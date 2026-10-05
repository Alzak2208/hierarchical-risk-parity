"""CPCV backtest of HRP against CLA, IVP and 1/N.

Runs the combinatorial purged cross-validation backtest of :mod:`src.backtest` either on the
synthetic sector universe (default) or on real prices (``--prices``), for instance a Databento
OHLCV-1d export of equities. It also measures how much the HRP tree changes from one training
set to the next.

Usage:
    python -m analysis.cpcv_backtest                      # synthetic universe -> results/
    python -m analysis.cpcv_backtest --prices data/equities.parquet   # real data -> results/real/

Options cover the CPCV scheme (groups, test groups, purge, embargo), the rebalancing frequency
and the transaction cost in basis points per unit of turnover.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plotting import COLORS, correlation_heatmap, save
from src.allocation import hrp_weights
from src.backtest import BacktestConfig, run_cpcv_backtest
from src.clustering import quasi_diagonal_order, tree_clustering, tree_clusters
from src.cpcv import CombinatorialPurgedCV
from src.data import load_prices, prices_to_returns
from src.metrics import weight_instability
from src.simulation import sector_universe

PANELS = [
    ("volatility", "annualised volatility"),
    ("max_drawdown", "maximum drawdown"),
    ("turnover", "turnover per year"),
    ("sharpe", "Sharpe ratio (net)"),
]


def tree_stability(
    returns: pd.DataFrame, cv: CombinatorialPurgedCV, sector: pd.Series | None = None
) -> pd.Series:
    """How much the HRP tree changes across the CPCV training sets.

    Counts the distinct trees, measures the share of the full-sample tree's clusters found in
    each training tree, and compares the HRP weight instability with the default and with the
    optimal leaf order. With sector labels, also counts the sectors that form a cluster of
    the full-sample tree.
    """
    data = returns.to_numpy(dtype=float)
    full_tree = tree_clusters(tree_clustering(returns.corr().to_numpy()))
    trees, shared = set(), []
    weights = {"default": [], "optimal": []}
    for split in cv.split(len(data)):
        train = data[split.train]
        cov, corr = np.cov(train, rowvar=False), np.corrcoef(train, rowvar=False)
        tree = tree_clusters(tree_clustering(corr))
        trees.add(tree)
        shared.append(len(tree & full_tree) / len(full_tree))
        weights["default"].append(hrp_weights(cov, corr))
        weights["optimal"].append(hrp_weights(cov, corr, optimal_ordering=True))
    stats = {
        "splits": cv.n_splits,
        "distinct_trees": len(trees),
        "mean_share_of_full_sample_clusters": round(float(np.mean(shared)), 4),
        "hrp_instability_default_order": round(weight_instability(np.array(weights["default"])), 4),
        "hrp_instability_optimal_order": round(weight_instability(np.array(weights["optimal"])), 4),
    }
    if sector is not None:
        labels = sector.reindex(returns.columns).to_numpy()
        groups = {frozenset(np.flatnonzero(labels == s)) for s in np.unique(labels)}
        stats["sectors"] = len(groups)
        stats["sectors_forming_a_cluster"] = len(groups & full_tree)
    return pd.Series(stats, name="value", dtype=object)


def plot_paths(paths: pd.DataFrame, summary: pd.DataFrame, title: str, path: Path) -> None:
    methods = list(summary.index)
    fig, axes = plt.subplots(1, len(PANELS) + 2, figsize=(19, 3.6))
    for ax, (column, label) in zip(axes, PANELS, strict=False):
        data = [paths.loc[paths["allocator"] == m, column] for m in methods]
        parts = ax.boxplot(
            data,
            tick_labels=methods,
            patch_artist=True,
            widths=0.5,
            showfliers=False,
            medianprops={"color": "black"},
        )
        for patch, m in zip(parts["boxes"], methods, strict=True):
            patch.set_facecolor(COLORS.get(m, "#999999"))
            patch.set_alpha(0.55)
        ax.set_title(label)
    for ax, (column, label) in zip(
        axes[len(PANELS) :],
        [
            ("weight_instability", "weight instability (L1)"),
            ("effective_assets", "effective number of assets"),
        ],
        strict=True,
    ):
        ax.bar(methods, summary[column], color=[COLORS.get(m, "#999999") for m in methods], alpha=0.8)
        ax.set_title(label)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_universe(returns: pd.DataFrame, path: Path) -> None:
    corr = returns.corr().to_numpy()
    order = quasi_diagonal_order(tree_clustering(corr))
    shuffled = np.random.default_rng(0).permutation(len(corr))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    correlation_heatmap(
        axes[0], corr[np.ix_(shuffled, shuffled)], returns.columns, "Correlation matrix (random order)", False
    )
    correlation_heatmap(
        axes[1],
        corr[np.ix_(order, order)],
        returns.columns[order],
        "After quasi-diagonalisation",
        show_labels=False,
    )
    fig.tight_layout()
    save(fig, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prices", type=Path, help="price file or directory (Parquet/CSV, long or wide)")
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--price-col", default="close")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--n-obs", type=int, default=2520, help="length of the synthetic history")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--purge", type=int, default=0)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.prices is not None:
        prices = load_prices(args.prices, symbol_col=args.symbol_col, price_col=args.price_col)
        returns = prices_to_returns(prices, min_coverage=args.min_coverage)
        sector = None
        tag, output_dir = "real", args.output_dir or Path("results/real")
        label = f"{returns.shape[1]} assets, {returns.index[0].date()} to {returns.index[-1].date()}"
    else:
        returns, sector = sector_universe(n_obs=args.n_obs, seed=args.seed)
        tag, output_dir = "synthetic", args.output_dir or Path("results")
        label = f"synthetic sector universe, {returns.shape[1]} assets, {len(returns)} days"

    cv = CombinatorialPurgedCV(
        args.groups, args.test_groups, purge=args.purge, embargo=int(args.embargo_pct * len(returns))
    )
    config = BacktestConfig(rebalance_every=args.rebalance_every, cost_bps=args.cost_bps)
    result = run_cpcv_backtest(returns, cv=cv, config=config)
    summary = result.summary()
    paths = result.path_metrics()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.round(6).to_csv(output_dir / f"cpcv_summary_{tag}.csv", index_label="allocator")
    paths.round(6).to_csv(output_dir / f"cpcv_paths_{tag}.csv", index=False)
    title = (
        f"CPCV backtest ({label}): {cv.n_splits} splits, {cv.n_paths} paths, "
        f"rebalancing every {config.rebalance_every} days, {config.cost_bps:g} bp costs"
    )
    plot_paths(paths, summary, title, output_dir / f"cpcv_backtest_{tag}.png")
    plot_universe(returns, output_dir / f"universe_clustered_{tag}.png")
    stability = tree_stability(returns, cv, sector)
    stability.to_csv(output_dir / f"cpcv_tree_stability_{tag}.csv", index_label="statistic")
    print(title)
    print(summary.round(4).to_string())
    print(stability.to_string())


if __name__ == "__main__":
    main()
