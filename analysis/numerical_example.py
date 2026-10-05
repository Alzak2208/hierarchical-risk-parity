"""Numerical example: the three HRP stages and the in-sample allocations.

Ten series: five independent ones and five noisy copies of randomly chosen ones. The script
shows the three HRP stages (correlation matrix, dendrogram, quasi-diagonal matrix) and
compares the CLA minimum-variance, HRP and inverse-variance allocations in-sample.

Usage:
    python -m analysis.numerical_example [--n-obs 10000] [--seed 12345] [--output-dir results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from analysis.plotting import COLORS, correlation_heatmap, save
from src.allocation import hrp_weights, inverse_variance_weights
from src.cla import min_variance_weights
from src.clustering import quasi_diagonal_order, tree_clustering
from src.metrics import top_k_concentration
from src.simulation import numerical_example


def run(n_obs: int, seed: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, parents = numerical_example(n_obs=n_obs, seed=seed)
    labels = list(data.columns)
    print("noisy copies (copy <- original):", [(6 + k, int(p) + 1) for k, p in enumerate(parents)])
    cov, corr = data.cov().to_numpy(), data.corr().to_numpy()

    link = tree_clustering(corr)
    order = quasi_diagonal_order(link)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    correlation_heatmap(axes[0], corr, labels, "1. Correlation matrix")
    dendrogram(link, labels=labels, ax=axes[1], color_threshold=0, above_threshold_color="#1f5f8b")
    axes[1].set_title("2. Tree clustering (single linkage)")
    axes[1].set_ylabel("distance between merged clusters")
    axes[1].grid(False)
    correlation_heatmap(
        axes[2], corr[np.ix_(order, order)], [labels[i] for i in order], "3. Quasi-diagonalised matrix"
    )
    fig.tight_layout()
    save(fig, output_dir / "hrp_stages.png")

    weights = pd.DataFrame(
        {
            "CLA": min_variance_weights(cov),
            "HRP": hrp_weights(cov, corr),
            "IVP": inverse_variance_weights(cov),
        },
        index=pd.Index(labels, name="asset"),
    )
    stats = pd.DataFrame(
        {
            name: {
                "in_sample_std": float(np.sqrt(w @ cov @ w)),
                "top5_share": top_k_concentration(w, 5),
                "zero_weights": int((w < 1e-6).sum()),
            }
            for name, w in weights.items()
        }
    )
    condition_number = float(np.linalg.cond(cov))
    print(f"condition number of the covariance matrix: {condition_number:.2f}")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.27
    x = np.arange(len(labels))
    for k, name in enumerate(weights.columns):
        ax.bar(x + (k - 1) * width, weights[name], width, label=name, color=COLORS[name])
    ax.set_xticks(x, labels=labels)
    ax.set_xlabel("asset")
    ax.set_ylabel("weight")
    ax.set_title("In-sample allocations")
    ax.legend(frameon=False)
    save(fig, output_dir / "numerical_example_weights.png")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights.round(6).to_csv(output_dir / "numerical_example_weights.csv")
    stats.loc["condition_number"] = condition_number
    stats.round(6).to_csv(output_dir / "numerical_example_stats.csv")
    print((weights * 100).round(2).to_string())
    print(stats.round(4).to_string())
    return weights, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-obs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    run(args.n_obs, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
