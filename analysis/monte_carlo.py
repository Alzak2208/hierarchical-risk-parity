"""Out-of-sample Monte Carlo of HRP, the CLA and IVP.

Each run simulates 520 daily observations of 10 series (5 independent, 5 noisy copies) with
jumps placed in the second half: a common shock on a series and its copy, and a specific
shock on one series. HRP, the CLA minimum-variance portfolio and the inverse-variance
portfolio are estimated on the previous 260 observations, rebalanced every 22 observations,
and held out-of-sample. The statistic of interest is the variance, across runs, of the
cumulative out-of-sample return of each portfolio.

Runs are seeded individually (numpy SeedSequence), so the results do not depend on the
number of worker processes.

Usage:
    python -m analysis.monte_carlo [--runs 10000] [--seed 2016] [--workers N] [--output-dir results]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plotting import COLORS, save
from src.allocation import hrp_weights, inverse_variance_weights
from src.cla import min_variance_weights
from src.simulation import monte_carlo_sample

METHODS = ("IVP", "HRP", "CLA")
N_OBS, S_LENGTH, REBALANCE = 520, 260, 22


def allocate(method: str, cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    if method == "IVP":
        return inverse_variance_weights(cov)
    if method == "HRP":
        return hrp_weights(cov, corr)
    return min_variance_weights(cov)


def run_one(seed: np.random.SeedSequence, keep_weights: bool = False):
    """Cumulative out-of-sample return of each method for one simulated history.

    Also returns the weights at each rebalance (when ``keep_weights``) and, for each noisy
    copy, the 0-based index of the series it was built from.
    """
    x, parents = monte_carlo_sample(np.random.default_rng(seed), n_obs=N_OBS, s_length=S_LENGTH)
    oos = {m: [] for m in METHODS}
    history = {m: [] for m in METHODS}
    for pointer in range(S_LENGTH, N_OBS, REBALANCE):
        window = x[pointer - S_LENGTH : pointer]
        cov, corr = np.cov(window, rowvar=False), np.corrcoef(window, rowvar=False)
        held = x[pointer : pointer + REBALANCE]
        for m in METHODS:
            w = allocate(m, cov, corr)
            oos[m].append(held @ w)
            if keep_weights:
                history[m].append(w)
    cumulative = {m: float(np.prod(1.0 + np.concatenate(oos[m])) - 1.0) for m in METHODS}
    return cumulative, (history if keep_weights else None), parents


def _run_chunk(seeds: list[np.random.SeedSequence]) -> list[dict[str, float]]:
    return [run_one(s)[0] for s in seeds]


def run(n_runs: int, seed: int, workers: int) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]], np.ndarray]:
    seeds = np.random.SeedSequence(seed).spawn(n_runs)
    _, first_history, first_parents = run_one(seeds[0], keep_weights=True)
    if workers <= 1:
        rows = _run_chunk(seeds)
    else:
        chunks = [list(c) for c in np.array_split(np.array(seeds, dtype=object), workers * 4) if len(c)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [row for chunk in pool.map(_run_chunk, chunks) for row in chunk]
    return pd.DataFrame(rows, columns=list(METHODS)), first_history, first_parents


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    var = runs.var()
    return pd.DataFrame(
        {
            "mean": runs.mean(),
            "std": runs.std(),
            "variance": var,
            "variance_vs_hrp": var / var["HRP"] - 1.0,
        }
    )


def plot(
    runs: pd.DataFrame, history: dict[str, list[np.ndarray]], parents: np.ndarray, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    rebalances = np.arange(1, len(history["HRP"]) + 1)
    series_colors = plt.get_cmap("tab10").colors
    size0 = np.array(history["HRP"]).shape[1] - len(parents)
    for ax, m in zip(axes, METHODS, strict=True):
        weights = np.array(history[m])
        for j in range(weights.shape[1]):
            label = f"{j + 1}" if j < size0 else f"{j + 1} (copy of {parents[j - size0] + 1})"
            ax.plot(rebalances, weights[:, j], lw=1.4, color=series_colors[j % 10], label=label)
        ax.set_title(m)
        ax.set_xlabel("rebalance")
    axes[0].set_ylabel("weight")
    axes[-1].legend(title="series", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"Weights at each rebalance, run 1: specific shock on series {parents[-1] + 1}, "
        f"common shock on series {parents[0] + 1} and {size0 + 1}",
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, output_dir / "monte_carlo_allocations.png")

    var = runs.var()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    bars = axes[0].bar(METHODS, [var[m] for m in METHODS], color=[COLORS[m] for m in METHODS], alpha=0.85)
    for bar, m in zip(bars, METHODS, strict=True):
        gap = var[m] / var["HRP"] - 1.0
        text = f"{var[m]:.4f}" + ("" if m == "HRP" else f"\n(+{gap:.0%} vs HRP)")
        axes[0].annotate(
            text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom"
        )
    axes[0].set_ylim(0, var.max() * 1.3)
    axes[0].set_ylabel("variance of the cumulative return")
    axes[0].set_title("Out-of-sample variance")
    parts = axes[1].boxplot(
        [runs[m] for m in METHODS],
        tick_labels=list(METHODS),
        showfliers=False,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black"},
    )
    for patch, m in zip(parts["boxes"], METHODS, strict=True):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.55)
    axes[1].set_ylabel("cumulative out-of-sample return")
    axes[1].set_title("Distribution across runs (outliers hidden)")
    fig.suptitle(f"Out-of-sample Monte Carlo, {len(runs):,} runs", fontweight="bold")
    fig.tight_layout()
    save(fig, output_dir / "monte_carlo_distribution.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2016)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    runs, history, parents = run(args.runs, args.seed, args.workers)
    summary = summarise(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "monte_carlo_runs.csv", index_label="run", float_format="%.6g")
    summary.round(6).to_csv(args.output_dir / "monte_carlo_summary.csv")
    plot(runs, history, parents, args.output_dir)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
