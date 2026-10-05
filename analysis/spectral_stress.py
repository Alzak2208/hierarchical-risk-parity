"""Spectral stress test: how much each allocation moves when the eigenvalues are re-scaled.

The covariance matrix is decomposed as ``V = W diag(lambda) W'`` and its eigenvalues are
randomly re-scaled, ``lambda~_n = N eps_n lambda_n / sum(eps)`` with ``eps ~ U[0, 1]``, while
the eigenvectors are kept. The multipliers average one but the total variance changes, which
does not matter here: none of the three allocations changes when the covariance matrix is
multiplied by a constant. Each method is re-run on the stressed matrix and the L1 distance
between the new and the original weights measures how much the allocation moves.

Two covariance matrices are stressed: the 10-asset numerical example and the 48-asset
synthetic sector universe used by the CPCV backtest.

Usage:
    python -m analysis.spectral_stress [--draws 2000] [--seed 163] [--output-dir results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plotting import COLORS, save
from src.allocation import hrp_weights, inverse_variance_weights
from src.cla import min_variance_weights
from src.simulation import numerical_example, sector_universe

METHODS = {
    "HRP": lambda cov: hrp_weights(cov),
    "CLA": lambda cov: min_variance_weights(cov),
    "IVP": lambda cov: inverse_variance_weights(cov),
}


def rescale_spectrum(cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eps = rng.uniform(0.0, 1.0, eigenvalues.size)
    stressed = eigenvalues.size * eps * eigenvalues / eps.sum()
    out = (eigenvectors * stressed) @ eigenvectors.T
    return 0.5 * (out + out.T)


def stress(cov: np.ndarray, n_draws: int, rng: np.random.Generator) -> pd.DataFrame:
    base = {name: f(cov) for name, f in METHODS.items()}
    rows = []
    for _ in range(n_draws):
        stressed = rescale_spectrum(cov, rng)
        rows.append({name: float(np.abs(f(stressed) - base[name]).sum()) for name, f in METHODS.items()})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=163)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    example, _ = numerical_example()
    universe, _ = sector_universe()
    covariances = {
        "numerical example (10 assets)": example.cov().to_numpy(),
        "sector universe (48 assets)": universe.cov().to_numpy(),
    }

    tables, distances = [], {}
    for name, cov in covariances.items():
        d = stress(cov, args.draws, rng)
        distances[name] = d
        table = d.describe(percentiles=[0.5, 0.95]).T[["mean", "50%", "95%"]]
        table.columns = ["mean_l1", "median_l1", "p95_l1"]
        table.insert(0, "covariance", name)
        tables.append(table)
    summary = pd.concat(tables).rename_axis("method").reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.output_dir / "spectral_stress.csv", index=False)
    print(summary.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (name, d) in zip(axes, distances.items(), strict=True):
        parts = ax.boxplot(
            [d[m] for m in METHODS],
            tick_labels=list(METHODS),
            showfliers=False,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "black"},
        )
        for patch, m in zip(parts["boxes"], METHODS, strict=True):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.55)
        ax.set_title(name)
        ax.set_ylabel("L1 change in weights")
    fig.suptitle(f"Re-scaled eigenvalues, {args.draws:,} draws per matrix", fontweight="bold")
    fig.tight_layout()
    save(fig, args.output_dir / "spectral_stress.png")


if __name__ == "__main__":
    main()
