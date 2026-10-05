"""Does the HRP allocation depend on the order of the assets?

The tree of stage 1 fixes which assets merge, but scipy decides which branch goes left from
the cluster ids, i.e. from the order in which the assets are listed. Because stage 3 bisects
the leaf order by position, listing the same assets in another order can change the weights.
This script permutes the columns many times and measures the L1 change in the weights, with
the default leaf order and with an optimal leaf ordering.

Usage:
    python -m analysis.order_invariance [--permutations 200] [--seed 31] [--output-dir results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.allocation import hrp_weights
from src.simulation import numerical_example, sector_universe


def permutation_sensitivity(
    cov: np.ndarray, n_permutations: int, rng: np.random.Generator, optimal_ordering: bool
) -> np.ndarray:
    reference = hrp_weights(cov, optimal_ordering=optimal_ordering)
    changes = np.empty(n_permutations)
    for k in range(n_permutations):
        p = rng.permutation(cov.shape[0])
        permuted = hrp_weights(cov[np.ix_(p, p)], optimal_ordering=optimal_ordering)
        restored = np.empty_like(permuted)
        restored[p] = permuted
        changes[k] = np.abs(restored - reference).sum()
    return changes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    covariances = {
        "numerical example (10 assets)": numerical_example()[0].cov().to_numpy(),
        "sector universe (48 assets)": sector_universe()[0].cov().to_numpy(),
    }
    rows = []
    for name, cov in covariances.items():
        for optimal in (False, True):
            changes = permutation_sensitivity(cov, args.permutations, rng, optimal)
            rows.append(
                {
                    "covariance": name,
                    "leaf_order": "optimal" if optimal else "default",
                    "mean_l1": changes.mean(),
                    "max_l1": changes.max(),
                }
            )
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(args.output_dir / "order_invariance.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
