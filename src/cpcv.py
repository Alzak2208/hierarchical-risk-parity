"""Combinatorial Purged Cross-Validation (CPCV).

The history is cut into ``N`` contiguous groups. Every combination of ``k`` groups is used
once as a test set, the remaining observations being the training set, which gives
``C(N, k)`` splits. Each group is tested in ``C(N-1, k-1)`` of them, so the test segments
can be stitched into ``phi = C(N-1, k-1)`` complete out-of-sample paths covering the whole
history, instead of the single path of a walk-forward backtest.

Training observations close to a test group are removed to limit leakage through serial
dependence: ``purge`` observations before every test group and an ``embargo`` of
observations after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Split:
    """One train/test configuration: training indices and the groups used as test set."""

    train: NDArray[np.intp]
    test_groups: tuple[int, ...]


class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation over a time index of length ``n_obs``.

    Args:
        n_groups: number of contiguous groups ``N``.
        n_test_groups: number of groups per test set ``k`` (``1 <= k < N``).
        purge: training observations dropped right before each test group.
        embargo: training observations dropped right after each test group.
    """

    def __init__(self, n_groups: int = 10, n_test_groups: int = 2, purge: int = 0, embargo: int = 0):
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must satisfy 1 <= k < n_groups")
        if purge < 0 or embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge = purge
        self.embargo = embargo

    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def group_bounds(self, n_obs: int) -> list[tuple[int, int]]:
        """Half-open ``[start, end)`` bounds of the contiguous groups (sizes differ by at most one)."""
        if n_obs < self.n_groups:
            raise ValueError("n_obs must be at least n_groups")
        edges = np.linspace(0, n_obs, self.n_groups + 1).round().astype(int)
        return [(int(edges[g]), int(edges[g + 1])) for g in range(self.n_groups)]

    def split(self, n_obs: int) -> list[Split]:
        """All ``C(N, k)`` splits, test groups in lexicographic order."""
        bounds = self.group_bounds(n_obs)
        splits = []
        for test_groups in combinations(range(self.n_groups), self.n_test_groups):
            keep = np.ones(n_obs, dtype=bool)
            for g in test_groups:
                start, end = bounds[g]
                keep[max(0, start - self.purge) : min(n_obs, end + self.embargo)] = False
            splits.append(Split(train=np.flatnonzero(keep), test_groups=test_groups))
        return splits

    def paths(self) -> NDArray[np.intp]:
        """Path table: entry ``[p, g]`` is the split whose test segment fills group ``g`` of path ``p``.

        Splits are enumerated in the order of :meth:`split`; the i-th split that tests group
        ``g`` feeds path ``i``. Every path covers every group exactly once.
        """
        table = np.full((self.n_paths, self.n_groups), -1, dtype=np.intp)
        filled = np.zeros(self.n_groups, dtype=np.intp)
        for s, test_groups in enumerate(combinations(range(self.n_groups), self.n_test_groups)):
            for g in test_groups:
                table[filled[g], g] = s
                filled[g] += 1
        return table
