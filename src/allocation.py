"""Hierarchical Risk Parity and simple risk-based benchmarks.

HRP allocates in three stages:

1. tree clustering of the assets from their correlations (:mod:`src.clustering`);
2. quasi-diagonalisation: assets are reordered along the dendrogram leaves;
3. recursive bisection: the ordered list is split in two halves, weight is shared between
   the halves in inverse proportion to their variance, and the process repeats inside each
   half until every subset holds a single asset.

The variance of a subset is measured with the inverse-variance portfolio of that subset,
which is optimal when the subset covariance is diagonal. Because no matrix is ever inverted,
HRP also works when the covariance matrix is ill-conditioned or singular.

All public functions accept numpy arrays or pandas DataFrames. When a DataFrame is passed,
the weights come back as a pandas Series indexed by the asset labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from src.clustering import quasi_diagonal_order, tree_clustering


def _unpack(matrix: ArrayLike | pd.DataFrame, name: str) -> tuple[NDArray[np.float64], pd.Index | None]:
    labels = matrix.columns if isinstance(matrix, pd.DataFrame) else None
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr, labels


def _pack(weights: NDArray[np.float64], labels: pd.Index | None) -> NDArray[np.float64] | pd.Series:
    if labels is None:
        return weights
    return pd.Series(weights, index=labels, name="weight")


def cov_to_corr(cov: ArrayLike) -> NDArray[np.float64]:
    """Correlation matrix implied by a covariance matrix."""
    c = np.asarray(cov, dtype=float)
    std = np.sqrt(np.diag(c))
    if np.any(std <= 0):
        raise ValueError("every asset needs a strictly positive variance")
    corr = np.clip(c / np.outer(std, std), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _ivp(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    variances = np.diag(cov)
    if np.any(variances <= 0):
        raise ValueError("every asset needs a strictly positive variance")
    inv = 1.0 / variances
    return inv / inv.sum()


def inverse_variance_weights(cov: ArrayLike | pd.DataFrame) -> NDArray[np.float64] | pd.Series:
    """Inverse-variance portfolio ``w_i = (1 / s_i^2) / sum_j (1 / s_j^2)``.

    It is the minimum-variance portfolio when the covariance matrix is diagonal, and the
    usual stand-in for traditional risk parity.
    """
    arr, labels = _unpack(cov, "cov")
    return _pack(_ivp(arr), labels)


def equal_weights(n_assets: int) -> NDArray[np.float64]:
    """Naive 1/N portfolio."""
    if n_assets < 1:
        raise ValueError("n_assets must be positive")
    return np.full(n_assets, 1.0 / n_assets)


def cluster_variance(cov: ArrayLike, items: Sequence[int]) -> float:
    """Variance of a subset of assets held through its own inverse-variance portfolio."""
    c = np.asarray(cov, dtype=float)
    idx = np.asarray(items, dtype=int)
    sub = c[np.ix_(idx, idx)]
    w = _ivp(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov: ArrayLike, order: Sequence[int]) -> NDArray[np.float64]:
    """Stage 3: split the weight top-down along the quasi-diagonal order.

    Every list of more than one asset is cut into two halves that keep the order. With
    ``V1`` and ``V2`` the variances of the halves, the first half is scaled by
    ``alpha = 1 - V1 / (V1 + V2)`` and the second by ``1 - alpha``, so the less risky half
    receives more capital. Weights stay in [0, 1] and sum to one by construction.
    """
    c = np.asarray(cov, dtype=float)
    weights = np.ones(c.shape[0])
    clusters = [list(order)]
    while clusters:
        next_level = []
        for items in clusters:
            if len(items) < 2:
                continue
            half = len(items) // 2
            left, right = items[:half], items[half:]
            var_left = cluster_variance(c, left)
            var_right = cluster_variance(c, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_level.extend((left, right))
        clusters = next_level
    return weights


def hrp_weights(
    cov: ArrayLike | pd.DataFrame,
    corr: ArrayLike | pd.DataFrame | None = None,
    method: str = "single",
    optimal_ordering: bool = False,
) -> NDArray[np.float64] | pd.Series:
    """Hierarchical Risk Parity weights.

    Args:
        cov: covariance matrix of asset returns (N x N).
        corr: correlation matrix used for the clustering. Derived from ``cov`` when omitted.
        method: linkage criterion for the tree clustering ("single" in the original method).
        optimal_ordering: reorder the dendrogram leaves optimally, which makes the weights
            independent of the order in which the assets are listed (see
            :func:`src.clustering.tree_clustering`). Off by default to match the original method.

    Returns:
        Long-only, fully invested weights in the original asset order.
    """
    c, labels = _unpack(cov, "cov")
    if c.shape[0] == 1:
        return _pack(np.ones(1), labels)
    rho = cov_to_corr(c) if corr is None else _unpack(corr, "corr")[0]
    if rho.shape != c.shape:
        raise ValueError("cov and corr must have the same shape")
    order = quasi_diagonal_order(tree_clustering(rho, method=method, optimal_ordering=optimal_ordering))
    return _pack(recursive_bisection(c, order), labels)
