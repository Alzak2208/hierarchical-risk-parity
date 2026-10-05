"""Performance, risk and allocation-quality metrics.

Return-based metrics take a 1-D array of simple periodic returns. Allocation metrics take
weight vectors (or a matrix with one allocation per row).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _returns(returns: ArrayLike) -> NDArray[np.float64]:
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2:
        raise ValueError("at least two returns are needed")
    return r


def wealth(returns: ArrayLike) -> NDArray[np.float64]:
    """Cumulative wealth of one unit invested, compounded."""
    return np.cumprod(1.0 + _returns(returns))


def cagr(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    r = _returns(returns)
    final = wealth(r)[-1]
    if final <= 0:
        return -1.0
    return float(final ** (periods_per_year / r.size) - 1.0)


def annualized_volatility(returns: ArrayLike, periods_per_year: int = 252) -> float:
    return float(np.std(_returns(returns), ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate."""
    r = _returns(returns)
    std = np.std(r, ddof=1)
    return float(np.mean(r) / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")


def max_drawdown(returns: ArrayLike) -> float:
    """Largest peak-to-trough loss of the wealth curve, as a positive fraction."""
    w = np.concatenate([[1.0], wealth(returns)])
    peaks = np.maximum.accumulate(w)
    return float(np.max(1.0 - w / peaks))


def effective_number_of_assets(weights: ArrayLike) -> float:
    """Inverse Herfindahl index ``1 / sum(w_i^2)``: N for 1/N, 1 for a single asset."""
    w = np.asarray(weights, dtype=float).ravel()
    return float(1.0 / np.sum(w**2))


def top_k_concentration(weights: ArrayLike, k: int = 5) -> float:
    """Share of the capital held by the ``k`` largest positions."""
    w = np.sort(np.asarray(weights, dtype=float).ravel())[::-1]
    return float(w[:k].sum())


def weight_instability(weight_matrix: ArrayLike) -> float:
    """Average L1 distance between each allocation (row) and the mean allocation.

    Applied to the allocations estimated on the different CPCV training sets, it measures
    how much an allocator reacts to estimation noise: 0 means the same weights every time.
    """
    w = np.asarray(weight_matrix, dtype=float)
    if w.ndim != 2:
        raise ValueError("weight_matrix must have one allocation per row")
    return float(np.mean(np.abs(w - w.mean(axis=0)).sum(axis=1)))
