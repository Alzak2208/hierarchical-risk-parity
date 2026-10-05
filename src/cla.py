"""Critical Line Algorithm.

Solves, for every ``lam >= 0`` at once, the box-constrained mean-variance problem

    min_w  1/2 w'Sw - lam * mu'w
    s.t.   sum(w) = 1,   lower <= w <= upper

The optimal weights are piecewise linear in ``lam``. Starting from ``lam = +inf`` (all the
capital on the highest-mean assets), the algorithm lowers ``lam`` and stops at every
turning point, where an asset either reaches one of its bounds (it leaves the free set) or
comes off a bound (it joins the free set). It ends at ``lam = 0``: the minimum-variance
portfolio, used as the quadratic-optimisation benchmark for HRP. That end point does not
depend on ``mu``, which only fixes the path.

With F the free assets and B the assets sitting at a bound, the first-order conditions give

    w_F(lam)   = S_FF^-1 (lam mu_F + gamma(lam) 1_F - S_FB w_B)
    gamma(lam) = (1 - 1'w_B + 1'S_FF^-1 S_FB w_B - lam 1'S_FF^-1 mu_F) / (1'S_FF^-1 1)

so between two turning points ``w_F(lam) = a + lam * s`` and ``gamma(lam) = g0 + lam * g1``:

- a free asset i reaches a bound when ``a_i + lam * s_i`` equals that bound;
- a bounded asset j comes off its bound when its reduced gradient
  ``(Sw)_j - lam mu_j - gamma(lam)`` crosses zero (it stays >= 0 at a lower bound and
  <= 0 at an upper bound for as long as the asset remains bounded).

The next turning point is the largest of those ``lam`` below the current one.

Only ``S_FF``, the covariance block of the free assets, is inverted. When it is singular or
ill-conditioned, :class:`SingularCovarianceError` is raised; HRP has no such requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

_BOUND_TOL = 1e-12
_SLOPE_TOL = 1e-14


class SingularCovarianceError(np.linalg.LinAlgError):
    """The covariance block of the free assets cannot be inverted reliably."""


@dataclass(frozen=True)
class TurningPoint:
    """A corner of the efficient frontier: weights, the ``lam`` at which they are reached
    and the free assets on the segment that follows."""

    lam: float
    weights: NDArray[np.float64]
    free: tuple[int, ...]


def _check_inputs(
    cov: ArrayLike,
    mean: ArrayLike | None,
    lower: ArrayLike | None,
    upper: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"cov must be a square matrix, got shape {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValueError("cov contains NaN or infinite values")
    if not np.allclose(c, c.T, rtol=1e-8, atol=1e-12):
        raise ValueError("cov must be symmetric")
    c = 0.5 * (c + c.T)
    n = c.shape[0]
    # Any vector with distinct entries works for the minimum-variance end point.
    mu = np.arange(n, dtype=float) if mean is None else np.asarray(mean, dtype=float).ravel()
    lo = np.zeros(n) if lower is None else np.broadcast_to(np.asarray(lower, dtype=float), (n,)).copy()
    up = np.ones(n) if upper is None else np.broadcast_to(np.asarray(upper, dtype=float), (n,)).copy()
    if mu.shape != (n,):
        raise ValueError("mean must have one entry per asset")
    if np.any(lo > up):
        raise ValueError("every lower bound must be below its upper bound")
    if lo.sum() > 1.0 + 1e-12 or up.sum() < 1.0 - 1e-12:
        raise ValueError("the bounds make a fully invested portfolio impossible")
    return c, mu, lo, up


def _initial_solution(
    mu: NDArray[np.float64], lo: NDArray[np.float64], up: NDArray[np.float64]
) -> tuple[NDArray[np.float64], int]:
    """Maximum expected return portfolio: fill the highest-mean assets up to their upper bound.

    The asset that receives the remaining budget is the first free asset.
    """
    w = lo.copy()
    budget = 1.0 - w.sum()
    for i in np.argsort(-mu, kind="stable"):
        room = up[i] - lo[i]
        if room >= budget:
            w[i] += budget
            return w, int(i)
        w[i] = up[i]
        budget -= room
    raise ValueError("the bounds make a fully invested portfolio impossible")


def _free_solution(
    c: NDArray[np.float64],
    mu: NDArray[np.float64],
    w: NDArray[np.float64],
    free: NDArray[np.intp],
    bounded: NDArray[np.intp],
    cond_limit: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Linear parametrisation ``w_F = a + lam * s`` and ``gamma = g0 + lam * g1``."""
    s_ff = c[np.ix_(free, free)]
    if free.size > 1 and np.linalg.cond(s_ff) > cond_limit:
        raise SingularCovarianceError("covariance of the free assets is singular or ill-conditioned")
    cross = c[np.ix_(free, bounded)] @ w[bounded] if bounded.size else np.zeros(free.size)
    rhs = np.column_stack([np.ones(free.size), mu[free], cross])
    try:
        inv_one, inv_mu, inv_cross = np.linalg.solve(s_ff, rhs).T
    except np.linalg.LinAlgError as exc:
        raise SingularCovarianceError("covariance of the free assets is singular") from exc
    c1 = inv_one.sum()
    g1 = -inv_mu.sum() / c1
    g0 = (1.0 - w[bounded].sum() + inv_cross.sum()) / c1
    a = g0 * inv_one - inv_cross
    s = inv_mu + g1 * inv_one
    return a, s, g0, g1


def _strictly_below(values: NDArray[np.float64], lam: float) -> NDArray[np.bool_]:
    margin = 0.0 if np.isinf(lam) else 1e-12 * max(1.0, abs(lam))
    return values < lam - margin


def critical_line(
    cov: ArrayLike,
    mean: ArrayLike | None = None,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    *,
    cond_limit: float = 1e12,
    max_iter: int | None = None,
) -> list[TurningPoint]:
    """Turning points of the efficient frontier, from ``lam = +inf`` down to ``lam = 0``.

    Args:
        cov: covariance matrix (N x N), symmetric positive definite.
        mean: expected returns. Defaults to ``0, 1, ..., N-1``, which is enough when only the
            minimum-variance end point is needed (it does not depend on the means).
        lower: lower bounds (scalar or vector), 0 by default (long only).
        upper: upper bounds (scalar or vector), 1 by default.
        cond_limit: condition number above which the free covariance block is treated as singular.
        max_iter: safety cap on the number of turning points.

    Returns:
        Turning points with strictly decreasing ``lam``. The first one has ``lam = inf``
        (maximum expected return), the last one ``lam = 0`` (minimum variance).
    """
    c, mu, lo, up = _check_inputs(cov, mean, lower, upper)
    n = mu.size
    w, first_free = _initial_solution(mu, lo, up)
    free = [first_free]
    lam = np.inf
    points = [TurningPoint(lam, w.copy(), (first_free,))]
    movable = (up - lo) > _BOUND_TOL
    limit = max_iter if max_iter is not None else 10 * n + 50

    for _ in range(limit):
        is_free = np.zeros(n, dtype=bool)
        is_free[free] = True
        f = np.flatnonzero(is_free)
        b = np.flatnonzero(~is_free)
        a, s, g0, g1 = _free_solution(c, mu, w, f, b, cond_limit)

        best, event = -np.inf, None

        # Case 1: a free asset reaches one of its bounds as lam decreases.
        if f.size > 1:
            moving = np.abs(s) > _SLOPE_TOL
            target = np.where(s > 0, lo[f], up[f])
            with np.errstate(divide="ignore", invalid="ignore"):
                lam_out = np.where(moving, (target - a) / s, -np.inf)
            lam_out = np.where(_strictly_below(lam_out, lam), lam_out, -np.inf)
            k = int(np.argmax(lam_out))
            if lam_out[k] > best:
                best, event = lam_out[k], ("out", int(f[k]), float(target[k]))

        # Case 2: a bounded asset comes off its bound (its reduced gradient crosses zero).
        if b.size:
            p = c[np.ix_(b, f)] @ a + c[np.ix_(b, b)] @ w[b] - g0
            q = c[np.ix_(b, f)] @ s - mu[b] - g1
            at_lower = np.abs(w[b] - lo[b]) <= np.abs(w[b] - up[b])
            leaving = np.where(at_lower, q > _SLOPE_TOL, q < -_SLOPE_TOL) & movable[b]
            with np.errstate(divide="ignore", invalid="ignore"):
                lam_in = np.where(leaving, -p / q, -np.inf)
            lam_in = np.where(_strictly_below(lam_in, lam), lam_in, -np.inf)
            k = int(np.argmax(lam_in))
            if lam_in[k] > best:
                best, event = lam_in[k], ("in", int(b[k]), None)

        if event is None or best <= 0.0:
            w_min = w.copy()
            w_min[f] = a  # lam = 0
            points.append(TurningPoint(0.0, w_min, tuple(int(i) for i in f)))
            return points

        lam = float(best)
        w[f] = a + lam * s
        kind, asset, bound = event
        if kind == "out":
            w[asset] = bound
            free.remove(asset)
        else:
            free.append(asset)
        points.append(TurningPoint(lam, w.copy(), tuple(sorted(free))))

    raise RuntimeError("the critical line algorithm did not reach lam = 0 within max_iter turning points")


def weights_at(points: list[TurningPoint], lam: float) -> NDArray[np.float64]:
    """Optimal weights for any ``lam >= 0``, interpolated between the turning points."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if len(points) < 2 or lam >= points[1].lam:
        return points[0].weights.copy()
    for hi, lo in zip(points[1:-1], points[2:], strict=True):
        if hi.lam >= lam >= lo.lam:
            t = (hi.lam - lam) / (hi.lam - lo.lam)
            return hi.weights + t * (lo.weights - hi.weights)
    return points[-1].weights.copy()


def min_variance_weights(
    cov: ArrayLike | pd.DataFrame,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
) -> NDArray[np.float64] | pd.Series:
    """Minimum-variance portfolio under bounds (long only by default), via the CLA.

    Raises:
        SingularCovarianceError: when the covariance matrix cannot be inverted reliably.
    """
    labels = cov.columns if isinstance(cov, pd.DataFrame) else None
    weights = critical_line(cov, None, lower, upper)[-1].weights
    if labels is None:
        return weights
    return pd.Series(weights, index=labels, name="weight")
