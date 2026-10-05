"""CPCV backtest of allocation methods.

For every CPCV split, each allocator receives the sample covariance and correlation matrices
of the training observations and returns long-only, fully invested weights. Those weights are
then held out-of-sample on the test groups of the split.

The test segments are stitched into complete paths (see :mod:`src.cpcv`). Along a path the
portfolio is rebalanced to its target at the start of every group (the target changes with
the split that fills the group) and every ``rebalance_every`` observations inside a group.
Between rebalances the weights drift with returns, and every rebalance pays ``cost_bps`` per
unit of turnover (sum of absolute weight changes).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src import metrics
from src.allocation import equal_weights, hrp_weights, inverse_variance_weights
from src.cla import min_variance_weights
from src.cpcv import CombinatorialPurgedCV

Allocator = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def default_allocators() -> dict[str, Allocator]:
    """HRP against the CLA minimum-variance and inverse-variance portfolios, plus 1/N."""
    return {
        "HRP": lambda cov, corr: np.asarray(hrp_weights(cov, corr)),
        "CLA": lambda cov, corr: np.asarray(min_variance_weights(cov)),
        "IVP": lambda cov, corr: np.asarray(inverse_variance_weights(cov)),
        "1/N": lambda cov, corr: equal_weights(cov.shape[0]),
    }


@dataclass(frozen=True)
class BacktestConfig:
    rebalance_every: int = 21
    cost_bps: float = 10.0
    periods_per_year: int = 252


@dataclass
class AllocatorRun:
    """Outputs for one allocator: weights per split and returns per path (NaN when unavailable)."""

    weights: NDArray[np.float64]  # (n_splits, n_assets)
    failures: int
    gross: NDArray[np.float64]  # (n_paths, n_obs)
    net: NDArray[np.float64]  # (n_paths, n_obs)
    turnover: NDArray[np.float64]  # (n_paths, n_obs)


@dataclass
class BacktestResult:
    runs: dict[str, AllocatorRun]
    index: pd.Index
    assets: pd.Index
    cv: CombinatorialPurgedCV
    config: BacktestConfig = field(default_factory=BacktestConfig)

    def path_metrics(self) -> pd.DataFrame:
        """One row per (allocator, path) with risk, performance and trading metrics."""
        ppy = self.config.periods_per_year
        rows = []
        for name, run in self.runs.items():
            for p in range(run.net.shape[0]):
                net, gross = run.net[p], run.gross[p]
                if np.isnan(net).any():
                    continue
                years = net.size / ppy
                rows.append(
                    {
                        "allocator": name,
                        "path": p,
                        "variance": metrics.annualized_volatility(net, ppy) ** 2,
                        "volatility": metrics.annualized_volatility(net, ppy),
                        "sharpe": metrics.sharpe_ratio(net, ppy),
                        "cagr": metrics.cagr(net, ppy),
                        "max_drawdown": metrics.max_drawdown(net),
                        "turnover": run.turnover[p].sum() / years,
                        "cost_drag": metrics.cagr(gross, ppy) - metrics.cagr(net, ppy),
                    }
                )
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Path metrics averaged over the CPCV paths, plus allocation-quality statistics."""
        paths = self.path_metrics()
        table = paths.drop(columns="path").groupby("allocator", sort=False).mean()
        quality = {}
        for name, run in self.runs.items():
            valid = run.weights[~np.isnan(run.weights).any(axis=1)]
            quality[name] = {
                "weight_instability": metrics.weight_instability(valid) if len(valid) else np.nan,
                "effective_assets": np.mean([metrics.effective_number_of_assets(w) for w in valid])
                if len(valid)
                else np.nan,
                "zero_weights": float(np.mean(valid < 1e-6)) if len(valid) else np.nan,
                "failed_splits": run.failures,
            }
        return table.join(pd.DataFrame.from_dict(quality, orient="index"), how="right")


def simulate_path(
    returns: NDArray[np.float64],
    bounds: list[tuple[int, int]],
    targets: NDArray[np.float64],
    rebalance_every: int,
    cost_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Hold target weights group by group, with periodic rebalancing and proportional costs.

    Returns:
        Gross returns, net returns and turnover, one value per observation.
    """
    n_obs, n_assets = returns.shape
    gross = np.zeros(n_obs)
    net = np.zeros(n_obs)
    turnover = np.zeros(n_obs)
    holdings = np.zeros(n_assets)  # start from cash
    for (start, end), target in zip(bounds, targets, strict=True):
        for t in range(start, end):
            if (t - start) % rebalance_every == 0:
                turnover[t] = np.abs(target - holdings).sum()
                holdings = target.copy()
            r_t = returns[t]
            port = float(holdings @ r_t)
            gross[t] = port
            net[t] = (1.0 - cost_rate * turnover[t]) * (1.0 + port) - 1.0
            value = 1.0 + port
            holdings = holdings * (1.0 + r_t) / value if value > 0 else np.zeros(n_assets)
    return gross, net, turnover


def _validate_weights(w: NDArray[np.float64], n_assets: int, name: str) -> None:
    if w.shape != (n_assets,) or not np.all(np.isfinite(w)):
        raise ValueError(f"allocator {name!r} must return {n_assets} finite weights")
    if abs(w.sum() - 1.0) > 1e-6 or w.min() < -1e-9:
        raise ValueError(f"allocator {name!r} must return long-only weights summing to one")


def run_cpcv_backtest(
    returns: pd.DataFrame,
    allocators: Mapping[str, Allocator] | None = None,
    cv: CombinatorialPurgedCV | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest allocators under combinatorial purged cross-validation.

    Args:
        returns: periodic simple returns, one column per asset, without missing values.
        allocators: name -> callable(cov, corr) returning weights. Defaults to
            :func:`default_allocators`.
        cv: the CPCV scheme. Defaults to 10 groups, 2 test groups and a 1% embargo.
        config: rebalancing frequency, cost rate and annualisation.

    An allocator that raises ``LinAlgError`` or ``ValueError`` on a split (for instance the
    CLA on a singular covariance matrix) is recorded as a failed split; the paths that need
    that split are then left out of its statistics.
    """
    data = returns.to_numpy(dtype=float)
    if np.isnan(data).any():
        raise ValueError("returns must not contain missing values")
    n_obs, n_assets = data.shape
    allocators = default_allocators() if allocators is None else dict(allocators)
    cv = cv if cv is not None else CombinatorialPurgedCV(10, 2, purge=0, embargo=int(0.01 * n_obs))
    config = config if config is not None else BacktestConfig()

    splits = cv.split(n_obs)
    bounds = cv.group_bounds(n_obs)
    table = cv.paths()
    cost_rate = config.cost_bps / 1e4

    estimates = []
    for split in splits:
        train = data[split.train]
        estimates.append((np.cov(train, rowvar=False), np.corrcoef(train, rowvar=False)))

    runs: dict[str, AllocatorRun] = {}
    for name, allocate in allocators.items():
        weights = np.full((len(splits), n_assets), np.nan)
        failures = 0
        for s, (cov, corr) in enumerate(estimates):
            try:
                w = np.asarray(allocate(cov, corr), dtype=float).ravel()
            except (np.linalg.LinAlgError, ValueError):
                failures += 1
                continue
            _validate_weights(w, n_assets, name)
            weights[s] = w

        gross = np.full((cv.n_paths, n_obs), np.nan)
        net = np.full_like(gross, np.nan)
        turnover = np.full_like(gross, np.nan)
        for p in range(cv.n_paths):
            targets = weights[table[p]]
            if np.isnan(targets).any():
                continue
            gross[p], net[p], turnover[p] = simulate_path(
                data, bounds, targets, config.rebalance_every, cost_rate
            )
        runs[name] = AllocatorRun(weights, failures, gross, net, turnover)

    return BacktestResult(runs, returns.index, returns.columns, cv, config)
