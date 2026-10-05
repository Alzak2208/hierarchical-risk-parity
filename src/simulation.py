"""Synthetic return generators.

- :func:`numerical_example`: ``size0`` independent series plus ``size1`` noisy copies of
  randomly chosen ones.
- :func:`monte_carlo_sample`: one draw of the out-of-sample Monte Carlo, the same structure
  plus two kinds of jumps placed in the out-of-sample half: a common shock
  hitting a series and its correlated copy, and a specific shock hitting a single series.
- :func:`sector_universe`: an equity-like universe (market factor, sector factors,
  fat-tailed idiosyncratic noise and rare jumps) used to demonstrate the CPCV backtest.

Every generator takes a seed or a :class:`numpy.random.Generator`, so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SeedLike = int | np.random.Generator | None


def _rng(seed: SeedLike) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def numerical_example(
    n_obs: int = 10_000,
    size0: int = 5,
    size1: int = 5,
    sigma1: float = 0.25,
    seed: SeedLike = 12345,
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Independent standard normal series plus noisy copies of some of them.

    Returns:
        Returns with columns labelled 1..N and, for each copy,
        the 0-based index of the series it was built from.
    """
    rng = _rng(seed)
    base = rng.normal(0.0, 1.0, size=(n_obs, size0))
    parents = rng.integers(0, size0, size=size1)
    copies = base[:, parents] + rng.normal(0.0, sigma1, size=(n_obs, size1))
    data = np.hstack([base, copies])
    return pd.DataFrame(data, columns=range(1, data.shape[1] + 1)), parents


def monte_carlo_sample(
    rng: np.random.Generator,
    n_obs: int = 520,
    s_length: int = 260,
    size0: int = 5,
    size1: int = 5,
    mu0: float = 0.0,
    sigma0: float = 1e-2,
    sigma1_factor: float = 0.25,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """One simulated history for the out-of-sample Monte Carlo.

    Jumps are only placed after ``s_length``, i.e. in the out-of-sample part of the history:
    a common shock on series ``parents[0]`` and on its correlated copy, and a specific shock
    on series ``parents[-1]`` alone. Each shock is a -50% and a +200% return on two random
    dates, so the +200% can come first.
    """
    base = rng.normal(mu0, sigma0, size=(n_obs, size0))
    parents = rng.integers(0, size0, size=size1)
    copies = base[:, parents] + rng.normal(0.0, sigma0 * sigma1_factor, size=(n_obs, size1))
    x = np.hstack([base, copies])
    dates = rng.integers(s_length, n_obs - 1, size=2)
    x[np.ix_(dates, [parents[0], size0])] = np.array([[-0.5, -0.5], [2.0, 2.0]])
    dates = rng.integers(s_length, n_obs - 1, size=2)
    x[dates, parents[-1]] = np.array([-0.5, 2.0])
    return x, parents


def sector_universe(
    n_sectors: int = 6,
    assets_per_sector: int = 8,
    n_obs: int = 2520,
    seed: SeedLike = 7,
    market_vol: float = 0.010,
    sector_vol: float = 0.007,
    idio_vol: float = 0.012,
    tail_df: float = 4.0,
    jump_prob: float = 0.002,
    jump_vol: float = 0.06,
    drift: float = 2e-4,
    demean: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns of an equity-like universe with a sector structure.

    ``r = drift + beta_m * market + beta_s * sector + idio + jumps``, with Student-t factor and
    idiosyncratic innovations (unit variance, then scaled), heterogeneous betas and
    idiosyncratic volatilities, and rare Gaussian jumps on single names.

    With ``demean=True`` the simulated factor and idiosyncratic innovations are centred over the
    sample, so every asset earns the same drift apart from its jumps. The covariance matrix is
    unchanged; only the luck of the draw is removed from average returns, which keeps the study
    focused on risk (a 10-year simulated market can otherwise trend down by chance).

    Returns:
        Returns (business days, one column per asset) and the sector of each asset.
    """
    rng = _rng(seed)
    n_assets = n_sectors * assets_per_sector
    scale = np.sqrt((tail_df - 2.0) / tail_df)  # unit-variance Student-t

    def shocks(size: tuple[int, ...]) -> NDArray[np.float64]:
        return rng.standard_t(tail_df, size=size) * scale

    market = market_vol * shocks((n_obs,))
    sectors = sector_vol * shocks((n_obs, n_sectors))
    sector_of = np.repeat(np.arange(n_sectors), assets_per_sector)
    beta_m = rng.uniform(0.6, 1.4, n_assets)
    beta_s = rng.uniform(0.5, 1.5, n_assets)
    idio_scale = idio_vol * rng.uniform(0.6, 1.6, n_assets)
    idio = shocks((n_obs, n_assets)) * idio_scale
    if demean:
        market -= market.mean()
        sectors -= sectors.mean(axis=0)
        idio -= idio.mean(axis=0)
    jumps = (rng.random((n_obs, n_assets)) < jump_prob) * rng.normal(0.0, jump_vol, (n_obs, n_assets))
    returns = drift + market[:, None] * beta_m + sectors[:, sector_of] * beta_s + idio + jumps

    names = [f"S{s + 1}_{k + 1:02d}" for s in range(n_sectors) for k in range(assets_per_sector)]
    index = pd.bdate_range("2015-01-01", periods=n_obs, name="date")
    frame = pd.DataFrame(returns, index=index, columns=names)
    sector = pd.Series([f"Sector {s + 1}" for s in sector_of], index=names, name="sector")
    return frame, sector
