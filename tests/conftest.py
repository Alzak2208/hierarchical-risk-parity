import numpy as np
import pytest


def random_covariance(rng: np.random.Generator, n: int, n_obs: int | None = None) -> np.ndarray:
    """Sample covariance of correlated Gaussian data, well conditioned."""
    n_obs = n_obs or 4 * n + 20
    loadings = rng.normal(size=(n, 3))
    x = rng.normal(size=(n_obs, 3)) @ loadings.T + rng.normal(scale=rng.uniform(0.5, 1.5, n), size=(n_obs, n))
    return np.cov(x, rowvar=False)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260930)


@pytest.fixture
def three_asset_corr() -> np.ndarray:
    """Three-asset correlation matrix whose distances and merges are known."""
    return np.array([[1.0, 0.7, 0.2], [0.7, 1.0, -0.2], [0.2, -0.2, 1.0]])
