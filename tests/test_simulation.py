import numpy as np

from src.simulation import monte_carlo_sample, numerical_example, sector_universe


def test_numerical_example_shape_and_structure():
    data, parents = numerical_example(n_obs=5000, seed=1)
    assert data.shape == (5000, 10) and list(data.columns) == list(range(1, 11))
    corr = data.corr().to_numpy()
    for k, parent in enumerate(parents):
        assert corr[parent, 5 + k] > 0.9  # each copy tracks the series it was built from


def test_numerical_example_is_reproducible():
    a, _ = numerical_example(n_obs=100, seed=3)
    b, _ = numerical_example(n_obs=100, seed=3)
    assert a.equals(b)


def test_monte_carlo_sample_places_shocks_out_of_sample():
    x, parents = monte_carlo_sample(np.random.default_rng(0))
    assert x.shape == (520, 10)
    in_sample = x[:260]
    assert np.abs(in_sample).max() < 0.1  # no jump in the estimation window
    common = x[260:, [parents[0], 5]]
    assert (common == 2.0).all(axis=1).any() and (common == -0.5).all(axis=1).any()
    assert (x[260:, parents[-1]] == 2.0).any() and (x[260:, parents[-1]] == -0.5).any()


def test_sector_universe_has_a_block_correlation_structure():
    returns, sector = sector_universe(n_sectors=3, assets_per_sector=4, n_obs=2000, seed=11)
    assert returns.shape == (2000, 12)
    corr = returns.corr().to_numpy()
    same = sector.to_numpy()[:, None] == sector.to_numpy()[None, :]
    off_diagonal = ~np.eye(12, dtype=bool)
    assert corr[same & off_diagonal].mean() > corr[~same].mean() + 0.1


def test_sector_universe_demeaning_keeps_the_covariance():
    centred, _ = sector_universe(n_obs=500, seed=5)
    raw, _ = sector_universe(n_obs=500, seed=5, demean=False)
    np.testing.assert_allclose(centred.cov(), raw.cov(), atol=1e-12)
    jumps_free_mean = centred.mean().median()
    assert abs(jumps_free_mean - 2e-4) < 2e-4
