import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, run_cpcv_backtest, simulate_path
from src.cpcv import CombinatorialPurgedCV


def test_zero_returns_only_pay_for_target_changes():
    returns = np.zeros((40, 2))
    bounds = [(0, 20), (20, 40)]
    targets = np.array([[0.5, 0.5], [0.8, 0.2]])
    gross, net, turnover = simulate_path(returns, bounds, targets, rebalance_every=5, cost_rate=0.001)
    assert np.all(gross == 0)
    assert turnover[0] == pytest.approx(1.0)  # initial build from cash
    assert turnover[20] == pytest.approx(0.6)  # |0.8 - 0.5| + |0.2 - 0.5|
    assert turnover.sum() == pytest.approx(1.6)  # no drift, so periodic rebalances are free
    assert net[0] == pytest.approx(-0.001) and net[20] == pytest.approx(-0.0006)


def test_drift_creates_turnover_at_periodic_rebalances():
    returns = np.zeros((10, 2))
    returns[0] = [0.10, 0.0]  # asset 0 outperforms on the first day
    _, _, turnover = simulate_path(returns, [(0, 10)], np.array([[0.5, 0.5]]), 5, 0.0)
    drifted = np.array([0.55, 0.5]) / 1.05
    assert turnover[5] == pytest.approx(np.abs(drifted - 0.5).sum())


def test_gross_return_is_the_weighted_return_on_rebalance_days():
    returns = np.array([[0.01, -0.02], [0.03, 0.01]])
    gross, _, _ = simulate_path(returns, [(0, 2)], np.array([[0.25, 0.75]]), 1, 0.0)
    np.testing.assert_allclose(gross, [0.25 * 0.01 - 0.75 * 0.02, 0.25 * 0.03 + 0.75 * 0.01])


def _toy_returns(rng, n_obs=300, n_assets=6):
    data = rng.normal(0.0003, 0.01, size=(n_obs, n_assets))
    index = pd.bdate_range("2020-01-01", periods=n_obs)
    return pd.DataFrame(data, index=index, columns=[f"A{i}" for i in range(n_assets)])


def test_backtest_end_to_end(rng):
    returns = _toy_returns(rng)
    cv = CombinatorialPurgedCV(6, 2, embargo=3)
    result = run_cpcv_backtest(returns, cv=cv, config=BacktestConfig(rebalance_every=10, cost_bps=5))
    summary = result.summary()
    assert list(summary.index) == ["HRP", "CLA", "IVP", "1/N"]
    assert (summary["failed_splits"] == 0).all()
    paths = result.path_metrics()
    assert len(paths) == 4 * cv.n_paths
    assert summary.loc["1/N", "weight_instability"] == pytest.approx(0.0)
    assert summary.loc["1/N", "effective_assets"] == pytest.approx(6.0)
    # costs can only reduce performance
    assert (summary["cost_drag"] >= 0).all()


def test_failed_splits_are_recorded_and_their_paths_dropped(rng):
    returns = _toy_returns(rng)
    cv = CombinatorialPurgedCV(4, 2)
    calls = {"n": 0}

    def fails_on_first_split(cov, corr):
        calls["n"] += 1
        if calls["n"] == 1:
            raise np.linalg.LinAlgError("singular")
        return np.full(cov.shape[0], 1.0 / cov.shape[0])

    result = run_cpcv_backtest(returns, {"fragile": fails_on_first_split}, cv=cv)
    assert result.summary().loc["fragile", "failed_splits"] == 1
    # split 0 tests groups (0, 1): the paths that need it are dropped
    needing_split_0 = int((cv.paths() == 0).any(axis=1).sum())
    assert len(result.path_metrics()) == cv.n_paths - needing_split_0


def test_hrp_runs_with_more_assets_than_training_observations(rng):
    returns = _toy_returns(rng, n_obs=60, n_assets=50)
    result = run_cpcv_backtest(returns, cv=CombinatorialPurgedCV(4, 2))
    assert result.summary().loc["HRP", "failed_splits"] == 0


def test_invalid_allocator_output_raises(rng):
    returns = _toy_returns(rng)
    with pytest.raises(ValueError):
        run_cpcv_backtest(returns, {"bad": lambda cov, corr: np.ones(cov.shape[0])})


def test_missing_values_are_rejected(rng):
    returns = _toy_returns(rng)
    returns.iloc[3, 2] = np.nan
    with pytest.raises(ValueError):
        run_cpcv_backtest(returns)
