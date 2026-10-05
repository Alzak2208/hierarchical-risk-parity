import numpy as np
import pytest

from src import metrics


def test_max_drawdown_of_a_known_path():
    returns = [0.10, -0.20, 0.05, 0.30]  # wealth 1.10, 0.88, 0.924, 1.2012
    assert metrics.max_drawdown(returns) == pytest.approx(0.20)


def test_max_drawdown_includes_a_loss_on_the_first_day():
    assert metrics.max_drawdown([-0.10, 0.05]) == pytest.approx(0.10)


def test_cagr_and_volatility():
    returns = np.full(252, 0.0004)
    assert metrics.cagr(returns) == pytest.approx(1.0004**252 - 1)
    assert metrics.annualized_volatility([0.01, -0.01, 0.01, -0.01], 4) == pytest.approx(
        np.std([0.01, -0.01, 0.01, -0.01], ddof=1) * 2
    )


def test_sharpe_ratio_sign_and_scale():
    returns = np.array([0.01, 0.02, 0.0, 0.01])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(returns) == pytest.approx(expected)
    assert np.isnan(metrics.sharpe_ratio([0.01, 0.01]))


def test_allocation_quality_metrics():
    assert metrics.effective_number_of_assets([0.25] * 4) == pytest.approx(4.0)
    assert metrics.effective_number_of_assets([1.0, 0.0]) == pytest.approx(1.0)
    assert metrics.top_k_concentration([0.1, 0.5, 0.2, 0.2], k=2) == pytest.approx(0.7)
    assert metrics.weight_instability(np.tile([0.5, 0.5], (3, 1))) == pytest.approx(0.0)
    assert metrics.weight_instability(np.array([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(1.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        metrics.cagr([0.01])
    with pytest.raises(ValueError):
        metrics.weight_instability([0.5, 0.5])
