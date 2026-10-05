import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from src.allocation import inverse_variance_weights
from src.cla import SingularCovarianceError, critical_line, min_variance_weights, weights_at
from tests.conftest import random_covariance


def exact_long_only_solution(cov, mean, lam):
    """Exact optimum of min 1/2 w'Sw - lam mu'w, sum(w) = 1, w >= 0, by enumerating free sets.

    For each candidate free set the equality-constrained KKT system is solved; the optimum is
    the candidate that is feasible (w >= 0) and whose excluded assets have a non-negative
    reduced gradient. Exponential in N, so only used on small problems.
    """
    n = cov.shape[0]
    for mask in range(1, 2**n):
        free = [i for i in range(n) if mask >> i & 1]
        kkt = np.zeros((len(free) + 1, len(free) + 1))
        kkt[:-1, :-1] = cov[np.ix_(free, free)]
        kkt[:-1, -1] = -1.0
        kkt[-1, :-1] = 1.0
        rhs = np.append(lam * mean[free], 1.0)
        sol = np.linalg.solve(kkt, rhs)
        w = np.zeros(n)
        w[free] = sol[:-1]
        gamma = sol[-1]
        gradient = cov @ w - lam * mean - gamma
        if w.min() >= -1e-10 and np.all(np.delete(gradient, free) >= -1e-10):
            return w
    raise AssertionError("no KKT point found")


def solve_qp(cov, mean, lam, lower=0.0, upper=1.0):
    """Reference solution of min 1/2 w'Sw - lam mu'w with SLSQP."""
    n = cov.shape[0]
    result = minimize(
        lambda w: 0.5 * w @ cov @ w - lam * mean @ w,
        np.full(n, 1.0 / n),
        jac=lambda w: cov @ w - lam * mean,
        bounds=[(lower, upper)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    assert result.success
    return result.x


@pytest.mark.parametrize("n", [3, 6, 10, 20])
def test_min_variance_matches_a_generic_qp_solver(rng, n):
    for _ in range(5):
        cov = random_covariance(rng, n)
        w = min_variance_weights(cov)
        np.testing.assert_allclose(w, solve_qp(cov, np.zeros(n), 0.0), atol=1e-6)


def test_whole_frontier_matches_a_generic_qp_solver(rng):
    for _ in range(10):
        n = int(rng.integers(3, 10))
        cov = random_covariance(rng, n)
        mean = rng.normal(0.05, 0.1, n)
        points = critical_line(cov, mean)
        finite = [p.lam for p in points if np.isfinite(p.lam)]
        for lam in rng.uniform(0.0, 1.2 * max(finite), size=4):
            np.testing.assert_allclose(
                weights_at(points, lam), exact_long_only_solution(cov, mean, lam), atol=1e-8
            )


def test_turning_points_are_feasible_and_ordered(rng):
    cov = random_covariance(rng, 12)
    points = critical_line(cov, rng.normal(size=12))
    lams = [p.lam for p in points]
    assert np.isinf(lams[0]) and lams[-1] == 0.0
    assert np.all(np.diff(lams[1:]) < 0)
    for p in points:
        assert p.weights.sum() == pytest.approx(1.0, abs=1e-10)
        assert p.weights.min() >= -1e-12 and p.weights.max() <= 1 + 1e-12


def test_first_turning_point_holds_the_highest_mean_asset(rng):
    mean = np.array([0.02, 0.09, 0.05, 0.01])
    points = critical_line(random_covariance(rng, 4), mean)
    np.testing.assert_allclose(points[0].weights, [0.0, 1.0, 0.0, 0.0])


def test_min_variance_is_ivp_when_covariance_is_diagonal(rng):
    cov = np.diag(rng.uniform(0.01, 0.1, 8))
    np.testing.assert_allclose(min_variance_weights(cov), inverse_variance_weights(cov), atol=1e-12)


def test_min_variance_does_not_depend_on_the_means(rng):
    cov = random_covariance(rng, 9)
    w_default = critical_line(cov)[-1].weights
    w_other = critical_line(cov, rng.normal(size=9))[-1].weights
    np.testing.assert_allclose(w_default, w_other, atol=1e-10)


def test_custom_bounds_are_respected(rng):
    cov = random_covariance(rng, 8)
    w = min_variance_weights(cov, lower=0.02, upper=0.25)
    assert w.min() >= 0.02 - 1e-12 and w.max() <= 0.25 + 1e-12
    np.testing.assert_allclose(w, solve_qp(cov, np.zeros(8), 0.0, 0.02, 0.25), atol=1e-6)


def test_a_free_block_that_cannot_be_inverted_raises():
    # a zero-variance asset attracts all the minimum-variance weight and makes the block singular
    cov = np.diag([0.04, 0.09, 0.0])
    with pytest.raises(SingularCovarianceError):
        min_variance_weights(cov)


def test_rank_deficient_covariance_either_raises_or_stays_feasible(rng):
    # fewer observations than assets: the full matrix is singular, but the long-only
    # constraint often keeps the free block invertible
    for _ in range(10):
        cov = np.cov(rng.normal(size=(15, 25)), rowvar=False)
        try:
            w = min_variance_weights(cov)
        except SingularCovarianceError:
            continue
        assert w.sum() == pytest.approx(1.0) and w.min() >= -1e-10


def test_dataframe_input_returns_series(rng):
    cov = pd.DataFrame(random_covariance(rng, 3), index=list("xyz"), columns=list("xyz"))
    w = min_variance_weights(cov)
    assert isinstance(w, pd.Series) and list(w.index) == list("xyz")


def test_infeasible_or_invalid_inputs_raise():
    with pytest.raises(ValueError):
        critical_line(np.eye(3), upper=0.2)  # cannot be fully invested
    with pytest.raises(ValueError):
        critical_line(np.eye(3), lower=0.5)
    with pytest.raises(ValueError):
        critical_line(np.array([[1.0, 0.5], [0.0, 1.0]]))  # not symmetric
    with pytest.raises(ValueError):
        weights_at(critical_line(np.eye(2)), -1.0)
