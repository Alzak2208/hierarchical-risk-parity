import numpy as np
import pandas as pd
import pytest

from src.allocation import (
    cluster_variance,
    cov_to_corr,
    equal_weights,
    hrp_weights,
    inverse_variance_weights,
    recursive_bisection,
)
from src.clustering import quasi_diagonal_order, tree_clustering
from tests.conftest import random_covariance


def test_inverse_variance_weights_formula():
    cov = np.diag([0.01, 0.04, 0.02])
    expected = np.array([100.0, 25.0, 50.0]) / 175.0
    np.testing.assert_allclose(inverse_variance_weights(cov), expected)


def test_hrp_equals_ivp_for_two_assets(rng):
    # with two assets the bisection split is the inverse-variance split
    cov = random_covariance(rng, 2)
    np.testing.assert_allclose(hrp_weights(cov), inverse_variance_weights(cov), atol=1e-14)


def test_hrp_equals_ivp_for_a_diagonal_covariance(rng):
    cov = np.diag(rng.uniform(0.01, 0.2, 15))
    np.testing.assert_allclose(hrp_weights(cov), inverse_variance_weights(cov), atol=1e-14)


@pytest.mark.parametrize("n", [3, 10, 37])
def test_hrp_weights_are_long_only_and_fully_invested(rng, n):
    for _ in range(10):
        cov = random_covariance(rng, n)
        w = hrp_weights(cov)
        assert w.shape == (n,)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert w.min() > 0 and w.max() <= 1


def test_hrp_handles_a_singular_covariance(rng):
    x = rng.normal(size=(300, 5))
    x = np.hstack([x, x[:, [0]]])  # an exact duplicate: the covariance matrix is singular
    cov = np.cov(x, rowvar=False)
    assert np.linalg.matrix_rank(cov) < cov.shape[0]
    w = hrp_weights(cov)
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0) and w.min() > 0
    order = quasi_diagonal_order(tree_clustering(cov_to_corr(cov)))
    assert abs(order.index(0) - order.index(5)) == 1  # the duplicates sit next to each other


def test_hrp_handles_more_assets_than_observations(rng):
    x = rng.normal(size=(30, 60))
    w = hrp_weights(np.cov(x, rowvar=False))
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)


def test_recursive_bisection_split_follows_cluster_variances():
    cov = np.diag([1.0, 1.0, 4.0, 4.0])
    w = recursive_bisection(cov, [0, 1, 2, 3])
    # halves {0,1} and {2,3}: IVP variances 0.5 and 2, so alpha = 1 - 0.5 / 2.5 = 0.8
    np.testing.assert_allclose(w, [0.4, 0.4, 0.1, 0.1])


def test_cluster_variance_of_a_single_asset_is_its_variance():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert cluster_variance(cov, [1]) == pytest.approx(0.09)


def test_dataframe_inputs_return_labelled_series(rng):
    cov = pd.DataFrame(random_covariance(rng, 4), index=list("ABCD"), columns=list("ABCD"))
    for w in (hrp_weights(cov), inverse_variance_weights(cov)):
        assert isinstance(w, pd.Series)
        assert list(w.index) == list("ABCD")


def test_explicit_correlation_is_used_for_the_clustering(rng):
    cov = random_covariance(rng, 6)
    np.testing.assert_allclose(hrp_weights(cov), hrp_weights(cov, cov_to_corr(cov)))


def test_single_asset_and_equal_weights():
    np.testing.assert_allclose(hrp_weights(np.array([[0.04]])), [1.0])
    np.testing.assert_allclose(equal_weights(4), [0.25] * 4)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        hrp_weights(np.ones((2, 3)))
    with pytest.raises(ValueError):
        inverse_variance_weights(np.diag([0.01, 0.0]))
    with pytest.raises(ValueError):
        hrp_weights(np.eye(3), corr=np.eye(2))
    with pytest.raises(ValueError):
        equal_weights(0)


def _weights_after_permutation(cov, permutation, **kwargs):
    w = hrp_weights(cov[np.ix_(permutation, permutation)], **kwargs)
    back = np.empty_like(w)
    back[permutation] = w
    return back


def test_optimal_leaf_ordering_makes_hrp_independent_of_the_asset_order(rng):
    cov = random_covariance(rng, 20)
    reference = hrp_weights(cov, optimal_ordering=True)
    for _ in range(10):
        permutation = rng.permutation(20)
        np.testing.assert_allclose(
            _weights_after_permutation(cov, permutation, optimal_ordering=True), reference, atol=1e-12
        )
