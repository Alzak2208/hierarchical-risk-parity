import numpy as np
import pytest
from scipy.cluster.hierarchy import leaves_list
from scipy.spatial.distance import squareform

from src.clustering import (
    correlation_distance,
    distance_of_distances,
    quasi_diagonal_order,
    tree_clustering,
    tree_clusters,
)


def test_correlation_distance_of_a_three_asset_example(three_asset_corr):
    expected = np.array([[0.0, 0.3873, 0.6325], [0.3873, 0.0, 0.7746], [0.6325, 0.7746, 0.0]])
    np.testing.assert_allclose(correlation_distance(three_asset_corr), expected, atol=5e-5)


def test_distance_of_distances_of_a_three_asset_example(three_asset_corr):
    expected = np.array([[0.0, 0.5659, 0.9747], [0.5659, 0.0, 1.1225], [0.9747, 1.1225, 0.0]])
    d_tilde = squareform(distance_of_distances(correlation_distance(three_asset_corr)))
    np.testing.assert_allclose(d_tilde, expected, atol=5e-5)


def test_single_linkage_merges_of_a_three_asset_example(three_asset_corr):
    link = tree_clustering(three_asset_corr, method="single")
    # assets 1 and 2 merge first at distance .5659, then asset 3 joins them at .9747
    assert sorted(link[0, :2].astype(int)) == [0, 1]
    assert link[0, 2] == pytest.approx(0.5659, abs=5e-5)
    assert sorted(link[1, :2].astype(int)) == [2, 3]
    assert link[1, 2] == pytest.approx(0.9747, abs=5e-5)
    assert link[1, 3] == 3


def test_tree_clusters_ignore_the_order_of_the_assets(rng, three_asset_corr):
    assert tree_clusters(tree_clustering(three_asset_corr)) == {frozenset({0, 1}), frozenset({0, 1, 2})}
    x = rng.normal(size=(300, 9))
    x[:, 5:] += x[:, :4]
    corr = np.corrcoef(x, rowvar=False)
    p = rng.permutation(9)
    permuted = tree_clusters(tree_clustering(corr[np.ix_(p, p)]))
    assert {frozenset(int(p[i]) for i in c) for c in permuted} == tree_clusters(tree_clustering(corr))


def test_correlation_distance_is_a_bounded_symmetric_metric(rng):
    x = rng.normal(size=(200, 8))
    d = correlation_distance(np.corrcoef(x, rowvar=False))
    assert np.allclose(d, d.T)
    assert np.all(np.diag(d) == 0)
    assert d.min() >= 0 and d.max() <= 1
    # triangle inequality on every triple
    n = d.shape[0]
    for i in range(n):
        assert np.all(d[i][:, None] <= d[i][None, :] + d + 1e-12)


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_quasi_diagonal_order_matches_dendrogram_leaves(rng, method):
    for _ in range(5):
        x = rng.normal(size=(300, 12))
        x[:, 6:] += x[:, :6] * rng.uniform(0.2, 1.0, 6)
        link = tree_clustering(np.corrcoef(x, rowvar=False), method=method)
        order = quasi_diagonal_order(link)
        assert order == leaves_list(link).tolist()
        assert sorted(order) == list(range(12))


def test_quasi_diagonal_order_groups_noisy_copies(rng):
    base = rng.normal(size=(2000, 3))
    x = np.hstack([base, base + 0.1 * rng.normal(size=(2000, 3))])  # asset k + 3 copies asset k
    order = quasi_diagonal_order(tree_clustering(np.corrcoef(x, rowvar=False)))
    position = {asset: i for i, asset in enumerate(order)}
    for k in range(3):
        assert abs(position[k] - position[k + 3]) == 1


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        correlation_distance(np.ones((2, 3)))
    with pytest.raises(ValueError):
        tree_clustering(np.eye(3), method="centroid")
    with pytest.raises(ValueError):
        tree_clustering(np.eye(1))
    with pytest.raises(ValueError):
        quasi_diagonal_order(np.ones((2, 3)))
    with pytest.raises(ValueError):
        tree_clusters(np.ones((2, 3)))
