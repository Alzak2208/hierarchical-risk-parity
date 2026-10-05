"""Tree clustering and quasi-diagonalisation (HRP stages 1 and 2).

Stage 1 turns a correlation matrix into a hierarchical tree:

1. correlation distance ``d_ij = sqrt((1 - rho_ij) / 2)``, a proper metric with values in [0, 1];
2. "distance of distances" ``d~_ij = ||D_i - D_j||_2``, the Euclidean distance between two
   columns of the distance matrix D. Each entry therefore depends on the whole correlation
   structure, not only on one pair of assets;
3. agglomerative clustering on ``d~`` with a linkage criterion (single linkage by default).

Stage 2 reorders the assets along the leaves of the resulting dendrogram, so that similar
assets sit next to each other and the covariance matrix becomes quasi-diagonal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist

LINKAGE_METHODS = ("single", "complete", "average", "ward")


def _as_square(matrix: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def correlation_distance(corr: ArrayLike) -> NDArray[np.float64]:
    """Correlation distance matrix ``d_ij = sqrt((1 - rho_ij) / 2)``.

    Perfectly correlated assets are at distance 0, uncorrelated ones at ``sqrt(1/2)`` and
    perfectly anti-correlated ones at 1. The diagonal is set to exactly 0.
    """
    rho = _as_square(corr, "corr")
    dist = np.sqrt(np.clip((1.0 - rho) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def distance_of_distances(dist: ArrayLike) -> NDArray[np.float64]:
    """Euclidean distance between the columns of a distance matrix, in condensed form.

    The condensed vector is what :func:`scipy.cluster.hierarchy.linkage` expects; use
    :func:`scipy.spatial.distance.squareform` to recover the square matrix.
    """
    d = _as_square(dist, "dist")
    return pdist(d, metric="euclidean")


def tree_clustering(
    corr: ArrayLike, method: str = "single", optimal_ordering: bool = False
) -> NDArray[np.float64]:
    """Stage 1: hierarchical tree of the assets, as a scipy linkage matrix.

    Each of the ``N - 1`` rows records the two merged clusters, the distance at which they
    merged and the number of original assets in the new cluster. Cluster ids ``>= N``
    refer to the cluster formed at row ``id - N``.

    The tree fixes which assets merge, but not which branch is drawn on the left: scipy
    decides that from the cluster ids, so the leaf order (and therefore the bisection of
    stage 3) depends on the order in which the assets are listed. With
    ``optimal_ordering=True`` the branches are flipped to minimise the distance between
    adjacent leaves (:func:`scipy.cluster.hierarchy.optimal_leaf_ordering`), which makes
    the leaf order a function of the data only.
    """
    if method not in LINKAGE_METHODS:
        raise ValueError(f"method must be one of {LINKAGE_METHODS}, got {method!r}")
    rho = _as_square(corr, "corr")
    if rho.shape[0] < 2:
        raise ValueError("tree clustering needs at least two assets")
    distances = distance_of_distances(correlation_distance(rho))
    link = linkage(distances, method=method)
    return optimal_leaf_ordering(link, distances) if optimal_ordering else link


def _check_linkage(link: ArrayLike) -> NDArray[np.float64]:
    z = np.asarray(link)
    if z.ndim != 2 or z.shape[1] != 4:
        raise ValueError("link must be a linkage matrix of shape (N - 1, 4)")
    return z


def tree_clusters(link: ArrayLike) -> frozenset[frozenset[int]]:
    """Clusters of the tree, each one given by the set of original assets it contains.

    Two linkage matrices describe the same tree when they give the same clusters, whatever
    the order of the merges and the side on which each branch is drawn.
    """
    z = _check_linkage(link)
    n_assets = z.shape[0] + 1
    members = [frozenset([i]) for i in range(n_assets)]
    for left, right in z[:, :2].astype(int):
        members.append(members[left] | members[right])
    return frozenset(members[n_assets:])


def quasi_diagonal_order(link: ArrayLike) -> list[int]:
    """Stage 2: order of the original assets along the leaves of the dendrogram.

    Starting from the root, every cluster is replaced by its two children (left first),
    depth first, until only original assets remain. Similar assets end up adjacent, which
    makes the reordered covariance matrix quasi-diagonal. The result matches
    :func:`scipy.cluster.hierarchy.leaves_list`.
    """
    z = _check_linkage(link)
    n_assets = z.shape[0] + 1
    order: list[int] = []
    stack = [2 * n_assets - 2]  # id of the root cluster
    while stack:
        node = stack.pop()
        if node < n_assets:
            order.append(node)
            continue
        left, right = (int(c) for c in z[node - n_assets, :2])
        stack.append(right)  # pushed first so that the left branch is visited first
        stack.append(left)
    return order
