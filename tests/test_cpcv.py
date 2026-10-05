from math import comb

import numpy as np
import pytest

from src.cpcv import CombinatorialPurgedCV


@pytest.mark.parametrize(("n_groups", "k"), [(6, 2), (10, 2), (8, 3)])
def test_split_and_path_counts(n_groups, k):
    cv = CombinatorialPurgedCV(n_groups, k)
    assert len(cv.split(500)) == cv.n_splits == comb(n_groups, k)
    assert cv.paths().shape == (cv.n_paths, n_groups)
    assert cv.n_paths == comb(n_groups - 1, k - 1)


def test_groups_partition_the_history():
    cv = CombinatorialPurgedCV(7, 2)
    bounds = cv.group_bounds(503)
    assert bounds[0][0] == 0 and bounds[-1][1] == 503
    assert all(prev[1] == nxt[0] for prev, nxt in zip(bounds, bounds[1:], strict=False))
    sizes = [end - start for start, end in bounds]
    assert max(sizes) - min(sizes) <= 1


def test_every_observation_is_tested_the_same_number_of_times():
    n_obs = 400
    cv = CombinatorialPurgedCV(8, 3)
    bounds = cv.group_bounds(n_obs)
    counts = np.zeros(n_obs, dtype=int)
    for split in cv.split(n_obs):
        for g in split.test_groups:
            counts[bounds[g][0] : bounds[g][1]] += 1
    assert np.all(counts == cv.n_paths)


def test_training_sets_exclude_test_purge_and_embargo_windows():
    n_obs, purge, embargo = 1000, 5, 12
    cv = CombinatorialPurgedCV(10, 2, purge=purge, embargo=embargo)
    bounds = cv.group_bounds(n_obs)
    for split in cv.split(n_obs):
        train = set(split.train.tolist())
        for g in split.test_groups:
            start, end = bounds[g]
            forbidden = range(max(0, start - purge), min(n_obs, end + embargo))
            assert train.isdisjoint(forbidden)
        # everything else is kept
        expected = n_obs - sum(
            min(n_obs, bounds[g][1] + embargo) - max(0, bounds[g][0] - purge) for g in split.test_groups
        )
        assert len(train) >= expected  # windows of adjacent test groups may overlap


def test_paths_cover_every_group_once_with_consistent_splits():
    cv = CombinatorialPurgedCV(9, 3)
    splits = cv.split(900)
    table = cv.paths()
    assert np.all(table >= 0)
    for path in table:
        for g, s in enumerate(path):
            assert g in splits[s].test_groups
    # each split fills exactly k (path, group) cells
    _, uses = np.unique(table, return_counts=True)
    assert np.all(uses == 3)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(1, 1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(5, 5)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(5, 2, embargo=-1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(10, 2).group_bounds(5)
