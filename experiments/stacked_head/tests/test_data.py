"""Roles and client partitions must be deterministic, disjoint and match the documented label-stress design."""

import numpy as np
import pytest

from experiments.stacked_head import data as task_data


def synthetic_labels(per_class=2000):
    return np.repeat(np.arange(10), per_class)


def test_parse_task_accepts_both_names_and_rejects_bad_ranges():
    assert task_data.parse_task("kmnist_classes4to7") == ("kmnist", 4)
    assert task_data.parse_task("mnist_digits0to3") == ("mnist", 0)
    for bad in ("kmnist_classes0to4", "cifar_classes0to3", "kmnist_0to3"):
        with pytest.raises(ValueError):
            task_data.parse_task(bad)


def test_roles_are_disjoint_balanced_and_deterministic():
    labels = synthetic_labels()
    roles = task_data.build_roles(labels, 4, 20261012)
    ids = roles.all_indices()
    assert len(np.unique(ids)) == len(ids)
    assert set(np.unique(labels[ids])) == {4, 5, 6, 7}
    for (budget, _), public in roles.public.items():
        assert np.array_equal(np.bincount(labels[public] - 4, minlength=4), [budget // 4] * 4)
    for pool, per_class in [(roles.validation, 128), (roles.cohorts["A"], 512), (roles.cohorts["B"], 512), (roles.evaluation, 512)]:
        assert np.array_equal(np.bincount(labels[pool] - 4, minlength=4), [per_class] * 4)
    again = task_data.build_roles(labels, 4, 20261012)
    assert all(np.array_equal(roles.public[key], again.public[key]) for key in roles.public)
    other = task_data.build_roles(labels, 4, 1)
    assert not np.array_equal(roles.validation, other.validation)


def test_roles_fail_loudly_when_a_class_is_too_small():
    with pytest.raises(ValueError, match="needed"):
        task_data.build_roles(synthetic_labels(per_class=100), 0, 1)


def test_client_partitions_follow_the_label_stress_design():
    labels = synthetic_labels()
    roles = task_data.build_roles(labels, 0, 7)
    cohort = roles.cohorts["A"]
    parts = task_data.client_partitions(labels[cohort], 7, 0)
    assert len(parts) == 8 and all(len(part) == task_data.CLIENT_SIZE for part in parts)
    for client, part in enumerate(parts):
        counts = np.bincount(labels[cohort][part], minlength=4)
        expected = np.full(4, 17)
        expected[client % 4] = 205
        assert np.array_equal(counts, expected)
    assert len(np.unique(np.concatenate(parts))) == 8 * task_data.CLIENT_SIZE


def test_extra_public_sets_never_move_earlier_roles_and_stay_disjoint():
    labels = synthetic_labels(per_class=4000)
    base = task_data.build_roles(labels, 4, 20261012)
    more = task_data.build_roles(labels, 4, 20261012, public_sets=30)
    assert len(more.public) == 2 * 30
    assert np.array_equal(base.validation, more.validation) and np.array_equal(base.evaluation, more.evaluation)
    assert all(np.array_equal(base.cohorts[c], more.cohorts[c]) for c in base.cohorts)
    assert all(np.array_equal(base.public[key], more.public[key]) for key in base.public)
    ids = more.all_indices()
    assert len(np.unique(ids)) == len(ids)
    for s in range(3, 30):
        assert np.array_equal(np.bincount(labels[more.public[(32, s)]] - 4, minlength=4), [8] * 4)
    with pytest.raises(ValueError):
        task_data.build_roles(labels, 4, 1, public_sets=2)
