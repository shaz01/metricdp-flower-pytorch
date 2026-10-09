"""Roles and client partitions must be deterministic, disjoint and match the documented label-stress design."""

import numpy as np
import pytest

from experiments.stacked_head import data as task_data


def synthetic_labels(per_class=6000):
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


def roles_digest(first_class):
    import hashlib

    r = task_data.build_roles(synthetic_labels(), first_class, 20261012)
    h = hashlib.sha256()
    parts = [r.validation, r.cohorts["A"], r.cohorts["B"], r.evaluation]
    parts += [r.public[(b, s)] for b in (32, 128) for s in range(3)] + [r.public[(32, s)] for s in range(3, 30)]
    for part in parts:
        h.update(np.asarray(part, dtype=np.int64).tobytes())
    return h.hexdigest()


def test_original_roles_are_pinned_so_every_earlier_result_keeps_its_data():
    # Digests of the layout used by the first gate/budgets matrices and the 30-set sweep; a change here silently
    # invalidates the committed results, so it must fail loudly.
    assert roles_digest(4) == "1fa7b1c71bef5eecf0d8140adfb90faa4708eeba1df1efa2c6cb81bb98b25fcb"
    assert roles_digest(0) == "0030b7c72a1108a7f99b0ee7a776f1ceeaad238f9bb9287c4a3d130b49aa2152"


def test_canonical_layout_has_all_regions_disjoint_and_balanced():
    labels = synthetic_labels()
    roles = task_data.build_roles(labels, 4, 20261012)
    assert set(k[0] for k in roles.public) == {32, 128, 512}
    assert len([k for k in roles.public if k[0] == 32]) == 63 and len([k for k in roles.public if k[0] == 128]) == 13 and len([k for k in roles.public if k[0] == 512]) == 10
    assert list(roles.cohorts) == ["A", "B", "C", "D"]
    ids = roles.all_indices()
    assert len(np.unique(ids)) == len(ids)
    for (budget, _), public in roles.public.items():
        assert np.array_equal(np.bincount(labels[public] - 4, minlength=4), [budget // 4] * 4)
    for cohort in roles.cohorts.values():
        assert np.array_equal(np.bincount(labels[cohort] - 4, minlength=4), [512] * 4)
    assert task_data.per_class_needed() == 4888


def test_test_split_evaluation_is_balanced_deterministic_and_class_restricted():
    test_labels = np.repeat(np.arange(10), 1000)
    a = task_data.test_evaluation_indices(test_labels, 4, 20261012)
    assert np.array_equal(a, task_data.test_evaluation_indices(test_labels, 4, 20261012))
    assert np.array_equal(np.bincount(test_labels[a] - 4, minlength=4), [500] * 4) and len(np.unique(a)) == 2000
    with pytest.raises(ValueError):
        task_data.test_evaluation_indices(np.repeat(np.arange(10), 100), 0, 1)
