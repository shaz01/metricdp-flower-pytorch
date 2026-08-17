"""Tests for the Fashion-MNIST removal-adjacency CIA stage builder."""

from __future__ import annotations

import pytest

from experiments.cia.scripts import fashion_mnist_remove


def test_build_combos_defaults_to_canonical_client_count() -> None:
    combos = fashion_mnist_remove.build_combos()
    in_combos = [c for c in combos if c.name_prefix.endswith("in-remove")]
    out_combos = [c for c in combos if c.name_prefix.endswith("out-remove")]

    assert len(combos) == len(fashion_mnist_remove.ADJACENCIES) * len(
        fashion_mnist_remove.PRIVACY_MODES
    )
    assert {c.num_clients for c in in_combos} == {
        fashion_mnist_remove.CANONICAL_NUM_CLIENTS
    }
    assert {c.num_clients for c in out_combos} == {
        fashion_mnist_remove.CANONICAL_NUM_CLIENTS - 1
    }


def test_build_combos_accepts_homogeneous_partition() -> None:
    combos = fashion_mnist_remove.build_combos(
        canonical_num_clients=8, partition="homogeneous"
    )
    assert {c.partition for c in combos} == {"homogeneous"}


def test_build_combos_defaults_to_non_iid_partition() -> None:
    combos = fashion_mnist_remove.build_combos(canonical_num_clients=8)
    assert {c.partition for c in combos} == {"non-iid"}


def test_combos_carry_the_calibrated_noise_multiplier() -> None:
    combos = fashion_mnist_remove.build_combos(canonical_num_clients=8)
    assert {c.noise_multiplier for c in combos} == {0.03211721152213856}


def test_fixed_ratio_scales_by_active_client_count() -> None:
    combos = fashion_mnist_remove.build_combos(
        canonical_num_clients=8,
        privacy_modes=("global-dp",),
        noise_ratio=0.001,
    )
    by_adjacency = {combo.name_prefix: combo for combo in combos}

    assert by_adjacency["fashion-mnist-in-remove"].noise_multiplier == pytest.approx(
        0.008
    )
    assert by_adjacency["fashion-mnist-out-remove"].noise_multiplier == pytest.approx(
        0.007
    )


def test_build_combos_rejects_degenerate_client_counts() -> None:
    with pytest.raises(ValueError):
        fashion_mnist_remove.build_combos(canonical_num_clients=1)
