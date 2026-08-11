"""Tests for the CIFAR-10 removal-adjacency CIA chunk builder."""

from __future__ import annotations

import pytest

from experiments.cia.scripts import cifar10_remove


def test_build_combos_defaults_to_canonical_client_count() -> None:
    combos = cifar10_remove.build_combos()
    in_combos = [c for c in combos if c.name_prefix.endswith("in-remove")]
    out_combos = [c for c in combos if c.name_prefix.endswith("out-remove")]

    assert len(combos) == len(cifar10_remove.ADJACENCIES) * len(
        cifar10_remove.PRIVACY_MODES
    )
    assert {c.num_clients for c in in_combos} == {cifar10_remove.CANONICAL_NUM_CLIENTS}
    assert {c.num_clients for c in out_combos} == {
        cifar10_remove.CANONICAL_NUM_CLIENTS - 1
    }


@pytest.mark.parametrize("canonical", [8, 16])
def test_out_view_drops_exactly_the_target_client(canonical: int) -> None:
    combos = cifar10_remove.build_combos(
        privacy_modes=("vanilla",), canonical_num_clients=canonical
    )
    by_adjacency = {c.name_prefix: c for c in combos}

    assert by_adjacency["cifar10-in-remove"].num_clients == canonical
    assert by_adjacency["cifar10-out-remove"].num_clients == canonical - 1


def test_build_combos_rejects_degenerate_client_counts() -> None:
    with pytest.raises(ValueError):
        cifar10_remove.build_combos(canonical_num_clients=1)


@pytest.mark.parametrize("canonical", [8, 16])
def test_both_views_resolve_the_same_canonical_partitioning(canonical: int) -> None:
    """The shadow split is derived from the canonical count, so a matched
    IN/OUT pair must agree on it despite training different client counts."""
    in_config = {"num-clients": canonical}
    out_config = {"num-clients": canonical - 1}

    assert cifar10_remove._canonical_clients(in_config, "in-remove") == canonical
    assert cifar10_remove._canonical_clients(out_config, "out-remove") == canonical


def test_combos_carry_the_calibrated_noise_multiplier() -> None:
    combos = cifar10_remove.build_combos(canonical_num_clients=8)
    assert {c.noise_multiplier for c in combos} == {0.0182}
