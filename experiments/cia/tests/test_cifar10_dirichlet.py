"""Tests for the Dirichlet-only CIFAR-10 CIA chunk builder."""

from __future__ import annotations

import sys

import pytest

from experiments.cia.scripts import cifar10_dirichlet


def test_build_combos_are_dirichlet_with_explicit_alpha() -> None:
    combos = cifar10_dirichlet.build_combos(dirichlet_alpha=0.5)
    in_combos = [c for c in combos if c.name_prefix.endswith("in-remove")]
    out_combos = [c for c in combos if c.name_prefix.endswith("out-remove")]

    assert len(combos) == len(cifar10_dirichlet.ADJACENCIES) * len(
        cifar10_dirichlet.PRIVACY_MODES
    )
    assert {c.partition for c in combos} == {"dirichlet"}
    assert {c.dirichlet_alpha for c in combos} == {0.5}
    assert {c.data_module for c in combos} == {
        "experiments.cia.scripts.cifar10_dirichlet:create_in_remove",
        "experiments.cia.scripts.cifar10_dirichlet:create_out_remove",
    }
    assert {c.num_clients for c in in_combos} == {
        cifar10_dirichlet.CANONICAL_NUM_CLIENTS
    }
    assert {c.num_clients for c in out_combos} == {
        cifar10_dirichlet.CANONICAL_NUM_CLIENTS - 1
    }


@pytest.mark.parametrize("canonical", [8, 16])
def test_out_view_drops_exactly_the_target_client(canonical: int) -> None:
    combos = cifar10_dirichlet.build_combos(
        privacy_modes=("vanilla",),
        dirichlet_alpha=0.1,
        canonical_num_clients=canonical,
    )
    by_adjacency = {c.name_prefix: c for c in combos}

    assert by_adjacency["cifar10-in-remove"].num_clients == canonical
    assert by_adjacency["cifar10-out-remove"].num_clients == canonical - 1


@pytest.mark.parametrize("alpha", [0.0, -0.1, float("inf"), float("nan")])
def test_build_combos_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        cifar10_dirichlet.build_combos(dirichlet_alpha=alpha)


def test_build_combos_rejects_degenerate_client_counts() -> None:
    with pytest.raises(ValueError):
        cifar10_dirichlet.build_combos(
            dirichlet_alpha=0.5, canonical_num_clients=1
        )


@pytest.mark.parametrize("canonical", [8, 16])
def test_both_views_resolve_the_same_canonical_partitioning(canonical: int) -> None:
    in_config = {"num-clients": canonical}
    out_config = {"num-clients": canonical - 1}

    assert cifar10_dirichlet._canonical_clients(in_config, "in-remove") == canonical
    assert cifar10_dirichlet._canonical_clients(out_config, "out-remove") == canonical


def test_combos_carry_the_calibrated_noise_multiplier() -> None:
    combos = cifar10_dirichlet.build_combos(
        dirichlet_alpha=0.5, canonical_num_clients=8
    )
    assert {c.noise_multiplier for c in combos} == {0.0182}


def test_fixed_ratio_scales_by_active_client_count() -> None:
    combos = cifar10_dirichlet.build_combos(
        canonical_num_clients=8,
        dirichlet_alpha=0.5,
        privacy_modes=("global-dp",),
        noise_ratio=0.00625,
    )
    by_adjacency = {combo.name_prefix: combo for combo in combos}

    assert by_adjacency["cifar10-in-remove"].noise_multiplier == pytest.approx(0.05)
    assert by_adjacency["cifar10-out-remove"].noise_multiplier == pytest.approx(0.04375)


def test_fixed_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError, match="noise_ratio"):
        cifar10_dirichlet.build_combos(
            dirichlet_alpha=0.5, noise_ratio=0
        )


def test_cli_requires_dirichlet_alpha() -> None:
    parser = cifar10_dirichlet._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--privacy", "vanilla"])


def test_results_path_is_derived_from_client_count() -> None:
    assert cifar10_dirichlet.RESULTS_ROOT == (
        cifar10_dirichlet.PROJECT_ROOT / "results" / "dirichlet" / "cifar10"
    )


def test_disjoint_chunks_receive_disjoint_output_directories(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cifar10_dirichlet, "RESULTS_ROOT", tmp_path)
    common = {
        "clients": 48,
        "privacy": "vanilla",
        "adjacencies": ("in-remove", "out-remove"),
        "seeds": (42,),
        "noise_ratio": None,
    }

    weak = cifar10_dirichlet._chunk_output_dir(
        dirichlet_alpha=10.0, **common
    )
    strong = cifar10_dirichlet._chunk_output_dir(
        dirichlet_alpha=0.1, **common
    )

    assert weak != strong
    assert weak.parent == tmp_path / "48_clients"
    assert strong.parent == tmp_path / "48_clients"


def test_main_writes_to_unique_chunk_below_client_count_directory(
    tmp_path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(cifar10_dirichlet, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(cifar10_dirichlet, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(
        cifar10_dirichlet,
        "run_attack",
        lambda **kwargs: calls.append(kwargs) or [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cifar10_dirichlet",
            "--privacy",
            "vanilla",
            "--dirichlet-alpha",
            "0.1",
            "--clients",
            "48",
        ],
    )

    cifar10_dirichlet.main()

    expected = (
        tmp_path
        / "48_clients"
        / "alpha-0p1__vanilla__in-remove-out-remove__seeds-42__noise-historical"
    )
    assert calls[0]["output_dir"] == expected
    assert calls[0]["log_path"] == expected / "progress.log"
    assert {combo.partition for combo in calls[0]["combos"]} == {"dirichlet"}
    assert {combo.dirichlet_alpha for combo in calls[0]["combos"]} == {0.1}
