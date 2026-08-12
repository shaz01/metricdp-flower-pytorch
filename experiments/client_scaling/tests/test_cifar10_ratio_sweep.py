"""Tests for the Stage A CIFAR-10 noise-ratio calibration sweep."""

from __future__ import annotations

import pytest

from experiments.client_scaling.scripts import cifar10_ratio_sweep as sweep


def test_fixed_ratio_pins_global_dp_noise_across_client_counts() -> None:
    """nm = ratio * n is what keeps global-DP's stdv (nm*clip/n) constant."""
    ratio, clip = 0.00625, sweep.HYPERPARAMS.clipping_norm
    stdvs = set()
    for clients in sweep.CLIENT_COUNTS:
        nm = sweep.noise_multiplier_for(
            privacy="global-dp", ratio=ratio, num_clients=clients
        )
        assert nm == pytest.approx(ratio * clients)
        stdvs.add(round(nm * clip / clients, 12))
    assert len(stdvs) == 1


def test_matched_noise_arm_cancels_the_distance_calibration() -> None:
    """Metric-privacy injects nm*clip/(d*n); scaling nm by d equalises it."""
    ratio, clients, distance = 0.00625, 48, 0.871
    nm = sweep.noise_multiplier_for(
        privacy="metric-privacy",
        ratio=ratio,
        num_clients=clients,
        arm="matched-noise",
        distance=distance,
    )

    clip = sweep.HYPERPARAMS.clipping_norm
    metric_stdv = nm * clip / (distance * clients)
    global_stdv = (ratio * clients) * clip / clients
    assert metric_stdv == pytest.approx(global_stdv)


def test_matched_noise_leaves_global_dp_untouched() -> None:
    fixed = sweep.noise_multiplier_for(
        privacy="global-dp", ratio=0.004, num_clients=8
    )
    matched = sweep.noise_multiplier_for(
        privacy="global-dp", ratio=0.004, num_clients=8,
        arm="matched-noise", distance=1.5,
    )
    assert fixed == matched


def test_matched_noise_requires_a_distance() -> None:
    with pytest.raises(ValueError):
        sweep.noise_multiplier_for(
            privacy="metric-privacy", ratio=0.004, num_clients=8, arm="matched-noise"
        )


def test_vanilla_ignores_the_ratio_and_runs_once() -> None:
    combos = sweep.build_combos(num_clients=8, ratios=(0.0025, 0.004, 0.00625))
    vanilla = [combo for combo in combos if combo.privacy == "vanilla"]

    assert len(vanilla) == 1
    assert vanilla[0].noise_multiplier == sweep.VANILLA_NOISE_MULTIPLIER


def test_build_combos_covers_every_ratio_for_each_dp_mechanism() -> None:
    ratios = (0.0025, 0.004, 0.00625)
    combos = sweep.build_combos(num_clients=48, ratios=ratios)

    for privacy in ("global-dp", "metric-privacy"):
        multipliers = sorted(
            combo.noise_multiplier for combo in combos if combo.privacy == privacy
        )
        assert multipliers == pytest.approx(sorted(r * 48 for r in ratios))
    assert len(combos) == 2 * len(ratios) + 1


def test_run_names_are_unique_and_record_the_arm() -> None:
    combos = sweep.build_combos(
        num_clients=8, ratios=(0.0025, 0.004), arm="matched-noise", distance=1.5
    )
    names = [combo.run_name() for combo in combos]

    assert len(set(names)) == len(names)
    assert all("matched-noise" in name for name in names)


def test_replace_view_uses_one_extra_canonical_partition() -> None:
    """IN-replace holds out the replacement client, so canonical = active + 1."""
    view = sweep.create_in_replace({"num-clients": 8})

    assert view.canonical_num_partitions == 9
    assert view.num_active_partitions == 8
    assert sweep.TARGET_PARTITION_ID in view.active_partition_ids


def test_replace_view_rejects_a_degenerate_federation() -> None:
    with pytest.raises(ValueError):
        sweep.create_in_replace({"num-clients": 1})


def test_build_combos_rejects_unknown_privacy_and_tiny_federations() -> None:
    with pytest.raises(ValueError):
        sweep.build_combos(num_clients=8, ratios=(0.004,), privacy_modes=("bogus",))
    with pytest.raises(ValueError):
        sweep.build_combos(num_clients=1, ratios=(0.004,))
