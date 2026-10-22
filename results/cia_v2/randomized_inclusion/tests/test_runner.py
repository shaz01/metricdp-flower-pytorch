"""Randomized-inclusion runner (CIA roadmap v2, M0): planning only, no data and no training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from results.cia_v2.randomized_inclusion import runner
from results.cia_v2.randomized_inclusion.surrogate import label_matched_indices


def plan(**kwargs):
    settings = dict(config="gdp-r1x", privacy="global-dp", noise_ratio=runner.FRONTIER_RATIO, replicates=8,
                    design_seed=0)
    settings.update(kwargs)
    return runner.build_runs(**settings)


def test_candidates_are_fixed_before_any_run():
    assert runner.CANDIDATES[:5] == (23, 7, 40, 18, 33)
    assert len(runner.CANDIDATES) == len(set(runner.CANDIDATES)) == 32
    # The original 16-candidate panel (before the 2026-10-17 escalation) is contained in the 32.
    assert set(runner.CANDIDATES) >= {23, 7, 40, 18, 33, 1, 22, 16, 31, 47, 28, 27, 8, 29, 25, 43}
    assert runner.TOP_DECILE == (23, 7, 40, 18, 33)


def test_candidate_rule_reproduces_the_fixed_list():
    sizes = {t: 1000 - t for t in range(48)}  # size falls with the index
    top, others = runner.select_candidates(sizes)
    assert top == (0, 1, 2, 3, 4)
    ascending = sorted(range(5, 48), key=lambda t: sizes[t])
    assert others == tuple(ascending[round((k + 0.5) * 43 / 27)] for k in range(27))


def test_every_run_has_thirty_two_active_clients_and_follows_the_design():
    runs = plan()
    design = runner.design_matrix(8, design_seed=0)
    for j, run in enumerate(runs):
        assert len(run.active) == 32
        assert len(run.included) == 16
        assert set(runner.CANDIDATES) - set(run.included) <= set(range(48)) - set(run.active)
        assert run.included == tuple(sorted(c for c, row in zip(runner.CANDIDATES, design) if row[j]))
        assert run.num_clients == 32


def test_replicates_draw_independent_private_seeds():
    runs = plan()
    assert len({run.client_seed for run in runs}) == len({run.noise_seed for run in runs}) == 8
    assert not {run.client_seed for run in runs} & {run.noise_seed for run in runs}


def test_configurations_share_replicate_seeds_and_inclusion():
    first, second = plan(), plan(config="gdp-r4x", noise_ratio=4 * runner.FRONTIER_RATIO)
    for a, b in zip(first, second, strict=True):
        assert (a.client_seed, a.noise_seed, a.included) == (b.client_seed, b.noise_seed, b.included)
        assert a.run_name() != b.run_name()


def test_global_dp_noise_std_is_ratio_times_clipping_norm():
    run = plan()[0]
    assert run.noise_multiplier * run.hyperparams.clipping_norm / run.num_clients == pytest.approx(
        runner.FRONTIER_RATIO * run.hyperparams.clipping_norm)


def test_vanilla_has_no_noise():
    assert all(run.noise_multiplier == 0.0 for run in plan(config="vanilla", privacy="vanilla", noise_ratio=0.0))


def test_runner_args_pin_seeds_layout_and_weights():
    run = plan()[3]
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                           checkpoint_rounds=runner.CHECKPOINT_ROUNDS)
    flags = dict(zip(args[::1], args[1::1]))
    assert flags["--client-seed"] == str(run.client_seed)
    assert flags["--noise-seed"] == str(run.noise_seed)
    assert flags["--layout-seed"] == flags["--init-seed"] == str(runner.LAYOUT_SEED)
    assert flags["--aggregation-weighting"] == "equal"
    assert "--log-noise-diagnostics" in args
    profile = json.loads(flags["--partition-profile"])
    assert profile == {"alpha": runner.ALPHA, "clients": 48, "partition_seed": runner.LAYOUT_SEED,
                       "active": list(run.active)}
    assert runner.CHECKPOINT_ROUNDS == (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)


def test_run_names_are_unique():
    runs = plan(replicates=32)
    assert len({run.run_name() for run in runs}) == 32


def test_replicate_selection_for_sharding():
    runs = plan(replicates=8, replicate_subset=[1, 6])
    assert [run.replicate for run in runs] == [1, 6]


def test_without_augmentation_switches_off_the_transform_through_subsets():
    from dataclasses import dataclass

    from torch.utils.data import Subset

    @dataclass(frozen=True)
    class Transform:
        augment: bool

    class Records:
        def __init__(self):
            self.transform = Transform(augment=True)

    original = Records()
    view = runner.without_augmentation(Subset(Subset(original, [0, 2, 4]), [1, 2]))
    assert view.dataset.dataset.transform.augment is False
    assert original.transform.augment is True
    assert list(view.indices) == [1, 2] and list(view.dataset.indices) == [0, 2, 4]


def test_label_matched_surrogates_follow_the_histogram():
    labels = np.repeat(np.arange(4), 100)
    chosen = label_matched_indices(labels, target_counts={0: 5, 2: 20, 3: 1}, seed=0)
    assert sorted(np.bincount(labels[chosen], minlength=4).tolist()) == sorted([5, 0, 20, 1])
    assert np.bincount(labels[chosen], minlength=4).tolist() == [5, 0, 20, 1]
    assert len(set(chosen.tolist())) == len(chosen)
    np.testing.assert_array_equal(chosen, label_matched_indices(labels, target_counts={0: 5, 2: 20, 3: 1}, seed=0))


def test_checkpoints_are_every_five_rounds_plus_the_last():
    assert runner.checkpoint_rounds(50) == list(runner.CHECKPOINT_ROUNDS)
    assert runner.checkpoint_rounds(2) == [0, 2]
    assert runner.checkpoint_rounds(12) == [0, 5, 10, 12]


def test_fixed_candidates_follow_the_rule_on_the_seed_42_layout():
    try:
        from results.cia_frontier.eurosat_frontier.data import partition_summary
        summary = partition_summary(runner.ALPHA, runner.SLOTS, runner.LAYOUT_SEED)
    except Exception:
        pytest.skip("EuroSAT is not available locally")
    top, others = runner.select_candidates({c["target"]: c["records"] for c in summary["clients"]})
    assert top + others == runner.CANDIDATES


def test_private_median_runs_pin_the_mechanism_settings():
    from experiments.reproduce import runner as reproduce_runner

    run = plan(config="pm", privacy="private-median")[0]
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                           checkpoint_rounds=runner.CHECKPOINT_ROUNDS)
    flags = dict(zip(args[::1], args[1::1]))
    assert flags["--privacy"] == "private-median"
    assert flags["--num-slots"] == str(run.num_clients) == "32"  # the public active count
    assert (flags["--median-quantile"], flags["--median-quantile-min"], flags["--median-quantile-max"]) == (
        "0.3", "0.03", "1.0")
    assert "--log-noise-diagnostics" in args
    config = reproduce_runner.build_run_config(reproduce_runner._parser().parse_args(list(args)))
    assert config["median-quantile-max"] == 1.0 and config["num-slots"] == 32


def test_median_settings_are_absent_for_other_mechanisms():
    args = plan()[0].runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                                 checkpoint_rounds=runner.CHECKPOINT_ROUNDS)
    assert not any(str(a).startswith("--median-") for a in args) and "--num-slots" not in args


def test_manifest_records_median_settings_only_for_private_median():
    median = runner.manifest(plan(config="pm", privacy="private-median")[0], surrogates=True)
    assert median["median_quantile"] == {"initial": 0.3, "minimum": 0.03, "maximum": 1.0, "num_slots": 32}
    assert "median_quantile" not in runner.manifest(plan()[0], surrogates=True)  # M2 manifests stay unchanged


# --- Alzheimer profile (owner decision 8) ------------------------------------------------------------


def test_alzheimer_runs_use_its_panel_model_and_layout():
    runs = plan(dataset="alzheimer", replicates=32)
    profile = runner.PROFILES["alzheimer"]
    assert profile.candidates[:5] == profile.top_decile and len(set(profile.candidates)) == 32
    assert all(len(run.active) == 32 and set(run.included) <= set(profile.candidates) for run in runs)
    assert all(set(range(48)) - set(profile.candidates) <= set(run.active) for run in runs)
    assert {run.model_module for run in runs} == {"experiments.reproduce.paper_cnn:create_model"}
    assert runs[0].hyperparams.learning_rate == 0.001 and runs[0].hyperparams.rounds == 100
    pinned = runner.manifest(runs[0], surrogates=False)
    assert pinned["checkpoint_rounds"] == list(range(0, 101, 10)) and pinned["alpha"] == 1.0
    eurosat = plan(replicates=32)
    assert [run.client_seed for run in runs] == [run.client_seed for run in eurosat]  # same design and seeds


def test_alzheimer_runner_args_and_manifest_name_the_dataset():
    run = plan(dataset="alzheimer")[0]
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                           checkpoint_rounds=runner.CHECKPOINT_ROUNDS)
    flags = dict(zip(args[::1], args[1::1]))
    assert json.loads(flags["--partition-profile"])["dataset"] == "alzheimer"
    pinned = runner.manifest(run, surrogates=False)
    assert pinned["dataset"] == "alzheimer" and pinned["shadow_fraction"] == 0.25
    assert pinned["candidates"] == list(runner.ALZHEIMER_CANDIDATES)
    assert pinned["top_decile"] == list(runner.ALZHEIMER_TOP_DECILE)
    assert "dataset" not in runner.manifest(plan()[0], surrogates=False)  # EuroSAT manifests unchanged


def test_alzheimer_candidates_follow_the_rule_on_the_seed_42_layout():
    try:
        from results.cia_frontier.side_channel_audit.data import partition_summary
        from results.cia_v2.randomized_inclusion.data import ALZHEIMER_MIN_RECORDS
        summary = partition_summary(runner.PROFILES["alzheimer"].alpha, runner.SLOTS, runner.LAYOUT_SEED,
                                    min_records=ALZHEIMER_MIN_RECORDS)
    except Exception:
        pytest.skip("Alzheimer MRI is not available locally")
    top, others = runner.select_candidates({c["target"]: c["records"] for c in summary["clients"]})
    assert top + others == runner.ALZHEIMER_CANDIDATES
    assert summary["min_records"] >= ALZHEIMER_MIN_RECORDS


def test_data_module_follows_the_profile_dataset():
    from results.cia_frontier.side_channel_audit.data import DirichletAlzheimer
    from results.cia_v2.randomized_inclusion import data

    module = data.base_module(runner.ALPHA, partition_seed=42, dataset="alzheimer")
    assert isinstance(module, DirichletAlzheimer) and module.min_records == 10
    with pytest.raises(ValueError):
        data.base_module(runner.ALPHA, dataset="cifar")


def test_alpha_override_reaches_the_partition_profile_and_manifest():
    run = plan(dataset="alzheimer", alpha=100.0)[0]
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                           checkpoint_rounds=runner.CHECKPOINT_ROUNDS)
    flags = dict(zip(args[::1], args[1::1]))
    assert json.loads(flags["--partition-profile"])["alpha"] == 100.0
    assert runner.manifest(run, surrogates=False)["alpha"] == 100.0
    assert "alpha" not in runner.manifest(plan()[0], surrogates=False)  # EuroSAT's 0.3 is not recorded


def test_alzheimer_private_median_uses_its_own_quantile_settings():
    run = plan(dataset="alzheimer", config="pm", privacy="private-median")[0]
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                           checkpoint_rounds=runner.CHECKPOINT_ROUNDS)
    flags = dict(zip(args[::1], args[1::1]))
    assert (flags["--median-quantile"], flags["--median-quantile-min"], flags["--median-quantile-max"]) == (
        "0.4", "0.04", "1.3")
    assert runner.manifest(run, surrogates=False)["median_quantile"]["maximum"] == 1.3


def test_a_run_with_a_failed_client_is_refused(tmp_path):
    full = {str(t): {"client-ids": list(range(32))} for t in range(1, 4)}
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"train_metrics": full}))
    runner.check_every_client_trained(path, 32)
    full["2"]["client-ids"] = list(range(31))
    path.write_text(json.dumps({"train_metrics": full}))
    with pytest.raises(RuntimeError, match="fewer than 32"):
        runner.check_every_client_trained(path, 32)
