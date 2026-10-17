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
    assert len(runner.CANDIDATES) == len(set(runner.CANDIDATES)) == 16
    assert runner.TOP_DECILE == (23, 7, 40, 18, 33)


def test_candidate_rule_reproduces_the_fixed_list():
    sizes = {t: 1000 - t for t in range(48)}  # size falls with the index
    top, others = runner.select_candidates(sizes)
    assert top == (0, 1, 2, 3, 4)
    ascending = sorted(range(5, 48), key=lambda t: sizes[t])
    assert others == tuple(ascending[round((k + 0.5) * 43 / 11)] for k in range(11))


def test_every_run_has_forty_active_clients_and_follows_the_design():
    runs = plan()
    design = runner.design_matrix(8, design_seed=0)
    for j, run in enumerate(runs):
        assert len(run.active) == 40
        assert set(runner.CANDIDATES) - set(run.included) <= set(range(48)) - set(run.active)
        assert run.included == tuple(sorted(c for c, row in zip(runner.CANDIDATES, design) if row[j]))
        assert run.num_clients == 40


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
    runs = plan(replicates=16)
    assert len({run.run_name() for run in runs}) == 16


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
