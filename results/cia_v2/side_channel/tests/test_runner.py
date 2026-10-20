"""A2 planning (no training)."""

from __future__ import annotations

from pathlib import Path

import pytest

from results.cia_v2.side_channel import runner


def test_cells_and_targets_are_fixed():
    assert runner.CELLS == {("alzheimer", 3): 2, ("alzheimer", 5): 1, ("alzheimer", 8): 5, ("eurosat", 3): 1}


def test_a_cell_has_one_in_and_one_out_trajectory_per_replicate():
    combos = runner.build_cell("alzheimer", 5, "global-dp", multiple=4, stage="main")
    assert len(combos) == 2 * runner.REPLICATES["main"]
    assert [c.out_target for c in combos[:4]] == [None, 1, None, 1]
    assert {c.replicate for c in combos} == set(range(runner.REPLICATES["main"]))
    assert all(c.partition_seed == 42 and c.num_clients == (5 if c.out_target is None else 4) for c in combos)


def test_noise_std_is_fixed_across_removal():
    combos = runner.build_cell("eurosat", 3, "metric-privacy", multiple=16, stage="pilot")
    for c in combos:
        assert c.noise_multiplier / c.num_clients == pytest.approx(16 * runner.BASE_RATIO["eurosat"])


def test_private_seeds_differ_between_worlds_and_replicates():
    combos = runner.build_cell("alzheimer", 3, "global-dp", multiple=1, stage="main")
    seeds = [(c.client_seed, c.noise_seed) for c in combos]
    assert len({s[0] for s in seeds}) == len({s[1] for s in seeds}) == len(combos)


def test_seeds_are_shared_across_mechanisms_and_levels_but_not_stages():
    a = runner.build_cell("alzheimer", 8, "global-dp", multiple=1, stage="main")
    b = runner.build_cell("alzheimer", 8, "metric-privacy", multiple=16, stage="main")
    assert [(x.client_seed, x.noise_seed) for x in a] == [(x.client_seed, x.noise_seed) for x in b]
    pilot = runner.build_cell("alzheimer", 8, "global-dp", multiple=1, stage="pilot")
    assert not {x.noise_seed for x in pilot} & {x.noise_seed for x in a}


def test_runner_args_pass_seeds_and_noise_diagnostics():
    combo = runner.build_cell("eurosat", 3, "global-dp", multiple=4, stage="main")[3]
    args = combo.runner_args(output_dir=Path("out"), max_parallel_clients=2, client_cpus=1.0)
    flags = dict(zip(args, args[1:]))
    assert flags["--client-seed"] == str(combo.client_seed)
    assert flags["--noise-seed"] == str(combo.noise_seed)
    assert "--log-noise-diagnostics" in args


def test_run_names_are_unique_within_a_cell():
    combos = runner.build_cell("alzheimer", 3, "global-dp", multiple=4, stage="main")
    assert len({c.run_name() for c in combos}) == len(combos)


def test_only_grid_multiples_are_allowed():
    with pytest.raises(ValueError):
        runner.build_cell("alzheimer", 3, "global-dp", multiple=2, stage="main")
