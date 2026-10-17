"""M1 planning (no training): v1's step-3 gdp arm with independent private seeds per world."""

from __future__ import annotations

from pathlib import Path

import pytest

from results.cia_frontier.consensus_kl import runner_dp
from results.cia_v2.measurement.m1 import runner


def test_twelve_trajectories_one_in_and_five_out_per_level():
    cells = runner.plan()
    assert [name for name, _ in cells] == ["gdp-r0.001546", "gdp-r0.006184"]
    for _, combos in cells:
        assert [c.out_target for c in combos] == [None, 23, 7, 1, 0, 3]


def test_every_world_has_its_own_private_seeds():
    for _, combos in runner.plan():
        assert len({c.client_seed for c in combos}) == len({c.noise_seed for c in combos}) == 6
        assert not {c.client_seed for c in combos} & {c.noise_seed for c in combos}


def test_noise_levels_share_the_world_seeds():
    (_, low), (_, high) = runner.plan()
    assert [(c.client_seed, c.noise_seed) for c in low] == [(c.client_seed, c.noise_seed) for c in high]


def test_only_the_private_seeds_and_names_differ_from_v1():
    for (name, combos), ratio in zip(runner.plan(), runner.RATIOS):
        for ours, theirs in zip(combos, runner_dp.build_dp_combos("gdp", ratio), strict=True):
            assert ours.name_prefix == "m1-" + theirs.name_prefix
            for field in ("seed", "num_clients", "noise_multiplier", "noise_ratio", "out_target", "partition_seed",
                          "aggregation_weighting", "hyperparams", "diagnostics", "config", "consensus_beta"):
                assert getattr(ours, field) == getattr(theirs, field), field


def test_runner_args_pass_the_seeds():
    combo = runner.plan()[1][1][2]
    args = combo.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0)
    flags = dict(zip(args, args[1:]))
    assert flags["--client-seed"] == str(combo.client_seed)
    assert flags["--noise-seed"] == str(combo.noise_seed)
    assert flags["--seed"] == "42"
    assert "--log-noise-diagnostics" in args


@pytest.mark.parametrize("ratio", [0.003092, 0.01])
def test_only_the_protocol_levels_are_planned(ratio):
    with pytest.raises(ValueError):
        runner.plan([ratio])
