"""Paper-protocol runner (v3 P0 item 4): planning only, no data and no training."""

from __future__ import annotations

from pathlib import Path

import pytest

from results.cia_v3.infra import paper_runner


def plan(**kwargs):
    settings = dict(config="gdp", privacy="global-dp", noise_multiplier=0.01, pairs=20, design_seed=0)
    settings.update(kwargs)
    return paper_runner.build_runs(**settings)


def test_every_pair_has_an_in_and_an_out_world():
    runs = plan()
    assert len(runs) == 40
    for inside, outside in zip(runs[::2], runs[1::2], strict=True):
        assert (inside.world, outside.world) == ("in", "out") and inside.pair == outside.pair
        assert inside.active == (0, 1, 2) and inside.num_clients == 3
        assert outside.active == (0, 1) and outside.num_clients == 2
        assert paper_runner.TARGET == 2


def test_no_two_worlds_share_private_randomness():
    runs = plan()
    clients, noises = [r.client_seed for r in runs], [r.noise_seed for r in runs]
    assert len(set(clients)) == len(set(noises)) == 40
    assert not set(clients) & set(noises)


def test_mechanisms_share_world_seeds_and_differ_in_names():
    gdp, metric = plan(), plan(config="mp", privacy="metric-privacy")
    for a, b in zip(gdp, metric, strict=True):
        assert (a.client_seed, a.noise_seed) == (b.client_seed, b.noise_seed)
        assert a.run_name() != b.run_name()
    assert len({r.run_name() for r in gdp}) == 40


def test_paper_noise_law_and_hyperparameters():
    run = plan()[0]
    assert run.noise_multiplier == 0.01  # the paper's z; the strategy divides z C by the active count
    hyper = run.hyperparams
    assert (hyper.rounds, hyper.local_epochs, hyper.batch_size, hyper.clipping_norm) == (20, 5, 32, 5.0)
    assert run.aggregation == "fedavg" and run.aggregation_weighting == "num-examples"
    assert run.model_module == "experiments.reproduce.paper_cnn:create_model"


def test_runner_args_pin_seeds_and_parse():
    from experiments.reproduce import runner as reproduce_runner

    run = plan()[1]
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=3, client_cpus=1.0,
                           checkpoint_rounds=tuple(paper_runner.checkpoint_rounds(20)))
    flags = dict(zip(args[::1], args[1::1]))
    assert flags["--client-seed"] == str(run.client_seed) and flags["--noise-seed"] == str(run.noise_seed)
    assert flags["--layout-seed"] == flags["--init-seed"] == "42"
    assert flags["--data-module"].endswith(":create_alzheimer_out") and flags["--num-clients"] == "2"
    assert "--require-all-clients" in args and "--log-noise-diagnostics" in args
    config = reproduce_runner.build_run_config(reproduce_runner._parser().parse_args(list(args)))
    assert config["client-seed"] == run.client_seed and config["noise-seed"] == run.noise_seed
    vanilla = plan(config="v", privacy="vanilla", noise_multiplier=0.0)[0]
    assert "--log-noise-diagnostics" not in vanilla.runner_args(output_dir=Path("out"), max_parallel_clients=3,
                                                                 client_cpus=1.0)


def test_manifest_records_worlds_seeds_and_every_round():
    pinned = paper_runner.manifest(plan()[1])
    assert pinned["world"] == "out" and pinned["active"] == [0, 1] and pinned["target"] == 2
    assert pinned["checkpoint_rounds"] == list(range(21))
    assert (pinned["shadow_fraction"], pinned["shadow_noise_std_fraction"]) == (0.10, 0.20)


def test_invalid_plans_are_refused():
    with pytest.raises(ValueError):
        plan(privacy="vanilla")
    with pytest.raises(ValueError):
        plan(noise_multiplier=0.0)
    with pytest.raises(ValueError):
        plan(pair_subset=[20])
    assert [r.pair for r in plan(pair_subset=[3, 1])] == [1, 1, 3, 3]


def test_data_module_factories_resolve():
    from metricdp_pytorch.data_module import load_data_module

    for world, factory in paper_runner.DATA_MODULES.items():
        module = load_data_module(factory, {})
        assert module.canonical_num_partitions == 3
        assert list(module.active_partition_ids) == ([0, 1, 2] if world == "in" else [0, 1])
