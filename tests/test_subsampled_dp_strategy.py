"""CIA v2 B1: global DP with Poisson client sampling and a public denominator. Synthetic, no training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from experiments.reproduce import runner
from experiments.reproduce.server import make_run_strategy, require_all_clients
from metricdp_pytorch.strategy_factory import UNIT_WEIGHT_KEY
from metricdp_pytorch.subsampled_dp_strategy import PoissonSampledGlobalDP
from results.cia_v2.randomized_inclusion import runner as inclusion


def model(*arrays):
    return ArrayRecord({f"layer-{i}": Array(np.asarray(a, dtype=float)) for i, a in enumerate(arrays)})


def replies(models):
    out = []
    for node_id, record in enumerate(models, start=1):
        request = Message(content=RecordDict(), message_type="train", dst_node_id=node_id)
        content = RecordDict({"arrays": record, "metrics": MetricRecord({UNIT_WEIGHT_KEY: 1, "client-id": node_id - 1})})
        out.append(Message(content=content, reply_to=request))
    return out


def strategy(num_clients=4, sampling_rate=0.5, noise_multiplier=1e-9, clipping_norm=100.0, seed=0):
    s = PoissonSampledGlobalDP(FedAvg(weighted_by_key=UNIT_WEIGHT_KEY), noise_multiplier=noise_multiplier,
                               clipping_norm=clipping_norm, num_clients=num_clients, sampling_rate=sampling_rate,
                               seed=seed)
    s.noise_rng = np.random.default_rng(0)
    return s


class Inner:
    """Stands in for FedAvg's sampling: one message per active client."""

    def __init__(self, n):
        self.n = n

    def configure_train(self, server_round, arrays, config, grid):
        return list(range(self.n))


def test_sampling_draws_each_client_with_probability_q_and_is_seeded():
    s = strategy(num_clients=32, sampling_rate=0.25, seed=7)
    s.strategy = Inner(32)
    counts = [len(s.configure_train(t, model([0.0]), {}, None)) for t in range(400)]
    assert s.sampled_this_round == counts[-1]
    assert np.mean(counts) == pytest.approx(8.0, abs=0.4)
    again = strategy(num_clients=32, sampling_rate=0.25, seed=7)
    again.strategy = Inner(32)
    assert [len(again.configure_train(t, model([0.0]), {}, None)) for t in range(400)] == counts


def test_the_sampled_mean_is_scaled_to_the_public_denominator():
    s = strategy(num_clients=4, sampling_rate=0.5)  # denominator 2
    s.current_arrays = model([0.0, 0.0])
    arrays, metrics = s.aggregate_train(1, replies([model([2.0, 0.0])]))  # one client sampled
    np.testing.assert_allclose(arrays["layer-0"].numpy(), [1.0, 0.0], atol=1e-6)  # 2.0 / (q N = 2)
    assert metrics["subsampled-dp-sampled-clients"] == 1 and metrics["subsampled-dp-denominator"] == 2.0


def test_noise_on_the_average_is_z_c_over_q_n():
    s = strategy(num_clients=32, sampling_rate=0.25, noise_multiplier=0.8, clipping_norm=5.0)
    assert s.current_noise_stdv == pytest.approx(0.8 * 5.0 / 8)


def test_an_empty_round_releases_noise_only():
    s = strategy(num_clients=4, sampling_rate=0.25, noise_multiplier=2.0, clipping_norm=1.0)
    s.current_arrays = model(np.zeros(20000))
    arrays, metrics = s.aggregate_train(3, [])
    released = arrays["layer-0"].numpy()
    assert released.std() == pytest.approx(2.0 / 1.0, rel=0.05)  # z C / (q N) = 2 / 1
    assert metrics["subsampled-dp-sampled-clients"] == 0


def test_the_fail_fast_check_expects_only_the_sampled_clients():
    s = strategy(num_clients=4, sampling_rate=0.5)
    s.current_arrays = model([0.0])
    s.sampled_this_round = 2
    checked = require_all_clients(s, 4)
    checked.aggregate_train(1, replies([model([1.0]), model([1.0])]))
    with pytest.raises(RuntimeError, match="of 2 clients"):
        checked.aggregate_train(2, replies([model([1.0])]))


BASE = ["--seed", "42", "--noise-multiplier", "0.5", "--clipping-norm", "5.0", "--num-clients", "4",
        "--model-module", "experiments.reproduce.eurosat_cnn:create_model", "--rounds", "2", "--local-epochs", "1"]


def test_runner_builds_the_mode_and_validates_it():
    config = runner.build_run_config(runner._parser().parse_args(
        [*BASE, "--privacy", "subsampled-global-dp", "--sampling-rate", "0.25", "--aggregation-weighting", "equal"]))
    assert config["sampling-rate"] == 0.25
    built = make_run_strategy(config | {"noise-seed": 5})
    assert isinstance(built, PoissonSampledGlobalDP) and built.expected_clients == 1.0
    for bad in (["--privacy", "subsampled-global-dp", "--aggregation-weighting", "equal"],
                ["--privacy", "global-dp", "--sampling-rate", "0.5"]):
        with pytest.raises(ValueError):
            runner.build_run_config(runner._parser().parse_args([*BASE, *bad]))


def test_inclusion_runs_scale_the_multiplier_and_record_the_rate():
    run = inclusion.build_runs(config="b1", privacy="subsampled-global-dp", noise_ratio=0.003092, replicates=2,
                               design_seed=2, sampling_rate=0.5)[0]
    assert run.noise_multiplier == pytest.approx(0.003092 * 32 * 0.5)  # std on the average stays r C
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                           checkpoint_rounds=inclusion.CHECKPOINT_ROUNDS)
    flags = dict(zip(args[::1], args[1::1]))
    assert flags["--sampling-rate"] == "0.5" and "--require-all-clients" in args
    assert inclusion.manifest(run, surrogates=False)["sampling_rate"] == 0.5
    plain = inclusion.build_runs(config="g", privacy="global-dp", noise_ratio=0.003092, replicates=2, design_seed=2)[0]
    assert "sampling_rate" not in inclusion.manifest(plain, surrogates=False)
    with pytest.raises(ValueError):
        inclusion.build_runs(config="g", privacy="global-dp", noise_ratio=0.003, replicates=2, design_seed=2,
                             sampling_rate=0.5)


def test_run_check_counts_the_sampled_clients(tmp_path):
    import json

    metrics = {"1": {"client-ids": [0, 3], "subsampled-dp-sampled-clients": 2},
               "2": {"subsampled-dp-sampled-clients": 0}}
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"train_metrics": metrics}))
    inclusion.check_every_client_trained(path, 32)
    metrics["1"]["client-ids"] = [0]
    path.write_text(json.dumps({"train_metrics": metrics}))
    with pytest.raises(RuntimeError):
        inclusion.check_every_client_trained(path, 32)
