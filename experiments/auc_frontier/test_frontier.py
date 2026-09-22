import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.auc_frontier.data import create_data_module, dirichlet_partitions
from experiments.auc_frontier.runner import build_combos, execute
from experiments.auc_frontier.analyze import statistics, summarize


def combos(**kwargs):
    options = dict(alpha=0.5, seeds=[42, 43], targets=[0, 47], clients=48,
                   privacy="global-dp", ratios=[0.001])
    options.update(kwargs)
    return build_combos(**options)


def test_plan_and_profile():
    runs = combos()
    assert len(runs) == 6
    assert len({c.run_name() for c in runs}) == 6
    for combo in runs:
        args = combo.runner_args(output_dir=None, max_parallel_clients=6, client_cpus=1)
        config = {"partition-profile": args[-1]}
        module = create_data_module(config)
        assert module.canonical_num_partitions == 48
        assert module.num_active_partitions == combo.num_clients
        assert (47 in module.active_partition_ids) == (combo.out_target != 47)
        assert combo.noise_multiplier == 0.001 * combo.num_clients


def test_counts_and_validation():
    assert len(combos(targets=list(range(20)), ratios=[0.001 * 2**i for i in range(6)])) == 252
    assert len(combos(privacy="vanilla", pilot=True)) == 2
    for bad in ({"alpha": float("nan")}, {"targets": [48]}, {"seeds": [42, 42]}, {"ratios": [-1]}):
        with pytest.raises(ValueError):
            combos(**bad)


def test_dirichlet_is_reproducible_disjoint_and_complete():
    labels = tuple(np.repeat(np.arange(10), 100))
    parts = dirichlet_partitions(labels, 8, 0.5, 42)
    assert parts == dirichlet_partitions(labels, 8, 0.5, 42)
    assert parts != dirichlet_partitions(labels, 8, 0.5, 43)
    assert sorted(i for p in parts for i in p) == list(range(1000))
    assert min(map(len, parts)) >= 10
    assert parts != dirichlet_partitions(labels, 8, 10.0, 42)


def test_metrics_not_folded_and_cluster_interval():
    assert statistics([[2]], [[1]])["paired_concordance"] == 0
    assert statistics([[1]], [[1]])["target_stratified_auc"] == 0.5
    assert summarize([[1], [2]], [[2], [3]])["ci95"] is None
    assert summarize([[1]] * 5, [[2]] * 5, resamples=10)["ci95"]["paired_concordance"] == [1, 1]
    # Cross-seed AUC is different from paired wins; do not mislabel either.
    result = statistics([[1], [3]], [[2], [4]])
    assert result["paired_concordance"] == 1
    assert result["target_stratified_auc"] == 0.75


def test_every_round_every_target_and_resume(tmp_path, monkeypatch):
    import experiments.cia.iter_combos as training
    import experiments.auc_frontier.data as data
    monkeypatch.setattr(data, "partition_summary", lambda *args: {})
    from experiments.cia import cia
    import metricdp_pytorch.utils.device as device
    monkeypatch.setattr(device, "resolve_device", lambda: "cpu")
    calls = []
    def fake_training(runs, **kwargs):
        calls.append(runs)
        assert kwargs["checkpoint_rounds"] == (1, 2, 3)
        paths = tuple(kwargs["output_dir"] / f"{r}.pt" for r in (1, 2, 3))
        for path in paths:
            path.touch()
        yield runs[0], True, paths
    monkeypatch.setattr(training, "iter_combos", fake_training)
    evaluations = []
    def fake_eval(path, **kwargs):
        evaluations.append(path)
        return (1.0, 2.0, 3.0, 10)
    monkeypatch.setattr(cia, "eval_model", fake_eval)
    combo = combos()[0]
    combo = replace(combo, hyperparams=replace(combo.hyperparams, rounds=3))
    execute([combo], [0, 47], tmp_path, 1)
    assert len(evaluations) == 6
    report = next(tmp_path.glob("*/measurements.json"))
    assert len(json.loads(report.read_text())) == 6
    assert not list(tmp_path.glob("*/*.pt"))
    execute([combo], [0, 47], tmp_path, 1)
    assert len(calls) == 1
    # Simulate an interruption after checkpoint deletion: incomplete coverage
    # must retrain, never accept a stale completion marker.
    report.write_text("[]")
    execute([combo], [0, 47], tmp_path, 1)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="Manifest mismatch"):
        execute([combo], [0], tmp_path, 1)
