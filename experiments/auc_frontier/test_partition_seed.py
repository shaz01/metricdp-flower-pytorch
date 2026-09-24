"""Partition seed fixes the data layout; the training seed varies only training randomness."""
import json
from dataclasses import replace

import numpy as np

import experiments.auc_frontier.data as data
from experiments.auc_frontier.data import DirichletEuroSAT, create_data_module
from experiments.auc_frontier.runner import build_combos, execute


def plan(**kwargs):
    options = dict(alpha=0.3, seeds=[43, 44], targets=list(range(10)), clients=48,
                   privacy="vanilla", ratios=None, adjacency="both", out_targets=[0, 1])
    options.update(kwargs)
    return build_combos(**options)


def test_default_is_unchanged_from_existing_runs():
    combo = plan(seeds=[42], partition_seed=None)[0]
    assert combo.run_name() == (
        "eurosat-dirichlet-a0.3-in-r0.0__non-iid__vanilla__fedavg__clients-48__seed-42__nm0__"
        "clip5__rounds-100__epochs-5__data__eurosat_cnn")
    profile = json.loads(combo.runner_args(output_dir=None, max_parallel_clients=6, client_cpus=1)[-1])
    assert "partition_seed" not in profile
    assert combo.layout_seed == 42


def test_partition_seed_names_profile_and_layout():
    runs = plan(partition_seed=42)
    assert len(runs) == 6  # 2 training seeds x (IN + OUT0 + OUT1)
    assert len({c.run_name() for c in runs}) == 6
    assert all("-p42-" in c.run_name() for c in runs)
    for combo in runs:
        assert combo.layout_seed == 42
        profile = json.loads(combo.runner_args(output_dir=None, max_parallel_clients=6, client_cpus=1)[-1])
        assert profile["partition_seed"] == 42
        module = create_data_module({"partition-profile": json.dumps(profile)})
        assert module.data_module.partition_seed == 42


def _fake_dataset(monkeypatch):
    labels = np.repeat(np.arange(10), 200)
    captured = []
    monkeypatch.setattr(DirichletEuroSAT, "dataset", property(lambda self: {"train": None, "test": None}))
    monkeypatch.setattr(data, "labels_from_records", lambda split, label_column: labels)
    monkeypatch.setattr(data, "EurosatDataset", lambda split, augment: augment)
    monkeypatch.setattr(data, "make_indexed_loader",
                        lambda dataset, indices, **kw: captured.append((dataset, tuple(indices), kw["seed"])))
    return captured


def test_layout_fixed_while_training_shuffle_varies(monkeypatch):
    captured = _fake_dataset(monkeypatch)
    module = DirichletEuroSAT(0.3, partition_seed=42)
    for training_seed in (43, 44):
        module.client_loaders(5, num_partitions=48, partition_mode="non-iid", batch_size=32, seed=training_seed)
    (train_a, test_a), (train_b, test_b) = captured[:2], captured[2:]
    assert train_a[1] == train_b[1] and test_a[1] == test_b[1]   # same records
    assert (train_a[2], train_b[2]) == (48, 49)                  # shuffle follows training seed


def test_default_layout_follows_seed(monkeypatch):
    captured = _fake_dataset(monkeypatch)
    module = DirichletEuroSAT(0.3)
    for seed in (43, 44):
        module.client_loaders(5, num_partitions=48, partition_mode="non-iid", batch_size=32, seed=seed)
    assert captured[0][1] != captured[2][1]


def test_server_split_uses_partition_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(data.EurosatDataModule, "server_loaders",
                        lambda self, *, batch_size, seed, max_samples=0: seen.append(seed))
    DirichletEuroSAT(0.3, partition_seed=42).server_loaders(batch_size=32, seed=44)
    DirichletEuroSAT(0.3).server_loaders(batch_size=32, seed=44)
    assert seen == [42, 44]


def test_execute_scores_on_partition_seed_data(tmp_path, monkeypatch):
    import experiments.cia.iter_combos as training
    from experiments.cia import cia
    import metricdp_pytorch.utils.device as device
    monkeypatch.setattr(device, "resolve_device", lambda: "cpu")
    summaries = []
    monkeypatch.setattr(data, "partition_summary", lambda *args: summaries.append(args) or {})

    def fake_training(runs, **kwargs):
        paths = tuple(kwargs["output_dir"] / f"{r}.pt" for r in kwargs["checkpoint_rounds"])
        for path in paths:
            path.touch()
        yield runs[0], True, paths
    monkeypatch.setattr(training, "iter_combos", fake_training)
    eval_seeds, layouts = [], []

    def fake_eval(path, **kwargs):
        eval_seeds.append(kwargs["combo"].seed)
        layouts.append(kwargs["clean_data_module"].data_module.partition_seed)
        return (1.0, 2.0, 3.0, 10)
    monkeypatch.setattr(cia, "eval_model", fake_eval)
    combo = plan(partition_seed=42, seeds=[44], out_targets=[0])[0]
    combo = replace(combo, hyperparams=replace(combo.hyperparams, rounds=2))
    execute([combo], list(range(10)), tmp_path, 1)
    assert set(eval_seeds) == {42} and set(layouts) == {42}
    assert summaries == [(0.3, 48, 42)]
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert manifest["partition_seed"] == 42 and manifest["seed"] == 44
