"""Stage B planning, CIFAR-10 subsampling and analysis; no dataset download or training."""
import json

import numpy as np
import pytest
import torch

from experiments.reproduce.dataset.cifar10 import Cifar10DataModule
from experiments.reproduce.runner import _parser, build_run_config
from results.cia_frontier.dataset_vs_model import analyze_stage_b, data, stage_b


def test_plan_size_and_axes():
    combos = stage_b.build_combos()
    # 2 datasets x 2 partitions x (1 vanilla + 2 mechanisms x 4 ratios), seed 42, IN only
    assert len(combos) == 2 * 2 * 9 == 36
    assert len({c.run_name() for c in combos}) == 36
    assert {c.seed for c in combos} == {42} and {c.out_target for c in combos} == {None}
    assert {c.num_clients for c in combos} == {48} and all(c.log_client_influence for c in combos)
    assert {c.model_module for c in combos} == {stage_b.MODELS["cifar10_cnn"]}
    assert {c.partition for c in combos} == {"homogeneous", "dirichlet"}
    assert {c.dirichlet_alpha for c in combos} == {0.3}
    assert {c.data_tag for c in combos} == {"cifar10s", "eurosat32"}
    assert {c.hyperparams.rounds for c in combos} == {20}
    ratios = {round(c.noise_multiplier / 48, 6) for c in combos if c.privacy != "vanilla"}
    assert ratios == {0.001, 0.0025, 0.004, 0.00625}
    assert all(c.noise_multiplier == 0.0 for c in combos if c.privacy == "vanilla")
    assert all(c.max_client_samples == 0 for c in combos)  # pool size does the matching
    assert len(stage_b.build_combos(seeds=[42, 43, 44])) == 108


def test_out_builder():
    outs = stage_b.build_combos(cells=["eurosat32+homogeneous"], privacy=["global-dp"],
                                ratios=[0.001], out_targets=[0, 3])
    assert [(c.out_target, c.num_clients) for c in outs] == [(0, 47), (3, 47)]
    assert outs[0].noise_multiplier == pytest.approx(0.001 * 47)
    assert "-out-3__" in outs[1].run_name()
    for bad in ([10], [0, 0], []):
        with pytest.raises(ValueError):
            stage_b.build_combos(out_targets=bad)


def test_run_names_carry_partition_and_arm():
    names = {c.run_name() for c in stage_b.build_combos(seeds=[42], cells=["cifar10s+dirichlet"])}
    assert len(names) == 9
    assert any(n.startswith("stage-b-cifar10s-dirichlet-vanilla-in__dirichlet__alpha-0p3__") for n in names)
    assert any(n.startswith("stage-b-cifar10s-dirichlet-metric-privacy-r0.00625-in__") for n in names)


def test_sharding_and_validation():
    with pytest.raises(ValueError):
        stage_b.build_combos(cells=["nope"])
    with pytest.raises(ValueError):
        stage_b.build_combos(seeds=[42, 42])
    with pytest.raises(ValueError):
        stage_b.build_combos(ratios=[0.001, 0.001])
    only_gdp = stage_b.build_combos(seeds=[42], privacy=["global-dp"], ratios=[0.001])
    assert len(only_gdp) == 4 and {c.privacy for c in only_gdp} == {"global-dp"}


def test_runner_args_reach_server_config(tmp_path):
    (combo,) = stage_b.build_combos(seeds=[42], cells=["eurosat32+dirichlet"],
                                    privacy=["metric-privacy"], ratios=[0.0025], out_targets=[4])
    args = combo.runner_args(output_dir=tmp_path, max_parallel_clients=6, client_cpus=1.0)
    config = build_run_config(_parser().parse_args(list(args)))
    assert config["partition-mode"] == "dirichlet" and config["dirichlet-alpha"] == 0.3
    assert config["num-server-rounds"] == 20 and config["noise-multiplier"] == pytest.approx(0.0025 * 47)
    assert config["log-client-influence"] is True and config["num-clients"] == 47
    view = data.create_stage_b_view(config)
    assert view.canonical_num_partitions == 48 and 4 not in view.active_partition_ids
    assert view.num_active_partitions == 47


class _Labels(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.targets = list(labels)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor([float(index)]), self.targets[index]


class _FakeBase:
    """Tiny labelled pool partitioned like the real plugins (seeded, canonical clients)."""

    def __init__(self, config=None):
        self.labels = np.repeat(np.arange(4), 400)
        self.train_subsample = 0

    def client_loaders(self, partition_id, *, num_partitions, partition_mode, batch_size, seed,
                       partition_profile="auto", client_weights=None, dirichlet_alpha=0.5, max_samples=0):
        from experiments.reproduce.dataset.cifar10 import create_partitions
        from metricdp_pytorch.utils.data import make_client_loaders
        assert partition_profile == "auto"
        parts = create_partitions(self.labels, num_partitions=num_partitions, mode=partition_mode,
                                  seed=seed, dirichlet_alpha=dirichlet_alpha)
        return make_client_loaders(_Labels(self.labels), self.labels, parts[partition_id],
                                   batch_size=batch_size, seed=seed + partition_id)

    def server_loaders(self, **kwargs):
        raise AssertionError("not needed")


@pytest.mark.parametrize("partition", ["homogeneous", "dirichlet"])
def test_in_and_out_score_identical_shadow_records(monkeypatch, partition):
    monkeypatch.setitem(data._BASES, "cifar10s", _FakeBase)
    kw = dict(seeds=[42], cells=[f"cifar10s+{partition}"], privacy=["vanilla"], clients=4, targets=[0, 1])
    (inside,) = stage_b.build_combos(**kw)
    (outside,) = stage_b.build_combos(**kw, out_targets=[1])
    in_prints = stage_b.shadow_fingerprints(inside, {t: stage_b.shadow_modules(inside, t) for t in (0, 1)})
    out_prints = stage_b.shadow_fingerprints(outside, {1: stage_b.shadow_modules(outside, 1)})
    assert in_prints["1"]["sha256"] == out_prints["1"]["sha256"]
    assert in_prints["1"]["in_training"] and not out_prints["1"]["in_training"]
    assert in_prints["0"]["sha256"] != in_prints["1"]["sha256"]
    # Another training seed gives another layout (one seed controls both, as in eurosat_frontier).
    (other,) = stage_b.build_combos(**{**kw, "seeds": [43]})
    assert stage_b.shadow_fingerprints(other, {1: stage_b.shadow_modules(other, 1)})["1"]["sha256"] \
        != in_prints["1"]["sha256"]


class _Split:
    """Minimal stand-in for a Hugging Face split: labels + select()."""

    def __init__(self, labels):
        self._labels = list(labels)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, key):
        if key == "label":
            return self._labels
        raise KeyError(key)

    def select(self, indices):
        return _Split(self._labels[i] for i in indices)


def test_cifar10_train_pool_is_stratified_fixed_subset():
    labels = np.repeat(np.arange(10), 500)  # 5,000 "images", 500 per class
    module = Cifar10DataModule(train_subsample=1000, subsample_seed=0)
    module._dataset = {"train": _Split(labels)}
    pool = module.train_pool
    assert len(pool) == 1000
    assert np.bincount(pool["label"]).tolist() == [100] * 10
    again = Cifar10DataModule(train_subsample=1000, subsample_seed=0)
    again._dataset = {"train": _Split(labels)}
    assert again.train_pool["label"] == pool["label"]
    full = Cifar10DataModule()
    full._dataset = {"train": _Split(labels)}
    assert len(full.train_pool) == 5000
    with pytest.raises(ValueError):
        Cifar10DataModule(train_subsample=-1)


def test_round_matched_score_synthetic():
    from results.cia_frontier.per_client_score import score_rows
    inside = [dict(round=r, target=t, clean_loss=float(r + t)) for r in (1, 2) for t in (0, 1)]
    outside = [dict(round=r, target=t, clean_loss=float(r + t + (1 if t == 0 else -1)))
               for r in (1, 2) for t in (0, 1)]
    scored = score_rows(inside, outside)
    assert scored["per_target"] == {"0": 1.0, "1": 0.0}
    assert scored["mean"] == 0.5 and scored["matched_rounds"] == 2


def test_stage_b_cifar10_module_is_eurosat_sized():
    module = data.create_cifar10_small({})
    assert module.train_subsample == 21_600 and module.subsample_seed == 0


def _fake_run(seed, acc):
    return {"metadata": {"seed": seed},
            "train_metrics": {"1": {"dp-update-norms-before-clipping": [3.0, 4.0],
                                    "metric-dp-pairwise-distances": [0.5, 0.7],
                                    "metric-dp-noise-stdv": 0.02}},
            "server_evaluate_metrics": {"0": {"accuracy": 0.1}, "1": {"accuracy": acc}}}


def test_analyze_groups_by_cell_and_arm(tmp_path):
    combos = stage_b.build_combos(seeds=[42, 43], cells=["cifar10s+homogeneous"],
                                  privacy=["vanilla", "global-dp"], ratios=[0.001])
    for combo in combos:
        sub = tmp_path / combo.privacy
        sub.mkdir(exist_ok=True)
        acc = 0.6 if combo.privacy == "vanilla" else 0.5
        combo.result_path(sub).write_text(json.dumps(_fake_run(combo.seed, acc + combo.seed / 1000)))
    (out,) = stage_b.build_combos(seeds=[42], cells=["cifar10s+homogeneous"], privacy=["vanilla"],
                                  out_targets=[0])
    out.result_path(tmp_path).write_text(json.dumps(_fake_run(42, 0.0)))  # OUT: not in accuracy
    summary = analyze_stage_b.summarize(tmp_path)
    assert list(summary) == ["cifar10s+homogeneous"]
    arms = summary["cifar10s+homogeneous"]
    assert set(arms) == {"vanilla", "global-dp-r0.001"}
    assert arms["vanilla"]["seeds"] == 2
    assert arms["vanilla"]["final_accuracy"] == pytest.approx(0.6425)
    assert arms["global-dp-r0.001"]["final_accuracy"] == pytest.approx(0.5425)
    assert arms["global-dp-r0.001"]["max_pairwise_distance"] == 0.7


def _trajectory(root, name, manifest, rows, shadows=None):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"pilot": False, "run_name": name, **manifest}))
    (folder / "measurements.json").write_text(json.dumps(rows))
    (folder / "complete.json").write_text("{}")
    if shadows is not None:
        (folder / "shadows.json").write_text(json.dumps(shadows))


def _rows(losses):
    return [dict(round=r, target=t, clean_loss=v, noisy_loss=v, aggregate_loss=1.0, shadow_size=5)
            for (r, t), v in losses.items()]


@pytest.mark.parametrize("style", ["eurosat_frontier", "stage_b"])
def test_per_client_score_on_synthetic_trajectories(tmp_path, style):
    from results.cia_frontier.per_client_score import score_directories
    setting = ({"alpha": 0.3, "privacy": "vanilla", "noise_ratio": 0.0, "clients": 48, "rounds": 3}
               if style == "eurosat_frontier" else
               {"dataset": "cifar10s", "partition": "dirichlet", "alpha": 0.3, "privacy": "vanilla",
                "noise_ratio": 0.0, "clients": 48, "rounds": 3})
    shadows = (lambda t: {str(t): {"sha256": f"h{t}"}}) if style == "stage_b" else (lambda t: None)
    rounds = (1, 2, 3)
    _trajectory(tmp_path / "in", "in", {**setting, "seed": 42, "out_target": None, "targets": [0, 1]},
                _rows({(r, t): 1.0 for r in rounds for t in (0, 1)}),
                {"0": {"sha256": "h0"}, "1": {"sha256": "h1"}} if style == "stage_b" else None)
    # target 0: IN lower in rounds 1, 2, tie in 3 -> 2.5/3; target 1: IN never lower -> 0
    _trajectory(tmp_path / "out0", "out0", {**setting, "seed": 42, "out_target": 0, "targets": [0]},
                _rows({(1, 0): 2.0, (2, 0): 3.0, (3, 0): 1.0}), shadows(0))
    _trajectory(tmp_path / "out1", "out1", {**setting, "seed": 42, "out_target": 1, "targets": [1]},
                _rows({(r, 1): 0.5 for r in rounds}), shadows(1))
    (result,) = score_directories(tmp_path)
    assert result["per_target"] == {"0": pytest.approx(2.5 / 3), "1": 0.0}
    assert result["mean"] == pytest.approx(2.5 / 6) and result["matched_rounds"] == 3
    if style == "stage_b":
        (tmp_path / "out1" / "out1" / "shadows.json").write_text(json.dumps(shadows(9) | {"1": {"sha256": "x"}}))
        with pytest.raises(ValueError, match="shadow records differ"):
            score_directories(tmp_path)
