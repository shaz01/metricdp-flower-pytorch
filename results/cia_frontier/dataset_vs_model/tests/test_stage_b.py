"""Stage B planning, CIFAR-10 subsampling and analysis; no dataset download or training."""
import json

import numpy as np
import pytest

from experiments.reproduce.dataset.cifar10 import Cifar10DataModule
from experiments.reproduce.runner import _parser, build_run_config
from results.cia_frontier.dataset_vs_model import analyze_stage_b, data, stage_b


def test_plan_size_and_axes():
    combos = stage_b.build_combos()
    # 2 datasets x 2 partitions x (1 vanilla + 2 mechanisms x 4 ratios) x 3 seeds
    assert len(combos) == 2 * 2 * 9 * 3 == 108
    assert len({c.run_name() for c in combos}) == 108
    assert {c.model_module for c in combos} == {stage_b.MODELS["cifar10_cnn"]}
    assert {c.partition for c in combos} == {"homogeneous", "dirichlet"}
    assert {c.dirichlet_alpha for c in combos} == {0.3}
    assert {c.data_tag for c in combos} == {"cifar10s", "eurosat32"}
    assert {c.hyperparams.rounds for c in combos} == {20}
    ratios = {round(c.noise_multiplier / 48, 6) for c in combos if c.privacy != "vanilla"}
    assert ratios == {0.001, 0.0025, 0.004, 0.00625}
    assert all(c.noise_multiplier == 0.0 for c in combos if c.privacy == "vanilla")
    assert all(c.max_client_samples == 0 for c in combos)  # pool size does the matching


def test_run_names_carry_partition_and_arm():
    names = {c.run_name() for c in stage_b.build_combos(seeds=[42], cells=["cifar10s+dirichlet"])}
    assert len(names) == 9
    assert any(n.startswith("stage-b-cifar10s-dirichlet-vanilla__dirichlet__alpha-0p3__") for n in names)
    assert any(n.startswith("stage-b-cifar10s-dirichlet-metric-privacy-r0.00625__") for n in names)


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
                                    privacy=["metric-privacy"], ratios=[0.0025])
    args = combo.runner_args(output_dir=tmp_path, max_parallel_clients=6, client_cpus=1.0)
    config = build_run_config(_parser().parse_args(list(args)))
    assert config["partition-mode"] == "dirichlet" and config["dirichlet-alpha"] == 0.3
    assert config["num-server-rounds"] == 20 and config["noise-multiplier"] == pytest.approx(0.12)


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


def test_pure_influence_diagnostics():
    from metricdp_pytorch.influence_diagnostics import per_client_influence
    original = [np.array([3., 0.]), np.array([0., 2.])]
    snapshot = [u.copy() for u in original]
    rows = per_client_influence(original, [8, 3], [1, 3], 2.)
    assert set(rows) == {"8", "3"}
    assert rows["8"]["clipped"] is True
    assert rows["3"]["aggregation_weight"] == 0.75
    assert rows["8"]["leave_one_out_influence_norm"] >= 0
    assert all(np.array_equal(a, b) for a, b in zip(original, snapshot))


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
    summary = analyze_stage_b.summarize(tmp_path)
    assert list(summary) == ["cifar10s+homogeneous"]
    arms = summary["cifar10s+homogeneous"]
    assert set(arms) == {"vanilla", "global-dp-r0.001"}
    assert arms["vanilla"]["seeds"] == 2
    assert arms["vanilla"]["final_accuracy"] == pytest.approx(0.6425)
    assert arms["global-dp-r0.001"]["final_accuracy"] == pytest.approx(0.5425)
    assert arms["global-dp-r0.001"]["max_pairwise_distance"] == 0.7
