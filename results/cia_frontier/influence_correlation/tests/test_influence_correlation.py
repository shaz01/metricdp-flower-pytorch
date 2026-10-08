"""Planning, data-pool, summary and analysis tests; no dataset download or training."""
import json

import pytest

from experiments.reproduce.dataset.eurosat import EurosatDataModule
from experiments.reproduce.runner import _parser, build_run_config
from results.cia_frontier.dataset_vs_model.data import stage_b_profile
from results.cia_frontier.influence_correlation import analyze, runner


def test_plan_counts_and_names():
    ins = runner.build_combos(seeds=[42])
    assert len(ins) == 6  # 2 datasets x 3 arms
    assert {c.num_clients for c in ins} == {16}
    assert {c.out_target for c in ins} == {None}
    assert {c.hyperparams.rounds for c in ins} == {50}
    assert {c.partition for c in ins} == {"dirichlet"} and {c.dirichlet_alpha for c in ins} == {0.3}
    assert {c.train_subsample for c in ins} == {7200}
    assert all(c.run_name().startswith("infl-") and "stage-" not in c.run_name() for c in ins)
    assert all(c.log_client_influence for c in ins)
    everything = runner.build_combos(seeds=[42], out_targets=range(16))
    assert len(everything) == 6 * 17 and len({c.run_name() for c in everything}) == 6 * 17
    outs = [c for c in everything if c.out_target is not None]
    assert {c.num_clients for c in outs} == {15}
    gdp_in = next(c for c in ins if c.privacy == "global-dp")
    assert gdp_in.noise_multiplier == pytest.approx(0.004 * 16)
    assert runner.build_combos(seeds=[42], with_in=False, out_targets=[3]) and \
        all(c.out_target == 3 for c in runner.build_combos(seeds=[42], with_in=False, out_targets=[3]))
    with pytest.raises(ValueError):
        runner.build_combos(seeds=[42], datasets=["nope"])
    with pytest.raises(ValueError):
        runner.build_combos(seeds=[42], clients=49)


def test_profile_reaches_runner_config_with_pool(tmp_path):
    (combo,) = [c for c in runner.build_combos(seeds=[42], datasets=["eurosat32"], privacy=["vanilla"])]
    args = combo.runner_args(output_dir=tmp_path, max_parallel_clients=4, client_cpus=1.0)
    config = build_run_config(_parser().parse_args(list(args)))
    profile = json.loads(config["partition-profile"])
    assert profile == {"dataset": "eurosat32", "clients": 16, "out_target": None, "train_subsample": 7200}
    assert config["num-clients"] == 16 and config["num-server-rounds"] == 50
    assert config["log-client-influence"] is True


def test_eurosat_train_subsample_is_stratified_and_fixed(monkeypatch):
    from datasets import Dataset
    labels = [i % 10 for i in range(200)]
    fake = Dataset.from_dict({"image": [0] * 200, "label": labels})
    monkeypatch.setattr("experiments.reproduce.dataset.eurosat.load_eurosat_dataset",
                        lambda cache_dir: {"train": fake, "test": fake})
    module = EurosatDataModule(train_subsample=50)
    pool = module.train_pool
    assert len(pool) == 50
    counts = {}
    for lab in pool["label"]:
        counts[lab] = counts.get(lab, 0) + 1
    assert set(counts.values()) == {5}
    again = EurosatDataModule(train_subsample=50).train_pool
    assert list(pool["label"]) == list(again["label"])
    assert len(EurosatDataModule().train_pool) == 200
    with pytest.raises(ValueError):
        EurosatDataModule(train_subsample=-1)
    assert json.loads(stage_b_profile("eurosat32", 16, None, 7200))["train_subsample"] == 7200
    with pytest.raises(ValueError):
        stage_b_profile("eurosat32", 16, None, 0)


def _fake_in_run(client_ids, norms, loo, losses):
    rounds = {}
    for r in (1, 2):
        rounds[str(r)] = {
            "influence-client-ids": client_ids,
            "influence-update-norm-before-clipping": [n * r for n in norms],
            "influence-clipped": [int(n * r > 5) for n in norms],
            "influence-distance-from-weighted-clipped-average": norms,
            "influence-cosine-with-weighted-clipped-average": [0.5] * len(norms),
            "influence-leave-one-out-influence-norm": loo,
            "influence-aggregation-weight": [1 / len(norms)] * len(norms),
            "influence-num-examples": [100] * len(norms),
            "client-ids": client_ids,
            "per-client-train_loss": losses,
        }
    return {"metadata": {"seed": 42}, "train_metrics": rounds, "server_evaluate_metrics": {}}


def test_client_features_aggregate_over_rounds():
    data = _fake_in_run([0, 1, 2], [1.0, 2.0, 6.0], [0.1, 0.2, 0.3], [2.0, 1.0, 0.5])
    summary = {"clients": [{"target": i, "train_records": 10 * (i + 1), "test_records": 2,
                            "class_counts": {"0": 5, "1": 5} if i < 2 else {"0": 10}}
                           for i in range(3)]}
    feats = analyze.client_features(data, summary)
    assert feats[2]["update_norm_mean"] == pytest.approx(9.0)  # (6 + 12) / 2
    assert feats[2]["update_norm_sum"] == pytest.approx(18.0)
    assert feats[2]["clipped_fraction"] == 1.0 and feats[0]["clipped_fraction"] == 0.0
    assert feats[1]["leave_one_out_influence_sum"] == pytest.approx(0.4)
    assert feats[0]["train_loss_mean"] == 2.0 and feats[1]["train_records"] == 20
    skew = analyze.label_skew(summary)
    assert skew[2] > skew[0] == skew[1]


def test_correlations_recover_a_monotone_relation(tmp_path):
    rows = [{"dataset": "d", "privacy": "vanilla", "noise_ratio": 0.0, "seed": 42, "client": i,
             "attack_score": 0.5 + i / 40, "update_norm_mean": float(i), "train_loss_mean": -i / 2,
             "cosine_with_average_mean": 0.3} for i in range(16)]
    out = analyze.correlations(rows)
    assert out["n_rows"] == 16
    assert out["pooled"]["update_norm_mean"]["spearman"] == pytest.approx(1.0)
    assert out["pooled"]["train_loss_mean"]["spearman"] == pytest.approx(-1.0)
    assert out["pooled"]["cosine_with_average_mean"]["spearman"] is None  # constant: undefined
    assert analyze.spearman([1, 2, 2, 3], [1, 3, 3, 4]) == pytest.approx(1.0)


def test_collect_joins_scores_with_in_run_features(tmp_path):
    combos = runner.build_combos(seeds=[42], datasets=["cifar10s"], privacy=["vanilla"],
                                 out_targets=[0, 1], clients=16)
    from results.cia_frontier.dataset_vs_model import stage_b
    in_combo = next(c for c in combos if c.out_target is None)
    for combo in combos:
        folder = tmp_path / combo.run_name()
        folder.mkdir()
        chosen = list(range(16)) if combo.out_target is None else [combo.out_target]
        (folder / "manifest.json").write_text(json.dumps(stage_b.manifest(combo, range(16))))
        loss = (lambda r, t: 1.0 + 0.01 * t) if combo.out_target is None else (lambda r, t: 2.0)
        (folder / "measurements.json").write_text(json.dumps(
            [{"round": r, "target": t, "clean_loss": loss(r, t), "noisy_loss": 1.0, "shadow_size": 10,
              "aggregate_loss": 1.0} for r in range(1, 51) for t in chosen]))
        (folder / "shadows.json").write_text(json.dumps(
            {str(t): {"size": 10, "sha256": f"h{t}", "in_training": combo.out_target is None} for t in chosen}))
        (folder / "complete.json").write_text(json.dumps({"complete": True}))
    folder = tmp_path / in_combo.run_name()
    (folder / f"{in_combo.run_name()}.json").write_text(json.dumps(
        _fake_in_run(list(range(16)), [float(i) for i in range(16)], [0.1] * 16, [1.0] * 16)))
    (folder / "partitions.json").write_text(json.dumps(
        {"clients": [{"target": i, "train_records": 100, "test_records": 10, "class_counts": {"0": 100}}
                     for i in range(16)]}))
    rows = analyze.collect(tmp_path)
    assert len(rows) == 2 and {r["client"] for r in rows} == {0, 1}
    assert all(r["attack_score"] == 1.0 for r in rows)
    assert rows[1]["update_norm_mean"] == pytest.approx(1.5) and rows[0]["label_skew"] == 0.0


def test_equal_weighting_reaches_runner_config_and_names(tmp_path):
    default = runner.build_combos(seeds=[42], out_targets=[0])
    equal = runner.build_combos(seeds=[42], out_targets=[0], weighting="equal")
    assert [c.run_name() for c in equal] == [c.run_name().replace("__fedavg__", "__fedavg-eqw__")
                                             for c in default]
    assert all("__fedavg__" in c.run_name() for c in default)
    combo = equal[0]
    config = build_run_config(_parser().parse_args(list(
        combo.runner_args(output_dir=tmp_path, max_parallel_clients=4, client_cpus=1.0))))
    assert config["aggregation-weighting"] == "equal"
    default_config = build_run_config(_parser().parse_args(list(
        default[0].runner_args(output_dir=tmp_path, max_parallel_clients=4, client_cpus=1.0))))
    assert default_config["aggregation-weighting"] == "num-examples"


@pytest.mark.parametrize("prefix", ["influence-pairwise", "metric-dp-pairwise"])
def test_distance_features_from_pairwise_matrix(prefix):
    # clients 0,1,2; pairs (0,1),(0,2),(1,2)
    rounds = {"1": [1.0, 4.0, 3.0], "2": [5.0, 2.0, 1.0]}
    data = {"train_metrics": {r: {f"{prefix}-distances": d, f"{prefix}-client-i": [0, 0, 1],
                                  f"{prefix}-client-j": [1, 2, 2]} for r, d in rounds.items()}}
    feats = analyze.client_features(data, None)
    assert feats[0]["pairwise_distance_mean"] == pytest.approx((2.5 + 3.5) / 2)
    assert feats[2]["pairwise_distance_max"] == pytest.approx((4.0 + 2.0) / 2)
    assert feats[0]["max_pair_share"] == 1.0  # max pair (0,2) then (0,1)
    assert feats[1]["max_pair_share"] == 0.5 and feats[2]["max_pair_share"] == 0.5


def test_no_pairwise_matrix_means_no_distance_features():
    feats = analyze.client_features(_fake_in_run([0, 1], [1.0, 2.0], [0.1, 0.2], [1.0, 1.0]), None)
    assert "pairwise_distance_mean" not in feats[0]
