"""Runner configuration, attack statistic calibration and gate logic."""

import argparse

import numpy as np
import pytest
from scipy.special import ndtr

from experiments.stacked_head import analysis, attack, runner


def namespace(**overrides):
    parser = runner._parser()
    args = parser.parse_args(["run"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_probe_seed_matches_the_research_formula():
    assert runner.probe_seed(32, 0, "A", 0.65) == 20261012 + 32000 + 0 + 0 + 65
    assert runner.probe_seed(128, 2, "B", 0.55) == 20261012 + 128000 + 200 + 10 + 55
    assert runner.probe_seed(32, 1, "A", 0.8) == 20261012 + 32000 + 100 + 0 + 80


def test_build_run_config_single_release_and_replicate_modes():
    single = runner.build_run_config(namespace(single_release=True))
    assert single["rounds"] == 1 and single["replicate"] is False and single["run-name"].endswith("_single")
    many = runner.build_run_config(namespace(rounds=64))
    assert many["rounds"] == 64 and many["replicate"] is True and many["probe-noise-rounds"] == 64
    with pytest.raises(ValueError):
        runner.build_run_config(namespace(risk=0.5))


def test_attack_auc_matches_the_calibrated_expectation():
    rng = np.random.default_rng(0)
    d, sigma, rounds, target = 6, 0.01, 6000, 3
    contributions = rng.normal(size=(8, d)) * 0.004
    shift = contributions[target]
    peer = attack.peer_of(target)

    def world(include):
        parts = contributions.copy()
        if not include:
            parts[target] = 0
        noise = rng.normal(size=(rounds, 8, d)) * sigma / np.sqrt(7)
        return parts[None] + noise

    in_scores, out_scores = attack.attack_scores(world(True), world(False), contributions, target, sigma)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(np.r_[np.ones(rounds), np.zeros(rounds)], np.r_[in_scores, out_scores])
    expected = float(ndtr(np.linalg.norm(shift) / (np.sqrt(2) * sigma)))
    assert peer == 7 and abs(auc - expected) < 0.02


def test_peer_of_target_seven_is_six():
    assert attack.peer_of(7) == 6 and attack.peer_of(0) == 7


def test_gate_requires_cells_threshold_mean_and_worst_cell():
    good = [0.012, 0.011, 0.01, 0.009, 0.02, 0.002]
    assert analysis.gate(good)["passed"]
    assert not analysis.gate([0.012, 0.011, 0.01, 0.009, 0.0005, 0.0005])["passed"]
    assert not analysis.gate([0.02, 0.02, 0.02, 0.02, 0.02, -0.004])["passed"]
    assert not analysis.gate(good[:5])["passed"]


def test_matrix_shards_partition_groups_without_splitting_a_bundle(capsys):
    def names(shard):
        args = runner._parser().parse_args(["matrix", "--preset", "sweep32", "--dry-run", "--shard", shard, "--output-dir", "/nonexistent-results-dir"])
        runner.cmd_matrix(args)
        return [line[5:] for line in capsys.readouterr().out.splitlines() if line.startswith("run: ")]

    shards = [names(f"{i}/3") for i in range(3)]
    everything = sum(shards, [])
    assert len(everything) == len(set(everything)) == 120
    for shard in shards:
        groups = {name.rsplit("_q", 1)[0] for name in shard}
        assert len(shard) == 2 * len(groups)  # both risks of a (task, set) group share one shard


def write_cell(directory, task, risk, public_set, gain, tag="sweep", fixed=None, alt=None, control_ce=0.6):
    import json

    name = f"{task}_b32_s{public_set}_A_q{int(round(risk * 100))}_{tag}"
    result = {
        "run_name": name, "config": {"task": task, "risk": risk, "public-set": public_set, "budget": 32},
        "control": {"ce": control_ce, "accuracy": 0.8},
        "summary": {"gain_over_control": gain, "accuracy_delta": 0.001, "pick_fraction": {"0.0": 0.1, "1.0": 0.9}},
        "summary_alt_validation": {"128": {"gain_over_control": gain if alt is None else alt}},
        "summary_fixed_multiplier_1": {"gain_over_control": gain if fixed is None else fixed},
    }
    (directory / f"{name}.json").write_text(json.dumps(result))
    return name


def test_wilson_interval_contains_the_estimate_and_widens_for_small_samples():
    low, high = analysis.wilson(8, 10)
    assert low < 0.8 < high
    wide, narrow = analysis.wilson(8, 10), analysis.wilson(80, 100)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    assert analysis.wilson(0, 0)[0] != analysis.wilson(0, 0)[0]  # NaN


def test_sweep_report_applies_the_prestated_reliability_rule_and_dataset_transfer(tmp_path):
    for s in range(10):
        write_cell(tmp_path, "mnist_classes0to3", 0.65, s, 0.01 + 0.001 * s)           # all wins
        write_cell(tmp_path, "mnist_classes0to3", 0.8, s, 0.02, fixed=-0.01, alt=0.0)   # wins when gated, losses without gating
        write_cell(tmp_path, "mnist_classes4to7", 0.65, s, -0.005 if s < 3 else 0.01)   # 30% losses -> not reliable
        write_cell(tmp_path, "mnist_classes4to7", 0.8, s, 0.01)
    report = analysis.sweep_report(tmp_path, "sweep")
    groups = {(g["task"], g["risk"]): g for g in report["groups"]}
    assert groups[("mnist_classes0to3", 0.65)]["gates"]["gated_val512"]["reliable"]
    assert groups[("mnist_classes0to3", 0.8)]["gates"]["gated_val512"]["reliable"]
    assert not groups[("mnist_classes0to3", 0.8)]["gates"]["fixed_multiplier_1"]["reliable"]
    assert not groups[("mnist_classes0to3", 0.8)]["gates"]["gated_val128"]["reliable"]  # median 0 is not > 0
    bad = groups[("mnist_classes4to7", 0.65)]["gates"]["gated_val512"]
    assert not bad["reliable"] and bad["p_loss"] == pytest.approx(0.3) and len(groups[("mnist_classes4to7", 0.65)]["failures"]) == 3
    transfer = analysis.dataset_transfer(report)
    assert transfer["mnist"] == {"pairs_reliable": 3, "pairs": 4, "transfers": True}
    assert "RELIABLE" in analysis.format_report({**report, "dataset_transfer": transfer})


def test_failure_anatomy_separates_gate_errors_from_missing_steps(tmp_path):
    name = write_cell(tmp_path, "kmnist_classes0to3", 0.8, 5, -0.006, control_ce=0.8)
    import json

    result = json.loads((tmp_path / f"{name}.json").read_text())
    multipliers = np.array([0.0, 0.1, 1.0])
    np.savez(tmp_path / f"{name}.releases.npz", multipliers=multipliers, eval_ce=np.tile(np.array([0.8, 0.78, 0.806]), (4, 1)))
    anatomy = analysis.failure_anatomy(result, tmp_path)
    assert anatomy["cause"] == "gate_error" and anatomy["best_fixed_multiplier"] == 0.1
    np.savez(tmp_path / f"{name}.releases.npz", multipliers=multipliers, eval_ce=np.tile(np.array([0.8, 0.801, 0.806]), (4, 1)))
    assert analysis.failure_anatomy(result, tmp_path)["cause"] == "no_usable_step"
