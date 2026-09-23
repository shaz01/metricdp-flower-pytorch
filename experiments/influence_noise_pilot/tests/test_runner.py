"""Planning/sharding/manifest tests; no dataset or training."""
import json

import pytest
from flwr.supercore.differential_privacy import compute_stdv

from experiments.influence_noise_pilot import runner
from experiments.reproduce.runner import _parser, build_run_config


def test_plan_names_and_sharding():
    full = runner.build_influence_combos(fraction=0.5, seeds=[42], targets=list(range(10)))
    assert len(full) == 11 and len({c.run_name() for c in full}) == 11
    control = runner.build_influence_combos(fraction=0.0, seeds=[42], targets=list(range(10)))
    assert not {c.run_name() for c in full} & {c.run_name() for c in control}
    (only_in,) = runner.build_influence_combos(fraction=0.5, seeds=[42], targets=list(range(10)), adjacency="in")
    (out0,) = runner.build_influence_combos(fraction=0.5, seeds=[42], targets=list(range(10)),
                                            adjacency="out", out_targets=[0])
    assert only_in.out_target is None and only_in.num_clients == 48
    assert out0.out_target == 0 and out0.num_clients == 47
    assert {only_in.run_name(), out0.run_name()} <= {c.run_name() for c in full}
    with pytest.raises(ValueError):
        runner.build_influence_combos(fraction=1.5, seeds=[42], targets=[0])


def test_tau_is_exact_global_dp_convention():
    for combo in runner.build_influence_combos(fraction=0.5, seeds=[42], targets=[0, 1]):
        pinned = runner.protocol(combo)
        n = combo.num_clients
        assert combo.noise_multiplier == 0.001546 * n
        assert pinned["tau"] == compute_stdv(0.001546 * n, 5.0, n)
        assert pinned["tau"] == pytest.approx(0.001546 * 5.0, rel=1e-12)
        assert pinned["influence_cap"] == 5.0 and pinned["dp_claim"] is False
        assert len(pinned["canonical_client_ids"]) == n
        assert combo.out_target not in pinned["canonical_client_ids"]


def test_runner_args_reach_server_config(tmp_path):
    (combo,) = runner.build_influence_combos(fraction=0.5, seeds=[42], targets=[0], adjacency="in")
    args = combo.runner_args(output_dir=tmp_path, max_parallel_clients=6, client_cpus=1.0)
    config = build_run_config(_parser().parse_args(["--model-module", "x:y", *args]))
    assert config["privacy"] == "influence-noise"
    assert config["influence-fraction"] == 0.5 and config["influence-cap"] == 5.0
    assert config["noise-multiplier"] == combo.noise_multiplier and config["num-clients"] == 48


def test_influence_flags_rejected_elsewhere():
    base = ["--model-module", "x:y", "--rounds", "1", "--local-epochs", "1", "--seed", "1",
            "--noise-multiplier", "0.1", "--clipping-norm", "5"]
    with pytest.raises(ValueError):
        build_run_config(_parser().parse_args([*base, "--privacy", "global-dp", "--influence-fraction", ".5"]))
    with pytest.raises(ValueError):
        build_run_config(_parser().parse_args([*base, "--privacy", "influence-noise"]))
    config = build_run_config(_parser().parse_args([*base, "--privacy", "global-dp"]))
    assert "influence-fraction" not in config


def test_protocol_mismatch_rejected(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("experiments.auc_frontier.runner.execute", lambda *a, **k: calls.append(a))
    (combo,) = runner.build_influence_combos(fraction=0.5, seeds=[42], targets=[0], adjacency="in")
    runner.execute([combo], [0], tmp_path, 6)
    path = tmp_path / combo.run_name() / "influence_protocol.json"
    assert json.loads(path.read_text())["fraction"] == 0.5 and len(calls) == 1
    runner.execute([combo], [0], tmp_path, 6)  # identical protocol resumes
    edited = json.loads(path.read_text()) | {"influence_cap": 1.0}
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError):
        runner.execute([combo], [0], tmp_path, 6)
    assert len(calls) == 2
