"""Baseline mechanisms on the same public base: local training, repository DP strategies, dummy clients, oracle AUC."""

import numpy as np
import pytest
from scipy.special import ndtr

from experiments.stacked_head import baseline, server as server_module
from experiments.stacked_head.cnn import embed
from experiments.stacked_head.tests.test_pipeline import base_model, execute, make_config, patched, world  # noqa: F401  (fixtures)
from experiments.stacked_head.tests.synthetic import labels_for, make_images

LOCAL = {"local-lr": 0.05, "local-steps": 5, "clip-norm": 0.1, "multipliers": ""}


def test_local_training_descends_and_is_deterministic():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 32)), rng.integers(0, 4, size=60)
    w0, b0 = rng.normal(size=(4, 32)) * 0.1, np.zeros(4)

    def loss(w, b):
        z = x @ w.T + b
        z -= z.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(60), y]))

    w1, b1 = baseline.local_train(x, y, w0, b0, 0.1, 20)
    assert loss(w1, b1) < loss(w0, b0)
    w2, b2 = baseline.local_train(x, y, w0, b0, 0.1, 20)
    assert np.array_equal(w1, w2) and np.array_equal(b1, b2)
    assert np.array_equal(baseline.local_train(x, y, w0, b0, 0.1, 0)[0], w0)


def test_matched_noise_multiplier_inverts_the_worst_case_oracle_auc():
    for risk in (0.55, 0.65, 0.8, 0.95):
        z = baseline.matched_noise_multiplier(risk)
        worst_case = baseline.oracle_auc(np.array([10.0]), clip_norm=1.0, noise_std=z * 1.0 / 8, num_clients=8)[0]   # fully clipped client
        assert worst_case == pytest.approx(risk, abs=1e-9)
    with pytest.raises(ValueError):
        baseline.matched_noise_multiplier(0.5)


def test_oracle_auc_uses_the_clipped_norm_and_handles_zero_noise():
    auc = baseline.oracle_auc(np.array([0.05, 0.5]), clip_norm=0.1, noise_std=0.02, num_clients=8)
    assert auc[0] == pytest.approx(float(ndtr(0.05 / 8 / (np.sqrt(2) * 0.02))))
    assert auc[1] == pytest.approx(float(ndtr(0.1 / 8 / (np.sqrt(2) * 0.02))))     # clipped at C
    assert list(baseline.oracle_auc(np.array([0.3, 0.0]), 0.1, 0.0, 8)) == [1.0, 0.5]


def test_vanilla_release_equals_the_plain_average_of_the_local_updates(world, patched, tmp_path):  # noqa: F811
    _, bundle, clients = world
    result, config = execute(world, tmp_path, rounds=1, **{**LOCAL, "mechanism": "vanilla", "multipliers": "0,1"})
    model = base_model(bundle)
    heads = []
    for i in range(8):
        features = embed(model, clients[i][0])[:, :-1]
        w, b = baseline.local_train(features, clients[i][1], bundle.state["head.weight"], bundle.state["head.bias"], 0.05, 5)
        heads.append((w.astype(np.float32).astype(float), b.astype(np.float32).astype(float)))
    mean_w, mean_b = np.mean([h[0] for h in heads], axis=0), np.mean([h[1] for h in heads], axis=0)
    from metricdp_pytorch.stacked_head_strategy import CONTRAST, head_scores

    theta = CONTRAST.T @ np.column_stack((mean_w, mean_b))
    ce = head_scores(bundle.evaluation_features, bundle.evaluation_labels, theta[None])[0][0]
    stored = np.load(tmp_path / "cell.releases.npz")
    assert stored["eval_ce"][0][1] == pytest.approx(float(ce), abs=2e-4)   # candidate multiplier 1 is the vanilla FedAvg head
    assert result["oracle"]["mean_noise_std"] == 0.0


def test_global_dp_uses_the_repository_noise_scale_and_clips_every_update(world, patched, tmp_path):  # noqa: F811
    # The synthetic base already fits its data (updates ~1e-7), so use a tiny clip norm to make every update clipped.
    result, _ = execute(world, tmp_path, rounds=4, replicate=True, **{**LOCAL, "clip-norm": 1e-9, "mechanism": "global-dp", "noise-multiplier": 0.7})
    assert result["oracle"]["mean_noise_std"] == pytest.approx(0.7 * 1e-9 / 8, rel=1e-6)
    assert result["oracle"]["fraction_clipped"] == 1.0
    stored = np.load(tmp_path / "cell.releases.npz")
    assert stored["update_norms"].shape == (4, 8) and stored["noise_std"].shape == (4,)
    assert result["baseline"]["multipliers"][0] == 0.0 and result["summary"]["num_rounds"] == 4


def test_global_dp_default_multiplier_is_matched_to_the_risk_and_metric_privacy_needs_one(world, patched, tmp_path):  # noqa: F811
    matched, _ = execute(world, tmp_path, rounds=2, replicate=True, risk=0.8, **{**LOCAL, "mechanism": "global-dp"})
    assert matched["baseline"]["noise_multiplier"] == pytest.approx(baseline.matched_noise_multiplier(0.8))
    with pytest.raises(ValueError, match="explicit"):
        execute(world, tmp_path, rounds=2, replicate=True, **{**LOCAL, "mechanism": "metric-privacy"})


def test_metric_privacy_noise_is_the_global_dp_noise_divided_by_the_client_distance(world, patched, tmp_path):  # noqa: F811
    result, _ = execute(world, tmp_path, rounds=3, replicate=True, **{**LOCAL, "mechanism": "metric-privacy", "noise-multiplier": 0.02})
    stored = np.load(tmp_path / "cell.releases.npz")
    expected = 0.02 / stored["distance"] * 0.1 / 8
    assert np.allclose(stored["noise_std"], expected, rtol=1e-4) and np.all(stored["distance"] > 0)


def test_dummy_client_sends_a_tiny_nonzero_update_and_runs_survive(world, patched, tmp_path):  # noqa: F811
    result, _ = execute(world, tmp_path, rounds=3, replicate=True, **{**LOCAL, "mechanism": "global-dp", "noise-multiplier": 0.7, "absent-clients": "3"})
    stored = np.load(tmp_path / "cell.releases.npz")
    assert stored["update_norms"][:, 3].max() == pytest.approx(baseline.DUMMY_EPSILON, rel=0.05)
    assert result["summary"]["num_rounds"] == 3 and np.all(np.isfinite(stored["update_norms"]))


def test_replicate_rounds_all_start_from_the_same_base_and_are_repeatable(world, patched, tmp_path):  # noqa: F811
    a, _ = execute(world, tmp_path, rounds=3, replicate=True, **{**LOCAL, "mechanism": "global-dp", "noise-multiplier": 0.5})
    b, _ = execute(world, tmp_path, rounds=3, replicate=True, **{**LOCAL, "mechanism": "global-dp", "noise-multiplier": 0.5, "run-name": "again"})
    assert [r["eval-ce"] for r in a["rounds"]] == [r["eval-ce"] for r in b["rounds"]]
    stored = np.load(tmp_path / "cell.releases.npz")["update_norms"]
    assert np.allclose(stored[0], stored[1], rtol=1e-5) and np.allclose(stored[0], stored[2], rtol=1e-5)   # same base -> same local updates every round
