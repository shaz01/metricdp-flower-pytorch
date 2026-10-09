"""Real ServerApp.run + ClientApp.train over the in-process grid on synthetic data."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.stacked_head import bundle as bundle_module
from experiments.stacked_head import client as client_module
from experiments.stacked_head import server as server_module
from experiments.stacked_head.cnn import Net, embed
from experiments.stacked_head.local import LocalGrid
from experiments.stacked_head.tests.synthetic import client_labels, labels_for, make_images
from metricdp_pytorch.stacked_head_strategy import StackedHeadConfig, client_contribution, head_scores, head_theta, ArrayRecord, Array

TASK, BUDGET, SET, ROLE_SEED, TRAIN_SEED = "kmnist_classes0to3", 32, 0, 11, 13
CAP, ETA, DIMENSION = 0.05, 20.0, 12


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bundles")
    public, validation, evaluation = labels_for(8), labels_for(16), labels_for(24)
    bundle = bundle_module.prepare_from_arrays(
        make_images(public, 1), public, make_images(validation, 2), validation, make_images(evaluation, 3), evaluation, seed_offset=BUDGET + SET, train_seed=TRAIN_SEED
    )
    bundle_module.save_bundle(bundle, bundle_module.bundle_path(directory, TASK, BUDGET, SET, ROLE_SEED, TRAIN_SEED))
    clients = {i: (make_images(client_labels(i, 5), 100 + i), client_labels(i, 5)) for i in range(8)}
    return directory, bundle, clients


@pytest.fixture
def patched(world, monkeypatch):
    _, _, clients = world
    monkeypatch.setattr(client_module, "client_data", lambda task, role_seed, cohort, client_id: clients[client_id])
    client_module.probe_block.cache_clear()


def make_config(world, tmp_path, **overrides):
    directory, *_ = world
    config = {
        "task": TASK, "budget": BUDGET, "public-set": SET, "cohort": "A", "risk": 0.65, "rounds": 1, "replicate": False,
        "noise-source": "independent", "seed": 3, "probe-noise-seed": 77, "probe-noise-rounds": 4, "role-seed": ROLE_SEED, "train-seed": TRAIN_SEED,
        "frozen-config": "", "cap": CAP, "eta": ETA, "dimension": DIMENSION, "mode": "target_center",
        "bundle-dir": str(directory), "output-dir": str(tmp_path), "run-name": "cell", "diagnostics": False, "log-messages": False,
        "absent-clients": "", "noise-free": False, "model-module": "experiments.stacked_head.cnn:create_model",
    }
    config.update(overrides)
    return config


def execute(world, tmp_path, **overrides):
    config = make_config(world, tmp_path, **overrides)
    return server_module.run(LocalGrid(8, config), config), config


def base_model(bundle):
    model = Net()
    model.load_state_dict({key: torch.tensor(value) for key, value in bundle.state.items()})
    return model.eval()


def test_noise_free_single_release_equals_direct_library_computation(world, patched, tmp_path):
    _, bundle, clients = world
    result, _ = execute(world, tmp_path, **{"noise-free": True})
    model = base_model(bundle)
    config = StackedHeadConfig(cap=CAP, eta=ETA, dimension=DIMENSION, risk=0.65, noise_free=True)
    basis = bundle.basis[:, :DIMENSION]
    theta0 = head_theta(ArrayRecord({k: Array(v) for k, v in bundle.state.items()}))
    total = sum(client_contribution(embed(model, clients[i][0]), clients[i][1], theta0, bundle.class_gradients, basis, config) for i in range(8))
    candidates = np.stack([theta0 - ETA * m * (total @ basis.T).reshape(3, -1) for m in config.multipliers])
    validation = head_scores(bundle.validation_features, bundle.validation_labels, candidates)[0]
    pick = int(validation.argmin())
    expected_ce = head_scores(bundle.evaluation_features, bundle.evaluation_labels, candidates[pick][None])[0][0]
    row = result["rounds"][0]
    assert row["multiplier"] == config.multipliers[pick]
    assert row["eval-ce"] == pytest.approx(float(expected_ce), abs=1e-5)
    assert result["control"]["ce"] == pytest.approx(bundle.meta["control"]["ce"], abs=1e-9)
    assert (tmp_path / "cell.json").exists() and json.loads((tmp_path / "cell.json").read_text())["summary"]["num_rounds"] == 1


def test_replicate_releases_are_independent_repeatable_and_never_beat_the_zero_step_on_validation(world, patched, tmp_path):
    a, _ = execute(world, tmp_path, rounds=5, replicate=True)
    b, _ = execute(world, tmp_path, rounds=5, replicate=True, **{"run-name": "again"})
    assert [r["eval-ce"] for r in a["rounds"]] == [r["eval-ce"] for r in b["rounds"]]
    base_validation = a["rounds"][0]["validation-ce-base"]
    assert all(r["validation-ce-chosen"] <= base_validation + 1e-12 for r in a["rounds"])
    assert a["summary"]["num_rounds"] == 5 and sum(a["summary"]["pick_fraction"].values()) == pytest.approx(1.0)
    seeds_differ, _ = execute(world, tmp_path, rounds=5, replicate=True, seed=99, **{"run-name": "other"})
    assert [r["eval-ce"] for r in seeds_differ["rounds"]] != [r["eval-ce"] for r in a["rounds"]]


def test_multiple_rounds_require_replicate_mode(world, patched, tmp_path):
    with pytest.raises(ValueError, match="replicate"):
        execute(world, tmp_path, rounds=2, replicate=False)


def test_absent_client_sends_only_its_noise_share_and_contributions_respect_the_cap(world, patched, tmp_path):
    result, _ = execute(world, tmp_path, **{"noise-free": True, "diagnostics": True, "absent-clients": "3"})
    contributions = {int(k): np.array(v) for k, v in result["contributions"].items()}
    assert np.all(contributions[3] == 0)
    assert all(np.linalg.norm(v) <= CAP / 8 + 1e-12 for v in contributions.values())
    assert sum(np.linalg.norm(v) > 0 for v in contributions.values()) == 7


def test_probe_noise_source_replays_the_research_stream(world, patched, tmp_path):
    result, config = execute(world, tmp_path, rounds=3, replicate=True, **{"noise-source": "probe", "diagnostics": True, "log-messages": True, "run-name": "probe"})
    construction = server_module.construction_from_config(config)
    block = np.random.default_rng(77).normal(size=(4, 8, 51))
    messages = np.load(tmp_path / "probe.messages.npz")["messages"]
    noise_free, _ = execute(world, tmp_path, **{"noise-free": True, "diagnostics": True, "run-name": "contrib"})
    contribution = np.array([noise_free["contributions"][str(i)] for i in range(8)])
    for r in range(3):
        expected_share = block[r, :, :DIMENSION] * construction.sigma / np.sqrt(7)
        assert np.allclose(messages[r] - contribution, expected_share, atol=1e-10)


def test_message_log_has_one_vector_per_client_and_round(world, patched, tmp_path):
    execute(world, tmp_path, rounds=2, replicate=True, **{"log-messages": True, "run-name": "log"})
    stored = np.load(tmp_path / "log.messages.npz")
    assert stored["messages"].shape == (2, 8, DIMENSION) and list(stored["rounds"]) == [1, 2]
    assert np.all(np.isfinite(stored["messages"]))


def test_independent_noise_differs_across_cells_but_repeats_within_one(world, patched, tmp_path):
    def shares(**overrides):
        result, _ = execute(world, tmp_path, rounds=2, replicate=True, **{"log-messages": True, "run-name": "n", "noise-free": False, **overrides})
        return np.load(tmp_path / "n.messages.npz")["messages"]

    a = shares(**{"cell-key": "cell-one"})
    assert np.array_equal(a, shares(**{"cell-key": "cell-one"}))
    assert not np.allclose(a, shares(**{"cell-key": "cell-two"}))


def test_default_frozen_config_is_read_for_the_risk_target():
    construction = server_module.construction_from_config({"risk": 0.65})
    assert (construction.cap, construction.eta, construction.dimension, construction.mode) == (0.003, 100.0, 51, "target_center")
    assert server_module.construction_from_config({"risk": 0.8}).eta == 100.0


def test_release_diagnostics_agree_with_the_strategy_gate_and_alternative_gates(world, patched, tmp_path):
    result, _ = execute(world, tmp_path, rounds=6, replicate=True, **{"run-name": "diag", "extra-validation-sizes": "64"})
    stored = np.load(tmp_path / "diag.releases.npz")
    multipliers = list(stored["multipliers"])
    assert stored["eval_ce"].shape == (6, len(multipliers)) and stored["val64_ce"].shape == stored["val_ce"].shape
    picks = stored["val_ce"].argmin(axis=1)
    assert [multipliers[i] for i in picks] == pytest.approx([r["multiplier"] for r in result["rounds"]])
    gated = stored["eval_ce"][np.arange(6), picks]
    assert gated == pytest.approx([r["eval-ce"] for r in result["rounds"]], abs=1e-5)
    assert result["summary_fixed_multiplier_1"]["mean_ce"] == pytest.approx(float(stored["eval_ce"][:, multipliers.index(1.0)].mean()), abs=1e-5)
    alt = result["summary_alt_validation"]["64"]
    alt_picks = stored["val64_ce"].argmin(axis=1)
    assert alt["mean_ce"] == pytest.approx(float(stored["eval_ce"][np.arange(6), alt_picks].mean()), abs=1e-5)
    assert result["summary"]["gain_over_control"] == pytest.approx(result["control"]["ce"] - float(gated.mean()), abs=1e-5)


def test_attack_statistics_equal_a_direct_recomputation_and_records_are_optional(world, patched, tmp_path, monkeypatch):
    from metricdp_pytorch.stacked_head_strategy import head_class_ce

    _, bundle, clients = world
    model = base_model(bundle)
    records = {t: (embed(model, clients[t][0]), clients[t][1]) for t in range(4)}
    monkeypatch.setattr(server_module, "target_records", lambda config, bundle_, targets: records)
    result, config = execute(world, tmp_path, rounds=3, replicate=True, **{"attack-records": True, "run-name": "atk"})
    stored = np.load(tmp_path / "atk.releases.npz")
    assert stored["attack_gain"].shape == (3, 4) and stored["attack_records_gain"].shape == (3, 4)
    # Recompute release 1 from the logged messages' sum: rerun with message logging for the same cell and noise.
    execute(world, tmp_path, rounds=3, replicate=True, **{"attack-records": True, "log-messages": True, "run-name": "atk2"})
    total = np.load(tmp_path / "atk2.messages.npz")["messages"][0].sum(axis=0)
    theta0 = head_theta(ArrayRecord({k: Array(v) for k, v in bundle.state.items()}))
    basis = bundle.basis[:, :DIMENSION]
    multipliers = server_module.construction_from_config(config).multipliers
    candidates = np.stack([theta0 - ETA * m * (total @ basis.T).reshape(3, -1) for m in multipliers])
    pick = int(head_scores(bundle.validation_features, bundle.validation_labels, candidates)[0].argmin())
    class_ce = head_class_ce(bundle.validation_features, bundle.validation_labels, candidates)
    mix = np.full((4, 4), 17.0)
    mix[np.arange(4), np.arange(4)] = 205.0
    mix /= 256.0
    assert stored["attack_gain"][0] == pytest.approx(mix @ (class_ce[0] - class_ce[pick]), abs=1e-5)
    direct = [head_scores(f, y, theta0[None])[0][0] - head_scores(f, y, candidates[pick][None])[0][0] for f, y in (records[t] for t in range(4))]
    assert stored["attack_records_gain"][0] == pytest.approx(direct, abs=1e-5)
    plain, _ = execute(world, tmp_path, rounds=2, replicate=True, **{"run-name": "plain"})
    assert "attack_records_gain" not in np.load(tmp_path / "plain.releases.npz").files   # only stored when the attacker simulation is requested
