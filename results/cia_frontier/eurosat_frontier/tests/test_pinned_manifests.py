"""Finished runs: run names and manifests must match the committed result folders exactly."""
import json

import pytest

HYPERPARAMS = {"clipping_norm": 5.0, "rounds": 100, "local_epochs": 5, "batch_size": 32,
               "learning_rate": 0.001, "initialization_epochs": 20, "weight_decay": 0.0,
               "lr_schedule": "none"}
FRONTIER = ("eurosat-dirichlet-a0.3-out-3-r0.001546__non-iid__global-dp__fedavg__clients-47__"
            "seed-42__nm0p072662__clip5__rounds-100__epochs-5__data__eurosat_cnn")
INFLUENCE = ("eurosat-dirichlet-a0.3-out-0-r0.001546-f0.0-cap5.0__non-iid__influence-noise__fedavg__"
             "clients-47__seed-42__nm0p072662__clip5__rounds-100__epochs-5__data__eurosat_cnn")


def _manifest(privacy, target, run_name):
    # Copied from the committed results/cia_frontier/{eurosat_frontier,influence_noise}/results/.
    return {"alpha": 0.3, "seed": 42, "privacy": privacy, "noise_ratio": 0.001546, "clients": 48,
            "out_target": target, "targets": [target], "rounds": 100, "pilot": False, "schema": 1,
            "hyperparams": HYPERPARAMS, "run_name": run_name,
            "score_direction": "lower loss indicates IN"}


@pytest.fixture
def no_training(monkeypatch):
    import experiments.cia.iter_combos as training
    import metricdp_pytorch.utils.device as device
    import results.cia_frontier.eurosat_frontier.data as data
    from experiments.cia import trajectories
    monkeypatch.setattr(device, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(data, "partition_summary", lambda *args: {})

    def fake_training(runs, **kwargs):
        paths = tuple(kwargs["output_dir"] / f"{r}.pt" for r in kwargs["checkpoint_rounds"])
        for path in paths:
            path.touch()
        yield runs[0], True, paths
    monkeypatch.setattr(training, "iter_combos", fake_training)
    monkeypatch.setattr(trajectories, "build_loaders", lambda shadows, combo: (None, shadows))
    monkeypatch.setattr(trajectories, "score_checkpoint", lambda path, *, shadow_loaders, **kwargs:
                        {t: (1.0, 2.0, 3.0, 10) for t in shadow_loaders})


def test_eurosat_frontier_run_name_and_manifest(tmp_path, no_training):
    from results.cia_frontier.eurosat_frontier.runner import build_combos, execute
    (combo,) = build_combos(alpha=0.3, seeds=[42], targets=list(range(10)), clients=48,
                            privacy="global-dp", ratios=[0.001546], adjacency="out", out_targets=[3])
    assert combo.run_name() == FRONTIER
    execute([combo], list(range(10)), tmp_path, 1)
    assert json.loads((tmp_path / FRONTIER / "manifest.json").read_text()) == \
        _manifest("global-dp", 3, FRONTIER)


def test_influence_noise_run_name_and_manifest(tmp_path, no_training):
    from results.cia_frontier.influence_noise.runner import build_influence_combos, execute
    (combo,) = build_influence_combos(fraction=0.0, seeds=[42], targets=list(range(10)),
                                      adjacency="out", out_targets=[0])
    assert combo.run_name() == INFLUENCE
    execute([combo], list(range(10)), tmp_path, 1)
    assert json.loads((tmp_path / INFLUENCE / "manifest.json").read_text()) == \
        _manifest("influence-noise", 0, INFLUENCE)
