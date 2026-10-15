import json
from dataclasses import replace

import numpy as np
import pytest

from results.cia_frontier.eurosat_frontier.data import create_data_module, dirichlet_partitions
from results.cia_frontier.eurosat_frontier.runner import build_combos, execute
from results.cia_frontier.eurosat_frontier.analyze import statistics, summarize


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
    import results.cia_frontier.eurosat_frontier.data as data
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


def test_adjacency_selection_keeps_fixed_panel(tmp_path, monkeypatch):
    panel = list(range(10))
    names = lambda runs: [(c.out_target, c.privacy) for c in runs]
    assert names(combos(targets=panel, adjacency="in")) == [(None, "global-dp")] * 2
    out = combos(targets=panel, adjacency="out", out_targets=[0], seeds=[42])
    assert names(out) == [(0, "global-dp")]
    assert out[0].noise_multiplier == 0.001 * 47
    assert len(combos(targets=panel, out_targets=[0, 3], seeds=[42])) == 3
    assert len(combos(targets=panel, seeds=[42])) == 11
    for bad in ({"adjacency": "in", "out_targets": [0]}, {"out_targets": [10]},
                {"out_targets": [0, 0]}, {"adjacency": "sideways"}):
        with pytest.raises(ValueError):
            combos(targets=panel, **bad)
    with pytest.raises(ValueError):
        combos(privacy="vanilla", pilot=True, adjacency="in")
    # Shared IN manifest records (and evaluates) the whole panel.
    import experiments.cia.iter_combos as training
    import results.cia_frontier.eurosat_frontier.data as data
    from experiments.cia import cia
    import metricdp_pytorch.utils.device as device
    monkeypatch.setattr(data, "partition_summary", lambda *args: {})
    monkeypatch.setattr(device, "resolve_device", lambda: "cpu")
    def fake_training(runs, **kwargs):
        paths = tuple(kwargs["output_dir"] / f"{r}.pt" for r in kwargs["checkpoint_rounds"])
        for path in paths:
            path.touch()
        yield runs[0], True, paths
    monkeypatch.setattr(training, "iter_combos", fake_training)
    monkeypatch.setattr(cia, "eval_model", lambda path, **kwargs: (1.0, 2.0, 3.0, 10))
    selected = [replace(c, hyperparams=replace(c.hyperparams, rounds=2)) for c in
                combos(targets=panel, seeds=[42], adjacency="in")
                + combos(targets=panel, seeds=[42], adjacency="out", out_targets=[0])]
    execute(selected[:1], panel, tmp_path / "in", 1)
    execute(selected[1:], panel, tmp_path / "out", 1)
    in_manifest = json.loads(next((tmp_path / "in").glob("*/manifest.json")).read_text())
    out_manifest = json.loads(next((tmp_path / "out").glob("*/manifest.json")).read_text())
    assert in_manifest["targets"] == panel and in_manifest["out_target"] is None
    assert out_manifest["targets"] == [0] and out_manifest["out_target"] == 0
    rows = json.loads(next((tmp_path / "in").glob("*/measurements.json")).read_text())
    assert {(r["round"], r["target"]) for r in rows} == {(r, t) for r in (1, 2) for t in panel}
    marker = json.loads(next((tmp_path / "in").glob("*/complete.json")).read_text())
    assert marker["complete"] is True and "evaluation_seconds" in marker


def test_combo_subclass_keeps_master_combo_and_historical_run_names():
    """FrontierCombo/InfluenceCombo must define on top of master's Combo (which ends with
    a defaulted ``dirichlet_alpha``) and still reproduce the committed run names."""
    from results.cia_frontier.influence_noise.runner import build_influence_combos
    from results.cia_frontier.eurosat_frontier.runner import FrontierCombo
    from experiments.reproduce.matrix.combo import Combo

    assert issubclass(FrontierCombo, Combo)
    (gdp_in,) = build_combos(alpha=0.3, seeds=[42], targets=list(range(10)), clients=48,
                             privacy="global-dp", ratios=[0.001546], adjacency="in")
    assert gdp_in.dirichlet_alpha == 0.5 and gdp_in.partition == "non-iid"
    assert gdp_in.run_name() == (
        "eurosat-dirichlet-a0.3-in-r0.001546__non-iid__global-dp__fedavg__clients-48__seed-42"
        "__nm0p074208__clip5__rounds-100__epochs-5__data__eurosat_cnn")
    (pilot_in,) = build_influence_combos(fraction=0.0, seeds=[42], targets=list(range(10)),
                                         adjacency="in")
    assert pilot_in.run_name() == (
        "eurosat-dirichlet-a0.3-in-r0.001546-f0.0-cap5.0__non-iid__influence-noise__fedavg"
        "__clients-48__seed-42__nm0p074208__clip5__rounds-100__epochs-5__data__eurosat_cnn")
    with pytest.raises(TypeError):
        FrontierCombo(*[None] * 15)  # subclass fields are keyword-only


def test_diagnostics_log_increments_without_changing_training_identity(tmp_path, monkeypatch):
    import experiments.cia.iter_combos as training
    import results.cia_frontier.eurosat_frontier.data as data
    from experiments.cia import cia
    import metricdp_pytorch.utils.device as device
    monkeypatch.setattr(data, "partition_summary", lambda *args: {})
    monkeypatch.setattr(device, "resolve_device", lambda: "cpu")
    plain, logged = combos(seeds=[42])[0], combos(seeds=[42], diagnostics=True)[0]
    assert plain.run_name() == logged.run_name()
    args = lambda c: c.runner_args(output_dir=tmp_path, max_parallel_clients=1, client_cpus=1)
    assert "--log-loo-residuals" not in args(plain)
    assert args(logged) == (*args(plain), "--log-loo-residuals")

    import torch

    def fake_training(runs, **kwargs):
        assert kwargs["checkpoint_rounds"] == (0, 1, 2)
        paths = tuple(kwargs["output_dir"] / f"round{r}.pt" for r in kwargs["checkpoint_rounds"])
        for r, path in zip(kwargs["checkpoint_rounds"], paths):
            # weights 3^r * (1, 1); an integer buffer that must be ignored
            torch.save({"w": torch.full((2,), 3.0 ** r), "steps": torch.tensor(r)}, path)
        yield runs[0], True, paths
    monkeypatch.setattr(training, "iter_combos", fake_training)
    # Loss at round r for target t: 10 - r + t (clean), 20 - 2r (noisy); grad norm 1 + r.
    def fake_eval(path, **kwargs):
        r = int(path.stem.removeprefix("round"))
        target = kwargs["clean_data_module"].target_partition_id
        return {"aggregate_loss": 1.0, "clean_loss": 10.0 - r + target,
                "noisy_loss": 20.0 - 2 * r, "shadow_size": 5,
                "clean_grad_norm": 1.0 + r, "noisy_grad_norm": 2.0 + r}
    monkeypatch.setattr(cia, "eval_model_with_gradients", fake_eval)
    monkeypatch.setattr(cia, "eval_model", lambda *a, **k: pytest.fail("plain evaluation used"))

    logged = replace(logged, hyperparams=replace(logged.hyperparams, rounds=2))
    execute([logged], [0, 47], tmp_path, 1)

    folder = next(tmp_path.glob("*/manifest.json")).parent
    assert json.loads((folder / "manifest.json").read_text())["diagnostics"] is True
    rows = {(r["round"], r["target"]): r for r in json.loads((folder / "measurements.json").read_text())}
    assert set(rows) == {(r, t) for r in (1, 2) for t in (0, 47)}
    row = rows[(2, 47)]
    assert row["clean_loss"] == 55.0 and row["clean_loss_before"] == 56.0
    assert row["clean_increment"] == -1.0 and row["noisy_increment"] == -2.0
    assert row["clean_grad_norm_start"] == 2.0 and row["noisy_grad_norm_start"] == 3.0
    assert rows[(1, 0)]["clean_loss_before"] == 10.0  # the round-0 (initial) model
    assert not list(tmp_path.glob("*/*.pt"))
    releases = json.loads((folder / "releases.json").read_text())
    assert [r["round"] for r in releases] == [1, 2]
    assert releases[0]["update_norm"] == pytest.approx(2 * 2 ** 0.5)  # (3 - 1) * sqrt(2)
    assert releases[1]["update_norm"] == pytest.approx(6 * 2 ** 0.5)  # (9 - 3) * sqrt(2)
    assert {r["parameter_count"] for r in releases} == {2}


def test_dirichlet_eurosat_accepts_every_keyword_the_partition_views_forward():
    """Since 2026-10-02 the partition views forward ``dirichlet_alpha``; the frontier
    module must accept it, or every client fails before training (round skipped)."""
    import inspect

    from experiments.cia.datasets.partitions import PartitionViewDataModule
    from results.cia_frontier.eurosat_frontier.data import DirichletEuroSAT

    forwarded = {name for name, parameter in inspect.signature(
        PartitionViewDataModule.client_loaders).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY}
    accepted = set(inspect.signature(DirichletEuroSAT.client_loaders).parameters)
    assert "dirichlet_alpha" in forwarded
    assert forwarded <= accepted
