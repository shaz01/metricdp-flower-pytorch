import json

import pytest

from experiments.auc_frontier.analyze import analyze


def test_final_only_and_incomplete_rejection(tmp_path):
    for seed in (42, 43):
        for out in (None, 0, 1):
            name = f"seed-{seed}-out-{out}"
            folder = tmp_path / name
            folder.mkdir()
            targets = [0, 1] if out is None else [out]
            manifest = dict(alpha=.3, privacy="global-dp", noise_ratio=.001,
                            clients=48, rounds=2, seed=seed, out_target=out,
                            targets=targets, run_name=name, pilot=False)
            (folder / "manifest.json").write_text(json.dumps(manifest))
            (folder / "complete.json").write_text("{}")
            # Early rounds give the opposite result: analysis must ignore them.
            rows = [dict(round=r, target=t, clean_loss=(1 if out is None else 2) * (1 if r == 2 else -1))
                    for r in (1, 2) for t in targets]
            (folder / "measurements.json").write_text(json.dumps(rows))
            (folder / f"{name}.json").write_text(json.dumps({
                "server_evaluate_metrics": {"2": {"accuracy": .8}}}))
    result = analyze(tmp_path)
    assert len(result) == 1
    assert result[0]["paired_concordance"] == 1
    assert result[0]["target_stratified_auc"] == 1
    assert result[0]["ci95"] is None
    assert analyze(tmp_path, [43])[0]["seed_ids"] == [43]
    path = next(tmp_path.glob("*/measurements.json"))
    path.write_text("[]")
    with pytest.raises(ValueError, match="Missing or duplicate"):
        analyze(tmp_path)


def _trajectory(folder, seed, out, targets, rounds=2):
    name = f"seed-{seed}-out-{out}"
    folder = folder / name
    folder.mkdir(parents=True)
    manifest = dict(alpha=.3, privacy="global-dp", noise_ratio=.001, clients=48,
                    rounds=rounds, seed=seed, out_target=out,
                    targets=targets if out is None else [out], run_name=name, pilot=False)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "complete.json").write_text("{}")
    rows = [dict(round=r, target=t, clean_loss=1 if out is None else 2)
            for r in range(1, rounds + 1) for t in manifest["targets"]]
    (folder / "measurements.json").write_text(json.dumps(rows))
    (folder / f"{name}.json").write_text(json.dumps(
        {"server_evaluate_metrics": {str(rounds): {"accuracy": .8}}}))


def test_nested_shards_and_partial_wave_rejected(tmp_path):
    panel = list(range(10))
    # First wave: shared IN on the whole panel plus only OUT target 0, each in its own shard.
    _trajectory(tmp_path / "gdp-in", 42, None, panel)
    _trajectory(tmp_path / "gdp-out-0", 42, 0, panel)
    with pytest.raises(ValueError, match="Incomplete or unbalanced"):
        analyze(tmp_path)
    for target in panel[1:]:
        _trajectory(tmp_path / f"gdp-out-{target}", 42, target, panel)
    result = analyze(tmp_path)
    assert len(result) == 1 and result[0]["target_ids"] == panel
    _trajectory(tmp_path / "duplicate", 42, 0, panel)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(tmp_path)
