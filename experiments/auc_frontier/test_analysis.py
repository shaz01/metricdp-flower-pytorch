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
