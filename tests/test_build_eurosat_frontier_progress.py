"""Synthetic tests for reports/build_eurosat_frontier_progress.py (no real data, no GPU)."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_eurosat_frontier_progress",
    Path(__file__).resolve().parents[1] / "reports" / "build_eurosat_frontier_progress.py")
bp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bp
SPEC.loader.exec_module(bp)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.invalid")
    git(tmp_path, "config", "user.name", "t")
    return tmp_path


def commit(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "synthetic")


def write_traj(repo, parent_rel, slot, in_losses=None, out_loss=1.0, acc=0.8, *, skip_round=None,
               nan=False, dup_row=False, extra_manifest=False, ratio=None):
    shard = repo / parent_rel / slot.shard
    run = f"run-{slot.shard}"
    traj = shard / run
    traj.mkdir(parents=True)
    targets = list(range(10)) if slot.out_target is None else [slot.out_target]
    man = dict(alpha=0.3, privacy=slot.privacy, seed=slot.seed, out_target=slot.out_target, rounds=100,
               pilot=False, targets=targets, run_name=run,
               noise_ratio=ratio if ratio is not None else (0.0 if slot.privacy == "vanilla" else 0.001546))
    (traj / "manifest.json").write_text(json.dumps(man))
    (traj / "complete.json").write_text(json.dumps({"complete": True}))
    rows = []
    for r in range(1, 101):
        if r == skip_round:
            continue
        for t in targets:
            loss = (in_losses[t] if slot.out_target is None else out_loss) if r == 100 else 2.0
            rows.append(dict(round=r, target=t, clean_loss=float("nan") if nan and r == 50 else loss,
                             noisy_loss=3.0, aggregate_loss=1.0, shadow_size=47))
    if dup_row:
        rows.append(dict(rows[0]))
    (traj / "measurements.json").write_text(json.dumps(rows))
    server = {str(r): {"accuracy": acc} for r in range(0, 101)}
    (traj / f"{run}.json").write_text(json.dumps({"server_evaluate_metrics": server,
                                                  "metadata": {"library_versions": {"torch": "x"}}}))
    per_class = {str(i): {"name": f"class{i}", "recall": acc} for i in range(10)}
    (traj / f"{run}.evaluation.json").write_text(json.dumps({"server_final_test": {
        "accuracy": acc, "per_class": per_class, "averages": {"macro": {"recall": acc}}}}))
    if slot.fraction is not None:
        (traj / "influence_protocol.json").write_text(json.dumps({"fraction": slot.fraction, "seed": 42}))
    if extra_manifest:
        (shard / "dup").mkdir()
        (shard / "dup" / "manifest.json").write_text(json.dumps(man))
    return traj


def collect_main(repo, slots):
    return bp.collect(repo, slots, bp.MAIN_REL, lambda s: bp.MAIN_REL / bp.COHORT_DIRS[s.cohort])


def test_paired_metric_direction_ties_and_duplicates():
    st = bp.paired([(0, 1.0, 2.0), (1, 2.0, 1.0), (2, 1.0, 1.0), (3, 0.5, 0.9)])
    assert st["n"] == 4 and st["in_lower"] == 2
    assert st["concordance"] == pytest.approx((2 + 0.5) / 4)
    assert st["mean_delta"] == pytest.approx((-1 + 1 + 0 - 0.4) / 4)
    with pytest.raises(ValueError):
        bp.paired([(0, 1.0, 2.0), (0, 1.0, 2.0)])
    with pytest.raises(ValueError):
        bp.paired([(0, math.nan, 1.0)])
    with pytest.raises(ValueError):
        bp.paired([])


def test_expected_cohorts_are_88_unique_with_documented_overlap():
    slots = bp.main_slots()
    assert len(slots) == len(set(slots)) == 88
    count = {c: sum(s.cohort == c for s in slots) for c in ("old", "pinned", "mixed")}
    assert count == {"old": 54, "pinned": 22, "mixed": 12}
    pinned = {(s.privacy, s.out_target) for s in slots if s.cohort == "pinned"}
    assert {t for p, t in pinned if p == "global-dp"} == {None, 0, 1, 2, 3}
    assert {t for p, t in pinned if p == "metric-privacy"} == {None, 0, 1, 2, 3, 4}
    assert len(bp.pilot_slots()) == 11 + 11 + 3


def test_validation_rejects_bad_trajectories(tmp_path):
    repo = make_repo(tmp_path)
    parent = bp.MAIN_REL / "frontier"
    good = bp.Slot("old", "global-dp", 42, 0)
    cases = {
        bp.Slot("old", "global-dp", 42, 1): dict(skip_round=37),
        bp.Slot("old", "global-dp", 42, 2): dict(nan=True),
        bp.Slot("old", "global-dp", 42, 3): dict(dup_row=True),
        bp.Slot("old", "global-dp", 42, 4): dict(extra_manifest=True),
        bp.Slot("old", "global-dp", 42, 5): dict(ratio=0.003092),
    }
    write_traj(repo, parent, good)
    for slot, kw in cases.items():
        write_traj(repo, parent, slot, **kw)
    commit(repo)
    uncommitted = bp.Slot("old", "global-dp", 42, 6)
    write_traj(repo, parent, uncommitted)
    missing = bp.Slot("old", "global-dp", 42, 7)
    coll = collect_main(repo, [good, *cases, uncommitted, missing])
    assert coll.status[good] == "valid"
    assert "coverage incomplete" in coll.status[bp.Slot("old", "global-dp", 42, 1)]
    assert "nonfinite" in coll.status[bp.Slot("old", "global-dp", 42, 2)]
    assert "duplicate" in coll.status[bp.Slot("old", "global-dp", 42, 3)]
    assert "2 manifests" in coll.status[bp.Slot("old", "global-dp", 42, 4)]
    assert "noise_ratio" in coll.status[bp.Slot("old", "global-dp", 42, 5)]
    assert "not committed" in coll.status[uncommitted]
    assert coll.status[missing] == "missing"
    assert set(coll.valid) == {good}


def test_modified_committed_file_is_rejected(tmp_path):
    repo = make_repo(tmp_path)
    slot = bp.Slot("old", "global-dp", 42, 0)
    traj = write_traj(repo, bp.MAIN_REL / "frontier", slot)
    commit(repo)
    (traj / "complete.json").write_text(json.dumps({"complete": True, "edited": 1}))
    assert "not committed" in collect_main(repo, [slot]).status[slot]


def test_group_selection_mixed_seed44_and_no_partial_aggregate(tmp_path):
    repo = make_repo(tmp_path)
    in_losses = {t: 1.0 for t in range(10)}
    write_traj(repo, bp.MAIN_REL / "frontier", bp.Slot("old", "metric-privacy", 44, None), in_losses)
    for t in range(10):
        cohort = "mixed" if t >= 4 else "old"
        out = 2.0 if t % 2 == 0 else 0.5  # IN lower on even targets
        write_traj(repo, bp.MAIN_REL / bp.COHORT_DIRS[cohort], bp.Slot(cohort, "metric-privacy", 44, t), out_loss=out)
    # seed 42 metric-privacy: IN plus only 3 OUTs -> must not produce an aggregate
    write_traj(repo, bp.MAIN_REL / "frontier", bp.Slot("old", "metric-privacy", 42, None), in_losses)
    for t in range(3):
        write_traj(repo, bp.MAIN_REL / "frontier", bp.Slot("old", "metric-privacy", 42, t), out_loss=2.0)
    commit(repo)
    coll = collect_main(repo, bp.main_slots())
    groups = {(g["privacy"], g["seed"]): g for g in bp.main_groups(coll)}
    g44 = groups[("metric-privacy", 44)]
    assert g44["complete"] and g44["kind"] == "mixed"
    assert g44["stats"]["in_lower"] == 5 and g44["stats"]["concordance"] == 0.5
    assert [p[3] for p in g44["pairs"]] == ["old"] * 4 + ["mixed"] * 6
    g42 = groups[("metric-privacy", 42)]
    assert not g42["complete"] and g42["stats"] is None and g42["out_acc_mean"] is None
    assert len(g42["pairs"]) == 3


def test_overlap_only_matched_targets(tmp_path):
    repo = make_repo(tmp_path)
    for cohort in ("old", "pinned"):
        parent = bp.MAIN_REL / bp.COHORT_DIRS[cohort]
        write_traj(repo, parent, bp.Slot(cohort, "global-dp", 42, None), {t: 1.0 for t in range(10)})
        write_traj(repo, parent, bp.Slot(cohort, "global-dp", 42, 0), out_loss=2.0 if cohort == "old" else 0.5)
    write_traj(repo, bp.MAIN_REL / "frontier", bp.Slot("old", "global-dp", 42, 1), out_loss=2.0)
    commit(repo)
    rows = bp.overlap_rows(collect_main(repo, bp.main_slots()))
    assert [(r["privacy"], r["target"]) for r in rows] == [("global-dp", 0)]
    assert rows[0]["same_sign"] is False


def test_pilot_partial_labels_common_subset_and_stopped_diagnostic(tmp_path):
    repo = make_repo(tmp_path)
    in_losses = {t: 1.0 for t in range(10)}
    avail = {0.0: range(7), 0.05: range(6), 0.5: range(2)}
    for f, targets in avail.items():
        write_traj(repo, bp.PILOT_REL, bp.Slot("pilot", "influence-noise", 42, None, f), in_losses, acc=0.9 - f)
        for t in targets:
            write_traj(repo, bp.PILOT_REL, bp.Slot("pilot", "influence-noise", 42, t, f), out_loss=2.0)
    commit(repo)
    coll = bp.collect(repo, bp.pilot_slots(), bp.PILOT_REL, lambda s: bp.PILOT_REL)
    ps = bp.pilot_summary(coll)
    assert ps["arms"][0.0]["full_stats"] is None and ps["arms"][0.05]["full_stats"] is None
    assert ps["common"] == [0, 1, 2, 3, 4, 5]
    assert ps["common_stats"][0.0]["n"] == 6 and ps["common_stats"][0.05]["n"] == 6
    assert ps["stopped_common"] == [0, 1]
    assert set(ps["stopped_diag"][0.5]) == {0, 1}
    assert ps["arms"][0.0]["pending"] == {7: "missing", 8: "missing", 9: "missing"}
    assert bp.pilot_status("missing") == "pending (not collected)"
    # protocol fraction mismatch is rejected
    bad = bp.Slot("pilot", "influence-noise", 42, 9, 0.0)
    traj = write_traj(repo, bp.PILOT_REL, bad, out_loss=2.0)
    (traj / "influence_protocol.json").write_text(json.dumps({"fraction": 0.5, "seed": 42}))
    commit(repo)
    coll = bp.collect(repo, bp.pilot_slots(), bp.PILOT_REL, lambda s: bp.PILOT_REL)
    assert "fraction" in coll.status[bad]


def test_build_writes_provisional_html_without_absolute_paths(tmp_path):
    repo = make_repo(tmp_path / "main")
    write_traj(repo, bp.MAIN_REL / "frontier", bp.Slot("old", "global-dp", 42, None), {t: 1.0 for t in range(10)})
    commit(repo)
    out = tmp_path / "out.html"
    data = bp.build(repo, tmp_path / "no-pilot-here", out)
    text = out.read_text()
    assert "PROVISIONAL" in text and "Pilot root not found" in text
    assert str(tmp_path) not in text
    assert sum(v == "valid" for v in data["coverage"].values()) == 1
