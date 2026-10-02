"""PROVISIONAL progress visualization for the EuroSAT alpha-.3 AUC-frontier run and the
separate influence-noise pilot. Not a finished-experiment report; no conclusions.

Every number is recomputed from git-tracked JSON artifacts at generation time:
manifest/complete/measurements/training-run/evaluation JSONs (plus
influence_protocol.json for the pilot). A trajectory is used only if every such file
is tracked and unmodified, rounds 1-100 are all present exactly once with finite
losses/accuracies, and its manifest matches the explicitly expected identity.
Missing, pending or invalid trajectories are listed, never imputed or averaged over.

Cohorts are selected explicitly (no recursive pooling of mixed roots):
  old     results/new_auc_frontier_eurosat/frontier/                       (54)
  pinned  results/new_auc_frontier_eurosat/frontier_torch210/              (22)
  mixed   results/new_auc_frontier_eurosat/frontier_torch210_s44_mixedenv/ (12)

Run from the repository root (safe to rerun any time; overwrites the HTML):
    uv run python reports/build_eurosat_frontier_progress.py [--pilot-root PATH]

--pilot-root defaults to the sibling worktree ../metricdp-pytorch-influence-noise-pilot;
if absent, the pilot panel says so. Only the directory basename, branch and commit of
each repository are written into the HTML.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_REL = Path("results/new_auc_frontier_eurosat")
PILOT_REL = Path("results/influence_noise_pilot")
DEFAULT_PILOT_ROOT = ROOT.parent / "metricdp-pytorch-influence-noise-pilot"
OUTPUT = ROOT / "reports" / "eurosat_frontier_progress.html"

ALPHA, RATIO, ROUNDS, TARGETS = 0.3, 0.001546, 100, tuple(range(10))
MECHS = ("global-dp", "metric-privacy")
COHORT_DIRS = {"old": "frontier", "pinned": "frontier_torch210",
               "mixed": "frontier_torch210_s44_mixedenv"}
ENV_LABEL = {"old": "old runtime", "pinned": "pinned torch 2.10",
             "mixed": "pinned torch 2.10 (seed-44 mixed-env OUT)"}
PILOT_ARMS = {0.0: TARGETS, 0.05: TARGETS, 0.5: (0, 1)}  # f=.5 early-stopped at OUT 0,1
COLORS = {"global-dp": "#2166ac", "metric-privacy": "#b2182b", "vanilla": "#222",
          0.0: "#4d4d4d", 0.05: "#1b9e77", 0.5: "#d95f02"}
SHORT = {"global-dp": "GDP", "metric-privacy": "MP", "vanilla": "Vanilla"}


class Invalid(Exception):
    pass


@dataclass(frozen=True)
class Slot:
    """One explicitly expected trajectory. `out_target` None = shared IN."""
    cohort: str
    privacy: str
    seed: int
    out_target: int | None
    fraction: float | None = None  # pilot only

    @property
    def shard(self) -> str:
        side = "in" if self.out_target is None else f"out-{self.out_target}"
        if self.fraction is not None:
            return f"f{self.fraction!r}-seed-{self.seed}-{side}"
        if self.privacy == "vanilla":
            return f"vanilla-seed-{self.seed}-{side}"
        return f"{self.privacy}-r{RATIO}-seed-{self.seed}-{side}"


@dataclass
class Traj:
    slot: Slot
    rel: str
    final_clean: dict
    accuracy: float
    eval_accuracy: float
    recall: dict
    macro_recall: float
    torch: str | None


@dataclass
class Collection:
    valid: dict = field(default_factory=dict)      # Slot -> Traj
    status: dict = field(default_factory=dict)     # Slot -> "valid" | "missing" | reason
    extras: list = field(default_factory=list)     # unselected shard dirs


# ---------------------------------------------------------------- expected cohorts
def main_slots() -> list[Slot]:
    slots = []
    for m in MECHS:
        for s in (42, 43):
            slots += [Slot("old", m, s, t) for t in (None, *TARGETS)]
        slots += [Slot("old", m, 44, t) for t in (None, 0, 1, 2, 3)]
        slots += [Slot("mixed", m, 44, t) for t in range(4, 10)]
    slots += [Slot("pinned", "global-dp", 42, t) for t in (None, 0, 1, 2, 3)]
    slots += [Slot("pinned", "metric-privacy", 42, t) for t in (None, 0, 1, 2, 3, 4)]
    slots += [Slot("pinned", "vanilla", 42, t) for t in (None, *TARGETS)]
    return slots


def pilot_slots() -> list[Slot]:
    return [Slot("pilot", "influence-noise", 42, t, f)
            for f, targets in PILOT_ARMS.items() for t in (None, *targets)]


# ---------------------------------------------------------------- git helpers
def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def committed_files(repo: Path, rel: Path) -> set[str]:
    """Tracked files under `rel` that have no staged/unstaged modification."""
    tracked = set(git(repo, "ls-files", "-z", "--", str(rel)).split("\0")) - {""}
    dirty = {line[3:] for line in git(repo, "status", "--porcelain=v1", "--", str(rel)).splitlines()}
    return tracked - dirty


def repo_info(repo: Path, rel: Path) -> dict:
    try:
        return dict(name=repo.name, branch=git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip(),
                    head=git(repo, "rev-parse", "--short=10", "HEAD").strip(),
                    results_commit=git(repo, "log", "-1", "--format=%h %cI", "--", str(rel)).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return dict(name=repo.name, branch="?", head="unavailable", results_commit="unavailable")


# ---------------------------------------------------------------- validation
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_traj(repo: Path, shard: Path, slot: Slot, committed: set[str]) -> Traj:
    manifests = sorted(shard.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise Invalid(f"{len(manifests)} manifests in shard (duplicate or incomplete)")
    traj = manifests[0].parent
    man = json.loads(manifests[0].read_text())
    run_name = man.get("run_name", "")
    names = ["manifest.json", "complete.json", "measurements.json",
             f"{run_name}.json", f"{run_name}.evaluation.json"]
    if slot.fraction is not None:
        names.append("influence_protocol.json")
    for name in names:
        path = traj / name
        if not path.exists():
            raise Invalid(f"missing {name}")
        if path.relative_to(repo).as_posix() not in committed:
            raise Invalid(f"{name} not committed")
    expect_targets = list(TARGETS) if slot.out_target is None else [slot.out_target]
    if slot.fraction is not None and slot.out_target is None:
        expect_targets = list(TARGETS)
    ratio = 0.0 if slot.privacy == "vanilla" else RATIO
    checks = dict(alpha=ALPHA, privacy=slot.privacy, seed=slot.seed, out_target=slot.out_target,
                  rounds=ROUNDS, pilot=False, targets=expect_targets, noise_ratio=ratio)
    for key, want in checks.items():
        if man.get(key) != want:
            raise Invalid(f"manifest {key}={man.get(key)!r}, expected {want!r}")
    if json.loads((traj / "complete.json").read_text()).get("complete") is not True:
        raise Invalid("complete.json not complete")
    if slot.fraction is not None:
        proto = json.loads((traj / "influence_protocol.json").read_text())
        if proto.get("fraction") != slot.fraction or proto.get("seed") != slot.seed:
            raise Invalid("influence_protocol fraction/seed mismatch")
    rows = json.loads((traj / "measurements.json").read_text())
    keys = [(r.get("round"), r.get("target")) for r in rows]
    expected = {(r, t) for r in range(1, ROUNDS + 1) for t in expect_targets}
    if len(keys) != len(set(keys)):
        raise Invalid("duplicate round/target measurement rows")
    if set(keys) != expected:
        raise Invalid("measurement round/target coverage incomplete")
    for r in rows:
        if not all(_finite(r.get(k)) for k in ("clean_loss", "noisy_loss", "aggregate_loss")):
            raise Invalid(f"nonfinite loss at round {r['round']} target {r['target']}")
    run = json.loads((traj / f"{run_name}.json").read_text())
    server = run.get("server_evaluate_metrics", {})
    for rnd in range(1, ROUNDS + 1):
        if not _finite(server.get(str(rnd), {}).get("accuracy")):
            raise Invalid(f"missing/nonfinite server accuracy round {rnd}")
    evaluation = json.loads((traj / f"{run_name}.evaluation.json").read_text())
    test = evaluation.get("server_final_test", {})
    recall = {v["name"]: v["recall"] for v in test.get("per_class", {}).values()}
    macro = test.get("averages", {}).get("macro", {}).get("recall")
    if len(recall) != 10 or not all(map(_finite, [*recall.values(), macro, test.get("accuracy")])):
        raise Invalid("evaluation per-class recall/accuracy missing or nonfinite")
    accuracy = float(server[str(ROUNDS)]["accuracy"])
    if abs(test["accuracy"] - accuracy) > 1e-9:
        raise Invalid("evaluation accuracy disagrees with round-100 run accuracy")
    final = {r["target"]: float(r["clean_loss"]) for r in rows if r["round"] == ROUNDS}
    return Traj(slot, traj.relative_to(repo).as_posix(), final, accuracy, float(test["accuracy"]),
                recall, float(macro), run.get("metadata", {}).get("library_versions", {}).get("torch"))


def collect(repo: Path, slots: list[Slot], root_rel: Path, cohort_dir) -> Collection:
    """`cohort_dir(slot)` -> directory (relative to repo) holding that slot's shard."""
    out = Collection()
    committed = committed_files(repo, root_rel) if (repo / ".git").exists() else set()
    seen_dirs, identities = set(), set()
    for slot in slots:
        shard = repo / cohort_dir(slot) / slot.shard
        seen_dirs.add(shard)
        identity = (slot.cohort, slot.privacy, slot.seed, slot.out_target, slot.fraction)
        if identity in identities:
            raise ValueError(f"duplicate expected slot {identity}")
        identities.add(identity)
        if not shard.exists():
            out.status[slot] = "missing"
            continue
        if not any(shard.glob("*/complete.json")):
            out.status[slot] = "incomplete (no completion marker)"
            continue
        try:
            out.valid[slot] = load_traj(repo, shard, slot, committed)
            out.status[slot] = "valid"
        except (Invalid, json.JSONDecodeError, KeyError, TypeError) as exc:
            out.status[slot] = f"invalid: {exc}"
    for parent in {repo / cohort_dir(s) for s in slots}:
        if parent.exists():
            out.extras += sorted(p.relative_to(repo).as_posix() for p in parent.iterdir()
                                 if p.is_dir() and p not in seen_dirs and p.name != "failures")
    return out


# ---------------------------------------------------------------- metrics
def paired(pairs: list[tuple[int, float, float]]) -> dict:
    """pairs = [(target, in_clean, out_clean)]. Lower loss = IN (fixed direction)."""
    deltas = [(t, i - o) for t, i, o in pairs]
    if not deltas or not all(math.isfinite(d) for _, d in deltas):
        raise ValueError("need nonempty finite pairs")
    if len({t for t, _ in deltas}) != len(deltas):
        raise ValueError("duplicate target in pairs")
    lower = sum(d < 0 for _, d in deltas)
    ties = sum(d == 0 for _, d in deltas)
    return dict(n=len(deltas), in_lower=lower, concordance=(lower + 0.5 * ties) / len(deltas),
                mean_delta=sum(d for _, d in deltas) / len(deltas),
                deltas={t: d for t, d in deltas})


def group_pairs(coll: Collection, in_slot: Slot, out_slots: dict[int, Slot]):
    """Available (target, in, out, out_cohort) pairs; never fills gaps."""
    if in_slot not in coll.valid:
        return []
    tin = coll.valid[in_slot]
    return [(t, tin.final_clean[t], coll.valid[s].final_clean[t], s.cohort)
            for t, s in sorted(out_slots.items()) if s in coll.valid]


def main_groups(coll: Collection) -> list[dict]:
    groups = []
    for m in MECHS:
        for seed in (42, 43, 44):
            outs = {t: Slot("mixed" if seed == 44 and t >= 4 else "old", m, seed, t) for t in TARGETS}
            label = "old IN + old OUT" if seed != 44 else "old IN; OUT 0-3 old, OUT 4-9 pinned (mixed)"
            groups.append(dict(privacy=m, seed=seed, kind="mixed" if seed == 44 else "old",
                               env=label, in_slot=Slot("old", m, seed, None), outs=outs))
    groups.append(dict(privacy="vanilla", seed=42, kind="pinned", env="pinned IN + pinned OUT",
                       in_slot=Slot("pinned", "vanilla", 42, None),
                       outs={t: Slot("pinned", "vanilla", 42, t) for t in TARGETS}))
    for g in groups:
        g["pairs"] = group_pairs(coll, g["in_slot"], g["outs"])
        g["complete"] = len(g["pairs"]) == len(TARGETS)
        g["stats"] = paired([p[:3] for p in g["pairs"]]) if g["complete"] else None
        g["in_accuracy"] = coll.valid[g["in_slot"]].accuracy if g["in_slot"] in coll.valid else None
        outs = [coll.valid[s].accuracy for s in g["outs"].values() if s in coll.valid]
        g["out_acc_mean"] = sum(outs) / len(outs) if g["complete"] else None
    return groups


def overlap_rows(coll: Collection) -> list[dict]:
    rows = []
    for m in MECHS:
        old_in, new_in = Slot("old", m, 42, None), Slot("pinned", m, 42, None)
        for t in TARGETS:
            old_out, new_out = Slot("old", m, 42, t), Slot("pinned", m, 42, t)
            if not all(s in coll.valid for s in (old_in, new_in, old_out, new_out)):
                continue
            v = coll.valid
            od = v[old_in].final_clean[t] - v[old_out].final_clean[t]
            nd = v[new_in].final_clean[t] - v[new_out].final_clean[t]
            rows.append(dict(privacy=m, target=t, old_in_acc=v[old_in].accuracy,
                             new_in_acc=v[new_in].accuracy, old_out_acc=v[old_out].accuracy,
                             new_out_acc=v[new_out].accuracy, old_delta=od, new_delta=nd,
                             same_sign=(od < 0) == (nd < 0)))
    return rows


def pilot_summary(coll: Collection) -> dict:
    arms = {}
    for f, planned in PILOT_ARMS.items():
        in_slot = Slot("pilot", "influence-noise", 42, None, f)
        outs = {t: Slot("pilot", "influence-noise", 42, t, f) for t in planned}
        pairs = group_pairs(coll, in_slot, outs)
        arms[f] = dict(planned=list(planned), in_valid=in_slot in coll.valid,
                       in_status=coll.status.get(in_slot, "missing"),
                       valid_targets=[p[0] for p in pairs],
                       pending={t: coll.status[s] for t, s in outs.items() if s not in coll.valid},
                       in_traj=coll.valid.get(in_slot), pairs=pairs,
                       full=len(pairs) == 10,
                       full_stats=paired([p[:3] for p in pairs]) if len(pairs) == 10 else None)
    common = sorted(set(arms[0.0]["valid_targets"]) & set(arms[0.05]["valid_targets"]))
    subset = {f: paired([p[:3] for p in arms[f]["pairs"] if p[0] in common]) if common else None
              for f in (0.0, 0.05)}
    stopped_common = sorted(set(arms[0.5]["valid_targets"]) & set(arms[0.0]["valid_targets"])
                            & set(arms[0.05]["valid_targets"]))
    diag = {f: {p[0]: p[1] - p[2] for p in arms[f]["pairs"] if p[0] in stopped_common}
            for f in PILOT_ARMS}
    return dict(arms=arms, common=common, common_stats=subset,
                stopped_common=stopped_common, stopped_diag=diag)


# ---------------------------------------------------------------- SVG helpers
W, H = 560, 340
M = dict(l=58, r=16, t=16, b=44)


def esc(x) -> str:
    return html.escape(str(x))


def pct(x) -> str:
    return "&mdash;" if x is None else f"{100 * x:.2f}%"


def num(x, d=4) -> str:
    return "&mdash;" if x is None else f"{x:+.{d}f}"


def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a) if hi != lo else (a + b) / 2


def axes(xlo, xhi, ylo, yhi, xticks, yticks, xlab, ylab, xfmt, yfmt) -> list[str]:
    x0, x1, y0, y1 = M["l"], W - M["r"], M["t"], H - M["b"]
    p = []
    for v in xticks:
        x = scale(v, xlo, xhi, x0, x1)
        p.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="#e3e3e3"/>'
                 f'<text x="{x:.1f}" y="{y1 + 15}" text-anchor="middle" font-size="11">{xfmt(v)}</text>')
    for v in yticks:
        y = scale(v, ylo, yhi, y1, y0)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e3e3e3"/>'
                 f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{yfmt(v)}</text>')
    p.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="#777"/>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{H - 8}" text-anchor="middle" font-size="12">{xlab}</text>')
    p.append(f'<text x="14" y="{(y0 + y1) / 2}" text-anchor="middle" font-size="12" '
             f'transform="rotate(-90 14 {(y0 + y1) / 2})">{ylab}</text>')
    return p


def ticks(lo, hi, n=5):
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def svg(parts, label, h=H) -> str:
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(label)}">{"".join(parts)}</svg>'


def marker(kind, x, y, color, title) -> str:
    t = f"<title>{esc(title)}</title>"
    if kind == "old":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}">{t}</circle>'
    if kind == "mixed":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#fff" stroke="{color}" stroke-width="2.2">{t}</circle>'
    return (f'<rect x="{x - 5.5:.1f}" y="{y - 5.5:.1f}" width="11" height="11" fill="#fff" stroke="{color}" '
            f'stroke-width="2.2" transform="rotate(45 {x:.1f} {y:.1f})">{t}</rect>')


# ---------------------------------------------------------------- panels
def panel_concordance(groups) -> str:
    done = [g for g in groups if g["complete"]]
    if not done:
        return '<p class="warn">No group has all 10 targets valid yet.</p>'
    accs = [g["in_accuracy"] for g in done]
    ylo, yhi = math.floor(min(accs) * 100 - 1) / 100, math.ceil(max(accs) * 100 + 1) / 100
    yt = [v / 100 for v in range(round(ylo * 100), round(yhi * 100) + 1)]
    p = axes(0, 1, ylo, yhi, ticks(0, 1), yt, "Paired concordance (clean loss, IN lower = hit), 10 targets",
             "Final IN accuracy (round 100)", lambda v: f"{v:.1f}", lambda v: f"{100 * v:.0f}%")
    x5 = scale(0.5, 0, 1, M["l"], W - M["r"])
    p.append(f'<line x1="{x5}" y1="{M["t"]}" x2="{x5}" y2="{H - M["b"]}" stroke="#999" stroke-dasharray="4 3"/>')
    for g in done:
        x = scale(g["stats"]["concordance"], 0, 1, M["l"], W - M["r"])
        y = scale(g["in_accuracy"], ylo, yhi, H - M["b"], M["t"])
        title = f'{g["privacy"]} seed {g["seed"]} ({g["env"]}): concordance {g["stats"]["concordance"]:.2f}, IN acc {100 * g["in_accuracy"]:.2f}%'
        p.append(marker(g["kind"], x, y, COLORS[g["privacy"]], title))
        p.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" font-size="10" fill="{COLORS[g["privacy"]]}">'
                 f'{SHORT[g["privacy"]]} s{g["seed"]}</text>')
    return svg(p, "Paired concordance vs final IN accuracy")


def panel_strip(groups) -> str:
    vals = [p[1] - p[2] for g in groups for p in g["pairs"]]
    if not vals:
        return '<p class="warn">No valid IN/OUT pairs.</p>'
    lim = max(abs(v) for v in vals) * 1.1
    row_h, top = 30, 16
    h = top + row_h * len(groups) + 44
    x0, x1 = 190, W - M["r"]
    p = []
    for v in ticks(-lim, lim, 6):
        x = scale(v, -lim, lim, x0, x1)
        p.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - 44}" stroke="{"#888" if abs(v) < 1e-12 else "#e3e3e3"}"/>'
                 f'<text x="{x:.1f}" y="{h - 30}" text-anchor="middle" font-size="10">{v:+.2f}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 10}" text-anchor="middle" font-size="12">'
             f'IN &minus; OUT clean loss at round 100 (left of 0 = IN lower)</text>')
    for i, g in enumerate(groups):
        y = top + row_h * i + row_h / 2
        p.append(f'<text x="{x0 - 8}" y="{y + 4}" text-anchor="end" font-size="11" fill="{COLORS[g["privacy"]]}">'
                 f'{SHORT[g["privacy"]]} s{g["seed"]} ({len(g["pairs"])}/10)</text>')
        for t, i_loss, o_loss, oc in g["pairs"]:
            kind = "old" if (g["kind"] == "old" or oc == "old") else ("mixed" if oc == "mixed" else "pinned")
            x = scale(i_loss - o_loss, -lim, lim, x0, x1)
            p.append(marker(kind, x, y, COLORS[g["privacy"]], f"target {t}: {i_loss - o_loss:+.4f} (OUT {ENV_LABEL[oc]})")
                     .replace('r="6"', 'r="4.5"'))
    return svg(p, "Per-client IN minus OUT clean loss", h)


def panel_overlap_svg(rows) -> str:
    if not rows:
        return '<p class="warn">No matched old/pinned overlap targets valid.</p>'
    lim = max(max(abs(r["old_delta"]), abs(r["new_delta"])) for r in rows) * 1.15
    p = axes(-lim, lim, -lim, lim, ticks(-lim, lim, 4), ticks(-lim, lim, 4),
             "old-runtime IN&minus;OUT clean loss", "pinned-runtime IN&minus;OUT clean loss",
             lambda v: f"{v:+.2f}", lambda v: f"{v:+.2f}")
    cx, cy = scale(0, -lim, lim, M["l"], W - M["r"]), scale(0, -lim, lim, H - M["b"], M["t"])
    p.append(f'<line x1="{M["l"]}" y1="{cy}" x2="{W - M["r"]}" y2="{cy}" stroke="#888"/>'
             f'<line x1="{cx}" y1="{M["t"]}" x2="{cx}" y2="{H - M["b"]}" stroke="#888"/>'
             f'<line x1="{M["l"]}" y1="{H - M["b"]}" x2="{W - M["r"]}" y2="{M["t"]}" stroke="#bbb" stroke-dasharray="4 3"/>')
    for r in rows:
        x = scale(r["old_delta"], -lim, lim, M["l"], W - M["r"])
        y = scale(r["new_delta"], -lim, lim, H - M["b"], M["t"])
        p.append(marker("old" if r["same_sign"] else "mixed", x, y, COLORS[r["privacy"]],
                        f'{r["privacy"]} target {r["target"]}'))
        if not r["same_sign"] or abs(r["old_delta"]) > 0.1:
            p.append(f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" font-size="10" fill="{COLORS[r["privacy"]]}">'
                     f'{SHORT[r["privacy"]]} t{r["target"]}</text>')
    return svg(p, "Seed-42 old vs pinned overlap")


def panel_recall_svg(pilot) -> str:
    arms = [(f, a["in_traj"]) for f, a in pilot["arms"].items() if a["in_traj"] is not None]
    if not arms:
        return '<p class="warn">No valid pilot IN evaluation yet.</p>'
    classes = list(arms[0][1].recall)
    h, x0, x1, top, bottom = 360, 56, W - 10, 16, 118
    p = []
    for v in (0, .25, .5, .75, 1):
        y = scale(v, 0, 1, h - bottom, top)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e3e3e3"/>'
                 f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{100 * v:.0f}%</text>')
    cats = [*classes, "MACRO"]
    slot_w = (x1 - x0) / len(cats)
    bw = slot_w * 0.8 / len(arms)
    for ci, c in enumerate(cats):
        for ai, (f, tr) in enumerate(arms):
            v = tr.macro_recall if c == "MACRO" else tr.recall[c]
            x = x0 + ci * slot_w + slot_w * 0.1 + ai * bw
            y = scale(v, 0, 1, h - bottom, top)
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h - bottom - y:.1f}" '
                     f'fill="{COLORS[f]}"{" fill-opacity=\"0.55\"" if f == 0.5 else ""}>'
                     f'<title>f={f}: {esc(c)} recall {100 * v:.1f}%</title></rect>')
        lx = x0 + (ci + 0.5) * slot_w
        p.append(f'<text x="{lx:.1f}" y="{h - bottom + 12}" font-size="10" text-anchor="end" '
                 f'transform="rotate(-45 {lx:.1f} {h - bottom + 12})">{esc(c if len(c) <= 24 else c[:22] + '…')}</text>')
    p.append(f'<text x="14" y="{(top + h - bottom) / 2}" text-anchor="middle" font-size="12" '
             f'transform="rotate(-90 14 {(top + h - bottom) / 2})">IN recall, server test split (n=1350)</text>')
    return svg(p, "Pilot IN per-class recall", h)


# ---------------------------------------------------------------- HTML
def table(head, rows) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def pilot_status(status: str) -> str:
    return "pending (not collected)" if status == "missing" else status


def coverage_rows(main: Collection, alpha: dict, pilot: Collection | None) -> list:
    rows = []
    for cohort in ("old", "pinned", "mixed"):
        slots = [s for s in main.status if s.cohort == cohort]
        ok = sum(main.status[s] == "valid" for s in slots)
        bad = [f"{s.shard}: {main.status[s]}" for s in slots if main.status[s] != "valid"]
        rows.append([f"main &middot; {ENV_LABEL[cohort]}", f"<code>{COHORT_DIRS[cohort]}/</code>",
                     len(slots), ok, esc("; ".join(bad)) or "&mdash;"])
    total = len(main.status)
    rows.append(["<b>main total</b>", "", total, sum(v == "valid" for v in main.status.values()),
                 esc(f"unselected dirs present: {', '.join(main.extras)}") if main.extras else "&mdash;"])
    rows.append(["alpha pilot (vanilla IN only, separate)", "<code>alpha_pilot/</code>",
                 alpha["expected"], alpha["valid"], esc("; ".join(alpha["problems"])) or "&mdash;"])
    if pilot is not None:
        for f, planned in PILOT_ARMS.items():
            slots = [s for s in pilot.status if s.fraction == f]
            ok = sum(pilot.status[s] == "valid" for s in slots)
            miss = [s.shard.split("seed-42-")[1] + f" ({pilot_status(pilot.status[s])})"
                    for s in slots if pilot.status[s] != "valid"]
            note = " (early-stopped: OUT 0,1 only)" if f == 0.5 else ""
            rows.append([f"influence pilot f={f}{note}", f"<code>{PILOT_REL.name}/</code>",
                         len(slots), ok, esc("; ".join(miss)) or "&mdash;"])
    return rows


def alpha_coverage(repo: Path) -> dict:
    committed = committed_files(repo, MAIN_REL / "alpha_pilot") if (repo / ".git").exists() else set()
    problems, valid, expected = [], 0, 0
    for a in ("0.1", "0.3", "1", "10"):
        for s in (42, 43, 44):
            expected += 1
            shard = repo / MAIN_REL / "alpha_pilot" / f"alpha-{a}-seed-{s}"
            try:
                (man,) = shard.glob("*/manifest.json")
                m = json.loads(man.read_text())
                run = man.parent / f'{m["run_name"]}.json'
                if run.relative_to(repo).as_posix() not in committed or not (man.parent / "complete.json").exists():
                    raise Invalid("uncommitted or incomplete")
                srv = json.loads(run.read_text())["server_evaluate_metrics"]
                if not all(_finite(srv.get(str(r), {}).get("accuracy")) for r in range(1, ROUNDS + 1)):
                    raise Invalid("round accuracies incomplete")
                valid += 1
            except (Invalid, ValueError, KeyError) as exc:
                problems.append(f"alpha-{a}-seed-{s}: {exc}")
    return dict(expected=expected, valid=valid, problems=problems)


def runtimes(coll: Collection) -> str:
    seen = {}
    for s, t in coll.valid.items():
        seen.setdefault(s.cohort, set()).add(t.torch or "unrecorded")
    return "; ".join(f"{ENV_LABEL[c]}: torch {', '.join(sorted(v))}" for c, v in sorted(seen.items()))


def build(main_repo: Path = ROOT, pilot_repo: Path | None = DEFAULT_PILOT_ROOT, output: Path = OUTPUT) -> dict:
    main = collect(main_repo, main_slots(), MAIN_REL, lambda s: MAIN_REL / COHORT_DIRS[s.cohort])
    groups = main_groups(main)
    overlap = overlap_rows(main)
    alpha = alpha_coverage(main_repo)
    pilot = pilot_repo if pilot_repo is not None and (pilot_repo / PILOT_REL).exists() else None
    pcoll = collect(pilot, pilot_slots(), PILOT_REL, lambda s: PILOT_REL) if pilot else None
    psum = pilot_summary(pcoll) if pcoll else None
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    minfo = repo_info(main_repo, MAIN_REL)
    pinfo = repo_info(pilot, PILOT_REL) if pilot else None

    # ---- panel 1 & 2 tables
    g_rows = [[f'<span style="color:{COLORS[g["privacy"]]}">{g["privacy"]}</span>', g["seed"], esc(g["env"]),
               f'{len(g["pairs"])}/10',
               f'{g["stats"]["in_lower"]}/10' if g["stats"] else "&mdash;",
               f'{g["stats"]["concordance"]:.2f}' if g["stats"] else "incomplete",
               num(g["stats"]["mean_delta"]) if g["stats"] else "&mdash;",
               pct(g["in_accuracy"]), pct(g["out_acc_mean"])] for g in groups]
    strip_rows = [[f'{SHORT[g["privacy"]]} s{g["seed"]}',
                   *[(f'{p[1] - p[2]:+.4f}' + ("<sup>p</sup>" if p[3] != "old" and g["kind"] == "mixed" else ""))
                     if (p := next((q for q in g["pairs"] if q[0] == t), None)) else "&mdash;" for t in TARGETS]]
                  for g in groups]
    ov_rows = [[r["privacy"], r["target"], pct(r["old_in_acc"]), pct(r["new_in_acc"]), pct(r["old_out_acc"]),
                pct(r["new_out_acc"]), num(r["old_delta"]), num(r["new_delta"]), "same" if r["same_sign"] else "<b>flipped</b>"]
               for r in overlap]
    ov_summary = "; ".join(f'{m}: targets {[r["target"] for r in overlap if r["privacy"] == m]}, '
                           f'sign agreement {sum(r["same_sign"] for r in overlap if r["privacy"] == m)}/'
                           f'{sum(r["privacy"] == m for r in overlap)}' for m in MECHS)

    # ---- panel 4
    if psum:
        a = psum["arms"]
        cov = [[f"f={f}", esc(v["in_status"]), f'{len(v["valid_targets"])}/{len(v["planned"])}',
                ", ".join(map(str, v["valid_targets"])) or "&mdash;",
                esc(", ".join(f"{t}: {pilot_status(s)}" for t, s in v["pending"].items())) or "&mdash;",
                pct(v["in_traj"].accuracy if v["in_traj"] else None),
                pct(v["in_traj"].macro_recall if v["in_traj"] else None)] for f, v in a.items()]
        agg = []
        for f in (0.0, 0.05):
            st = a[f]["full_stats"]
            agg.append([f"f={f}", "all 10 targets" if st else f'not full ({len(a[f]["valid_targets"])}/10): no full aggregate',
                        f'{st["in_lower"]}/10' if st else "&mdash;", f'{st["concordance"]:.2f}' if st else "&mdash;",
                        num(st["mean_delta"]) if st else "&mdash;"])
            cs = psum["common_stats"][f]
            agg.append([f"f={f}", f'common-target subset {psum["common"]} (n={len(psum["common"])})',
                        f'{cs["in_lower"]}/{cs["n"]}' if cs else "&mdash;",
                        f'{cs["concordance"]:.2f}' if cs else "&mdash;", num(cs["mean_delta"]) if cs else "&mdash;"])
        sc = psum["stopped_common"]
        diag = [[f"f={f}", *[num(psum["stopped_diag"][f].get(t)) for t in sc]] for f in PILOT_ARMS]
        pilot_html = f"""
  <p class="note">Separate experiment ({esc(pinfo["name"])}, branch <code>{esc(pinfo["branch"])}</code> @ <code>{esc(pinfo["head"])}</code>).
  Influence-directed matched-energy noise, seed 42 only, same &tau; as main global-dp but its own clipping path and
  noise RNG: f=0 is the pilot's own isotropic control and is <b>not</b> interchangeable with the main global-dp runs,
  so no pilot arm is plotted against main-experiment points. Not DP; no privacy claim.</p>
  <h3>Current coverage and final IN utility (round 100, server test split)</h3>
  {table(["arm", "IN", "valid OUT", "valid OUT targets", "not yet valid", "IN accuracy", "IN macro recall"], cov)}
  <h3>IN per-class recall</h3>
  <div class="panel">{panel_recall_svg(psum)}</div>
  <div class="legend">{"".join(f'<span class="key"><span class="sq" style="background:{COLORS[f]}"></span>f={f}{" (early-stopped)" if f == 0.5 else ""}</span>' for f in PILOT_ARMS if a[f]["in_traj"])}</div>
  <h3>Paired IN&minus;OUT clean loss (f=0 and f=.05 only)</h3>
  <p class="note">A full aggregate is shown only when all 10 targets are valid; otherwise only the explicit common-target subset.</p>
  {table(["arm", "target set", "IN lower", "paired concordance", "mean IN&minus;OUT"], agg)}
  <h3>Early-stopped f=.5: separate diagnostic on common targets {sc} only</h3>
  <p class="note">Per-target IN&minus;OUT clean loss for the targets every arm has. Two targets: not a headline, not
  rankable against 10-target results.</p>
  {table(["arm", *[f"target {t}" for t in sc]], diag) if sc else '<p class="warn">No common targets yet.</p>'}"""
    else:
        pilot_html = '<p class="warn">Pilot root not found; pass <code>--pilot-root</code>. Panel omitted.</p>'

    data = dict(
        generated=now, main_repo=minfo, pilot_repo=pinfo,
        coverage={s.shard + f" [{s.cohort}]": v for s, v in main.status.items()},
        pilot_coverage={s.shard: v for s, v in pcoll.status.items()} if pcoll else None,
        alpha_pilot=alpha,
        groups=[dict(privacy=g["privacy"], seed=g["seed"], env=g["env"], n_pairs=len(g["pairs"]),
                     in_accuracy=g["in_accuracy"], out_accuracy_mean=g["out_acc_mean"],
                     stats=g["stats"], pairs=[dict(target=p[0], in_clean=p[1], out_clean=p[2], out_cohort=p[3])
                                              for p in g["pairs"]]) for g in groups],
        overlap=overlap,
        pilot=None if not psum else dict(
            arms={str(f): dict(valid_targets=v["valid_targets"], in_accuracy=v["in_traj"].accuracy if v["in_traj"] else None,
                               in_macro_recall=v["in_traj"].macro_recall if v["in_traj"] else None,
                               in_recall=v["in_traj"].recall if v["in_traj"] else None, full_stats=v["full_stats"],
                               pairs=[dict(target=p[0], in_clean=p[1], out_clean=p[2]) for p in v["pairs"]])
                  for f, v in psum["arms"].items()},
            common_targets=psum["common"], common_stats={str(k): v for k, v in psum["common_stats"].items()},
            stopped_common=psum["stopped_common"],
            stopped_diag={str(k): v for k, v in psum["stopped_diag"].items()}))
    blob = base64.b64encode(json.dumps(data, indent=1, allow_nan=False, default=str).encode()).decode()

    legend = (f'<span class="key"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="#555"/></svg>seeds 42/43: old runtime, IN &amp; OUT</span>'
              f'<span class="key"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#fff" stroke="#555" stroke-width="2"/></svg>seed 44 MIXED: old IN + OUT 0&ndash;3, pinned OUT 4&ndash;9</span>'
              f'<span class="key"><svg width="14" height="14"><rect x="3" y="3" width="8" height="8" fill="#fff" stroke="#222" stroke-width="2" transform="rotate(45 7 7)"/></svg>vanilla seed 42: PINNED runtime only (not runtime-matched to old)</span>'
              f'<span class="key"><span class="sq" style="background:{COLORS["global-dp"]}"></span>global-dp</span>'
              f'<span class="key"><span class="sq" style="background:{COLORS["metric-privacy"]}"></span>metric-privacy</span>')

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PROVISIONAL &mdash; EuroSAT AUC-frontier progress</title>
<style>
 body {{ font-family: Arial, Helvetica, sans-serif; max-width: 1180px; margin: 0 auto; padding: 24px; color: #111; }}
 h1 {{ font-size: 25px; margin: 0 0 6px; }} h2 {{ font-size: 19px; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 32px; }}
 h3 {{ font-size: 15px; margin: 18px 0 6px; }}
 .banner {{ background: #fff3cd; border: 2px solid #c79a00; padding: 10px 14px; font-weight: bold; margin-bottom: 10px; }}
 .note, .meta {{ color: #444; font-size: 13px; line-height: 1.45; }} .warn {{ color: #a00; }}
 .panel {{ border: 1px solid #ccc; padding: 10px; max-width: 760px; }}
 svg {{ width: 100%; height: auto; display: block; }}
 table {{ border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; margin: 6px 0; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: 4px 7px; text-align: left; vertical-align: top; }}
 thead th {{ border-bottom: 2px solid #888; }}
 .legend svg {{ width: 14px; height: 14px; display: inline; }}
 .legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; margin: 6px 0; }}
 .key {{ display: inline-flex; align-items: center; gap: 5px; }} .sq {{ width: 11px; height: 11px; display: inline-block; }}
 ul.lim li {{ font-size: 13px; line-height: 1.45; }}
</style></head><body>
<div class="banner">PROVISIONAL progress view &mdash; experiments not declared finished; no final conclusions.</div>
<h1>EuroSAT alpha&nbsp;.3 AUC frontier &mdash; progress snapshot</h1>
<p class="meta">Generated {esc(now)} &middot; main: {esc(minfo["name"])} branch <code>{esc(minfo["branch"])}</code> @ <code>{esc(minfo["head"])}</code>
(latest results commit {esc(minfo["results_commit"])})
{f'&middot; pilot: {esc(pinfo["name"])} <code>{esc(pinfo["branch"])}</code> @ <code>{esc(pinfo["head"])}</code> (latest results commit {esc(pinfo["results_commit"])})' if pinfo else ''}
&middot; noise ratio {RATIO}, 48 clients, targets 0&ndash;9, round {ROUNDS} only &middot;
<a download="eurosat_frontier_progress.json" href="data:application/json;base64,{blob}">download all plotted values (JSON)</a></p>

<h2>Coverage</h2>
<p class="note">Counts of explicitly expected trajectories that pass validation (all JSONs committed; rounds 1&ndash;100 present once;
finite losses/accuracies; manifest identity matches). Main expected 88 = 54 old + 22 pinned + 12 mixed.</p>
{table(["group", "source", "expected", "valid", "not valid / notes"], coverage_rows(main, alpha, pcoll))}
<p class="note">Recorded runtime per cohort (run metadata): {esc(runtimes(main))}.</p>

<h2>1. Per-seed paired concordance vs final IN accuracy (main)</h2>
<p class="note">One point per mechanism &times; seed; only groups with all 10 targets valid are plotted. Fixed direction: lower
clean-shadow loss = IN. One noise level (ratio {RATIO}), so there is no noise frontier to connect. Seeds are not folded or
pooled; no intervals with fewer than 5 seeds. Values below .5 indicate the fixed direction is reversed on those
targets; they are not evidence of privacy.</p>
<div class="panel">{panel_concordance(groups)}</div>
<div class="legend">{legend}</div>
{table(["mechanism", "seed", "IN/OUT runtime pairing", "valid pairs", "IN lower", "paired concordance", "mean IN&minus;OUT clean", "IN acc", "mean OUT acc"], g_rows)}

<h2>2. Client-level IN&minus;OUT clean loss (main)</h2>
<p class="note">Each dot is one target client within one seed (same-seed IN vs its own leave-one-out OUT); no cross-client
pooled AUC. Seed-44 hollow dots / <sup>p</sup>: OUT from the pinned runtime, paired with the old-runtime IN.</p>
<div class="panel">{panel_strip(groups)}</div>
{table(["group", *[f"t{t}" for t in TARGETS]], strip_rows)}

<h2>3. Seed-42 old vs pinned overlap (sensitivity diagnostic)</h2>
<p class="note">Matched targets where both runtimes have IN and OUT: {esc(ov_summary)}. Hollow = sign of IN&minus;OUT differs
between runtimes (labeled, with |old| &gt; 0.1 points; all values in the table). Two realizations per target; this is not an equivalence test and does not validate or invalidate either runtime.</p>
<div class="panel">{panel_overlap_svg(overlap)}</div>
{table(["mechanism", "target", "old IN acc", "pinned IN acc", "old OUT acc", "pinned OUT acc", "old IN&minus;OUT", "pinned IN&minus;OUT", "sign"], ov_rows)}

<h2>4. Influence-noise pilot (separate experiment)</h2>
{pilot_html}

<h2>Limitations</h2>
<ul class="lim">
<li>Old-cohort runs record torch in run metadata but no pip freeze or full runtime capture; exact old environments are
unknown. That is a reproducibility limitation, not grounds to discard them.</li>
<li>Seed 44 pairs an old-runtime IN with pinned-runtime OUTs for targets 4&ndash;9. It is labeled, not assumed equivalent; no
formal statistical equivalence between runtimes is established (see panel 3).</li>
<li>Vanilla has only seed 42, pinned runtime only; it is a reference point, not a runtime-matched baseline.</li>
<li>2&ndash;3 seeds &times; 10 correlated targets: descriptive only; no confidence intervals, no privacy certification, and attack
metrics near or below .5 are not proof of privacy.</li>
<li>Pilot: seed 42 only, own RNG/clipping; the utility screen used the same server test split that is reported here.</li>
</ul>
<p class="meta">Generated by <code>reports/build_eurosat_frontier_progress.py</code>; rerun to refresh. Failed-attempt metadata under
<code>failures/</code> is not counted as trajectories.</p>
</body></html>
"""
    output.write_text(doc)
    return data


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT,
                        help="Influence-pilot worktree root (default: sibling metricdp-pytorch-influence-noise-pilot)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    data = build(ROOT, args.pilot_root.resolve(), args.output)
    valid = sum(v == "valid" for v in data["coverage"].values())
    print(f"wrote {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output.name}; "
          f"main valid {valid}/{len(data['coverage'])}")
    for g in data["groups"]:
        s = g["stats"]
        print(f"  {g['privacy']:15s} s{g['seed']} pairs={g['n_pairs']:2d} "
              f"conc={s['concordance'] if s else None} IN acc={g['in_accuracy']}")
    if data["pilot_coverage"]:
        print("  pilot valid", sum(v == "valid" for v in data["pilot_coverage"].values()), "/", len(data["pilot_coverage"]))


if __name__ == "__main__":
    main()
