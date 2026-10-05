"""Round-matched per-client IN/OUT score from frontier ``measurements.json`` files.

For each (seed, target): the fraction of rounds where the target's clean-shadow loss
under the shared IN trajectory is lower than under that target's OUT trajectory
(ties count 0.5), then the mean over targets per seed. 0.5 = no signal. Works on any
root holding eurosat_frontier or dataset_vs_model Stage B/C trajectory folders
(``manifest.json`` + ``measurements.json``), nested at any depth.

    python -m results.cia_frontier.per_client_score <root>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

# Manifest fields that must agree between an IN and its OUT trajectories.
_SETTING = ("dataset", "partition", "alpha", "privacy", "noise_ratio", "clients", "rounds",
            "partition_seed", "train_subsample")


def score_rows(in_rows, out_rows):
    """Per-target fraction of matched rounds with IN clean loss < OUT clean loss, and mean."""
    def keyed(rows):
        result = {}
        for row in rows:
            key = (int(row["round"]), int(row["target"]))
            if key in result:
                raise ValueError(f"Duplicate measurement {key}")
            value = float(row["clean_loss"])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite loss at {key}")
            result[key] = value
        return result
    inside, outside = keyed(in_rows), keyed(out_rows)
    if not outside or not set(outside) <= set(inside):
        raise ValueError("Every OUT round/target needs an IN measurement")
    if {r for r, _ in inside} != {r for r, _ in outside}:
        raise ValueError("IN and OUT must cover the same rounds")
    per_target = {}
    for target in sorted({t for _, t in outside}):
        keys = [k for k in outside if k[1] == target]
        wins = sum(1.0 if inside[k] < outside[k] else 0.5 if inside[k] == outside[k] else 0.0
                   for k in keys)
        per_target[str(target)] = wins / len(keys)
    return {"per_target": per_target, "mean": sum(per_target.values()) / len(per_target),
            "matched_rounds": len({r for r, _ in outside})}


def _load(root: Path):
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("pilot"):
            continue
        if not (path.parent / "complete.json").exists():
            raise ValueError(f"Incomplete trajectory: {path.parent}")
        shadows = path.parent / "shadows.json"
        yield (manifest, json.loads((path.parent / "measurements.json").read_text()),
               json.loads(shadows.read_text()) if shadows.exists() else None)


def score_directories(root: Path) -> list[dict]:
    """One entry per (setting, seed): per-target scores for every target with an OUT run."""
    ins, outs = {}, defaultdict(dict)
    for manifest, rows, shadows in _load(root):
        setting = tuple(json.dumps(manifest.get(k)) for k in _SETTING)
        key = (setting, manifest["seed"])
        if manifest["out_target"] is None:
            if key in ins:
                raise ValueError(f"Duplicate IN trajectory: {manifest['run_name']}")
            ins[key] = (manifest, rows, shadows)
        else:
            outs[key][manifest["out_target"]] = (rows, shadows)
    output = []
    for key, by_target in sorted(outs.items()):
        if key not in ins:
            raise ValueError(f"Missing IN trajectory for {by_target and next(iter(by_target))}")
        manifest, in_rows, in_shadows = ins[key]
        out_rows = []
        for target, (rows, shadows) in sorted(by_target.items()):
            if in_shadows is not None or shadows is not None:
                if (in_shadows or {}).get(str(target), {}).get("sha256") != \
                        (shadows or {}).get(str(target), {}).get("sha256"):
                    raise ValueError(f"IN and OUT shadow records differ for target {target}")
            out_rows += [r for r in rows if r["target"] == target]
        scored = score_rows(in_rows, out_rows)
        output.append({**{k: manifest.get(k) for k in _SETTING if manifest.get(k) is not None},
                       "seed": manifest["seed"], **scored})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_directories(args.root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
