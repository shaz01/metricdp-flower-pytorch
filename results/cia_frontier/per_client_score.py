"""Round-matched clean-shadow IN/OUT score for frontier measurement directories."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def score_rows(in_rows, out_rows):
    """Return per-target fraction of matched rounds IN loss < OUT loss and mean."""
    def keyed(rows):
        result = {}
        for row in rows:
            key = (int(row["round"]), int(row["target"]))
            if key in result:
                raise ValueError(f"Duplicate measurement {key}")
            value = float(row["clean_loss"])
            if not __import__("math").isfinite(value):
                raise ValueError(f"Nonfinite loss at {key}")
            result[key] = value
        return result
    inside, outside = keyed(in_rows), keyed(out_rows)
    if not inside or set(inside) != set(outside):
        raise ValueError("IN and OUT measurements must have identical nonempty round/target coverage")
    targets = sorted({t for _, t in inside})
    per_target = {str(t): sum(inside[r, t] < outside[r, t] for r, target in inside if target == t) /
                  sum(target == t for _, target in inside) for t in targets}
    return {"per_target": per_target, "mean": sum(per_target.values()) / len(per_target),
            "matched_rounds": len(inside) // len(targets)}


def score_directories(root: Path):
    manifests = [json.loads(p.read_text()) | {"_path": p.parent} for p in root.rglob("manifest.json")]
    ins = {}
    outs = {}
    for m in manifests:
        if m.get("pilot"): continue
        rows = json.loads((m["_path"] / "measurements.json").read_text())
        targets = m.get("targets", [])
        if m.get("out_target") is None:
            for target in targets: ins[(m["seed"], target)] = [r for r in rows if r["target"] == target]
        else: outs[(m["seed"], m["out_target"])] = rows
    output = []
    for key, out in sorted(outs.items()):
        if key not in ins: raise ValueError(f"Missing IN run for seed/target {key}")
        output.append({"seed": key[0], "target": key[1], **score_rows(ins[key], out)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_directories(args.root), indent=2, allow_nan=False))

if __name__ == "__main__": main()
