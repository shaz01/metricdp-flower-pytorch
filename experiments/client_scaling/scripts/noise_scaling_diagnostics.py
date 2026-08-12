"""Check whether a fixed noise ratio really holds the injected noise constant.

The noise-ratio system in ``PLAN.md`` defines ``ratio = noise_multiplier /
num_clients``.  It is built on Flower's global-DP noise rule,
``compute_stdv(nm, clipping_norm, n) = nm * clipping_norm / n``: holding the
ratio fixed makes that standard deviation ``ratio * clipping_norm``, i.e.
independent of the client count, which is what makes runs at different ``n``
comparable.

Metric-privacy does not obey that rule.  It first rescales the multiplier by
the measured maximum pairwise client-model distance ``d``
(``metricdp_strategy.py``: ``calibrated_multiplier = noise_multiplier /
distance``), so its standard deviation is ``nm * clipping_norm / (d * n)`` --
smaller by a factor ``d``, and ``d`` is data-dependent and changes with client
count and partitioning.  A fixed ratio therefore pins global-DP's noise but not
metric-privacy's, which is a candidate explanation for metric-privacy's
accuracy advantage shrinking (and, on the 4-class runs, reversing) as ``n``
grows.

This script quantifies that from the diagnostics already recorded in every run
JSON -- ``metric-dp-distance``, ``metric-dp-noise-stdv``,
``global-dp-noise-stdv``, ``dp-noise-to-signal-ratio`` -- rather than
re-running anything.  For each run it reports the configured ratio, the noise
standard deviation a global-DP run would use at that ratio, the standard
deviation actually injected, and, for metric-privacy, the *effective* ratio
``ratio / d`` -- the ratio a global-DP run would need in order to inject the
same noise.

Usage:
    uv run python -m experiments.client_scaling.scripts.noise_scaling_diagnostics
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOTS = (
    PROJECT_ROOT / "results" / "cia" / "cifar10_remove",
    PROJECT_ROOT / "results" / "client_scaling" / "cifar10_homogeneous",
    PROJECT_ROOT / "results" / "planned_runs" / "cifar",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "client_scaling" / "noise_scaling_diagnostics.json"

# Sidecar artifacts that live beside run JSONs but are not run JSONs.
SKIP_SUFFIXES = (".evaluation.json",)
SKIP_NAMES = frozenset(
    {"cia.json", "colab_run.json", "chunk_manifest.json", "comparison.json"}
)


def is_run_json(path: Path) -> bool:
    """A run JSON is the one carrying both metadata and per-round train metrics."""
    if path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES):
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(document, dict)
        and "metadata" in document
        and "train_metrics" in document
    )


def find_run_jsons(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if is_run_json(path))


def _round_values(train_metrics: dict[str, Any], key: str) -> list[float]:
    """Collect one diagnostic across rounds, skipping rounds that lack it."""
    values = []
    for round_metrics in train_metrics.values():
        value = round_metrics.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _repo_relative(path: Path) -> Path:
    """Path relative to the repo when it lives there, else the path itself.

    Result roots are normally inside the repo, but the scanner is also pointed
    at directories outside it (tmp dirs in tests, an external results mount),
    where ``relative_to`` would raise.
    """
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def source_label(path: Path) -> str:
    """Short provenance label: which experiment's results a run came from."""
    parts = _repo_relative(path).parts
    return "/".join(parts[1:3]) if len(parts) > 2 else parts[-1]


def summarize_run(path: Path) -> dict[str, Any]:
    """Extract one run's configured and realized noise scale."""
    document = json.loads(path.read_text(encoding="utf-8"))
    metadata = document["metadata"]
    train_metrics = document["train_metrics"]

    num_clients = int(metadata["num_clients"])
    noise_multiplier = float(metadata["noise_multiplier"])
    clipping_norm = float(metadata["clipping_norm"])
    privacy = str(metadata["privacy"])

    ratio = noise_multiplier / num_clients
    # Flower's compute_stdv: the noise a global-DP run injects at this config.
    global_dp_stdv = noise_multiplier * clipping_norm / num_clients

    distances = _round_values(train_metrics, "metric-dp-distance")
    observed_stdv = _mean(
        _round_values(train_metrics, "metric-dp-noise-stdv")
        or _round_values(train_metrics, "global-dp-noise-stdv")
    )
    mean_distance = _mean(distances)

    entry: dict[str, Any] = {
        "path": str(_repo_relative(path)),
        "source": source_label(path),
        "privacy": privacy,
        "partition_mode": metadata.get("partition_mode"),
        "num_clients": num_clients,
        "noise_multiplier": noise_multiplier,
        "clipping_norm": clipping_norm,
        "ratio": ratio,
        "global_dp_stdv_at_this_ratio": global_dp_stdv,
        "observed_noise_stdv": observed_stdv,
        "mean_max_pairwise_distance": mean_distance,
        "effective_ratio": ratio / mean_distance if mean_distance else ratio,
        "mean_noise_to_signal": _mean(
            _round_values(train_metrics, "dp-noise-to-signal-ratio")
        ),
        "mean_signal_update_norm": _mean(
            _round_values(train_metrics, "dp-signal-update-norm")
        ),
    }
    if privacy == "vanilla":
        # Vanilla injects nothing; its recorded multiplier is a naming artifact.
        entry["ratio"] = None
        entry["effective_ratio"] = None
        entry["global_dp_stdv_at_this_ratio"] = None
    return entry


def build_summary(roots: list[Path]) -> list[dict[str, Any]]:
    summary = []
    for root in roots:
        if not root.exists():
            print(f"WARNING: {root} does not exist -- skipping.")
            continue
        for path in find_run_jsons(root):
            summary.append(summarize_run(path))
    return summary


def aggregate_by_config(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average repeated runs of one config (seeds, IN/OUT views) into one row."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for entry in summary:
        if entry["privacy"] == "vanilla":
            continue
        key = (
            entry["source"],
            entry["privacy"],
            entry["partition_mode"],
            entry["num_clients"],
            round(entry["ratio"], 6),
        )
        groups[key].append(entry)

    aggregated = []
    for (source, privacy, partition, clients, ratio), entries in groups.items():
        distances = [
            entry["mean_max_pairwise_distance"]
            for entry in entries
            if entry["mean_max_pairwise_distance"] is not None
        ]
        aggregated.append(
            {
                "source": source,
                "privacy": privacy,
                "partition_mode": partition,
                "num_clients": clients,
                "ratio": ratio,
                "runs": len(entries),
                "global_dp_stdv_at_this_ratio": entries[0][
                    "global_dp_stdv_at_this_ratio"
                ],
                "mean_observed_noise_stdv": _mean(
                    [
                        entry["observed_noise_stdv"]
                        for entry in entries
                        if entry["observed_noise_stdv"] is not None
                    ]
                ),
                "mean_max_pairwise_distance": _mean(distances),
                "mean_effective_ratio": _mean(
                    [entry["effective_ratio"] for entry in entries]
                ),
            }
        )
    aggregated.sort(
        key=lambda e: (e["privacy"], e["source"], e["num_clients"], e["ratio"])
    )
    return aggregated


def format_table(summary: list[dict[str, Any]]) -> str:
    """Render the per-config view (vanilla omitted: it injects no noise)."""
    lines = [
        "Configured vs. realized noise scale, averaged over repeats of each config",
        "(global-DP stdv = ratio x clipping_norm, constant by construction;",
        " metric-privacy divides by the measured distance d, so it is not)",
        "",
        f"{'privacy':15s} {'source':28s} {'part':11s} {'n':>4} {'ratio':>9} "
        f"{'gDP stdv':>9} {'obs stdv':>9} {'d':>7} {'eff.ratio':>10}",
    ]
    for entry in aggregate_by_config(summary):
        distance = entry["mean_max_pairwise_distance"]
        observed = entry["mean_observed_noise_stdv"]
        lines.append(
            f"{entry['privacy']:15s} {entry['source']:28s} "
            f"{str(entry['partition_mode']):11s} {entry['num_clients']:4d} "
            f"{entry['ratio']:9.5f} "
            f"{entry['global_dp_stdv_at_this_ratio']:9.5f} "
            f"{(observed if observed is not None else float('nan')):9.5f} "
            f"{(distance if distance is not None else float('nan')):7.3f} "
            f"{entry['mean_effective_ratio']:10.5f}"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Result directories to scan recursively for run JSONs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = build_summary([root.resolve() for root in args.roots])
    if not summary:
        raise SystemExit("No run JSONs found under the given roots.")
    print(format_table(summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"runs": summary, "by_config": aggregate_by_config(summary)}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
