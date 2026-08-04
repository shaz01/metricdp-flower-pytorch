"""Run the CIA contest matrix twice with one seed and compare exact results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cia.attack_runner import run_attack
from experiments.cia.datasets.partitions import PartitionViewDataModule, in_remove
from experiments.cia.result import CiaResult
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.alzheimer import (
    create_data_module as create_alzheimer_data_module,
)
from experiments.reproduce.matrix import Hyperparams, Matrix
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "cia" / "check_determinism"

NUM_CLIENTS = 4
TARGET_PARTITION_ID = 0
SHADOW_FRACTION = 0.10
NOISE_STD_FRACTION = 0.10

SEED = 42
ROUNDS = 20
CHECKPOINT_ROUNDS = (1, ROUNDS)
RUN_PREFIXES = ("check-determinism-run-1", "check-determinism-run-2")


def create_data_module(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Build this script's IN-remove participant view over Alzheimer MRI."""
    return in_remove(
        create_alzheimer_data_module(config),
        canonical_num_partitions=int(config["num-clients"]),
        target_partition_id=TARGET_PARTITION_ID,
    )


MATRIX = Matrix(
    partitions=("homogeneous",),
    privacy_modes=("vanilla", "global-dp", "metric-privacy"),
    aggregations=("fedavg",),
    seeds=(SEED,),
    noise_multipliers=(0.01,),
    hyperparams=Hyperparams(
        clipping_norm=5.0,
        rounds=ROUNDS,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        initialization_epochs=20,
    ),
    data_module="experiments.cia.scripts.check_determinism:create_data_module",
    model_module="experiments.reproduce.paper_cnn:create_model",
)


def _clean_shadow_dataset(combo: Any) -> Any:
    return clean_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
    )


def _noisy_shadow_dataset(combo: Any) -> Any:
    return noisy_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
        std_fraction=NOISE_STD_FRACTION,
    )


def _run_once(name_prefix: str) -> list[CiaResult]:
    combos = MATRIX.list_combos(name_prefix=name_prefix, num_clients=NUM_CLIENTS)
    return run_attack(
        combos=combos,
        output_dir=OUTPUT_DIR,
        log_path=OUTPUT_DIR / "progress.log",
        max_parallel_clients=4,
        force=False,
        start_message=f"Determinism {name_prefix}: {len(combos)} combinations",
        clean_data_module_factory=_clean_shadow_dataset,
        noisy_data_module_factory=_noisy_shadow_dataset,
        device=resolve_device(),
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name=f"{name_prefix}.json",
    )


def _canonical_results(results: list[CiaResult]) -> list[dict[str, Any]]:
    """Remove the deliberately different run name and sort for exact comparison."""
    canonical = []
    for result in results:
        values = asdict(result)
        values.pop("run_name")
        canonical.append(values)
    return sorted(
        canonical,
        key=lambda value: (
            value["privacy"],
            value["aggregation"],
            value["server_round"],
        ),
    )


def _normalize_artifact_json(value: Any) -> Any:
    """Remove only intentional run labels and random temporary-venv paths."""
    if isinstance(value, str):
        for prefix in RUN_PREFIXES:
            value = value.replace(prefix, "check-determinism-run-X")
        return value
    if isinstance(value, list):
        return [_normalize_artifact_json(item) for item in value]
    if isinstance(value, dict):
        normalized = {
            key: _normalize_artifact_json(item) for key, item in value.items()
        }
        metadata = normalized.get("metadata")
        if isinstance(metadata, dict):
            # The runner records its fresh, randomly named temporary venv.
            metadata.pop("python_executable", None)
        return normalized
    return value


def _compare_artifacts() -> dict[str, Any]:
    """Compare per-round JSON, detailed evaluations, and prediction arrays."""
    json_equal = True
    evaluation_equal = True
    predictions_equal = True
    checked_pairs = 0
    first_combos = MATRIX.list_combos(
        name_prefix=RUN_PREFIXES[0], num_clients=NUM_CLIENTS
    )
    second_combos = MATRIX.list_combos(
        name_prefix=RUN_PREFIXES[1], num_clients=NUM_CLIENTS
    )
    for first, second in zip(first_combos, second_combos, strict=True):
        first_base = OUTPUT_DIR / first.run_name()
        second_base = OUTPUT_DIR / second.run_name()
        for suffix, report_key in (
            (".json", "json"),
            (".evaluation.json", "evaluation"),
        ):
            left = _normalize_artifact_json(
                json.loads(first_base.with_suffix(suffix).read_text(encoding="utf-8"))
            )
            right = _normalize_artifact_json(
                json.loads(second_base.with_suffix(suffix).read_text(encoding="utf-8"))
            )
            if report_key == "json":
                json_equal &= left == right
            else:
                evaluation_equal &= left == right

        with (
            np.load(first_base.with_suffix(".predictions.npz")) as left_arrays,
            np.load(second_base.with_suffix(".predictions.npz")) as right_arrays,
        ):
            predictions_equal &= left_arrays.files == right_arrays.files and all(
                np.array_equal(left_arrays[key], right_arrays[key], equal_nan=True)
                for key in left_arrays.files
            )
        checked_pairs += 1

    return {
        "training_json_equal": json_equal,
        "evaluation_json_equal": evaluation_equal,
        "prediction_arrays_equal": predictions_equal,
        "checked_configuration_pairs": checked_pairs,
        "ignored_differences": [
            "intentional run-name prefix",
            "metadata.python_executable random temporary-venv path",
        ],
    }


def _write_comparison(
    first: list[CiaResult], second: list[CiaResult]
) -> dict[str, Any]:
    combo_count = len(MATRIX.list_combos(name_prefix="x", num_clients=NUM_CLIENTS))
    expected_count = combo_count * len(CHECKPOINT_ROUNDS)
    first_values = _canonical_results(first)
    second_values = _canonical_results(second)
    artifact_comparison = _compare_artifacts()
    artifacts_equal = all(
        artifact_comparison[key]
        for key in (
            "training_json_equal",
            "evaluation_json_equal",
            "prediction_arrays_equal",
        )
    )
    report = {
        "deterministic": first_values == second_values and artifacts_equal,
        "comparison": (
            "exact CIA results plus canonical training/evaluation JSON and exact "
            "prediction arrays"
        ),
        "seed": SEED,
        "expected_results_per_run": expected_count,
        "actual_results": {
            RUN_PREFIXES[0]: len(first_values),
            RUN_PREFIXES[1]: len(second_values),
        },
        "run_prefixes": list(RUN_PREFIXES),
        "artifact_comparison": artifact_comparison,
        "first": first_values,
        "second": second_values,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Determinism comparison: {path}")
    print(f"Exact match: {report['deterministic']}")

    if len(first_values) != expected_count or len(second_values) != expected_count:
        raise RuntimeError(
            "The determinism check is incomplete: "
            f"expected {expected_count} evaluated checkpoints per run."
        )
    if not report["deterministic"]:
        raise RuntimeError("The two same-seed CIA runs produced different results.")
    return report


def main() -> None:
    first = _run_once(RUN_PREFIXES[0])
    second = _run_once(RUN_PREFIXES[1])
    _write_comparison(first, second)


if __name__ == "__main__":
    main()
