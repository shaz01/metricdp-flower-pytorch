"""Run small, resumable CIFAR-10 PLAN.md chunks on Colab GPUs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.cia.attack_runner import run_attack
from experiments.cia.datasets.partitions import (
    PartitionViewDataModule,
    in_replace,
    out_remove,
    out_replace,
)
from experiments.cia.result import CiaResult
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.cifar10 import Cifar10DataModule
from experiments.reproduce.matrix import Combo, Hyperparams, is_complete, run_combos
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "planned_runs" / "cifar_chunks"

SEEDS = (42, 43, 44)
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
ROUNDS = 20
CHECKPOINT_ROUNDS = tuple(range(1, ROUNDS + 1))
SHADOW_FRACTION = 0.10
NOISE_STD_FRACTION = 0.20
BASE_NOISE_MULTIPLIER = 0.01
TARGET_PARTITION_ID = 0

HYPERPARAMS = Hyperparams(
    clipping_norm=5.0,
    rounds=ROUNDS,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    initialization_epochs=20,
)


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_cifar_out_remove_all(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Partition all filtered CIFAR records among three canonical clients."""
    return out_remove(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=3,
        target_partition_id=TARGET_PARTITION_ID,
    )


def _replacement_ids(config: Mapping[str, Any]) -> tuple[int, int]:
    active_clients = int(config["num-clients"])
    if active_clients < 2:
        raise ValueError("Replacement adjacency requires at least two active clients.")
    canonical_clients = active_clients + 1
    return canonical_clients, canonical_clients - 1


def create_cifar_in_replace(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create an IN-replace view with one excluded replacement partition."""
    canonical_clients, replacement_id = _replacement_ids(config)
    return in_replace(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=canonical_clients,
        target_partition_id=TARGET_PARTITION_ID,
        replacement_partition_id=replacement_id,
    )


def create_cifar_out_replace(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create an OUT-replace view with the target swapped for its replacement."""
    canonical_clients, replacement_id = _replacement_ids(config)
    return out_replace(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=canonical_clients,
        target_partition_id=TARGET_PARTITION_ID,
        replacement_partition_id=replacement_id,
    )


def _clean_shadow(combo: Combo) -> Any:
    return clean_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
    )


def _noisy_shadow(combo: Combo) -> Any:
    return noisy_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
        std_fraction=NOISE_STD_FRACTION,
    )


def _data_module(task: str, adjacency: str | None) -> str:
    module = "experiments.cia.scripts.cifar_chunks"
    if task == "max-records":
        return f"{module}:create_cifar_out_remove_all"
    if adjacency == "in-replace":
        return f"{module}:create_cifar_in_replace"
    if adjacency == "out-replace":
        return f"{module}:create_cifar_out_replace"
    raise ValueError("Replacement tasks require --adjacency.")


def _name_prefix(task: str, adjacency: str | None) -> str:
    if task == "max-records":
        return "cifar-max-records-out-remove"
    return f"cifar-{task}-{adjacency}"


def build_combos(
    *,
    task: str,
    adjacency: str | None,
    active_clients: int,
    seed: int,
    privacy_modes: Sequence[str],
    noise_ratio: float | None,
) -> list[Combo]:
    """Build the one- or two-trajectory unit assigned to one Colab session."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}.")
    if not privacy_modes or any(mode not in PRIVACY_MODES for mode in privacy_modes):
        raise ValueError(f"privacy modes must come from {PRIVACY_MODES}.")
    if len(set(privacy_modes)) != len(privacy_modes):
        raise ValueError("privacy modes must not contain duplicates.")
    if task == "max-records":
        if active_clients != 2 or adjacency not in (None, "out-remove"):
            raise ValueError("max-records uses three canonical and two active clients.")
        noise_multiplier = BASE_NOISE_MULTIPLIER
        adjacency = "out-remove"
    else:
        if adjacency not in ("in-replace", "out-replace"):
            raise ValueError("replacement tasks require IN- or OUT-replace adjacency.")
        if noise_ratio is None or noise_ratio <= 0:
            if any(mode != "vanilla" for mode in privacy_modes):
                raise ValueError("non-vanilla replacement runs require a positive ratio.")
            noise_multiplier = BASE_NOISE_MULTIPLIER
        else:
            noise_multiplier = noise_ratio * active_clients

    return [
        Combo(
            name_prefix=_name_prefix(task, adjacency),
            num_clients=active_clients,
            partition="non-iid",
            privacy=privacy,
            aggregation="fedavg",
            seed=seed,
            noise_multiplier=noise_multiplier,
            hyperparams=HYPERPARAMS,
            data_module=_data_module(task, adjacency),
            model_module="experiments.reproduce.cifar10_cnn:create_model",
        )
        for privacy in privacy_modes
    ]


def _expected_cia_keys(combos: Sequence[Combo]) -> set[tuple[str, int]]:
    return {
        (combo.run_name(), round_number)
        for combo in combos
        for round_number in CHECKPOINT_ROUNDS
    }


def _actual_cia_keys(results: Sequence[CiaResult]) -> set[tuple[str, int]]:
    return {(result.run_name, result.server_round) for result in results}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("max-records", "noise-sweep", "validation", "full"),
        required=True,
    )
    parser.add_argument(
        "--adjacency",
        choices=("out-remove", "in-replace", "out-replace"),
    )
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--privacy",
        choices=PRIVACY_MODES,
        nargs="+",
        required=True,
    )
    parser.add_argument("--noise-ratio", type=float)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = build_combos(
        task=args.task,
        adjacency=args.adjacency,
        active_clients=args.clients,
        seed=args.seed,
        privacy_modes=args.privacy,
        noise_ratio=args.noise_ratio,
    )
    accuracy_only = args.task in ("noise-sweep", "validation")
    max_parallel_clients = min(args.clients, 12 if args.clients >= 48 else 8)

    if accuracy_only:
        run_combos(
            combos,
            output_dir=output_dir / "runs",
            max_parallel_clients=max_parallel_clients,
        )
        complete = all(
            is_complete(
                combo.result_path(output_dir / "runs"), expected_rounds=ROUNDS
            )
            for combo in combos
        )
        actual_evaluations = None
    else:
        results = run_attack(
            combos=combos,
            output_dir=output_dir / "runs",
            log_path=output_dir / "runs" / "progress.log",
            max_parallel_clients=max_parallel_clients,
            force=False,
            start_message=f"Starting {args.task} chunk: {len(combos)} trajectories",
            clean_data_module_factory=_clean_shadow,
            noisy_data_module_factory=_noisy_shadow,
            device=resolve_device(),
            checkpoint_rounds=CHECKPOINT_ROUNDS,
            report_name="cia.json",
        )
        actual = _actual_cia_keys(results)
        complete = _expected_cia_keys(combos) <= actual
        actual_evaluations = len(actual)

    manifest = {
        "task": args.task,
        "adjacency": args.adjacency,
        "active_clients": args.clients,
        "canonical_clients": args.clients + 1,
        "seed": args.seed,
        "privacy": list(args.privacy),
        "noise_ratio": args.noise_ratio,
        "noise_multiplier": combos[0].noise_multiplier,
        "accuracy_only": accuracy_only,
        "rounds": ROUNDS,
        "checkpoint_rounds": [] if accuracy_only else list(CHECKPOINT_ROUNDS),
        "expected_trajectories": len(combos),
        "expected_evaluations": None if accuracy_only else len(combos) * ROUNDS,
        "actual_evaluations": actual_evaluations,
        "complete": complete,
    }
    (output_dir / "chunk_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not complete:
        raise RuntimeError("CIFAR chunk did not produce all expected artifacts.")


if __name__ == "__main__":
    main()
