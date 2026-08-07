"""Run the unblocked experiment groups specified in ``PLAN.md``.

The replacement-adjacency client-scaling experiments are intentionally absent:
their dataset-size and noise-ratio choices are still blocked in the plan.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from experiments.cia.attack_runner import run_attack
from experiments.cia.datasets.paper import (
    PAPER_CIA_CLIENT_COUNTS,
    PAPER_CIA_NUM_CLIENTS,
    PAPER_CIA_TARGET_PARTITION_ID,
    PaperCiaDataModule,
)
from experiments.cia.datasets.partitions import (
    PartitionViewDataModule,
    in_remove,
    out_remove,
)
from experiments.cia.result import CiaResult
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.alzheimer import create_data_module as alzheimer
from experiments.reproduce.dataset.cifar4 import Cifar4DataModule, Cifar4Dataset
from experiments.reproduce.dataset.fashion_mnist import (
    FashionMNISTDataModule,
    FashionMNISTDataset,
)
from experiments.reproduce.matrix import Hyperparams, Matrix
from experiments.reproduce.matrix.run_combo import is_complete
from experiments.reproduce.matrix.run_matrix import run_combos
from metricdp_pytorch.utils.data import labels_from_records, make_client_loaders
from metricdp_pytorch.utils.device import resolve_device
from metricdp_pytorch.utils.split_data import partition_by_class_counts

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "planned_runs"

SEEDS = (42, 43, 44)
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
ROUNDS = 20
CHECKPOINT_ROUNDS = tuple(range(1, ROUNDS + 1))
SHADOW_FRACTION = 0.10
NOISE_STD_FRACTION = 0.20
NOISE_MULTIPLIER = 0.01
TARGET_PARTITION_ID = PAPER_CIA_TARGET_PARTITION_ID

HYPERPARAMS = Hyperparams(
    clipping_norm=5.0,
    rounds=ROUNDS,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    initialization_epochs=20,
)


class FashionTable9DataModule(FashionMNISTDataModule):
    """Four-class Fashion-MNIST with the paper CIA's exact client counts."""

    def client_loaders(
        self,
        partition_id: int,
        *,
        num_partitions: int,
        partition_mode: str,
        batch_size: int,
        seed: int,
        partition_profile: str = "auto",
        client_weights: Sequence[float] | None = None,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        del partition_mode, partition_profile
        if num_partitions != PAPER_CIA_NUM_CLIENTS:
            raise ValueError("The Table-9 transfer distribution requires three clients.")
        if client_weights is not None:
            raise ValueError("The Table-9 transfer distribution does not use client weights.")
        if not 0 <= partition_id < PAPER_CIA_NUM_CLIENTS:
            raise ValueError("partition_id must identify one of the three clients.")

        split = self.dataset["train"]
        labels = labels_from_records(split)
        partitions = partition_by_class_counts(
            labels, PAPER_CIA_CLIENT_COUNTS, seed=seed
        )
        return make_client_loaders(
            FashionMNISTDataset(split),
            labels,
            partitions[partition_id],
            batch_size=batch_size,
            seed=seed + partition_id,
            train_fraction=self.train_fraction,
            max_samples=max_samples,
        )


class CifarTable9DataModule(Cifar4DataModule):
    """Four-class CIFAR-10 with the paper CIA's exact client counts."""

    def client_loaders(
        self,
        partition_id: int,
        *,
        num_partitions: int,
        partition_mode: str,
        batch_size: int,
        seed: int,
        partition_profile: str = "auto",
        client_weights: Sequence[float] | None = None,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        del partition_mode, partition_profile
        if num_partitions != PAPER_CIA_NUM_CLIENTS:
            raise ValueError("The Table-9 transfer distribution requires three clients.")
        if client_weights is not None:
            raise ValueError("The Table-9 transfer distribution does not use client weights.")
        if not 0 <= partition_id < PAPER_CIA_NUM_CLIENTS:
            raise ValueError("partition_id must identify one of the three clients.")

        split = self.dataset["train"]
        labels = labels_from_records(split)
        partitions = partition_by_class_counts(
            labels, PAPER_CIA_CLIENT_COUNTS, seed=seed
        )
        return make_client_loaders(
            Cifar4Dataset(split),
            labels,
            partitions[partition_id],
            batch_size=batch_size,
            seed=seed + partition_id,
            train_fraction=self.train_fraction,
            max_samples=max_samples,
        )


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_alzheimer_in(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the canonical three-client paper CIA IN-remove view."""
    return in_remove(
        PaperCiaDataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=PAPER_CIA_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_alzheimer_out(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the canonical three-client paper CIA OUT-remove view."""
    return out_remove(
        PaperCiaDataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=PAPER_CIA_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_fashion_in(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the three-client Fashion-MNIST IN-remove transfer view."""
    return in_remove(
        FashionTable9DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=PAPER_CIA_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_fashion_out(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the three-client Fashion-MNIST OUT-remove transfer view."""
    return out_remove(
        FashionTable9DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=PAPER_CIA_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_cifar_in(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the three-client CIFAR-10 IN-remove transfer view."""
    return in_remove(
        CifarTable9DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=PAPER_CIA_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_cifar_out(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the three-client CIFAR-10 OUT-remove transfer view."""
    return out_remove(
        CifarTable9DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=PAPER_CIA_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def _matrix(*, data_module: str, model_module: str) -> Matrix:
    return Matrix(
        partitions=("non-iid",),
        privacy_modes=PRIVACY_MODES,
        aggregations=("fedavg",),
        seeds=SEEDS,
        noise_multipliers=(NOISE_MULTIPLIER,),
        hyperparams=HYPERPARAMS,
        data_module=data_module,
        model_module=model_module,
    )


REPRODUCTION_MATRIX = Matrix(
    partitions=("homogeneous",),
    privacy_modes=PRIVACY_MODES,
    aggregations=("fedavg",),
    seeds=SEEDS,
    noise_multipliers=(NOISE_MULTIPLIER,),
    hyperparams=HYPERPARAMS,
    data_module="experiments.reproduce.dataset.alzheimer:create_data_module",
    model_module="experiments.reproduce.paper_cnn:create_model",
)

CIA_GROUPS = (
    (
        "alzheimer-in-remove",
        PAPER_CIA_NUM_CLIENTS,
        _matrix(
            data_module="experiments.cia.scripts.planned_runs:create_alzheimer_in",
            model_module="experiments.reproduce.paper_cnn:create_model",
        ),
    ),
    (
        "alzheimer-out-remove",
        PAPER_CIA_NUM_CLIENTS - 1,
        _matrix(
            data_module="experiments.cia.scripts.planned_runs:create_alzheimer_out",
            model_module="experiments.reproduce.paper_cnn:create_model",
        ),
    ),
    (
        "fashion-in-remove",
        PAPER_CIA_NUM_CLIENTS,
        _matrix(
            data_module="experiments.cia.scripts.planned_runs:create_fashion_in",
            model_module="experiments.reproduce.fashion_mnist_cnn:create_model",
        ),
    ),
    (
        "fashion-out-remove",
        PAPER_CIA_NUM_CLIENTS - 1,
        _matrix(
            data_module="experiments.cia.scripts.planned_runs:create_fashion_out",
            model_module="experiments.reproduce.fashion_mnist_cnn:create_model",
        ),
    ),
    (
        "cifar-in-remove",
        PAPER_CIA_NUM_CLIENTS,
        _matrix(
            data_module="experiments.cia.scripts.planned_runs:create_cifar_in",
            model_module="experiments.reproduce.cifar4_cnn:create_model",
        ),
    ),
    (
        "cifar-out-remove",
        PAPER_CIA_NUM_CLIENTS - 1,
        _matrix(
            data_module="experiments.cia.scripts.planned_runs:create_cifar_out",
            model_module="experiments.reproduce.cifar4_cnn:create_model",
        ),
    ),
)


def _clean_shadow(combo: Any) -> Any:
    return clean_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
    )


def _noisy_shadow(combo: Any) -> Any:
    return noisy_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
        std_fraction=NOISE_STD_FRACTION,
    )


def _expected_cia_keys(combos: Sequence[Any]) -> set[tuple[str, int]]:
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
        "--suite",
        choices=("all", "reproduction", "alzheimer", "fashion", "cifar"),
        default="all",
        help="independent subset to run in a dedicated Colab job",
    )
    parser.add_argument(
        "--group",
        choices=tuple(name for name, _clients, _matrix in CIA_GROUPS),
        help="run exactly one CIA adjacency group (overrides --suite)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verdict: dict[str, Any] = {
        "configuration": {
            "seeds": list(SEEDS),
            "rounds": ROUNDS,
            "shadow_fraction": SHADOW_FRACTION,
            "shadow_noise_std_fraction": NOISE_STD_FRACTION,
            "noise_multiplier": NOISE_MULTIPLIER,
            "replacement_runs_included": False,
            "suite": args.suite,
        },
        "groups": {},
    }
    all_complete = True
    if args.group is None and args.suite in ("all", "reproduction"):
        reproduction_dir = output_dir / "original_reproduction"
        reproduction_combos = REPRODUCTION_MATRIX.list_combos(
            name_prefix="original-reproduction", num_clients=4
        )
        run_combos(
            reproduction_combos,
            output_dir=reproduction_dir,
            max_parallel_clients=4,
        )
        reproduction_complete = all(
            is_complete(combo.result_path(reproduction_dir), expected_rounds=ROUNDS)
            for combo in reproduction_combos
        )
        verdict["groups"]["original-reproduction"] = {
            "expected_trajectories": len(reproduction_combos),
            "complete": reproduction_complete,
        }
        all_complete &= reproduction_complete

    for name_prefix, active_clients, matrix in CIA_GROUPS:
        if args.group is not None and name_prefix != args.group:
            continue
        dataset_name = name_prefix.split("-", 1)[0]
        if args.group is None and args.suite == "reproduction":
            continue
        if (
            args.group is None
            and args.suite in ("alzheimer", "fashion", "cifar")
            and dataset_name != args.suite
        ):
            continue
        group_dir = output_dir / name_prefix
        combos = matrix.list_combos(
            name_prefix=name_prefix, num_clients=active_clients
        )
        results = run_attack(
            combos=combos,
            output_dir=group_dir,
            log_path=group_dir / "progress.log",
            max_parallel_clients=4,
            force=False,
            start_message=f"Starting {name_prefix}: {len(combos)} trajectories",
            clean_data_module_factory=_clean_shadow,
            noisy_data_module_factory=_noisy_shadow,
            device=resolve_device(),
            checkpoint_rounds=CHECKPOINT_ROUNDS,
            report_name="cia.json",
        )
        missing = sorted(_expected_cia_keys(combos) - _actual_cia_keys(results))
        group_complete = not missing
        all_complete &= group_complete
        verdict["groups"][name_prefix] = {
            "active_clients": active_clients,
            "canonical_clients": PAPER_CIA_NUM_CLIENTS,
            "expected_trajectories": len(combos),
            "expected_evaluations": len(combos) * len(CHECKPOINT_ROUNDS),
            "actual_evaluations": len(_actual_cia_keys(results)),
            "missing": [list(key) for key in missing],
            "complete": group_complete,
        }

    verdict["complete"] = all_complete
    (output_dir / "planned_runs_manifest.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    if not all_complete:
        raise RuntimeError("One or more unblocked PLAN.md experiment groups failed.")


if __name__ == "__main__":
    main()
