"""Deterministic task roles, label-stress client partitions and image access for the stacked-head experiment.

A task is a 4-class slice of a 28x28 grayscale dataset, named ``<dataset>_classes<a>to<b>`` (``digits`` is accepted as
an alias). Roles are carved out of the TRAIN split per class with one seeded shuffle, so the server (public
examples, validation, held-out evaluation) and every client (its slice of one private cohort) derive the same
disjoint index sets from the labels alone, without sharing files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from metricdp_pytorch.utils.split_data import partition_by_class_counts

DATASET_IDS = {
    "kmnist": "tanganke/kmnist",
    "mnist": "ylecun/mnist",
    "fmnist": "zalando-datasets/fashion_mnist",
}
DEFAULT_BUDGETS = (32, 128)
PUBLIC_SETS = 3
NUM_CLIENTS = 8
DOMINANT_COUNT = 205
OFF_CLASS_COUNT = 17
CLIENT_SIZE = DOMINANT_COUNT + 3 * OFF_CLASS_COUNT
VALIDATION_PER_CLASS = 128
COHORT_PER_CLASS = 512
EVALUATION_PER_CLASS = 512
COHORTS = ("A", "B")
_TASK = re.compile(r"^(kmnist|mnist|fmnist)_(?:classes|digits)(\d)to(\d)$")


def parse_task(name: str) -> tuple[str, int]:
    """Return ``(dataset, first_class)`` for a task name; exactly four consecutive classes are required."""
    match = _TASK.match(name)
    if match is None or int(match.group(3)) != int(match.group(2)) + 3:
        raise ValueError(f"Task {name!r} must look like 'kmnist_classes0to3' (four consecutive classes).")
    return match.group(1), int(match.group(2))


@dataclass(frozen=True)
class TaskRoles:
    """Disjoint original-index sets for one task (indices into the dataset's TRAIN split)."""

    first_class: int
    public: dict[tuple[int, int], np.ndarray]
    validation: np.ndarray
    cohorts: dict[str, np.ndarray]
    evaluation: np.ndarray
    role_seed: int

    def all_indices(self) -> np.ndarray:
        parts = [*self.public.values(), self.validation, *self.cohorts.values(), self.evaluation]
        return np.concatenate(parts)


def build_roles(raw_labels: np.ndarray, first_class: int, role_seed: int, budgets: tuple[int, ...] = DEFAULT_BUDGETS) -> TaskRoles:
    """Carve per-class roles from the labels. The shuffle order is part of the experiment's identity."""
    rng = np.random.default_rng(role_seed + first_class)
    public = {(b, s): [] for b in budgets for s in range(PUBLIC_SETS)}
    validation, cohorts, evaluation = [], {c: [] for c in COHORTS}, []
    for k in range(4):
        ids = np.where(raw_labels == first_class + k)[0].copy()
        needed = sum(b // 4 for b in budgets) * PUBLIC_SETS + VALIDATION_PER_CLASS + len(COHORTS) * COHORT_PER_CLASS + EVALUATION_PER_CLASS
        if len(ids) < needed:
            raise ValueError(f"Class {first_class + k} has {len(ids)} examples; {needed} are needed.")
        rng.shuffle(ids)
        position = 0
        for b in budgets:
            for s in range(PUBLIC_SETS):
                public[(b, s)].append(ids[position : position + b // 4])
                position += b // 4
        for target, count in [(validation, VALIDATION_PER_CLASS), *[(cohorts[c], COHORT_PER_CLASS) for c in COHORTS], (evaluation, EVALUATION_PER_CLASS)]:
            target.append(ids[position : position + count])
            position += count
    roles = TaskRoles(
        first_class=first_class,
        public={key: np.concatenate(value) for key, value in public.items()},
        validation=np.concatenate(validation),
        cohorts={c: np.concatenate(v) for c, v in cohorts.items()},
        evaluation=np.concatenate(evaluation),
        role_seed=role_seed,
    )
    everything = roles.all_indices()
    if len(np.unique(everything)) != len(everything):
        raise AssertionError("Task roles must be disjoint.")
    return roles


def client_partitions(cohort_labels: np.ndarray, role_seed: int, first_class: int) -> list[np.ndarray]:
    """Label-stress partition of one cohort pool: client i is dominated by class i % 4 (205 vs 17 per other class)."""
    counts = np.full((NUM_CLIENTS, 4), OFF_CLASS_COUNT)
    counts[np.arange(NUM_CLIENTS), np.arange(NUM_CLIENTS) % 4] = DOMINANT_COUNT
    parts = partition_by_class_counts(cohort_labels, counts, seed=role_seed + first_class)
    if not all(len(part) == CLIENT_SIZE for part in parts) or len({int(i) for part in parts for i in part}) != NUM_CLIENTS * CLIENT_SIZE:
        raise AssertionError("Client partitions must be disjoint and of the documented size.")
    return parts


@lru_cache(maxsize=None)
def load_split(dataset: str):
    """Hugging Face TRAIN split for ``dataset`` (downloaded once, then served from the local cache)."""
    from datasets import load_dataset

    return load_dataset(DATASET_IDS[dataset])["train"]


@lru_cache(maxsize=None)
def split_labels(dataset: str) -> np.ndarray:
    return np.array(load_split(dataset)["label"])


def load_images(dataset: str, indices: np.ndarray) -> np.ndarray:
    """Pixels in [-0.5, 0.5] as float32, shape (n, 1, 28, 28), in the order of ``indices``."""
    data = load_split(dataset)
    pictures = data[[int(i) for i in indices]]["image"]
    return np.stack([np.asarray(picture, dtype=np.float32) for picture in pictures])[:, None] / 255 - 0.5


@lru_cache(maxsize=None)
def task_roles(task: str, role_seed: int, budgets: tuple[int, ...] = DEFAULT_BUDGETS) -> TaskRoles:
    dataset, first = parse_task(task)
    return build_roles(split_labels(dataset), first, role_seed, budgets)


def relabel(raw: np.ndarray, first_class: int) -> np.ndarray:
    return raw - first_class


def client_indices(task: str, role_seed: int, cohort: str, client_id: int, budgets: tuple[int, ...] = DEFAULT_BUDGETS) -> np.ndarray:
    """Original TRAIN-split indices held by one client of ``cohort``."""
    dataset, first = parse_task(task)
    roles = task_roles(task, role_seed, budgets)
    pool = roles.cohorts[cohort]
    parts = client_partitions(relabel(split_labels(dataset)[pool], first), role_seed, first)
    return pool[parts[client_id]]
