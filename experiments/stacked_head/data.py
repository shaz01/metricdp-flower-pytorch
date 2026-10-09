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
CORE_BUDGETS = (32, 128)
PUBLIC_SETS = 3                       # sets 0..2 of the core budgets sit in the original layout
EXTRA_SETS = {32: 60, 128: 10}        # appended sets 3..(2 + n) per core budget
BUDGET_512_SETS = 10                  # appended 512-image sets 0..9
NUM_CLIENTS = 8
DOMINANT_COUNT = 205
OFF_CLASS_COUNT = 17
CLIENT_SIZE = DOMINANT_COUNT + 3 * OFF_CLASS_COUNT
VALIDATION_PER_CLASS = 128
COHORT_PER_CLASS = 512
EVALUATION_PER_CLASS = 512
BASE_COHORTS = ("A", "B")
EXTRA_COHORTS = ("C", "D")
COHORTS = BASE_COHORTS + EXTRA_COHORTS
TEST_EVALUATION_PER_CLASS = 500
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


def per_class_needed() -> int:
    base = sum(b // 4 for b in CORE_BUDGETS) * PUBLIC_SETS + VALIDATION_PER_CLASS + len(BASE_COHORTS) * COHORT_PER_CLASS + EVALUATION_PER_CLASS
    appended = sum((b // 4) * EXTRA_SETS[b] for b in CORE_BUDGETS) + (512 // 4) * BUDGET_512_SETS + len(EXTRA_COHORTS) * COHORT_PER_CLASS
    return base + appended


def build_roles(raw_labels: np.ndarray, first_class: int, role_seed: int) -> TaskRoles:
    """Carve every role from the labels with one seeded shuffle per class (the shuffle order is the experiment's identity).

    Layout per class, in order (each region has a FIXED capacity, so no later request can move an earlier role):
      1. original: public sets 0..2 of budgets 32 and 128, validation, cohorts A and B, evaluation
      2. budget-32 sets 3..62     3. budget-128 sets 3..12     4. budget-512 sets 0..9     5. cohorts C and D
    """
    rng = np.random.default_rng(role_seed + first_class)
    public: dict[tuple[int, int], list] = {}
    validation, evaluation = [], []
    cohorts: dict[str, list] = {c: [] for c in COHORTS}
    needed = per_class_needed()
    for k in range(4):
        ids = np.where(raw_labels == first_class + k)[0].copy()
        if len(ids) < needed:
            raise ValueError(f"Class {first_class + k} has {len(ids)} examples; {needed} are needed.")
        rng.shuffle(ids)
        position = 0

        def take(count: int) -> np.ndarray:
            nonlocal position
            chunk = ids[position : position + count]
            position += count
            return chunk

        for b in CORE_BUDGETS:
            for s in range(PUBLIC_SETS):
                public.setdefault((b, s), []).append(take(b // 4))
        validation.append(take(VALIDATION_PER_CLASS))
        for c in BASE_COHORTS:
            cohorts[c].append(take(COHORT_PER_CLASS))
        evaluation.append(take(EVALUATION_PER_CLASS))
        for b in CORE_BUDGETS:
            for s in range(PUBLIC_SETS, PUBLIC_SETS + EXTRA_SETS[b]):
                public.setdefault((b, s), []).append(take(b // 4))
        for s in range(BUDGET_512_SETS):
            public.setdefault((512, s), []).append(take(512 // 4))
        for c in EXTRA_COHORTS:
            cohorts[c].append(take(COHORT_PER_CLASS))
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


def test_evaluation_indices(test_labels: np.ndarray, first_class: int, role_seed: int) -> np.ndarray:
    """A fixed class-balanced held-out set from the dataset's TEST split (never used by any role above or any earlier result)."""
    rng = np.random.default_rng(role_seed + first_class + 7919)
    picks = []
    for k in range(4):
        ids = np.where(test_labels == first_class + k)[0].copy()
        if len(ids) < TEST_EVALUATION_PER_CLASS:
            raise ValueError(f"Test split class {first_class + k} has only {len(ids)} examples.")
        rng.shuffle(ids)
        picks.append(ids[:TEST_EVALUATION_PER_CLASS])
    return np.concatenate(picks)


def client_partitions(cohort_labels: np.ndarray, role_seed: int, first_class: int) -> list[np.ndarray]:
    """Label-stress partition of one cohort pool: client i is dominated by class i % 4 (205 vs 17 per other class)."""
    counts = np.full((NUM_CLIENTS, 4), OFF_CLASS_COUNT)
    counts[np.arange(NUM_CLIENTS), np.arange(NUM_CLIENTS) % 4] = DOMINANT_COUNT
    parts = partition_by_class_counts(cohort_labels, counts, seed=role_seed + first_class)
    if not all(len(part) == CLIENT_SIZE for part in parts) or len({int(i) for part in parts for i in part}) != NUM_CLIENTS * CLIENT_SIZE:
        raise AssertionError("Client partitions must be disjoint and of the documented size.")
    return parts


@lru_cache(maxsize=None)
def load_split(dataset: str, split: str = "train"):
    """Hugging Face split for ``dataset`` (downloaded once, then served from the local cache)."""
    from datasets import load_dataset

    return load_dataset(DATASET_IDS[dataset])[split]


@lru_cache(maxsize=None)
def split_labels(dataset: str, split: str = "train") -> np.ndarray:
    return np.array(load_split(dataset, split)["label"])


def load_images(dataset: str, indices: np.ndarray, split: str = "train") -> np.ndarray:
    """Pixels in [-0.5, 0.5] as float32, shape (n, 1, 28, 28), in the order of ``indices``."""
    data = load_split(dataset, split)
    pictures = data[[int(i) for i in indices]]["image"]
    return np.stack([np.asarray(picture, dtype=np.float32) for picture in pictures])[:, None] / 255 - 0.5


@lru_cache(maxsize=None)
def task_roles(task: str, role_seed: int) -> TaskRoles:
    dataset, first = parse_task(task)
    return build_roles(split_labels(dataset), first, role_seed)


def relabel(raw: np.ndarray, first_class: int) -> np.ndarray:
    return raw - first_class


def client_indices(task: str, role_seed: int, cohort: str, client_id: int) -> np.ndarray:
    """Original TRAIN-split indices held by one client of ``cohort``."""
    dataset, first = parse_task(task)
    roles = task_roles(task, role_seed)
    pool = roles.cohorts[cohort]
    parts = client_partitions(relabel(split_labels(dataset)[pool], first), role_seed, first)
    return pool[parts[client_id]]
