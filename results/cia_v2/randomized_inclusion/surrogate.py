"""Label-matched surrogate sets (ROADMAP.md B5 item 1): public images resampled to a candidate's label histogram."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def label_matched_indices(labels, *, target_counts: Mapping[int, int], seed: int) -> np.ndarray:
    """Indices into ``labels`` with exactly ``target_counts[label]`` records per label, drawn without replacement."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(target_counts):
        count = int(target_counts[label])
        pool = np.flatnonzero(labels == label)
        if count > len(pool):
            raise ValueError(f"label {label}: {count} records requested, {len(pool)} available")
        chosen.append(rng.choice(pool, size=count, replace=False))
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=int)
