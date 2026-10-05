"""Deterministic per-client influence diagnostics for clipped model updates."""
from __future__ import annotations

import numpy as np


def per_client_influence(updates, client_ids, num_examples, clipping_norm):
    """Return ID-keyed diagnostics; inputs are flattened update vectors.

    Computes no random values and does not mutate inputs, so it is safe to call
    alongside aggregation without changing training RNG state or updates.
    """
    updates = [np.asarray(u, dtype=np.float64).reshape(-1) for u in updates]
    if not (len(updates) == len(client_ids) == len(num_examples)) or not updates:
        raise ValueError("Updates, IDs and example counts must be nonempty and aligned")
    if len(set(client_ids)) != len(client_ids) or any(n <= 0 for n in num_examples):
        raise ValueError("Client IDs must be unique and example counts positive")
    if any(u.shape != updates[0].shape for u in updates) or clipping_norm <= 0:
        raise ValueError("Updates must have equal shapes and clipping norm must be positive")
    norms = [float(np.linalg.norm(u)) for u in updates]
    clipped = [u * min(1., clipping_norm / norm) if norm else u.copy()
               for u, norm in zip(updates, norms, strict=True)]
    total = float(sum(num_examples))
    weights = [n / total for n in num_examples]
    average = sum((w * u for w, u in zip(weights, clipped, strict=True)), np.zeros_like(clipped[0]))
    avg_norm = float(np.linalg.norm(average))
    rows = {}
    for cid, n, w, norm, u, c in zip(client_ids, num_examples, weights, norms, updates, clipped, strict=True):
        delta = c - average
        delta_norm = float(np.linalg.norm(delta))
        rows[str(cid)] = {
            "update_norm_before_clipping": norm,
            "clipped": norm > clipping_norm,
            "distance_from_weighted_clipped_average": delta_norm,
            "cosine_with_weighted_clipped_average": (float(np.dot(c, average) / (np.linalg.norm(c) * avg_norm))
                                                       if np.linalg.norm(c) and avg_norm else 0.0),
            "leave_one_out_influence_norm": float(w / (1 - w) * delta_norm) if w < 1 else 0.0,
            "aggregation_weight": w,
            "num_examples": int(n),
        }
    return rows
