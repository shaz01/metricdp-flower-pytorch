"""Randomized-inclusion design for one-run CIA audits (CIA roadmap v2, Section 2 item 3)."""

from __future__ import annotations

import numpy as np


def balanced_design(candidates: int, replicates: int, *, seed: int, fixed_count: bool = True) -> np.ndarray:
    """Boolean ``(candidates, replicates)`` matrix; entry ``[c, j]`` says candidate ``c`` is IN replicate ``j``.

    Every candidate is IN in exactly half of the replicates. With ``fixed_count`` every
    replicate also includes exactly half of the candidates, so all runs have the same number
    of active clients: replicates come in pairs, and the second of a pair includes exactly the
    candidates the first leaves out (a fresh random half per pair).
    """
    if replicates < 2 or replicates % 2:
        raise ValueError("replicates must be a positive even number")
    if fixed_count and candidates % 2:
        raise ValueError("a fixed count needs an even number of candidates")
    rng = np.random.default_rng(seed)
    design = np.zeros((candidates, replicates), dtype=bool)
    if fixed_count:
        for pair in range(replicates // 2):
            half = np.isin(np.arange(candidates), rng.permutation(candidates)[: candidates // 2])
            design[:, 2 * pair], design[:, 2 * pair + 1] = half, ~half
    else:
        for candidate in range(candidates):
            design[candidate, rng.permutation(replicates)[: replicates // 2]] = True
    return design
