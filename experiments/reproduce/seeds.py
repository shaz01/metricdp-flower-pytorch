"""Separate seeds for the parts of a run (CIA roadmap v2, measurement standard Section 2).

``layout-seed`` draws the data layout (partition, client train/test splits, server
evaluation split), ``init-seed`` the initial model, ``client-seed`` the client round seeds
(batches, augmentation, dropout) and ``noise-seed`` the server noise. Each defaults to the
run's ``seed``, so runs that set only ``seed`` behave exactly as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SEED_KEYS = ("layout-seed", "init-seed", "client-seed", "noise-seed")


@dataclass(frozen=True)
class RunSeeds:
    layout: int
    init: int
    client: int
    noise: int
    # Server noise uses a dedicated generator only when the noise seed is set explicitly;
    # otherwise it keeps NumPy's global state (seeded by ``seed``), as committed runs did.
    noise_explicit: bool


def run_seeds(config: Mapping[str, Any]) -> RunSeeds:
    seed = int(config.get("seed", 42))
    layout, init, client, noise = (int(config.get(key, seed)) for key in SEED_KEYS)
    return RunSeeds(layout, init, client, noise, noise_explicit="noise-seed" in config)
