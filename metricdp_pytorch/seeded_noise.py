"""Server noise from a dedicated generator, and diagnostics of the noise actually drawn (CIA roadmap v2, M0).

Flower's ``add_gaussian_noise_inplace`` draws from NumPy's global state, which
``seed_training(seed)`` sets, so two trajectories with the same seed draw identical noise
(research/cia_roadmap_v2/REVIEW.md Section 3). ``set_noise_seed`` gives a strategy its own
generator; strategies without one keep the global draws, so committed runs replay unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from flwr.supercore.differential_privacy import add_gaussian_noise_inplace

# Fixed coordinate blocks for the realized-noise-energy statistic: two whole-model values per
# 2-round run would make every correlation +-1; 64 blocks per round give 128 points per run.
NOISE_BLOCKS = 64


def add_gaussian_noise(arrays: Sequence[np.ndarray], std: float, rng: np.random.Generator | None) -> None:
    """Add N(0, std^2) noise in place; ``rng=None`` keeps Flower's global-state draws."""
    if rng is None:
        add_gaussian_noise_inplace(list(arrays), std)
        return
    for array in arrays:
        array += rng.normal(0.0, std, array.shape).astype(array.dtype)


def set_noise_seed(strategy, noise_seed: int) -> None:
    """Draw all of ``strategy``'s server noise from one generator seeded by ``noise_seed``.

    Strategies that already own a generator (private median, residual clipping) use it for
    every draw, so the noise seed replaces their ``seed``.
    """
    strategy.noise_rng = np.random.default_rng(int(noise_seed))
    if hasattr(strategy, "rng"):
        strategy.rng = strategy.noise_rng


def _flat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def noise_diagnostics(
    aggregated: Sequence[np.ndarray],
    released: Sequence[np.ndarray],
    current: Sequence[np.ndarray],
    *,
    std: float,
) -> dict[str, float | list[float]]:
    """Released-update norm and, per block, ``||Delta_b||^2 - ||s_b||^2 - std^2 D_b``.

    ``Delta = released - current`` is what every participant sees; ``s = aggregated - current``
    is the pre-noise signal. The block statistic equals ``||xi_b||^2 - E||xi_b||^2 + 2 <s_b, xi_b>``
    (the statistic of ``research/cia_roadmap_v2/scripts/measurement_audit.py`` per block).
    """
    current_flat = _flat(current)
    delta = _flat(released) - current_flat
    signal = _flat(aggregated) - current_flat
    blocks = [float(d @ d - s @ s - std**2 * len(d))
              for d, s in zip(np.array_split(delta, NOISE_BLOCKS), np.array_split(signal, NOISE_BLOCKS), strict=True)]
    return {"dp-released-update-norm": float(np.linalg.norm(delta)), "dp-noise-energy-blocks": blocks}


class SeededNoiseMixin:
    """Attributes and the noise step shared by the server-side DP strategies."""

    noise_rng: np.random.Generator | None = None
    log_noise_diagnostics: bool = False

    def _noisy_copy(self, aggregated: list[np.ndarray], std: float) -> list[np.ndarray]:
        """Return ``aggregated`` plus noise; fills ``noise_round_diagnostics`` when logging."""
        released = [np.array(a, copy=True) for a in aggregated]
        add_gaussian_noise(released, std, self.noise_rng)
        self.noise_round_diagnostics = (
            noise_diagnostics(aggregated, released, self.current_arrays.to_numpy_ndarrays(), std=std)
            if self.log_noise_diagnostics else {}
        )
        return released
