"""G0: the fixed-noise options give the same server noise with the target IN and OUT.

The noise is drawn by the real strategy wrappers (built by ``make_strategy`` with each
combo's own multiplier and active client count) from the same RNG state, so any
difference can only come from the noise standard deviation.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord

from metricdp_pytorch.strategy_factory import make_strategy
from results.cia_frontier.eurosat_frontier.runner import build_combos
from results.contest_at_scale.plan_suite import planned_runs

CLIPPING_NORM = 5.0


def _drawn_noise(privacy: str, noise_multiplier: float, num_clients: int) -> tuple[float, np.ndarray]:
    strategy = make_strategy(
        "fedavg", privacy, num_clients=num_clients, fraction_evaluate=1.0,
        noise_multiplier=noise_multiplier, clipping_norm=CLIPPING_NORM,
    )
    zeros = ArrayRecord({"w": Array(np.zeros((64, 32), dtype=np.float32))})
    strategy.current_arrays = zeros
    if privacy == "metric-privacy":
        strategy.current_distance = 0.37  # the distance factor is held equal on purpose
    np.random.seed(1234)
    noisy = strategy._add_noise_to_aggregated_arrays(zeros)
    return strategy.current_noise_stdv, noisy.to_numpy_ndarrays()[0]


def _assert_same_noise(privacy: str, settings: list[tuple[float, int]]) -> None:
    draws = [_drawn_noise(privacy, multiplier, clients) for multiplier, clients in settings]
    reference_stdv, reference_noise = draws[0]
    assert reference_stdv > 0
    for stdv, noise in draws[1:]:
        assert stdv == pytest.approx(reference_stdv, rel=1e-12)
        np.testing.assert_array_equal(noise, reference_noise)


@pytest.mark.parametrize("privacy", ["global-dp", "metric-privacy"])
def test_plan_suite_fixed_noise_stdv_is_identical_in_and_out(privacy) -> None:
    groups = {name: (clients, matrix) for name, clients, matrix in
              planned_runs.cia_groups(fixed_noise_stdv=True)}
    for dataset in ("alzheimer", "fashion", "cifar"):
        settings = []
        for adjacency in ("in", "out"):
            clients, matrix = groups[f"{dataset}-{adjacency}-remove"]
            (combo,) = [c for c in matrix.list_combos(name_prefix="t", num_clients=clients)
                        if c.privacy == privacy and c.seed == 42]
            settings.append((combo.noise_multiplier, combo.num_clients))
        assert [clients for _, clients in settings] == [3, 2]
        _assert_same_noise(privacy, settings)


@pytest.mark.parametrize("privacy", ["global-dp", "metric-privacy"])
def test_plan_suite_default_keeps_the_count_dependent_noise(privacy) -> None:
    """Default behaviour is unchanged: z = 0.01 for 3 and 2 clients (std ratio 3/2)."""
    assert planned_runs.cia_groups() is planned_runs.CIA_GROUPS
    multipliers = {matrix.noise_multipliers for _, _, matrix in planned_runs.CIA_GROUPS}
    assert multipliers == {(planned_runs.NOISE_MULTIPLIER,)}
    in_stdv, _ = _drawn_noise(privacy, planned_runs.NOISE_MULTIPLIER, 3)
    out_stdv, _ = _drawn_noise(privacy, planned_runs.NOISE_MULTIPLIER, 2)
    assert out_stdv / in_stdv == pytest.approx(1.5)


@pytest.mark.parametrize("privacy", ["global-dp", "metric-privacy"])
@pytest.mark.parametrize("clients", [3, 5, 8, 48])
def test_frontier_runner_noise_is_identical_in_and_out(privacy, clients) -> None:
    runs = build_combos(alpha=0.3, seeds=[42], targets=list(range(clients)), clients=clients,
                        privacy=privacy, ratios=[0.0015])
    assert [c.num_clients for c in runs] == [clients] + [clients - 1] * clients
    _assert_same_noise(privacy, [(c.noise_multiplier, c.num_clients) for c in runs[:2]])
