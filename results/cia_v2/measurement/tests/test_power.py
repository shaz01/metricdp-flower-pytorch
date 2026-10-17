"""M0 power table helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from results.cia_v2.measurement import power
from results.cia_v2.randomized_inclusion.runner import CANDIDATES, TOP_DECILE


def test_minimum_detectable_interpolates_at_eighty_percent():
    grid = pd.DataFrame({"difference": [0.0, 0.1, 0.2, 0.3], "detected": [0.05, 0.4, 0.9, 1.0]})
    assert power.minimum_detectable(grid) == pytest.approx(0.1 + 0.4 / 0.5 * 0.1)
    assert np.isnan(power.minimum_detectable(grid.assign(detected=[0.0, 0.1, 0.2, 0.3])))


def test_top_decile_candidates_take_top_decile_shifts():
    for candidate, source in zip(CANDIDATES, power.sources()):
        assert (source in power.TOP_SOURCES) == (candidate in TOP_DECILE)


@pytest.mark.skipif(not power.WP2.exists(), reason="WP2 trajectories not checked out")
def test_wp2_model_has_four_independent_seeds_per_world():
    trajectories = power.load_wp2()
    assert len(trajectories) == 2 * 5 * 4
    model = power.empirical_model(trajectories)
    assert model["covariance"].shape == (10, 10)
    assert np.all(np.linalg.eigvalsh(model["covariance"]) > 0)
    # Without DP the IN world has the lower shadow loss for the largest client.
    assert np.mean(model["means"][("in", 23)] - model["means"][("out", 23)]) < 0
