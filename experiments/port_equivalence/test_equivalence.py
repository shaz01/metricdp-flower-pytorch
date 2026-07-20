"""Cross-version reproducibility test for the metric-aware DP port."""

import pytest

from experiments.port_equivalence.compare import SEEDS, compare_seed

TOLERANCE = 1e-15


@pytest.mark.reproducibility
@pytest.mark.parametrize("seed", SEEDS)
def test_port_matches_original_strategy(seed: int) -> None:
    """Match distance, noise scale, and noised aggregate for one RNG seed."""
    differences = compare_seed(seed)

    assert differences["distance_diff"] <= TOLERANCE
    assert differences["stdv_diff"] <= TOLERANCE
    assert differences["output_max_diff"] <= TOLERANCE
