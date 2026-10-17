"""A1 (CIA roadmap v2): WP3's private-median metric privacy as a runner privacy mode (``--privacy private-median``)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.reproduce import runner
from experiments.reproduce.server import make_run_strategy
from metricdp_pytorch.private_metric_strategy import PrivateMedianMetricPrivacyServerSideFixedClipping

BASE_ARGS = ["--seed", "42", "--noise-multiplier", "0.5", "--clipping-norm", "5.0", "--num-clients", "4",
             "--model-module", "experiments.reproduce.eurosat_cnn:create_model", "--rounds", "2", "--local-epochs", "1"]
MEDIAN_ARGS = ["--privacy", "private-median", "--aggregation-weighting", "equal", "--num-slots", "4",
               "--median-quantile", "0.3", "--median-quantile-min", "0.03", "--median-quantile-max", "1.0"]


def _config(*extra):
    return runner.build_run_config(runner._parser().parse_args([*BASE_ARGS, *extra]))


def test_runner_passes_the_private_median_settings():
    config = _config(*MEDIAN_ARGS)
    assert config["privacy"] == "private-median"
    assert {key: config[key] for key in ("num-slots", "median-quantile", "median-quantile-min",
                                         "median-quantile-max")} == {
        "num-slots": 4, "median-quantile": 0.3, "median-quantile-min": 0.03, "median-quantile-max": 1.0}
    assert "median-bit-noise-std" not in config
    assert _config(*MEDIAN_ARGS, "--median-bit-noise-std", "20")["median-bit-noise-std"] == 20.0


@pytest.mark.parametrize("change", [
    ("--aggregation-weighting", "num-examples"),  # the mechanism needs equal weights
    ("--median-quantile", "2.0"),  # initial quantile above the maximum
])
def test_runner_rejects_invalid_private_median_settings(change):
    args = list(MEDIAN_ARGS)
    flag, value = change
    args[args.index(flag) + 1] = value
    with pytest.raises(ValueError):
        _config(*args)


def test_runner_requires_the_quantile_bounds():
    with pytest.raises(ValueError):
        _config(*MEDIAN_ARGS[:-2])


def test_median_flags_apply_only_to_private_median():
    with pytest.raises(ValueError):
        _config("--privacy", "global-dp", "--median-quantile", "0.3")


def _strategy_config(**extra):
    config = {"aggregation": "fedavg", "privacy": "private-median", "num-clients": 4, "noise-multiplier": 0.5,
              "clipping-norm": 5.0, "aggregation-weighting": "equal", "seed": 42, "num-slots": 4,
              "median-quantile": 0.3, "median-quantile-min": 0.03, "median-quantile-max": 1.0}
    config.update(extra)
    return config


def test_run_strategy_builds_the_private_median_mechanism():
    strategy = make_run_strategy(_strategy_config(**{"log-noise-diagnostics": True}))
    assert isinstance(strategy, PrivateMedianMetricPrivacyServerSideFixedClipping)
    assert strategy.num_slots == 4 and strategy.noise_multiplier == 0.5 and strategy.clipping_norm == 5.0
    assert (strategy.quantile, strategy.minimum_quantile, strategy.maximum_quantile) == (0.3, 0.03, 1.0)
    assert strategy.log_noise_diagnostics is True
    explicit = make_run_strategy(_strategy_config(**{"median-bit-noise-std": 20.0}))
    assert explicit.bit_noise_std == 20.0


def test_private_median_run_strategy_draws_from_the_noise_seed():
    a = make_run_strategy(_strategy_config(seed=1, **{"noise-seed": 5}))
    b = make_run_strategy(_strategy_config(seed=2, **{"noise-seed": 5}))
    np.testing.assert_array_equal(a.rng.normal(size=5), b.rng.normal(size=5))
