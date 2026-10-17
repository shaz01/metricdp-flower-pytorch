"""M0 (CIA roadmap v2): separate layout, init, client and noise seeds, each defaulting to ``seed``."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.reproduce import runner
from experiments.reproduce.seeds import SEED_KEYS, run_seeds
from experiments.reproduce.server import make_run_strategy

BASE_ARGS = ["--seed", "42", "--noise-multiplier", "0.5", "--clipping-norm", "5.0",
             "--model-module", "experiments.reproduce.eurosat_cnn:create_model", "--privacy", "global-dp",
             "--rounds", "2", "--local-epochs", "1"]


def test_every_seed_defaults_to_seed():
    seeds = run_seeds({"seed": 7})
    assert (seeds.layout, seeds.init, seeds.client, seeds.noise) == (7, 7, 7, 7)
    assert seeds.noise_explicit is False


def test_missing_seed_defaults_to_42():
    assert run_seeds({}).client == 42


def test_explicit_seeds_override_seed():
    seeds = run_seeds({"seed": 7, "layout-seed": 1, "init-seed": 2, "client-seed": 3, "noise-seed": 4})
    assert (seeds.layout, seeds.init, seeds.client, seeds.noise) == (1, 2, 3, 4)
    assert seeds.noise_explicit is True


def test_runner_passes_explicit_seeds_only():
    parser = runner._parser()
    plain = runner.build_run_config(parser.parse_args(BASE_ARGS))
    assert not set(SEED_KEYS) & set(plain)
    seeded = runner.build_run_config(parser.parse_args(
        [*BASE_ARGS, "--layout-seed", "1", "--init-seed", "2", "--client-seed", "3", "--noise-seed", "4"]))
    assert {key: seeded[key] for key in SEED_KEYS} == {
        "layout-seed": 1, "init-seed": 2, "client-seed": 3, "noise-seed": 4}
    assert seeded["seed"] == 42


def test_runner_passes_the_noise_diagnostics_flag():
    parser = runner._parser()
    assert "log-noise-diagnostics" not in runner.build_run_config(parser.parse_args(BASE_ARGS))
    config = runner.build_run_config(parser.parse_args([*BASE_ARGS, "--log-noise-diagnostics"]))
    assert config["log-noise-diagnostics"] is True


def _config(**extra):
    config = {"aggregation": "fedavg", "privacy": "global-dp", "num-clients": 4, "noise-multiplier": 0.5,
              "clipping-norm": 5.0, "aggregation-weighting": "equal", "seed": 42}
    config.update(extra)
    return config


def test_run_strategy_without_noise_seed_uses_the_global_state():
    strategy = make_run_strategy(_config())
    assert strategy.noise_rng is None
    assert strategy.log_noise_diagnostics is False


def test_run_strategy_with_noise_seed_gets_its_own_generator():
    first = make_run_strategy(_config(**{"noise-seed": 9, "log-noise-diagnostics": True}))
    second = make_run_strategy(_config(**{"noise-seed": 9}))
    np.testing.assert_array_equal(first.noise_rng.normal(size=5), second.noise_rng.normal(size=5))
    assert first.log_noise_diagnostics is True


def test_residual_clipping_run_strategy_draws_from_the_noise_seed():
    residual = {"privacy": "residual-clipping", "num-slots": 4, "residual-center": "previous",
                "residual-rho": 1.0}
    a = make_run_strategy(_config(**residual, seed=1, **{"noise-seed": 5}))
    b = make_run_strategy(_config(**residual, seed=2, **{"noise-seed": 5}))
    np.testing.assert_array_equal(a.rng.normal(size=5), b.rng.normal(size=5))


@pytest.mark.parametrize("privacy", ["global-dp", "metric-privacy"])
def test_run_strategy_keeps_loo_logging_outermost(privacy):
    strategy = make_run_strategy(_config(privacy=privacy, **{"noise-seed": 1, "log-loo-residuals": True}))
    assert strategy.strategy.noise_rng is not None
