"""Unit tests for the v3 re-analysis statistics (synthetic data only; the integration check is the reproduction
of C1's published gate numbers inside ``reanalyze.main``)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("reanalyze", Path(__file__).resolve().parents[1] / "reanalyze.py")
reanalyze = importlib.util.module_from_spec(_spec)
sys.modules["reanalyze"] = reanalyze  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(reanalyze)


def _trajectories(rng, schedules, effect, difficulty):
    """Shadow losses at 11 checkpoints: a common decay, a per-candidate offset, noise, and a drop per sampled round."""
    c, r, rounds = schedules.shape
    base = 2.3 * np.exp(-np.arange(11) / 4.0)
    counts = schedules.reshape(c, r, 10, 5).sum(axis=3)
    extra = np.concatenate([np.zeros((c, r, 1)), np.cumsum(effect * counts, axis=2)], axis=2)
    return base + difficulty[:, None, None] + 0.02 * rng.standard_normal((c, r, 11)) - extra


def test_statistic_is_centred_without_participation():
    rng = np.random.default_rng(0)
    schedules = rng.random((32, 32, 50)) < 0.25
    losses = _trajectories(rng, np.zeros_like(schedules), 0.0, rng.normal(0, 0.5, 32))
    z = reanalyze.schedule_statistic(losses, schedules, 0.25)
    assert abs(z.mean()) < 3 * z.std() / np.sqrt(z.size)


def test_statistic_is_positive_when_training_follows_the_schedule():
    rng = np.random.default_rng(1)
    schedules = rng.random((32, 32, 50)) < 0.25
    losses = _trajectories(rng, schedules, 0.01, rng.normal(0, 0.5, 32))
    z = reanalyze.schedule_statistic(losses, schedules, 0.25)
    assert z.mean() > 5 * z.std() / np.sqrt(z.size)


def test_windows_match_checkpoints():
    schedules = np.zeros((1, 2, 50), dtype=bool)
    schedules[0, 0, :5] = True  # replicate 0 sampled in every round of the first window only
    losses = np.zeros((1, 2, 11))
    losses[0, 0, 0] = 1.0  # replicate 0 drops by 1 in window 1; replicate 1 never drops
    z = reanalyze.schedule_statistic(losses, schedules, 0.25)
    # The candidate's mean window-1 drop is 0.5, so replicate 0's centred drop is 0.5 there and -0 elsewhere.
    assert np.isclose(z[0, 0], (5 - 1.25) * 0.5)
    assert np.isclose(z[0, 1], (0 - 1.25) * -0.5)
