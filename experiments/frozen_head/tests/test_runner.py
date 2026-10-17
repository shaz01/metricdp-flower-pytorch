"""Runner planning, selection and gate logic on synthetic records (no training)."""

from __future__ import annotations

import numpy as np

from experiments.frozen_head import runner


def test_tuning_plan_is_complete_and_unique():
    jobs = runner.tuning_jobs()
    assert len(jobs) == len(runner.settings()) * 2 * 3 * 3 * 2 * len(runner.TUNING_SEEDS) == 1440
    assert len({job.name for job in jobs}) == len(jobs)
    assert all(job.layout_seed == runner.TUNING_LAYOUT for job in jobs)


def test_evaluation_seeds_are_independent_per_replicate_and_shared_across_settings():
    chosen = {(48, 10, 1.0): dict(local_epochs=1, local_lr=0.2, server_lr=1.0, clip_multiplier=1.0),
              (48, 10, 2.0): dict(local_epochs=5, local_lr=0.05, server_lr=3.0, clip_multiplier=0.5)}
    jobs = runner.evaluation_jobs(chosen)
    assert len(jobs) == 2 * runner.REPLICATES
    first = [job.seeds for job in jobs if job.mu_total == 1.0]
    second = [job.seeds for job in jobs if job.mu_total == 2.0]
    assert first == second and len(set(first)) == runner.REPLICATES
    assert all(job.layout_seed == runner.EVALUATION_LAYOUT for job in jobs)
    assert {s for pair in first for s in pair}.isdisjoint({1_000, 1_001, 2_000, 2_001})  # not tuning seeds


def _tuning_record(server_lr, epochs, lr, clip, seed, accuracy):
    job = dict(clients=48, rounds=10, mu_total=1.0, server_lr=server_lr, local_epochs=epochs, local_lr=lr,
               clip_multiplier=clip, replicate=seed)
    return {"job": job, "final_pooled_local_test": accuracy}


def test_select_takes_best_mean_and_breaks_ties_towards_the_smaller_step():
    records = [_tuning_record(3.0, 1, 0.2, 1.0, s, 0.90) for s in (0, 1)]
    records += [_tuning_record(1.0, 5, 0.2, 1.0, s, 0.90) for s in (0, 1)]
    records += [_tuning_record(10.0, 1, 0.2, 1.0, s, a) for s, a in ((0, 0.95), (1, 0.80))]  # mean 0.875
    chosen = runner.select(records)[(48, 10, 1.0)]
    assert chosen["server_lr"] == 1.0 and chosen["local_epochs"] == 5


def _eval_record(mu, replicate, correct, baseline):
    return {"job": dict(clients=48, rounds=10, mu_total=mu, replicate=replicate), "server_test_correct": correct,
            "public_only": {"server_test_correct": baseline},
            "per_client_local_test": {"min": 0.5, "p10": 0.7, "median": 0.9}, "privacy": None}


def test_analyze_gate_reading():
    rng = np.random.default_rng(0)
    baseline = (rng.random(1000) < 0.90).astype(int).tolist()
    strong = [_eval_record(1.0, r, (rng.random(1000) < 0.99).astype(int).tolist(), baseline) for r in range(8)]
    weak = [_eval_record(2.0, r, (rng.random(1000) < 0.92).astype(int).tolist(), baseline) for r in range(8)]
    result = runner.analyze(strong + weak, {})
    rows = {row["mu_total"]: row for row in result["settings"]}
    assert rows[1.0]["gate_reached"] and rows[1.0]["gate_robust"]
    assert not rows[2.0]["gate_reached"]
    assert result["verdict"] == "pass" and result["passing_settings"] == [(48, 10, 1.0)]
    low, high = rows[2.0]["gain_ci95_pp"]
    assert low < rows[2.0]["gain_pp"] < high
