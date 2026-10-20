"""Confirmatory run (owner decision 11): plan and confirmation rule on synthetic records (no training)."""

from __future__ import annotations

from experiments.frozen_head import confirm, runner


def _chosen():
    grid = dict(local_epochs=1, local_lr=0.2, server_lr=1.0, clip_multiplier=1.0)
    return {size: {setting: dict(grid) for setting in runner.settings()} for size in runner.PUBLIC_SIZES}


def test_confirm_plan_uses_the_fresh_layout_seeds_and_mu_at_most_2():
    jobs = runner.confirm_jobs(_chosen())
    assert len(jobs) == 2 * 12 * runner.PUBLIC_SUBSETS * runner.PUBLIC_REPLICATES == 480
    assert len({job.name for job in jobs}) == len(jobs)
    assert {job.public_size for job in jobs} == {20, 50}
    assert all(job.mu_total <= 2 for job in jobs)
    assert all(job.layout_seed == runner.CONFIRM_LAYOUT == 44 for job in jobs)
    seeds = {job.seeds for job in jobs}
    assert seeds == {(30_000 + r, 40_000 + r) for r in range(runner.PUBLIC_REPLICATES)}
    assert all(job.path.parent.name == "confirm_public" for job in jobs)


def test_original_stages_are_unchanged():
    job = runner.public_evaluation_jobs({20: {(48, 10, 1.0): _chosen()[20][(48, 10, 1.0)]}})[0]
    assert job.layout_seed == runner.EVALUATION_LAYOUT and job.seeds == (10_000, 20_000)


def _reading(lows):
    return {"settings": [{"public_size": size, "clients": n, "rounds": t, "mu_total": mu, "gain_ci95_pp": [low, low + 5]}
                         for (size, n, t, mu), low in lows.items()]}


def test_confirmation_rule():
    original = {20: [(48, 10, 1.0), (480, 10, 1.0)], 50: [(480, 10, 2.0)]}
    lows = {(20, 48, 10, 1.0): 6.0, (20, 480, 10, 1.0): 5.0, (50, 480, 10, 2.0): 4.9}
    verdicts = confirm.confirmation(_reading(lows), original)
    assert verdicts[20]["verdict"] == "confirmed"
    assert verdicts[50]["verdict"] == "not confirmed"
    lows[(20, 48, 10, 1.0)] = 1.0
    partly = confirm.confirmation(_reading(lows), original)[20]
    assert partly["verdict"] == "partly confirmed" and partly["held"] == [(480, 10, 1.0)]
