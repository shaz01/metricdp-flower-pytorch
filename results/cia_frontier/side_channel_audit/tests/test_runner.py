"""Plan-level tests for the WP1 audit runner and its Alzheimer data module (no training)."""

from __future__ import annotations

import json

import pytest

from results.cia_frontier.side_channel_audit import data, runner


def test_full_plan_matches_the_protocol() -> None:
    cells = runner.plan(list(runner.DATASETS), runner.CLIENT_COUNTS, runner.MECHANISMS, runner.SEEDS)
    combos = [combo for _, _, cell in cells for combo in cell]
    assert len(cells) == 12 and len(combos) == 228  # 2 x 3 seeds x (4 + 6 + 9) x 2 datasets
    assert len({combo.run_name() for combo in combos}) == 228
    for path, targets, cell in cells:
        dataset, clients, privacy = path.parts
        n = int(clients.removeprefix("clients-"))
        assert targets == list(range(n))
        assert sum(combo.out_target is None for combo in cell) == 3
        assert sorted(c.out_target for c in cell if c.out_target is not None) == sorted(list(range(n)) * 3)
        for combo in cell:
            assert combo.privacy == privacy and combo.canonical_clients == n
            assert combo.diagnostics and combo.aggregation_weighting == "equal"
            assert combo.hyperparams.rounds == 50 and combo.alpha == 0.3
            assert combo.name_prefix.startswith(dataset + "-dirichlet-a0.3-")
            assert combo.data_module == runner.DATASETS[dataset]["data_module"]
            # Fixed noise std: multiplier / active clients is the same IN and OUT.
            assert combo.noise_multiplier / combo.num_clients == pytest.approx(runner.DATASETS[dataset]["ratio"])
            args = combo.runner_args(output_dir="o", max_parallel_clients=8, client_cpus=1)
            assert "--log-loo-residuals" in args and args[args.index("--aggregation-weighting") + 1] == "equal"


def test_smoke_rounds_change_names_and_never_touch_the_protocol_runs() -> None:
    full = runner.build_audit_combos("alzheimer", 3, "global-dp", seeds=[42])
    smoke = runner.build_audit_combos("alzheimer", 3, "global-dp", seeds=[42], rounds=2)
    assert {c.hyperparams.rounds for c in smoke} == {2}
    assert not {c.run_name() for c in full} & {c.run_name() for c in smoke}


def test_alzheimer_partitions_are_canonical_across_removal_and_respect_the_minimum() -> None:
    labels = tuple([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)
    parts = data.dirichlet_partitions(labels, 5, 0.3, 42, data.MIN_RECORDS)
    assert min(map(len, parts)) >= data.MIN_RECORDS
    assert sorted(i for part in parts for i in part) == list(range(len(labels)))
    profile = {"alpha": 0.3, "clients": 5, "out_target": 2}
    view = data.create_data_module({"partition-profile": json.dumps(profile)})
    assert view.num_active_partitions == 4 and 2 not in view.active_partition_ids


def _record(combo, targets, output, max_parallel_clients):
    (output / "executed.txt").open("a").write(combo.run_name() + "\n")


def _hang(combo, targets, output, max_parallel_clients):
    """Hang with a grandchild in its own process group (like the isolated runner and Ray)."""
    import subprocess
    import time

    child = subprocess.Popen(["sleep", "600"], start_new_session=True)
    (output / "grandchild.pid").write_text(str(child.pid))
    time.sleep(600)


def test_execute_shared_skips_a_trajectory_another_worker_holds(tmp_path) -> None:
    import fcntl
    import hashlib
    import tempfile
    from pathlib import Path

    held, free = runner.build_audit_combos("alzheimer", 3, "global-dp", seeds=[42])[:2]
    identity = str((tmp_path / held.run_name()).resolve()).encode()
    lock = Path(tempfile.gettempdir()) / ("auc-frontier-" + hashlib.sha256(identity).hexdigest() + ".lock")
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.execute_shared([held, free], [0, 1, 2], tmp_path, 1, worker=_record, poll_seconds=0.1)
    assert (tmp_path / "executed.txt").read_text().split() == [free.run_name()]


def test_execute_shared_kills_a_stalled_trajectory_and_moves_on(tmp_path) -> None:
    import time

    hung, after = runner.build_audit_combos("alzheimer", 3, "global-dp", seeds=[42])[:2]
    started = time.monotonic()
    stalled = runner.execute_shared([hung], [0, 1, 2], tmp_path, 1, worker=_hang,
                                    stall_minutes=0.1, poll_seconds=0.2)
    assert stalled == [hung.run_name()]
    assert time.monotonic() - started < 90
    import psutil

    grandchild = int((tmp_path / "grandchild.pid").read_text())
    assert not psutil.pid_exists(grandchild) or psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE
    assert runner.execute_shared([after], [0, 1, 2], tmp_path, 1, worker=_record, poll_seconds=0.1) == []
    assert (tmp_path / "executed.txt").read_text().split() == [after.run_name()]
