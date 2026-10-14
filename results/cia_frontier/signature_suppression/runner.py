"""WP2 (D1 Stage A) signature-suppression screen. Plans unless --execute.

Design (PROTOCOL.md): EuroSAT, label-Dirichlet alpha 0.3, 48 canonical clients, seed 42,
50 rounds, no DP, removal adjacency (one IN trajectory, one OUT trajectory per target),
five fixed targets, ten local-training configurations, equal aggregation weights except
the size-weighted vanilla reference, WP0 diagnostics on (round-0 checkpoint, shadow
increments and gradient norms, server-side leave-one-out residual norms).

Several workers may run the same command (e.g. one per GPU): every trajectory is locked
while it trains, a locked trajectory is skipped, and completed ones are skipped on restart.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path

from results.cia_frontier.eurosat_frontier.runner import FrontierCombo, atomic_json, build_combos
from results.contest_at_scale.auc_frontier.eurosat_remove import HYPERPARAMS

ALPHA = 0.3
CLIENTS = 48
SEEDS = (42,)
LAYOUT_SEED = 42
ROUNDS = 50
# Fixed before any run (PROTOCOL.md Section 3): the two largest clients of the seed-42
# federation (23: 1,150 records, 7: 1,089), then clients 1 (149), 0 (304) and 3 (588).
TARGETS = (23, 7, 1, 0, 3)
# Fixed local step budget: the mean of the vanilla per-client step counts at seed 42
# (5 epochs x ceil(train records / 32); 2,800 steps over 48 clients = 58.3), so the
# federation takes about as many optimizer steps per round as vanilla.
FIXED_LOCAL_STEPS = 58
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "cia_frontier" / "signature_suppression" / "results" / "stage_a"
CONFIG_SCHEMA = 1

# name -> local-training changes relative to vanilla with equal weights.
CONFIGS = {
    "vanilla": {},
    "vanilla-size-weighted": {"aggregation_weighting": "num-examples"},
    "weight-decay-5e-4": {"weight_decay": 5e-4},
    "local-epochs-1": {"local_epochs": 1},
    "fedprox-mu-0.01": {"proximal_mu": 0.01},
    "fedprox-mu-0.1": {"proximal_mu": 0.1},
    "fedprox-mu-1": {"proximal_mu": 1.0},
    "fixed-steps": {"local_steps": FIXED_LOCAL_STEPS},
    "consensus-kl-0.3": {"consensus_beta": 0.3},
    "consensus-kl-1": {"consensus_beta": 1.0},
}
# FedProx in the FedProx paper's convention (mu/2) ||w - w_t||^2 (PROTOCOL.md Section 3).
PROXIMAL_NORMALIZATION = "sum"
# Roadmap unit: one 48-client, 100-round EuroSAT trajectory on the reference machine.
TRAJECTORY_HOURS_PER_ROUND = 0.74 / 100


@dataclass(frozen=True, kw_only=True)
class SuppressionCombo(FrontierCombo):
    config: str
    proximal_mu: float | None = None
    consensus_beta: float = 0.0
    local_steps: int = 0

    def runner_args(self, **kwargs):
        args = super().runner_args(**kwargs)
        if self.proximal_mu is not None:
            args = (*args, "--proximal-mu", repr(self.proximal_mu),
                    "--proximal-normalization", PROXIMAL_NORMALIZATION)
        if self.consensus_beta:
            args = (*args, "--consensus-kl-beta", repr(self.consensus_beta))
        if self.local_steps:
            args = (*args, "--local-steps", str(self.local_steps))
        return args


def build_suppression_combos(config, seeds=SEEDS, targets=TARGETS, adjacency="both", out_targets=None,
                             rounds=ROUNDS):
    """All trajectories of one configuration. ``rounds`` other than ``ROUNDS`` is for smoke tests."""
    changes = dict(CONFIGS[config])
    weighting = changes.pop("aggregation_weighting", "equal")
    hyperparams = replace(HYPERPARAMS, rounds=rounds,
                          weight_decay=changes.pop("weight_decay", HYPERPARAMS.weight_decay),
                          local_epochs=changes.pop("local_epochs", HYPERPARAMS.local_epochs))
    # Seed 42 keeps the frontier convention (one seed for layout and training, run names
    # unchanged); extension seeds keep the seed-42 layout (PROTOCOL.md Section 6).
    combos = [combo for seed in seeds for combo in build_combos(
        alpha=ALPHA, seeds=[seed], targets=list(targets), clients=CLIENTS, privacy="vanilla", ratios=None,
        adjacency=adjacency, out_targets=out_targets, diagnostics=True,
        partition_seed=None if seed == LAYOUT_SEED else LAYOUT_SEED)]
    return [SuppressionCombo(**{
        **{field.name: getattr(combo, field.name) for field in fields(FrontierCombo)},
        "name_prefix": f"{config}-{combo.name_prefix}",
        "aggregation": "fedprox" if "proximal_mu" in changes else "fedavg",
        "aggregation_weighting": weighting,
        "hyperparams": hyperparams,
        "config": config, **changes,
    }) for combo in combos]


def pinned_config(combo):
    """Per-trajectory record of the local-training configuration; a change needs a new output."""
    return {"schema": CONFIG_SCHEMA, "config": combo.config, "aggregation": combo.aggregation,
            "aggregation_weighting": combo.aggregation_weighting,
            "local_epochs": combo.hyperparams.local_epochs, "weight_decay": combo.hyperparams.weight_decay,
            "proximal_mu": combo.proximal_mu,
            "proximal_normalization": PROXIMAL_NORMALIZATION if combo.proximal_mu is not None else None,
            "consensus_kl_beta": combo.consensus_beta, "local_steps": combo.local_steps,
            "run_name": combo.run_name()}


def plan(configs, seeds=SEEDS, adjacency="both", out_targets=None, rounds=ROUNDS):
    """``[(output subdirectory, combos)]`` for every selected configuration."""
    return [(Path(config), build_suppression_combos(config, seeds, TARGETS, adjacency, out_targets, rounds))
            for config in configs]


def trajectory_hours(combo):
    return combo.hyperparams.rounds * TRAJECTORY_HOURS_PER_ROUND


def _run_isolated(combo, targets, output, max_parallel_clients):
    """Child-process entry point for one trajectory."""
    from results.cia_frontier.eurosat_frontier.runner import _execute

    folder = output / combo.run_name()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "suppression_config.json"
    pinned = pinned_config(combo)
    if path.exists() and json.loads(path.read_text()) != pinned:
        raise ValueError(f"Configuration mismatch: {folder}; use a separate output directory")
    atomic_json(path, pinned)
    _execute([combo], targets, output, max_parallel_clients)


def _kill_tree(pid):
    """SIGKILL a process and every descendant, collected before anything dies.

    A process group is not enough: the reproduction runner relaunches itself in an
    isolated venv in a new process group, and Ray's processes live under that one.
    """
    import psutil

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    processes = root.children(recursive=True) + [root]
    for process in processes:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(processes, timeout=30)


def _last_activity(folder):
    import os

    try:
        return max((entry.stat().st_mtime for entry in os.scandir(folder)), default=None)
    except FileNotFoundError:
        return None


def execute_shared(combos, targets, output, max_parallel_clients, *, stall_minutes=30.0,
                   worker=_run_isolated, poll_seconds=60.0):
    """Train and evaluate each trajectory unless another worker holds it.

    Same stall guard as the WP1 runner (``feature/cia-audit``): each trajectory runs in a
    child process; if its folder shows no file activity for ``stall_minutes``, the child
    and all its descendants are killed and the worker moves on, leaving the trajectory
    without ``complete.json`` for the next run of the same command (Ray occasionally
    deadlocks while forking its dashboard process).
    """
    import fcntl
    import hashlib
    import multiprocessing
    import signal
    import tempfile
    import time

    context = multiprocessing.get_context("spawn")
    stalled = []
    for combo in combos:
        folder = output / combo.run_name()
        if (folder / "complete.json").exists():
            continue
        identity = str(folder.resolve()).encode()
        lock = Path(tempfile.gettempdir()) / ("auc-frontier-" + hashlib.sha256(identity).hexdigest() + ".lock")
        with lock.open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[WP2] skip (another worker holds it): {combo.run_name()}", flush=True)
                continue
            process = context.Process(target=worker, args=(combo, targets, output, max_parallel_clients))
            process.start()
            seen, since = _last_activity(folder), time.monotonic()
            while True:
                process.join(poll_seconds)
                if not process.is_alive():
                    break
                activity = _last_activity(folder)
                if activity != seen:
                    seen, since = activity, time.monotonic()
                elif time.monotonic() - since > stall_minutes * 60:
                    print(f"[WP2] stalled for {stall_minutes} min, killing: {combo.run_name()}", flush=True)
                    _kill_tree(process.pid)
                    process.join()
                    stalled.append(combo.run_name())
                    break
            if process.exitcode not in (0, -signal.SIGKILL) and combo.run_name() not in stalled:
                raise RuntimeError(f"Trajectory failed (exit {process.exitcode}): {combo.run_name()}")
    if stalled:
        print(f"[WP2] {len(stalled)} stalled trajectories; rerun the same command to retry them", flush=True)
    return stalled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="42 is the protocol; 43 only as the declared extension")
    parser.add_argument("--adjacency", choices=("both", "in", "out"), default="both")
    parser.add_argument("--out-targets", type=int, nargs="+")
    parser.add_argument("--smoke-rounds", type=int,
                        help="smoke test only: at most 5 rounds, requires an --output outside results/")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=8)
    parser.add_argument("--stall-minutes", type=float, default=30.0,
                        help="kill a trajectory whose folder shows no file activity for this long")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    rounds = ROUNDS
    output = args.output or DEFAULT_OUTPUT
    if args.smoke_rounds is not None:
        if not 1 <= args.smoke_rounds <= 5:
            parser.error("--smoke-rounds must be between 1 and 5")
        if args.output is None or (PROJECT_ROOT / "results") in args.output.resolve().parents:
            parser.error("--smoke-rounds requires an --output outside results/")
        rounds = args.smoke_rounds
    cells = plan(args.configs, args.seeds, args.adjacency, args.out_targets, rounds)
    combos = [combo for _, cell in cells for combo in cell]
    complete = sum((output / sub / combo.run_name() / "complete.json").exists()
                   for sub, cell in cells for combo in cell)
    print(json.dumps({
        "trajectories": len(combos), "complete": complete, "execute": args.execute,
        "rounds": rounds, "targets": list(TARGETS), "seeds": args.seeds,
        "trajectory_hours_planned": round(sum(trajectory_hours(c) for c in combos), 2),
        "configs": {str(sub): [pinned_config(c) for c in cell][:1] for sub, cell in cells},
    }, indent=2))
    if args.execute:
        for sub, cell in cells:
            execute_shared(cell, list(TARGETS), output / sub, args.max_parallel_clients,
                           stall_minutes=args.stall_minutes)


if __name__ == "__main__":
    main()
