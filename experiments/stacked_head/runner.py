"""Run the stacked-head experiment: prepare public bundles, launch Flower simulations, run matrices and attack checks.

Subcommands
  prepare  train/cache the public base for one (task, budget, public-set)
  run      one cell: N independent one-shot releases (replicate rounds) through the real ServerApp/ClientApp
  matrix   a preset grid of cells, resumable (finished result files are skipped)
  attack   IN/OUT known-alternative attack check on the real message path (three runs + analysis)

Backends: ``ray`` (Flower simulation, the repo's standard) or ``inprocess`` (same server/client code, sequential).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from metricdp_pytorch.utils.runtime import RUN_CONFIG_ENV  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_SEED_BASE = 20261012
COHORTS = ("A", "B", "C", "D")
DEFAULT_RESULTS = "results/stacked_head"
DEFAULT_BUNDLES = ".stacked_head_cache/bundles"
FROZEN = "results/client_specific_noise/stacked_constructor_freeze.json"
PRESETS = {
    "smoke": {"tasks": ["kmnist_classes0to3"], "budgets": [32], "sets": [0], "cohorts": ["A"], "risks": [0.65], "rounds": 16},
    "gate": {"tasks": ["kmnist_classes0to3", "kmnist_classes4to7"], "budgets": [32], "sets": [0, 1, 2], "cohorts": ["A", "B"], "risks": [0.65, 0.8], "rounds": 512},
    "budgets": {"tasks": ["kmnist_classes0to3", "kmnist_classes4to7"], "budgets": [32, 128], "sets": [0, 1, 2], "cohorts": ["A", "B"], "risks": [0.65, 0.8], "rounds": 512},
    # Protocol experiments/stacked_head/protocols/2026-10-09_public_set_sweep_and_transfer.md
    "sweep32": {"tasks": ["kmnist_classes0to3", "kmnist_classes4to7"], "budgets": [32], "sets": list(range(30)), "cohorts": ["A"], "risks": [0.65, 0.8], "rounds": 256, "tag": "sweep"},
    "transfer32": {"tasks": ["mnist_classes0to3", "mnist_classes4to7", "fmnist_classes4to7"], "budgets": [32], "sets": list(range(10)), "cohorts": ["A"], "risks": [0.65, 0.8], "rounds": 256, "tag": "transfer"},
    # Protocol experiments/stacked_head/protocols/2026-10-09_accuracy_aware_gate.md: fresh public sets AND fresh (test-split) evaluation images.
    "confirm_sweep": {"tasks": ["kmnist_classes0to3", "kmnist_classes4to7"], "budgets": [32], "sets": list(range(30, 60)), "cohorts": ["A"], "risks": [0.65, 0.8], "rounds": 256, "tag": "confirmsweep", "eval_split": "test"},
    "confirm_transfer": {"tasks": ["mnist_classes0to3", "mnist_classes4to7", "fmnist_classes4to7"], "budgets": [32], "sets": list(range(10, 20)), "cohorts": ["A"], "risks": [0.65, 0.8], "rounds": 256, "tag": "confirmtransfer", "eval_split": "test"},
    # Protocol 2026-10-09_budgets_and_cohorts.md
    "budget_map": {"tasks": ["kmnist_classes0to3", "kmnist_classes4to7", "mnist_classes0to3", "mnist_classes4to7", "fmnist_classes4to7"], "budgets": [128, 512], "sets": list(range(10)), "cohorts": ["A"], "risks": [0.65, 0.8], "rounds": 256, "tag": "budgetmap"},
    "cohorts": {"tasks": ["kmnist_classes0to3", "kmnist_classes4to7"], "budgets": [32], "sets": list(range(10)), "cohorts": ["B", "C", "D"], "risks": [0.65, 0.8], "rounds": 256, "tag": "cohorts"},
}


def probe_seed(budget: int, public_set: int, cohort: str, risk: float) -> int:
    """Seed of the research probe's noise stream for this cell (used only for ``noise-source=probe``)."""
    return PROBE_SEED_BASE + 1000 * budget + 100 * public_set + 10 * COHORTS.index(cohort) + int(risk * 100)


def run_name(task: str, budget: int, public_set: int, cohort: str, risk: float, tag: str) -> str:
    return f"{task}_b{budget}_s{public_set}_{cohort}_q{int(round(risk * 100))}_{tag}"


def _add_cell_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="kmnist_classes0to3")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--public-set", type=int, default=0)
    parser.add_argument("--cohort", choices=COHORTS, default="A")
    parser.add_argument("--risk", type=float, default=0.65, help="calibrated conditional attack AUC target in (0.5, 1)")
    parser.add_argument("--rounds", type=int, default=512, help="independent releases (replicate) or 1 for a single one-shot release")
    parser.add_argument("--single-release", action="store_true", help="one real one-shot release (rounds=1, no replication)")
    parser.add_argument("--noise-source", choices=("independent", "probe"), default="independent")
    parser.add_argument("--seed", type=int, default=42, help="base seed of the per-client independent noise streams")
    parser.add_argument("--role-seed", type=int, default=20261012)
    parser.add_argument("--eval-split", choices=("train", "test"), default="train", help="held-out evaluation images: the task's train-split role, or a fixed subset of the dataset's test split")
    parser.add_argument("--train-seed", type=int, default=20261012)
    parser.add_argument("--frozen-config", default=FROZEN)
    parser.add_argument("--frozen-key", default="", help="entry of the frozen config to use (default: the one for --risk, e.g. noisy_65)")
    parser.add_argument("--bundle-dir", default=DEFAULT_BUNDLES)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--tag", default=None, help="result-name suffix (default: the noise source)")
    parser.add_argument("--diagnostics", action="store_true", help="clients also return their noise-free contribution (audit only; leaks private-dependent values)")
    parser.add_argument("--log-messages", action="store_true", help="save every client message of every round next to the result")
    parser.add_argument("--absent-clients", default="", help="comma-separated client ids that send only their noise share (dummy world)")
    parser.add_argument("--noise-free", action="store_true")
    parser.add_argument("--attack-records", action="store_true", help="also record the own-records membership statistic for targets 0-3 (attacker simulation; needs the clients' data on the server host)")
    parser.add_argument("--backend", choices=("ray", "inprocess"), default="ray")
    parser.add_argument("--max-parallel-clients", type=int, default=8)
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    rounds = 1 if args.single_release else int(args.rounds)
    tag = args.tag or ("single" if args.single_release else args.noise_source)
    config: dict[str, Any] = {
        "task": args.task,
        "budget": int(args.budget),
        "public-set": int(args.public_set),
        "cohort": args.cohort,
        "risk": float(args.risk),
        "rounds": rounds,
        "replicate": rounds > 1,
        "noise-source": args.noise_source,
        "seed": int(args.seed),
        "probe-noise-seed": probe_seed(args.budget, args.public_set, args.cohort, args.risk),
        "probe-noise-rounds": rounds,
        "role-seed": int(args.role_seed),
        "train-seed": int(args.train_seed),
        "frozen-config": args.frozen_config,
        "frozen-key": args.frozen_key,
        "attack-records": bool(getattr(args, "attack_records", False)),
        "bundle-dir": str(args.bundle_dir),
        "output-dir": str(args.output_dir),
        "eval-split": args.eval_split,
        "extra-validation-sizes": "128",
        "run-name": run_name(args.task, args.budget, args.public_set, args.cohort, args.risk, tag),
        "cell-key": run_name(args.task, args.budget, args.public_set, args.cohort, args.risk, ""),
        "diagnostics": bool(args.diagnostics),
        "log-messages": bool(args.log_messages),
        "absent-clients": args.absent_clients,
        "noise-free": bool(args.noise_free),
        "model-module": "experiments.stacked_head.cnn:create_model",
    }
    if not 0.5 < config["risk"] < 1:
        raise ValueError("--risk must lie in (0.5, 1).")
    return config


def prepare(config: dict[str, Any], *, force: bool = False) -> Path:
    from experiments.stacked_head.bundle import prepare_bundle

    return prepare_bundle(
        config["task"], config["budget"], config["public-set"], Path(config["bundle-dir"]),
        role_seed=config["role-seed"], train_seed=config["train-seed"], eval_split=config.get("eval-split", "train"), force=force,
    )


def _run_worker(config: dict[str, Any], *, max_parallel_clients: int, client_cpus: float, verbose: bool) -> None:
    os.environ[RUN_CONFIG_ENV] = json.dumps(config)
    from flwr.simulation import run_simulation

    from experiments.stacked_head.client import app as client_app
    from experiments.stacked_head.server import app as server_app

    num_supernodes = 8
    parallel = min(num_supernodes, max_parallel_clients)
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_supernodes,
        backend_config={"init_args": {"num_cpus": max(1, math.ceil(parallel * client_cpus))}, "client_resources": {"num_cpus": client_cpus, "num_gpus": 0.0}},
        verbose_logging=verbose,
    )


def _launch_isolated(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Re-exec through a no-space venv path so Ray works from checkouts whose path contains spaces."""
    with tempfile.TemporaryDirectory(prefix="stacked-head-") as temporary:
        directory = Path(temporary)
        config_path = directory / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        link = directory / "venv"
        os.symlink(sys.prefix, link, target_is_directory=True)
        interpreter = link / Path(sys.executable).relative_to(Path(sys.prefix))
        command = [str(interpreter), "-m", "experiments.stacked_head.runner", "--worker-config", str(config_path),
                   "--max-parallel-clients", str(args.max_parallel_clients), "--client-cpus", str(args.client_cpus)]
        if args.verbose:
            command.append("--verbose")
        environment = os.environ.copy()
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        environment.setdefault("PYTHONHASHSEED", "0")
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def execute(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    """Prepare the public bundle, run the simulation on the chosen backend, return the result path."""
    prepare(config)
    result_path = Path(config["output-dir"]) / f"{config['run-name']}.json"
    if args.backend == "inprocess":
        from experiments.stacked_head.local import LocalGrid
        from experiments.stacked_head.server import run

        run(LocalGrid(8, config), config)
    else:
        Path(config["output-dir"]).mkdir(parents=True, exist_ok=True)
        _launch_isolated(args, config)
    return result_path


def _print_summary(result_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = result["summary"]
    print(f"Result: {result_path}")
    if summary:
        print(f"  control CE {result['control']['ce']:.5f}  gated CE {summary['mean_gated_ce']:.5f}  gain {summary['gain_over_control']:+.5f}  "
              f"accuracy delta {summary['accuracy_delta']:+.4f}  releases {summary['num_rounds']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-parallel-clients", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--client-cpus", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command")
    prepare_parser = sub.add_parser("prepare", help="train and cache the public base")
    _add_cell_arguments(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true")
    run_parser = sub.add_parser("run", help="run one cell")
    _add_cell_arguments(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")
    matrix = sub.add_parser("matrix", help="run a preset grid of cells (resumable)")
    _add_cell_arguments(matrix)
    matrix.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    matrix.add_argument("--overwrite", action="store_true", help="re-run cells whose result file already exists (results are deterministic)")
    matrix.add_argument("--shard", default="0/1", help="I/N: run only every N-th (task, budget, public-set) group, so parallel processes never share a bundle")
    matrix.add_argument("--dry-run", action="store_true")
    frontier = sub.add_parser("frontier", help="model-only attack (IN/OUT) and utility along a range of risk targets; see protocols/")
    _add_cell_arguments(frontier)
    frontier.add_argument("--shard", default="0/1")
    frontier.add_argument("--overwrite", action="store_true")
    frontier.add_argument("--dry-run", action="store_true")
    attack = sub.add_parser("attack", help="IN/OUT known-alternative attack check on the real message path")
    _add_cell_arguments(attack)
    attack.add_argument("--target", type=int, default=3, help="client whose contribution the attacker tests for")
    return parser


def cmd_matrix(args: argparse.Namespace) -> None:
    preset = PRESETS[args.preset]
    index, count = (int(part) for part in args.shard.split("/"))
    if not 0 <= index < count:
        raise ValueError("--shard must look like I/N with 0 <= I < N.")
    group = -1
    for task in preset["tasks"]:
        for budget in preset["budgets"]:
            for public_set in preset["sets"]:
                group += 1
                if group % count != index:
                    continue
                for cohort in preset["cohorts"]:
                    for risk in preset["risks"]:
                        cell = argparse.Namespace(**{**vars(args), "task": task, "budget": budget, "public_set": public_set, "cohort": cohort, "risk": risk,
                                                     "rounds": preset["rounds"], "tag": preset.get("tag", args.tag),
                                                     "eval_split": preset.get("eval_split", args.eval_split)})
                        config = build_run_config(cell)
                        path = Path(config["output-dir"]) / f"{config['run-name']}.json"
                        if path.exists() and not args.overwrite:
                            print(f"skip (done): {path.name}", flush=True)
                            continue
                        print(f"run: {config['run-name']}", flush=True)
                        if args.dry_run:
                            continue
                        _print_summary(execute(cell, config))


FRONTIER = {"tasks": ["kmnist_classes0to3", "kmnist_classes4to7"], "budget": 32, "sets": [0, 1, 2, 3, 4], "cohort": "A", "risks": [0.55, 0.65, 0.8, 0.9, 0.95, "nf"],
            "targets": [0, 1, 2, 3], "rounds": 256, "frozen_key": "noisy_65"}


def frontier_cells() -> list[tuple[str, int, object, str, str]]:
    """(task, public set, risk, world tag, absent-clients) for the protocol's IN world and one OUT world per target."""
    cells = []
    for task in FRONTIER["tasks"]:
        for public_set in FRONTIER["sets"]:
            for risk in FRONTIER["risks"]:
                stem = "frontiernf" if risk == "nf" else "frontier"
                cells.append((task, public_set, risk, f"{stem}in", ""))
                cells.extend((task, public_set, risk, f"{stem}out{t}", str(t)) for t in FRONTIER["targets"])
    return cells


def cmd_frontier(args: argparse.Namespace) -> None:
    index, count = (int(part) for part in args.shard.split("/"))
    group = -1
    last = None
    for task, public_set, risk, tag, absent in frontier_cells():
        if (task, public_set) != last:
            group, last = group + 1, (task, public_set)
        if group % count != index:
            continue
        noise_free = risk == "nf"
        cell = argparse.Namespace(**{**vars(args), "task": task, "budget": FRONTIER["budget"], "public_set": public_set, "cohort": FRONTIER["cohort"],
                                     "risk": 0.65 if noise_free else risk, "rounds": 1 if noise_free else FRONTIER["rounds"], "single_release": noise_free,
                                     "tag": tag, "absent_clients": absent, "noise_free": noise_free, "frozen_key": FRONTIER["frozen_key"], "attack_records": True})
        config = build_run_config(cell)
        path = Path(config["output-dir"]) / f"{config['run-name']}.json"
        if path.exists() and not args.overwrite:
            print(f"skip (done): {path.name}", flush=True)
            continue
        print(f"run: {config['run-name']}", flush=True)
        if args.dry_run:
            continue
        _print_summary(execute(cell, config))


def cmd_attack(args: argparse.Namespace) -> None:
    from experiments.stacked_head.attack import evaluate_attack

    common = dict(vars(args))
    contributions = argparse.Namespace(**{**common, "noise_free": True, "diagnostics": True, "single_release": True, "tag": f"attack_contrib", "log_messages": False, "absent_clients": ""})
    world_in = argparse.Namespace(**{**common, "tag": "attack_in", "log_messages": True, "diagnostics": False, "absent_clients": "", "single_release": False})
    world_out = argparse.Namespace(**{**common, "tag": "attack_out", "log_messages": True, "diagnostics": False, "absent_clients": str(args.target), "single_release": False})
    paths = [execute(world, build_run_config(world)) for world in (contributions, world_in, world_out)]
    report = evaluate_attack(*paths, target=args.target)
    out = Path(args.output_dir) / f"{paths[1].stem.replace('_attack_in', '')}_attack_t{args.target}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.worker_config is not None:
        _run_worker(json.loads(args.worker_config.read_text(encoding="utf-8")), max_parallel_clients=args.max_parallel_clients, client_cpus=args.client_cpus, verbose=args.verbose)
        return
    if args.command is None:
        parser.error("choose a subcommand: prepare, run, matrix, attack")
    if args.command == "matrix":
        return cmd_matrix(args)
    if args.command == "attack":
        return cmd_attack(args)
    if args.command == "frontier":
        return cmd_frontier(args)
    try:
        config = build_run_config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.command == "prepare":
        print(prepare(config, force=args.force))
        return
    print(json.dumps(config, indent=1), flush=True)
    if args.dry_run:
        print("Dry run only; remove --dry-run to start.")
        return
    _print_summary(execute(args, config))


if __name__ == "__main__":
    main()
