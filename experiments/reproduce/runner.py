"""Run one end-to-end Alzheimer MRI Flower reproduction experiment.

This runner connects ``experiments.reproduce.server:app`` to
``experiments.reproduce.client:app`` through Flower's local Ray simulation.
It is independent of where it runs, so the same command works locally and on a
Lightning.ai pod.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

# Set this before importing PyTorch-dependent modules or starting CUDA. The
# isolated worker and Ray client processes inherit the environment.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from experiments.reproduce.seeds import SEED_KEYS
from metricdp_pytorch.utils.runtime import RUN_CONFIG_ENV
from metricdp_pytorch.strategy_factory import (
    AGGREGATION_METHODS,
    AGGREGATION_WEIGHTINGS,
    aggregation_token,
    EXPERIMENTAL_PRIVACY_MODES,
    PRIVACY_MODES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep the summed per-actor GPU share just under 1.0 so float rounding cannot
# make Ray refuse to schedule the last concurrent client.
_GPU_SHARE_SAFETY = 0.99



# WP6 residual clipping (metricdp_pytorch.residual_clipping_strategy), built by experiments.reproduce.server.
RESIDUAL_PRIVACY = "residual-clipping"
# WP3 private-median metric privacy (metricdp_pytorch.private_metric_strategy), built by experiments.reproduce.server.
PRIVATE_MEDIAN_PRIVACY = "private-median"
SUBSAMPLED_PRIVACY = "subsampled-global-dp"  # CIA v2 B1: global DP with Poisson client sampling

def _auto_client_gpus(*, num_clients: int, max_parallel_clients: int) -> float:
    """Split one CUDA device across the concurrently-running client actors.

    Ray overrides ``CUDA_VISIBLE_DEVICES`` inside any actor whose ``num_gpus``
    is 0, so leaving this at zero silently pins client training to the CPU even
    on a GPU host. Returns 0.0 when there is no CUDA device, which keeps CPU and
    Apple-MPS hosts on their previous behaviour.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    parallel = max(1, min(num_clients, max_parallel_clients))
    return _GPU_SHARE_SAFETY / parallel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        default="homogeneous",
        help="partition mode interpreted by the selected data module",
    )
    parser.add_argument(
        "--privacy",
        choices=PRIVACY_MODES + EXPERIMENTAL_PRIVACY_MODES + (RESIDUAL_PRIVACY, PRIVATE_MEDIAN_PRIVACY, SUBSAMPLED_PRIVACY),
        default="vanilla",
    )
    parser.add_argument(
        "--aggregation", choices=AGGREGATION_METHODS, default="fedavg"
    )
    parser.add_argument(
        "--aggregation-weighting",
        choices=AGGREGATION_WEIGHTINGS,
        default="num-examples",
        help="num-examples: size-weighted FedAvg (default); equal: every client weighs 1/n",
    )
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    residual = parser.add_argument_group("residual clipping (--privacy residual-clipping, WP6)")
    residual.add_argument("--num-slots", type=int, help="public slot count (denominator); default num-clients")
    residual.add_argument("--residual-center", choices=("previous", "zero"), default="previous")
    residual.add_argument("--residual-rho", type=float, help="fixed residual clipping norm")
    residual.add_argument("--residual-quantile", type=float, help="target quantile of a private rho")
    residual.add_argument("--residual-rho-min", type=float)
    residual.add_argument("--residual-rho-max", type=float)
    residual.add_argument("--residual-bit-noise-std", type=float)
    median = parser.add_argument_group("private-median metric privacy (--privacy private-median, WP3; also --num-slots)")
    median.add_argument("--median-quantile", type=float, help="initial private median distance q_0")
    median.add_argument("--median-quantile-min", type=float)
    median.add_argument("--median-quantile-max", type=float)
    median.add_argument("--median-bit-noise-std", type=float, help="default: 10%% of the round's GDP budget")
    parser.add_argument("--proximal-mu", type=float,
                        help="FedProx mu (fedprox only; default 0.5, the paper value)")
    parser.add_argument("--proximal-normalization", choices=("mean", "sum"),
                        help="proximal term: mean squared difference (default) or the "
                             "FedProx paper's sum of squares")
    parser.add_argument("--consensus-kl-beta", type=float, default=0.0,
                        help="weight of KL(p_global || p_local) on local data (0 = off)")
    parser.add_argument("--sampling-rate", type=float,
                        help="subsampled-global-dp: probability q that a client joins a round (denominator q N)")
    parser.add_argument("--require-all-clients", action="store_true",
                        help="stop the run if a client fails or does not reply in any round (default: Flower "
                             "aggregates the remaining clients)")
    parser.add_argument("--local-steps", type=int, default=0,
                        help="fixed optimizer steps per round instead of --local-epochs (0 = off)")
    parser.add_argument("--contribution-cap", type=int, default=0,
                        help="CIA v2 B5: at most this many training records per client per round (0 = off)")
    parser.add_argument("--logit-adjustment-tau", type=float, default=0.0,
                        help="CIA v2 B5: logit-adjusted local loss with the client's label prior (0 = off)")
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "cosine"),
        default="none",
        help="none keeps a fixed --learning-rate; cosine decays it per round",
    )
    parser.add_argument("--seed", type=int, required=True)
    for part, what in (("layout", "data layout"), ("init", "initial model"),
                       ("client", "client round seeds"), ("noise", "server noise draws")):
        parser.add_argument(f"--{part}-seed", type=int,
                            help=f"seed for the {what} (default --seed; CIA roadmap v2)")
    parser.add_argument("--noise-multiplier", type=float, required=True)
    parser.add_argument("--clipping-norm", type=float, required=True)
    parser.add_argument("--influence-fraction", type=float,
                        help="influence-noise only: orientation fraction f in [0, 1]")
    parser.add_argument("--influence-cap", type=float,
                        help="influence-noise only: fixed L2 cap on leave-one-out influence")
    parser.add_argument(
        "--partition-profile",
        default="auto",
        help="optional profile interpreted by the selected data module",
    )
    parser.add_argument(
        "--client-weights",
        default="",
        help="comma-separated non-IID quantity weights, one per client",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.5,
        help="positive concentration for --partition dirichlet (smaller means stronger skew)",
    )
    parser.add_argument("--initialization-epochs", type=int, default=20)
    parser.add_argument("--initialization-batch-size", type=int, default=32)
    parser.add_argument("--initialization-learning-rate", type=float, default=0.001)
    parser.add_argument("--max-client-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--data-module",
        default="experiments.reproduce.dataset.alzheimer:create_data_module",
        help="pluggable dataset factory in package.module:factory format",
    )
    parser.add_argument(
        "--model-module",
        required=True,
        help="pluggable model factory in package.module:factory format",
    )
    parser.add_argument(
        "--data-cache-dir",
        default="",
        help="optional cache directory passed to the data-module factory",
    )
    parser.add_argument(
        "--target-partition-id",
        type=int,
        help="optional target partition interpreted by CIA data modules",
    )
    parser.add_argument(
        "--shadow-fraction",
        type=float,
        help="optional target-shadow fraction interpreted by CIA data modules",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        help="optional client train fraction interpreted by data modules",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results-reproduce-paper/reproduce"))
    parser.add_argument("--run-name")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--log-loo-residuals",
        action="store_true",
        help="log each client's update minus the leave-one-out mean update (norm, before clipping)",
    )
    parser.add_argument(
        "--log-noise-diagnostics",
        action="store_true",
        help="log the released-update norm and per-block realized noise energy (DP strategies)",
    )
    parser.add_argument(
        "--checkpoint-rounds",
        nargs="+",
        type=int,
        default=(),
        help="server rounds whose aggregated models should be saved",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one round/epoch with capped data (32 client, 64 server samples)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        default=2,
        help=(
            "cap simultaneous Ray actors to control memory use. Also sets each "
            "actor's automatically calculated GPU share. Ray's num_gpus is only "
            "bookkeeping -- it never reserves or limits VRAM, so "
            "raising this past what the device holds will OOM rather than queue. "
            "Each actor cost ~1.06 GiB of VRAM on PaperCNN, i.e. roughly 42 actors "
            "on a 46 GiB card; a larger model lowers that ceiling"
        ),
    )
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    return parser


def _project_defaults() -> dict[str, Any]:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    return dict(project["tool"]["flwr"]["app"]["config"])


def _validate_private_median(args: argparse.Namespace) -> None:
    bounds = (args.median_quantile, args.median_quantile_min, args.median_quantile_max)
    if args.privacy != PRIVATE_MEDIAN_PRIVACY:
        if any(value is not None for value in (*bounds, args.median_bit_noise_std)):
            raise ValueError("--median-* flags apply only to --privacy private-median.")
        return
    if args.aggregation_weighting != "equal":
        raise ValueError("private-median requires --aggregation-weighting equal.")
    if None in bounds:
        raise ValueError("private-median needs --median-quantile, --median-quantile-min and --median-quantile-max.")
    if not 0 < args.median_quantile_min <= args.median_quantile <= args.median_quantile_max:
        raise ValueError("Require 0 < --median-quantile-min <= --median-quantile <= --median-quantile-max.")
    if args.num_slots is not None and args.num_slots < args.num_clients:
        raise ValueError("num-slots must be at least num-clients.")
    if args.noise_multiplier <= 0:
        raise ValueError("private-median needs a positive noise multiplier.")


def _validate_subsampled(args: argparse.Namespace) -> None:
    if args.privacy != SUBSAMPLED_PRIVACY:
        if args.sampling_rate is not None:
            raise ValueError("--sampling-rate applies only to --privacy subsampled-global-dp.")
        return
    if args.aggregation_weighting != "equal":
        raise ValueError("subsampled-global-dp requires --aggregation-weighting equal.")
    if args.sampling_rate is None or not 0 < args.sampling_rate <= 1:
        raise ValueError("subsampled-global-dp needs --sampling-rate in (0, 1].")
    if args.noise_multiplier <= 0:
        raise ValueError("subsampled-global-dp needs a positive noise multiplier.")


def _validate_residual(args: argparse.Namespace) -> None:
    slots = None if args.privacy == PRIVATE_MEDIAN_PRIVACY else args.num_slots
    options = (slots, args.residual_rho, args.residual_quantile, args.residual_rho_min,
               args.residual_rho_max, args.residual_bit_noise_std)
    if args.privacy != RESIDUAL_PRIVACY:
        if any(value is not None for value in options) or args.residual_center != "previous":
            raise ValueError("--num-slots and --residual-* flags apply only to --privacy residual-clipping.")
        return
    if args.aggregation_weighting != "equal":
        raise ValueError("residual-clipping requires --aggregation-weighting equal.")
    if args.num_slots is not None and args.num_slots < args.num_clients:
        raise ValueError("num-slots must be at least num-clients.")
    if (args.residual_rho is None) == (args.residual_quantile is None):
        raise ValueError("residual-clipping needs exactly one of --residual-rho and --residual-quantile.")
    if args.residual_quantile is not None and None in (
        args.residual_rho_min, args.residual_rho_max, args.residual_bit_noise_std
    ):
        raise ValueError("--residual-quantile needs --residual-rho-min, --residual-rho-max and --residual-bit-noise-std.")
    if args.noise_multiplier <= 0:
        raise ValueError("residual-clipping needs a positive noise multiplier.")


def _validate(args: argparse.Namespace) -> None:
    _validate_residual(args)
    _validate_private_median(args)
    _validate_subsampled(args)
    if args.num_clients < 2:
        raise ValueError("num-clients must be at least two.")
    if not 0.0 < args.fraction_evaluate <= 1.0:
        raise ValueError("fraction-evaluate must be in (0, 1].")
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("rounds and local-epochs must be positive.")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative.")
    if args.aggregation != "fedprox" and (
        args.proximal_mu is not None or args.proximal_normalization is not None
    ):
        raise ValueError("--proximal-* flags apply only to --aggregation fedprox.")
    if args.proximal_mu is not None and not (math.isfinite(args.proximal_mu) and args.proximal_mu >= 0):
        raise ValueError("proximal-mu must be finite and non-negative.")
    if not (math.isfinite(args.consensus_kl_beta) and args.consensus_kl_beta >= 0):
        raise ValueError("consensus-kl-beta must be finite and non-negative.")
    if args.local_steps < 0:
        raise ValueError("local-steps must be non-negative.")
    if not math.isfinite(args.dirichlet_alpha) or args.dirichlet_alpha <= 0:
        raise ValueError("dirichlet-alpha must be positive and finite.")
    if len(set(args.checkpoint_rounds)) != len(args.checkpoint_rounds):
        raise ValueError("checkpoint-rounds must not contain duplicates.")
    # Round 0 is the initial (pre-training) global model, evaluated by Strategy.start().
    if any(round_number < 0 or round_number > args.rounds for round_number in args.checkpoint_rounds):
        raise ValueError("checkpoint-rounds must be between 0 and rounds.")
    if args.batch_size < 1 or args.initialization_batch_size < 1:
        raise ValueError("batch sizes must be positive.")
    if args.initialization_epochs < 1:
        raise ValueError("initialization-epochs must be positive.")
    if args.max_parallel_clients < 1:
        raise ValueError("max-parallel-clients must be positive.")
    if args.client_cpus <= 0:
        raise ValueError("client-cpus must be positive.")
    if args.target_partition_id is not None and not (
        0 <= args.target_partition_id < args.num_clients
    ):
        raise ValueError("target-partition-id must identify a configured client.")
    for name, fraction in (
        ("shadow-fraction", args.shadow_fraction),
        ("train-fraction", args.train_fraction),
    ):
        if fraction is not None and not 0.0 < fraction < 1.0:
            raise ValueError(f"{name} must be in (0, 1).")
    influence_given = args.influence_fraction is not None or args.influence_cap is not None
    if args.privacy == "influence-noise":
        if args.influence_fraction is None or args.influence_cap is None:
            raise ValueError("influence-noise requires --influence-fraction and --influence-cap.")
        if not 0.0 <= args.influence_fraction <= 1.0:
            raise ValueError("influence-fraction must be in [0, 1].")
        if not math.isfinite(args.influence_cap) or args.influence_cap <= 0:
            raise ValueError("influence-cap must be finite and positive.")
    elif influence_given:
        raise ValueError("--influence-* flags apply only to --privacy influence-noise.")
    if args.client_weights:
        weights = [part.strip() for part in args.client_weights.split(",") if part.strip()]
        if len(weights) != args.num_clients:
            raise ValueError("client-weights must contain one value per client.")
        if any(float(weight) < 0 for weight in weights):
            raise ValueError("client-weights must be non-negative.")


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI choices into the project's Flower run-config defaults."""
    _validate(args)
    rounds = 1 if args.smoke else args.rounds
    local_epochs = 1 if args.smoke else args.local_epochs
    initialization_epochs = 1 if args.smoke else args.initialization_epochs
    max_client_samples = (
        32 if args.smoke and args.max_client_samples == 0 else args.max_client_samples
    )
    max_test_samples = (
        64 if args.smoke and args.max_test_samples == 0 else args.max_test_samples
    )
    partition_token = args.partition
    if args.partition == "dirichlet":
        encoded_alpha = f"{args.dirichlet_alpha:g}".replace(".", "p")
        partition_token = f"{partition_token}__alpha-{encoded_alpha}"
    run_name = args.run_name or (
        f"paper__{partition_token}__{args.privacy}__"
        f"{aggregation_token(args.aggregation, args.aggregation_weighting)}"
        f"__clients-{args.num_clients}__seed-{args.seed}"
    )
    output_dir = args.output_dir.resolve()
    cache_dir = str(args.data_cache_dir).strip()
    if cache_dir:
        cache_dir = str(Path(cache_dir).expanduser().resolve())

    config = _project_defaults()
    config.update(
        {
            "partition-mode": args.partition,
            "partition-profile": args.partition_profile,
            "client-weights": args.client_weights,
            "dirichlet-alpha": args.dirichlet_alpha,
            "privacy": args.privacy,
            "aggregation": args.aggregation,
            "aggregation-weighting": args.aggregation_weighting,
            "seed": args.seed,
            "num-clients": args.num_clients,
            "fraction-evaluate": args.fraction_evaluate,
            "num-server-rounds": rounds,
            "local-epochs": local_epochs,
            "learning-rate": args.learning_rate,
            "weight-decay": args.weight_decay,
            "lr-schedule": args.lr_schedule,
            "batch-size": args.batch_size,
            "initialization-batch-size": args.initialization_batch_size,
            "initialization-epochs": initialization_epochs,
            "initialization-learning-rate": args.initialization_learning_rate,
            "max-client-samples": max_client_samples,
            "max-test-samples": max_test_samples,
            "noise-multiplier": args.noise_multiplier,
            "clipping-norm": args.clipping_norm,
            "data-module": args.data_module,
            "model-module": args.model_module,
            "data-cache-dir": cache_dir,
            "output-dir": str(output_dir),
            "run-name": run_name,
            "save-model": args.save_model,
            "checkpoint-rounds": sorted(args.checkpoint_rounds),
        }
    )
    if args.privacy == "influence-noise":
        config["influence-fraction"] = args.influence_fraction
        config["influence-cap"] = args.influence_cap
    if args.target_partition_id is not None:
        config["target-partition-id"] = args.target_partition_id
    if args.shadow_fraction is not None:
        config["shadow-fraction"] = args.shadow_fraction
    if args.train_fraction is not None:
        config["train-fraction"] = args.train_fraction
    if args.log_loo_residuals:
        config["log-loo-residuals"] = True
    if args.log_noise_diagnostics:
        config["log-noise-diagnostics"] = True
    for key in SEED_KEYS:
        value = getattr(args, key.replace("-", "_"))
        if value is not None:
            config[key] = value
    if args.privacy == RESIDUAL_PRIVACY:
        config["num-slots"] = args.num_slots or args.num_clients
        config["residual-center"] = args.residual_center
        for key in ("rho", "quantile", "rho-min", "rho-max", "bit-noise-std"):
            value = getattr(args, "residual_" + key.replace("-", "_"))
            if value is not None:
                config["residual-" + key] = value
    if args.privacy == SUBSAMPLED_PRIVACY:
        config["sampling-rate"] = args.sampling_rate
    if args.privacy == PRIVATE_MEDIAN_PRIVACY:
        config["num-slots"] = args.num_slots or args.num_clients
        for key in ("quantile", "quantile-min", "quantile-max", "bit-noise-std"):
            value = getattr(args, "median_" + key.replace("-", "_"))
            if value is not None:
                config["median-" + key] = value
    if args.proximal_mu is not None:
        config["proximal-mu"] = args.proximal_mu
    if args.proximal_normalization is not None:
        config["proximal-normalization"] = args.proximal_normalization
    if args.consensus_kl_beta:
        config["consensus-kl-beta"] = args.consensus_kl_beta
    if args.local_steps:
        config["local-steps"] = args.local_steps
    if args.contribution_cap:
        config["contribution-cap"] = args.contribution_cap
    if args.logit_adjustment_tau:
        config["logit-adjustment-tau"] = args.logit_adjustment_tau
    if args.require_all_clients:
        config["require-all-clients"] = True
    return config


def _run_worker(
    config: dict[str, Any],
    *,
    num_supernodes: int,
    max_parallel_clients: int,
    client_cpus: float,
    client_gpus: float,
    verbose: bool,
) -> None:
    os.environ[RUN_CONFIG_ENV] = json.dumps(config)

    from flwr.simulation import run_simulation

    from experiments.reproduce.client import app as client_app
    from experiments.reproduce.server import app as server_app

    parallel_clients = min(num_supernodes, max_parallel_clients)
    total_cpus = max(1, math.ceil(parallel_clients * client_cpus))
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_supernodes,
        backend_config={
            "init_args": {"num_cpus": total_cpus},
            "client_resources": {
                "num_cpus": client_cpus,
                "num_gpus": client_gpus,
            },
        },
        verbose_logging=verbose,
    )


def _launch_isolated(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Re-exec through a no-space venv path so Ray works from spaced paths."""
    with tempfile.TemporaryDirectory(prefix="paper-reproduction-") as temporary:
        temporary_dir = Path(temporary)
        config_path = temporary_dir / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        venv_link = temporary_dir / "venv"
        os.symlink(sys.prefix, venv_link, target_is_directory=True)
        # sys.executable's path relative to sys.prefix gives the venv's own
        # interpreter regardless of platform layout (POSIX: bin/python;
        # Windows: Scripts/python.exe) -- hardcoding bin/python here always
        # raised FileNotFoundError on Windows, since this venv has no bin/
        # directory at all.
        # Deliberately NOT .resolve()d: uv-managed venvs symlink their own
        # bin/python straight through to a shared interpreter store outside
        # the venv (e.g. ~/.local/share/uv/python/...), so resolving before
        # taking the relative path follows that symlink past sys.prefix
        # entirely and relative_to() raises ValueError. sys.executable is
        # already venv-relative as reported (e.g. ".venv/bin/python3") --
        # resolving is neither needed nor safe here.
        python_relative = Path(sys.executable).relative_to(Path(sys.prefix))
        command = [
            str(venv_link / python_relative),
            "-m",
            "experiments.reproduce.runner",
            "--worker-config",
            str(config_path),
            "--num-clients",
            str(args.num_clients),
            "--max-parallel-clients",
            str(args.max_parallel_clients),
            "--client-cpus",
            str(args.client_cpus),
            "--seed",
            str(args.seed),
            "--noise-multiplier",
            str(args.noise_multiplier),
            "--clipping-norm",
            str(args.clipping_norm),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--model-module",
            str(args.model_module),
        ]
        if args.verbose:
            command.append("--verbose")
        child_env = os.environ.copy()
        child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        child_env.setdefault("PYTHONHASHSEED", "0")
        subprocess.run(command, cwd=PROJECT_ROOT, env=child_env, check=True)


def _print_result(config: dict[str, Any]) -> None:
    result_path = Path(config["output-dir"]) / f"{config['run-name']}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = result["server_evaluate_metrics"]
    completed = [int(round_number) for round_number in history if int(round_number) > 0]
    final_metrics = history[str(max(completed))] if completed else history.get("0", {})
    print(f"Result: {result_path}")
    if final_metrics:
        print(f"Final server metrics: {final_metrics}")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.worker_config is not None:
        config = json.loads(args.worker_config.read_text(encoding="utf-8"))
        _run_worker(
            config,
            num_supernodes=args.num_clients,
            max_parallel_clients=args.max_parallel_clients,
            client_cpus=args.client_cpus,
            client_gpus=_auto_client_gpus(
                num_clients=args.num_clients,
                max_parallel_clients=args.max_parallel_clients,
            ),
            verbose=args.verbose,
        )
        return

    try:
        config = build_run_config(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; remove --dry-run to start the simulation.")
        return
    Path(config["output-dir"]).mkdir(parents=True, exist_ok=True)
    _launch_isolated(args, config)
    _print_result(config)


if __name__ == "__main__":
    main()
