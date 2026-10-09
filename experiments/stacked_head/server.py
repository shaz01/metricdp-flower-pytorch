"""Flower ServerApp for the stacked-head experiment: one public base, one noisy aggregate per round, gated step."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp

from experiments.stacked_head import data as task_data
from experiments.stacked_head.bundle import Bundle, bundle_path, load_bundle
from experiments.stacked_head.release import ReleaseRecorder, target_records
from metricdp_pytorch.stacked_head_strategy import (
    DEFAULT_MULTIPLIERS,
    StackedHeadConfig,
    StackedHeadStrategy,
    head_class_ce,
    head_scores,
    head_theta,
)
from metricdp_pytorch.utils.runtime import runtime_config

app = ServerApp()

DEFAULT_FROZEN_CONFIG = "results/client_specific_noise/stacked_constructor_freeze.json"


def construction_from_config(config: dict[str, Any]) -> StackedHeadConfig:
    """Frozen construction for the run's risk target (from the frozen JSON), with explicit keys taking precedence."""
    risk = float(config["risk"])
    frozen: dict[str, Any] = {}
    path = config.get("frozen-config", DEFAULT_FROZEN_CONFIG)
    if path and not all(key in config for key in ("cap", "eta", "dimension", "mode")):
        key = config.get("frozen-key") or f"noisy_{int(round(risk * 100))}"
        frozen = json.loads(Path(path).read_text(encoding="utf-8"))["frozen"][key]
    pick = lambda key: config[key] if key in config else frozen[key]  # noqa: E731
    multipliers = config.get("multipliers", "")
    return StackedHeadConfig(
        cap=float(pick("cap")),
        eta=float(pick("eta")),
        dimension=int(pick("dimension")),
        risk=risk,
        mode=str(pick("mode")),
        peers=int(config.get("peers", task_data.NUM_CLIENTS - 1)),
        multipliers=tuple(float(m) for m in str(multipliers).split(",")) if multipliers else DEFAULT_MULTIPLIERS,
        noise_free=bool(config.get("noise-free", False)),
    )


class RecordingStrategy(StackedHeadStrategy):
    """Collects the clients' optional diagnostic contributions before delegating to the gated aggregation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.contributions: dict[int, np.ndarray] = {}

    def aggregate_train(self, server_round: int, replies) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        for reply in replies:
            if reply.has_error() or "diagnostics" not in reply.content:
                continue
            client_id = int(next(iter(reply.content.metric_records.values()))["client-id"])
            self.contributions[client_id] = reply.content["diagnostics"]["contribution"].numpy().astype(float)
        return super().aggregate_train(server_round, replies)


def clean(value: Any) -> Any:
    """Recursively replace NaN/inf floats with ``None`` so results serialise as strict JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def provenance() -> dict[str, Any]:
    packages = {}
    for package in ("flwr", "numpy", "torch", "datasets"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unavailable", None
    return {"python_version": platform.python_version(), "platform": platform.platform(), "library_versions": packages, "git_commit": commit, "git_dirty": dirty}


def load_run_bundle(config: dict[str, Any]) -> Bundle:
    path = bundle_path(
        Path(config["bundle-dir"]), str(config["task"]), int(config["budget"]), int(config["public-set"]),
        int(config.get("role-seed", 20261012)), int(config.get("train-seed", 20261012)), str(config.get("eval-split", "train")),
    )
    if not path.exists():
        raise FileNotFoundError(f"Public bundle {path} is missing; run the runner's prepare step first.")
    return load_bundle(path)


def summarize(control: dict[str, float], rounds: list[dict[str, float]], multipliers: tuple[float, ...]) -> dict[str, Any]:
    ce = np.array([r["eval-ce"] for r in rounds])
    accuracy = np.array([r["eval-accuracy"] for r in rounds])
    picks = np.array([r["multiplier"] for r in rounds])
    return {
        "num_rounds": len(rounds),
        "mean_gated_ce": float(ce.mean()),
        "gain_over_control": float(control["ce"] - ce.mean()),
        "gated_ce_std_across_releases": float(ce.std(ddof=1)) if len(ce) > 1 else 0.0,
        "mean_gated_accuracy": float(accuracy.mean()),
        "accuracy_delta": float(accuracy.mean() - control["accuracy"]),
        "pick_fraction": {str(m): float((picks == m).mean()) for m in multipliers},
    }


def run(grid: Grid, config: dict[str, Any]) -> dict[str, Any]:
    """Execute the configured rounds, evaluate each released model on the held-out images, and write the result JSON."""
    if str(config.get("mechanism", "stacked")) != "stacked":
        from experiments.stacked_head.baseline import run_baseline

        return run_baseline(grid, config)
    bundle = load_run_bundle(config)
    construction = construction_from_config(config)
    num_clients = task_data.NUM_CLIENTS
    rounds = int(config.get("rounds", 1))
    replicate = bool(config.get("replicate", False))
    if rounds > 1 and not replicate:
        raise ValueError("rounds > 1 requires replicate=true: the one-shot release is not a multi-round protocol.")
    basis = bundle.basis[:, : construction.dimension]
    history: list[dict[str, float]] = []
    message_log: dict[int, np.ndarray] = {}

    recorder = ReleaseRecorder(config, bundle, construction.multipliers, target_records(config, bundle, targets=range(4)) if bool(config.get("attack-records", False)) else None)
    strategy_box: list[StackedHeadStrategy] = []

    def on_replies(server_round: int, vectors: dict[int, np.ndarray], metrics: dict[str, float]) -> None:
        history.append({"round": server_round, **metrics})
        total = np.zeros(construction.dimension)
        for client_id in sorted(vectors):
            total += vectors[client_id]
        recorder.record(strategy_box[0].candidates_for(total), strategy_box[0].candidates_for(np.zeros(construction.dimension))[0])
        if bool(config.get("log-messages", False)):
            message_log[server_round] = np.stack([vectors.get(i, np.full(construction.dimension, np.nan)) for i in range(num_clients)])

    evaluations: dict[int, dict[str, float]] = {}

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        ce, accuracy = head_scores(bundle.evaluation_features, bundle.evaluation_labels, head_theta(arrays)[None])
        evaluations[server_round] = {"eval-ce": float(ce[0]), "eval-accuracy": float(accuracy[0])}
        return MetricRecord(evaluations[server_round])

    strategy = RecordingStrategy(
        construction,
        basis,
        bundle.class_gradients,
        lambda thetas: head_scores(bundle.validation_features, bundle.validation_labels, thetas)[0],
        replicate=replicate,
        reply_callback=on_replies,
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=num_clients,
        min_available_nodes=num_clients,
    )
    strategy_box.append(strategy)
    initial = ArrayRecord({key: Array(value) for key, value in bundle.state.items()})
    strategy.start(grid, initial_arrays=initial, num_rounds=rounds, train_config=ConfigRecord(), evaluate_fn=evaluate)

    control = {"ce": evaluations[0]["eval-ce"], "accuracy": evaluations[0]["eval-accuracy"]}
    released = [{**row, **evaluations[row["round"]]} for row in sorted(history, key=lambda r: r["round"])]
    multipliers = construction.multipliers
    releases = recorder.arrays()
    summary_alt, summary_fixed = recorder.summaries(control)
    result = {
        "kind": "Stacked-head one-shot release rounds (independent releases from one public base when replicate=true).",
        "run_name": config.get("run-name"),
        "config": {key: config[key] for key in sorted(config) if key not in ("frozen-config",)},
        "construction": {
            "cap": construction.cap, "eta": construction.eta, "dimension": construction.dimension, "risk": construction.risk,
            "mode": construction.mode, "peers": construction.peers, "sigma": construction.sigma, "multipliers": list(construction.multipliers),
            "noise_free": construction.noise_free,
        },
        "bundle": bundle.meta,
        "control": control,
        "rounds": released,
        "summary": summarize(control, released, construction.multipliers) if released else None,
        "summary_alt_validation": summary_alt,
        "summary_fixed_multiplier_1": summary_fixed,
        "contributions": {str(k): v.tolist() for k, v in sorted(strategy.contributions.items())} if strategy.contributions else None,
        "provenance": provenance(),
    }
    output = Path(config["output-dir"])
    output.mkdir(parents=True, exist_ok=True)
    name = str(config["run-name"])
    if releases:
        np.savez_compressed(output / f"{name}.releases.npz", multipliers=np.array(multipliers), control=np.array([control["ce"], control["accuracy"]]),
                            **{key: value.astype(np.float32) for key, value in releases.items()})
    if message_log:
        np.savez_compressed(output / f"{name}.messages.npz", rounds=np.array(sorted(message_log)), messages=np.stack([message_log[r] for r in sorted(message_log)]))
    (output / f"{name}.json").write_text(json.dumps(clean(result), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return result


@app.main()
def main(grid: Grid, context: Context) -> None:
    run(grid, runtime_config(context))
