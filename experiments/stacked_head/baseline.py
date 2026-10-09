"""Server-side-DP baselines on the same public base, data and releases as the stacked construction.

The federated model is the 4-class linear head on the frozen public body (two arrays, ``head.weight`` and ``head.bias``).
Each client runs a few full-batch gradient steps on its own records, the server applies the repository's REAL strategies
(``make_strategy("fedavg", privacy, ...)``: vanilla FedAvg, Flower's fixed-clipping global DP, or the paper's metric-privacy
calibration) and then the same public validation gate chooses a multiplier on the aggregated update. Releases, attack worlds
and held-out scoring are recorded by the shared ``ReleaseRecorder``.

Dummy (OUT-world) clients send the base head plus a 1e-6 perturbation: Flower's clipping divides by the update norm, so an
exactly zero update is not allowed, and 1e-6 is far below every noise scale used here.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from scipy.special import ndtr, ndtri

from experiments.stacked_head import data as task_data
from experiments.stacked_head.bundle import Bundle, bundle_path, load_bundle
from experiments.stacked_head.cnn import Net, embed
from experiments.stacked_head.release import ReleaseRecorder, target_records
from metricdp_pytorch.dp_diagnostics import model_update_l2_norm
from metricdp_pytorch.stacked_head_strategy import head_scores, head_theta
from metricdp_pytorch.strategy_factory import make_strategy

MECHANISMS = ("stacked", "global-dp", "metric-privacy", "vanilla")
BASELINES = MECHANISMS[1:]
BASELINE_MULTIPLIERS = (0.0, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0)
DUMMY_EPSILON = 1e-6
WEIGHT_KEY, BIAS_KEY = "head.weight", "head.bias"


def local_train(features: np.ndarray, labels: np.ndarray, weight: np.ndarray, bias: np.ndarray, lr: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch softmax-CE gradient descent on a linear head (float64)."""
    w, b = weight.astype(float).copy(), bias.astype(float).copy()
    onehot, n = np.eye(4)[labels], len(labels)
    for _ in range(steps):
        z = features @ w.T + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - onehot) / n
        w -= lr * (g.T @ features)
        b -= lr * g.sum(axis=0)
    return w, b


def head_record(weight: np.ndarray, bias: np.ndarray) -> ArrayRecord:
    return ArrayRecord({WEIGHT_KEY: Array(np.asarray(weight, dtype=np.float32)), BIAS_KEY: Array(np.asarray(bias, dtype=np.float32))})


def initial_head(bundle: Bundle) -> ArrayRecord:
    return head_record(bundle.state[WEIGHT_KEY], bundle.state[BIAS_KEY])


def bundle_file(config: dict[str, Any]) -> Path:
    return bundle_path(Path(config["bundle-dir"]), str(config["task"]), int(config["budget"]), int(config["public-set"]),
                       int(config.get("role-seed", 20261012)), int(config.get("train-seed", 20261012)), str(config.get("eval-split", "train")))


@lru_cache(maxsize=4)
def public_body(path: str) -> Net:
    """The frozen public body of the base model (public artifact), loaded once per process."""
    state = load_bundle(Path(path)).state
    net = Net()
    net.load_state_dict({key: torch.tensor(value) for key, value in state.items()})
    return net.eval()


@lru_cache(maxsize=64)
def client_features(bundle_file_name: str, task: str, role_seed: int, cohort: str, client_id: int, loader) -> tuple[np.ndarray, np.ndarray]:
    """A client's frozen-body features (no bias column) and labels; the public body never changes within a run, so compute them once."""
    images, labels = loader(task, role_seed, cohort, client_id)
    return embed(public_body(bundle_file_name), images)[:, :-1], labels


def baseline_train(msg: Message, context: Context, run_config: dict[str, Any], client_data) -> Message:
    """ClientApp handler for the baselines: local head training, or the dummy update for an absent client."""
    from experiments.stacked_head.client import parse_ids

    client_id = int(context.node_config["partition-id"])
    arrays = msg.content["arrays"]
    weight, bias = arrays[WEIGHT_KEY].numpy().astype(float), arrays[BIAS_KEY].numpy().astype(float)
    if client_id in parse_ids(run_config.get("absent-clients", "")):
        weight = weight.copy()
        weight[0, 0] += DUMMY_EPSILON
    else:
        torch.set_num_threads(1)
        features, labels = client_features(str(bundle_file(run_config)), str(run_config["task"]), int(run_config.get("role-seed", 20261012)), str(run_config["cohort"]), client_id, client_data)
        weight, bias = local_train(features, labels, weight, bias, float(run_config["local-lr"]), int(run_config["local-steps"]))
    content = {
        "arrays": head_record(weight, bias),
        "metrics": MetricRecord({"client-id": client_id, "num-examples": task_data.CLIENT_SIZE}),
    }
    return Message(content=RecordDict(content), reply_to=msg)


def matched_noise_multiplier(risk: float) -> float:
    """Global-DP noise multiplier whose worst-case (fully clipped) known-alternative AUC equals ``risk``: Phi(1 / (sqrt(2) z))."""
    if not 0.5 < risk < 1:
        raise ValueError("risk must lie in (0.5, 1).")
    return float(1 / (np.sqrt(2) * ndtri(risk)))


def oracle_auc(update_norm: np.ndarray, clip_norm: float, noise_std: float, num_clients: int) -> np.ndarray:
    """Known-alternative (full-knowledge) attack AUC against server-side Gaussian noise: Phi(shift / (sqrt(2) sigma)),
    with shift = min(||update||, C) / n (a client's clipped contribution to the average). Infinite when there is no noise."""
    shift = np.minimum(np.asarray(update_norm, dtype=float), clip_norm) / num_clients
    if noise_std <= 0:
        return np.where(shift > 0, 1.0, 0.5)
    return ndtr(shift / (np.sqrt(2) * noise_std))


class GatedBaselineStrategy(Strategy):
    """Delegates to a repository DP strategy, then applies the public validation gate to the aggregated head update."""

    def __init__(self, inner: Strategy, bundle: Bundle, recorder: ReleaseRecorder, multipliers: tuple[float, ...], *, replicate: bool, clip_norm: float, num_clients: int) -> None:
        self.inner, self.bundle, self.recorder, self.multipliers = inner, bundle, recorder, tuple(multipliers)
        self.replicate, self.clip_norm, self.num_clients = replicate, clip_norm, num_clients
        self.base: ArrayRecord | None = None
        self.current: ArrayRecord | None = None
        self.history: list[dict[str, float]] = []

    def __repr__(self) -> str:
        return f"Public-validation-gated baseline over {self.inner!r}"

    def summary(self) -> None:
        self.inner.summary()

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        if self.base is None:
            self.base = arrays
        if self.replicate:
            arrays = self.base
        self.current = arrays
        return self.inner.configure_train(server_round, arrays, config, grid)

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid):
        return self.inner.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round: int, replies):
        return self.inner.aggregate_evaluate(server_round, replies)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        ordered = sorted((r for r in replies if not r.has_error()), key=lambda r: int(next(iter(r.content.metric_records.values()))["client-id"]))
        norms = np.array([model_update_l2_norm(r.content["arrays"], self.current) for r in ordered])
        aggregated, metrics = self.inner.aggregate_train(server_round, replies)
        if aggregated is None:
            return None, metrics
        theta0, theta_aggregated = head_theta(self.current), head_theta(aggregated)
        candidates = np.stack([theta0 + m * (theta_aggregated - theta0) for m in self.multipliers])
        pick = self.recorder.record(candidates, theta0)
        multiplier = self.multipliers[pick]
        base_w, base_b = self.current[WEIGHT_KEY].numpy().astype(float), self.current[BIAS_KEY].numpy().astype(float)
        released = head_record(base_w + multiplier * (aggregated[WEIGHT_KEY].numpy().astype(float) - base_w), base_b + multiplier * (aggregated[BIAS_KEY].numpy().astype(float) - base_b))
        noise_std = float(getattr(self.inner, "current_noise_stdv", 0.0) or 0.0)
        self.recorder.add_extra("update_norms", norms)
        self.recorder.add_extra("noise_std", noise_std)
        self.recorder.add_extra("distance", float(getattr(self.inner, "current_distance", np.nan) or np.nan))
        row = {"round": server_round, "multiplier": float(multiplier), "validation-ce-base": float(self.recorder.per_release["val_ce"][-1][0]) if self.multipliers[0] == 0 else float("nan"),
               "validation-ce-chosen": float(self.recorder.per_release["val_ce"][-1][pick]), "num-replies": float(len(ordered)), "noise-std": noise_std}
        self.history.append(row)
        return released, MetricRecord({k: v for k, v in row.items() if k != "round"})


def build_inner(config: dict[str, Any], num_clients: int) -> tuple[Strategy, dict[str, float]]:
    mechanism = str(config["mechanism"])
    clip = float(config.get("clip-norm") or 0.0)
    multiplier = 0.0
    if mechanism == "global-dp":
        multiplier = float(config["noise-multiplier"]) if config.get("noise-multiplier") not in (None, "") else matched_noise_multiplier(float(config["risk"]))
    elif mechanism == "metric-privacy":
        if config.get("noise-multiplier") in (None, ""):
            raise ValueError("metric-privacy needs an explicit --noise-multiplier (its meaning depends on the client-model distance).")
        multiplier = float(config["noise-multiplier"])
    privacy = "vanilla" if mechanism == "vanilla" else mechanism
    inner = make_strategy("fedavg", privacy, num_clients=num_clients, fraction_evaluate=0.0, noise_multiplier=multiplier, clipping_norm=max(clip, 1e-12))
    return inner, {"mechanism": mechanism, "clip_norm": clip, "noise_multiplier": multiplier, "local_lr": float(config["local-lr"]), "local_steps": int(config["local-steps"])}


def run_baseline(grid: Grid, config: dict[str, Any]) -> dict[str, Any]:
    from experiments.stacked_head.server import clean, load_run_bundle, provenance

    bundle = load_run_bundle(config)
    num_clients = task_data.NUM_CLIENTS
    rounds = int(config.get("rounds", 1))
    replicate = bool(config.get("replicate", False))
    if rounds > 1 and not replicate:
        raise ValueError("rounds > 1 requires replicate=true: the one-shot release is not a multi-round protocol.")
    multipliers = tuple(float(m) for m in str(config["multipliers"]).split(",")) if config.get("multipliers") else BASELINE_MULTIPLIERS
    np.random.seed((zlib.crc32(str(config.get("cell-key", "")).encode()) + 7919 * int(config.get("seed", 42))) % 2**32)  # Flower draws DP noise from the global NumPy RNG
    recorder = ReleaseRecorder(config, bundle, multipliers, target_records(config, bundle, targets=range(4)) if bool(config.get("attack-records", False)) else None)
    inner, parameters = build_inner(config, num_clients)
    strategy = GatedBaselineStrategy(inner, bundle, recorder, multipliers, replicate=replicate, clip_norm=parameters["clip_norm"], num_clients=num_clients)
    evaluations: dict[int, dict[str, float]] = {}

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        ce, accuracy = head_scores(bundle.evaluation_features, bundle.evaluation_labels, head_theta(arrays)[None])
        evaluations[server_round] = {"eval-ce": float(ce[0]), "eval-accuracy": float(accuracy[0])}
        return MetricRecord(evaluations[server_round])

    strategy.start(grid, initial_arrays=initial_head(bundle), num_rounds=rounds, train_config=ConfigRecord(), evaluate_fn=evaluate)
    from experiments.stacked_head.server import summarize

    control = {"ce": evaluations[0]["eval-ce"], "accuracy": evaluations[0]["eval-accuracy"]}
    released = [{**row, **evaluations[row["round"]]} for row in sorted(strategy.history, key=lambda r: r["round"])]
    releases = recorder.arrays()
    summary_alt, summary_fixed = recorder.summaries(control)
    oracle = None
    if "update_norms" in releases:
        mean_auc = np.array([oracle_auc(n, parameters["clip_norm"], s, num_clients) for n, s in zip(releases["update_norms"], releases["noise_std"])]).mean(axis=0)
        oracle = {"per_client_mean_oracle_auc": [float(x) for x in mean_auc], "mean_update_norm": float(releases["update_norms"].mean()), "mean_noise_std": float(releases["noise_std"].mean()),
                  "fraction_clipped": float((releases["update_norms"] > parameters["clip_norm"]).mean()) if parameters["clip_norm"] > 0 else 0.0}
    result = {
        "kind": "Server-side-DP baseline one-shot head update (independent releases from one public base when replicate=true).",
        "run_name": config.get("run-name"),
        "config": {key: config[key] for key in sorted(config) if key not in ("frozen-config",)},
        "baseline": {**parameters, "multipliers": list(multipliers)},
        "bundle": bundle.meta,
        "control": control,
        "rounds": released,
        "summary": summarize(control, released, multipliers) if released else None,
        "summary_alt_validation": summary_alt,
        "summary_fixed_multiplier_1": summary_fixed,
        "oracle": oracle,
        "provenance": provenance(),
    }
    output = Path(config["output-dir"])
    output.mkdir(parents=True, exist_ok=True)
    name = str(config["run-name"])
    if releases:
        np.savez_compressed(output / f"{name}.releases.npz", multipliers=np.array(multipliers), control=np.array([control["ce"], control["accuracy"]]), **{k: v.astype(np.float32) for k, v in releases.items()})
    (output / f"{name}.json").write_text(json.dumps(clean(result), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return result
