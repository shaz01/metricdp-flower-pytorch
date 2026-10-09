"""Flower ClientApp for the stacked-head experiment.

A client embeds its local images with the broadcast base model's frozen body, forms one clipped, public-subspace
projected, class-balanced head gradient, adds its own Gaussian noise share, and replies with that single vector.
Raw data, features and the unclipped gradient never leave the client. Diagnostic fields (``diagnostics = true``) leak
private-dependent quantities (the clip factor, the noise-free contribution) and exist only for audits and tests.
"""

from __future__ import annotations

import zlib
from functools import lru_cache

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from experiments.stacked_head import data as task_data
from experiments.stacked_head.cnn import embed
from metricdp_pytorch.model_module import load_model
from metricdp_pytorch.stacked_head_strategy import (
    UPDATE_KEY,
    StackedHeadConfig,
    client_contribution,
    head_theta,
)
from metricdp_pytorch.utils.runtime import runtime_config

app = ClientApp()

DEFAULT_MODEL_MODULE = "experiments.stacked_head.cnn:create_model"
PROBE_STREAM_WIDTH = 51


def parse_ids(value) -> set[int]:
    """Accept '', '3', '3,4' or an iterable of ints."""
    if value in ("", None):
        return set()
    if isinstance(value, str):
        return {int(part) for part in value.split(",") if part.strip()}
    return {int(part) for part in value}


@lru_cache(maxsize=None)
def client_data(task: str, role_seed: int, cohort: str, client_id: int):
    """This client's images (float32) and labels (0..3); cached per worker process."""
    dataset, first = task_data.parse_task(task)
    indices = task_data.client_indices(task, role_seed, cohort, client_id)
    labels = task_data.relabel(task_data.split_labels(dataset)[indices], first)
    return task_data.load_images(dataset, indices), labels


@lru_cache(maxsize=8)
def probe_block(seed: int, rounds: int) -> np.ndarray:
    """Replay of the research probe's noise stream: block[round - 1, client, :] ~ N(0, 1)."""
    return np.random.default_rng(seed).normal(size=(rounds, task_data.NUM_CLIENTS, PROBE_STREAM_WIDTH))


def noise_share(config: StackedHeadConfig, run_config: dict, server_round: int, client_id: int) -> np.ndarray:
    """One client's Gaussian share. ``independent`` streams are per (seed, round, client); ``probe`` replays the research draws."""
    if config.noise_free:
        return np.zeros(config.dimension)
    scale = config.sigma / np.sqrt(config.peers)
    source = str(run_config.get("noise-source", "independent"))
    if source == "probe":
        block = probe_block(int(run_config["probe-noise-seed"]), int(run_config["probe-noise-rounds"]))
        return block[server_round - 1, client_id, : config.dimension] * scale
    if source != "independent":
        raise ValueError(f"Unknown noise-source {source!r}; choose 'independent' or 'probe'.")
    # The cell key keeps different cells (task/budget/set/cohort/risk/tag) on independent noise even with one base seed.
    cell = zlib.crc32(str(run_config.get("cell-key", "")).encode())
    stream = np.random.default_rng(np.random.SeedSequence([int(run_config.get("seed", 42)), cell, server_round, client_id, 0x5AFE]))
    return stream.normal(size=config.dimension) * scale


def message_config(train_config) -> StackedHeadConfig:
    return StackedHeadConfig(
        cap=float(train_config["cap"]),
        eta=1.0,
        dimension=int(train_config["dimension"]),
        risk=float(train_config["risk"]),
        mode=str(train_config["mode"]),
        peers=int(train_config["peers"]),
        noise_free=bool(train_config["noise-free"]),
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    run_config = runtime_config(context)
    client_id = int(context.node_config["partition-id"])
    train_config = msg.content["config"]
    server_round = int(train_config["server-round"])
    config = message_config(train_config)
    if client_id >= task_data.NUM_CLIENTS:
        raise ValueError(f"Partition {client_id} is outside the {task_data.NUM_CLIENTS} configured clients.")

    torch.set_num_threads(1)
    model = load_model(str(run_config.get("model-module", DEFAULT_MODEL_MODULE)))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.eval()
    images, labels = client_data(str(run_config["task"]), int(run_config.get("role-seed", 20261012)), str(run_config["cohort"]), client_id)
    absent = client_id in parse_ids(run_config.get("absent-clients", ""))
    public = msg.content["public"]
    basis = public["basis"].numpy()
    contribution = np.zeros(config.dimension) if absent else client_contribution(
        embed(model, images), labels, head_theta(msg.content["arrays"]), public["class-gradients"].numpy(), basis, config
    )
    share = noise_share(config, run_config, server_round, client_id)

    content = {
        UPDATE_KEY: ArrayRecord({"vector": Array(contribution + share)}),
        "metrics": MetricRecord({"client-id": client_id}),
    }
    if bool(run_config.get("diagnostics", False)):
        content["diagnostics"] = ArrayRecord({"contribution": Array(contribution), "share": Array(share)})
        content["metrics"] = MetricRecord({"client-id": client_id, "contribution-norm": float(np.linalg.norm(contribution)), "absent": int(absent)})
    return Message(content=RecordDict(content), reply_to=msg)
