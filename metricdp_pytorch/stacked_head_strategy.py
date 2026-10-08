"""Stacked public-plus-private head-step strategy for Flower (one-shot, Gaussian peer shares).

Each client sends one clipped, public-subspace-projected, class-balanced head gradient plus its own Gaussian noise
share. The server sums the replies, decodes the total, and applies it to the head of the public base model with a
step multiplier chosen on a public validation set. The validation choice only reads the released sum and public
data (post-processing).

The mathematics mirrors ``research/calculations/stacked_constructor_probe.py``; ``tests/test_stacked_head_strategy.py``
checks equivalence against it. The head is an identifiable 4-class linear layer: logits = features @ (C @ theta).T
with ``theta`` of shape (3, D), D = feature dimension + bias column.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from logging import INFO, WARNING

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from scipy.special import ndtri

NUM_CLASSES = 4
DEFAULT_MULTIPLIERS = (0.0, 1 / 30, 1 / 10, 1 / 3, 1.0, 3.0, 10.0, 30.0)
HEAD_WEIGHT_KEY = "head.weight"
HEAD_BIAS_KEY = "head.bias"
UPDATE_KEY = "update"
PUBLIC_KEY = "public"


def contrast_matrix() -> np.ndarray:
    """Orthonormal 4x3 contrast basis whose columns are orthogonal to the all-ones vector."""
    basis = np.zeros((NUM_CLASSES, NUM_CLASSES - 1))
    for j in range(NUM_CLASSES - 1):
        scale = np.sqrt((j + 1) * (j + 2))
        basis[: j + 1, j] = 1 / scale
        basis[j + 1, j] = -(j + 1) / scale
    return basis


CONTRAST = contrast_matrix()


@dataclass(frozen=True)
class StackedHeadConfig:
    """Frozen construction parameters (public)."""

    cap: float
    eta: float
    dimension: int
    risk: float
    mode: str = "target_center"
    peers: int = 7
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS
    noise_free: bool = False

    @property
    def sigma(self) -> float:
        """Per-coordinate std calibrating the optimal known-alternative attack to AUC ``risk``."""
        if self.noise_free:
            return 0.0
        if self.cap <= 0 or not 0.5 < self.risk < 1:
            raise ValueError("Positive cap and risk in (0.5, 1) required.")
        return self.cap / (8 * np.sqrt(2) * ndtri(self.risk))


def class_probabilities(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    logits = (features @ theta.T) @ CONTRAST.T
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def example_gradients(features: np.ndarray, labels: np.ndarray, theta: np.ndarray) -> np.ndarray:
    residual = class_probabilities(features, theta) - np.eye(NUM_CLASSES)[labels]
    return np.einsum("na,nd->nad", residual @ CONTRAST, features)


def public_class_gradients(features: np.ndarray, labels: np.ndarray, theta: np.ndarray) -> np.ndarray:
    gradients = example_gradients(features, labels, theta)
    return np.stack([gradients[labels == k].mean(axis=0) for k in range(NUM_CLASSES)])


def head_scores(features: np.ndarray, labels: np.ndarray, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean cross-entropy and accuracy of each head in ``thetas`` (shape (M, 3, D)) on one labelled set."""
    weights = np.einsum("ca,nad->ncd", CONTRAST, thetas)
    logits = np.einsum("md,ncd->nmc", features, weights)
    top = logits.max(axis=2)
    log_sum = top + np.log(np.exp(logits - top[:, :, None]).sum(axis=2))
    ce = (log_sum - logits[:, np.arange(len(labels)), labels]).mean(axis=1)
    return ce, (logits.argmax(axis=2) == labels).mean(axis=1)


def projection_basis(features: np.ndarray, theta: np.ndarray, dimension: int) -> np.ndarray:
    """Top-``dimension`` eigenvectors of the public head Hessian, with a deterministic sign convention."""
    probs = class_probabilities(features, theta)
    curvature = -probs[:, :, None] * probs[:, None, :]
    curvature[:, np.arange(NUM_CLASSES), np.arange(NUM_CLASSES)] += probs
    contrast = np.einsum("ca,ncd,db->nab", CONTRAST, curvature, CONTRAST)
    size = theta.size
    hessian = np.einsum("nab,ni,nj->aibj", contrast, features, features, optimize=True).reshape(size, size) / len(features)
    values, vectors = np.linalg.eigh(hessian)
    basis = vectors[:, np.argsort(values)[::-1][:dimension]].copy()
    basis *= np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(dimension)])
    return basis


def client_contribution(
    features: np.ndarray,
    labels: np.ndarray,
    theta: np.ndarray,
    class_gradients: np.ndarray,
    basis: np.ndarray,
    config: StackedHeadConfig,
) -> np.ndarray:
    """One client's clipped, projected, class-balanced head gradient, scaled by 1/8 (zeros if the client is empty)."""
    if len(labels) == 0:
        return np.zeros(basis.shape[1])
    gradients = example_gradients(features, labels, theta)
    means = np.stack(
        [gradients[labels == k].mean(axis=0) if np.any(labels == k) else class_gradients[k] for k in range(NUM_CLASSES)]
    )
    query = means.mean(axis=0)
    if config.mode == "target_center":
        query = query - class_gradients.mean(axis=0)
    elif config.mode != "target_balanced":
        raise ValueError(f"Unknown mode {config.mode!r}.")
    projected = query.ravel() @ basis
    norm = np.linalg.norm(projected)
    return projected * min(1.0, config.cap / max(norm, 1e-100)) / 8


def noise_share(config: StackedHeadConfig, rng: np.random.Generator) -> np.ndarray:
    """One client's Gaussian share; shares of ``peers`` clients sum to std ``sigma`` per coordinate."""
    return rng.normal(size=config.dimension) * config.sigma / np.sqrt(config.peers)


def client_update(
    features: np.ndarray,
    labels: np.ndarray,
    theta: np.ndarray,
    class_gradients: np.ndarray,
    basis: np.ndarray,
    config: StackedHeadConfig,
    noise: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Vector a client sends: clipped contribution plus its own noise share."""
    share = noise if noise is not None else noise_share(config, rng if rng is not None else np.random.default_rng())
    return client_contribution(features, labels, theta, class_gradients, basis, config) + share


def head_theta(arrays: ArrayRecord) -> np.ndarray:
    weight = arrays[HEAD_WEIGHT_KEY].numpy().astype(float)
    bias = arrays[HEAD_BIAS_KEY].numpy().astype(float)
    return CONTRAST.T @ np.column_stack((weight, bias))


def with_head(arrays: ArrayRecord, theta: np.ndarray) -> ArrayRecord:
    """Return ``arrays`` with the head replaced by the (softmax-equivalent) weights realising ``theta``."""
    weights = CONTRAST @ theta
    out = ArrayRecord({key: array for key, array in arrays.items()})
    dtype = arrays[HEAD_WEIGHT_KEY].numpy().dtype
    out[HEAD_WEIGHT_KEY] = Array(weights[:, :-1].astype(dtype))
    out[HEAD_BIAS_KEY] = Array(weights[:, -1].astype(dtype))
    return out


class StackedHeadStrategy(FedAvg):
    """One-shot head-step strategy: sum client replies, decode, choose the multiplier on public validation data."""

    def __init__(
        self,
        config: StackedHeadConfig,
        basis: np.ndarray,
        class_gradients: np.ndarray,
        validation_ce: Callable[[np.ndarray], np.ndarray],
        *,
        replicate: bool = False,
        reply_callback: Callable[[int, dict[int, np.ndarray], dict[str, float]], None] | None = None,
        **fedavg_kwargs,
    ) -> None:
        """Build the strategy.

        ``replicate`` re-sends the ORIGINAL base model every round, so each round is an independent
        release from the same base (used to estimate the distribution of a one-shot release in one
        simulation). It is not a multi-round federated protocol: privacy is not composed across rounds.
        ``reply_callback(server_round, {client_id: vector}, metrics)`` is a diagnostics hook.
        """
        super().__init__(**fedavg_kwargs)
        if basis.shape[1] != config.dimension:
            raise ValueError("basis columns must equal config.dimension.")
        self.config = config
        self.basis = basis
        self.class_gradients = class_gradients
        self.validation_ce = validation_ce
        self.replicate = replicate
        self.reply_callback = reply_callback
        self.base_arrays: ArrayRecord | None = None
        self.current_arrays: ArrayRecord | None = None
        self.last_round_metrics: dict[str, float] = {}

    def __repr__(self) -> str:
        return "Stacked head-step strategy (public validation gated)"

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.replicate:
            if self.base_arrays is None:
                self.base_arrays = arrays
            arrays = self.base_arrays
        self.current_arrays = arrays
        node_ids, num_total = sample_nodes(
            grid, self.min_available_nodes, max(int(len(list(grid.get_node_ids())) * self.fraction_train), self.min_train_nodes)
        )
        log(INFO, "configure_train: Sampled %s nodes (out of %s)", len(node_ids), len(num_total))
        config["server-round"] = server_round
        config["cap"] = float(self.config.cap)
        config["dimension"] = int(self.config.dimension)
        config["risk"] = float(self.config.risk)
        config["mode"] = self.config.mode
        config["peers"] = int(self.config.peers)
        config["noise-free"] = bool(self.config.noise_free)
        public = ArrayRecord({"basis": Array(self.basis), "class-gradients": Array(self.class_gradients)})
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config, PUBLIC_KEY: public})
        return self._construct_messages(record, node_ids, "train")

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        if self.current_arrays is None:
            raise RuntimeError("configure_train must run before aggregate_train.")
        good = [reply for reply in replies if not reply.has_error()]
        if len(good) < max(self.min_train_nodes, 1):
            log(WARNING, "aggregate_train: %s of the required %s clients replied; skipping the release.", len(good), self.min_train_nodes)
            return None, None
        good.sort(key=lambda reply: int(next(iter(reply.content.metric_records.values()))["client-id"]))
        total = np.zeros(self.config.dimension)
        vectors: dict[int, np.ndarray] = {}
        for reply in good:
            vector = reply.content[UPDATE_KEY]["vector"].numpy().astype(float)
            vectors[int(next(iter(reply.content.metric_records.values()))["client-id"])] = vector
            total += vector
        theta0 = head_theta(self.current_arrays)
        delta = (total @ self.basis.T).reshape(theta0.shape)
        candidates = np.stack([theta0 - self.config.eta * m * delta for m in self.config.multipliers])
        validation = np.asarray(self.validation_ce(candidates), dtype=float)
        pick = int(validation.argmin())
        self.last_round_metrics = {
            "multiplier": float(self.config.multipliers[pick]),
            "validation-ce-base": float(validation[0]) if self.config.multipliers[0] == 0 else float("nan"),
            "validation-ce-chosen": float(validation[pick]),
            "num-replies": float(len(good)),
        }
        if self.reply_callback is not None:
            self.reply_callback(server_round, vectors, dict(self.last_round_metrics))
        return with_head(self.current_arrays, candidates[pick]), MetricRecord(self.last_round_metrics)
