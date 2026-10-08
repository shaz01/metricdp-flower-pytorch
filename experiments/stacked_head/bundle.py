"""Public bundle: the base model and every public artifact the server needs, prepared once per (task, budget, set).

Everything here is computed from PUBLIC data only (the public training examples, the public validation set and the
held-out evaluation images used for reporting). The bundle is self-contained, so the server never reads the dataset;
clients receive the base model and public geometry inside Flower messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.stacked_head import data as task_data
from experiments.stacked_head.cnn import EPOCHS, LEARNING_RATES, embed, train_public
from metricdp_pytorch.stacked_head_strategy import CONTRAST, head_scores, projection_basis, public_class_gradients

MAX_DIMENSION = 51
DEFAULT_ROLE_SEED = 20261012
DEFAULT_TRAIN_SEED = 20261012
STATE_PREFIX = "state/"


@dataclass
class Bundle:
    state: dict[str, np.ndarray]
    theta0: np.ndarray
    class_gradients: np.ndarray
    basis: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    evaluation_features: np.ndarray
    evaluation_labels: np.ndarray
    meta: dict


def head_of(net) -> np.ndarray:
    """Identifiable contrast parameterisation (3, D) of a network's linear head."""
    weight = net.head.weight.detach().numpy().astype(float)
    bias = net.head.bias.detach().numpy().astype(float)
    return CONTRAST.T @ np.column_stack((weight, bias))


def prepare_from_arrays(
    public_images: np.ndarray,
    public_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    evaluation_images: np.ndarray,
    evaluation_labels: np.ndarray,
    *,
    seed_offset: int,
    train_seed: int = DEFAULT_TRAIN_SEED,
    max_dimension: int = MAX_DIMENSION,
) -> Bundle:
    """Train the candidate CNNs on the public set, keep the one with the best public-validation CE, and freeze its geometry."""
    candidates = []
    for epochs in EPOCHS:
        for learning_rate in LEARNING_RATES:
            net = train_public(public_images, public_labels, epochs, learning_rate, train_seed + seed_offset + epochs)
            theta = head_of(net)
            ce = float(head_scores(embed(net, validation_images), validation_labels, theta[None])[0][0])
            candidates.append((ce, net, theta, {"epochs": epochs, "lr": learning_rate, "validation_ce": ce}))
    ce, net, theta0, base = min(candidates, key=lambda candidate: candidate[0])
    public_features = embed(net, public_images)
    validation_features = embed(net, validation_images)
    evaluation_features = embed(net, evaluation_images)
    control_ce, control_accuracy = head_scores(evaluation_features, evaluation_labels, theta0[None])
    meta = {
        "base": base,
        "candidates": [candidate[3] for candidate in candidates],
        "control": {"ce": float(control_ce[0]), "accuracy": float(control_accuracy[0])},
        "num_public": len(public_labels),
        "num_validation": len(validation_labels),
        "num_evaluation": len(evaluation_labels),
        "train_seed": train_seed,
        "seed_offset": seed_offset,
    }
    return Bundle(
        state={key: value.detach().numpy().copy() for key, value in net.state_dict().items()},
        theta0=theta0,
        class_gradients=public_class_gradients(public_features, public_labels, theta0),
        basis=projection_basis(public_features, theta0, max_dimension),
        validation_features=validation_features,
        validation_labels=validation_labels,
        evaluation_features=evaluation_features,
        evaluation_labels=evaluation_labels,
        meta=meta,
    )


def save_bundle(bundle: Bundle, path: Path) -> None:
    arrays = {STATE_PREFIX + key: value for key, value in bundle.state.items()}
    arrays.update(
        theta0=bundle.theta0,
        class_gradients=bundle.class_gradients,
        basis=bundle.basis,
        validation_features=bundle.validation_features,
        validation_labels=bundle.validation_labels,
        evaluation_features=bundle.evaluation_features,
        evaluation_labels=bundle.evaluation_labels,
        meta=np.array(json.dumps(bundle.meta)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_bundle(path: Path) -> Bundle:
    stored = np.load(path, allow_pickle=False)
    return Bundle(
        state={key[len(STATE_PREFIX) :]: stored[key] for key in stored.files if key.startswith(STATE_PREFIX)},
        theta0=stored["theta0"],
        class_gradients=stored["class_gradients"],
        basis=stored["basis"],
        validation_features=stored["validation_features"],
        validation_labels=stored["validation_labels"],
        evaluation_features=stored["evaluation_features"],
        evaluation_labels=stored["evaluation_labels"],
        meta=json.loads(str(stored["meta"])),
    )


def bundle_path(directory: Path, task: str, budget: int, public_set: int, role_seed: int, train_seed: int) -> Path:
    return Path(directory) / f"{task}_b{budget}_s{public_set}_r{role_seed}_t{train_seed}.npz"


def prepare_bundle(
    task: str,
    budget: int,
    public_set: int,
    directory: Path,
    *,
    role_seed: int = DEFAULT_ROLE_SEED,
    train_seed: int = DEFAULT_TRAIN_SEED,
    budgets: tuple[int, ...] = task_data.DEFAULT_BUDGETS,
    force: bool = False,
) -> Path:
    """Build (or reuse) the bundle for one public set of one task. Reads the dataset; the server then needs only the file."""
    path = bundle_path(directory, task, budget, public_set, role_seed, train_seed)
    if path.exists() and not force:
        return path
    dataset, first = task_data.parse_task(task)
    roles = task_data.task_roles(task, role_seed, budgets)
    labels = task_data.split_labels(dataset)
    public_ids = roles.public[(budget, public_set)]
    bundle = prepare_from_arrays(
        task_data.load_images(dataset, public_ids),
        task_data.relabel(labels[public_ids], first),
        task_data.load_images(dataset, roles.validation),
        task_data.relabel(labels[roles.validation], first),
        task_data.load_images(dataset, roles.evaluation),
        task_data.relabel(labels[roles.evaluation], first),
        seed_offset=budget + public_set,
        train_seed=train_seed,
    )
    bundle.meta.update(task=task, budget=budget, public_set=public_set, role_seed=role_seed, budgets=list(budgets))
    save_bundle(bundle, path)
    return path
