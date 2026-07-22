"""Detailed final-model evaluation for the Alzheimer paper reproduction.

The normal Flower evaluation path intentionally records only scalar per-round
loss and accuracy. This module runs once after training from a transient final
checkpoint and persists enough information to recompute richer metrics without
retaining the model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule, CLASS_NAMES
from experiments.reproduce.paper_cnn import PaperCNN


def predict_probabilities(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer labels and class probabilities for one loader."""
    labels_out: list[np.ndarray] = []
    probabilities_out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            probabilities_out.append(model(images).cpu().numpy())
            labels_out.append(labels.numpy())
    if not labels_out:
        raise ValueError("Evaluation loader must not be empty.")
    return (
        np.concatenate(labels_out).astype(np.int64),
        np.concatenate(probabilities_out).astype(np.float32),
    )


def binary_roc_curve(
        labels: np.ndarray,
        scores: np.ndarray,
) -> dict[str, Any] | None:
    """Compute a binary ROC curve and trapezoidal AUC without scikit-learn."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_labels)[threshold_indices].astype(float)
    false_positives = (1 + threshold_indices - true_positives).astype(float)
    true_positive_rate = np.r_[0.0, true_positives / positives, 1.0]
    false_positive_rate = np.r_[0.0, false_positives / negatives, 1.0]

    # JSON cannot represent infinity. Null denotes the two synthetic endpoints.
    thresholds: list[float | None] = [
        None,
        *sorted_scores[threshold_indices].tolist(),
        None,
    ]
    return {
        "auc": float(np.trapezoid(true_positive_rate, false_positive_rate)),
        "fpr": false_positive_rate.tolist(),
        "tpr": true_positive_rate.tolist(),
        "thresholds": thresholds,
    }


def classification_metrics(
        labels: np.ndarray,
        probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute confusion, F1/precision/recall, log loss, and OVR ROC/AUC."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("Probabilities must have one column per paper class.")
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("Labels and probabilities must have equal non-zero length.")

    predictions = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    support = confusion.sum(axis=1)
    true_positives = np.diag(confusion).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    precision = np.divide(
        true_positives,
        predicted,
        out=np.zeros_like(true_positives),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positives,
        support,
        out=np.zeros_like(true_positives),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positives),
        where=(precision + recall) != 0,
    )
    total = len(labels)
    weights = support / total
    selected_probabilities = np.clip(
        probabilities[np.arange(total), labels],
        np.finfo(np.float64).tiny,
        1.0,
    )

    per_class_roc: dict[str, dict[str, Any] | None] = {}
    aucs: list[float] = []
    auc_weights: list[float] = []
    for class_id in range(num_classes):
        roc = binary_roc_curve(labels == class_id, probabilities[:, class_id])
        per_class_roc[str(class_id)] = roc
        if roc is not None:
            aucs.append(float(roc["auc"]))
            auc_weights.append(float(weights[class_id]))

    micro_roc = binary_roc_curve(
        np.eye(num_classes, dtype=np.int8)[labels].ravel(),
        probabilities.ravel(),
    )
    auc_weight_total = sum(auc_weights)
    weighted_auc = (
        sum(auc * weight for auc, weight in zip(aucs, auc_weights, strict=True))
        / auc_weight_total
        if auc_weight_total
        else None
    )
    micro_value = float(true_positives.sum() / total)
    return {
        "num_examples": total,
        "accuracy": float((predictions == labels).mean()),
        "log_loss": float(-np.log(selected_probabilities).mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            str(class_id): {
                "name": CLASS_NAMES[class_id],
                "support": int(support[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
            }
            for class_id in range(num_classes)
        },
        "averages": {
            "macro": {
                "precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1": float(f1.mean()),
            },
            "weighted": {
                "precision": float(np.dot(precision, weights)),
                "recall": float(np.dot(recall, weights)),
                "f1": float(np.dot(f1, weights)),
            },
            "micro": {
                "precision": micro_value,
                "recall": micro_value,
                "f1": micro_value,
            },
        },
        "roc_ovr": {
            "macro_auc": float(np.mean(aucs)) if aucs else None,
            "weighted_auc": float(weighted_auc) if weighted_auc is not None else None,
            "micro": micro_roc,
            "per_class": per_class_roc,
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    os.replace(temporary, path)


def _evaluate_model(
    *,
    model: PaperCNN,
    run: dict[str, Any],
    run_json_path: Path,
    evaluation_json_path: Path,
    predictions_path: Path,
    data_module: AlzheimerDataModule,
    device: torch.device | None,
) -> dict[str, Any]:
    """Evaluate a loaded final model and atomically persist detailed artifacts."""
    metadata = run["metadata"]
    seed = int(metadata["seed"])
    partition_mode = str(metadata["partition_mode"])
    num_clients = int(metadata.get("num_clients", 4))
    batch_size = int(metadata.get("batch_size", 32))
    server_batch_size = int(metadata.get("initialization_batch_size", 32))
    max_client_samples = int(metadata.get("max_client_samples", 0))
    max_test_samples = int(metadata.get("max_test_samples", 0))
    partition_profile = str(metadata.get("partition_profile", "auto"))
    encoded_weights = str(metadata.get("client_weights", "")).strip()
    client_weights = (
        [float(weight.strip()) for weight in encoded_weights.split(",")]
        if encoded_weights
        else None
    )
    selected_device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(selected_device)

    _, server_loader = data_module.server_loaders(
        batch_size=server_batch_size,
        seed=seed,
        max_samples=max_test_samples,
    )
    server_labels, server_probabilities = predict_probabilities(
        model, server_loader, selected_device
    )
    arrays = {
        "server_y_true": server_labels,
        "server_probabilities": server_probabilities,
        "server_y_pred": server_probabilities.argmax(1).astype(np.int64),
    }

    client_results: list[dict[str, Any]] = []
    all_client_labels: list[np.ndarray] = []
    all_client_probabilities: list[np.ndarray] = []
    for client_id in range(num_clients):
        _, client_loader = data_module.client_loaders(
            client_id,
            num_partitions=num_clients,
            partition_mode=partition_mode,
            batch_size=batch_size,
            seed=seed,
            partition_profile=partition_profile,
            client_weights=client_weights,
            max_samples=max_client_samples,
        )
        labels, probabilities = predict_probabilities(
            model, client_loader, selected_device
        )
        all_client_labels.append(labels)
        all_client_probabilities.append(probabilities)
        arrays[f"client_{client_id}_y_true"] = labels
        arrays[f"client_{client_id}_probabilities"] = probabilities
        arrays[f"client_{client_id}_y_pred"] = probabilities.argmax(1).astype(
            np.int64
        )
        client_results.append(
            {"client_id": client_id, **classification_metrics(labels, probabilities)}
        )

    combined_labels = np.concatenate(all_client_labels)
    combined_probabilities = np.concatenate(all_client_probabilities)
    arrays["clients_combined_y_true"] = combined_labels
    arrays["clients_combined_probabilities"] = combined_probabilities
    arrays["clients_combined_y_pred"] = combined_probabilities.argmax(1).astype(
        np.int64
    )

    output = {
        "schema_version": 1,
        "run_name": metadata["run_name"],
        "source_run_json": str(run_json_path.resolve()),
        "class_names": list(CLASS_NAMES),
        "server_final_test": classification_metrics(
            server_labels, server_probabilities
        ),
        "clients_combined_test": classification_metrics(
            combined_labels, combined_probabilities
        ),
        "clients": client_results,
        "raw_predictions": str(predictions_path.resolve()),
    }

    history = run["server_evaluate_metrics"]
    final_round = max(
        int(round_number) for round_number in history if int(round_number) > 0
    )
    recorded_accuracy = float(history[str(final_round)]["accuracy"])
    postprocessed_accuracy = float(output["server_final_test"]["accuracy"])
    if not math.isclose(
        postprocessed_accuracy,
        recorded_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Postprocessed accuracy {postprocessed_accuracy} does not match "
            f"recorded accuracy {recorded_accuracy}."
        )
    output["validated_against_run_json"] = {
        "round": final_round,
        "accuracy": recorded_accuracy,
    }

    _atomic_npz(predictions_path, arrays)
    _atomic_json(evaluation_json_path, output)
    json.loads(evaluation_json_path.read_text(encoding="utf-8"))
    with np.load(predictions_path) as persisted:
        if len(persisted["server_y_true"]) != output["server_final_test"][
            "num_examples"
        ]:
            raise ValueError("Persisted server predictions failed validation.")
    return output


def evaluate_state_dict(
    *,
    state_dict: dict[str, torch.Tensor],
    run: dict[str, Any],
    run_json_path: Path,
    evaluation_json_path: Path,
    predictions_path: Path,
    data_module: AlzheimerDataModule | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Persist detailed evaluation directly from in-memory final parameters."""
    model = PaperCNN()
    model.load_state_dict(state_dict)
    return _evaluate_model(
        model=model,
        run=run,
        run_json_path=run_json_path,
        evaluation_json_path=evaluation_json_path,
        predictions_path=predictions_path,
        data_module=data_module or AlzheimerDataModule(),
        device=device,
    )


def evaluate_saved_model(
    *,
    model_path: Path,
    run_json_path: Path,
    evaluation_json_path: Path,
    predictions_path: Path,
    delete_model_on_success: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate one final checkpoint and atomically persist detailed artifacts."""
    output = evaluate_state_dict(
        state_dict=torch.load(model_path, map_location="cpu", weights_only=True),
        run=json.loads(run_json_path.read_text(encoding="utf-8")),
        run_json_path=run_json_path,
        evaluation_json_path=evaluation_json_path,
        predictions_path=predictions_path,
        device=device,
    )
    if delete_model_on_success:
        model_path.unlink()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--delete-model-on-success", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_saved_model(
        model_path=args.model,
        run_json_path=args.run_json,
        evaluation_json_path=args.evaluation_json,
        predictions_path=args.predictions,
        delete_model_on_success=args.delete_model_on_success,
    )
    print(
        json.dumps(
            {
                "server_accuracy": result["server_final_test"]["accuracy"],
                "server_macro_f1": result["server_final_test"]["averages"][
                    "macro"
                ]["f1"],
                "clients_accuracy": result["clients_combined_test"]["accuracy"],
            }
        )
    )


if __name__ == "__main__":
    main()
