"""Per-release bookkeeping shared by every mechanism: validation/held-out scores of all step multipliers and attack statistics.

A "release" is one noisy aggregate turned into candidate heads (one per step multiplier); the gate picks one by public
validation CE. Everything recorded here is a function of the candidates and public data, so the recorder is mechanism-neutral.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.stacked_head import data as task_data
from experiments.stacked_head.bundle import Bundle
from metricdp_pytorch.stacked_head_strategy import head_class_ce, head_scores


def validation_subset(labels: np.ndarray, size: int) -> np.ndarray:
    """Indices of the first ``size // 4`` validation examples of each class (a smaller public validation set)."""
    return np.concatenate([np.where(labels == k)[0][: size // 4] for k in range(4)])


def gated_summary(control: dict[str, float], eval_ce: np.ndarray, eval_accuracy: np.ndarray, picks: np.ndarray) -> dict[str, float]:
    """Mean outcome of choosing ``picks[r]`` (a multiplier index) at every release r."""
    rows = np.arange(len(picks))
    ce, accuracy = eval_ce[rows, picks], eval_accuracy[rows, picks]
    return {"mean_ce": float(ce.mean()), "gain_over_control": float(control["ce"] - ce.mean()), "mean_accuracy": float(accuracy.mean()), "accuracy_delta": float(accuracy.mean() - control["accuracy"])}


def target_records(config: dict[str, Any], bundle: Bundle, targets) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Features (under the public base's frozen body) and labels of the clients' own training records, for the attacker simulation."""
    import torch

    from experiments.stacked_head.cnn import Net, embed

    model = Net()
    model.load_state_dict({key: torch.tensor(value) for key, value in bundle.state.items()})
    model.eval()
    task = str(config["task"])
    dataset, first = task_data.parse_task(task)
    out = {}
    for t in targets:
        indices = task_data.client_indices(task, int(config.get("role-seed", 20261012)), str(config["cohort"]), int(t))
        out[int(t)] = (embed(model, task_data.load_images(dataset, indices)), task_data.relabel(task_data.split_labels(dataset)[indices], first))
    return out


class ReleaseRecorder:
    def __init__(self, config: dict[str, Any], bundle: Bundle, multipliers: tuple[float, ...], attack_records=None) -> None:
        self.bundle = bundle
        self.multipliers = tuple(multipliers)
        self.alt_sizes = [int(size) for size in str(config.get("extra-validation-sizes", "128")).split(",") if size.strip()]
        self.subsets = {size: validation_subset(bundle.validation_labels, size) for size in self.alt_sizes}
        # Shadow mix of target client k: its dominant class k (205 examples) against 17 of each other class.
        self.mix = np.full((task_data.NUM_CLIENTS // 2, 4), task_data.OFF_CLASS_COUNT, dtype=float)
        self.mix[np.arange(4), np.arange(4)] = task_data.DOMINANT_COUNT
        self.mix /= task_data.CLIENT_SIZE
        self.attack_records = attack_records or {}
        self.per_release: dict[str, list[np.ndarray]] = {
            "val_ce": [], "val_accuracy": [], "eval_ce": [], "eval_accuracy": [], "attack_gain": [], "attack_records_gain": [],
            **{f"val{size}_ce": [] for size in self.alt_sizes}, **{f"val{size}_accuracy": [] for size in self.alt_sizes},
        }
        self.extra: dict[str, list[np.ndarray]] = {}

    def add_extra(self, name: str, value) -> None:
        self.extra.setdefault(name, []).append(np.asarray(value, dtype=float))

    def record(self, candidates: np.ndarray, base_theta: np.ndarray) -> int:
        """Score every candidate on validation and held-out data, update the attack statistics, return the gate's pick."""
        bundle = self.bundle
        val_ce, val_accuracy = head_scores(bundle.validation_features, bundle.validation_labels, candidates)
        self.per_release["val_ce"].append(val_ce)
        self.per_release["val_accuracy"].append(val_accuracy)
        for size, subset in self.subsets.items():
            sub_ce, sub_accuracy = head_scores(bundle.validation_features[subset], bundle.validation_labels[subset], candidates)
            self.per_release[f"val{size}_ce"].append(sub_ce)
            self.per_release[f"val{size}_accuracy"].append(sub_accuracy)
        pick = int(val_ce.argmin())
        # Practical model-only attack statistic: shadow-weighted CE decrease of the released head relative to the base.
        class_ce = head_class_ce(bundle.validation_features, bundle.validation_labels, candidates)
        base_ce = head_class_ce(bundle.validation_features, bundle.validation_labels, base_theta[None])[0]
        self.per_release["attack_gain"].append(self.mix @ (base_ce - class_ce[pick]))
        if self.attack_records:
            chosen = candidates[pick][None]
            self.per_release["attack_records_gain"].append(np.array([
                head_scores(f, y, base_theta[None])[0][0] - head_scores(f, y, chosen)[0][0] for f, y in (self.attack_records[t] for t in range(4))
            ]))
        ce, accuracy = head_scores(bundle.evaluation_features, bundle.evaluation_labels, candidates)
        self.per_release["eval_ce"].append(ce)
        self.per_release["eval_accuracy"].append(accuracy)
        return pick

    def arrays(self) -> dict[str, np.ndarray]:
        out = {key: np.stack(value) for key, value in self.per_release.items() if value}
        out.update({key: np.stack(value) for key, value in self.extra.items() if value})
        return out

    def summaries(self, control: dict[str, float]) -> tuple[dict | None, dict | None]:
        releases = self.arrays()
        if "eval_ce" not in releases:
            return None, None
        alt = {str(size): gated_summary(control, releases["eval_ce"], releases["eval_accuracy"], releases[f"val{size}_ce"].argmin(axis=1)) for size in self.alt_sizes}
        fixed = None
        if 1.0 in self.multipliers:
            fixed = gated_summary(control, releases["eval_ce"], releases["eval_accuracy"], np.full(len(releases["eval_ce"]), self.multipliers.index(1.0)))
        return alt, fixed
