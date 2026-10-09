"""Known-alternative attack check on the real Flower message path.

The attacker (a peer of the target) knows the target's noise-free contribution ``shift`` and the noise-free part of
everyone else's, sees the released messages of all clients except the designated peer, and decides whether the target
contributed (IN) or sent only its noise share (OUT, the dummy world). The Gaussian likelihood-ratio statistic is
optimal; its AUC should match ``ndtr(||shift|| / (sqrt(2) sigma))`` and must not exceed the calibrated risk target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score, roc_curve


def peer_of(target: int) -> int:
    return 6 if target == 7 else 7


def attack_scores(in_messages: np.ndarray, out_messages: np.ndarray, contributions: np.ndarray, target: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Likelihood-ratio scores for IN and OUT worlds; messages have shape (rounds, clients, d)."""
    peer = peer_of(target)
    others = [i for i in range(contributions.shape[0]) if i not in (target, peer)]
    base = contributions[others].sum(axis=0)
    shift = contributions[target]

    def score(messages: np.ndarray) -> np.ndarray:
        observed = messages.sum(axis=1) - messages[:, peer]
        return (observed - base - shift / 2) @ shift / sigma**2

    return score(in_messages), score(out_messages)


def evaluate_attack(contribution_run: Path, in_run: Path, out_run: Path, *, target: int) -> dict:
    """Combine the noise-free contribution run with the IN/OUT message logs of the same cell."""
    contribution_result = json.loads(Path(contribution_run).read_text(encoding="utf-8"))
    contributions = np.array([contribution_result["contributions"][str(i)] for i in range(8)])
    in_result = json.loads(Path(in_run).read_text(encoding="utf-8"))
    sigma = float(in_result["construction"]["sigma"])
    risk = float(in_result["construction"]["risk"])
    in_messages = np.load(Path(in_run).with_suffix(".messages.npz"))["messages"]
    out_messages = np.load(Path(out_run).with_suffix(".messages.npz"))["messages"]
    in_scores, out_scores = attack_scores(in_messages, out_messages, contributions, target, sigma)
    labels = np.r_[np.ones(len(in_scores)), np.zeros(len(out_scores))]
    scores = np.r_[in_scores, out_scores]
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    auc = float(roc_auc_score(labels, scores))
    shift_norm = float(np.linalg.norm(contributions[target]))
    expected = float(ndtr(shift_norm / (np.sqrt(2) * sigma))) if sigma > 0 else float("nan")
    se = float(np.sqrt(auc * (1 - auc) / min(len(in_scores), len(out_scores))))
    return {
        "target": target, "peer": peer_of(target), "releases_per_world": int(len(in_scores)), "sigma": sigma, "risk_target": risk,
        "shift_norm": shift_norm, "expected_auc": expected, "empirical_auc": auc, "auc_standard_error_rough": se,
        "within_target": bool(auc <= risk + 2 * se),
        "tpr_at_fpr_0.01": float(tpr[fpr <= 0.01].max()), "tpr_at_fpr_0.05": float(tpr[fpr <= 0.05].max()),
    }
