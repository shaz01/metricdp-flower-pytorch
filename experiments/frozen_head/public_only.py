"""The public-only baseline: the same linear head trained on the server's public set only.

Features are standardized with the public set's mean and standard deviation (a public statistic,
applied to every split). The head is multinomial logistic regression (scikit-learn, L-BFGS) whose
L2 strength is chosen by stratified 5-fold cross-validation on the public set; no private record
and no server test record is used. Its parameters also initialize every federated run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from experiments.frozen_head.fedavg import Head

# Inverse L2 strengths searched by cross-validation (scikit-learn's ``C``).
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CV_FOLDS = 5
CV_SEED = 0


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> Standardizer:
        return cls(features.mean(axis=0), features.std(axis=0) + 1e-6)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=float) - self.mean) / self.std


@dataclass(frozen=True)
class PublicOnly:
    theta: np.ndarray
    inverse_l2: float
    cv_accuracy: dict[float, float]


def fit_public_only(head: Head, features: np.ndarray, labels: np.ndarray) -> PublicOnly:
    """Cross-validate the L2 strength on the public set, then refit on all of it."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    scores = {c: float(np.mean(cross_val_score(_model(c), features, labels, cv=folds))) for c in C_GRID}
    best = max(C_GRID, key=lambda c: (scores[c], -c))  # ties: stronger regularization
    model = _model(best).fit(features, labels)
    return PublicOnly(head.pack(model.coef_, model.intercept_), best, scores)


def _model(inverse_l2: float) -> LogisticRegression:
    return LogisticRegression(C=inverse_l2, max_iter=5000, tol=1e-6)
