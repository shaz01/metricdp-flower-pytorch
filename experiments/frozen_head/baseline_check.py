"""Strength of the public-only baseline (B4, PROTOCOL.md Amendment 2, part E). No federated runs.

For every public size and evaluation subset of Amendment 1's sweep, five public-only heads are fit on that subset only
and scored on the server test set: ``protocol`` (recomputed, must match the recorded accuracy), ``wide_cv``,
``l2norm_cv``, ``nearest_mean`` and ``oracle_c`` (L2 chosen on the server test: an optimistic ceiling, not a
legitimate baseline). Part A's gains and per-size verdicts are then recomputed, with Part A's bootstrap, against
``best_legit`` (per subset, the most accurate of the first four) and against ``oracle_c``.

    python -m experiments.frozen_head.baseline_check
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestCentroid

from experiments.frozen_head import runner
from experiments.frozen_head.public_only import CV_FOLDS, CV_SEED, Standardizer, _model, fit_public_only

WIDE_GRID = tuple(10 ** (k / 2) for k in range(-10, 9))  # 1e-5 .. 1e4
LEGITIMATE = ("protocol", "wide_cv", "l2norm_cv", "nearest_mean")


def _cv_choice(features: np.ndarray, labels: np.ndarray, grid: tuple[float, ...]) -> float:
    """The protocol's rule (``fit_public_only``) over ``grid``."""
    folds = StratifiedKFold(n_splits=min(CV_FOLDS, int(np.bincount(labels).min())), shuffle=True,
                            random_state=CV_SEED)
    scores = {c: float(np.mean(cross_val_score(_model(c), features, labels, cv=folds))) for c in grid}
    return max(grid, key=lambda c: (scores[c], -c))


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def fit_alternatives(public_x: np.ndarray, public_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict:
    """Raw (unstandardized) features in; per baseline, its L2 choice and per-image server-test correctness."""
    from experiments.frozen_head.fedavg import Head

    scale = Standardizer.fit(public_x)
    xs, ts = scale(public_x), scale(test_x)
    head = Head(10, public_x.shape[1])
    protocol = fit_public_only(head, xs, public_y)
    out = {"protocol": (protocol.inverse_l2, head.predict(protocol.theta, ts))}
    c = _cv_choice(xs, public_y, WIDE_GRID)
    out["wide_cv"] = (c, _model(c).fit(xs, public_y).predict(ts))
    xn, tn = _l2_normalize(public_x), _l2_normalize(test_x)
    c = _cv_choice(xn, public_y, WIDE_GRID)
    out["l2norm_cv"] = (c, _model(c).fit(xn, public_y).predict(tn))
    out["nearest_mean"] = (None, NearestCentroid().fit(xs, public_y).predict(ts))
    test_acc = {c: float(np.mean(_model(c).fit(xs, public_y).predict(ts) == test_y)) for c in WIDE_GRID}
    c = max(WIDE_GRID, key=lambda c: (test_acc[c], -c))
    out["oracle_c"] = (c, _model(c).fit(xs, public_y).predict(ts))
    return {name: {"inverse_l2": c, "server_test_correct": (pred == test_y).astype(int).tolist()}
            for name, (c, pred) in out.items()}


def fits(layout_seed: int = runner.EVALUATION_LAYOUT, sizes: tuple[int, ...] = runner.PUBLIC_SIZES
         ) -> dict[tuple[int, int], dict]:
    from experiments.frozen_head.features import load_split_features

    test_x, test_y = load_split_features("test")
    result = {}
    for size in sizes:
        for subset in range(runner.PUBLIC_SUBSETS):
            fed = runner.federation(48, layout_seed, size, subset)
            public, server = fed["public_indices"], fed["layout"].server_test
            result[(size, subset)] = fit_alternatives(test_x[public], test_y[public], test_x[server], test_y[server])
    return result


def with_baseline(records: list[dict], baselines: dict[tuple[int, int], list[int]]) -> list[dict]:
    """Records whose public-only correctness is replaced by ``baselines[(size, subset)]``."""
    return [{**r, "public_only": {"server_test_correct": baselines[(r["job"]["public_size"], r["job"]["public_subset"])]}}
            for r in records]


def best_legitimate(alternatives: dict) -> str:
    return max(LEGITIMATE, key=lambda name: (np.mean(alternatives[name]["server_test_correct"]), name == "protocol"))


def main() -> None:
    records = runner._load("evaluate_public")
    alternatives = fits()
    for record in records:
        key = (record["job"]["public_size"], record["job"]["public_subset"])
        if record["public_only"]["server_test_correct"] != alternatives[key]["protocol"]["server_test_correct"]:
            raise RuntimeError(f"{record['name']}: recomputed protocol baseline differs from the recorded one.")
    chosen_by_size = runner.select_by_public_size(runner._load("tune_public"))
    picks = {key: best_legitimate(alt) for key, alt in alternatives.items()}
    readings = {}
    for reading, pick in (("best_legit", picks), ("oracle_c", {key: "oracle_c" for key in alternatives})):
        baselines = {key: alternatives[key][pick[key]]["server_test_correct"] for key in alternatives}
        readings[reading] = runner.analyze_public(with_baseline(records, baselines), chosen_by_size)
    result = {
        "fits": [{"public_size": size, "subset": subset, "best_legit": picks[(size, subset)],
                  **{name: {"inverse_l2": alt["inverse_l2"], "server_test_acc": float(np.mean(alt["server_test_correct"]))}
                     for name, alt in alternatives[(size, subset)].items()}}
                 for size, subset in sorted(alternatives)],
        "readings": readings,
    }
    runner._atomic_json(runner.RESULTS.parent / "analysis_baseline_check.json", result)
    for reading, value in readings.items():
        print(reading, json.dumps({size: v["verdict"] for size, v in value["by_size"].items()}))


if __name__ == "__main__":
    main()
