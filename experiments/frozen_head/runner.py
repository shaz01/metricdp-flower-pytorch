"""B4 runner: tuning, evaluation and analysis of head-only DP-FedAvg (``results/cia_v2/formal_regime/PROTOCOL.md``).

    python -m experiments.frozen_head.runner tune               # plan only
    python -m experiments.frozen_head.runner tune --execute --workers 8
    python -m experiments.frozen_head.runner evaluate --execute --workers 8
    python -m experiments.frozen_head.runner analyze

Every run writes one JSON (atomically) under ``results/cia_v2/formal_regime/results/<stage>/``; a run whose
file exists is skipped, so every stage resumes. CPU only, one thread per worker.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results" / "cia_v2" / "formal_regime" / "results"
TUNING_LAYOUT, EVALUATION_LAYOUT = 43, 42
SPLIT_FACTOR = {48: 1, 480: 10}
CLIENT_COUNTS, ROUNDS, MU_TOTALS = (48, 480), (10, 20), (1.0, 2.0, 4.0, 8.0)
NO_NOISE = 0.0  # mu_total key of the no-noise reference
BATCH_SIZE = 32
# Tuning grid (PROTOCOL.md Section 5); clip = multiplier x median round-0 update norm on the tuning layout.
LOCAL_EPOCHS, LOCAL_LRS, SERVER_LRS, CLIP_MULTIPLIERS = (1, 5), (0.05, 0.2, 0.5), (1.0, 3.0, 10.0), (0.5, 1.0)
TUNING_SEEDS = (0, 1)
REPLICATES = 8
CLIENT_SEED_BASE, NOISE_SEED_BASE = 10_000, 20_000
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 2000, 20261017
GATE_PP = 5.0


@dataclass(frozen=True)
class Job:
    stage: str  # "tune" | "evaluate"
    clients: int
    rounds: int
    mu_total: float  # 0 = no noise
    local_epochs: int
    local_lr: float
    server_lr: float
    clip_multiplier: float
    replicate: int

    @property
    def name(self) -> str:
        mu = "nonoise" if self.mu_total == NO_NOISE else f"mu{self.mu_total:g}"
        return (f"n{self.clients}_T{self.rounds}_{mu}_E{self.local_epochs}_lr{self.local_lr:g}"
                f"_slr{self.server_lr:g}_c{self.clip_multiplier:g}_r{self.replicate}")

    @property
    def path(self) -> Path:
        return RESULTS / self.stage / f"{self.name}.json"

    @property
    def layout_seed(self) -> int:
        return TUNING_LAYOUT if self.stage == "tune" else EVALUATION_LAYOUT

    @property
    def seeds(self) -> tuple[int, int]:
        """(client_seed, noise_seed): independent per replicate, shared across settings."""
        if self.stage == "tune":
            return 1_000 + self.replicate, 2_000 + self.replicate
        return CLIENT_SEED_BASE + self.replicate, NOISE_SEED_BASE + self.replicate


def settings() -> list[tuple[int, int, float]]:
    return [(n, t, mu) for n in CLIENT_COUNTS for t in ROUNDS for mu in (NO_NOISE, *MU_TOTALS)]


def tuning_jobs() -> list[Job]:
    return [Job("tune", n, t, mu, e, lr, slr, c, seed)
            for (n, t, mu), e, lr, slr, c, seed in itertools.product(
                settings(), LOCAL_EPOCHS, LOCAL_LRS, SERVER_LRS, CLIP_MULTIPLIERS, TUNING_SEEDS)]


# --- Shared context (cached per worker process) ----------------------------------------------------

@lru_cache(maxsize=4)
def federation(clients: int, layout_seed: int) -> dict:
    from experiments.frozen_head.data import build_layout
    from experiments.frozen_head.features import fingerprint, load_split_features
    from experiments.frozen_head.fedavg import Head
    from experiments.frozen_head.public_only import Standardizer, fit_public_only

    train_x, train_y = load_split_features("train")
    test_x, test_y = load_split_features("test")
    layout = build_layout(tuple(train_y.tolist()), tuple(test_y.tolist()), num_clients=clients,
                          layout_seed=layout_seed, split_factor=SPLIT_FACTOR[clients])
    scale = Standardizer.fit(test_x[layout.public])
    head = Head(10, 512)
    public = fit_public_only(head, scale(test_x[layout.public]), test_y[layout.public])
    local_test_idx = np.concatenate(layout.client_test)
    return {
        "layout": layout, "head": head, "public": public,
        "train": [(scale(train_x[i]), train_y[i]) for i in layout.client_train],
        "local_test": [(scale(train_x[i]), train_y[i]) for i in layout.client_test],
        "pooled_local_test": (scale(train_x[local_test_idx]), train_y[local_test_idx]),
        "server_test": (scale(test_x[layout.server_test]), test_y[layout.server_test]),
        "fingerprints": {"train": fingerprint(train_x), "test": fingerprint(test_x)},
    }


@lru_cache(maxsize=16)
def base_clip(clients: int, local_epochs: int, local_lr: float) -> float:
    """Median round-0 update norm on the tuning layout from the public-only head (public hyperparameter)."""
    from experiments.frozen_head.fedavg import HeadConfig, run

    fed = federation(clients, TUNING_LAYOUT)
    probe = HeadConfig(rounds=1, local_epochs=local_epochs, local_lr=local_lr, batch_size=BATCH_SIZE, clip=1e12,
                       server_lr=1.0, noise_multiplier=0.0, num_slots=clients)
    return float(run(fed["head"], fed["public"].theta, fed["train"], probe, client_seed=0, noise_seed=0)
                 ["median_update_norm"][0])


def execute(job: Job) -> str:
    from metricdp_pytorch import accounting
    from experiments.frozen_head.fedavg import HeadConfig, run

    if job.path.exists():
        return f"skip {job.name}"
    started = time.time()
    fed = federation(job.clients, job.layout_seed)
    clip = job.clip_multiplier * base_clip(job.clients, job.local_epochs, job.local_lr)
    z = 0.0 if job.mu_total == NO_NOISE else accounting.noise_multiplier_for_mu(job.mu_total, rounds=job.rounds)
    config = HeadConfig(rounds=job.rounds, local_epochs=job.local_epochs, local_lr=job.local_lr,
                        batch_size=BATCH_SIZE, clip=clip, server_lr=job.server_lr, noise_multiplier=z,
                        num_slots=job.clients)
    client_seed, noise_seed = job.seeds
    evaluate = {"pooled_local_test": fed["pooled_local_test"]}
    if job.stage == "evaluate":
        evaluate["server_test"] = fed["server_test"]
    out = run(fed["head"], fed["public"].theta, fed["train"], config, client_seed=client_seed,
              noise_seed=noise_seed, evaluate=evaluate)
    head, theta = fed["head"], out["theta"]
    record = {
        "job": job.__dict__, "name": job.name, "layout_seed": job.layout_seed, "config": config.as_dict(),
        "client_seed": client_seed, "noise_seed": noise_seed, "noise_std": out["noise_std"],
        "privacy": (accounting.report(accounting.SubsampledGaussian(z, 1.0, job.rounds), with_rdp=False)
                    if z > 0 else None),
        "accuracy_by_round": out["accuracy"], "median_update_norm": out["median_update_norm"],
        "clipped_fraction": out["clipped_fraction"],
        "final_pooled_local_test": out["accuracy"]["pooled_local_test"][-1],
        "public_only": {"inverse_l2": fed["public"].inverse_l2,
                        "pooled_local_test": head.accuracy(fed["public"].theta, *fed["pooled_local_test"])},
        "feature_fingerprints": fed["fingerprints"], "seconds": round(time.time() - started, 2),
    }
    if job.stage == "evaluate":
        test_x, test_y = fed["server_test"]
        record["final_server_test"] = out["accuracy"]["server_test"][-1]
        record["server_test_correct"] = (head.predict(theta, test_x) == test_y).astype(int).tolist()
        record["public_only"]["server_test_correct"] = (head.predict(fed["public"].theta, test_x) == test_y).astype(int).tolist()
        record["public_only"]["server_test"] = float(np.mean(record["public_only"]["server_test_correct"]))
        per_client = [head.accuracy(theta, x, y) for x, y in fed["local_test"]]
        record["per_client_local_test"] = {"min": float(np.min(per_client)), "p10": float(np.quantile(per_client, 0.1)),
                                           "median": float(np.median(per_client))}
    _atomic_json(job.path, record)
    return f"done {job.name} {record['seconds']}s"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


def _worker_init() -> None:
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass


# --- Selection and evaluation -----------------------------------------------------------------------

def select(tuning_records: list[dict]) -> dict[tuple[int, int, float], dict]:
    """Per setting, the grid point with the best mean final pooled local-test accuracy over tuning seeds.

    Ties: smaller server step, then fewer local epochs, then smaller local lr, then smaller clip.
    """
    groups: dict[tuple, list[float]] = {}
    for record in tuning_records:
        job = record["job"]
        key = (job["clients"], job["rounds"], job["mu_total"], job["server_lr"], job["local_epochs"],
               job["local_lr"], job["clip_multiplier"])
        groups.setdefault(key, []).append(record["final_pooled_local_test"])
    chosen: dict[tuple[int, int, float], dict] = {}
    for key, values in sorted(groups.items()):
        if len(values) != len(TUNING_SEEDS):
            raise ValueError(f"Incomplete tuning cell {key}: {len(values)} of {len(TUNING_SEEDS)} seeds")
        setting, grid = key[:3], key[3:]
        score = float(np.mean(values))
        best = chosen.get(setting)
        if best is None or score > best["score"]:  # sorted keys make ties keep the earliest grid point
            chosen[setting] = {"score": score, "server_lr": grid[0], "local_epochs": grid[1], "local_lr": grid[2],
                               "clip_multiplier": grid[3]}
    return chosen


def evaluation_jobs(chosen: dict[tuple[int, int, float], dict]) -> list[Job]:
    return [Job("evaluate", n, t, mu, c["local_epochs"], c["local_lr"], c["server_lr"], c["clip_multiplier"], r)
            for (n, t, mu), c in sorted(chosen.items()) for r in range(REPLICATES)]


def paired_bootstrap(correct: np.ndarray, baseline: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two-way bootstrap of mean_r (acc_r - acc_public): replicates and test images resampled independently."""
    replicates, images = correct.shape
    draws = np.empty(BOOTSTRAP_RESAMPLES)
    for b in range(BOOTSTRAP_RESAMPLES):
        rows = rng.integers(0, replicates, replicates)
        cols = rng.integers(0, images, images)
        draws[b] = correct[np.ix_(rows, cols)].mean() - baseline[cols].mean()
    return draws


def analyze(records: list[dict], chosen: dict) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by_setting: dict[tuple, list[dict]] = {}
    for record in records:
        job = record["job"]
        by_setting.setdefault((job["clients"], job["rounds"], job["mu_total"]), []).append(record)
    rows = []
    for key, group in sorted(by_setting.items()):
        group.sort(key=lambda r: r["job"]["replicate"])
        correct = np.array([r["server_test_correct"] for r in group], dtype=float)
        baseline = np.array(group[0]["public_only"]["server_test_correct"], dtype=float)
        diff = 100 * (correct.mean() - baseline.mean())
        draws = 100 * paired_bootstrap(correct, baseline, rng)
        low, high = np.quantile(draws, [0.025, 0.975])
        clients, rounds, mu = key
        rows.append({
            "clients": clients, "rounds": rounds, "mu_total": mu, "replicates": len(group),
            "selected": chosen.get(key) if chosen else None,
            "public_only_acc": float(baseline.mean()),
            "dp_acc_mean": float(correct.mean()), "dp_acc_sd_over_replicates": float(correct.mean(axis=1).std(ddof=1)),
            "gain_pp": float(diff), "gain_ci95_pp": [float(low), float(high)],
            "per_client_min_mean": float(np.mean([r["per_client_local_test"]["min"] for r in group])),
            "per_client_p10_mean": float(np.mean([r["per_client_local_test"]["p10"] for r in group])),
            "privacy": group[0]["privacy"],
            "gate_reached": bool(mu != NO_NOISE and mu <= 8 and diff >= GATE_PP),
            "gate_robust": bool(mu != NO_NOISE and mu <= 8 and low >= GATE_PP),
        })
    dp_rows = [r for r in rows if r["mu_total"] != NO_NOISE]
    passing = [r for r in dp_rows if r["gate_reached"]]
    verdict = ("fail" if not passing else
               "pass" if any(r["gate_robust"] for r in passing) else "pass, not robust")
    return {"gate": "G-B4", "threshold_pp": GATE_PP, "verdict": verdict,
            "passing_settings": [(r["clients"], r["rounds"], r["mu_total"]) for r in passing], "settings": rows}


def _load(stage: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((RESULTS / stage).glob("*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("tune", "evaluate", "analyze"))
    parser.add_argument("--execute", action="store_true", help="run the jobs (default: plan only)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "analyze":
        chosen = select(_load("tune"))
        result = analyze(_load("evaluate"), chosen)
        result["selected"] = {f"n{k[0]}_T{k[1]}_mu{k[2]:g}": v for k, v in chosen.items()}
        _atomic_json(RESULTS.parent / "analysis.json", result)
        print(json.dumps({k: result[k] for k in ("verdict", "passing_settings")}))
        return
    jobs = tuning_jobs() if args.stage == "tune" else evaluation_jobs(select(_load("tune")))
    pending = [job for job in jobs if not job.path.exists()]
    print(f"{args.stage}: {len(jobs)} jobs, {len(pending)} pending")
    if not args.execute:
        for job in pending[:5]:
            print("  plan", job.name)
        return
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        for count, message in enumerate(pool.map(execute, pending, chunksize=4), start=1):
            if count % 25 == 0 or count == len(pending):
                print(f"[{count}/{len(pending)}] {message}", flush=True)


if __name__ == "__main__":
    main()
