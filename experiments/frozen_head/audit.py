"""Secondary empirical audit of the frozen-head regime (B4, PROTOCOL.md Amendment 1, part C).

Randomized inclusion as in M0's resolved design (``results/cia_v2/measurement/ESCALATION.md`` Resolution):

* 32 candidates chosen by M0's rule (``select_candidates``: the 5 largest clients, then 27 by size quantile); at
  ``n = 48`` this is M0's fixed panel ``CANDIDATES`` (checked at run time);
* ``R = 32`` replicates from ``balanced_design(32, 32, seed=DESIGN_SEED)``: every candidate IN in half of them, every
  replicate including exactly 16, so the active count (``n - 16``) is a public constant and is the denominator;
* replicate ``j`` has its own ``(client_seed, noise_seed)`` from ``SeedSequence([DESIGN_SEED, j])``, shared across
  configurations and never between the runs of one configuration;
* each candidate's shadow sample is a stratified 10% of its training records (``split_stratified`` on the training
  records in loader order, ``seed = layout_seed``, as M0's ``ShadowDataModule``), at least 2 records;
* the shadow loss is scored after every round ``1..T`` (round 0 is the public initial head, identical in all runs).

Attackers and intervals are Agent A's (``experiments/cia/audit_attacks.py``, imported unchanged): the calibrated loss
attacker (primary), the uncalibrated threshold, the pair jackknife at ``GATE_LEVEL`` and the two-way bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import norm

from experiments.frozen_head import runner

AUDIT_DIR = runner.RESULTS / "audit"
DESIGN_SEED = 2026101801
AUDIT_REPLICATES = 32
CANDIDATE_COUNT = 32
SHADOW_FRACTION = 0.10
MIN_SHADOW = 2


@dataclass(frozen=True)
class AuditJob:
    clients: int
    rounds: int
    mu_total: float
    replicate: int

    @property
    def name(self) -> str:
        mu = "nonoise" if self.mu_total == runner.NO_NOISE else f"mu{self.mu_total:g}"
        return f"n{self.clients}_T{self.rounds}_{mu}_r{self.replicate:02d}"

    @property
    def path(self) -> Path:
        return AUDIT_DIR / f"{self.name}.json"


def replicate_seeds(replicate: int) -> tuple[int, int]:
    client_seed, noise_seed = np.random.SeedSequence([DESIGN_SEED, replicate]).generate_state(2)
    return int(client_seed), int(noise_seed)


def design() -> np.ndarray:
    from experiments.cia.inclusion import balanced_design

    return balanced_design(CANDIDATE_COUNT, AUDIT_REPLICATES, seed=DESIGN_SEED)


@lru_cache(maxsize=2)
def panel(clients: int) -> dict:
    """Candidates, top-5 subset and shadow records (train-split indices) for the evaluation layout."""
    from experiments.frozen_head.features import load_split_features
    from metricdp_pytorch.utils.split_data import split_stratified
    from results.cia_frontier.eurosat_frontier.data import dirichlet_partitions
    from results.cia_v2.randomized_inclusion.runner import CANDIDATES, select_candidates

    fed = runner.federation(clients, runner.EVALUATION_LAYOUT)
    layout = fed["layout"]
    sizes = {t: int(s) for t, s in enumerate(layout.client_sizes)}
    top, others = select_candidates(sizes, others=CANDIDATE_COUNT - 5)
    candidates = top + others
    if clients == 48 and tuple(candidates) != tuple(CANDIDATES):
        raise RuntimeError("The 48-client panel differs from M0's fixed CANDIDATES.")
    _, train_y = load_split_features("train")
    labels = tuple(train_y.tolist())
    seed = runner.EVALUATION_LAYOUT
    factor = runner.SPLIT_FACTOR[clients]
    parts = dirichlet_partitions(labels, clients // factor, 0.3, seed)
    if factor > 1:
        parts = [chunk.tolist() for t, part in enumerate(parts)
                 for chunk in np.array_split(np.random.default_rng([seed, t]).permutation(part), factor)]
    shadows = []
    for c in candidates:
        train_order, _ = split_stratified(labels, parts[c], 0.8, seed=seed + c)  # loader order, as M0
        if sorted(train_order) != layout.client_train[c].tolist():
            raise RuntimeError(f"Candidate {c}: training records differ from the layout.")
        count = max(MIN_SHADOW, int(SHADOW_FRACTION * len(train_order)))
        if count >= len(train_order):
            shadow = list(train_order)
        else:
            positions, _ = split_stratified([labels[i] for i in train_order], list(range(len(train_order))),
                                            count / len(train_order), seed=seed)
            shadow = [train_order[p] for p in positions]
        shadows.append(np.asarray(shadow, dtype=np.int64))
    return {"candidates": tuple(int(c) for c in candidates), "top": tuple(int(c) for c in top), "shadows": shadows}


def audit_jobs() -> list[AuditJob]:
    return [AuditJob(n, t, mu, r) for n, t, mu in runner.settings() for r in range(AUDIT_REPLICATES)]


def execute(job: AuditJob) -> str:
    from experiments.frozen_head.features import load_split_features
    from experiments.frozen_head.fedavg import HeadConfig, run
    from metricdp_pytorch import accounting

    if job.path.exists():
        return f"skip {job.name}"
    started = time.time()
    chosen = _chosen()[(job.clients, job.rounds, job.mu_total)]
    fed = runner.federation(job.clients, runner.EVALUATION_LAYOUT)
    info = panel(job.clients)
    member = design()[:, job.replicate]
    included = {c for c, m in zip(info["candidates"], member) if m}
    active = [t for t in range(job.clients) if t not in info["candidates"] or t in included]
    clip = chosen["clip_multiplier"] * runner.base_clip(job.clients, chosen["local_epochs"], chosen["local_lr"])
    z = 0.0 if job.mu_total == runner.NO_NOISE else accounting.noise_multiplier_for_mu(job.mu_total, rounds=job.rounds)
    config = HeadConfig(rounds=job.rounds, local_epochs=chosen["local_epochs"], local_lr=chosen["local_lr"],
                        batch_size=runner.BATCH_SIZE, clip=clip, server_lr=chosen["server_lr"], noise_multiplier=z,
                        num_slots=len(active))
    client_seed, noise_seed = replicate_seeds(job.replicate)
    out = run(fed["head"], fed["public"].theta, fed["train"], config, client_seed=client_seed, noise_seed=noise_seed,
              active=active, evaluate={"server_test": fed["server_test"]}, record_thetas=True)
    train_x, train_y = load_split_features("train")
    scale = _scale(job.clients)
    losses = [[fed["head"].loss_and_grad(theta, scale(train_x[shadow]), train_y[shadow])[0] for theta in out["thetas"]]
              for shadow in info["shadows"]]
    record = {"job": asdict(job), "name": job.name, "design_seed": DESIGN_SEED, "candidates": list(info["candidates"]),
              "top5": list(info["top"]),
              "membership": member.astype(int).tolist(), "active_count": len(active), "config": config.as_dict(),
              "selected": chosen, "client_seed": client_seed, "noise_seed": noise_seed,
              "shadow_losses": losses, "final_server_test": out["accuracy"]["server_test"][-1],
              "mu_total": job.mu_total if z > 0 else None, "seconds": round(time.time() - started, 2)}
    runner._atomic_json(job.path, record)
    return f"done {job.name} {record['seconds']}s"


@lru_cache(maxsize=1)
def _chosen() -> dict:
    return runner.select(runner._load("tune"))


@lru_cache(maxsize=2)
def _scale(clients: int):
    from experiments.frozen_head.features import load_split_features
    from experiments.frozen_head.public_only import Standardizer

    test_x, _ = load_split_features("test")
    return Standardizer.fit(test_x[runner.federation(clients, runner.EVALUATION_LAYOUT)["layout"].public])


def gdp_bounds(mu: float) -> dict:
    """Bounds on any attacker against mu-GDP: AUC of G_mu, TPR = 1 - G_mu(alpha)."""
    return {"auc": float(norm.cdf(mu / math.sqrt(2))), "tpr_at_5pct": float(norm.cdf(mu - norm.ppf(0.95))),
            "tpr_at_1pct": float(norm.cdf(mu - norm.ppf(0.99)))}


def analyze(records: list[dict]) -> dict:
    from experiments.cia.audit_attacks import (GATE_LEVEL, calibrated_scores, pair_jackknife, roc_auc,
                                               threshold_scores, tpr_at_fpr, two_way_bootstrap)

    groups: dict[tuple, list[dict]] = {}
    for record in records:
        job = record["job"]
        groups.setdefault((job["clients"], job["rounds"], job["mu_total"]), []).append(record)
    metrics = {"auc": roc_auc, "tpr_at_5pct": lambda s, y: tpr_at_fpr(s, y, 0.05),
               "tpr_at_1pct": lambda s, y: tpr_at_fpr(s, y, 0.01)}
    rows = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: r["job"]["replicate"])
        if len(group) != AUDIT_REPLICATES:
            raise ValueError(f"{key}: {len(group)} of {AUDIT_REPLICATES} replicates")
        losses = np.transpose(np.array([r["shadow_losses"] for r in group]), (1, 0, 2))  # (candidates, reps, rounds)
        membership = np.array([r["membership"] for r in group], dtype=bool).T
        top = np.isin(group[0]["candidates"], group[0]["top5"])
        scores = {"calibrated": calibrated_scores(losses, membership), "threshold": threshold_scores(losses)}
        row = {"clients": key[0], "rounds": key[1], "mu_total": key[2],
               "bounds": gdp_bounds(key[2]) if key[2] != runner.NO_NOISE else None,
               "server_test_acc_mean": float(np.mean([r["final_server_test"] for r in group]))}
        for attacker, score in scores.items():
            for subset, mask in (("all", np.ones(len(top), bool)), ("top5", top)):
                for name, metric in metrics.items():
                    gate = pair_jackknife(score[mask], membership[mask], metric, level=GATE_LEVEL)
                    boot = two_way_bootstrap(score[mask], membership[mask], metric, seed=DESIGN_SEED)
                    row[f"{attacker}_{subset}_{name}"] = {"estimate": gate.estimate,
                                                          "jackknife_975": [gate.low, gate.high],
                                                          "bootstrap_95": [boot.low, boot.high]}
        auc = row["calibrated_all_auc"]["estimate"]
        row["empirical_mu_from_auc"] = float(math.sqrt(2) * norm.ppf(min(max(auc, 1e-9), 1 - 1e-9)))
        if row["bounds"] is not None:
            row["bound_violated"] = bool(row["calibrated_all_auc"]["jackknife_975"][0] > row["bounds"]["auc"])
        rows.append(row)
    return {"design_seed": DESIGN_SEED, "replicates": AUDIT_REPLICATES, "candidates": CANDIDATE_COUNT,
            "gate_level": GATE_LEVEL, "settings": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("run", "analyze"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "analyze":
        records = [json.loads(p.read_text()) for p in sorted(AUDIT_DIR.glob("*.json"))]
        result = analyze(records)
        runner._atomic_json(runner.RESULTS.parent / "analysis_audit.json", result)
        for row in result["settings"]:
            print(row["clients"], row["rounds"], row["mu_total"], "AUC", round(row["calibrated_all_auc"]["estimate"], 3),
                  "bound", None if row["bounds"] is None else round(row["bounds"]["auc"], 3))
        return
    jobs = audit_jobs()
    pending = [job for job in jobs if not job.path.exists()]
    print(f"audit: {len(jobs)} jobs, {len(pending)} pending")
    if not args.execute:
        for job in pending[:5]:
            print("  plan", job.name)
        return
    with ProcessPoolExecutor(max_workers=args.workers, initializer=runner._worker_init) as pool:
        for count, message in enumerate(pool.map(execute, pending, chunksize=4), start=1):
            if count % 25 == 0 or count == len(pending):
                print(f"[{count}/{len(pending)}] {message}", flush=True)


if __name__ == "__main__":
    main()
