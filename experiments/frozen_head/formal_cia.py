"""Roadmap v3 P5: client inference in the formal regime (``results/cia_v3/formal_cia/PROTOCOL.md``).

B4's frozen-feature linear head at 20 public images, ``T = 10``, ``n`` in {48, 480}, randomized inclusion as B4 Part
C/D (32 candidates, ``R = 32``, 16 IN per replicate) with a fresh design seed, against four mechanisms:

* ``vanilla``: no noise;
* ``gdp``: client-level DP-FedAvg, exact ``mu``-GDP;
* ``mp``: the paper's distance calibration at global DP's noise multiplier (``fedavg.run(calibration="metric")``);
* ``ss``: Poisson client sampling at ``q = 0.25``, noise for the same dominating ``mu`` (``fedavg.run(sampling_rate=)``).

Replicate ``j`` draws ``(client_seed, noise_seed, sampling_seed)`` from ``SeedSequence([P5_DESIGN_SEED, j])``: no two
runs of a configuration share private randomness; replicate ``j`` is shared across configurations (pairing).

    PYTHONPATH="$PWD" uv run --no-sync python -m experiments.frozen_head.formal_cia reproduce
    PYTHONPATH="$PWD" uv run --no-sync python -m experiments.frozen_head.formal_cia run --execute --workers 6
    PYTHONPATH="$PWD" uv run --no-sync python -m experiments.frozen_head.formal_cia accuracy
    PYTHONPATH="$PWD" uv run --no-sync python -m experiments.frozen_head.formal_cia analyze
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

from experiments.frozen_head import audit, runner

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "cia_v3" / "formal_cia"
RUNS_DIR = OUT_DIR / "results" / "runs"
P5_DESIGN_SEED = 2026103001
PUBLIC_SIZE, ROUNDS, SAMPLING_RATE = 20, 10, 0.25
CLIENT_COUNTS = (48, 480)
LEVELS = (0.5, 1.0, 2.0, 4.0)  # nominal mu_total of every noisy mechanism
# Declared extension (STATE.md): levels added once, on accuracy alone, before any leakage statistic.
EXTRA_LEVELS: dict[tuple[int, str], tuple[float, ...]] = {}
MECHANISMS = ("gdp", "mp", "ss")
# B4 Part D's global-DP mean final accuracy at mu 2 and 1 (results/cia_v2/formal_regime/results/audit_public20).
TARGETS = {48: (0.843, 0.714), 480: (0.921, 0.910)}
SCHEDULE_SEED = 20261030  # reanalyze.py's seed for the OUT trials' simulated schedules
DELTA = 1e-5
BEATS = 0.05
B4_PUBLISHED = {"auc": 0.638, "low": 0.587, "high": 0.689}  # B4 FINDINGS, n = 48, T = 10, mu = 2, 20 public images
# PROTOCOL.md Amendment 1: the owner PC's feature cache (float32 rounding differs from B4's; ESCALATION.md).
LOCAL_FINGERPRINTS = {"train": "8b56243fcb32340f", "test": "4ae40e575a25aef2"}
# Reproduction check (a) under Amendment 1: B4 Part D runs (clients, rounds, mu_total, replicate), tolerance on losses.
REPRODUCTION_RUNS = ((48, 10, 2.0, 0), (48, 10, 1.0, 7), (48, 10, 0.0, 12), (480, 10, 2.0, 3), (480, 10, 1.0, 20))
LOSS_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Job:
    mechanism: str  # "vanilla" | "gdp" | "mp" | "ss"
    level: float  # nominal mu_total (0 for vanilla)
    clients: int
    replicate: int

    @property
    def config(self) -> str:
        return "vanilla" if self.mechanism == "vanilla" else f"{self.mechanism}-mu{self.level:g}"

    @property
    def name(self) -> str:
        return f"{self.config}_n{self.clients}_r{self.replicate:02d}"

    @property
    def path(self) -> Path:
        return RUNS_DIR / f"{self.name}.json"


def levels(clients: int, mechanism: str) -> tuple[float, ...]:
    return tuple(sorted(LEVELS + EXTRA_LEVELS.get((clients, mechanism), ())))


def configs(clients: int) -> list[tuple[str, float]]:
    return [("vanilla", 0.0)] + [(m, level) for m in MECHANISMS for level in levels(clients, m)]


def jobs() -> list[Job]:
    return [Job(m, level, n, j) for n in CLIENT_COUNTS for m, level in configs(n)
            for j in range(audit.AUDIT_REPLICATES)]


def design() -> np.ndarray:
    from experiments.cia.inclusion import balanced_design

    membership = balanced_design(audit.CANDIDATE_COUNT, audit.AUDIT_REPLICATES, seed=P5_DESIGN_SEED)
    if any((membership[:, 2 * p] == membership[:, 2 * p + 1]).any() for p in range(membership.shape[1] // 2)):
        raise RuntimeError("Replicates 2p and 2p + 1 must be complementary.")
    return membership


def replicate_seeds(replicate: int) -> tuple[int, int, int]:
    """``(client_seed, noise_seed, sampling_seed)``, independent per replicate."""
    return tuple(int(s) for s in np.random.SeedSequence([P5_DESIGN_SEED, replicate]).generate_state(3))


@lru_cache(maxsize=32)
def noise_multiplier(mechanism: str, level: float) -> float:
    from metricdp_pytorch import accounting

    if mechanism == "vanilla":
        return 0.0
    rate = SAMPLING_RATE if mechanism == "ss" else 1.0
    return accounting.noise_multiplier_for_mu(level, sampling_rate=rate, rounds=ROUNDS)


def selection(clients: int, mechanism: str, level: float) -> dict:
    """B4 Part A's selection at 20 public images; levels below 1 take mu = 1's, the rest the nominal level's."""
    key = runner.NO_NOISE if mechanism == "vanilla" else max(level, 1.0)
    return audit._chosen(PUBLIC_SIZE)[(clients, ROUNDS, key)]


def privacy(mechanism: str, level: float) -> dict | None:
    from metricdp_pytorch import accounting

    if mechanism in ("vanilla", "mp"):
        return None
    z = noise_multiplier(mechanism, level)
    rate = SAMPLING_RATE if mechanism == "ss" else 1.0
    mechanism_ = accounting.SubsampledGaussian(z, rate, ROUNDS)
    mu = accounting.dominating_mu(mechanism_).mu
    return {"noise_multiplier": z, "sampling_rate": rate, "mu": mu, "mu_delta_floor": 0.0 if rate == 1 else 1e-10,
            "epsilon": accounting.epsilon(mechanism_, DELTA), "delta": DELTA, "auc_bound": float(norm.cdf(mu / math.sqrt(2)))}


def execute(job: Job) -> str:
    from experiments.frozen_head.features import load_split_features
    from experiments.frozen_head.fedavg import HeadConfig, run

    if job.path.exists():
        return f"skip {job.name}"
    started = time.time()
    chosen = selection(job.clients, job.mechanism, job.level)
    fed = runner.federation(job.clients, runner.EVALUATION_LAYOUT, PUBLIC_SIZE, audit.PUBLIC_SUBSET)
    if fed["fingerprints"] != LOCAL_FINGERPRINTS:
        raise RuntimeError(f"Feature cache {fed['fingerprints']} is not PROTOCOL.md Amendment 1's {LOCAL_FINGERPRINTS}.")
    info = audit.panel(job.clients)
    member = design()[:, job.replicate]
    included = {c for c, m in zip(info["candidates"], member) if m}
    active = [t for t in range(job.clients) if t not in info["candidates"] or t in included]
    clip = chosen["clip_multiplier"] * runner.base_clip(job.clients, chosen["local_epochs"], chosen["local_lr"],
                                                        PUBLIC_SIZE)
    config = HeadConfig(rounds=ROUNDS, local_epochs=chosen["local_epochs"], local_lr=chosen["local_lr"],
                        batch_size=runner.BATCH_SIZE, clip=clip, server_lr=chosen["server_lr"],
                        noise_multiplier=noise_multiplier(job.mechanism, job.level), num_slots=len(active))
    client_seed, noise_seed, sampling_seed = replicate_seeds(job.replicate)
    options = {"calibration": "metric"} if job.mechanism == "mp" else {}
    if job.mechanism == "ss":
        options = {"sampling_rate": SAMPLING_RATE, "sampling_seed": sampling_seed}
    out = run(fed["head"], fed["public"].theta, fed["train"], config, client_seed=client_seed, noise_seed=noise_seed,
              active=active, evaluate={"server_test": fed["server_test"]}, record_thetas=True, **options)
    head = fed["head"]
    train_x, train_y = load_split_features("train")
    scale = audit._scale(job.clients, PUBLIC_SIZE)
    thetas = [fed["public"].theta, *out["thetas"]]
    losses = [[head.loss_and_grad(theta, scale(train_x[shadow]), train_y[shadow])[0] for theta in thetas]
              for shadow in info["shadows"]]
    per_client = [head.accuracy(out["theta"], x, y) for x, y in fed["local_test"]]
    record = {"job": asdict(job), "name": job.name, "config_key": job.config, "design_seed": P5_DESIGN_SEED,
              "candidates": list(info["candidates"]), "top5": list(info["top"]),
              "membership": member.astype(int).tolist(), "active": active, "active_count": len(active),
              "config": config.as_dict(), "selected": chosen, "client_seed": client_seed, "noise_seed": noise_seed,
              "sampling_seed": sampling_seed if job.mechanism == "ss" else None,
              "shadow_losses": losses, "loss_rounds": list(range(ROUNDS + 1)),
              "server_test_by_round": out["accuracy"]["server_test"],
              "final_server_test": out["accuracy"]["server_test"][-1],
              "per_client_local_test": {"min": float(np.min(per_client)), "p10": float(np.quantile(per_client, 0.1)),
                                        "median": float(np.median(per_client))},
              "noise_std": out["noise_std"], "noise_std_by_round": out.get("noise_std_by_round"),
              "distance_by_round": out.get("distance_by_round"), "sampled_by_round": out.get("sampled_by_round"),
              "median_update_norm": out["median_update_norm"], "clipped_fraction": out["clipped_fraction"],
              "public_size": len(fed["public_indices"]), "feature_fingerprints": fed["fingerprints"],
              "seconds": round(time.time() - started, 2)}
    runner._atomic_json(job.path, record)
    return f"done {job.name} {record['seconds']}s"


# --- Analysis ---------------------------------------------------------------------------------------

@dataclass
class Config:
    """One configuration's trials: losses ``(candidates, replicates, T + 1)``, membership, per-replicate values."""
    key: str
    clients: int
    losses: np.ndarray
    membership: np.ndarray
    top5: np.ndarray
    accuracy: np.ndarray
    min_client: np.ndarray
    p10_client: np.ndarray
    schedules: np.ndarray | None  # (candidates, replicates, T) realized participation of IN trials


def load_configs(records: list[dict]) -> dict[tuple[int, str], Config]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["job"]["clients"], record["config_key"]), []).append(record)
    out = {}
    for key, group in groups.items():
        group.sort(key=lambda r: r["job"]["replicate"])
        if [r["job"]["replicate"] for r in group] != list(range(audit.AUDIT_REPLICATES)):
            raise ValueError(f"{key}: replicates {[r['job']['replicate'] for r in group]}")
        candidates = group[0]["candidates"]
        schedules = None
        if group[0]["sampled_by_round"] is not None:
            position = {c: i for i, c in enumerate(candidates)}
            schedules = np.zeros((len(candidates), len(group), ROUNDS), dtype=bool)
            for j, r in enumerate(group):
                for t, sampled in enumerate(r["sampled_by_round"]):
                    for client in sampled:
                        if client in position:
                            schedules[position[client], j, t] = True
        out[key] = Config(
            key=key[1], clients=key[0],
            losses=np.transpose(np.array([r["shadow_losses"] for r in group]), (1, 0, 2)),
            membership=np.array([r["membership"] for r in group], dtype=bool).T,
            top5=np.isin(candidates, group[0]["top5"]),
            accuracy=np.array([r["final_server_test"] for r in group]),
            min_client=np.array([r["per_client_local_test"]["min"] for r in group]),
            p10_client=np.array([r["per_client_local_test"]["p10"] for r in group]),
            schedules=schedules)
    return out


def simulated_schedules(config: Config, candidates: list[int]) -> np.ndarray:
    """IN trials keep the realized schedule; OUT trials get i.i.d. Bernoulli(q) (``reanalyze.simulated_schedules``)."""
    out = config.schedules.copy()
    for c, candidate in enumerate(candidates):
        for j in range(out.shape[1]):
            if not config.membership[c, j]:
                out[c, j] = np.random.default_rng([SCHEDULE_SEED, int(candidate), j]).random(out.shape[2]) < SAMPLING_RATE
    return out


def schedule_statistic(losses: np.ndarray, schedules: np.ndarray, rate: float) -> np.ndarray:
    """``reanalyze.schedule_statistic`` with one round per window: ``sum_t (s_t - q)(d_t - dbar_t)``."""
    drops = losses[..., :-1] - losses[..., 1:]
    drops = drops - drops.mean(axis=1, keepdims=True)
    return ((schedules - rate) * drops).sum(axis=2)


def scores(config: Config, attacker: str, candidates: list[int] | None = None) -> np.ndarray:
    """Calibrated attacker on the losses after rounds ``1..T``; ``sampling-aware`` appends the schedule statistic."""
    from experiments.cia.audit_attacks import calibrated_scores, threshold_scores

    losses = config.losses[..., 1:]
    if attacker == "calibrated":
        return calibrated_scores(losses, config.membership)
    if attacker == "threshold":
        return threshold_scores(losses)
    if attacker == "sampling-aware":
        z = schedule_statistic(config.losses, simulated_schedules(config, candidates), SAMPLING_RATE)
        z = z / z.std()
        return calibrated_scores(np.concatenate([losses, z[..., None]], axis=2), config.membership)
    raise ValueError(attacker)


def config_metrics(score: np.ndarray, membership: np.ndarray, top5: np.ndarray) -> dict:
    from experiments.cia.audit_attacks import GATE_LEVEL, pair_jackknife, roc_auc, tpr_at_fpr, two_way_bootstrap

    metrics = {"auc": roc_auc, "tpr_at_5pct": lambda s, y: tpr_at_fpr(s, y, 0.05),
               "tpr_at_1pct": lambda s, y: tpr_at_fpr(s, y, 0.01)}
    out = {}
    for subset, mask in (("all", np.ones(len(top5), bool)), ("top5", top5)):
        for name, metric in metrics.items():
            gate = pair_jackknife(score[mask], membership[mask], metric, level=GATE_LEVEL)
            boot = two_way_bootstrap(score[mask], membership[mask], metric, seed=P5_DESIGN_SEED)
            out[f"{subset}_{name}"] = {"estimate": gate.estimate, "jackknife_975": [gate.low, gate.high],
                                       "bootstrap_95": [boot.low, boot.high]}
    return out


@dataclass
class Point:
    scores: np.ndarray
    membership: np.ndarray
    accuracy: np.ndarray


def curve_at(points: list[Point], target: float, index=None) -> float:
    from experiments.cia.audit_attacks import roc_auc
    from results.cia_v2.frontier.a1.analyze import curve_value

    values = []
    for p in points:
        s, m, a = p.scores, p.membership, p.accuracy
        if index is not None:
            s, m, a = s[index], m[index], a[index[1].ravel()]
        values.append((float(a.mean()), roc_auc(s.ravel(), m.ravel())))
    return curve_value(values, target)


def matched(mech: list[Point], ref: list[Point], target: float, resamples: int = runner.BOOTSTRAP_RESAMPLES) -> dict:
    """A1's statistic on precomputed scores (as ``reanalyze.matched``) with P5's testability labels."""
    from experiments.cia.audit_attacks import GATE_LEVEL, _complementary_pairs, _jackknife_interval, _resampled_indices
    from results.cia_v2.frontier.a1.analyze import outcome

    candidates, replicates = mech[0].scores.shape
    rows = np.arange(candidates)[:, None]
    estimate = curve_at(mech, target) - curve_at(ref, target)
    pairs = _complementary_pairs(mech[0].membership)
    leave_out = [(rows, np.setdiff1d(np.arange(replicates), pair)[None, :]) for pair in pairs]
    values = np.array([curve_at(mech, target, i) - curve_at(ref, target, i) for i in leave_out])
    if math.isnan(estimate):
        testable, gate = "not testable", {"estimate": estimate, "low": math.nan, "high": math.nan}
    elif np.isnan(values).any():
        testable, gate = "not testable by the gate interval", {"estimate": estimate, "low": math.nan, "high": math.nan}
    else:
        iv = _jackknife_interval(estimate, values, GATE_LEVEL)
        testable, gate = "testable", {"estimate": iv.estimate, "low": iv.low, "high": iv.high}
    draws = np.array([curve_at(mech, target, i) - curve_at(ref, target, i)
                      for i in _resampled_indices(candidates, replicates, resamples, P5_DESIGN_SEED)])
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (math.nan, math.nan)
    result = {"target": target, "mechanism_auc": curve_at(mech, target), "reference_auc": curve_at(ref, target),
              "testability": testable, "gate": gate,
              "two_way": {"low": float(low), "high": float(high), "bracketed": int(len(finite))}}
    if testable == "testable":
        result["a1_outcome"] = outcome(gate)
        result["beats"] = bool(gate["estimate"] <= -BEATS and gate["high"] < 0)
    return result


def curve_points(configs_: dict, clients: int, mechanism: str, score_of, subset=None) -> list[Point]:
    keys = ["vanilla"] + [f"{mechanism}-mu{level:g}" for level in levels(clients, mechanism)]
    points = []
    for key in keys:
        c = configs_[(clients, key)]
        mask = np.ones(len(c.top5), bool) if subset is None else subset
        points.append(Point(score_of(c)[mask], c.membership[mask], c.accuracy))
    return points


def interpolate(points: list[tuple[float, float]], target: float) -> float:
    from results.cia_v2.frontier.a1.analyze import curve_value

    return curve_value(points, target)


def analyze(records: list[dict]) -> dict:
    from experiments.cia.audit_attacks import pair_jackknife_difference, roc_auc

    configs_ = load_configs(records)
    candidates = {n: records_candidates(records, n) for n in CLIENT_COUNTS}
    cache: dict[tuple[int, str, str], np.ndarray] = {}

    def score_of(attacker):
        def get(c: Config) -> np.ndarray:
            key = (c.clients, c.key, attacker if c.schedules is not None else "calibrated")
            if key not in cache:
                cache[key] = scores(c, key[2], candidates[c.clients])
            return cache[key]
        return get

    standard, aware = score_of("calibrated"), score_of("sampling-aware")
    out = {"design_seed": P5_DESIGN_SEED, "replicates": audit.AUDIT_REPLICATES, "candidates": audit.CANDIDATE_COUNT,
           "public_size": PUBLIC_SIZE, "rounds": ROUNDS, "sampling_rate": SAMPLING_RATE, "targets": TARGETS,
           "configs": [], "gate": {}, "matched": {}, "matched_guarantee": [], "fairness": {}}
    for (n, key), c in sorted(configs_.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mechanism, level = (key, 0.0) if key == "vanilla" else (key.split("-mu")[0], float(key.split("-mu")[1]))
        row = {"clients": n, "config": key, "mechanism": mechanism, "level": level,
               "privacy": privacy(mechanism, level),
               "accuracy_mean": float(c.accuracy.mean()), "accuracy_sd": float(c.accuracy.std(ddof=1)),
               "min_client_mean": float(c.min_client.mean()), "p10_client_mean": float(c.p10_client.mean()),
               "calibrated": config_metrics(standard(c), c.membership, c.top5),
               "threshold": {"all_auc": roc_auc(scores(c, "threshold").ravel(), c.membership.ravel())}}
        auc = row["calibrated"]["all_auc"]["estimate"]
        row["implied_mu"] = float(math.sqrt(2) * norm.ppf(min(max(auc, 1e-9), 1 - 1e-9)))
        if row["privacy"] is not None:
            row["bound_violated"] = bool(row["calibrated"]["all_auc"]["jackknife_975"][0] > row["privacy"]["auc_bound"])
        if c.schedules is not None:
            row["sampling_aware"] = config_metrics(aware(c), c.membership, c.top5)
            row["realized_in_rate"] = float(c.schedules[c.membership].mean())
        out["configs"].append(row)
    for n in CLIENT_COUNTS:
        gdp2 = next(r for r in out["configs"] if r["clients"] == n and r["config"] == "gdp-mu2")
        upper = gdp2["calibrated"]["all_auc"]["jackknife_975"][1]
        out["gate"][n] = {"auc": gdp2["calibrated"]["all_auc"]["estimate"], "jackknife_975": gdp2["calibrated"]["all_auc"]["jackknife_975"],
                          "verdict": "protective regime confirmed" if upper <= 0.60 else "not confirmed"}
        comparisons = {"mp-gdp": ("mp", standard, "gdp", standard), "ss-gdp": ("ss", standard, "gdp", standard),
                       "ss_sa-gdp": ("ss", aware, "gdp", standard), "ss-mp": ("ss", standard, "mp", standard),
                       "ss_sa-mp": ("ss", aware, "mp", standard)}
        top5 = configs_[(n, "vanilla")].top5
        out["matched"][n] = {}
        for name, (m1, s1, m2, s2) in comparisons.items():
            out["matched"][n][name] = {}
            for subset_name, subset in (("all", None), ("top5", top5)):
                results = [matched(curve_points(configs_, n, m1, s1, subset), curve_points(configs_, n, m2, s2, subset), t)
                           for t in TARGETS[n]]
                if subset_name == "all":
                    tested = [r for r in results if r["testability"] == "testable"]
                    summary = ("not testable" if not tested
                               else "beats" if all(r["beats"] for r in tested) and len(tested) == len(results)
                               else "mixed or does not beat")
                    out["matched"][n][name]["summary"] = summary
                out["matched"][n][name][subset_name] = results
        for level in LEVELS:
            a, b = configs_[(n, f"ss-mu{level:g}")], configs_[(n, f"gdp-mu{level:g}")]
            for attacker, get in (("calibrated", standard), ("sampling-aware", aware)):
                diff = pair_jackknife_difference(get(a), a.membership, standard(b), b.membership, roc_auc)
                out["matched_guarantee"].append({"clients": n, "mu": level, "attacker": attacker,
                                                 "auc_difference": [diff.estimate, diff.low, diff.high],
                                                 "accuracy_difference": float(a.accuracy.mean() - b.accuracy.mean())})
        out["fairness"][n] = {}
        for mechanism in MECHANISMS:
            keys = ["vanilla"] + [f"{mechanism}-mu{level:g}" for level in levels(n, mechanism)]
            pts = [(configs_[(n, k)].accuracy.mean(), configs_[(n, k)].min_client.mean()) for k in keys]
            p10 = [(configs_[(n, k)].accuracy.mean(), configs_[(n, k)].p10_client.mean()) for k in keys]
            out["fairness"][n][mechanism] = {str(t): {"min_client": interpolate(pts, t), "p10_client": interpolate(p10, t)}
                                             for t in TARGETS[n]}
    return out


def records_candidates(records: list[dict], clients: int) -> list[int]:
    return next(r["candidates"] for r in records if r["job"]["clients"] == clients)


def reproduce_b4() -> dict:
    """Reproduction check (b): B4 Part D's n = 48, T = 10, mu = 2 calibrated AUC from its committed records."""
    paths = sorted(audit.audit_dir(PUBLIC_SIZE).glob("n48_T10_mu2_pub20_r*.json"))
    group = [json.loads(p.read_text()) for p in paths]
    group.sort(key=lambda r: r["job"]["replicate"])
    losses = np.transpose(np.array([r["shadow_losses"] for r in group]), (1, 0, 2))
    membership = np.array([r["membership"] for r in group], dtype=bool).T
    from experiments.cia.audit_attacks import calibrated_scores

    m = config_metrics(calibrated_scores(losses, membership), membership, np.isin(group[0]["candidates"], group[0]["top5"]))
    auc, (low, high) = m["all_auc"]["estimate"], m["all_auc"]["jackknife_975"]
    match = all(round(x, 3) == B4_PUBLISHED[k] for x, k in ((auc, "auc"), (low, "low"), (high, "high")))
    return {"runs": len(group), "auc": auc, "jackknife_975": [low, high], "published": B4_PUBLISHED, "match": match}


def reproduce_run() -> dict:
    """Reproduction check (a), PROTOCOL.md Amendment 1: re-run B4 Part D runs with the current ``fedavg``."""
    import tempfile

    original = audit.audit_dir
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        audit.audit_dir = lambda public_size=0: Path(tmp) / f"audit_public{public_size}"
        try:
            for args in REPRODUCTION_RUNS:
                job = audit.AuditJob(*args, PUBLIC_SIZE)
                committed = json.loads((original(PUBLIC_SIZE) / f"{job.name}.json").read_text())
                audit.execute(job)
                fresh = json.loads(job.path.read_text())
                difference = float(np.max(np.abs(np.array(fresh["shadow_losses"]) - np.array(committed["shadow_losses"]))))
                rows.append({"run": job.name, "final_server_test": [fresh["final_server_test"], committed["final_server_test"]],
                             "max_shadow_loss_difference": difference,
                             "match": fresh["final_server_test"] == committed["final_server_test"]
                             and difference <= LOSS_TOLERANCE})
        finally:
            audit.audit_dir = original
    return {"tolerance": LOSS_TOLERANCE, "runs": rows, "match": all(r["match"] for r in rows)}


def accuracy_table(records: list[dict]) -> list[dict]:
    configs_ = load_configs(records)
    rows = []
    for (n, key), c in sorted(configs_.items()):
        rows.append({"clients": n, "config": key, "accuracy": float(c.accuracy.mean())})
    for n in CLIENT_COUNTS:
        for mechanism in MECHANISMS:
            accs = [configs_[(n, f"{mechanism}-mu{level:g}")].accuracy.mean() for level in levels(n, mechanism)
                    if (n, f"{mechanism}-mu{level:g}") in configs_]
            accs.append(configs_[(n, "vanilla")].accuracy.mean())
            for t in TARGETS[n]:
                if not min(accs) <= t <= max(accs):
                    rows.append({"clients": n, "mechanism": mechanism, "target": t, "bracketed": False,
                                 "range": [float(min(accs)), float(max(accs))]})
    return rows


def _load_records() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("run", "reproduce", "accuracy", "analyze"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "reproduce":
        result = {"run": reproduce_run(), "b4_auc": reproduce_b4()}
        runner._atomic_json(OUT_DIR / "reproduction.json", result)
        print(json.dumps(result, indent=1))
        return
    if args.stage == "accuracy":
        for row in accuracy_table(_load_records()):
            print(row)
        return
    if args.stage == "analyze":
        result = analyze(_load_records())
        runner._atomic_json(OUT_DIR / "analysis.json", result)
        for n, gate in result["gate"].items():
            print(n, gate)
        return
    pending = [job for job in jobs() if not job.path.exists()]
    print(f"formal_cia: {len(jobs())} jobs, {len(pending)} pending")
    if not args.execute:
        for job in pending[:5]:
            print("  plan", job.name)
        return
    with ProcessPoolExecutor(max_workers=args.workers, initializer=runner._worker_init) as pool:
        for count, message in enumerate(pool.map(execute, pending, chunksize=2), start=1):
            if count % 25 == 0 or count == len(pending):
                print(f"[{count}/{len(pending)}] {message}", flush=True)


if __name__ == "__main__":
    main()
