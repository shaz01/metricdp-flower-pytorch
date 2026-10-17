"""Tables behind ``accounting.md`` (no training).

    uv run --with dp-accounting==0.6.0 python research/cia_roadmap_v2/theory/accounting_checks.py [published] [floor] [calibration]

Writes ``tables/accounting_published.csv``, ``tables/accounting_floor.csv`` and
``tables/accounting_calibration.csv`` beside this file.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from metricdp_pytorch import accounting as acc
from metricdp_pytorch.accounting import SubsampledGaussian

TABLES = Path(__file__).resolve().parent / "tables"
MNIST_N, MNIST_BATCH = 60_000, 256

# (source, q, z, steps, published epsilon at delta = 1e-5)
PUBLISHED = [
    ("Abadi et al. 2016 Sec. 5.1, E=100", 0.01, 4.0, 10_000, 1.26),
    ("Abadi et al. 2016 Sec. 5.1, E=400", 0.01, 4.0, 40_000, 2.55),
    ("TF Privacy MNIST, noise 1.1, 60 epochs", MNIST_BATCH / MNIST_N, 1.1, math.ceil(60 * MNIST_N / MNIST_BATCH), 3.01),
    ("TF Privacy MNIST, noise 1.3, 15 epochs", MNIST_BATCH / MNIST_N, 1.3, math.ceil(15 * MNIST_N / MNIST_BATCH), 1.19),
    ("TF Privacy MNIST, noise 0.7, 45 epochs", MNIST_BATCH / MNIST_N, 0.7, math.ceil(45 * MNIST_N / MNIST_BATCH), 7.10),
]
FLOORS = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
FLOOR_MECHANISMS = [(1.0, 0.01, 10_000), (2.0, 0.25, 50), (1.0, 0.25, 50), (1.5, 0.25, 100), (0.8, 0.5, 50)]
TARGET_MUS = (1.0, 2.0, 4.0, 8.0)
CALIBRATION = [(q, rounds) for q in (1.0, 0.5, 0.25) for rounds in (10, 20, 50, 100)]


def _write(name: str, rows: list[dict]) -> None:
    TABLES.mkdir(exist_ok=True)
    with (TABLES / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {TABLES / name} ({len(rows)} rows)")


def published() -> list[dict]:
    rows = []
    for source, q, z, steps, value in PUBLISHED:
        mechanism = SubsampledGaussian(z, q, steps)
        rows.append({
            "source": source, "q": round(q, 6), "z": z, "steps": steps, "published_epsilon": value,
            "rdp_classical": round(acc.epsilon(mechanism, 1e-5, method="rdp-classical"), 4),
            "rdp_improved": round(acc.epsilon(mechanism, 1e-5, method="rdp"), 4),
            "pld": round(acc.epsilon(mechanism, 1e-5, method="pld"), 4),
        })
    return rows


def floors() -> list[dict]:
    rows = []
    for z, q, rounds in FLOOR_MECHANISMS:
        mechanism = SubsampledGaussian(z, q, rounds)
        row = {"z": z, "q": q, "rounds": rounds, "mu_full_participation": round(acc.gdp_mu(z, rounds), 4),
               "mu_clt": round(acc.mu_clt(z, q, rounds), 4)}
        for floor in FLOORS:
            row[f"mu_dominating_floor_{floor:.0e}"] = round(acc.dominating_mu(mechanism, delta_floor=floor).mu, 4)
        row["epsilon_pld_1e-5"] = round(acc.epsilon(mechanism, 1e-5), 4)
        rows.append(row)
    return rows


CALIBRATION_RTOL = 1e-2


def _calibration_cell(cell: tuple[float, int, float]) -> dict:
    q, rounds, mu = cell
    z = acc.noise_multiplier_for_mu(mu, sampling_rate=q, rounds=rounds, rtol=CALIBRATION_RTOL)
    mechanism = SubsampledGaussian(z, q, rounds)
    return {"q": q, "rounds": rounds, "target_mu": mu, "noise_multiplier": round(z, 4),
            "rtol": CALIBRATION_RTOL, "mu_dominating": round(acc.dominating_mu(mechanism).mu, 4),
            "mu_full_participation_same_z": round(acc.gdp_mu(z, rounds), 4),
            "epsilon_pld_1e-5": round(acc.epsilon(mechanism, 1e-5), 4)}


def calibration(workers: int = 6) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    cells = [(q, rounds, mu) for q, rounds in CALIBRATION for mu in TARGET_MUS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_calibration_cell, cells))


if __name__ == "__main__":
    import sys

    parts = sys.argv[1:] or ["published", "floor", "calibration"]
    if "published" in parts:
        _write("accounting_published.csv", published())
    if "floor" in parts:
        _write("accounting_floor.csv", floors())
    if "calibration" in parts:
        _write("accounting_calibration.csv", calibration())
