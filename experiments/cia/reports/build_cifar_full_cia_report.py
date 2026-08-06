"""Build the data-only report for completed CIFAR full replacement CIA runs."""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
FULL_DIR = ROOT / "results" / "planned_runs" / "cifar" / "full"
OUT_DIR = ROOT / "results" / "planned_runs" / "cifar"
STEM = "cifar_full_cia_report"
OUT_TEX = OUT_DIR / f"{STEM}.tex"
OUT_PDF = OUT_DIR / f"{STEM}.pdf"
OUT_PNG = OUT_DIR / "cifar-full-pooled-auc.png"
OUT_DISTANCE_PNG = OUT_DIR / "cifar-full-metric-dp-distance.png"
SEEDS = (42, 43, 44)
CLIENTS = (3, 8, 16)
VARIANTS = (
    ("vanilla", None),
    ("global-dp", 0.0025),
    ("metric-privacy", 0.0025),
    ("global-dp", 0.003333),
    ("metric-privacy", 0.003333),
    ("global-dp", 0.00625),
    ("metric-privacy", 0.00625),
)
LABELS = {"vanilla": "Vanilla", "global-dp": "Global-DP", "metric-privacy": "Metric-privacy"}
COLORS = {
    ("vanilla", None): "#2e6db4",
    ("global-dp", 0.0025): "#d65a31",
    ("global-dp", 0.003333): "#f08b5b",
    ("global-dp", 0.00625): "#9f3b1f",
    ("metric-privacy", 0.0025): "#258968",
    ("metric-privacy", 0.003333): "#55ae8c",
    ("metric-privacy", 0.00625): "#126348",
}


def ratio_from_path(path: Path, privacy: str) -> float | None:
    if privacy == "vanilla":
        return None
    token = next(part for part in path.parts if part.startswith("ratio-"))
    return float(token.removeprefix("ratio-").replace("p", "."))


def load() -> dict[tuple[str, int, str, float | None, int], list[dict]]:
    grouped: dict[tuple[str, int, str, float | None, int], list[dict]] = defaultdict(list)
    for path in sorted(FULL_DIR.rglob("cia.json")):
        adjacency = next(part for part in path.parts if part in {"in-replace", "out-replace"})
        clients = int(next(part for part in path.parts if part.startswith("clients-")).split("-")[1])
        for row in json.loads(path.read_text()):
            privacy = str(row["privacy"])
            grouped[(adjacency, clients, privacy, ratio_from_path(path, privacy), int(row["seed"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["server_round"]))
    expected = {
        (adjacency, clients, privacy, ratio, seed)
        for adjacency in ("in-replace", "out-replace")
        for clients in CLIENTS
        for privacy, ratio in VARIANTS
        for seed in SEEDS
    }
    if set(grouped) != expected or any(len(rows) != 20 for rows in grouped.values()):
        raise RuntimeError(f"Expected {len(expected)} complete trajectories; found {len(grouped)}.")
    return grouped


def auc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    return float(np.mean(in_scores[:, None] > out_scores[None, :]) + 0.5 * np.mean(in_scores[:, None] == out_scores[None, :]))


def scores(rows: list[dict], key: str) -> np.ndarray:
    return -np.asarray([float(row[key]) for row in rows])


def pooled_rows(data: dict, adjacency: str, clients: int, privacy: str, ratio: float | None) -> list[dict]:
    return [row for seed in SEEDS for row in data[(adjacency, clients, privacy, ratio, seed)]]


def pooled_auc(data: dict, clients: int, privacy: str, ratio: float | None, key: str) -> float:
    return auc(
        scores(pooled_rows(data, "in-replace", clients, privacy, ratio), key),
        scores(pooled_rows(data, "out-replace", clients, privacy, ratio), key),
    )


def utility(data: dict, adjacency: str, clients: int, privacy: str, ratio: float | None) -> tuple[float, float]:
    values: list[tuple[float, float]] = []
    for seed in SEEDS:
        run_name = data[(adjacency, clients, privacy, ratio, seed)][0]["run_name"]
        path = next(FULL_DIR.rglob(f"{run_name}.json"))
        run = json.loads(path.read_text())
        values.extend(
            (float(run["server_evaluate_metrics"][str(round_number)]["accuracy"]), float(run["server_evaluate_metrics"][str(round_number)]["f1"]))
            for round_number in range(16, 21)
        )
    return statistics.fmean(value[0] for value in values), statistics.fmean(value[1] for value in values)


def table(data: dict, clients: int) -> str:
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Approach & Ratio & Noise multiplier & Clean AUC & Noisy AUC & Acc. IN & Acc. OUT & F1 IN & F1 OUT \\",
        r"\midrule",
    ]
    for privacy, ratio in VARIANTS:
        acc_in, f1_in = utility(data, "in-replace", clients, privacy, ratio)
        acc_out, f1_out = utility(data, "out-replace", clients, privacy, ratio)
        lines.append(
            f"{LABELS[privacy]} & {('--' if ratio is None else f'{ratio:.6f}')} & "
            f"{('' if ratio is None else f'{ratio * clients:.6f}')} & "
            f"{pooled_auc(data, clients, privacy, ratio, 'target_clean_shadow_loss'):.3f} & "
            f"{pooled_auc(data, clients, privacy, ratio, 'target_noisy_shadow_loss'):.3f} & "
            f"{acc_in:.3f} & {acc_out:.3f} & {f1_in:.3f} & {f1_out:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def write_plot(data: dict) -> None:
    image = Image.new("RGB", (1800, 950), "white")
    draw = ImageDraw.Draw(image)
    title, body, small = font(34, True), font(21), font(17)
    draw.text((900, 35), "Pooled ROC AUC across active-client counts", font=title, fill="#222", anchor="ma")
    for index, (key, heading) in enumerate((("target_clean_shadow_loss", "Clean shadow"), ("target_noisy_shadow_loss", "Noisy shadow"))):
        left, right, top, bottom = 150 + index * 870, 770 + index * 870, 165, 680
        draw.text(((left + right) // 2, 110), heading, font=title, fill="#222", anchor="ma")
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - tick * (bottom - top)
            draw.line((left, y, right, y), fill="#e4e8ec", width=2)
            draw.text((left - 12, y), f"{tick:.2f}", font=small, fill="#555", anchor="rm")
        draw.rectangle((left, top, right, bottom), outline="#555", width=2)
        for client in CLIENTS:
            x = left + (client - CLIENTS[0]) / (CLIENTS[-1] - CLIENTS[0]) * (right - left)
            draw.line((x, top, x, bottom), fill="#f0f2f4", width=1)
            draw.text((x, bottom + 14), str(client), font=body, fill="#555", anchor="ma")
        for variant_index, (privacy, ratio) in enumerate(VARIANTS):
            points = []
            for client in CLIENTS:
                x = left + (client - CLIENTS[0]) / (CLIENTS[-1] - CLIENTS[0]) * (right - left)
                y = bottom - pooled_auc(data, client, privacy, ratio, key) * (bottom - top)
                points.append((x, y))
            color = COLORS[(privacy, ratio)]
            draw.line(points, fill=color, width=5)
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            lx = left + (variant_index % 2) * 300
            ly = 740 + (variant_index // 2) * 42
            draw.line((lx, ly, lx + 32, ly), fill=color, width=5)
            label = LABELS[privacy] if ratio is None else f"{LABELS[privacy]} r={ratio:.6f}"
            draw.text((lx + 44, ly), label, font=small, fill="#333", anchor="lm")
    draw.text((900, 710), "Active clients", font=body, fill="#333", anchor="ma")
    image.save(OUT_PNG)


def write_distance_plot(data: dict) -> None:
    image = Image.new("RGB", (1800, 820), "white")
    draw = ImageDraw.Draw(image)
    title, body, small = font(32, True), font(20), font(16)
    draw.text((900, 32), "Metric-privacy maximum pairwise client-model distance", font=title, fill="#222", anchor="ma")
    all_values = []
    trajectories = {}
    for clients in CLIENTS:
        for adjacency in ("in-replace", "out-replace"):
            for ratio in (0.0025, 0.003333, 0.00625):
                series = []
                for seed in SEEDS:
                    run_name = data[(adjacency, clients, "metric-privacy", ratio, seed)][0]["run_name"]
                    run = json.loads(next(FULL_DIR.rglob(f"{run_name}.json")).read_text())
                    series.append([float(run["train_metrics"][str(round_number)]["metric-dp-distance"]) for round_number in range(1, 21)])
                trajectories[(clients, adjacency, ratio)] = np.mean(series, axis=0)
                all_values.extend(np.ravel(series))
    low, high = min(all_values), max(all_values)
    pad = max((high - low) * 0.1, 0.02)
    low, high = low - pad, high + pad
    for panel, clients in enumerate(CLIENTS):
        left, right = 105 + panel * 575, 605 + panel * 575
        top, bottom = 150, 580
        draw.text(((left + right) // 2, 105), f"{clients} active clients", font=body, fill="#222", anchor="ma")
        for tick in np.linspace(low, high, 5):
            y = bottom - (tick - low) * (bottom - top) / (high - low)
            draw.line((left, y, right, y), fill="#e4e8ec", width=1)
            draw.text((left - 9, y), f"{tick:.2f}", font=small, fill="#555", anchor="rm")
        draw.rectangle((left, top, right, bottom), outline="#555", width=2)
        for adjacency, color in (("in-replace", "#2166ac"), ("out-replace", "#d6604d")):
            for ratio_index, ratio in enumerate((0.0025, 0.003333, 0.00625)):
                series = trajectories[(clients, adjacency, ratio)]
                points = [(left + i * (right - left) / 19, bottom - (value - low) * (bottom - top) / (high - low)) for i, value in enumerate(series)]
                draw.line(points, fill=color, width=2 + ratio_index * 2)
        for round_number in (1, 5, 10, 15, 20):
            x = left + (round_number - 1) * (right - left) / 19
            draw.text((x, bottom + 12), str(round_number), font=small, fill="#555", anchor="ma")
    draw.line((615, 670, 650, 670), fill="#2166ac", width=5); draw.text((662, 670), "IN-replace", font=body, fill="#333", anchor="lm")
    draw.line((900, 670, 935, 670), fill="#d6604d", width=5); draw.text((947, 670), "OUT-replace", font=body, fill="#333", anchor="lm")
    draw.text((900, 715), "Line width: ratio 0.002500, 0.003333, 0.006250 (thin to thick)", font=small, fill="#555", anchor="ma")
    draw.text((900, 765), "Communication round", font=body, fill="#333", anchor="ma")
    image.save(OUT_DISTANCE_PNG)


def tex(data: dict) -> str:
    pages = "\n\\clearpage\n".join(
        rf"""\section*{{{clients} active clients}}
\begin{{table}}[h]
\centering
\caption{{Pooled 60 IN versus 60 OUT raw-loss ROC AUC; utility is the mean over rounds 16--20 and seeds 42--44.}}
{table(data, clients)}
\end{{table}}"""
        for clients in CLIENTS
    )
    return rf"""\documentclass[10pt,a4paper,landscape]{{article}}
\usepackage[margin=17mm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\captionsetup{{font=small,labelfont=bf}}
\setlength{{\tabcolsep}}{{6pt}}
\renewcommand{{\arraystretch}}{{1.22}}
\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries CIFAR-10 full replacement CIA results}}\\[4pt]
{{\small IN-replace / OUT-replace \quad non-IID \quad FedAvg \quad seeds 42, 43, 44 \quad 20 rounds \quad 5 local epochs}}
\end{{center}}
\begin{{table}}[h]
\centering
\caption*{{\bfseries Recorded coverage}}
\begin{{tabular}}{{lccc}}
\toprule
Active clients & 3 & 8 & 16 \\
\midrule
Variants per client count & Vanilla + 3 Global-DP + 3 Metric-privacy & Vanilla + 3 Global-DP + 3 Metric-privacy & Vanilla + 3 Global-DP + 3 Metric-privacy \\
Noise ratios (DP) & 0.002500, 0.003333, 0.006250 & 0.002500, 0.003333, 0.006250 & 0.002500, 0.003333, 0.006250 \\
Completed IN/OUT trajectories & 42 / 42 & 42 / 42 & 42 / 42 \\
\bottomrule
\end{{tabular}}
\end{{table}}
{pages}
\clearpage
\begin{{center}}\includegraphics[width=0.94\textwidth]{{results/planned_runs/cifar/{OUT_PNG.name}}}\end{{center}}
\clearpage
\begin{{center}}\includegraphics[width=0.94\textwidth]{{results/planned_runs/cifar/{OUT_DISTANCE_PNG.name}}}\end{{center}}
\vfill
\begin{{center}}\small Source: \texttt{{results/planned\_runs/cifar/full/}}\end{{center}}
\end{{document}}
"""


def main() -> None:
    data = load()
    write_plot(data)
    write_distance_plot(data)
    OUT_TEX.write_text(tex(data))
    engine = shutil.which("pdflatex")
    if engine is None:
        raise RuntimeError("pdflatex is required")
    with tempfile.TemporaryDirectory() as temp:
        for _ in range(2):
            subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", temp, str(OUT_TEX)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        shutil.copyfile(Path(temp) / f"{STEM}.pdf", OUT_PDF)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
