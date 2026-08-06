"""Build the data-only CIFAR-10 noise-sweep and 48-client validation report."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "planned_runs" / "cifar"
SWEEP_DIR = RESULTS_DIR / "noise_sweep"
VALIDATION_DIR = RESULTS_DIR / "validation_48"
REPORT_STEM = "cifar_noise_sweep_48_validation_report"
OUT_TEX = RESULTS_DIR / f"{REPORT_STEM}.tex"
OUT_PDF = RESULTS_DIR / f"{REPORT_STEM}.pdf"
OUT_PNG = RESULTS_DIR / "cifar-noise-sweep-48-validation-accuracy.png"
ROUNDS = (1, 5, 10, 15, 20)
RATIOS = (0.0025, 0.003333, 0.00625)
COLORS = {
    "vanilla": "#2e6db4",
    "global-dp": "#d65a31",
    "metric-privacy": "#24936d",
}


@dataclass(frozen=True)
class Result:
    clients: int
    privacy: str
    ratio: float | None
    noise_multiplier: float
    accuracy: float
    f1: float
    log_loss: float
    auc: float
    trajectory: tuple[float, ...]


def discover_results(root: Path) -> list[Result]:
    results: list[Result] = []
    for run_path in sorted(root.rglob("*.json")):
        if run_path.name.endswith(".evaluation.json") or run_path.name in {
            "chunk_manifest.json",
            "colab_run.json",
        }:
            continue
        run = json.loads(run_path.read_text())
        if "metadata" not in run:
            continue
        metadata = run["metadata"]
        evaluation_path = run_path.with_suffix(".evaluation.json")
        evaluation = json.loads(evaluation_path.read_text())
        privacy = metadata["privacy"]
        clients = int(metadata["num_clients"])
        noise_multiplier = float(metadata["noise_multiplier"])
        ratio = None if privacy == "vanilla" else noise_multiplier / clients
        server_test = evaluation["server_final_test"]
        trajectory = tuple(
            float(run["server_evaluate_metrics"][str(round_number)]["accuracy"])
            for round_number in ROUNDS
        )
        results.append(
            Result(
                clients=clients,
                privacy=privacy,
                ratio=ratio,
                noise_multiplier=noise_multiplier,
                accuracy=float(server_test["accuracy"]),
                f1=float(server_test["averages"]["macro"]["f1"]),
                log_loss=float(server_test["log_loss"]),
                auc=float(server_test["roc_ovr"]["macro_auc"]),
                trajectory=trajectory,
            )
        )
    return results


def fmt(value: float) -> str:
    return f"{value:.4f}"


def display_privacy(privacy: str) -> str:
    return {"vanilla": "Vanilla", "global-dp": "Global-DP", "metric-privacy": "Metric-privacy"}[privacy]


def ordered(results: list[Result]) -> list[Result]:
    order = {"vanilla": 0, "global-dp": 1, "metric-privacy": 2}
    return sorted(results, key=lambda item: (item.ratio is not None, item.ratio or 0, order[item.privacy]))


def write_plot(sweep: list[Result], validation: list[Result]) -> None:
    width, height = 1700, 940
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
    label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    draw.text((width // 2, 30), "Server-test accuracy over communication rounds", fill="#202124", font=title_font, anchor="ma")
    panels = ((sweep, "3 active clients"), (validation, "48 active clients"))
    for panel_index, (results, label) in enumerate(panels):
        left = 125 + panel_index * 825
        top, right, bottom = 175, left + 650, 780
        draw.text(((left + right) // 2, 120), label, fill="#202124", font=title_font, anchor="ma")
        for tick in (0.4, 0.5, 0.6, 0.7, 0.8):
            y = bottom - (tick - 0.2) / 0.65 * (bottom - top)
            draw.line((left, y, right, y), fill="#e6e9ed", width=2)
            draw.text((left - 15, y), f"{tick:.1f}", fill="#555", font=small_font, anchor="rm")
        draw.rectangle((left, top, right, bottom), outline="#555", width=2)
        for round_number in ROUNDS:
            x = left + (round_number - 1) / 19 * (right - left)
            draw.line((x, top, x, bottom), fill="#f1f3f5", width=1)
            draw.text((x, bottom + 14), str(round_number), fill="#555", font=small_font, anchor="ma")
        for result in ordered(results):
            points = []
            for round_number, accuracy in zip(ROUNDS, result.trajectory, strict=True):
                x = left + (round_number - 1) / 19 * (right - left)
                y = bottom - (accuracy - 0.2) / 0.65 * (bottom - top)
                points.append((x, y))
            draw.line(points, fill=COLORS[result.privacy], width=5)
            for point in points:
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=COLORS[result.privacy])
            ratio_label = "vanilla" if result.ratio is None else f"r={result.ratio:.6f}"
            legend_index = ordered(results).index(result)
            legend_x = left + (0 if legend_index < 4 else 330)
            legend_y = 842 + (legend_index % 4) * 25
            draw.line((legend_x, legend_y, legend_x + 28, legend_y), fill=COLORS[result.privacy], width=5)
            draw.text((legend_x + 38, legend_y), f"{display_privacy(result.privacy)} ({ratio_label})", fill="#333", font=small_font, anchor="lm")
        draw.text(((left + right) // 2, bottom + 25), "Communication round", fill="#333", font=label_font, anchor="ma")
    draw.text((48, (top + bottom) // 2), "Accuracy", fill="#333", font=label_font, anchor="mm")
    image.save(OUT_PNG)


def result_rows(results: list[Result]) -> str:
    return "\n".join(
        " & ".join(
            (
                display_privacy(item.privacy),
                "--" if item.ratio is None else f"{item.ratio:.6f}",
                f"{item.noise_multiplier:.6f}",
                fmt(item.accuracy),
                fmt(item.f1),
                fmt(item.log_loss),
                fmt(item.auc),
                " / ".join(fmt(value) for value in item.trajectory),
            )
        )
        + r" \\"
        for item in ordered(results)
    )


def delta_rows(sweep: list[Result], validation: list[Result]) -> str:
    rows: list[str] = []
    for ratio in RATIOS:
        row = [f"{ratio:.6f}"]
        for results in (sweep, validation):
            vanilla = next(item.accuracy for item in results if item.privacy == "vanilla")
            for privacy in ("global-dp", "metric-privacy"):
                result = next(item for item in results if item.privacy == privacy and abs(item.ratio - ratio) < 1e-7)
                row.append(f"{result.accuracy - vanilla:+.4f}")
        rows.append(" & ".join(row) + r" \\")
    return "\n".join(rows)


def build_tex(sweep: list[Result], validation: list[Result]) -> str:
    return rf"""\documentclass[10pt,a4paper,landscape]{{article}}
\usepackage[margin=13mm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{array}}
\usepackage{{xcolor}}
\usepackage{{caption}}
\pagestyle{{plain}}
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.20}}
\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries CIFAR-10 noise-ratio sweep and 48-client validation}}\\[4pt]
{{\small IN-replace \quad non-IID \quad FedAvg \quad seed 42 \quad 20 rounds \quad 5 local epochs \quad clipping norm 5.0}}
\end{{center}}

\begin{{table}}[h]
\centering
\caption*{{\bfseries Recorded configuration}}
\begin{{tabular}}{{lcc}}
\toprule
& 3-client sweep & 48-client validation \\
\midrule
Active / canonical clients & 3 / 4 & 48 / 49 \\
Dataset / server test & 4-class CIFAR-10 / 2,000 examples & 4-class CIFAR-10 / 2,000 examples \\
Privacy modes & Vanilla, Global-DP, Metric-privacy & Vanilla, Global-DP, Metric-privacy \\
Noise ratios & 0.002500, 0.003333, 0.006250 & 0.002500, 0.003333, 0.006250 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[h]
\centering
\caption*{{\bfseries Noise calibration: ratio = noise multiplier / active clients; sigma = ratio $\times$ 5}}
\begin{{tabular}}{{rrrr}}
\toprule
Ratio & 3-client multiplier & 48-client multiplier & Applied sigma \\
\midrule
0.002500 & 0.007500 & 0.120000 & 0.012500 \\
0.003333 & 0.009999 & 0.159984 & 0.016665 \\
0.006250 & 0.018750 & 0.300000 & 0.031250 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[h]
\centering
\caption*{{\bfseries Final server-test metrics: 3 active clients}}
\scriptsize
\begin{{tabular}}{{llrrrrrrl}}
\toprule
Privacy & Ratio & Multiplier & Accuracy & Macro F1 & Log loss & Macro OVR AUC & Accuracy at rounds 1 / 5 / 10 / 15 / 20 \\
\midrule
{result_rows(sweep)}
\bottomrule
\end{{tabular}}
\end{{table}}

\newpage
\begin{{table}}[h]
\centering
\caption*{{\bfseries Final server-test metrics: 48 active clients}}
\scriptsize
\begin{{tabular}}{{llrrrrrrl}}
\toprule
Privacy & Ratio & Multiplier & Accuracy & Macro F1 & Log loss & Macro OVR AUC & Accuracy at rounds 1 / 5 / 10 / 15 / 20 \\
\midrule
{result_rows(validation)}
\bottomrule
\end{{tabular}}
\end{{table}}

\newpage
\begin{{center}}
\includegraphics[width=0.75\textwidth]{{{OUT_PNG.name}}}
\end{{center}}

\begin{{table}}[h]
\centering
\caption*{{\bfseries Final-accuracy difference from vanilla (DP accuracy minus vanilla accuracy)}}
\begin{{tabular}}{{lrrrr}}
\toprule
Ratio & 3-client Global-DP & 3-client Metric-privacy & 48-client Global-DP & 48-client Metric-privacy \\
\midrule
{delta_rows(sweep, validation)}
\bottomrule
\end{{tabular}}
\end{{table}}

\vfill
\begin{{center}}\small Source: \texttt{{results/planned\_runs/cifar/noise\_sweep/}} and \texttt{{results/planned\_runs/cifar/validation\_48/}}\end{{center}}
\end{{document}}
"""


def main() -> None:
    sweep = discover_results(SWEEP_DIR)
    validation = discover_results(VALIDATION_DIR)
    if len(sweep) != 7 or len(validation) != 7:
        raise RuntimeError(f"Expected seven results per group; found {len(sweep)} and {len(validation)}.")
    write_plot(sweep, validation)
    OUT_TEX.write_text(build_tex(sweep, validation), encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", OUT_TEX.name],
        check=True,
        cwd=RESULTS_DIR,
        stdout=subprocess.DEVNULL,
    )
    for suffix in (".aux", ".log"):
        (RESULTS_DIR / f"{REPORT_STEM}{suffix}").unlink(missing_ok=True)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
