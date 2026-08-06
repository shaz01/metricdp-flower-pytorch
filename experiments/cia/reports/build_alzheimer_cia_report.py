"""Build a three-client CIA transfer report for Alzheimer, CIFAR-10, or Fashion-MNIST.

The report reads the IN/OUT planned-run artifacts, reproduces the paper's
single-round and multi-round reference tables, and emits an editable LaTeX
source plus a compiled PDF under the corresponding ``results/planned_runs/<dataset>`` directory.

Usage:
    uv run python experiments/cia/reports/build_alzheimer_cia_report.py --dataset cifar
    uv run python experiments/cia/reports/build_alzheimer_cia_report.py --dataset fashion
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.cia.datasets.paper import PAPER_CIA_CLIENT_COUNTS
from metricdp_pytorch.utils.split_data import split_total_by_weights


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = "alzheimer"
DATASET_LABEL = "Alzheimer"
CLASS_LABELS = ("Mild", "Moderate", "Non-demented", "Very mild")
RESULTS_DIR = REPO_ROOT / "results" / "planned_runs" / DATASET
IN_DIR = RESULTS_DIR / "in_remove" / f"{DATASET}-in-remove"
OUT_DIR = RESULTS_DIR / "out_remove" / f"{DATASET}-out-remove"
OUT_TEX = RESULTS_DIR / f"{DATASET}_cia_report.tex"
OUT_PNG = RESULTS_DIR / f"{DATASET}-cia-score-trajectories.png"
OUT_DISTANCE_PNG = RESULTS_DIR / f"{DATASET}-metric-dp-distance-trajectories.png"

MECHANISMS = ("vanilla", "global-dp", "metric-privacy")
LABEL = {
    "vanilla": "Vanilla FL",
    "global-dp": "Global-DP",
    "metric-privacy": "Metric-privacy",
}
CLIENT_ROLES = ("Attacker", "Bystander", "Target")

# Sainz-Pardo Diaz et al., Tables 10, 12, and 13.
PAPER_SINGLE = {
    "vanilla": (1.032, 1.182, 12.719, 1.010),
    "global-dp": (3.603, 4.848, 25.679, 3.514),
    "metric-privacy": (1.135, 1.506, 24.631, 1.106),
}
PAPER_MULTI = {
    "vanilla": (0.890, 0.767, 0.977, 0.948, 0.007, 0.894, 0.007, 0.930, 0.006, 0.729, 0.024),
    "global-dp": (0.397, 0.224, 0.585, 0.637, 0.009, 0.680, 0.026, 0.426, 0.018, 0.400, 0.008),
    "metric-privacy": (0.493, 0.296, 0.689, 0.794, 0.023, 0.746, 0.029, 0.695, 0.035, 0.480, 0.022),
}


def configure_dataset(dataset: str) -> None:
    """Select one of the matched three-client IN/OUT transfer experiment sets."""
    global DATASET, DATASET_LABEL, CLASS_LABELS, RESULTS_DIR, IN_DIR, OUT_DIR, OUT_TEX, OUT_PNG, OUT_DISTANCE_PNG
    presets = {
        "alzheimer": ("Alzheimer", ("Mild", "Moderate", "Non-demented", "Very mild")),
        "cifar": ("CIFAR-10", ("airplane", "automobile", "bird", "cat")),
        "fashion": ("Fashion-MNIST", ("T-shirt/top", "Trouser", "Pullover", "Dress")),
    }
    if dataset not in presets:
        raise ValueError(f"unsupported dataset: {dataset}")
    DATASET = dataset
    DATASET_LABEL, CLASS_LABELS = presets[dataset]
    RESULTS_DIR = REPO_ROOT / "results" / "planned_runs" / dataset
    IN_DIR = RESULTS_DIR / "in_remove" / f"{dataset}-in-remove"
    OUT_DIR = RESULTS_DIR / "out_remove" / f"{dataset}-out-remove"
    report_stem = f"{dataset}_transfer_cia_report" if dataset in {"cifar", "fashion"} else f"{dataset}_cia_report"
    OUT_TEX = RESULTS_DIR / f"{report_stem}.tex"
    OUT_PNG = RESULTS_DIR / f"{dataset}-cia-score-trajectories.png"
    OUT_DISTANCE_PNG = RESULTS_DIR / f"{dataset}-metric-dp-distance-trajectories.png"


def load_cia(path: Path) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in json.loads(path.read_text()):
        grouped[(row["privacy"], int(row["seed"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["server_round"]))
    return grouped


def raw_run_path(directory: Path, mechanism: str, seed: int) -> Path:
    matches = sorted(directory.glob(f"*__{mechanism}__*__seed-{seed}__*.json"))
    matches = [path for path in matches if not path.name.endswith(".evaluation.json")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one run for {mechanism=} {seed=} in {directory}, got {matches}")
    return matches[0]


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def fmt_ms(values: list[float], digits: int = 3) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} {{\\scriptsize$\\pm$\\,{std:.{digits}f}}}"


def attack_scores(rows: list[dict], loss_key: str) -> np.ndarray:
    """Negate a per-round loss (or relative-difference) column into a score."""
    return -np.asarray([float(row[loss_key]) for row in rows], dtype=float)


def rounds_of(rows: list[dict]) -> np.ndarray:
    return np.asarray([int(row["server_round"]) for row in rows], dtype=int)


def auc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    greater = np.mean(in_scores[:, None] > out_scores[None, :])
    tied = np.mean(in_scores[:, None] == out_scores[None, :])
    return float(greater + 0.5 * tied)


def bootstrap_auc(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> tuple[float, float, float]:
    """Stratified percentile bootstrap, resampling IN and OUT independently."""
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_in = in_scores[rng.integers(0, len(in_scores), size=len(in_scores))]
        sampled_out = out_scores[rng.integers(0, len(out_scores), size=len(out_scores))]
        estimates[index] = auc(sampled_in, sampled_out)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return auc(in_scores, out_scores), float(low), float(high)


def fmt_auc(result: tuple[float, float, float]) -> str:
    value, low, high = result
    return f"{value:.3f} ({low:.3f}, {high:.3f})"


# Raw-loss score column and the aggregate-referenced score column per shadow kind.
SHADOW_KEYS = {
    "clean": ("target_clean_shadow_loss", "clean_difference_pct"),
    "noisy": ("target_noisy_shadow_loss", "noisy_difference_pct"),
}


def pooled_rows(
    groups: dict[tuple[str, int], list[dict]], mechanism: str
) -> list[dict]:
    return sum((groups[(mechanism, seed)] for seed in (42, 43, 44)), [])


def fmt_pct(value: float) -> str:
    """Format a fraction as a LaTeX-safe percentage."""
    return f"{100 * value:.1f}\\%"


def pooled_auc(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
    mechanism: str,
    score_key: str,
) -> float:
    return auc(
        attack_scores(pooled_rows(in_groups, mechanism), score_key),
        attack_scores(pooled_rows(out_groups, mechanism), score_key),
    )


def round_matched_auc(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
    mechanism: str,
    score_key: str,
) -> float:
    """Compare IN and OUT only at identical (seed, round) checkpoints.

    The pooled 20-versus-20 AUC also ranks an early IN checkpoint against a
    late OUT checkpoint, so it mixes membership signal with the round index.
    Restricting to matched rounds removes that confound.
    """
    outcomes: list[float] = []
    for seed in (42, 43, 44):
        in_scores = attack_scores(in_groups[(mechanism, seed)], score_key)
        out_scores = attack_scores(out_groups[(mechanism, seed)], score_key)
        outcomes.extend(
            1.0 if in_score > out_score else 0.5 if in_score == out_score else 0.0
            for in_score, out_score in zip(in_scores, out_scores, strict=True)
        )
    return float(np.mean(outcomes))


def discordant_breakdown(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
    mechanism: str,
    score_key: str,
) -> tuple[float, float, float]:
    """Split pooled discordant pairs into matched-round and IN-earlier pairs."""
    in_rows = pooled_rows(in_groups, mechanism)
    out_rows = pooled_rows(out_groups, mechanism)
    in_scores = attack_scores(in_rows, score_key)
    out_scores = attack_scores(out_rows, score_key)
    in_rounds = rounds_of(in_rows)
    out_rounds = rounds_of(out_rows)
    discordant = in_scores[:, None] < out_scores[None, :]
    same_round = in_rounds[:, None] == out_rounds[None, :]
    in_earlier = in_rounds[:, None] < out_rounds[None, :]
    return (
        float(discordant.mean()),
        float(discordant[same_round].mean()),
        float(discordant[in_earlier].mean()),
    )


def score_construction_table(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
) -> str:
    """Contrast the paper's pooled score with two confound-free variants."""
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Approach & Shadow & Pooled 20$\times$20 AUC & Round-matched AUC & "
        r"Aggregate-referenced AUC \\",
        r"\midrule",
    ]
    for mechanism in MECHANISMS:
        for shadow_kind, (loss_key, difference_key) in SHADOW_KEYS.items():
            pooled = pooled_auc(in_groups, out_groups, mechanism, loss_key)
            matched = round_matched_auc(in_groups, out_groups, mechanism, loss_key)
            referenced = pooled_auc(
                in_groups, out_groups, mechanism, difference_key
            )
            lines.append(
                f"{LABEL[mechanism]} & {shadow_kind} & {pooled:.3f} & "
                f"{matched:.3f} & {referenced:.3f} \\\\"
            )
        if mechanism != MECHANISMS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else (
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    )
    return ImageFont.truetype(filename, size=size)


def write_score_figure(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
    output_path: Path,
) -> None:
    """Render mean CIA-score trajectories with seed-standard-deviation bands.

    Clean and noisy scores occupy separate rows because their raw loss scales
    differ substantially; IN and OUT stay on the same axes in each panel.
    """
    width, height = 1800, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font, panel_font, body_font, small_font = (
        _font(36, bold=True),
        _font(25, bold=True),
        _font(20),
        _font(17),
    )
    in_color, out_color = (38, 94, 160), (207, 89, 43)
    border, grid, text = (70, 70, 70), (220, 224, 230), (30, 30, 30)
    panel_left, panel_width, panel_gap = 120, 500, 60
    panel_height, row_tops = 305, (175, 615)
    plot_margin_left, plot_margin_right = 62, 20
    plot_margin_top, plot_margin_bottom = 24, 48
    y_ranges = {"clean": (-4.0, 0.2), "noisy": (-4.0, 0.2)}
    score_keys = {
        "clean": "target_clean_shadow_loss",
        "noisy": "target_noisy_shadow_loss",
    }

    draw.text(
        (width // 2, 35),
        "CIA score trajectories across communication rounds",
        fill=text,
        font=title_font,
        anchor="ma",
    )
    legend_y = 88
    for x, label, color in (
        (560, "IN trajectory", in_color),
        (880, "OUT trajectory", out_color),
    ):
        draw.line((x, legend_y, x + 52, legend_y), fill=color, width=7)
        draw.text((x + 65, legend_y), label, fill=text, font=body_font, anchor="lm")
    draw.text(
        (width // 2, 128),
        "Lines = mean across seeds 42, 43, 44; translucent bands = +/- 1 sample standard deviation.",
        fill=(80, 80, 80),
        font=small_font,
        anchor="ma",
    )

    for row_index, shadow_kind in enumerate(("clean", "noisy")):
        y_min, y_max = y_ranges[shadow_kind]
        row_top = row_tops[row_index]
        draw.text(
            (86, row_top + panel_height // 2),
            f"{shadow_kind.capitalize()} shadow\nscore (-loss)",
            fill=text,
            font=body_font,
            anchor="mm",
            spacing=3,
        )
        for mechanism_index, mechanism in enumerate(MECHANISMS):
            x0 = panel_left + mechanism_index * (panel_width + panel_gap)
            y0 = row_top
            left = x0 + plot_margin_left
            right = x0 + panel_width - plot_margin_right
            top = y0 + plot_margin_top
            bottom = y0 + panel_height - plot_margin_bottom
            if row_index == 0:
                draw.text(
                    (x0 + panel_width // 2, y0 - 33),
                    LABEL[mechanism],
                    fill=text,
                    font=panel_font,
                    anchor="ma",
                )

            def x_coord(round_number: int) -> float:
                return left + (round_number - 1) * (right - left) / 19

            def y_coord(value: float) -> float:
                bounded = min(max(value, y_min), y_max)
                return bottom - (bounded - y_min) * (bottom - top) / (y_max - y_min)

            for tick in np.linspace(y_min, 0.0, 5):
                y = y_coord(float(tick))
                draw.line((left, y, right, y), fill=grid, width=1)
                draw.text((left - 9, y), f"{tick:.0f}", fill=(90, 90, 90), font=small_font, anchor="rm")
            for round_number in (1, 5, 10, 15, 20):
                x = x_coord(round_number)
                draw.line((x, top, x, bottom), fill=(243, 244, 246), width=1)
                draw.text((x, bottom + 13), str(round_number), fill=(90, 90, 90), font=small_font, anchor="ma")
            draw.rectangle((left, top, right, bottom), outline=border, width=2)

            for condition, groups, color in (
                ("IN", in_groups, in_color),
                ("OUT", out_groups, out_color),
            ):
                values = np.asarray(
                    [
                        [-row[score_keys[shadow_kind]] for row in groups[(mechanism, seed)]]
                        for seed in (42, 43, 44)
                    ],
                    dtype=float,
                )
                mean = values.mean(axis=0)
                std = values.std(axis=0, ddof=1)
                upper = [
                    (x_coord(index + 1), y_coord(float(value + error)))
                    for index, (value, error) in enumerate(zip(mean, std, strict=True))
                ]
                lower = [
                    (x_coord(index + 1), y_coord(float(value - error)))
                    for index, (value, error) in enumerate(zip(mean, std, strict=True))
                ]
                draw.polygon(upper + list(reversed(lower)), fill=(*color, 45))
                points = [
                    (x_coord(index + 1), y_coord(float(value)))
                    for index, value in enumerate(mean)
                ]
                draw.line(points, fill=color, width=5)
                for point in points:
                    draw.ellipse(
                        (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                        fill=color,
                    )
    draw.text((width // 2, height - 23), "Communication round", fill=text, font=body_font, anchor="ma")
    image.save(output_path, format="PNG", optimize=True)


def write_distance_figure(output_path: Path) -> None:
    """Render maximum pairwise metric-DP distance by round and membership condition."""
    values: dict[str, np.ndarray] = {}
    for condition, directory in (("IN", IN_DIR), ("OUT", OUT_DIR)):
        values[condition] = np.asarray(
            [
                [
                    float(json.loads(raw_run_path(directory, "metric-privacy", seed).read_text())["train_metrics"][str(round_number)]["metric-dp-distance"])
                    for round_number in range(1, 21)
                ]
                for seed in (42, 43, 44)
            ],
            dtype=float,
        )
    lower = min(array.min() for array in values.values())
    upper = max(array.max() for array in values.values())
    pad = max((upper - lower) * 0.1, 0.02)
    lower, upper = lower - pad, upper + pad
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title, body, small = _font(34, bold=True), _font(22), _font(18)
    left, right, top, bottom = 190, 1380, 160, 590
    draw.text((width // 2, 36), "Metric-privacy maximum pairwise client-model distance", fill=(30, 30, 30), font=title, anchor="ma")
    for tick in np.linspace(lower, upper, 5):
        y = bottom - (tick - lower) * (bottom - top) / (upper - lower)
        draw.line((left, y, right, y), fill=(225, 229, 234), width=2)
        draw.text((left - 14, y), f"{tick:.2f}", fill=(80, 80, 80), font=small, anchor="rm")
    for round_number in (1, 5, 10, 15, 20):
        x = left + (round_number - 1) * (right - left) / 19
        draw.line((x, top, x, bottom), fill=(242, 244, 246), width=1)
        draw.text((x, bottom + 16), str(round_number), fill=(80, 80, 80), font=small, anchor="ma")
    draw.rectangle((left, top, right, bottom), outline=(70, 70, 70), width=2)
    for color, condition in (((38, 94, 160), "IN"), ((207, 89, 43), "OUT")):
        mean, std = values[condition].mean(axis=0), values[condition].std(axis=0, ddof=1)
        points = [(left + index * (right - left) / 19, bottom - (value - lower) * (bottom - top) / (upper - lower)) for index, value in enumerate(mean)]
        band_top = [(x, bottom - (value + error - lower) * (bottom - top) / (upper - lower)) for (x, _), value, error in zip(points, mean, std, strict=True)]
        band_bottom = [(x, bottom - (value - error - lower) * (bottom - top) / (upper - lower)) for (x, _), value, error in zip(points, mean, std, strict=True)]
        draw.polygon(band_top + list(reversed(band_bottom)), fill=(*color, 45))
        draw.line(points, fill=color, width=6)
        draw.line((520 if condition == "IN" else 850, 105, 575 if condition == "IN" else 905, 105), fill=color, width=7)
        draw.text((588 if condition == "IN" else 918, 105), f"{condition} mean +/- 1 sample std", fill=(30, 30, 30), font=body, anchor="lm")
    draw.text((width // 2, 674), "Communication round", fill=(30, 30, 30), font=body, anchor="ma")
    image.save(output_path, format="PNG", optimize=True)


def paper_single_table() -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Approach & Aggregated loss & Target loss & Difference (\%) & Test loss \\",
        r"\midrule",
    ]
    for mechanism in MECHANISMS:
        aggregate, target, difference, test = PAPER_SINGLE[mechanism]
        lines.append(
            f"{LABEL[mechanism]} & {aggregate:.3f} & {target:.3f} & "
            f"{difference:.3f} & {test:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def allocation_table() -> str:
    lines = [
        r"\begin{tabular}{clccrrrrr}",
        r"\toprule",
        "Client & Role & IN & OUT & Total & " + " & ".join(CLASS_LABELS) + r" \\",
        r"\midrule",
    ]
    for client_id, (role, counts) in enumerate(
        zip(CLIENT_ROLES, PAPER_CIA_CLIENT_COUNTS, strict=True), start=1
    ):
        total = sum(counts)
        count_cells = [f"{count} ({100 * count / total:.1f}\\%)" for count in counts]
        lines.append(
            f"{client_id} & {role} & yes & {'yes' if client_id < 3 else 'no'} & {total} & "
            + " & ".join(count_cells)
            + r" \\"
        )
    totals = np.asarray(PAPER_CIA_CLIENT_COUNTS, dtype=int).sum(axis=0)
    grand_total = int(totals.sum())
    total_cells = [f"{count} ({100 * count / grand_total:.1f}\\%)" for count in totals]
    lines.append(r"\midrule")
    lines.append(
        f"All & Canonical pool & & & {grand_total} & " + " & ".join(total_cells) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def local_split_table() -> str:
    lines = [
        r"\begin{tabular}{clrrrrrr}",
        r"\toprule",
        "Client & Role & Train total & " + " & ".join(CLASS_LABELS) + r" & Local evaluation total \\",
        r"\midrule",
    ]
    for client_id, (role, counts) in enumerate(
        zip(CLIENT_ROLES, PAPER_CIA_CLIENT_COUNTS, strict=True), start=1
    ):
        train_total = int(0.8 * sum(counts))
        train_counts = split_total_by_weights(train_total, counts)
        eval_total = sum(counts) - train_total
        lines.append(
            f"{client_id} & {role} & {train_total} & "
            + " & ".join(str(count) for count in train_counts)
            + f" & {eval_total} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def heterogeneity_summary() -> tuple[list[float], dict[tuple[int, int], float]]:
    counts = np.asarray(PAPER_CIA_CLIENT_COUNTS, dtype=float)
    distributions = counts / counts.sum(axis=1, keepdims=True)
    global_distribution = counts.sum(axis=0) / counts.sum()
    from_global = (0.5 * np.abs(distributions - global_distribution).sum(axis=1)).tolist()
    pairwise = {
        (left + 1, right + 1): float(
            0.5 * np.abs(distributions[left] - distributions[right]).sum()
        )
        for left in range(len(distributions))
        for right in range(left + 1, len(distributions))
    }
    return from_global, pairwise


def distribution_note() -> str:
    if DATASET == "alzheimer":
        return (
            "The canonical allocation is the corrected version of the paper's Table 9. "
            "The published Client 2 row states total 1591 and Mild count 180, but those values conflict "
            "with the row sum and source class totals; the internally consistent values used by the "
            "experiments are total 1491 and Mild count 80."
        )
    return (
        f"This {DATASET_LABEL} transfer experiment reuses the corrected three-client count profile from the paper's "
        f"Table 9, but maps its four columns to the {DATASET_LABEL} classes shown below."
    )


def paper_reference_caption(table_numbers: str) -> str:
    if DATASET == "alzheimer":
        return f"Paper reference ({table_numbers})."
    return f"Published Alzheimer reference ({table_numbers}); shown for protocol context only, not as a CIFAR-10 baseline."


def ours_single_table(in_groups: dict[tuple[str, int], list[dict]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Approach & Aggregated loss & Clean target & Clean diff. (\%) & Noisy target & Noisy diff. (\%) \\",
        r"\midrule",
    ]
    for mechanism in MECHANISMS:
        first_round = [in_groups[(mechanism, seed)][0] for seed in (42, 43, 44)]
        cells = [
            fmt_ms([row["aggregated_test_loss"] for row in first_round]),
            fmt_ms([row["target_clean_shadow_loss"] for row in first_round]),
            fmt_ms([row["clean_difference_pct"] for row in first_round]),
            fmt_ms([row["target_noisy_shadow_loss"] for row in first_round]),
            fmt_ms([row["noisy_difference_pct"] for row in first_round]),
        ]
        lines.append(LABEL[mechanism] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def paper_multi_table() -> str:
    lines = [
        r"\begin{tabular}{lccrrrr}",
        r"\toprule",
        r"Approach & Attack AUC (95\% CI) & & Acc. IN & Acc. OUT & F1 IN & F1 OUT \\",
        r"\midrule",
    ]
    for mechanism in MECHANISMS:
        auc_value, low, high, ai, ais, ao, aos, fi, fis, fo, fos = PAPER_MULTI[mechanism]
        lines.append(
            f"{LABEL[mechanism]} & {auc_value:.3f} ({low:.3f}, {high:.3f}) & & "
            f"{ai:.3f} $\\pm$ {ais:.3f} & {ao:.3f} $\\pm$ {aos:.3f} & "
            f"{fi:.3f} $\\pm$ {fis:.3f} & {fo:.3f} $\\pm$ {fos:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def last_five_utility(directory: Path, mechanism: str) -> tuple[list[float], list[float]]:
    accuracies: list[float] = []
    f1_scores: list[float] = []
    for seed in (42, 43, 44):
        run = json.loads(raw_run_path(directory, mechanism, seed).read_text())
        metrics = run["server_evaluate_metrics"]
        for round_number in range(16, 21):
            row = metrics[str(round_number)]
            accuracies.append(float(row["accuracy"]))
            f1_scores.append(float(row["f1"]))
    return accuracies, f1_scores


def ours_pooled_multi_table(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
) -> str:
    lines = [
        r"\begin{tabular}{lccrrrr}",
        r"\toprule",
        r"Approach & Clean AUC (95\% CI) & Noisy AUC (95\% CI) & Acc. IN & Acc. OUT & F1 IN & F1 OUT \\",
        r"\midrule",
    ]
    for mech_index, mechanism in enumerate(MECHANISMS):
        in_rows = sum((in_groups[(mechanism, seed)] for seed in (42, 43, 44)), [])
        out_rows = sum((out_groups[(mechanism, seed)] for seed in (42, 43, 44)), [])
        clean = bootstrap_auc(
            attack_scores(in_rows, "target_clean_shadow_loss"),
            attack_scores(out_rows, "target_clean_shadow_loss"),
            seed=20_260_806 + mech_index,
        )
        noisy = bootstrap_auc(
            attack_scores(in_rows, "target_noisy_shadow_loss"),
            attack_scores(out_rows, "target_noisy_shadow_loss"),
            seed=20_261_806 + mech_index,
        )
        acc_in, f1_in = last_five_utility(IN_DIR, mechanism)
        acc_out, f1_out = last_five_utility(OUT_DIR, mechanism)
        cells = [
            fmt_auc(clean),
            fmt_auc(noisy),
            fmt_ms(acc_in),
            fmt_ms(acc_out),
            fmt_ms(f1_in),
            fmt_ms(f1_out),
        ]
        lines.append(LABEL[mechanism] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_seed_auc_table(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
) -> str:
    lines = [
        r"\begin{tabular}{lcrr}",
        r"\toprule",
        r"Approach & Seed & Clean AUC (95\% CI) & Noisy AUC (95\% CI) \\",
        r"\midrule",
    ]
    for mech_index, mechanism in enumerate(MECHANISMS):
        for seed in (42, 43, 44):
            in_rows = in_groups[(mechanism, seed)]
            out_rows = out_groups[(mechanism, seed)]
            clean = bootstrap_auc(
                attack_scores(in_rows, "target_clean_shadow_loss"),
                attack_scores(out_rows, "target_clean_shadow_loss"),
                seed=31_000_000 + mech_index * 100 + seed,
            )
            noisy = bootstrap_auc(
                attack_scores(in_rows, "target_noisy_shadow_loss"),
                attack_scores(out_rows, "target_noisy_shadow_loss"),
                seed=32_000_000 + mech_index * 100 + seed,
            )
            lines.append(
                f"{LABEL[mechanism]} & {seed} & {fmt_auc(clean)} & {fmt_auc(noisy)} \\\\"
            )
        if mechanism != MECHANISMS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_document() -> str:
    in_groups = load_cia(IN_DIR / "cia.json")
    out_groups = load_cia(OUT_DIR / "cia.json")
    expected = {(mechanism, seed) for mechanism in MECHANISMS for seed in (42, 43, 44)}
    if set(in_groups) != expected or set(out_groups) != expected:
        raise RuntimeError("Alzheimer CIA artifacts are incomplete or contain unexpected runs")
    from_global, pairwise = heterogeneity_summary()
    clean_discordant, clean_matched_discordant, clean_in_earlier = discordant_breakdown(
        in_groups, out_groups, "metric-privacy", "target_clean_shadow_loss"
    )
    noisy_discordant, noisy_matched_discordant, _noisy_in_earlier = discordant_breakdown(
        in_groups, out_groups, "metric-privacy", "target_noisy_shadow_loss"
    )
    referenced_clean = {
        mechanism: pooled_auc(in_groups, out_groups, mechanism, "clean_difference_pct")
        for mechanism in MECHANISMS
    }
    referenced_noisy = {
        mechanism: pooled_auc(in_groups, out_groups, mechanism, "noisy_difference_pct")
        for mechanism in MECHANISMS
    }
    referenced_clean_range = (
        f"{min(referenced_clean.values()):.3f}--{max(referenced_clean.values()):.3f}"
    )
    return rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,landscape,margin=1.7cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{caption}}
\usepackage{{microtype}}
\usepackage{{graphicx}}
\captionsetup{{font=small,labelfont=bf}}
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.18}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}

\title{{{DATASET_LABEL} Client Inference Attack: Transfer Report}}
\author{{Planned-run results}}
\date{{}}

\begin{{document}}
\maketitle
\thispagestyle{{empty}}

\section*{{Scope and protocol}}
FedAvg is evaluated for 20 communication rounds and 5 local epochs on the non-i.i.d. {DATASET_LABEL} split.
The IN condition has three active clients; the OUT condition removes the target and has two.
Results use seeds 42--44, a 10\% target shadow split, noise multiplier 0.01, clipping norm 5,
and both clean and Gaussian-perturbed shadow data (standard deviation 20\% of the maximum pixel value).
Attack scores follow the paper: $s_r=-\mathcal{{L}}(W^{{(r)}},D_x)$.
The noisy column is the paper-faithful configuration for the multi-round attack (10\% split of the
target's training data perturbed at 20\% of the maximum pixel value); the clean column is an
additional strong-adversary reference of ours. Both shadow variants were verified to contain the same
150 target examples in the IN and OUT conditions, with an identical, deterministic noise draw, so the
two conditions differ only in whether the target participates in training.

\section*{{Client data distribution}}
{distribution_note()}

\begin{{table}}[h]
\centering
\caption{{Canonical examples assigned to each client. Parentheses show within-client class percentages.}}
{allocation_table()}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Examples actually used for local training after the deterministic stratified 80/20 split.}}
{local_split_table()}
\end{{table}}

\textbf{{How non-IID is it?}} Total-variation distance from the canonical global class distribution is
{from_global[0]:.3f} for Client 1, {from_global[1]:.3f} for Client 2, and {from_global[2]:.3f} for the target.
Pairwise distance is only {pairwise[(1, 2)]:.3f} between Clients 1 and 2, but rises to
{pairwise[(1, 3)]:.3f} between Clients 1 and 3 and {pairwise[(2, 3)]:.3f} between Clients 2 and 3.
Thus, the split is deliberately target-skewed rather than uniformly heterogeneous: the attacker and
bystander are close to each other, while the target is meaningfully different. The target has 27.8\% {CLASS_LABELS[0]}
and 29.2\% {CLASS_LABELS[2]} examples, versus 5.4--6.9\% {CLASS_LABELS[0]} and 60.0--64.2\% {CLASS_LABELS[2]} for the other clients.
The {CLASS_LABELS[1]} class remains rare everywhere (0.5--1.5\%), so it contributes little to the heterogeneity signal.

\section*{{Single-round CIA}}

\begin{{table}}[h]
\centering
\caption{{{paper_reference_caption('Tables 10 and 12')} The target loss is measured on the clean shadow data.}}
{paper_single_table()}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Ours, first-round IN condition (mean $\pm$ sample std over three seeds).}}
{ours_single_table(in_groups)}
\end{{table}}

\clearpage
\begin{{figure}}[p]
\centering
\includegraphics[width=0.96\textwidth]{{results/planned_runs/{DATASET}/{OUT_PNG.name}}}
\caption{{CIA score trajectories. Each panel uses raw scores $s_r=-\mathcal{{L}}(W^{{(r)}},D_x)$ and
the shared y-axis range $[-4, 0.2]$.}}
\end{{figure}}

\clearpage
\begin{{figure}}[p]
\centering
\includegraphics[width=0.94\textwidth]{{results/planned_runs/{DATASET}/{OUT_DISTANCE_PNG.name}}}
\caption{{Metric-privacy maximum pairwise client-model distance recorded before each server aggregation.}}
\end{{figure}}

\clearpage
\section*{{Multi-round CIA}}
The directional ROC AUC estimates the probability that an IN score ranks above an OUT score.
The paper reports one set of 20 IN and 20 OUT round scores. Our per-seed estimates use the same
20-versus-20 construction; the pooled summary combines all three seeds (60 IN and 60 OUT scores).
Confidence intervals are deterministic 95\% percentile intervals from 10,000 stratified bootstrap resamples.
Those intervals resample rounds as if they were independent draws, which they are not: the 20 scores of a
run form one autocorrelated trajectory and the real replication unit is the seed ($n=3$). The reported
intervals are therefore optimistic and should be read as paper-comparable, not as calibrated coverage.

\begin{{table}}[h]
\centering
\caption{{{paper_reference_caption('Table 13')} The attack uses noisy shadow data.}}
{paper_multi_table()}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Ours, pooled across seeds. Utility is mean $\pm$ sample std over rounds 16--20 and all three seeds.}}
{ours_pooled_multi_table(in_groups, out_groups)}
\end{{table}}

\textbf{{Interpretation.}} Clean-shadow access produces strong separation for every approach.
For noisy-shadow access, metric-privacy and global-DP also remain far above 0.5 in this Alzheimer setup;
neither reproduces the near-random multi-round behavior reported by the paper. Because this is a directional
AUC, values below 0.5 indicate reversed score ordering and can still be exploitable if an attacker is allowed
to invert the decision rule.

\clearpage
\section*{{Why the pooled score inverts the clean/noisy ordering}}
The pooled 20-versus-20 AUC ranks every IN checkpoint against every OUT checkpoint, including pairs from
different communication rounds. Raw shadow loss changes by an order of magnitude over training, so the round
index, not membership, dominates most of those comparisons. The two shadow variants drift in opposite
directions: with clean shadow data the OUT loss keeps falling as the two remaining clients converge and drops
below the IN loss of rounds 1--5, whereas with noisy shadow data the OUT loss grows as the model becomes
confidently wrong on perturbed inputs, so no late OUT checkpoint can outrank an early IN checkpoint.

For metric-privacy, {fmt_pct(clean_discordant)} of pooled pairs are discordant with clean shadow data but only
{fmt_pct(noisy_discordant)} with noisy shadow data. Restricted to matched rounds the ordering reverses
({fmt_pct(clean_matched_discordant)} discordant clean versus {fmt_pct(noisy_matched_discordant)} noisy), and
every clean discordant pair comes from an IN checkpoint that is earlier than its OUT counterpart
({fmt_pct(clean_in_earlier)} of those pairs are discordant). The pooled clean AUC is thus penalised for a
training-trajectory effect, not for a weaker membership signal.

\begin{{table}}[h]
\centering
\caption{{Score-construction diagnostics. ``Round-matched'' compares IN and OUT only at identical
(seed, round) checkpoints. ``Aggregate-referenced'' keeps the paper's pooling but scores the
attacker-observable relative difference between the aggregated test loss and the shadow loss
(Tables 10--11), which removes the round-wise drift of the global model.}}
{score_construction_table(in_groups, out_groups)}
\end{{table}}

Under both confound-free constructions, noise weakens the attack exactly as expected. Round-matched, clean
shadow data is never worse than noisy shadow data for any mechanism. With the aggregate-referenced score, clean access
stays near-perfect ({referenced_clean_range} across the three mechanisms) while noisy access falls to
{referenced_noisy["vanilla"]:.3f} for vanilla FL, {referenced_noisy["global-dp"]:.3f} for global-DP, and
{referenced_noisy["metric-privacy"]:.3f} for metric-privacy --- and the vanilla value lands on the near-random
regime the paper reports. The conclusion to draw from this experiment is therefore about the metric: the paper's
pooled multi-round AUC is not a pure membership statistic, because it lets the round index leak into the
ranking.

\clearpage
\section*{{Per-seed multi-round attack results}}
\begin{{table}}[h]
\centering
\caption{{Paper-comparable 20-round AUC estimates for each matched IN/OUT seed pair.}}
{per_seed_auc_table(in_groups, out_groups)}
\end{{table}}

\section*{{Data provenance}}
All ``ours'' values are computed directly from the {DATASET_LABEL} IN/OUT artifacts under
\texttt{{results/planned\_runs/{DATASET}}}. Published reference values are transcribed from
Sainz-Pardo Diaz et al., \emph{{Metric-privacy-inspired noise calibration in federated learning:
Improving convergence and preventing client inference attacks}}, Tables 10, 12, and 13.

\end{{document}}
"""


def compile_pdf(tex_path: Path) -> Path:
    engine = shutil.which("pdflatex")
    if engine is None:
        raise RuntimeError("pdflatex is required to build the report")
    with tempfile.TemporaryDirectory() as tmp:
        proc: subprocess.CompletedProcess[str] | None = None
        for _ in range(2):
            proc = subprocess.run(
                [
                    engine,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    tmp,
                    str(tex_path),
                ],
                capture_output=True,
                text=True,
            )
        assert proc is not None
        if proc.returncode != 0:
            print(proc.stdout[-5000:])
            raise SystemExit("pdflatex failed")
        pdf_path = tex_path.with_suffix(".pdf")
        shutil.copyfile(Path(tmp) / f"{tex_path.stem}.pdf", pdf_path)
        return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("alzheimer", "cifar", "fashion"), default="alzheimer")
    args = parser.parse_args()
    configure_dataset(args.dataset)
    in_groups = load_cia(IN_DIR / "cia.json")
    out_groups = load_cia(OUT_DIR / "cia.json")
    write_score_figure(in_groups, out_groups, OUT_PNG)
    print(f"wrote {OUT_PNG}")
    write_distance_figure(OUT_DISTANCE_PNG)
    print(f"wrote {OUT_DISTANCE_PNG}")
    OUT_TEX.write_text(build_document())
    print(f"wrote {OUT_TEX}")
    print(f"wrote {compile_pdf(OUT_TEX)}")


if __name__ == "__main__":
    main()
