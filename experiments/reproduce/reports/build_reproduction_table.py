"""Build a LaTeX/PDF comparison of the paper's Table 6 against our reproduction runs.

Reads the detailed evaluation artifacts in
``results/planned_runs/reproduction/original_reproduction/`` and emits
``results/planned_runs/reproduction/reproduction_table.{tex,pdf}``.

Every number in the "ours" tables is pulled from the committed JSON artifacts;
nothing is hard-coded except the paper's own published Table 6.

Usage:  uv run python experiments/reproduce/reports/build_reproduction_table.py
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "planned_runs" / "reproduction"
RUNS_DIR = RESULTS_DIR / "original_reproduction"
PREV_DIR = REPO_ROOT / "results" / "reproduce_paper" / "evaluations"
OUT_TEX = RESULTS_DIR / "reproduction_table.tex"

MECHANISMS = ["vanilla", "global-dp", "metric-privacy"]
MECH_LABEL = {
    "vanilla": "Vanilla FL",
    "global-dp": "Global-DP",
    "metric-privacy": "Metric-privacy",
}
STRATEGY_LABEL = {
    "fedavg": "FedAvg",
    "fedavgm": "FedAvgM",
    "fedmedian": "FedMedian",
    "fedprox": "FedProx",
    "fedopt": "FedOpt",
    "fedyogi": "FedYogi",
}
STRATEGY_ORDER = ["fedavg", "fedavgm", "fedmedian", "fedprox", "fedopt", "fedyogi"]

# Paper (Alvarez et al.), Table 6: accuracy / F1 / precision, homogeneous clients.
PAPER_TABLE6 = {
    "fedavg": {
        "vanilla": (0.909, 0.903, 0.950),
        "global-dp": (0.884, 0.862, 0.918),
        "metric-privacy": (0.894, 0.843, 0.943),
    },
    "fedavgm": {
        "vanilla": (0.884, 0.768, 0.932),
        "global-dp": (0.762, 0.554, 0.607),
        "metric-privacy": (0.823, 0.608, 0.631),
    },
    "fedmedian": {
        "vanilla": (0.932, 0.899, 0.946),
        "global-dp": (0.875, 0.827, 0.913),
        "metric-privacy": (0.895, 0.843, 0.943),
    },
    "fedprox": {
        "vanilla": (0.909, 0.848, 0.873),
        "global-dp": (0.877, 0.853, 0.914),
        "metric-privacy": (0.903, 0.877, 0.944),
    },
    "fedopt": {
        "vanilla": (0.950, 0.936, 0.970),
        "global-dp": (0.918, 0.877, 0.911),
        "metric-privacy": (0.930, 0.917, 0.950),
    },
    "fedyogi": {
        "vanilla": (0.933, 0.914, 0.957),
        "global-dp": (0.908, 0.841, 0.884),
        "metric-privacy": (0.917, 0.888, 0.941),
    },
}


def load_runs() -> tuple[list[dict], dict]:
    """Return per-run records and the shared configuration metadata."""
    records: list[dict] = []
    config: dict = {}
    for eval_path in sorted(RUNS_DIR.glob("*.evaluation.json")):
        evaluation = json.loads(eval_path.read_text())
        run_path = eval_path.with_name(eval_path.name.replace(".evaluation.json", ".json"))
        meta = json.loads(run_path.read_text())["metadata"]
        test = evaluation["server_final_test"]
        records.append(
            {
                "run_name": evaluation["run_name"],
                "mechanism": meta["privacy"],
                "strategy": meta["aggregation"],
                "seed": meta["seed"],
                "partition": meta["partition_mode"],
                "accuracy": test["accuracy"],
                "log_loss": test["log_loss"],
                "w_f1": test["averages"]["weighted"]["f1"],
                "w_precision": test["averages"]["weighted"]["precision"],
                "w_recall": test["averages"]["weighted"]["recall"],
                "m_f1": test["averages"]["macro"]["f1"],
                "m_precision": test["averages"]["macro"]["precision"],
                "m_recall": test["averages"]["macro"]["recall"],
                "auc": test["roc_ovr"]["weighted_auc"],
                "num_examples": test["num_examples"],
            }
        )
        if not config:
            config = {
                key: meta[key]
                for key in (
                    "num_clients",
                    "rounds",
                    "local_epochs",
                    "batch_size",
                    "noise_multiplier",
                    "clipping_norm",
                    "partition_mode",
                    "data_module",
                    "model_module",
                    "device_name",
                )
            }
            config["library_versions"] = meta["library_versions"]
    return records, config


def load_previous_runs() -> list[dict]:
    """Load the earlier `results/reproduce_paper/` homogeneous main-grid runs."""
    records: list[dict] = []
    for eval_path in sorted(PREV_DIR.glob("main__homogeneous__*noise-0.01__*.evaluation.json")):
        evaluation = json.loads(eval_path.read_text())
        run_path = (
            PREV_DIR.parent / "runs" / eval_path.name.replace(".evaluation.json", ".json")
        )
        meta = json.loads(run_path.read_text())["metadata"]
        test = evaluation["server_final_test"]
        records.append(
            {
                "run_name": evaluation["run_name"],
                "mechanism": meta["privacy"],
                "strategy": meta["aggregation"],
                "seed": meta["seed"],
                "accuracy": test["accuracy"],
                "w_f1": test["averages"]["weighted"]["f1"],
                "w_precision": test["averages"]["weighted"]["precision"],
                "m_f1": test["averages"]["macro"]["f1"],
                "m_precision": test["averages"]["macro"]["precision"],
            }
        )
    return records


def group(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["strategy"], rec["mechanism"])].append(rec)
    for runs in grouped.values():
        runs.sort(key=lambda r: r["seed"])
    return grouped


def mean_std(values: list[float]) -> tuple[float, float | None]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def fmt_ms(values: list[float]) -> str:
    mean, std = mean_std(values)
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} {{\\scriptsize$\\pm$\\,{std:.3f}}}"


def fmt_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else "$-$"
    return f"{sign}{abs(delta):.3f}"


def paper_table() -> str:
    lines = [
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\emph{Vanilla FL}} & \multicolumn{3}{c}{\emph{Global-DP}}"
        r" & \multicolumn{3}{c}{\emph{Metric-privacy}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"\emph{Strategy} & \emph{Acc.} & \emph{F1} & \emph{Prec.}"
        r" & \emph{Acc.} & \emph{F1} & \emph{Prec.}"
        r" & \emph{Acc.} & \emph{F1} & \emph{Prec.} \\",
        r"\midrule",
    ]
    for strategy in STRATEGY_ORDER:
        cells = []
        for mech in MECHANISMS:
            cells.extend(f"{v:.3f}" for v in PAPER_TABLE6[strategy][mech])
        lines.append(rf"\emph{{{STRATEGY_LABEL[strategy]}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ours_table(grouped, keys: tuple[str, str, str], strategies: list[str]) -> str:
    acc_key, f1_key, prec_key = keys
    lines = [
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\emph{Vanilla FL}} & \multicolumn{3}{c}{\emph{Global-DP}}"
        r" & \multicolumn{3}{c}{\emph{Metric-privacy}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"\emph{Strategy} & \emph{Acc.} & \emph{F1} & \emph{Prec.}"
        r" & \emph{Acc.} & \emph{F1} & \emph{Prec.}"
        r" & \emph{Acc.} & \emph{F1} & \emph{Prec.} \\",
        r"\midrule",
    ]
    for strategy in strategies:
        cells = []
        for mech in MECHANISMS:
            runs = grouped.get((strategy, mech), [])
            if not runs:
                cells.extend(["--", "--", "--"])
                continue
            cells.append(fmt_ms([r[acc_key] for r in runs]))
            cells.append(fmt_ms([r[f1_key] for r in runs]))
            cells.append(fmt_ms([r[prec_key] for r in runs]))
        lines.append(rf"\emph{{{STRATEGY_LABEL[strategy]}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def delta_table(grouped, strategies: list[str]) -> str:
    lines = [
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\emph{Vanilla FL}} & \multicolumn{3}{c}{\emph{Global-DP}}"
        r" & \multicolumn{3}{c}{\emph{Metric-privacy}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"\emph{Strategy} & $\Delta$Acc. & $\Delta$F1 & $\Delta$Prec."
        r" & $\Delta$Acc. & $\Delta$F1 & $\Delta$Prec."
        r" & $\Delta$Acc. & $\Delta$F1 & $\Delta$Prec. \\",
        r"\midrule",
    ]
    for strategy in strategies:
        cells = []
        for mech in MECHANISMS:
            runs = grouped.get((strategy, mech), [])
            if not runs:
                cells.extend(["--", "--", "--"])
                continue
            paper = PAPER_TABLE6[strategy][mech]
            ours = (
                statistics.fmean([r["accuracy"] for r in runs]),
                statistics.fmean([r["w_f1"] for r in runs]),
                statistics.fmean([r["w_precision"] for r in runs]),
            )
            cells.extend(fmt_delta(o - p) for o, p in zip(ours, paper))
        lines.append(rf"\emph{{{STRATEGY_LABEL[strategy]}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_seed_table(records: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{l l c ccc ccc c}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{Weighted avg.} & \multicolumn{3}{c}{Macro avg.} & \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Mechanism & Strategy & Seed & Acc. & F1 & Prec. & F1 & Prec. & Rec. & Log-loss \\",
        r"\midrule",
    ]
    ordered = sorted(
        records,
        key=lambda r: (MECHANISMS.index(r["mechanism"]), STRATEGY_ORDER.index(r["strategy"]), r["seed"]),
    )
    previous = None
    for rec in ordered:
        if previous is not None and rec["mechanism"] != previous:
            lines.append(r"\midrule")
        previous = rec["mechanism"]
        lines.append(
            " & ".join(
                [
                    MECH_LABEL[rec["mechanism"]],
                    STRATEGY_LABEL[rec["strategy"]],
                    str(rec["seed"]),
                    f"{rec['accuracy']:.4f}",
                    f"{rec['w_f1']:.4f}",
                    f"{rec['w_precision']:.4f}",
                    f"{rec['m_f1']:.4f}",
                    f"{rec['m_precision']:.4f}",
                    f"{rec['m_recall']:.4f}",
                    f"{rec['log_loss']:.4f}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_document(records: list[dict], config: dict) -> str:
    grouped = group(records)
    covered = [s for s in STRATEGY_ORDER if any((s, m) in grouped for m in MECHANISMS)]
    seeds = sorted({r["seed"] for r in records})

    previous = load_previous_runs()
    prev_grouped = group(previous)
    prev_covered = [s for s in STRATEGY_ORDER if any((s, m) in prev_grouped for m in MECHANISMS)]
    prev_seeds = sorted({r["seed"] for r in previous})

    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1.9cm,landscape]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{caption}}
\captionsetup{{font=small,labelfont=bf}}
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.15}}

\title{{Reproduction of the Paper's Table~6}}
\date{{}}

\begin{{document}}
\maketitle
\thispagestyle{{empty}}

\begin{{table}}[h]
\centering
\caption{{Paper, Table 6.}}
{paper_table()}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Ours (mean $\pm$ std over {len(seeds)} seeds).}}
{ours_table(grouped, ('accuracy', 'w_f1', 'w_precision'), covered)}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Difference (ours $-$ paper).}}
{delta_table(grouped, covered)}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Ours, macro-averaged F1/precision.}}
{ours_table(grouped, ('accuracy', 'm_f1', 'm_precision'), covered)}
\end{{table}}

\clearpage
\begin{{table}}[h]
\centering
\caption{{Previous runs, \texttt{{results/reproduce\_paper/}} (homogeneous, noise 0.01,
mean $\pm$ std over {len(prev_seeds)} seeds).}}
{ours_table(prev_grouped, ('accuracy', 'w_f1', 'w_precision'), prev_covered)}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Previous runs, macro-averaged F1/precision.}}
{ours_table(prev_grouped, ('accuracy', 'm_f1', 'm_precision'), prev_covered)}
\end{{table}}

\begin{{table}}[h]
\centering
\caption{{Per-seed final-model test metrics.}}
{per_seed_table(records)}
\end{{table}}

\end{{document}}
"""


def compile_pdf(tex_path: Path) -> Path | None:
    engine = shutil.which("pdflatex")
    if engine is None:
        print("pdflatex not found; wrote .tex only")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(2):
            proc = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory", tmp, str(tex_path)],
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            raise SystemExit("pdflatex failed")
        pdf_src = Path(tmp) / (tex_path.stem + ".pdf")
        pdf_dst = tex_path.with_suffix(".pdf")
        shutil.copyfile(pdf_src, pdf_dst)
        return pdf_dst


def main() -> None:
    records, config = load_runs()
    if not records:
        raise SystemExit(f"no evaluation artifacts found in {RUNS_DIR}")
    OUT_TEX.write_text(build_document(records, config))
    print(f"wrote {OUT_TEX}")
    pdf = compile_pdf(OUT_TEX)
    if pdf:
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
