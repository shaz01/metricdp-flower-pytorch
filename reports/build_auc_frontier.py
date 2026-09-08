"""Build the full 4-dataset attack-AUC-vs-accuracy frontier from the AUC-targeted noise sweep.

All 16 curves (4 datasets x 2 partitions x 2 privacy modes) are complete. Every number
here is read straight from the committed search_state.json/vanilla_reference.json files
under results/auc_target_sweep/ -- nothing is hand-transcribed. Re-run this script any
time those files change.

Handles all three non-"landed" outcomes a curve can end in:
  - landed: search_stages has the full trail, confirmation_stages has 2 more seeds
    at the landing ratio.
  - collapsed-before-target: search_stages has the full trail ending in the stage
    whose accuracy triggered the collapse guard; no landing point, no confirmation.
  - anchor-not-found: search_stages is empty (the step-up phase never started);
    anchor_stages holds the halving attempts instead -- used as the plotted trail.

Run from the repository root:
    uv run python reports/build_auc_frontier.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "results" / "auc_target_sweep"
OUTPUT = ROOT / "reports" / "auc_frontier.html"

DATASETS = ("eurosat", "alzheimer", "fashion-mnist", "cifar10")
DATASET_LABELS = {
    "eurosat": "EuroSAT",
    "alzheimer": "Alzheimer-MRI",
    "fashion-mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
}
PARTITIONS = ("homogeneous", "non-iid")
COLORS = {"vanilla": "#111", "global-dp": "#2166ac", "metric-privacy": "#b2182b"}
LABELS = {"vanilla": "Vanilla", "global-dp": "Global-DP", "metric-privacy": "Metric-privacy"}
RANDOM_BASELINE = {"eurosat": 0.10, "alzheimer": 0.25, "fashion-mnist": 0.25, "cifar10": 0.10}

STATUS_LABELS = {
    "landed": "landed",
    "collapsed-before-target": "collapsed before target",
    "anchor-not-found": "anchor not found",
}

# Chart geometry (shared by all panels).
VB_W, VB_H = 460, 360
MARGIN = {"left": 54, "right": 16, "top": 16, "bottom": 46}
PLOT_W = VB_W - MARGIN["left"] - MARGIN["right"]
PLOT_H = VB_H - MARGIN["top"] - MARGIN["bottom"]
X_DOMAIN = (0.42, 1.03)  # attack AUC


def load_curve(dataset: str, partition: str, privacy: str) -> dict:
    path = SWEEP_ROOT / dataset / partition / privacy / "search_state.json"
    return json.loads(path.read_text())


def load_vanilla(dataset: str, partition: str) -> dict:
    path = SWEEP_ROOT / dataset / partition / "vanilla_reference.json"
    return json.loads(path.read_text())


def curve_points(state: dict) -> list[dict]:
    """All distinct (ratio, seed) points for one curve, tagged with a plotting role.

    Falls back to anchor_stages when search_stages is empty (anchor-not-found curves
    never reached the step-up phase, so anchor_stages is the only trail there is).
    """
    trail = state["search_stages"] or state["anchor_stages"]
    points = list(trail)
    seen = {(p["noise_ratio"], p["seed"]) for p in points}
    for p in state["confirmation_stages"]:
        key = (p["noise_ratio"], p["seed"])
        if key not in seen:
            points.append(p)
            seen.add(key)
    landing_ratio = state.get("landing_ratio")
    for p in points:
        if landing_ratio is not None and p["noise_ratio"] == landing_ratio:
            p["role"] = "landed" if p["seed"] == 42 else "confirm"
        else:
            p["role"] = "search"
    return points


def y_domain_for(dataset: str) -> tuple[float, float]:
    """Accuracy axis range: wide enough to show a collapse down toward the random
    baseline, tight enough to still resolve the healthy region for datasets that
    never collapse."""
    baseline = RANDOM_BASELINE[dataset]
    return (max(0.0, baseline - 0.05), 1.0 if dataset in ("alzheimer", "fashion-mnist") else 0.95)


def sx(auc: float) -> float:
    return MARGIN["left"] + (auc - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0]) * PLOT_W


def sy(accuracy: float, y_domain: tuple[float, float]) -> float:
    return MARGIN["top"] + (1 - (accuracy - y_domain[0]) / (y_domain[1] - y_domain[0])) * PLOT_H


def render_panel(dataset: str, partition: str) -> str:
    vanilla = load_vanilla(dataset, partition)
    curves = {privacy: load_curve(dataset, partition, privacy) for privacy in ("global-dp", "metric-privacy")}
    y_domain = y_domain_for(dataset)
    baseline = RANDOM_BASELINE[dataset]

    parts: list[str] = []
    x0, y0 = MARGIN["left"], MARGIN["top"]
    x1, y1 = MARGIN["left"] + PLOT_W, MARGIN["top"] + PLOT_H

    # Target band, AUC 0.45-0.55.
    bx0, bx1 = sx(0.45), sx(0.55)
    parts.append(f'<rect x="{bx0:.1f}" y="{y0}" width="{bx1 - bx0:.1f}" height="{PLOT_H}" fill="#f2f2f2"/>')

    # Random-baseline accuracy line (dashed) -- context for any collapsed curve.
    by = sy(baseline, y_domain)
    if y0 <= by <= y1:
        parts.append(f'<line x1="{x0}" y1="{by:.1f}" x2="{x1}" y2="{by:.1f}" stroke="#ccc" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{x1 - 4}" y="{by - 4:.1f}" text-anchor="end" font-size="9.5" fill="#999">random baseline</text>')

    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        x = sx(v)
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="#ddd"/>')
        parts.append(f'<text x="{x:.1f}" y="{y1 + 16}" text-anchor="middle" font-size="11">{v:.1f}</text>')
    y_ticks = [round(y_domain[0] + i * (y_domain[1] - y_domain[0]) / 4, 2) for i in range(5)]
    for v in y_ticks:
        y = sy(v, y_domain)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 3:.1f}" text-anchor="end" font-size="11">{v * 100:.0f}%</text>')

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#555"/>')
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#555"/>')
    parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{VB_H - 6}" text-anchor="middle" font-size="12">Attack AUC</text>')
    parts.append(
        f'<text x="14" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {(y0 + y1) / 2:.1f})">Accuracy</text>'
    )

    vx, vy = sx(vanilla["auc"]), sy(vanilla["accuracy"], y_domain)
    parts.append(
        f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="6" fill="none" stroke="{COLORS["vanilla"]}" stroke-width="2"/>'
    )

    status_notes = []
    for privacy, state in curves.items():
        color = COLORS[privacy]
        for p in curve_points(state):
            cx, cy = sx(p["auc"]), sy(p["accuracy"], y_domain)
            r = 5 if p["role"] != "search" else 3.5
            opacity = "1" if p["role"] != "search" else "0.55"
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" fill-opacity="{opacity}"/>')
        if state["status"] != "landed":
            status_notes.append(f'{LABELS[privacy]}: {STATUS_LABELS[state["status"]]}')

    body = "".join(parts)
    caption = "Homogeneous partition" if partition == "homogeneous" else "Non-IID partition"
    sub = f"vanilla &middot; accuracy {vanilla['accuracy'] * 100:.1f}% &middot; auc {vanilla['auc'] * 100:.1f}%"
    if status_notes:
        sub += f' &mdash; <span style="color:#a33">{"; ".join(status_notes)}</span>'
    return (
        f'<div class="panel"><h3>{caption}</h3><p class="note">{sub}</p>'
        f'<svg viewBox="0 0 {VB_W} {VB_H}" role="img" aria-label="{caption}: attack AUC vs accuracy">{body}</svg></div>'
    )


def render_dataset_section(dataset: str) -> str:
    panels = "".join(render_panel(dataset, partition) for partition in PARTITIONS)
    return (
        f'<section id="{dataset}"><h2>{DATASET_LABELS[dataset]}</h2>'
        f'<div class="grid">{panels}</div>'
        f'<div class="legend">'
        f'<span class="key"><span class="swatch ring" style="border-color:{COLORS["vanilla"]}"></span>Vanilla (no defense)</span>'
        f'<span class="key"><span class="swatch" style="background:{COLORS["global-dp"]}"></span>Global-DP</span>'
        f'<span class="key"><span class="swatch" style="background:{COLORS["metric-privacy"]}"></span>Metric-privacy</span>'
        f"</div></section>"
    )


def render_table_rows() -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        for partition in PARTITIONS:
            vanilla = load_vanilla(dataset, partition)
            rows.append(
                f"<tr><td>{DATASET_LABELS[dataset]}</td><td>{partition}</td><td>Vanilla</td>"
                f"<td>&mdash;</td><td>&mdash;</td><td>reference</td>"
                f"<td>{vanilla['auc'] * 100:.1f}</td><td>{vanilla['accuracy'] * 100:.1f}</td></tr>"
            )
            for privacy in ("global-dp", "metric-privacy"):
                state = load_curve(dataset, partition, privacy)
                for p in curve_points(state):
                    rows.append(
                        f"<tr><td>{DATASET_LABELS[dataset]}</td><td>{partition}</td><td>{LABELS[privacy]}</td>"
                        f"<td>{p['noise_ratio']:.3e}</td><td>{p['seed']}</td><td>{p['role']}</td>"
                        f"<td>{p['auc'] * 100:.1f}</td><td>{p['accuracy'] * 100:.1f}</td></tr>"
                    )
    return "".join(rows)


def render_status_summary() -> str:
    counts: dict[str, int] = {}
    cells = []
    for dataset in DATASETS:
        for partition in PARTITIONS:
            for privacy in ("global-dp", "metric-privacy"):
                state = load_curve(dataset, partition, privacy)
                status = state["status"]
                counts[status] = counts.get(status, 0) + 1
                cells.append(
                    f"<tr><td>{DATASET_LABELS[dataset]}</td><td>{partition}</td><td>{LABELS[privacy]}</td>"
                    f"<td>{STATUS_LABELS[status]}</td></tr>"
                )
    badges = " ".join(
        f'<span class="badge {"done" if k == "landed" else "warn"}">{STATUS_LABELS[k]} &middot; {v}</span>'
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    return badges, "".join(cells)


def build() -> None:
    sections = "".join(render_dataset_section(dataset) for dataset in DATASETS)
    table_rows = render_table_rows()
    badges, status_rows = render_status_summary()

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AUC-Targeted Noise Sweep: Full Frontier</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, Helvetica, sans-serif; }}
    body {{ max-width: 1180px; margin: 0 auto; padding: 28px; color: #111; background: #fff; }}
    header {{ border-bottom: 1px solid #bbb; margin-bottom: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 36px; padding-bottom: 6px; border-bottom: 1px solid #ccc; font-size: 21px; }}
    h3 {{ font-size: 17px; margin: 0 0 4px; }}
    p, li {{ line-height: 1.45; }}
    .meta, .note {{ color: #444; font-size: 14px; }}
    .status {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 0; }}
    .badge {{ font-size: 12px; padding: 3px 10px; border-radius: 100px; border: 1px solid #bbb; color: #444; }}
    .badge.done {{ border-color: #1a7a3c; color: #1a7a3c; }}
    .badge.warn {{ border-color: #a33; color: #a33; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; }}
    .panel {{ border: 1px solid #ccc; padding: 12px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3),
    th:nth-child(6), td:nth-child(6) {{ text-align: left; }}
    thead th {{ border-bottom: 2px solid #888; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px; font-size: 12px; }}
    .key {{ display: inline-flex; align-items: center; gap: 5px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    .swatch.ring {{ background: none; border: 2px solid; }}
    footer {{ margin-top: 36px; padding-top: 10px; border-top: 1px solid #bbb; font-size: 12px; color: #555; }}
    nav.toc {{ font-size: 13px; margin-top: 14px; }}
    nav.toc a {{ margin-right: 14px; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    @media print {{ body {{ max-width: none; padding: 12mm; }} section {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
<header>
  <h1>AUC-Targeted Noise Sweep: Full Frontier</h1>
  <p class="meta">All 16 curves (4 datasets x 2 partitions x 2 privacy modes). X-axis: round-matched
  clean-shadow attack AUC (direction-reversal allowed). Y-axis: model accuracy. Shaded band: AUC
  0.45&ndash;0.55, the target. Faint dots: single-seed search steps. Solid dots: landing point (seed 42)
  and its 2-seed confirmation. Dashed line: this dataset's random-guessing accuracy floor, shown where
  a curve collapsed toward it.</p>
  <div class="status">{badges}</div>
  <nav class="toc">
    <a href="#eurosat">EuroSAT</a><a href="#alzheimer">Alzheimer-MRI</a>
    <a href="#fashion-mnist">Fashion-MNIST</a><a href="#cifar10">CIFAR-10</a>
    <a href="#outcomes">Outcomes</a><a href="#data">All data</a>
  </nav>
</header>

{sections}

<section id="outcomes">
  <h2>Every curve's outcome</h2>
  <div class="panel"><table><thead><tr><th>Dataset</th><th>Partition</th><th>Mechanism</th><th>Status</th></tr></thead><tbody>{status_rows}</tbody></table></div>
</section>

<section id="data">
  <h2>Every point</h2>
  <div class="panel"><table><thead><tr><th>Dataset</th><th>Partition</th><th>Mechanism</th><th>Noise ratio</th><th>Seed</th><th>Role</th><th>Attack AUC (%)</th><th>Accuracy (%)</th></tr></thead><tbody>{table_rows}</tbody></table></div>
</section>

<footer>Source: committed JSON artifacts under <code>results/auc_target_sweep/&lt;dataset&gt;/</code>
(<code>search_state.json</code> per partition/privacy, <code>vanilla_reference.json</code> per partition).
Generated by <code>reports/build_auc_frontier.py</code>.</footer>
</body>
</html>
"""
    OUTPUT.write_text(html)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
