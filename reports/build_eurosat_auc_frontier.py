"""Build the EuroSAT attack-AUC-vs-accuracy frontier from the AUC-targeted noise sweep.

Interim view: EuroSAT is the only dataset with all 4 curves (homogeneous/non-iid x
global-dp/metric-privacy) landed at the time this was generated -- Alzheimer,
Fashion-MNIST, and CIFAR-10 are still running. Rebuild once they land by re-running
this script; it always reads straight from the committed search_state.json/
vanilla_reference.json files, nothing here is hand-transcribed.

Run from the repository root:
    uv run python reports/build_eurosat_auc_frontier.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "results" / "auc_target_sweep" / "eurosat"
OUTPUT = ROOT / "reports" / "eurosat_auc_frontier.html"

PARTITIONS = ("homogeneous", "non-iid")
COLORS = {"vanilla": "#111", "global-dp": "#2166ac", "metric-privacy": "#b2182b"}
LABELS = {"vanilla": "Vanilla", "global-dp": "Global-DP", "metric-privacy": "Metric-privacy"}

# Chart geometry (shared by both panels).
VB_W, VB_H = 460, 360
MARGIN = {"left": 54, "right": 16, "top": 16, "bottom": 46}
PLOT_W = VB_W - MARGIN["left"] - MARGIN["right"]
PLOT_H = VB_H - MARGIN["top"] - MARGIN["bottom"]
X_DOMAIN = (0.42, 1.03)   # attack AUC
Y_DOMAIN = (0.78, 0.925)  # accuracy


def load_curve(partition: str, privacy: str) -> dict:
    path = SWEEP_ROOT / partition / privacy / "search_state.json"
    return json.loads(path.read_text())


def load_vanilla(partition: str) -> dict:
    path = SWEEP_ROOT / partition / "vanilla_reference.json"
    return json.loads(path.read_text())


def curve_points(state: dict) -> list[dict]:
    """All distinct (ratio, seed) points for one curve: the search trail plus confirmation seeds."""
    points = list(state["search_stages"])
    seen = {(p["noise_ratio"], p["seed"]) for p in points}
    for p in state["confirmation_stages"]:
        key = (p["noise_ratio"], p["seed"])
        if key not in seen:
            points.append(p)
            seen.add(key)
    confirmed_ratio = state.get("landing_ratio")
    for p in points:
        p["role"] = "confirm" if p["noise_ratio"] == confirmed_ratio and p["seed"] != 42 else (
            "landed" if p["noise_ratio"] == confirmed_ratio else "search"
        )
    return points


def sx(auc: float) -> float:
    return MARGIN["left"] + (auc - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0]) * PLOT_W


def sy(accuracy: float) -> float:
    return MARGIN["top"] + (1 - (accuracy - Y_DOMAIN[0]) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * PLOT_H


def render_panel(partition: str) -> str:
    vanilla = load_vanilla(partition)
    curves = {privacy: load_curve(partition, privacy) for privacy in ("global-dp", "metric-privacy")}

    parts: list[str] = []
    x0, y0 = MARGIN["left"], MARGIN["top"]
    x1, y1 = MARGIN["left"] + PLOT_W, MARGIN["top"] + PLOT_H

    # Target band, AUC 0.45-0.55.
    bx0, bx1 = sx(0.45), sx(0.55)
    parts.append(f'<rect x="{bx0:.1f}" y="{y0}" width="{bx1 - bx0:.1f}" height="{PLOT_H}" fill="#f2f2f2"/>')

    # Gridlines + x ticks (AUC).
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        x = sx(v)
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="#ddd"/>')
        parts.append(f'<text x="{x:.1f}" y="{y1 + 16}" text-anchor="middle" font-size="11">{v:.1f}</text>')
    # y ticks (accuracy).
    for v in (0.80, 0.84, 0.88, 0.92):
        y = sy(v)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 3:.1f}" text-anchor="end" font-size="11">{v * 100:.0f}%</text>')

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#555"/>')
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#555"/>')
    parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{VB_H - 6}" text-anchor="middle" font-size="12">Attack AUC</text>')
    parts.append(
        f'<text x="14" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {(y0 + y1) / 2:.1f})">Accuracy</text>'
    )

    # Vanilla reference: open circle.
    vx, vy = sx(vanilla["auc"]), sy(vanilla["accuracy"])
    parts.append(
        f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="6" fill="none" stroke="{COLORS["vanilla"]}" stroke-width="2"/>'
    )

    for privacy, state in curves.items():
        color = COLORS[privacy]
        for p in curve_points(state):
            cx, cy = sx(p["auc"]), sy(p["accuracy"])
            r = 5 if p["role"] != "search" else 3.5
            opacity = "1" if p["role"] != "search" else "0.55"
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" fill-opacity="{opacity}"/>')

    body = "".join(parts)
    caption = f"Homogeneous partition" if partition == "homogeneous" else "Non-IID partition"
    sub = f"vanilla reference &middot; accuracy {vanilla['accuracy'] * 100:.1f}% &middot; auc {vanilla['auc'] * 100:.1f}%"
    return (
        f'<div class="panel"><h3>{caption}</h3><p class="note">{sub}</p>'
        f'<svg viewBox="0 0 {VB_W} {VB_H}" role="img" aria-label="{caption}: attack AUC vs accuracy">{body}</svg></div>'
    )


def render_table_rows() -> str:
    rows: list[str] = []
    for partition in PARTITIONS:
        vanilla = load_vanilla(partition)
        rows.append(
            f"<tr><td>{partition}</td><td>Vanilla</td><td>&mdash;</td><td>&mdash;</td>"
            f"<td>reference</td><td>{vanilla['auc'] * 100:.1f}</td><td>{vanilla['accuracy'] * 100:.1f}</td></tr>"
        )
        for privacy in ("global-dp", "metric-privacy"):
            state = load_curve(partition, privacy)
            for p in curve_points(state):
                role = "confirm" if p["role"] in ("confirm", "landed") and p["seed"] != 42 else p["role"]
                rows.append(
                    f"<tr><td>{partition}</td><td>{LABELS[privacy]}</td>"
                    f"<td>{p['noise_ratio']:.3e}</td><td>{p['seed']}</td><td>{role}</td>"
                    f"<td>{p['auc'] * 100:.1f}</td><td>{p['accuracy'] * 100:.1f}</td></tr>"
                )
    return "".join(rows)


def build() -> None:
    panels = "".join(render_panel(partition) for partition in PARTITIONS)
    table_rows = render_table_rows()

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EuroSAT Attack-AUC-vs-Accuracy Frontier</title>
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
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; }}
    .panel {{ border: 1px solid #ccc; padding: 12px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(5), td:nth-child(5) {{ text-align: left; }}
    thead th {{ border-bottom: 2px solid #888; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px; font-size: 12px; }}
    .key {{ display: inline-flex; align-items: center; gap: 5px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    .swatch.ring {{ background: none; border: 2px solid; }}
    footer {{ margin-top: 36px; padding-top: 10px; border-top: 1px solid #bbb; font-size: 12px; color: #555; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    @media print {{ body {{ max-width: none; padding: 12mm; }} section {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
<header>
  <h1>EuroSAT: Attack AUC vs. Accuracy Frontier</h1>
  <p class="meta">AUC-targeted noise sweep. X-axis: round-matched clean-shadow attack AUC (direction-reversal
  allowed). Y-axis: model accuracy. Shaded band: AUC 0.45&ndash;0.55, the target &mdash; attack performance
  indistinguishable from a coin flip. Faint dots: single-seed search steps. Solid dots: 3-seed confirmation
  runs at the noise level each curve landed on.</p>
  <div class="status">
    <span class="badge done">EuroSAT &middot; 4/4 curves landed</span>
    <span class="badge">Alzheimer &middot; in progress</span>
    <span class="badge">Fashion-MNIST &middot; queued</span>
    <span class="badge">CIFAR-10 &middot; queued</span>
  </div>
</header>

<section id="frontier">
  <h2>Frontier by partition</h2>
  <div class="grid">{panels}</div>
  <div class="legend">
    <span class="key"><span class="swatch ring" style="border-color:{COLORS['vanilla']}"></span>Vanilla (no defense)</span>
    <span class="key"><span class="swatch" style="background:{COLORS['global-dp']}"></span>Global-DP</span>
    <span class="key"><span class="swatch" style="background:{COLORS['metric-privacy']}"></span>Metric-privacy</span>
  </div>
</section>

<section id="data">
  <h2>Every point</h2>
  <div class="panel"><table><thead><tr><th>Partition</th><th>Mechanism</th><th>Noise ratio</th><th>Seed</th><th>Role</th><th>Attack AUC (%)</th><th>Accuracy (%)</th></tr></thead><tbody>{table_rows}</tbody></table></div>
</section>

<footer>Source: committed JSON artifacts under <code>results/auc_target_sweep/eurosat/</code>
(<code>search_state.json</code> per partition/privacy, <code>vanilla_reference.json</code> per partition).
Generated by <code>reports/build_eurosat_auc_frontier.py</code> &mdash; interim, EuroSAT only; the sweep
also covers Alzheimer, Fashion-MNIST, and CIFAR-10, still running at generation time.</footer>
</body>
</html>
"""
    OUTPUT.write_text(html)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
