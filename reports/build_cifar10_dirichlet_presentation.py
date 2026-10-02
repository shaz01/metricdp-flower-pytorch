"""Build the standalone CIFAR-10 Dirichlet removal-CIA results presentation."""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dirichlet" / "cifar10" / "8_clients"
OUTPUT = ROOT / "reports" / "cifar10_dirichlet_presentation.html"
ROUNDS = range(1, 21)
METHODS = ("vanilla", "global-dp", "metric-privacy")
METHOD_LABEL = {
    "vanilla": "Vanilla",
    "global-dp": "Global-DP",
    "metric-privacy": "Metric privacy",
}
COLORS = {None: "#222222", 0.0025: "#0072B2", 0.004: "#E69F00", 0.00625: "#CC79A7"}
DASHES = {"vanilla": "3 5", "global-dp": None, "metric-privacy": "10 5"}


def parse_folder(path: Path) -> tuple[float, str, float]:
    match = re.fullmatch(
        r"alpha-(?P<alpha>[^_]+)__(?P<privacy>[^_]+)__in-remove-out-remove__seeds-42__noise-(?P<ratio>.+)",
        path.name,
    )
    if not match:
        raise ValueError(f"Unexpected result folder: {path}")
    number = lambda value: float(value.replace("p", "."))
    return number(match["alpha"]), match["privacy"], number(match["ratio"])


def load() -> dict[tuple[float, str, float], dict]:
    data: dict[tuple[float, str, float], dict] = {}
    for cia_path in sorted(RESULTS.glob("*/cia.json")):
        alpha, privacy, ratio = parse_folder(cia_path.parent)
        rows = json.loads(cia_path.read_text())
        by_adjacency: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            adjacency = "in" if row["run_name"].startswith("cifar10-in-remove__") else "out"
            by_adjacency[adjacency].append(row)
        if set(by_adjacency) != {"in", "out"} or any(len(values) != 20 for values in by_adjacency.values()):
            raise RuntimeError(f"Expected 20 IN and 20 OUT rows in {cia_path}")
        for values in by_adjacency.values():
            values.sort(key=lambda row: int(row["server_round"]))
        runs = {}
        for adjacency in ("in", "out"):
            run_name = by_adjacency[adjacency][0]["run_name"]
            run_path = cia_path.parent / f"{run_name}.json"
            runs[adjacency] = json.loads(run_path.read_text())
        data[(alpha, privacy, ratio)] = {"rows": by_adjacency, "runs": runs}
    if not data:
        raise RuntimeError(f"No completed results found under {RESULTS}")
    return data


def variants(data: dict, alpha: float) -> list[tuple[str, float | None, dict]]:
    """Return one baseline plus each available private run for an alpha."""
    available = {(privacy, ratio): value for (a, privacy, ratio), value in data.items() if a == alpha}
    rows: list[tuple[str, float | None, dict]] = []
    vanilla_ratios = sorted(ratio for privacy, ratio in available if privacy == "vanilla")
    if vanilla_ratios:
        # Vanilla ignores the supplied ratio. Repeated runs are deterministic duplicates.
        rows.append(("vanilla", None, available[("vanilla", vanilla_ratios[0])]))
    for ratio in sorted({ratio for privacy, ratio in available if privacy != "vanilla"}):
        for privacy in ("global-dp", "metric-privacy"):
            if (privacy, ratio) in available:
                rows.append((privacy, ratio, available[(privacy, ratio)]))
    return rows


def fmt_alpha(value: float) -> str:
    return f"{value:g}"


def fmt_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.5f}"


def td(value: str | float, digits: int = 3) -> str:
    rendered = f"{value:.{digits}f}" if isinstance(value, float) else html.escape(str(value))
    return f"<td>{rendered}</td>"


def auc(in_values: list[float], out_values: list[float]) -> float:
    wins = sum(left > right for left in in_values for right in out_values)
    ties = sum(left == right for left in in_values for right in out_values)
    return (wins + 0.5 * ties) / (len(in_values) * len(out_values))


def loss_auc(value: dict, key: str) -> float:
    # Lower loss is the membership score used by the removal CIA.
    in_scores = [-float(row[key]) for row in value["rows"]["in"]]
    out_scores = [-float(row[key]) for row in value["rows"]["out"]]
    return auc(in_scores, out_scores)


def table(headers: str, body: list[str]) -> str:
    return f"<table><thead>{headers}</thead><tbody>{''.join(body)}</tbody></table>"


def utility_section(data: dict, alphas: list[float]) -> str:
    panels = []
    for alpha in alphas:
        body = []
        for privacy, ratio, value in variants(data, alpha):
            metrics = value["runs"]["in"]["server_evaluate_metrics"]["20"]
            cls = ' class="pair-start"' if privacy == "global-dp" else ""
            nm = "—" if ratio is None else f"{8 * ratio:.3f}"
            body.append(
                f"<tr{cls}>{td(METHOD_LABEL[privacy])}{td(fmt_ratio(ratio))}{td(nm)}"
                f"{td(float(metrics['accuracy']))}{td(float(metrics['f1']))}{td(float(metrics['loss']))}</tr>"
            )
        headers = "<tr><th>Method</th><th>Ratio</th><th>IN noise multiplier</th><th>Round-20 accuracy</th><th>Round-20 F1</th><th>Round-20 loss</th></tr>"
        panels.append(f'<div class="panel"><h3>Dirichlet α = {fmt_alpha(alpha)}</h3>{table(headers, body)}</div>')
    return f'<section id="utility"><h2>Server utility</h2><p class="note">Utility values are from the 8-client IN-remove runs at round 20. Vanilla appears once per α because its repeated ratio-labelled runs are identical and use no noise.</p><div class="grid">{"".join(panels)}</div></section>'


def single_round_section(data: dict, alphas: list[float]) -> str:
    panels = []
    for alpha in alphas:
        body = []
        for privacy, ratio, value in variants(data, alpha):
            values = []
            for index in (0, 19):
                row = value["rows"]["in"][index]
                values.extend(
                    [row["aggregated_test_loss"], row["target_clean_shadow_loss"], row["clean_difference_pct"], row["target_noisy_shadow_loss"], row["noisy_difference_pct"]]
                )
            cls = ' class="pair-start"' if privacy == "global-dp" else ""
            body.append(f"<tr{cls}>{td(METHOD_LABEL[privacy])}{td(fmt_ratio(ratio))}{''.join(td(float(item), 2 if i in {2,4,7,9} else 3) for i,item in enumerate(values))}</tr>")
        headers = "<tr><th rowspan=\"2\">Method</th><th rowspan=\"2\">Ratio</th><th colspan=\"5\">Round 1</th><th colspan=\"5\">Round 20</th></tr><tr>" + "".join(f"<th>{label}</th>" for label in ("Agg. loss", "Clean loss", "Clean Δ%", "Noisy loss", "Noisy Δ%") * 2) + "</tr>"
        panels.append(f'<div class="panel"><h3>Dirichlet α = {fmt_alpha(alpha)}</h3>{table(headers, body)}</div>')
    note = "Relative difference is (target shadow loss − aggregated test loss) / target shadow loss × 100. Values closer to 0 indicate a weaker loss-based signal."
    return f'<section id="single-round"><h2>Single-round CIA: rounds 1 and 20</h2><p class="note">{note}</p><div class="grid">{"".join(panels)}</div></section>'


def line_chart(series: list[tuple[str, list[float], str, str | None]], title: str, ylabel: str) -> str:
    width, height, left, right, top, bottom = 820, 390, 72, 802, 35, 335
    all_values = [item for _, values, _, _ in series for item in values]
    low, high = min(all_values), max(all_values)
    pad = max((high - low) * 0.08, 0.01)
    low, high = low - pad, high + pad
    x = lambda index: left + index / 19 * (right - left)
    y = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">', f'<text x="410" y="20" text-anchor="middle" font-size="15" font-weight="bold">{html.escape(title)}</text>']
    for tick in range(5):
        value = low + tick / 4 * (high - low)
        yy = y(value)
        lines += [f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#e4e8ec"/>', f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>']
    for round_number in (1, 5, 10, 15, 20):
        xx = x(round_number - 1)
        lines += [f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" stroke="#f0f2f4"/>', f'<text x="{xx:.1f}" y="353" text-anchor="middle" font-size="11">{round_number}</text>']
    lines += [f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#555"/>', '<text x="410" y="378" text-anchor="middle" font-size="12">Round</text>', f'<text x="15" y="195" transform="rotate(-90 15 195)" text-anchor="middle" font-size="12">{html.escape(ylabel)}</text>']
    for _, values, color, dash in series:
        points = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(values))
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dashed}/>' )
    lines.append("</svg>")
    return "".join(lines)


def legend(include_vanilla: bool = True) -> str:
    keys = []
    if include_vanilla:
        keys.append('<span class="key"><i style="background:#222"></i>Vanilla (dotted)</span>')
    for ratio, color in COLORS.items():
        if ratio is not None:
            keys.append(f'<span class="key"><i style="background:{color}"></i>Ratio {ratio:g}</span>')
    if include_vanilla:
        keys += ['<span class="key"><i class="solid"></i>Global-DP (solid)</span>', '<span class="key"><i class="dashed"></i>Metric privacy (dashed)</span>']
    return f'<div class="legend">{"".join(keys)}</div>'


def all_rounds_section(data: dict, alphas: list[float]) -> str:
    groups = []
    for key, heading in (("clean_difference_pct", "Clean-shadow relative difference"), ("noisy_difference_pct", "Noisy-shadow relative difference")):
        panels = []
        for alpha in alphas:
            chart_series = []
            for privacy, ratio, value in variants(data, alpha):
                values = [float(row[key]) for row in value["rows"]["in"]]
                chart_series.append((METHOD_LABEL[privacy], values, COLORS[ratio], DASHES[privacy]))
            title = f"{heading} — α = {fmt_alpha(alpha)}"
            panels.append(f'<div class="panel"><h3>Dirichlet α = {fmt_alpha(alpha)}</h3>{line_chart(chart_series, title, "Relative difference (%)")}</div>')
        groups.append(f'<h3>{heading}</h3><div class="grid">{"".join(panels)}</div>')
    return f'<section id="all-rounds"><h2>Single-round CIA across all rounds</h2><p class="note">Clean and noisy shadow sets are plotted separately. Lines use IN-remove results.</p>{legend()}{"".join(groups)}</section>'


def auc_section(data: dict, alphas: list[float]) -> str:
    body = []
    for alpha in alphas:
        first = True
        for privacy, ratio, value in variants(data, alpha):
            clean = loss_auc(value, "target_clean_shadow_loss")
            noisy = loss_auc(value, "target_noisy_shadow_loss")
            cls = ' class="alpha-start"' if first else (' class="pair-start"' if privacy == "global-dp" else "")
            first = False
            body.append(f"<tr{cls}>{td(fmt_alpha(alpha))}{td(METHOD_LABEL[privacy])}{td(fmt_ratio(ratio))}{td(clean)}{td(max(clean,1-clean))}{td(noisy)}{td(max(noisy,1-noisy))}</tr>")
    headers = "<tr><th>Dirichlet α</th><th>Method</th><th>Ratio</th><th>Clean pooled AUC</th><th>Clean effective AUC</th><th>Noisy pooled AUC</th><th>Noisy effective AUC</th></tr>"
    note = "Pooled AUC uses all 20 IN and 20 OUT checkpoints. Effective AUC lets the attacker invert the loss direction, so values closer to 0.5 indicate better protection."
    return f'<section id="cia-auc"><h2>Multi-round CIA ROC-AUC</h2><p class="note">{note}</p><div class="panel">{table(headers, body)}</div></section>'


def distance_section(data: dict, alphas: list[float]) -> str:
    panels, body = [], []
    for alpha in alphas:
        chart_series = []
        for privacy, ratio, value in variants(data, alpha):
            if privacy != "metric-privacy":
                continue
            values = [float(value["runs"]["in"]["train_metrics"][str(round_number)]["metric-dp-distance"]) for round_number in ROUNDS]
            chart_series.append((f"Ratio {ratio:g}", values, COLORS[ratio], None))
            body.append(f"<tr>{td(fmt_alpha(alpha))}{td(fmt_ratio(ratio))}{td(8*ratio)}{td(values[0])}{td(values[-1])}{td(min(values))}{td(fmean(values[5:]))}</tr>")
        if chart_series:
            title = f"Metric-privacy distance — α = {fmt_alpha(alpha)}"
            panels.append(f'<div class="panel"><h3>Dirichlet α = {fmt_alpha(alpha)}</h3>{line_chart(chart_series, title, "dᵣ")}</div>')
    headers = "<tr><th>Dirichlet α</th><th>Ratio</th><th>IN noise multiplier</th><th>d₁</th><th>d₂₀</th><th>Minimum</th><th>Mean, rounds 6–20</th></tr>"
    return f'<section id="distance"><h2>Metric-privacy distance d<sub>r</sub></h2><p class="note">Maximum pairwise client-model distance used by the metric-privacy mechanism, shown for each available ratio.</p>{legend(False)}<div class="grid">{"".join(panels)}</div><div class="panel">{table(headers, body)}</div></section>'


def build() -> str:
    data = load()
    alphas = sorted({key[0] for key in data})
    styles = """:root{font-family:Arial,Helvetica,sans-serif;color:#111}body{max-width:1500px;margin:0 auto;padding:28px}header{border-bottom:1px solid #bbb;margin-bottom:28px}h1{margin:0 0 8px;font-size:28px}h2{margin-top:36px;padding-bottom:6px;border-bottom:1px solid #ccc;font-size:21px}h3{font-size:17px}.meta,.note{color:#444;font-size:14px;line-height:1.45}.grid{display:grid;gap:20px}.panel{border:1px solid #ccc;padding:12px;overflow-x:auto;margin-bottom:20px}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #ddd;padding:6px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{border-bottom:2px solid #888}th[colspan]{text-align:center}tbody tr.pair-start td{border-top:2px solid #aaa}tbody tr.alpha-start:not(:first-child) td{border-top:4px solid #555}svg{width:100%;height:auto;display:block}.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 16px;font-size:12px}.key{display:inline-flex;align-items:center;gap:5px}.key i{width:18px;height:3px;display:inline-block}.key i.solid{background:#333}.key i.dashed{background:repeating-linear-gradient(to right,#333 0 10px,transparent 10px 15px)}#utility .grid,#single-round .grid{grid-template-columns:1fr}#all-rounds .grid,#distance .grid{grid-template-columns:repeat(2,minmax(0,1fr))}footer{margin-top:36px;padding-top:10px;border-top:1px solid #bbb;font-size:12px;color:#555}@media(max-width:1000px){#all-rounds .grid,#distance .grid{grid-template-columns:1fr}}@media print{body{max-width:none;padding:12mm}section{break-inside:avoid}}"""
    alpha_text = ", ".join(fmt_alpha(alpha) for alpha in alphas)
    sections = "".join((utility_section(data, alphas), single_round_section(data, alphas), all_rounds_section(data, alphas), auc_section(data, alphas), distance_section(data, alphas)))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CIFAR-10 Dirichlet Removal Results</title><style>{styles}</style></head>
<body><header><h1>CIFAR-10 Dirichlet Removal Results</h1><p class="meta">Dirichlet non-IID partitioning, FedAvg, seed 42; IN/OUT removal adjacency; 8 canonical clients; 20 rounds; 5 local epochs; α ∈ {{{alpha_text}}}; ratios {{0.0025, 0.004, 0.00625}}. Vanilla is the non-private baseline. IN noise multiplier = 8 × ratio; OUT noise multiplier = 7 × ratio.</p></header>{sections}<footer>Source: committed JSON artifacts under <code>results/dirichlet/cifar10/8_clients/</code>. Generated by <code>reports/build_cifar10_dirichlet_presentation.py</code>.</footer></body></html>'''


if __name__ == "__main__":
    OUTPUT.write_text(build())
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
