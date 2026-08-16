"""Build the two-takeaway CIA report requested in supervisor feedback on
reports/accuracy_vs_roc_auc.html:

    1. As the number of clients goes up, attack AUC goes down (checked so far
       only on CIFAR-10).
    2. Noise (Global-DP / Metric-privacy) lowers attack AUC, at a real
       accuracy cost — checked on CIFAR-10 (full noise sweep) plus a single
       vanilla-vs-DP comparison point each on EuroSAT, CIFAR-100, Alzheimer,
       and Fashion-MNIST.

Every number below is computed from committed result artifacts, the same way
build_accuracy_vs_roc_auc.py does it -- nothing here is hand-transcribed.

Run from the repository root:
    uv run python reports/build_cia_takeaways.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from experiments.cia.reports import build_alzheimer_cia_report as transfer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "cia_takeaways.html"
CIFAR10_REMOVE_DIR = ROOT / "results" / "cia" / "cifar10_remove"
CIFAR10_RATIO_SWEEP_DIR = ROOT / "results" / "cia" / "cifar10_remove_ratio_sweep"
METHODS = ("vanilla", "global-dp", "metric-privacy")
LABELS = {
    "vanilla": "Vanilla",
    "global-dp": "Global-DP",
    "metric-privacy": "Metric privacy",
}


def effective_auc(value: float) -> float:
    """AUC after letting an attacker choose the more revealing score direction."""
    return max(value, 1.0 - value)


# ---------------------------------------------------------------------------
# Claim 1: attack AUC vs. client count (CIFAR-10 only -- see report caveat)
# ---------------------------------------------------------------------------

def load_client_count_trend() -> dict:
    """Round-matched (fair, same-round) clean-shadow AUC by client count, all
    three privacy modes, from the actual CIFAR-10 client-removal sweep."""
    rows = json.loads((CIFAR10_REMOVE_DIR / "cia_analysis.json").read_text())
    by_clients: dict[int, dict[str, float]] = {}
    for row in rows:
        clients = int(row["num_clients_canonical"])
        auc = effective_auc(row["multi_round"]["clean"]["round_matched_auc"])
        by_clients.setdefault(clients, {})[row["privacy"]] = auc
    clients_sorted = sorted(by_clients)
    vanilla = [by_clients[c]["vanilla"] for c in clients_sorted]
    gdp = [by_clients[c]["global-dp"] for c in clients_sorted]
    mp = [by_clients[c]["metric-privacy"] for c in clients_sorted]
    band_low = [min(g, m) for g, m in zip(gdp, mp, strict=True)]
    band_high = [max(g, m) for g, m in zip(gdp, mp, strict=True)]
    band_mean = [(g + m) / 2 for g, m in zip(gdp, mp, strict=True)]
    return {
        "clients": clients_sorted,
        "vanilla": vanilla,
        "bandLow": band_low,
        "bandHigh": band_high,
        "bandMean": band_mean,
        "raw": {"global-dp": gdp, "metric-privacy": mp},
    }


# ---------------------------------------------------------------------------
# Claim 2, panel A: CIFAR-10 noise sweep (attack AUC + accuracy vs. noise)
# ---------------------------------------------------------------------------

def round_matched_auc_from_rows(rows: list[dict]) -> float:
    trajectories: dict[str, list[dict]] = {}
    for row in rows:
        trajectories.setdefault(str(row["run_name"]), []).append(row)
    scores: dict[str, list[float]] = {}
    for name, trajectory in trajectories.items():
        trajectory.sort(key=lambda row: int(row["server_round"]))
        scores[name] = [-float(row["target_clean_shadow_loss"]) for row in trajectory]
    in_scores = next(value for name, value in scores.items() if "in-remove" in name)
    out_scores = next(value for name, value in scores.items() if "out-remove" in name)
    directional_auc = statistics.fmean(
        1.0 if score_in > score_out else 0.5 if score_in == score_out else 0.0
        for score_in, score_out in zip(in_scores, out_scores, strict=True)
    )
    return effective_auc(directional_auc)


def late_accuracy(directory: Path) -> float:
    values: list[float] = []
    for path in directory.glob("*.json"):
        if path.name == "cia.json" or path.name.endswith(".evaluation.json"):
            continue
        payload = json.loads(path.read_text())
        metrics = payload["server_evaluate_metrics"]
        for server_round in range(16, 21):
            values.append(float(metrics[str(server_round)]["accuracy"]))
    if len(values) != 10:
        raise RuntimeError(f"Expected two complete 20-round runs in {directory}, found {len(values)} late metrics.")
    return statistics.fmean(values)


def ratio_sweep_directory(clients: int, privacy: str, ratio: float | None) -> Path:
    if privacy == "vanilla":
        suffix = "vanilla"
    else:
        short = {"global-dp": "gdp", "metric-privacy": "mp"}[privacy]
        ratio_token = str(ratio).replace(".", "p")
        suffix = f"{short}-r{ratio_token}"
    directory = CIFAR10_RATIO_SWEEP_DIR / f"n{clients}-{suffix}" / "runs"
    if not directory.is_dir():
        raise RuntimeError(f"Missing CIFAR-10 ratio-sweep run directory: {directory}")
    return directory


def load_cifar10_noise_sweep(clients: int = 48) -> list[dict]:
    """Actual CIFAR-10 removal ratio sweep: attack AUC + accuracy vs. noise level."""
    panels: list[dict] = []
    for privacy in ("global-dp", "metric-privacy"):
        points = []
        for level, ratio in (
            ("No noise", None),
            ("Low", 0.0025),
            ("Medium", 0.004),
            ("High", 0.00625),
        ):
            selected_privacy = "vanilla" if ratio is None else privacy
            directory = ratio_sweep_directory(clients, selected_privacy, ratio)
            points.append({
                "level": level,
                "attack": round_matched_auc_from_rows(json.loads((directory / "cia.json").read_text())),
                "accuracy": late_accuracy(directory),
            })
        panels.append({"privacy": privacy, "label": LABELS[privacy], "clients": clients, "points": points})
    return panels


# ---------------------------------------------------------------------------
# Claim 2, panels B-E: single vanilla-vs-DP snapshot per remaining dataset
# ---------------------------------------------------------------------------

def transfer_snapshot(dataset: str, label: str) -> dict:
    """3-client IN/OUT transfer snapshot (Alzheimer, Fashion-MNIST)."""
    transfer.configure_dataset(dataset)
    in_groups = transfer.load_cia(transfer.IN_DIR / "cia.json")
    out_groups = transfer.load_cia(transfer.OUT_DIR / "cia.json")
    values = []
    for privacy in METHODS:
        attack_auc = effective_auc(transfer.round_matched_auc(
            in_groups, out_groups, privacy, "target_clean_shadow_loss",
        ))
        accuracy_in, _ = transfer.last_five_utility(transfer.IN_DIR, privacy)
        accuracy_out, _ = transfer.last_five_utility(transfer.OUT_DIR, privacy)
        values.append({
            "privacy": privacy,
            "attack": attack_auc,
            "accuracy": statistics.fmean(accuracy_in + accuracy_out),
        })
    return {
        "dataset": label,
        "subtitle": "3-client transfer · non-IID",
        "evidence": "3 seeds · 60 same-round pairs",
        "values": values,
    }


def scaling_accuracy(directory: Path, partition: str, privacy: str) -> float:
    values: list[float] = []
    for path in directory.glob("*.json"):
        if path.name.startswith("cia_"):
            continue
        payload = json.loads(path.read_text())
        metadata = payload.get("metadata")
        if not metadata:
            continue
        if metadata.get("partition_mode") != partition or metadata.get("privacy") != privacy:
            continue
        metrics = payload["server_evaluate_metrics"]
        final_round = max(metrics, key=lambda value: int(value))
        values.append(float(metrics[final_round]["accuracy"]))
    if not values:
        raise RuntimeError(f"No scaling accuracy for {directory=}, {partition=}, {privacy=}")
    return statistics.fmean(values)


def scaling_snapshot(*, dataset: str, clients: int, directory_name: str, seeds: int, pairs: int) -> dict:
    """Single-point vanilla-vs-DP snapshot (EuroSAT, CIFAR-100 scaling sweeps)."""
    directory = ROOT / "results" / directory_name
    analysis = json.loads((directory / "cia_analysis.json").read_text())
    by_privacy = {row["privacy"]: row for row in analysis if row["partition_mode"] == "homogeneous"}
    values = []
    for privacy in METHODS:
        values.append({
            "privacy": privacy,
            "attack": effective_auc(by_privacy[privacy]["target_clean_shadow_loss"]["round_matched_auc"]),
            "accuracy": scaling_accuracy(directory, "homogeneous", privacy),
        })
    return {
        "dataset": dataset,
        "subtitle": f"{clients} clients · homogeneous",
        "evidence": f"{seeds} seed{'s' if seeds != 1 else ''} · {pairs} same-round pairs",
        "values": values,
    }


def load_dataset_snapshots() -> list[dict]:
    return [
        scaling_snapshot(dataset="EuroSAT", clients=48, directory_name="cia_eurosat_scaling", seeds=3, pairs=33),
        scaling_snapshot(dataset="CIFAR-100", clients=100, directory_name="cia_cifar100_scaling", seeds=1, pairs=26),
        transfer_snapshot("alzheimer", "Alzheimer"),
        transfer_snapshot("fashion", "Fashion-MNIST"),
    ]


def build() -> None:
    data = {
        "clientTrend": load_client_count_trend(),
        "cifar10Sweep": load_cifar10_noise_sweep(),
        "datasetSnapshots": load_dataset_snapshots(),
    }
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUTPUT.write_text(TEMPLATE.replace("__DATA__", data_json))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CIA: two clear trends</title>
<style>
  :root {
    color-scheme: light;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --surface: #fcfcfb;
    --page: #f9f9f7;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --ink-muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --vanilla: #e34948;
    --gdp: #2a78d6;
    --mp: #1baf7a;
    --attack: #e34948;
    --accuracy: #2a78d6;
    --context: #898781;
  }
  * { box-sizing: border-box; }
  body { max-width: 1180px; margin: 0 auto; padding: 28px; color: var(--ink); background: var(--page); }
  header { border-bottom: 1px solid var(--axis); margin-bottom: 26px; }
  h1 { margin: 0 0 8px; font-size: 27px; }
  h2 { margin-top: 40px; padding-bottom: 6px; border-bottom: 1px solid var(--grid); font-size: 21px; }
  h3 { margin: 0 0 3px; font-size: 16px; }
  p, li { line-height: 1.5; }
  .lead { max-width: 900px; font-size: 16px; color: var(--ink-2); }
  .meta { color: var(--ink-2); font-size: 14px; }
  .caveat { margin: 14px 0 4px; padding: 11px 14px; border-left: 4px solid var(--ink-muted); background: var(--surface); font-size: 13.5px; color: var(--ink-2); line-height: 1.5; }
  .panel { border: 1px solid var(--border); border-radius: 6px; padding: 14px; background: var(--surface); }
  .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }
  .grid-1 { display: grid; grid-template-columns: minmax(0, 620px); gap: 16px; justify-content: center; }
  .card-head { margin-bottom: 4px; }
  .card-head p { margin: 2px 0 0; color: var(--ink-2); font-size: 12.5px; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .gridline { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1.2; }
  .tick { fill: var(--ink-muted); font-size: 12px; }
  .axis-label { fill: var(--ink-2); font-size: 12.5px; font-weight: 600; }
  .value-label { fill: var(--ink); font-size: 12px; font-weight: 700; }
  .chance { stroke: var(--ink-muted); stroke-width: 1.5; stroke-dasharray: 5 4; }
  .chance-label { fill: var(--ink-muted); font-size: 11.5px; font-style: italic; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 10px 0 16px; font-size: 13px; color: var(--ink-2); }
  .key { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 20px; height: 4px; border-radius: 2px; display: inline-block; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  footer { margin-top: 40px; padding-top: 10px; border-top: 1px solid var(--axis); color: var(--ink-2); font-size: 12px; line-height: 1.6; }
  code { font-size: 11px; }
  table.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 6px; }
  table.data-table th, table.data-table td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; }
  table.data-table th:first-child, table.data-table td:first-child { text-align: left; }
  table.data-table thead th { border-bottom: 2px solid var(--axis); color: var(--ink-2); font-weight: 600; }
  details { margin-top: 10px; }
  summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
  @media (max-width: 760px) {
    body { padding: 16px; }
    .grid-2 { grid-template-columns: 1fr; }
  }
  @media print {
    body { max-width: none; padding: 10mm; background: #fff; }
    h2 { break-before: page; }
    .panel { break-inside: avoid; }
  }
</style>
</head>
<body data-palette="#e34948,#2a78d6,#1baf7a" data-mode="light">
<header>
  <h1>Client inference attack: two clear trends</h1>
  <p class="lead">Follow-up on the earlier attack-AUC-vs-accuracy report. Two questions, one plot per dataset each, built only from committed run data.</p>
  <p class="meta"><strong>How to read every chart:</strong> Attack AUC measures how well an attacker can tell whether a client took part in training, after letting the attacker pick the more revealing direction. <strong>50% = pure guessing</strong> (safe); <strong>100% = attacker always right</strong> (unsafe). Model accuracy is plain test accuracy: higher is better.</p>
</header>

<section>
  <h2>1 &middot; More clients &rarr; harder attack</h2>
  <p class="meta">Actual CIFAR-10 client-removal sweep, non-IID, one seed, same protocol at every client count. Red is vanilla FL (no privacy mechanism) &mdash; the case the trend is clearest for. The gray band is the range covered by Global-DP and Metric-privacy at the same client count, shown only as supporting context.</p>
  <div class="legend">
    <span class="key"><i class="swatch" style="background:var(--vanilla)"></i>Vanilla FL</span>
    <span class="key"><i class="swatch" style="background:var(--context)"></i>Global-DP &amp; Metric-privacy (range)</span>
    <span class="key"><i class="swatch" style="height:1px;background:var(--ink-muted)"></i>50% = random guess</span>
  </div>
  <div class="grid-1"><div id="client-trend-plot" class="panel"></div></div>
  <div class="caveat"><strong>Caveat:</strong> this exact test (several client counts, everything else fixed) has only been run on CIFAR-10 so far. We don't yet know if the same drop happens on our other datasets &mdash; that's the natural next experiment, not an established fact yet.</div>
</section>

<section>
  <h2>2 &middot; More noise &rarr; harder attack, but a real accuracy cost</h2>
  <p class="meta">First, a full noise sweep on CIFAR-10 (48 clients, non-IID) so the trade-off is visible as noise actually increases. Then one vanilla-vs-DP comparison point each on four other datasets, to check the same direction holds elsewhere.</p>
  <div class="legend">
    <span class="key"><i class="swatch" style="background:var(--attack)"></i>Attack AUC &mdash; lower is safer</span>
    <span class="key"><i class="swatch" style="background:var(--accuracy)"></i>Model accuracy &mdash; higher is better</span>
  </div>
  <h3 style="margin-top:18px">CIFAR-10 &middot; full noise sweep</h3>
  <div class="grid-2" id="cifar10-sweep-plots"></div>

  <h3 style="margin-top:28px">Vanilla vs. DP, one dataset at a time</h3>
  <p class="meta">Three bars per group: Vanilla, Global-DP, Metric-privacy.</p>
  <div class="legend">
    <span class="key"><i class="dot" style="background:var(--vanilla)"></i>Vanilla</span>
    <span class="key"><i class="dot" style="background:var(--gdp)"></i>Global-DP</span>
    <span class="key"><i class="dot" style="background:var(--mp)"></i>Metric privacy</span>
  </div>
  <div class="grid-2" id="dataset-snapshot-plots"></div>
  <div class="caveat">The size of the accuracy cost varies a lot by dataset in this data &mdash; small on EuroSAT and Fashion-MNIST, large on CIFAR-100 &mdash; so "noise lowers the attack" looks solid here, but "at a very heavy cost" isn't true everywhere we've checked.</div>
</section>

<section>
  <details>
    <summary>Full numbers behind every chart (for anyone who wants exact values instead of reading the plots)</summary>
    <div id="data-tables"></div>
  </details>
</section>

<footer>
  Attack metric: clean-shadow raw-loss round-matched (fair, same-round) ROC AUC, reported after allowing the attacker to reverse its score direction. Accuracy is averaged over matched IN/OUT trajectories (late rounds for 20-round experiments; final checkpoints for EuroSAT/CIFAR-100 scaling). Sources: <code>results/cia/cifar10_remove/cia_analysis.json</code>, <code>results/cia/cifar10_remove_ratio_sweep/</code>, <code>results/cia_eurosat_scaling/</code>, <code>results/cia_cifar100_scaling/</code>, <code>results/planned_runs/{alzheimer,fashion}/</code>. CIFAR-100 has one seed and should be treated as preliminary. Generated by <code>reports/build_cia_takeaways.py</code>.
</footer>

<script id="report-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const colors = {'vanilla':'#e34948','global-dp':'#2a78d6','metric-privacy':'#1baf7a'};
  const labels = {'vanilla':'Vanilla','global-dp':'Global-DP','metric-privacy':'Metric privacy'};
  const el = (name, attrs={}, text='') => {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    return node;
  };
  const pct = value => `${Math.round(value * 100)}%`;

  function card(title, subtitle) {
    const section = document.createElement('article');
    section.className = 'panel';
    const head = document.createElement('div');
    head.className = 'card-head';
    head.innerHTML = `<h3>${title}</h3><p>${subtitle}</p>`;
    const svg = el('svg', {viewBox:'0 0 520 340', role:'img', 'aria-label':title});
    section.append(head, svg);
    return {section, svg};
  }

  function axes(svg, {left=58, right=495, top=20, bottom=270, min=.45, max=1, ticks=[.5,.6,.7,.8,.9,1]}) {
    const y = value => bottom - (value-min)/(max-min)*(bottom-top);
    ticks.forEach(value => {
      const yy = y(value);
      svg.append(el('line', {x1:left, y1:yy, x2:right, y2:yy, class:'gridline'}));
      svg.append(el('text', {x:left-9, y:yy+4, 'text-anchor':'end', class:'tick'}, pct(value)));
    });
    svg.append(el('line', {x1:left, y1:top, x2:left, y2:bottom, class:'axis'}));
    svg.append(el('line', {x1:left, y1:bottom, x2:right, y2:bottom, class:'axis'}));
    return y;
  }

  function linePath(points) {
    return points.map((point, index) => `${index ? 'L' : 'M'}${point[0]},${point[1]}`).join(' ');
  }

  function dot(svg, x, y, color) {
    svg.append(el('circle', {cx:x, cy:y, r:5, fill:color, stroke:'#fcfcfb', 'stroke-width':2}));
  }

  function roundedTopBar(svg, x, w, yTop, yBottom, color) {
    const r = 4;
    const d = `M${x},${yBottom} L${x},${yTop+r} Q${x},${yTop} ${x+r},${yTop} `
      + `L${x+w-r},${yTop} Q${x+w},${yTop} ${x+w},${yTop+r} L${x+w},${yBottom} Z`;
    svg.append(el('path', {d, fill:color}));
  }

  // ---- Section 1: client-count trend (emphasis: vanilla vs. gray DP band) ----
  (() => {
    const plot = data.clientTrend;
    const {section, svg} = card(
      'CIFAR-10 · client-removal sweep',
      'Non-IID · 1 seed · attack AUC, round-matched'
    );
    const left=58, right=495, top=20, bottom=270;
    const y = axes(svg, {left,right,top,bottom,min:.45,max:1});
    const n = plot.clients.length;
    const xs = plot.clients.map((_, i) => left + i * (right-left)/(n-1));
    plot.clients.forEach((clients, i) => {
      svg.append(el('text',{x:xs[i],y:bottom+23,'text-anchor':'middle',class:'tick'},`${clients} clients`));
    });
    svg.append(el('text',{x:(left+right)/2,y:bottom+46,'text-anchor':'middle',class:'axis-label'},'Number of clients'));
    svg.append(el('line', {x1:left,y1:y(.5),x2:right,y2:y(.5),class:'chance'}));
    svg.append(el('text', {x:right-4,y:y(.5)-6,'text-anchor':'end',class:'chance-label'},'50% = random guess'));

    // gray context band (min/max) + mean line
    const bandTop = plot.bandHigh.map((v,i) => [xs[i], y(v)]);
    const bandBottom = plot.bandLow.map((v,i) => [xs[i], y(v)]).reverse();
    svg.append(el('path', {d: linePath(bandTop) + ' L' + bandBottom.map(p=>p.join(',')).join(' L') + ' Z', fill:'#898781', opacity:0.14}));
    const meanPts = plot.bandMean.map((v,i) => [xs[i], y(v)]);
    svg.append(el('path', {d: linePath(meanPts), fill:'none', stroke:'#898781', 'stroke-width':2}));
    meanPts.forEach(([x,yy]) => dot(svg, x, yy, '#898781'));

    // vanilla, bold + labeled
    const vPts = plot.vanilla.map((v,i) => [xs[i], y(v)]);
    svg.append(el('path', {d: linePath(vPts), fill:'none', stroke:'#e34948', 'stroke-width':2}));
    vPts.forEach(([x,yy], i) => {
      dot(svg, x, yy, '#e34948');
      svg.append(el('text',{x,y:yy-11,'text-anchor':'middle',class:'value-label'}, pct(plot.vanilla[i])));
    });
    document.getElementById('client-trend-plot').replaceWith(section);
    section.id = 'client-trend-plot';
  })();

  // ---- Section 2a: CIFAR-10 noise sweep (attack + accuracy per privacy mode) ----
  data.cifar10Sweep.forEach(panel => {
    const {section, svg} = card(
      `${panel.label} · CIFAR-10`,
      `${panel.clients} clients · non-IID · 1 seed · noise increases left to right`
    );
    const left=58, right=495, top=20, bottom=270;
    const y = axes(svg, {left,right,top,bottom,min:0,max:1,ticks:[0,.25,.5,.75,1]});
    const n = panel.points.length;
    const xs = panel.points.map((_, i) => left + i * (right-left)/(n-1));
    panel.points.forEach((point, i) => {
      svg.append(el('text',{x:xs[i],y:bottom+23,'text-anchor':'middle',class:'tick'}, point.level));
    });
    svg.append(el('text',{x:(left+right)/2,y:bottom+46,'text-anchor':'middle',class:'axis-label'},'Noise level'));
    svg.append(el('line', {x1:left,y1:y(.5),x2:right,y2:y(.5),class:'chance'}));
    svg.append(el('text', {x:right-4,y:y(.5)-6,'text-anchor':'end',class:'chance-label'},'50% = random guess'));

    [{key:'attack', color:'#e34948'}, {key:'accuracy', color:'#2a78d6'}].forEach(series => {
      const pts = panel.points.map((point,i) => [xs[i], y(point[series.key])]);
      svg.append(el('path', {d: linePath(pts), fill:'none', stroke:series.color, 'stroke-width':2}));
      pts.forEach(([x,yy], i) => {
        dot(svg, x, yy, series.color);
        if (i === 0 || i === panel.points.length - 1) {
          svg.append(el('text',{x,y:yy-11,'text-anchor':'middle',class:'value-label'}, pct(panel.points[i][series.key])));
        }
      });
    });
    document.getElementById('cifar10-sweep-plots').append(section);
  });

  // ---- Section 2b: per-dataset vanilla-vs-DP grouped bars ----
  data.datasetSnapshots.forEach(plot => {
    const {section, svg} = card(plot.dataset, `${plot.subtitle} · ${plot.evidence}`);
    const left=58, right=495, top=20, bottom=270;
    const y = axes(svg, {left,right,top,bottom,min:0,max:1,ticks:[0,.25,.5,.75,1]});
    const centers = [180, 385];
    const groupNames = ['Attack AUC ↓', 'Model accuracy ↑'];
    const keys = ['attack', 'accuracy'];
    const barWidth = 30, gap = 8;
    svg.append(el('line', {x1:85,y1:y(.5),x2:275,y2:y(.5),class:'chance'}));
    svg.append(el('text', {x:278,y:y(.5)+4,class:'chance-label'}, 'random'));
    centers.forEach((center, groupIndex) => {
      const start = center - (barWidth*3 + gap*2)/2;
      plot.values.forEach((value, index) => {
        const x = start + index*(barWidth+gap);
        const valueY = y(value[keys[groupIndex]]);
        roundedTopBar(svg, x, barWidth, valueY, bottom, colors[value.privacy]);
        svg.append(el('text',{x:x+barWidth/2,y:valueY-7,'text-anchor':'middle',class:'value-label'}, pct(value[keys[groupIndex]])));
      });
      svg.append(el('text',{x:center,y:bottom+25,'text-anchor':'middle',class:'axis-label'}, groupNames[groupIndex]));
    });
    document.getElementById('dataset-snapshot-plots').append(section);
  });

  // ---- Appendix: raw numbers table ----
  (() => {
    const container = document.getElementById('data-tables');

    const t1 = document.createElement('table');
    t1.className = 'data-table';
    t1.innerHTML = '<caption style="text-align:left;font-weight:600;margin-bottom:4px">1 · Client-count trend (CIFAR-10, attack AUC)</caption>'
      + '<thead><tr><th>Clients</th><th>Vanilla</th><th>Global-DP</th><th>Metric privacy</th></tr></thead>';
    const tbody1 = document.createElement('tbody');
    data.clientTrend.clients.forEach((c, i) => {
      const row = document.createElement('tr');
      row.innerHTML = `<td>${c}</td><td>${pct(data.clientTrend.vanilla[i])}</td>`
        + `<td>${pct(data.clientTrend.raw['global-dp'][i])}</td><td>${pct(data.clientTrend.raw['metric-privacy'][i])}</td>`;
      tbody1.append(row);
    });
    t1.append(tbody1);
    container.append(t1);

    data.cifar10Sweep.forEach(panel => {
      const t = document.createElement('table');
      t.className = 'data-table';
      t.innerHTML = `<caption style="text-align:left;font-weight:600;margin:14px 0 4px">2 · CIFAR-10 noise sweep — ${panel.label}</caption>`
        + '<thead><tr><th>Noise level</th><th>Attack AUC</th><th>Accuracy</th></tr></thead>';
      const tbody = document.createElement('tbody');
      panel.points.forEach(point => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${point.level}</td><td>${pct(point.attack)}</td><td>${pct(point.accuracy)}</td>`;
        tbody.append(row);
      });
      t.append(tbody);
      container.append(t);
    });

    const t2 = document.createElement('table');
    t2.className = 'data-table';
    t2.innerHTML = '<caption style="text-align:left;font-weight:600;margin:14px 0 4px">2 · Vanilla-vs-DP snapshots by dataset</caption>'
      + '<thead><tr><th>Dataset</th><th>Method</th><th>Attack AUC</th><th>Accuracy</th></tr></thead>';
    const tbody2 = document.createElement('tbody');
    data.datasetSnapshots.forEach(plot => {
      plot.values.forEach(value => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${plot.dataset}</td><td>${labels[value.privacy]}</td><td>${pct(value.attack)}</td><td>${pct(value.accuracy)}</td>`;
        tbody2.append(row);
      });
    });
    t2.append(tbody2);
    container.append(t2);
  })();
})();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    build()
