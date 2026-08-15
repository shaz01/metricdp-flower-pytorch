"""Build the standalone CIA big-picture visualization report.

Run from the repository root:
    uv run python reports/build_accuracy_vs_roc_auc.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from experiments.cia.reports import build_alzheimer_cia_report as transfer
from experiments.cia.reports import build_cifar_full_cia_report as full


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "accuracy_vs_roc_auc.html"
METHODS = ("vanilla", "global-dp", "metric-privacy")
LABELS = {
    "vanilla": "Vanilla",
    "global-dp": "Global-DP",
    "metric-privacy": "Metric privacy",
}


def round_matched_full_auc(
    data: dict,
    clients: int,
    privacy: str,
    ratio: float | None,
) -> float:
    """Compare clean-shadow IN/OUT scores only at the same seed and round."""
    outcomes: list[float] = []
    for seed in full.SEEDS:
        in_scores = full.scores(
            data[("in-replace", clients, privacy, ratio, seed)],
            "target_clean_shadow_loss",
        )
        out_scores = full.scores(
            data[("out-replace", clients, privacy, ratio, seed)],
            "target_clean_shadow_loss",
        )
        outcomes.extend(
            1.0 if in_score > out_score else 0.5 if in_score == out_score else 0.0
            for in_score, out_score in zip(in_scores, out_scores, strict=True)
        )
    return statistics.fmean(outcomes)


def full_accuracy(
    data: dict,
    clients: int,
    privacy: str,
    ratio: float | None,
) -> float:
    """Average late-round task accuracy over matched IN and OUT trajectories."""
    accuracy_in, _ = full.utility(data, "in-replace", clients, privacy, ratio)
    accuracy_out, _ = full.utility(data, "out-replace", clients, privacy, ratio)
    return statistics.fmean((accuracy_in, accuracy_out))


def load_client_trends(full_data: dict) -> list[dict]:
    replacement = {
        "dataset": "CIFAR-10",
        "protocol": "Full replacement",
        "note": "3 seeds · vanilla only",
        "clients": list(full.CLIENTS),
        "pooled": [
            full.pooled_auc(
                full_data,
                clients,
                "vanilla",
                None,
                "target_clean_shadow_loss",
            )
            for clients in full.CLIENTS
        ],
        "matched": [
            round_matched_full_auc(full_data, clients, "vanilla", None)
            for clients in full.CLIENTS
        ],
    }

    removal_analysis = json.loads(
        (ROOT / "results" / "cia" / "cifar10_remove" / "cia_analysis.json").read_text()
    )
    vanilla_rows = sorted(
        (row for row in removal_analysis if row["privacy"] == "vanilla"),
        key=lambda row: row["num_clients_canonical"],
    )
    removal = {
        "dataset": "CIFAR-10",
        "protocol": "Client removal",
        "note": "1 seed · vanilla only",
        "clients": [row["num_clients_canonical"] for row in vanilla_rows],
        "pooled": [row["multi_round"]["clean"]["pooled_auc"] for row in vanilla_rows],
        "matched": [
            row["multi_round"]["clean"]["round_matched_auc"]
            for row in vanilla_rows
        ],
    }
    return [replacement, removal]


def load_noise_sweep(full_data: dict) -> list[dict]:
    rows: list[dict] = []
    for privacy in ("global-dp", "metric-privacy"):
        values = [
            {
                "level": "No noise",
                "ratio": 0.0,
                "sigma": 0.0,
                "attack": round_matched_full_auc(full_data, 16, "vanilla", None),
                "accuracy": full_accuracy(full_data, 16, "vanilla", None),
            }
        ]
        for level, ratio in zip(("Low", "Medium", "High"), (0.0025, 0.003333, 0.00625), strict=True):
            values.append(
                {
                    "level": level,
                    "ratio": ratio,
                    "sigma": ratio * 5.0,
                    "attack": round_matched_full_auc(full_data, 16, privacy, ratio),
                    "accuracy": full_accuracy(full_data, 16, privacy, ratio),
                }
            )
        rows.append({"privacy": privacy, "label": LABELS[privacy], "values": values})
    return rows


def planned_snapshot(dataset: str) -> dict:
    transfer.configure_dataset(dataset)
    in_groups = transfer.load_cia(transfer.IN_DIR / "cia.json")
    out_groups = transfer.load_cia(transfer.OUT_DIR / "cia.json")
    values = []
    for privacy in METHODS:
        attack_auc = transfer.round_matched_auc(
            in_groups,
            out_groups,
            privacy,
            "target_clean_shadow_loss",
        )
        accuracy_in, _ = transfer.last_five_utility(transfer.IN_DIR, privacy)
        accuracy_out, _ = transfer.last_five_utility(transfer.OUT_DIR, privacy)
        values.append(
            {
                "privacy": privacy,
                "attack": attack_auc,
                "accuracy": statistics.fmean(accuracy_in + accuracy_out),
            }
        )
    labels = {"alzheimer": "Alzheimer", "fashion": "Fashion-MNIST"}
    return {
        "dataset": labels[dataset],
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


def scaling_snapshot(
    *,
    dataset: str,
    clients: int,
    directory_name: str,
    prefix: str,
    seeds: int,
    pairs: int,
) -> dict:
    directory = ROOT / "results" / directory_name
    analysis = json.loads((directory / "cia_analysis.json").read_text())
    by_privacy = {
        row["privacy"]: row
        for row in analysis
        if row["partition_mode"] == "homogeneous"
    }
    values = []
    for privacy in METHODS:
        values.append(
            {
                "privacy": privacy,
                "attack": by_privacy[privacy]["target_clean_shadow_loss"]["round_matched_auc"],
                "accuracy": scaling_accuracy(directory, "homogeneous", privacy),
            }
        )
    return {
        "dataset": dataset,
        "subtitle": f"{clients} clients · homogeneous",
        "evidence": f"{seeds} seed{'s' if seeds != 1 else ''} · {pairs} same-round pairs",
        "values": values,
        "prefix": prefix,
    }


def load_snapshots() -> list[dict]:
    return [
        planned_snapshot("alzheimer"),
        planned_snapshot("fashion"),
        scaling_snapshot(
            dataset="EuroSAT",
            clients=48,
            directory_name="cia_eurosat_scaling",
            prefix="eurosat",
            seeds=3,
            pairs=33,
        ),
        scaling_snapshot(
            dataset="CIFAR-100",
            clients=100,
            directory_name="cia_cifar100_scaling",
            prefix="cifar100",
            seeds=1,
            pairs=26,
        ),
    ]


def build() -> None:
    full_data = full.load()
    data = {
        "clientTrends": load_client_trends(full_data),
        "noiseSweep": load_noise_sweep(full_data),
        "snapshots": load_snapshots(),
    }
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUTPUT.write_text(TEMPLATE.replace("__DATA__", data_json))
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)} with "
        f"{len(data['clientTrends'])} client-count plots, "
        f"{len(data['noiseSweep'])} noise-sweep plots, and "
        f"{len(data['snapshots'])} dataset checks."
    )


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIA: clients, noise, and accuracy</title>
  <style>
    :root { color-scheme: light; font-family: Arial, Helvetica, sans-serif; }
    * { box-sizing: border-box; }
    body { max-width: 1180px; margin: 0 auto; padding: 28px; color: #111; background: #fff; }
    header { border-bottom: 1px solid #bbb; margin-bottom: 28px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    h2 { margin-top: 36px; padding-bottom: 6px; border-bottom: 1px solid #ccc; font-size: 22px; }
    h3 { margin: 0 0 5px; font-size: 17px; }
    p { line-height: 1.45; }
    .meta, .note { color: #444; font-size: 14px; }
    .lead { max-width: 900px; font-size: 16px; }
    .panel { border: 1px solid #ccc; padding: 14px; background: #fff; }
    .verdicts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 20px 0 8px; }
    .verdict { border: 1px solid #ccc; border-top-width: 5px; padding: 14px; }
    .verdict.caution { border-top-color: #c57b00; }
    .verdict.supported { border-top-color: #18733c; }
    .verdict strong { display: block; margin-bottom: 5px; font-size: 17px; }
    .verdict p { margin: 0; font-size: 14px; color: #333; }
    .plot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .dataset-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .plot-card { min-width: 0; }
    .card-head { min-height: 51px; margin-bottom: 4px; }
    .card-head p { margin: 0; color: #555; font-size: 13px; }
    svg { display: block; width: 100%; height: auto; overflow: visible; }
    .gridline { stroke: #e2e2e2; stroke-width: 1; }
    .axis { stroke: #777; stroke-width: 1.2; }
    .tick { fill: #444; font-size: 12px; }
    .axis-label { fill: #222; font-size: 13px; font-weight: bold; }
    .value-label { fill: #222; font-size: 12px; font-weight: bold; }
    .chance { stroke: #999; stroke-width: 1.5; stroke-dasharray: 5 4; }
    .legend { display: flex; flex-wrap: wrap; gap: 9px 20px; margin: 10px 0 14px; font-size: 13px; }
    .key { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 22px; height: 4px; display: inline-block; }
    .swatch.dashed { height: 0; border-top: 3px dashed #888; }
    .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
    .callout { margin: 16px 0; padding: 12px 14px; border-left: 5px solid #c57b00; background: #faf8f3; font-size: 14px; line-height: 1.45; }
    .callout.good { border-left-color: #18733c; background: #f5faf7; }
    .plain-list { margin: 10px 0 0; padding-left: 21px; }
    .plain-list li { margin: 6px 0; line-height: 1.4; }
    .method-label { font-size: 13px; font-weight: bold; }
    .takeaway-table { width: 100%; margin-top: 18px; border-collapse: collapse; font-size: 14px; }
    .takeaway-table th, .takeaway-table td { padding: 9px 10px; border-bottom: 1px solid #ddd; text-align: left; vertical-align: top; }
    .takeaway-table thead th { border-bottom: 2px solid #888; }
    .yes { color: #18733c; font-weight: bold; }
    .mixed { color: #9a5b00; font-weight: bold; }
    footer { margin-top: 36px; padding-top: 10px; border-top: 1px solid #bbb; color: #555; font-size: 12px; line-height: 1.5; }
    code { font-size: 11px; }
    @media (max-width: 760px) {
      body { padding: 16px; }
      .verdicts, .plot-grid, .dataset-grid { grid-template-columns: 1fr; }
    }
    @media print {
      body { max-width: none; padding: 10mm; }
      h2 { break-before: page; }
      .plot-card { break-inside: avoid; }
    }
  </style>
</head>
<body>
<header>
  <h1>Client inference: the big picture</h1>
  <p class="meta">Two simple questions: Do more clients make the attack harder? Does noise buy privacy by sacrificing model accuracy?</p>
</header>

<section>
  <p class="lead"><strong>How to read the plots:</strong> attack AUC measures attacker success. <strong>50% means random guessing</strong>, so lower is safer. Model accuracy measures useful prediction performance, so higher is better.</p>
  <div class="verdicts">
    <div class="verdict caution">
      <strong>1 · More clients: not yet confirmed</strong>
      <p>The downward trend appears only in the older pooled score. A fair same-round check does not show a smooth decline, and only CIFAR-10 has been tested at several client counts.</p>
    </div>
    <div class="verdict supported">
      <strong>2 · More noise: supported, with limits</strong>
      <p>In the controlled 16-client CIFAR-10 sweep, more noise lowers attack AUC and accuracy together. The cost is heavy at the highest noise, but not at every tested setting or dataset.</p>
    </div>
  </div>
</section>

<section>
  <h2>1 · Does attack AUC fall as the number of clients rises?</h2>
  <p class="note">Vanilla only, as suggested. Each protocol gets its own plot. The gray line is the older score that compares checkpoints from different training rounds; the black line compares IN and OUT at the same seed and round.</p>
  <div class="legend">
    <span class="key"><i class="swatch" style="background:#111"></i>Fair comparison: same round</span>
    <span class="key"><i class="swatch dashed"></i>Old comparison: mixed rounds</span>
    <span class="key"><i class="swatch" style="height:1px;background:#999"></i>50% = random guess</span>
  </div>
  <div id="client-plots" class="plot-grid"></div>
  <div class="callout">
    <strong>Actionable takeaway:</strong> we should not claim a general client-count privacy benefit yet. The next clean test is the same multi-count CIA sweep on at least one non-CIFAR dataset, with the same-round AUC chosen before running it.
  </div>
</section>

<section>
  <h2>2 · Does more noise reduce attack AUC at an accuracy cost?</h2>
  <p class="note">Controlled subset: CIFAR-10 full replacement, 16 active clients, non-IID, three seeds. Only noise changes within each plot. Both measures use the same 0–100% scale.</p>
  <div class="legend">
    <span class="key"><i class="swatch" style="background:#a3212b"></i>Attack AUC — lower is safer</span>
    <span class="key"><i class="swatch" style="background:#18733c"></i>Model accuracy — higher is better</span>
  </div>
  <div id="noise-plots" class="plot-grid"></div>
  <div class="callout good">
    <strong>Clear trade-off:</strong> at the highest tested noise, Global-DP cuts attack AUC by about 18 percentage points and accuracy by about 11 points; Metric privacy cuts attack AUC by about 12 points and accuracy by about 8 points.
  </div>

  <h3 style="margin-top:28px">Does noise help on other datasets?</h3>
  <p class="note">One separate plot per dataset, using the clean-shadow same-round attack AUC. These are mechanism snapshots at each dataset's tested noise—not additional noise sweeps. Homogeneous partitions are used for EuroSAT and CIFAR-100 because they show the clearest vanilla leakage.</p>
  <div class="legend">
    <span class="key"><i class="dot" style="background:#111"></i>Vanilla</span>
    <span class="key"><i class="dot" style="background:#2166ac"></i>Global-DP</span>
    <span class="key"><i class="dot" style="background:#b2182b"></i>Metric privacy</span>
  </div>
  <div id="dataset-plots" class="dataset-grid"></div>

  <table class="takeaway-table">
    <thead><tr><th>Dataset</th><th>Does DP lower attack AUC?</th><th>Accuracy cost</th></tr></thead>
    <tbody>
      <tr><td>Alzheimer</td><td class="yes">Yes</td><td>Small for Metric privacy; larger for Global-DP</td></tr>
      <tr><td>Fashion-MNIST</td><td class="mixed">Barely</td><td>Essentially none</td></tr>
      <tr><td>EuroSAT</td><td class="yes">Yes</td><td>Small to moderate</td></tr>
      <tr><td>CIFAR-100</td><td class="yes">Yes</td><td>Very large</td></tr>
    </tbody>
  </table>
  <div class="callout">
    <strong>Overall:</strong> “noise can reduce CIA AUC” is supported across several datasets. “It always comes at a very heavy accuracy cost” is too strong: that is clear at high noise and on CIFAR-100, but not on Fashion-MNIST, Alzheimer, or EuroSAT at their tested settings.
  </div>
</section>

<footer>
  Attack metric: directional clean-shadow raw-loss ROC AUC, with the same-round value treated as the primary check. Accuracy is averaged over matched IN/OUT trajectories (late rounds for 20-round experiments; final checkpoints for EuroSAT/CIFAR-100). CIFAR-100 has one seed and should be treated as preliminary. Sources: <code>results/planned_runs/</code>, <code>results/cia/cifar10_remove/cia_analysis.json</code>, <code>results/cia_eurosat_scaling/cia_analysis.json</code>, and <code>results/cia_cifar100_scaling/cia_analysis.json</code>.
</footer>

<script id="report-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const colors = {'vanilla':'#111','global-dp':'#2166ac','metric-privacy':'#b2182b'};
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
    section.className = 'panel plot-card';
    const head = document.createElement('div');
    head.className = 'card-head';
    head.innerHTML = `<h3>${title}</h3><p>${subtitle}</p>`;
    const svg = el('svg', {viewBox:'0 0 520 330', role:'img', 'aria-label':title});
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

  function path(points) {
    return points.map((point, index) => `${index ? 'L' : 'M'}${point[0]},${point[1]}`).join(' ');
  }

  data.clientTrends.forEach(plot => {
    const {section, svg} = card(`${plot.dataset} · ${plot.protocol}`, plot.note);
    const left=58, right=495, top=20, bottom=270;
    const y = axes(svg, {left,right,top,bottom,min:.45,max:1});
    const xs = plot.clients.map((_, index) => left + index * (right-left)/(plot.clients.length-1));
    svg.append(el('line', {x1:left,y1:y(.5),x2:right,y2:y(.5),class:'chance'}));
    plot.clients.forEach((clients,index) => {
      svg.append(el('text',{x:xs[index],y:bottom+23,'text-anchor':'middle',class:'tick'},String(clients)));
    });
    svg.append(el('text',{x:(left+right)/2,y:bottom+51,'text-anchor':'middle',class:'axis-label'},'Number of clients'));
    const series = [
      {values:plot.pooled,color:'#888',dash:'7 5',width:3},
      {values:plot.matched,color:'#111',dash:'',width:4},
    ];
    series.forEach(item => {
      const points=item.values.map((value,index)=>[xs[index],y(value)]);
      svg.append(el('path',{d:path(points),fill:'none',stroke:item.color,'stroke-width':item.width,'stroke-dasharray':item.dash}));
      points.forEach(([xpos,ypos],index)=>{
        svg.append(el('circle',{cx:xpos,cy:ypos,r:5,fill:'#fff',stroke:item.color,'stroke-width':3}));
        if (item.color === '#111') svg.append(el('text',{x:xpos,y:ypos-10,'text-anchor':'middle',class:'value-label'},pct(item.values[index])));
      });
    });
    document.getElementById('client-plots').append(section);
  });

  data.noiseSweep.forEach(plot => {
    const {section, svg} = card(plot.label, 'CIFAR-10 · 16 clients · same-round AUC');
    const left=58, right=495, top=20, bottom=270;
    const y = axes(svg,{left,right,top,bottom,min:.65,max:1,ticks:[.7,.8,.9,1]});
    const xs=plot.values.map((_,index)=>left+index*(right-left)/(plot.values.length-1));
    plot.values.forEach((value,index)=>{
      svg.append(el('text',{x:xs[index],y:bottom+22,'text-anchor':'middle',class:'tick'},value.level));
      const noise = index === 0 ? 'σ = 0' : `σ = ${value.sigma.toFixed(4)}`;
      svg.append(el('text',{x:xs[index],y:bottom+39,'text-anchor':'middle',class:'tick'},noise));
    });
    const series=[
      {key:'attack',color:'#a3212b'},
      {key:'accuracy',color:'#18733c'},
    ];
    series.forEach(item=>{
      const points=plot.values.map((value,index)=>[xs[index],y(value[item.key])]);
      svg.append(el('path',{d:path(points),fill:'none',stroke:item.color,'stroke-width':4}));
      points.forEach(([xpos,ypos],index)=>{
        svg.append(el('circle',{cx:xpos,cy:ypos,r:5,fill:item.color,stroke:'#fff','stroke-width':2}));
        svg.append(el('text',{x:xpos,y:ypos+(item.key==='attack'?-10:18),'text-anchor':'middle',class:'value-label'},pct(plot.values[index][item.key])));
      });
    });
    document.getElementById('noise-plots').append(section);
  });

  data.snapshots.forEach(plot => {
    const {section, svg} = card(plot.dataset, `${plot.subtitle} · ${plot.evidence}`);
    const left=58, right=495, top=20, bottom=270;
    const y=axes(svg,{left,right,top,bottom,min:0,max:1,ticks:[0,.25,.5,.75,1]});
    const centers=[180,385];
    const groupNames=['Attack AUC ↓','Model accuracy ↑'];
    const keys=['attack','accuracy'];
    const barWidth=33, gap=7;
    svg.append(el('line',{x1:85,y1:y(.5),x2:275,y2:y(.5),class:'chance'}));
    svg.append(el('text',{x:278,y:y(.5)+4,class:'tick'},'random'));
    centers.forEach((center,groupIndex)=>{
      const start=center-(barWidth*3+gap*2)/2;
      plot.values.forEach((value,index)=>{
        const xpos=start+index*(barWidth+gap), valueY=y(value[keys[groupIndex]]);
        svg.append(el('rect',{x:xpos,y:valueY,width:barWidth,height:bottom-valueY,fill:colors[value.privacy]}));
        svg.append(el('text',{x:xpos+barWidth/2,y:valueY-6,'text-anchor':'middle',class:'value-label'},pct(value[keys[groupIndex]])));
      });
      svg.append(el('text',{x:center,y:bottom+25,'text-anchor':'middle',class:'axis-label'},groupNames[groupIndex]));
    });
    document.getElementById('dataset-plots').append(section);
  });
})();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    build()
