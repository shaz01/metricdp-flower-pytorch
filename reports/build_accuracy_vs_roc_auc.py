"""Build the standalone interactive accuracy-versus-ROC-AUC report.

Run from the repository root:
    uv run python reports/build_accuracy_vs_roc_auc.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "accuracy_vs_roc_auc.html"
RESULT_SETS = (
    ("cifar-remove", "CIFAR-10 removal", "results/cia/cifar10_remove", True),
    ("cifar-homogeneous", "CIFAR-10 homogeneous scaling", "results/client_scaling/cifar10_homogeneous", True),
    ("alzheimer", "Alzheimer", "results/planned_runs/alzheimer", True),
    ("fashion", "Fashion-MNIST", "results/planned_runs/fashion", True),
    ("cifar-full", "CIFAR full replacement", "results/planned_runs/cifar/full", True),
    ("cifar-validation-48", "CIFAR 48-client validation", "results/planned_runs/cifar/validation_48", True),
    ("cifar-max-records", "CIFAR max records", "results/planned_runs/cifar/max_records", True),
    ("cifar-noise-sweep", "CIFAR noise sweep", "results/planned_runs/cifar/noise_sweep", True),
    # Keep partial sweep artifacts out until the result set is declared ready.
    # Change the final value to True and rerun this generator at that point.
    ("cifar-remove-ratio-sweep", "CIFAR-10 removal ratio sweep", "results/cia/cifar10_remove_ratio_sweep", False),
)
PRIVACY_LABELS = {
    "vanilla": "Vanilla",
    "global-dp": "Global-DP",
    "metric-privacy": "Metric privacy",
}


def path_ratio(path: Path) -> float | None:
    for part in path.parts:
        if part.startswith("ratio-"):
            try:
                return float(part.removeprefix("ratio-").replace("p", "."))
            except ValueError:
                return None
    return None


def parse_run_name(name: str) -> dict[str, object]:
    parts = name.split("__")
    values: dict[str, object] = {"experiment": parts[0]}
    for part in parts[1:]:
        if part in PRIVACY_LABELS:
            values["privacy"] = part
        elif part.startswith("clients-"):
            values["clients"] = int(part.removeprefix("clients-"))
        elif part.startswith("seed-"):
            values["seed"] = int(part.removeprefix("seed-"))
        elif part.startswith("nm"):
            try:
                values["noise_multiplier"] = float(part.removeprefix("nm").replace("p", "."))
            except ValueError:
                pass
    experiment = str(values["experiment"])
    for adjacency in ("in-remove", "out-remove", "in-replace", "out-replace"):
        if adjacency in experiment:
            values["adjacency"] = adjacency
            break
    return values


def load_points() -> tuple[list[dict], list[dict]]:
    points: list[dict] = []
    sets: list[dict] = []
    for set_id, label, relative, include in RESULT_SETS:
        directory = ROOT / relative
        discovered = sorted(directory.rglob("*.evaluation.json")) if directory.exists() else []
        files = discovered if include else []
        sets.append({
            "id": set_id,
            "label": label,
            "path": relative,
            "count": len(files),
            "discovered": len(discovered),
            "pending": not include,
        })
        for path in files:
            payload = json.loads(path.read_text())
            run_name = str(payload["run_name"])
            meta = parse_run_name(run_name)
            test = payload["server_final_test"]
            roc = test.get("roc_ovr", {})
            if "macro_auc" not in roc or "accuracy" not in test or "privacy" not in meta:
                continue
            ratio = path_ratio(path)
            adjacency = meta.get("adjacency")
            direction = (
                "in" if isinstance(adjacency, str) and adjacency.startswith("in-")
                else "out" if isinstance(adjacency, str) and adjacency.startswith("out-")
                else "homogeneous" if "__homogeneous__" in run_name
                else "other"
            )
            point = {
                "id": len(points),
                "set": set_id,
                "setLabel": label,
                "path": str(path.relative_to(ROOT)),
                "run": run_name,
                "privacy": meta["privacy"],
                "accuracy": float(test["accuracy"]),
                "auc": float(roc["macro_auc"]),
                "experiment": meta.get("experiment", ""),
                "adjacency": meta.get("adjacency", "—"),
                "direction": direction,
                "clients": meta.get("clients"),
                "seed": meta.get("seed"),
                "noise": meta.get("noise_multiplier"),
                "ratio": ratio,
            }
            points.append(point)

    return points, sets


def build() -> None:
    points, sets = load_points()
    active_sets = [item for item in sets if item["count"]]
    pending_sets = [item for item in sets if not item["count"]]
    controls = "".join(
        f'<label class="toggle"><input type="checkbox" data-set="{html.escape(item["id"])}" checked> '
        f'<span>{html.escape(item["label"])}</span><small>{item["count"]}</small></label>'
        for item in active_sets
    )
    pending = "".join(
        f'<label class="toggle pending"><input type="checkbox" disabled> '
        f'<span>{html.escape(item["label"])}</span><small>'
        f'{("partial data excluded" if item["pending"] and item["discovered"] else "awaiting data")}</small></label>'
        for item in pending_sets
    )
    data_json = json.dumps({"points": points, "sets": sets}, separators=(",", ":")).replace("</", "<\\/")
    document = TEMPLATE.replace("__CONTROLS__", controls).replace("__PENDING__", pending).replace("__DATA__", data_json)
    OUTPUT.write_text(document)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(points)} points from {len(active_sets)} result sets.")


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Accuracy vs. ROC AUC</title>
  <style>
    :root { color-scheme: light; font-family: Arial, Helvetica, sans-serif; }
    * { box-sizing: border-box; }
    body { max-width: 1180px; margin: 0 auto; padding: 28px; color: #111; background: #fff; }
    header { border-bottom: 1px solid #bbb; margin-bottom: 28px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    h2 { margin-top: 32px; padding-bottom: 6px; border-bottom: 1px solid #ccc; font-size: 21px; }
    p { line-height: 1.45; }
    .meta, .note { color: #444; font-size: 14px; }
    .panel { border: 1px solid #ccc; padding: 14px; }
    .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 8px 14px; }
    .toggle { display: flex; align-items: center; gap: 7px; padding: 7px 9px; border: 1px solid #ddd; cursor: pointer; font-size: 13px; }
    .toggle span { flex: 1; }
    .toggle small { color: #666; font-variant-numeric: tabular-nums; }
    .toggle.pending { color: #777; background: #fafafa; cursor: not-allowed; }
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin: 12px 0 2px; font-size: 13px; }
    button { border: 1px solid #999; background: #fff; color: #111; padding: 6px 10px; cursor: pointer; }
    button:hover { background: #f3f3f3; }
    .filter-groups { display: grid; gap: 8px; width: 100%; }
    .filter-group { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 12px; margin: 0; padding: 7px 9px; border: 1px solid #ddd; }
    .filter-group legend { padding: 0 5px; font-weight: bold; }
    .filter-group label { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
    #chart { width: 100%; min-height: 590px; display: block; }
    .gridline { stroke: #e2e2e2; stroke-width: 1; }
    .axis { stroke: #777; stroke-width: 1.2; }
    .tick, .axis-label { fill: #333; font-size: 12px; }
    .axis-label { font-size: 14px; font-weight: bold; }
    .tradeoff { fill: none; stroke-width: 2; opacity: .58; }
    .set-mean { fill: #fff; stroke-width: 2; opacity: .85; }
    .point { stroke: #fff; stroke-width: 1.5; cursor: crosshair; }
    .point:hover { stroke: #111; stroke-width: 2.5; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 5px 0 12px; font-size: 13px; }
    .key { display: inline-flex; align-items: center; gap: 7px; }
    .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
    #summary { color: #444; font-size: 13px; margin-left: auto; }
    .stats { margin-top: 20px; }
    .stat-cards { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin: 12px 0; }
    .stat-card { border: 1px solid #ccc; padding: 12px; }
    .stat-card strong { display: block; font-size: 24px; font-variant-numeric: tabular-nums; }
    .stat-card span { color: #555; font-size: 12px; }
    .stat-card.metric { border-top: 4px solid #b2182b; }
    .stat-card.global { border-top: 4px solid #2166ac; }
    .stat-card.mixed { border-top: 4px solid #888; }
    .stat-card.total { border-top: 4px solid #111; }
    .stats table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
    .stats th, .stats td { border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; }
    .stats th:first-child, .stats td:first-child { text-align: left; }
    .stats thead th { border-bottom: 2px solid #888; }
    .tradeoff-summary { display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 10px; margin: 14px 0; }
    .tradeoff-box { border: 1px solid #ddd; padding: 10px; font-size: 13px; }
    .tradeoff-box strong { display: block; margin-bottom: 4px; }
    .tradeoff-box span { color: #555; }
    details { margin-top: 12px; }
    summary { cursor: pointer; font-weight: bold; }
    .table-scroll { max-height: 360px; margin-top: 8px; overflow: auto; border: 1px solid #ddd; }
    .table-scroll thead { position: sticky; top: 0; background: #fff; }
    .delta-good { color: #18733c; font-weight: bold; }
    .delta-bad { color: #a3212b; font-weight: bold; }
    .empty-stat { color: #666; padding: 12px 0; }
    #tooltip { position: fixed; z-index: 10; display: none; max-width: 390px; padding: 9px 11px; color: #111; background: rgba(255,255,255,.98); border: 1px solid #777; box-shadow: 0 2px 8px #0002; font-size: 12px; line-height: 1.45; pointer-events: none; }
    #tooltip strong { font-size: 13px; }
    #tooltip .path { color: #666; overflow-wrap: anywhere; }
    footer { margin-top: 36px; padding-top: 10px; border-top: 1px solid #bbb; font-size: 12px; color: #555; }
    @media (max-width: 700px) { body { padding: 16px; } #chart { min-height: 470px; } #summary { width: 100%; margin-left: 0; } .stat-cards, .tradeoff-summary { grid-template-columns: repeat(2, 1fr); } }
    @media print { body { max-width: none; padding: 12mm; } .controls, .toolbar button { display: none; } }
  </style>
</head>
<body>
<header>
  <h1>Accuracy vs. ROC AUC</h1>
  <p class="meta">Final server-test accuracy against one-vs-rest macro ROC AUC. Each dot is one evaluation artifact. Lines connect the per-result-set means of the same privacy method across result sets.</p>
</header>

<section>
  <h2>Result sets</h2>
  <div class="panel controls">__CONTROLS____PENDING__</div>
  <p class="note"><strong>Data update:</strong> partial artifacts under <code>results/cia/cifar10_remove_ratio_sweep/</code> are intentionally excluded. Once that result set is fully ready, enable its entry in <code>reports/build_accuracy_vs_roc_auc.py</code> and rerun the generator.</p>
</section>

<section>
  <h2>Utility map</h2>
  <div class="toolbar">
    <button id="all">Enable all</button><button id="none">Disable all</button><button id="clear-filters">Clear filters</button>
    <label><input id="lines" type="checkbox" checked> same-privacy set lines</label>
    <span id="summary"></span>
    <div class="filter-groups">
      <fieldset id="direction-filter" class="filter-group"><legend>Adjacency / partition</legend></fieldset>
      <fieldset id="ratio-filter" class="filter-group"><legend>Noise ratio</legend></fieldset>
      <fieldset id="noise-filter" class="filter-group"><legend>Noise multiplier</legend></fieldset>
    </div>
  </div>
  <div class="legend"><span class="key"><i class="dot" style="background:#111"></i>Vanilla</span><span class="key"><i class="dot" style="background:#2166ac"></i>Global-DP</span><span class="key"><i class="dot" style="background:#b2182b"></i>Metric privacy</span></div>
  <div class="panel"><svg id="chart" role="img" aria-label="Scatter plot of accuracy versus macro ROC AUC"></svg></div>
  <div class="panel stats">
    <h3>Metric-DP vs. Global-DP matched-point dominance</h3>
    <p class="note">Interactive summary for the currently visible points. A method dominates when its accuracy is at least as high and its macro ROC AUC is at least as low, with one strictly better. Runs are matched within result set by adjacency, clients, seed, and noise ratio.</p>
    <div id="dominance-stats"></div>
  </div>
</section>
<div id="tooltip"></div>
<footer>Standalone interactive report generated from committed <code>*.evaluation.json</code> artifacts. Axes rescale when result sets are toggled.</footer>

<script id="report-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const svg = document.getElementById('chart'), tooltip = document.getElementById('tooltip');
  const directionFilter = document.getElementById('direction-filter');
  const ratioFilter = document.getElementById('ratio-filter'), noiseFilter = document.getElementById('noise-filter');
  const colors = {'vanilla':'#111','global-dp':'#2166ac','metric-privacy':'#b2182b'};
  const labels = {'vanilla':'Vanilla','global-dp':'Global-DP','metric-privacy':'Metric privacy'};
  const enabled = new Set(data.sets.filter(s => s.count).map(s => s.id));
  let fixedDomain = null;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (name, attrs={}, text='') => { const n=document.createElementNS(NS,name); Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v)); if(text)n.textContent=text; return n; };
  const fmt = v => Number(v).toFixed(4);
  const filterValue = v => v == null ? 'na' : String(v);
  function populateCheckboxes(container, key, labelFor, sortValues) {
    const counts = new Map();
    data.points.forEach(p => counts.set(filterValue(p[key]), (counts.get(filterValue(p[key])) || 0) + 1));
    container.insertAdjacentHTML('beforeend', `<label><input type="checkbox" data-all checked> All (${data.points.length})</label>`);
    [...counts].sort((a,b) => sortValues(a[0],b[0])).forEach(([value,count]) => {
      const label=document.createElement('label'), input=document.createElement('input');
      input.type='checkbox'; input.value=value; input.checked=true;
      label.append(input, document.createTextNode(` ${labelFor(value)} (${count})`)); container.append(label);
    });
  }
  const numericSort=(a,b) => a === 'na' ? -1 : b === 'na' ? 1 : Number(a)-Number(b);
  populateCheckboxes(directionFilter, 'direction', v => ({in:'IN',out:'OUT',homogeneous:'Homogeneous',other:'Other'}[v] || v), (a,b) => ['in','out','homogeneous','other'].indexOf(a)-['in','out','homogeneous','other'].indexOf(b));
  populateCheckboxes(ratioFilter, 'ratio', v => v === 'na' ? 'Not specified' : v, numericSort);
  populateCheckboxes(noiseFilter, 'noise', v => v === 'na' ? 'Not specified' : v, numericSort);
  const selected = container => new Set([...container.querySelectorAll('input:not([data-all]):checked')].map(input => input.value));
  function domains(points) {
    if (fixedDomain) return fixedDomain;
    if (!points.length) return {x:[0,1],y:[0,1]};
    const extent = key => { const vs=points.map(p=>p[key]), lo=Math.min(...vs), hi=Math.max(...vs), pad=Math.max((hi-lo)*.09,.006); return [Math.max(0,lo-pad),Math.min(1,hi+pad)]; };
    return {x:extent('accuracy'),y:extent('auc')};
  }
  function renderDominance(points) {
    const groups=new Map();
    points.filter(p=>p.privacy==='global-dp'||p.privacy==='metric-privacy').forEach(p=>{
      const key=[p.set,p.adjacency,p.clients,p.seed,p.ratio].join('|');
      if(!groups.has(key)) groups.set(key,{}); groups.get(key)[p.privacy]=p;
    });
    const rows=[];
    groups.forEach(pair=>{
      const global=pair['global-dp'], metric=pair['metric-privacy'];
      if(!global||!metric) return;
      const da=metric.accuracy-global.accuracy, du=metric.auc-global.auc;
      const outcome=da>=0&&du<=0&&(da>0||du<0) ? 'metric' : da<=0&&du>=0&&(da<0||du>0) ? 'global' : 'mixed';
      rows.push({
        set:metric.setLabel, outcome, da, du,
        adjacency:metric.adjacency, clients:metric.clients, seed:metric.seed, ratio:metric.ratio
      });
    });
    const counts={metric:0,global:0,mixed:0}; rows.forEach(r=>counts[r.outcome]++);
    const target=document.getElementById('dominance-stats');
    if(!rows.length) { target.innerHTML='<div class="empty-stat">No visible matched Global-DP/Metric-DP pairs. Broaden the active filters to compare both methods.</div>'; return; }
    const pct=n=>`${(100*n/rows.length).toFixed(1)}%`;
    const bySet=new Map(); rows.forEach(r=>{if(!bySet.has(r.set))bySet.set(r.set,{metric:0,global:0,mixed:0});bySet.get(r.set)[r.outcome]++;});
    const table=[...bySet].map(([set,c])=>`<tr><td>${set}</td><td>${c.metric}</td><td>${c.global}</td><td>${c.mixed}</td><td>${c.metric+c.global+c.mixed}</td></tr>`).join('');
    const mixed=rows.filter(r=>r.outcome==='mixed');
    const metricUtility=mixed.filter(r=>r.da>0&&r.du>0);
    const globalUtility=mixed.filter(r=>r.da<0&&r.du<0);
    const mean=(items,key)=>items.length ? items.reduce((sum,r)=>sum+r[key],0)/items.length : 0;
    const signed=(value,digits=4)=>`${value>=0?'+':''}${value.toFixed(digits)}`;
    const mixedRows=mixed.map(r=>`<tr><td>${r.set}</td><td>${r.adjacency}</td><td>${r.clients??'—'}</td><td>${r.seed??'—'}</td><td>${r.ratio??'—'}</td><td class="${r.da>=0?'delta-good':'delta-bad'}">${signed(100*r.da,2)} pp</td><td class="${r.du<=0?'delta-good':'delta-bad'}">${signed(r.du)}</td></tr>`).join('');
    target.innerHTML=`<div class="stat-cards">
      <div class="stat-card metric"><strong>${counts.metric}</strong><span>Metric-DP dominates · ${pct(counts.metric)}</span></div>
      <div class="stat-card global"><strong>${counts.global}</strong><span>Global-DP dominates · ${pct(counts.global)}</span></div>
      <div class="stat-card mixed"><strong>${counts.mixed}</strong><span>Mixed tradeoff or tie · ${pct(counts.mixed)}</span></div>
      <div class="stat-card total"><strong>${rows.length}</strong><span>Matched comparisons</span></div>
    </div><table><thead><tr><th>Result set</th><th>Metric-DP</th><th>Global-DP</th><th>Mixed/tie</th><th>Total</th></tr></thead><tbody>${table}</tbody></table>
    <div class="tradeoff-summary">
      <div class="tradeoff-box"><strong>Metric-DP gains accuracy, loses ROC AUC: ${metricUtility.length}</strong><span>Mean Metric − Global: ${signed(100*mean(metricUtility,'da'),2)} accuracy pp, ${signed(mean(metricUtility,'du'))} AUC</span></div>
      <div class="tradeoff-box"><strong>Global-DP gains accuracy, loses ROC AUC: ${globalUtility.length}</strong><span>Mean Metric − Global: ${signed(100*mean(globalUtility,'da'),2)} accuracy pp, ${signed(mean(globalUtility,'du'))} AUC</span></div>
    </div>
    ${mixed.length ? `<details><summary>Show all ${mixed.length} mixed tradeoffs</summary><p class="note">Deltas are Metric-DP minus Global-DP. Green is favorable to Metric-DP: positive accuracy or negative ROC AUC.</p><div class="table-scroll"><table><thead><tr><th>Result set</th><th>Adjacency</th><th>Clients</th><th>Seed</th><th>Ratio</th><th>Δ accuracy</th><th>Δ ROC AUC</th></tr></thead><tbody>${mixedRows}</tbody></table></div></details>` : ''}`;
  }
  function draw() {
    const directions=selected(directionFilter), ratios=selected(ratioFilter), noises=selected(noiseFilter);
    const points=data.points.filter(p => enabled.has(p.set)
      && directions.has(filterValue(p.direction))
      && ratios.has(filterValue(p.ratio))
      && noises.has(filterValue(p.noise)));
    svg.replaceChildren();
    const box=svg.getBoundingClientRect(), W=Math.max(650,Math.round(box.width)||1000), H=Math.max(470,Math.min(660,Math.round(W*.62)));
    svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
    const m={l:72,r:24,t:18,b:64}, pw=W-m.l-m.r, ph=H-m.t-m.b, d=domains(points);
    const X=v=>m.l+(v-d.x[0])/(d.x[1]-d.x[0]||1)*pw, Y=v=>m.t+ph-(v-d.y[0])/(d.y[1]-d.y[0]||1)*ph;
    for(let i=0;i<=5;i++) {
      const xv=d.x[0]+i*(d.x[1]-d.x[0])/5, yv=d.y[0]+i*(d.y[1]-d.y[0])/5;
      svg.append(el('line',{x1:X(xv),y1:m.t,x2:X(xv),y2:m.t+ph,class:'gridline'}));
      svg.append(el('line',{x1:m.l,y1:Y(yv),x2:m.l+pw,y2:Y(yv),class:'gridline'}));
      svg.append(el('text',{x:X(xv),y:m.t+ph+22,'text-anchor':'middle',class:'tick'},xv.toFixed(3)));
      svg.append(el('text',{x:m.l-10,y:Y(yv)+4,'text-anchor':'end',class:'tick'},yv.toFixed(3)));
    }
    svg.append(el('line',{x1:m.l,y1:m.t+ph,x2:m.l+pw,y2:m.t+ph,class:'axis'}));
    svg.append(el('line',{x1:m.l,y1:m.t,x2:m.l,y2:m.t+ph,class:'axis'}));
    svg.append(el('text',{x:m.l+pw/2,y:H-12,'text-anchor':'middle',class:'axis-label'},'Final server-test accuracy'));
    const yl=el('text',{x:17,y:m.t+ph/2,'text-anchor':'middle',class:'axis-label',transform:`rotate(-90 17 ${m.t+ph/2})`},'Macro ROC AUC (one-vs-rest)'); svg.append(yl);
    if(document.getElementById('lines').checked) {
      ['vanilla','global-dp','metric-privacy'].forEach(privacy => {
        const means = data.sets.map(set => {
          const rows = points.filter(p => p.set === set.id && p.privacy === privacy);
          if (!rows.length) return null;
          return {
            set: set.label,
            accuracy: rows.reduce((sum,p)=>sum+p.accuracy,0)/rows.length,
            auc: rows.reduce((sum,p)=>sum+p.auc,0)/rows.length,
            count: rows.length
          };
        }).filter(Boolean);
        if(means.length > 1) {
          svg.append(el('polyline',{points:means.map(p=>`${X(p.accuracy)},${Y(p.auc)}`).join(' '),class:'tradeoff',stroke:colors[privacy]}));
          means.forEach(p => svg.append(el('circle',{cx:X(p.accuracy),cy:Y(p.auc),r:4,class:'set-mean',stroke:colors[privacy]})));
        }
      });
    }
    points.forEach(p=>{
      const c=el('circle',{cx:X(p.accuracy),cy:Y(p.auc),r:5.5,fill:colors[p.privacy],class:'point',tabindex:'0'});
      const show=e=>{tooltip.innerHTML=`<strong>${p.setLabel} · ${labels[p.privacy]}</strong><br>Accuracy: ${fmt(p.accuracy)} · Macro AUC: ${fmt(p.auc)}<br>Adjacency: ${p.adjacency} · Clients: ${p.clients ?? '—'} · Seed: ${p.seed ?? '—'}<br>Noise multiplier: ${p.noise ?? '—'} · Ratio: ${p.ratio ?? '—'}<br><span class="path">${p.path}</span>`; tooltip.style.display='block'; move(e);};
      const move=e=>{const x=(e.clientX??W/2)+14,y=(e.clientY??H/2)+14; tooltip.style.left=Math.min(x,innerWidth-tooltip.offsetWidth-8)+'px';tooltip.style.top=Math.min(y,innerHeight-tooltip.offsetHeight-8)+'px';};
      c.addEventListener('mouseenter',show); c.addEventListener('mousemove',move); c.addEventListener('mouseleave',()=>tooltip.style.display='none'); c.addEventListener('focus',show); c.addEventListener('blur',()=>tooltip.style.display='none'); svg.append(c);
    });
    const activeSetCount = new Set(points.map(p => p.set)).size;
    document.getElementById('summary').textContent=`${points.length} points · ${activeSetCount} visible result sets`;
    renderDominance(points);
  }
  document.querySelectorAll('[data-set]').forEach(input=>input.addEventListener('change',()=>{input.checked?enabled.add(input.dataset.set):enabled.delete(input.dataset.set);fixedDomain=null;draw();}));
  document.getElementById('lines').addEventListener('change',draw);
  [directionFilter, ratioFilter, noiseFilter].forEach(group => group.addEventListener('change', event => {
    const boxes=[...group.querySelectorAll('input:not([data-all])')], all=group.querySelector('input[data-all]');
    if(event.target === all) boxes.forEach(box => box.checked=all.checked);
    else all.checked=boxes.every(box => box.checked);
    fixedDomain=null; draw();
  }));
  document.getElementById('all').onclick=()=>{document.querySelectorAll('[data-set]').forEach(i=>{i.checked=true;enabled.add(i.dataset.set)});fixedDomain=null;draw();};
  document.getElementById('none').onclick=()=>{document.querySelectorAll('[data-set]').forEach(i=>{i.checked=false;enabled.delete(i.dataset.set)});fixedDomain=null;draw();};
  document.getElementById('clear-filters').onclick=()=>{[directionFilter,ratioFilter,noiseFilter].forEach(group=>group.querySelectorAll('input').forEach(input=>input.checked=true));fixedDomain=null;draw();};
  new ResizeObserver(draw).observe(svg.parentElement); draw();
})();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    build()
