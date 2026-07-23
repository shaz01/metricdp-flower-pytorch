#!/usr/bin/env python3
from __future__ import annotations
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('results-reproduce-paper')

def name_from(row):
 p,privacy,agg,seed,noise,family=row
 return f'{family}__{p}__{privacy}__{agg}__noise-{noise}__seed-{seed}'

tasks=[line.rstrip('\n').split('\t') for line in (ROOT/'tasks.tsv').open() if line.strip()]
records=[]; missing=[]
for task in tasks:
 name=name_from(task); rp=ROOT/'runs'/f'{name}.json'; ep=ROOT/'evaluations'/f'{name}.evaluation.json'; pp=ROOT/'evaluations'/f'{name}.predictions.npz'
 try:
  r=json.loads(rp.read_text()); e=json.loads(ep.read_text()); h=r['server_evaluate_metrics']; fr=max(int(k) for k in h if int(k)>0); m=h[str(fr)]; s=e['server_final_test']; c=e['clients_combined_test']
  assert abs(float(m['accuracy'])-float(s['accuracy']))<1e-12 and pp.stat().st_size>0
  records.append({'name':name,'partition':task[0],'privacy':task[1],'aggregation':task[2],'seed':int(task[3]),'noise':float(task[4]),'family':task[5],'accuracy':float(s['accuracy']),'loss':float(s['log_loss']),'macro_f1':float(s['averages']['macro']['f1']),'weighted_f1':float(s['averages']['weighted']['f1']),'macro_precision':float(s['averages']['macro']['precision']),'weighted_precision':float(s['averages']['weighted']['precision']),'macro_auc':s['roc_ovr']['macro_auc'],'clients_accuracy':float(c['accuracy'])})
 except Exception as exc:
  missing.append({'name':name,'error':repr(exc)})
summary={'expected':len(tasks),'complete':len(records),'missing':missing,'records':records}
(ROOT/'paper_reproduction_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
with (ROOT/'paper_reproduction_results.md').open('a') as f:
 f.write('\n## Final status\n\n')
 f.write(f'- Expected: {len(tasks)}\n- Complete and validated: {len(records)}\n- Missing/failed: {len(missing)}\n- Retained final `.pt` files (failure recovery only): {len(list((ROOT/"runs").glob("*.pt")))}\n')
 if missing:
  f.write('\n### Missing or failed\n\n')
  for item in missing: f.write(f"- `{item['name']}` — {item['error']}\n")
 groups=defaultdict(list)
 for r in records:
  if r['family']=='main' and r['partition']=='homogeneous' and r['noise']==0.01: groups[(r['privacy'],r['aggregation'])].append(r)
 f.write('\n## Homogeneous five-seed server final-test summary\n\n| Privacy | Aggregator | Accuracy mean ± SD | Macro F1 mean ± SD | Macro precision mean ± SD | Macro AUC mean ± SD | n |\n|---|---|---:|---:|---:|---:|---:|\n')
 def ms(rows,key):
  vals=[float(x[key]) for x in rows if x[key] is not None]; return f'{statistics.mean(vals):.6f} ± {statistics.stdev(vals):.6f}' if len(vals)>1 else (f'{vals[0]:.6f} ± 0.000000' if vals else 'NA')
 for key,rows in sorted(groups.items()): f.write(f'| {key[0]} | {key[1]} | {ms(rows,"accuracy")} | {ms(rows,"macro_f1")} | {ms(rows,"macro_precision")} | {ms(rows,"macro_auc")} | {len(rows)} |\n')
 f.write('\n## Non-IID fixed-seed server final-test results-reproduce-paper\n\n| Privacy | Aggregator | Accuracy | Macro F1 | Macro precision | Macro AUC |\n|---|---|---:|---:|---:|---:|\n')
 for r in sorted((x for x in records if x['family']=='main' and x['partition']=='non-iid'),key=lambda x:(x['privacy'],x['aggregation'])): f.write(f"| {r['privacy']} | {r['aggregation']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['macro_precision']:.6f} | {r['macro_auc']:.6f} |\n")
 f.write('\n## Appendix B server final-test results-reproduce-paper\n\n| Noise | Privacy | Aggregator | Accuracy | Macro F1 | Macro precision | Macro AUC |\n|---:|---|---|---:|---:|---:|---:|\n')
 for r in sorted((x for x in records if x['family']=='appendix-b'),key=lambda x:(x['noise'],x['privacy'],x['aggregation'])): f.write(f"| {r['noise']} | {r['privacy']} | {r['aggregation']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['macro_precision']:.6f} | {r['macro_auc']:.6f} |\n")
print(json.dumps({'expected':len(tasks),'complete':len(records),'missing':len(missing)}))
raise SystemExit(0 if not missing else 1)
