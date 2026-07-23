#!/usr/bin/env python3
import json,statistics
from collections import defaultdict
from pathlib import Path

def name(row):
 root,clients,partition,privacy,agg,seed,noise,family=row
 return f'clients-{clients}__{family}__{partition}__{privacy}__{agg}__noise-{noise}__seed-{seed}'
tasks=[x.rstrip().split('\t') for x in Path('results-reproduce-new-matrices-tasks.tsv').open() if x.strip()]
records=[]; missing=[]
for t in tasks:
 root,clients,partition,privacy,agg,seed,noise,family=t; n=name(t); base=Path(root)/'runs'; rp=base/f'{n}.json'; ep=base/f'{n}.evaluation.json'; pp=base/f'{n}.predictions.npz'
 try:
  r=json.loads(rp.read_text()); e=json.loads(ep.read_text()); assert int(r['metadata']['num_clients'])==int(clients) and len(e['clients'])==int(clients) and pp.stat().st_size>0
  s=e['server_final_test']; records.append({'name':n,'clients':int(clients),'partition':partition,'privacy':privacy,'aggregation':agg,'seed':int(seed),'noise':float(noise),'family':family,'accuracy':s['accuracy'],'macro_f1':s['averages']['macro']['f1'],'macro_precision':s['averages']['macro']['precision'],'macro_auc':s['roc_ovr']['macro_auc']})
 except Exception as exc: missing.append({'name':n,'error':repr(exc)})
summary={'expected':len(tasks),'complete':len(records),'missing':missing,'records':records}; Path('results-reproduce-new-matrices-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
with Path('results-reproduce-new-matrices.md').open('a') as f:
 f.write(f'\n## Final status\n\n- Expected: {len(tasks)}\n- Complete: {len(records)}\n- Missing: {len(missing)}\n')
 if missing:
  for x in missing: f.write(f"- `{x['name']}` — {x['error']}\n")
 def ms(rows,key):
  vals=[float(r[key]) for r in rows if r[key] is not None]; return f'{statistics.mean(vals):.6f} ± {statistics.stdev(vals):.6f}' if len(vals)>1 else (f'{vals[0]:.6f}' if vals else 'NA')
 groups=defaultdict(list)
 for r in records:
  if r['family']=='main' and r['partition']=='homogeneous': groups[(r['clients'],r['privacy'],r['aggregation'])].append(r)
 f.write('\n## Homogeneous main five-seed summary\n\n| Clients | Privacy | Aggregator | Accuracy mean ± SD | Macro F1 mean ± SD | Macro precision mean ± SD | Macro AUC mean ± SD | n |\n|---:|---|---|---:|---:|---:|---:|---:|\n')
 for k,rows in sorted(groups.items()): f.write(f'| {k[0]} | {k[1]} | {k[2]} | {ms(rows,"accuracy")} | {ms(rows,"macro_f1")} | {ms(rows,"macro_precision")} | {ms(rows,"macro_auc")} | {len(rows)} |\n')
 f.write('\n## Fixed-seed non-IID and Appendix-B results\n\n| Clients | Family | Partition | Noise | Privacy | Aggregator | Accuracy | Macro F1 | Macro precision | Macro AUC |\n|---:|---|---|---:|---|---|---:|---:|---:|---:|\n')
 for r in sorted((x for x in records if not (x['family']=='main' and x['partition']=='homogeneous')),key=lambda x:(x['clients'],x['family'],x['privacy'],x['aggregation'])): f.write(f"| {r['clients']} | {r['family']} | {r['partition']} | {r['noise']} | {r['privacy']} | {r['aggregation']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['macro_precision']:.6f} | {r['macro_auc']:.6f} |\n")
print(json.dumps({'expected':len(tasks),'complete':len(records),'missing':len(missing)})); raise SystemExit(0 if not missing else 1)
