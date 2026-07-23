#!/usr/bin/env python3
import glob,json,statistics
from pathlib import Path
root=Path('results-reproduce-fedyogi-rerun-4clients'); rows=[]
for p in sorted((root/'runs').glob('*.json')):
 if p.name.endswith('.evaluation.json'): continue
 e=json.loads(p.with_name(p.stem+'.evaluation.json').read_text()); r=json.loads(p.read_text()); m=r['metadata']; s=e['server_final_test']
 rows.append({'run_name':m['run_name'],'partition':m['partition_mode'],'privacy':m['privacy'],'aggregation':m['aggregation'],'seed':m['seed'],'noise':m['noise_multiplier'],'accuracy':s['accuracy'],'macro_f1':s['averages']['macro']['f1'],'macro_precision':s['averages']['macro']['precision'],'macro_auc':s['roc_ovr']['macro_auc']})
(root/'summary.json').write_text(json.dumps({'expected':20,'complete':len(rows),'records':rows},indent=2,allow_nan=False)+'\n')
lines=['# Four-client FedYogi rerun','','Twenty FedYogi cells from the paper 120-run matrix: 15 homogeneous main, 3 non-IID main, and 2 Appendix-B noise=0.003. FedOpt is excluded.','',f'Complete: **{len(rows)}/20**. Detailed metrics and raw predictions are stored in `runs/`.','', '## Homogeneous main five-seed summary','','| Privacy | Accuracy mean ± SD | Macro F1 mean ± SD | Macro precision mean ± SD | Macro AUC mean ± SD | n |','|---|---:|---:|---:|---:|---:|']
def ms(rs,k):
 v=[float(x[k]) for x in rs]; return f'{statistics.mean(v):.6f} ± {statistics.stdev(v):.6f}'
for privacy in ('vanilla','global-dp','metric-privacy'):
 rs=[x for x in rows if x['partition']=='homogeneous' and x['noise']==0.01 and x['privacy']==privacy]
 lines.append(f'| {privacy} | {ms(rs,"accuracy")} | {ms(rs,"macro_f1")} | {ms(rs,"macro_precision")} | {ms(rs,"macro_auc")} | {len(rs)} |')
lines += ['', '## Fixed-seed non-IID and Appendix-B results','', '| Partition | Noise | Privacy | Accuracy | Macro F1 | Macro precision | Macro AUC |','|---|---:|---|---:|---:|---:|---:|']
for x in rows:
 if not (x['partition']=='homogeneous' and x['noise']==0.01): lines.append(f"| {x['partition']} | {x['noise']} | {x['privacy']} | {x['accuracy']:.6f} | {x['macro_f1']:.6f} | {x['macro_precision']:.6f} | {x['macro_auc']:.6f} |")
(root/'README.md').write_text('\n'.join(lines)+'\n')
print(len(rows))
