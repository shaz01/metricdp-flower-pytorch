"""Post-hoc matched-public-offset attribution on already evaluated images."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from research.calculations.class_conditional_probe import weighted_scores, transform, objective, MODES
from research.calculations.public_residual_probe import digest, public_geometry
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.client_geometry_audit import features
from research.calculations.private_descriptor_probe import ROOT

PROTOCOL=Path('research/proposals/2026-10-08_class_conditional_attribution_protocol.md')


def main():
    epath=ROOT/'class_conditional_evaluation.json'; ebankpath=epath.with_suffix('.npz'); dpath=ROOT/'class_conditional_development.json'; dbankpath=dpath.with_suffix('.npz')
    e=json.loads(epath.read_text()); bank=np.load(ebankpath); dbank=np.load(dbankpath)
    source,*_=prepare(False); data=Dataset.from_file(str(source)); labels=np.array(data['label']); ids=bank['utility_original_indices']; x=features(data,ids); y=labels[ids]; b=dbank['reference']; h=dbank['public_hessian']
    groups={}
    for row in e['rows']:
        if row['target']<4:
            token=tuple(row[k] for k in ('seed','risk','scenario','reference_source','arm'))
            groups.setdefault(token,[]).append(row)
    rows=[]
    for (seed,risk,scenario,route,arm),cells in groups.items():
        prefix=f'seed{seed}_{scenario}_{route}'; name=f'{prefix}_q{int(risk*100)}_{arm}'; config=cells[0]['configuration']
        offset=bank[name+'_offset']; ce,acc=weighted_scores(transform(x,scenario in ('mask','both')),y,(b-config['eta']*offset)[None],objective(scenario))
        strata=[bank[r['archive_key']+f'_world{w}_utility_ce'] for r in cells for w in (0,1)]
        mean=float(np.mean([a.mean() for a in strata])); no=float(np.mean([r['noiseless'][str(w)]['ce'] for r in cells for w in (0,1)]))
        se=float(np.sqrt(sum(a.var(ddof=1)/len(a) for a in strata))/8); gain=float(ce[0]-mean)
        g=public_geometry(h,config['dimension'],config['metric']); raw=dbank[prefix+'_raw_query'].reshape(8,51)@g['encoder']; center=dbank[prefix+'_class_center_query'].reshape(8,51)@g['encoder']
        rows.append({'seed':seed,'risk':risk,'scenario':scenario,'reference_source':route,'arm':arm,'configuration':config,
            'matched_public_offset_ce':float(ce[0]),'matched_public_offset_accuracy':float(acc[0]),'protected_ce':mean,
            'protected_gain_over_matched_offset':gain,'gain_se':se,'gain_ci95':[gain-1.96*se,gain+1.96*se],
            'noiseless_ce':no,'noiseless_gain_over_matched_offset':float(ce[0]-no),'encoded_raw_client_norms':np.linalg.norm(raw,axis=1).tolist(),
            'encoded_class_center_client_norms':np.linalg.norm(center,axis=1).tolist(),'clip_factors':cells[0]['clip_factors']})
    assert len(rows)==147
    (ROOT/'class_conditional_attribution.json').write_text(json.dumps({'kind':'Post-hoc matched public-offset diagnostic, same320images; no new confirmation/tuning.',
        'hashes':{str(p):digest(p) for p in (PROTOCOL,Path('research/calculations/class_conditional_attribution.py'),epath,ebankpath,dpath,dbankpath,Path('research/calculations/class_conditional_probe.py'))},'source_hash':digest(source),'utility_count':len(ids),'rows':rows},indent=2,allow_nan=False)+'\n')
    for r in rows:
        if r['risk']==.55 and r['arm']=='class_center': print(r['seed'],r['scenario'],r['reference_source'],'private gain',r['protected_gain_over_matched_offset'],'noiseless',r['noiseless_gain_over_matched_offset'],flush=True)


if __name__=='__main__': main()
