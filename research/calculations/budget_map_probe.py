"""Frozen stacked constructor applied unchanged at public budgets 32/128/512 on the two fresh tasks (descriptive map, no tuning)."""
import json
from pathlib import Path
import numpy as np
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import scores
from research.calculations.client_geometry_audit import features
from research.calculations.public_residual_probe import digest
from research.calculations.limited_public_probe import mean_scores, RISKS
from research.calculations.stacked_constructor_probe import stacked_inputs, stacked_theta, load_task, TASKS, FREEZE as STACKED_FREEZE, DRAWS_CONFIRM
from research.calculations.frozen_constructor_probe import public_candidates

PROTOCOL=Path('research/proposals/2026-10-08_budget_map_protocol.md')
CODE=Path('research/calculations/budget_map_probe.py')
OUT=ROOT/'budget_map.json'
NPZ=ROOT/'budget_map.npz'
BUDGETS=(32,128,512)
SEED=20261010


def roles(raw,first):
    rng=np.random.default_rng(SEED); out={f'public{b}_{s}':[] for b in BUDGETS for s in range(3)}; out.update(selection=[],A=[],B=[],evaluation=[])
    for k in range(4):
        ids=np.where(raw==first+k)[0].copy(); rng.shuffle(ids); pos=0
        for b in BUDGETS:
            for s in range(3): out[f'public{b}_{s}'].append(ids[pos:pos+b//4]); pos+=b//4
        for n,m in (('selection',128),('A',512),('B',512),('evaluation',512)): out[n].append(ids[pos:pos+m]); pos+=m
    out={n:np.concatenate(v) for n,v in out.items()}; allids=np.concatenate(list(out.values())); assert len(set(allids.tolist()))==len(allids)
    return out


def main():
    fz=json.loads(STACKED_FREEZE.read_text())['frozen']; saved={}; tasks={}
    for name in TASKS:
        data,raw,first=load_task(name); r=roles(raw,first); lab=lambda ids:raw[ids]-first
        sel=(features(data,r['selection']),lab(r['selection'])); ex=features(data,r['evaluation']); ey=lab(r['evaluation']); coh={}
        counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
        for c in 'AB':
            cx=features(data,r[c]); cy=lab(r[c]); coh[c]=(cx,cy,partition_by_class_counts(cy,counts,seed=SEED)); assert all(len(p)==256 for p in coh[c][2])
        cells=[]
        for b in BUDGETS:
            for s in range(3):
                ids=r[f'public{b}_{s}']; px=features(data,ids); py=lab(ids); _b,_g,_h,family,control=public_candidates(px,py,sel); cce,cacc=mean_scores((ex,ey),control[None])
                for c in 'AB':
                    cx,cy,parts=coh[c]; queries,h=stacked_inputs(px,py,control,cx,cy,parts); row={'task':name,'budget':b,'subset':s,'cohort':c,'public_family':family,'control':{'ce':cce,'accuracy':cacc},'risks':{}}
                    for risk in RISKS:
                        key=f'noisy_{int(risk*100)}'; cfg=fz[key]; seed=801000000+b*1000+s*100+'AB'.index(c)*10+int(risk*100)
                        ce,acc=scores(ex,ey,stacked_theta(control,h,queries[cfg['mode']],cfg,draws=DRAWS_CONFIRM,seed=seed)); saved[f'{name}_b{b}_s{s}_{c}_{key}_ce']=ce
                        row['risks'][str(risk)]={'ce':float(ce.mean()),'accuracy':float(acc.mean()),'gain':cce-float(ce.mean())}
                    cells.append(row)
                print(name,b,s,'done',flush=True)
        tasks[name]=cells
        saved[name+'_roles']=np.concatenate(list(r.values()))
    np.savez_compressed(NPZ,**saved)
    OUT.write_text(json.dumps({'kind':'Descriptive public-budget map of the frozen stacked constructor; no retuning.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'freeze_sha':digest(STACKED_FREEZE),'draws':DRAWS_CONFIRM,'tasks':tasks},indent=1,allow_nan=False)+'\n')


if __name__=='__main__': main()
