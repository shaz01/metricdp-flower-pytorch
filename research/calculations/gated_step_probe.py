"""Validation-gated step size for the frozen stacked constructor on KMNIST (fresh) plus seen-pool replicates."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset, load_dataset
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import scores
from research.calculations.client_geometry_audit import features
from research.calculations.public_residual_probe import digest
from research.calculations.limited_public_probe import RISKS
from research.calculations.stacked_constructor_probe import stacked_inputs, stacked_theta, FREEZE as STACKED_FREEZE, DRAWS_CONFIRM
from research.calculations.frozen_constructor_probe import public_candidates

PROTOCOL=Path('research/proposals/2026-10-08_gated_step_protocol.md')
CODE=Path('research/calculations/gated_step_probe.py')
OUT=ROOT/'gated_step.json'
NPZ=ROOT/'gated_step.npz'
BUDGETS=(32,128,512)
MULTIPLIERS=(0.,.25,.5,1.,2.)
TASKS=(('kmnist_classes0to3','kmnist',0,True),('kmnist_classes4to7','kmnist',4,True),('fmnist_classes4to7','fmnist',4,False),('mnist_digits0to3','mnist',0,False))
SEED=20261011


def load(source):
    if source=='kmnist': data=load_dataset('tanganke/kmnist')['train']
    elif source=='mnist': data=load_dataset('ylecun/mnist')['train']
    else:
        cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'; data=Dataset.from_file(str(next(cache.rglob('fashion_mnist-train.arrow'))))
    return data,np.array(data['label'])


def roles(raw,first):
    rng=np.random.default_rng(SEED+first); out={f'public{b}_{s}':[] for b in BUDGETS for s in range(3)}; out.update(validation=[],A=[],B=[],evaluation=[])
    for k in range(4):
        ids=np.where(raw==first+k)[0].copy(); rng.shuffle(ids); pos=0
        for b in BUDGETS:
            for s in range(3): out[f'public{b}_{s}'].append(ids[pos:pos+b//4]); pos+=b//4
        for n,m in (('validation',128),('A',512),('B',512),('evaluation',512)): out[n].append(ids[pos:pos+m]); pos+=m
    out={n:np.concatenate(v) for n,v in out.items()}; allids=np.concatenate(list(out.values())); assert len(set(allids.tolist()))==len(allids)
    return out


def gated(val_ce,eval_ce):
    """val_ce/eval_ce: (multipliers, draws). Choose the multiplier per draw by validation CE, score on evaluation."""
    pick=val_ce.argmin(axis=0); return eval_ce[pick,np.arange(eval_ce.shape[1])],pick


def main():
    fz=json.loads(STACKED_FREEZE.read_text())['frozen']; saved={}; tasks={}
    for name,source,first,fresh in TASKS:
        data,raw=load(source); r=roles(raw,first); lab=lambda ids:raw[ids]-first
        vx=features(data,r['validation']); vy=lab(r['validation']); sel=(vx,vy); ex=features(data,r['evaluation']); ey=lab(r['evaluation'])
        sub=np.concatenate([np.where(vy==k)[0][:32] for k in range(4)]); coh={}
        counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
        for c in 'AB':
            cx=features(data,r[c]); cy=lab(r[c]); coh[c]=(cx,cy,partition_by_class_counts(cy,counts,seed=SEED+first)); assert all(len(p)==256 for p in coh[c][2])
        cells=[]
        for b in BUDGETS:
            for s in range(3):
                ids=r[f'public{b}_{s}']; px=features(data,ids); py=lab(ids); _b,_g,_h,family,control=public_candidates(px,py,sel)
                cce=float(scores(ex,ey,control[None])[0].mean()); cacc=float(scores(ex,ey,control[None])[1].mean())
                for c in 'AB':
                    cx,cy,parts=coh[c]; queries,h=stacked_inputs(px,py,control,cx,cy,parts); row={'task':name,'fresh_dataset':fresh,'budget':b,'subset':s,'cohort':c,'public_family':family,'control':{'ce':cce,'accuracy':cacc},'risks':{}}
                    for risk in RISKS:
                        key=f'noisy_{int(risk*100)}'; cfg=fz[key]; seed=901000000+b*1000+s*100+'AB'.index(c)*10+int(risk*100)
                        vce=[];ece=[];eacc=[];vce128=[]
                        for m in MULTIPLIERS:
                            theta=stacked_theta(control,h,queries[cfg['mode']],dict(cfg,eta=cfg['eta']*m),draws=DRAWS_CONFIRM,seed=seed)
                            vce.append(scores(vx,vy,theta)[0]); vce128.append(scores(vx[sub],vy[sub],theta)[0]); e,a=scores(ex,ey,theta); ece.append(e); eacc.append(a)
                        vce,vce128,ece,eacc=map(np.array,(vce,vce128,ece,eacc)); out={'fixed_m1':{'ce':float(ece[3].mean()),'gain':cce-float(ece[3].mean()),'accuracy':float(eacc[3].mean())}}
                        for label,v in (('gated_val512',vce),('gated_val128',vce128)):
                            pick=v.argmin(axis=0); g=ece[pick,np.arange(ece.shape[1])]; ga=eacc[pick,np.arange(ece.shape[1])]
                            out[label]={'ce':float(g.mean()),'gain':cce-float(g.mean()),'accuracy':float(ga.mean()),'pick_fraction':[float((pick==i).mean()) for i in range(len(MULTIPLIERS))]}
                        out['oracle_best_multiplier_mean_ce']=float(ece.mean(axis=1).min()); out['by_multiplier_ce']=[float(x) for x in ece.mean(axis=1)]
                        row['risks'][str(risk)]=out; saved[f'{name}_b{b}_s{s}_{c}_{key}_eval_ce']=ece
                    cells.append(row)
                print(name,b,s,'done',flush=True)
        tasks[name]=cells; saved[name+'_roles']=np.concatenate(list(r.values()))
    np.savez_compressed(NPZ,**saved)
    OUT.write_text(json.dumps({'kind':'Validation-gated step size on frozen stacked constructor; grid fixed a priori; no retuning.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'freeze_sha':digest(STACKED_FREEZE),'multipliers':MULTIPLIERS,'draws':DRAWS_CONFIRM,'tasks':tasks},indent=1,allow_nan=False)+'\n')


if __name__=='__main__': main()
