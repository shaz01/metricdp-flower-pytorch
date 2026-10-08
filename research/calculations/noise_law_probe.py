"""Matched-strength Gaussian vs radial-Laplace noise on the stacked constructor: radial freeze, fresh KMNIST comparison, attack verification."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score, roc_curve
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import scores
from research.calculations.client_geometry_audit import features
from research.calculations.public_residual_probe import digest, public_geometry, bounded_public, public_scale
from research.calculations.limited_public_probe import mean_scores, CAPS, DIMS
from research.calculations.matched_cia_probe import radial_scale_ratio, radial_ranks
from research.calculations.headroom_probe import development_halves
from research.calculations.stacked_constructor_probe import stacked_inputs, stacked_theta, MODES, FREEZE as STACKED_FREEZE, DRAWS_CONFIRM
from research.calculations.frozen_constructor_probe import public_candidates, ETAS
from research.calculations.gated_step_probe import load, roles, MULTIPLIERS, SEED as GATED_SEED

PROTOCOL=Path('research/proposals/2026-10-08_noise_law_protocol.md')
CODE=Path('research/calculations/noise_law_probe.py')
FREEZE=ROOT/'noise_law_radial_freeze.json'
OUT=ROOT/'noise_law_comparison.json'
NPZ=ROOT/'noise_law_comparison.npz'
LAW_RISKS=(.65,.80)
SELECTION_DRAWS=32
ATTACK_DRAWS=65536


def radial_total(draws,seed,d,cap,risk):
    """Sum of eight Gamma-share radial-Laplace shares (peer contract: 7 shares calibrated, 8 released)."""
    scale=(cap/8)*radial_scale_ratio(d,risk); rng=np.random.default_rng(seed)
    z=rng.normal(size=(draws,8,d))*np.sqrt(rng.gamma((d+1)/2/7,2*scale**2,size=(draws,8,1)))
    return z.sum(1)


def law_theta(control,h,query,cfg,law,draws,seed):
    d=cfg['dimension']; g=public_geometry(h,d,'euclidean'); signal=bounded_public(query@g['encoder'],cfg['cap'])[0].sum(0)
    if law=='gaussian':
        noise=np.random.default_rng(seed).normal(size=(draws,8,51))[:,:,:d].sum(1)*public_scale(cfg['cap'],cfg['risk'])/np.sqrt(7)
    else: noise=radial_total(draws,seed,d,cfg['cap'],cfg['risk'])
    return control[None]-cfg['eta']*((signal[None]+noise)@g['decoder']).reshape(-1,3,17)


def freeze():
    from research.calculations.stacked_constructor_probe import HEAD, BANK
    from datasets import Dataset
    source,ids,public,private,y,dev,*_=prepare(False); labels=np.array(Dataset.from_file(str(source))['label'])
    sid,_=development_halves(ids['development'],labels); look={int(v):i for i,v in enumerate(ids['development'])}; sel=(dev[0][[look[int(v)] for v in sid]],labels[sid])
    head=json.loads(HEAD.read_text()); bank=np.load(BANK); plook={int(v):i for i,v in enumerate(ids['public'])}; rlook={int(v):i for i,v in enumerate(ids['private'])}
    controls={(r['budget'],r['subset_seed']):bank[r['selected']['archive_key']] for r in head['public_rows']}; table={}
    for seed in (42,43,44):
        for cohort in 'AB':
            pid=bank[f'budget32_seed{seed}_public_original_indices']; px=public[[plook[int(v)] for v in pid]]; py=labels[pid]
            parts=[np.array([rlook[int(v)] for v in row]) for row in bank[f'cohort{cohort}_original_indices']]; control=controls[(32,seed)]
            queries,h=stacked_inputs(px,py,control,private,y,parts); base=611000000+seed*100+'AB'.index(cohort)
            for mode in MODES:
                for d in DIMS:
                    g=public_geometry(h,d,'euclidean'); q=queries[mode]@g['encoder']
                    for cap in CAPS:
                        signal=bounded_public(q,cap)[0].sum(0)
                        for risk in LAW_RISKS:
                            noise=radial_total(SELECTION_DRAWS,base,d,cap,risk)
                            for eta in ETAS: table.setdefault((mode,d,cap,eta,risk),[]).append(mean_scores(sel,control[None]-eta*((signal[None]+noise)@g['decoder']).reshape(-1,3,17))[0])
    pooled=[{'mode':k[0],'dimension':k[1],'cap':k[2],'eta':k[3],'risk':k[4],'pooled_selection_ce':float(np.mean(v))} for k,v in table.items()]
    frozen={f'noisy_{int(r*100)}':min((p for p in pooled if p['risk']==r),key=lambda p:p['pooled_selection_ce']) for r in LAW_RISKS}
    for n,c in frozen.items(): print('radial frozen',n,c,'EDGE' if c['eta']==ETAS[-1] else '',flush=True)
    FREEZE.write_text(json.dumps({'kind':'Radial-Laplace pooled development freeze; selection CE only.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'frozen':frozen,'pooled':pooled},indent=1,allow_nan=False)+'\n')


def attack_verification(d,risk,law,seed):
    """Optimal known-alternative statistic for a unit shift: Gaussian projection onto the shift, or radial ranks; AUC and low-FPR TPRs."""
    rng=np.random.default_rng(seed)
    if law=='gaussian':
        sd=1/(np.sqrt(2)*ndtri(risk)); x=[rng.normal(size=(ATTACK_DRAWS,d))*sd for _ in (0,1)]; x[1][:,0]+=1; ranks=[x[0][:,0],x[1][:,0]]
    else:
        ratio=radial_scale_ratio(d,risk); z=[rng.normal(size=(ATTACK_DRAWS,d))*np.sqrt(rng.gamma((d+1)/2,2,size=(ATTACK_DRAWS,1))) for _ in (0,1)]
        ranks=[radial_ranks(z[0],ratio,0),radial_ranks(z[1],ratio,1)]
    labels=np.r_[np.zeros(ATTACK_DRAWS),np.ones(ATTACK_DRAWS)]; score=np.r_[ranks[0],ranks[1]]; fpr,tpr,_=roc_curve(labels,score,drop_intermediate=False)
    return {'law':law,'dimension':d,'risk':risk,'auc':float(roc_auc_score(labels,score)),**{f'tpr_at_fpr{k}':float(tpr[fpr<=f].max()) for k,f in (('001',.001),('01',.01),('05',.05))}}


def compare():
    fz=json.loads(STACKED_FREEZE.read_text())['frozen']; rf=json.loads(FREEZE.read_text()); assert digest(PROTOCOL)==rf['protocol_sha'] and digest(CODE)==rf['calculator_sha']
    configs={'gaussian':fz,'radial':rf['frozen']}; saved={}; tasks={}; verification=[]
    for law in ('gaussian','radial'):
        for risk in LAW_RISKS: verification.append(attack_verification(configs[law][f'noisy_{int(risk*100)}']['dimension'],risk,law,991000000+int(risk*100)))
    gated=json.loads((ROOT/'gated_step.json').read_text())
    for name,source,first in (('kmnist_classes0to3','kmnist',0),('kmnist_classes4to7','kmnist',4)):
        data,raw=load(source); r=roles(raw,first); lab=lambda ids:raw[ids]-first
        vx=features(data,r['validation']); vy=lab(r['validation']); ex=features(data,r['evaluation']); ey=lab(r['evaluation']); coh={}
        counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
        for c in 'AB':
            cx=features(data,r[c]); cy=lab(r[c]); coh[c]=(cx,cy,partition_by_class_counts(cy,counts,seed=GATED_SEED+first))
        cells=[]
        for s in range(3):
            ids=r[f'public32_{s}']; px=features(data,ids); py=lab(ids); *_,family,control=public_candidates(px,py,(vx,vy)); cce=float(scores(ex,ey,control[None])[0].mean())
            for c in 'AB':
                cx,cy,parts=coh[c]; queries,h=stacked_inputs(px,py,control,cx,cy,parts); row={'task':name,'subset':s,'cohort':c,'control_ce':cce,'risks':{}}
                for risk in LAW_RISKS:
                    key=f'noisy_{int(risk*100)}'; seed=901000000+32*1000+s*100+'AB'.index(c)*10+int(risk*100); out={}
                    for law in ('gaussian','radial'):
                        cfg=configs[law][key]; ece=[];vce=[]
                        for m in MULTIPLIERS:
                            theta=law_theta(control,h,queries[cfg['mode']],dict(cfg,eta=cfg['eta']*m),law,DRAWS_CONFIRM,seed)
                            vce.append(scores(vx,vy,theta)[0]); ece.append(scores(ex,ey,theta)[0])
                        vce,ece=np.array(vce),np.array(ece); pick=vce.argmin(0); g=ece[pick,np.arange(ece.shape[1])]
                        out[law]={'configuration':cfg,'fixed_m1_ce':float(ece[3].mean()),'gated_ce':float(g.mean()),'gated_gain_over_control':cce-float(g.mean())}
                        saved[f'{name}_s{s}_{c}_{key}_{law}_gated_ce']=g
                    out['gaussian_minus_radial_gated_ce']=out['gaussian']['gated_ce']-out['radial']['gated_ce']
                    out['radial_advantage']=-out['gaussian_minus_radial_gated_ce']
                    old=next(x for x in gated['tasks'][name] if x['budget']==32 and x['subset']==s and x['cohort']==c)['risks'][str(risk)]['gated_val512']['gain']
                    assert abs(old-out['gaussian']['gated_gain_over_control'])<1e-9
                    row['risks'][str(risk)]=out
                cells.append(row); print(name,s,c,{k:round(v['radial_advantage'],5) for k,v in row['risks'].items()},flush=True)
        tasks[name]=cells
    np.savez_compressed(NPZ,**saved)
    OUT.write_text(json.dumps({'kind':'Matched-AUC Gaussian vs radial Laplace on the stacked constructor; each law frozen separately; fresh KMNIST.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'radial_freeze_sha':digest(FREEZE),'attack_verification':verification,'tasks':tasks},indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=('freeze','compare'),required=True); a=p.parse_args()
    freeze() if a.stage=='freeze' else compare()
