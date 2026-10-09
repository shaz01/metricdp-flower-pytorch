"""Stacked public-plus-private constructor: pooled development freeze and fresh-task transfer (no re-tuning)."""
import argparse
import json
from pathlib import Path
import numpy as np
from datasets import Dataset, load_dataset
from scipy.special import ndtr
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import scores
from research.calculations.client_geometry_audit import features
from research.calculations.public_residual_probe import digest, public_geometry, public_hessian, bounded_public, public_scale, random_shares
from research.calculations.public_residual_signal_diagnostic import example_gradients
from research.calculations.class_conditional_probe import client_queries, public_offset
from research.calculations.headroom_probe import development_halves
from research.calculations.limited_public_probe import noise_total, step, mean_scores, CAPS, DIMS, RISKS
from research.calculations.frozen_constructor_probe import public_candidates, construct as unstacked_construct, ETAS, FREEZE as UNSTACKED_FREEZE
from research.calculations.matched_cia_probe import auc_statistics

PROTOCOL=Path('research/proposals/2026-10-08_stacked_constructor_protocol.md')
CODE=Path('research/calculations/stacked_constructor_probe.py')
FREEZE=ROOT/'stacked_constructor_freeze.json'
TRANSFER=ROOT/'stacked_constructor_transfer.json'
TRANSFER_NPZ=ROOT/'stacked_constructor_transfer.npz'
HEAD=ROOT/'headroom_development.json'
BANK=ROOT/'headroom_development.npz'
MODES=('target_balanced','target_center')
DRAWS_SELECTION=32
DRAWS_CONFIRM=512
ATTACK_DRAWS=2048
TASKS=('fmnist_classes4to7','mnist_digits0to3')


def stacked_inputs(px,py,control,cx,cy,parts):
    """Client queries at the strongest public model T0, with public class gradients at T0 restoring the target_center mode."""
    gamma=np.stack([example_gradients(px[py==k],py[py==k],control).mean(0) for k in range(4)])
    queries,_=client_queries(cx,cy,parts,control,gamma,np.ones(4)/4)
    return {m:queries[m].reshape(8,51) for m in MODES},public_hessian(px,control)


def stacked_theta(control,h,query,cfg,draws=0,seed=0,noiseless=False):
    d=cfg['dimension']; g=public_geometry(h,d,'euclidean'); signal=bounded_public(query@g['encoder'],cfg['cap'])[0].sum(0)
    total=signal[None]+noise_total(draws,seed,d,public_scale(cfg['cap'],cfg['risk'])) if draws and not noiseless else signal
    return control[None]-cfg['eta']*(np.atleast_2d(total)@g['decoder']).reshape(-1,3,17)


def attack(signal,d,sigma,seed):
    out=[]
    for target in range(8):
        peer=6 if target==7 else 7; absent=signal.copy(); absent[target]=0; shift=signal[target]; base=absent.sum(0)-absent[peer]; worlds=[]
        for world in (0,1):
            active=signal if world else absent; shares=random_shares(ATTACK_DRAWS,seed+10*target+world)[:,:,:d]
            obj=active[None]+shares*sigma/np.sqrt(7); observed=obj.sum(1)-obj[:,peer]; worlds.append((observed-base-shift/2)@shift/sigma**2)
        a=auc_statistics(worlds[1],worlds[0]); norm=float(np.linalg.norm(shift)); a.update(target=target,shift_norm=norm,expected_auc=float(ndtr(norm/(np.sqrt(2)*sigma)))); out.append(a)
    return out


def freeze():
    source,ids,public,private,y,dev,*_=prepare(False); labels=np.array(Dataset.from_file(str(source))['label'])
    sid,aid=development_halves(ids['development'],labels); look={int(v):i for i,v in enumerate(ids['development'])}
    sel=(dev[0][[look[int(v)] for v in sid]],labels[sid])
    head=json.loads(HEAD.read_text()); bank=np.load(BANK); assert digest(BANK)==head['bank_sha']
    plook={int(v):i for i,v in enumerate(ids['public'])}; rlook={int(v):i for i,v in enumerate(ids['private'])}
    controls={(r['budget'],r['subset_seed']):bank[r['selected']['archive_key']] for r in head['public_rows']}
    table={}; cells=[(s,c) for s in (42,43,44) for c in 'AB']
    for seed,cohort in cells:
        pid=bank[f'budget32_seed{seed}_public_original_indices']; px=public[[plook[int(v)] for v in pid]]; py=labels[pid]
        parts=[np.array([rlook[int(v)] for v in row]) for row in bank[f'cohort{cohort}_original_indices']]; control=controls[(32,seed)]
        queries,h=stacked_inputs(px,py,control,private,y,parts); base=601000000+seed*100+'AB'.index(cohort)
        record_zero=mean_scores(sel,control[None])[0]; table.setdefault(('public_control',None,None,None,None,None),[]).append(record_zero)
        for mode in MODES:
            for d in DIMS:
                g=public_geometry(h,d,'euclidean'); q=queries[mode]@g['encoder']
                for cap in CAPS:
                    signal=bounded_public(q,cap)[0].sum(0)
                    for eta in ETAS:
                        table.setdefault(('clipped',mode,d,cap,eta,None),[]).append(mean_scores(sel,control[None]-eta*(signal@g['decoder']).reshape(1,3,17))[0])
                        for risk in RISKS:
                            total=signal[None]+noise_total(DRAWS_SELECTION,base,d,public_scale(cap,risk))
                            table.setdefault(('noisy',mode,d,cap,eta,risk),[]).append(mean_scores(sel,control[None]-eta*(total@g['decoder']).reshape(-1,3,17))[0])
    pooled=[{'stage':k[0],'mode':k[1],'dimension':k[2],'cap':k[3],'eta':k[4],'risk':k[5],'pooled_selection_ce':float(np.mean(v)),'cell_selection_ce':[float(x) for x in v]} for k,v in table.items()]
    frozen={}
    for risk in RISKS: frozen[f'noisy_{int(risk*100)}']=min((p for p in pooled if p['stage']=='noisy' and p['risk']==risk),key=lambda p:p['pooled_selection_ce'])
    control_ce=float(np.mean(table[('public_control',None,None,None,None,None)]))
    for n,c in frozen.items(): print('frozen',n,{k:c[k] for k in ('mode','dimension','cap','eta','pooled_selection_ce')},'control',round(control_ce,5),'EDGE' if c['eta']==ETAS[-1] else '',flush=True)
    FREEZE.write_text(json.dumps({'kind':'Stacked pooled development freeze; selection CE only; no assessment/test/fresh-task access.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'headroom_sha':digest(HEAD),'bank_sha':digest(BANK),
        'eta_grid':ETAS,'pooled_control_selection_ce':control_ce,'frozen':frozen,'cells':[f'{s}{c}' for s,c in cells],'pooled':pooled},indent=1,allow_nan=False)+'\n')


def load_task(name):
    if name=='fmnist_classes4to7':
        cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'; data=Dataset.from_file(str(next(cache.rglob('fashion_mnist-train.arrow')))); first=4
    else:
        data=load_dataset('ylecun/mnist')['train']; first=0
    raw=np.array(data['label']); return data,raw,first


def task_roles(raw,first,seed):
    rng=np.random.default_rng(seed); out={'public':[[],[],[]],'selection':[],'A':[],'B':[],'evaluation':[]}
    for k in range(4):
        ids=np.where(raw==first+k)[0].copy(); rng.shuffle(ids)
        for s in range(3): out['public'][s].append(ids[8*s:8*s+8])
        out['selection'].append(ids[24:152]); out['A'].append(ids[152:664]); out['B'].append(ids[664:1176]); out['evaluation'].append(ids[1176:1688])
    roles={'public':[np.concatenate(p) for p in out['public']],**{n:np.concatenate(out[n]) for n in ('selection','A','B','evaluation')}}
    allids=np.concatenate(roles['public']+[roles[n] for n in ('selection','A','B','evaluation')]); assert len(set(allids.tolist()))==len(allids)
    return roles


def transfer():
    fz=json.loads(FREEZE.read_text()); assert digest(PROTOCOL)==fz['protocol_sha'] and digest(CODE)==fz['calculator_sha']
    unstacked=json.loads(UNSTACKED_FREEZE.read_text())['frozen']; frozen=fz['frozen']; saved={}; tasks={}
    for name in TASKS:
        data,raw,first=load_task(name); roles=task_roles(raw,first,20261009)
        lab=lambda ids:raw[ids]-first
        sel=(features(data,roles['selection']),lab(roles['selection'])); ex=features(data,roles['evaluation']); ey=lab(roles['evaluation']); cells=[]
        cohorts={}
        for c in 'AB':
            cx=features(data,roles[c]); cy=lab(roles[c]); counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
            parts=partition_by_class_counts(cy,counts,seed=20261009); assert all(len(p)==256 for p in parts); cohorts[c]=(cx,cy,parts)
        for s,pids in enumerate(roles['public']):
            px=features(data,pids); py=lab(pids); b,gamma,h_b,family,control=public_candidates(px,py,sel)
            control_ce,control_acc=mean_scores((ex,ey),control[None])
            for c in 'AB':
                cx,cy,parts=cohorts[c]; queries,h=stacked_inputs(px,py,control,cx,cy,parts)
                ub_queries,_=client_queries(cx,cy,parts,b,gamma,np.ones(4)/4)
                row={'task':name,'subset':s,'cohort':c,'public_family':family,'strong_public_control':{'ce':control_ce,'accuracy':control_acc},'risks':{}}
                for risk in RISKS:
                    key=f'noisy_{int(risk*100)}'; cfg=frozen[key]; seed=701000000+s*1000+('AB'.index(c))*100+int(risk*100)
                    ces={}
                    for arm,kw in (('clipped_noiseless',dict(noiseless=True)),('noisy',dict(draws=DRAWS_CONFIRM,seed=seed))):
                        ce,acc=scores(ex,ey,stacked_theta(control,h,queries[cfg['mode']],cfg,**kw)); ces[arm]={'ce':float(ce.mean()),'accuracy':float(acc.mean())}
                        if arm=='noisy': saved[f'{name}_s{s}_{c}_{key}_noisy_ce']=ce
                    ucfg=unstacked[key]; g=public_geometry(h_b,ucfg['dimension'],'euclidean')
                    uoff=public_offset(ucfg['mode'],gamma.mean(0),g); uq=ub_queries[ucfg['mode']].reshape(8,51)@g['encoder']
                    ce,acc=scores(ex,ey,unstacked_construct(b,uoff,g['decoder'],uq,ucfg,draws=DRAWS_CONFIRM,seed=seed)); ces['unstacked_noisy']={'ce':float(ce.mean()),'accuracy':float(acc.mean())}
                    gg=public_geometry(h,cfg['dimension'],'euclidean'); signal,factors=bounded_public(queries[cfg['mode']]@gg['encoder'],cfg['cap']); sigma=public_scale(cfg['cap'],risk)
                    row['risks'][str(risk)]={'configuration':cfg,'arms':ces,'clip_factor_mean':float(factors.mean()),'attack':attack(signal,cfg['dimension'],sigma,seed+9000000),
                        'gain_over_strong_public_control':control_ce-ces['noisy']['ce'],'gain_over_unstacked':ces['unstacked_noisy']['ce']-ces['noisy']['ce']}
                    print(name,'subset',s,c,'risk',risk,'gain vs control',round(control_ce-ces['noisy']['ce'],5),'vs unstacked',round(ces['unstacked_noisy']['ce']-ces['noisy']['ce'],5),flush=True)
                cells.append(row)
        saved[name+'_roles']=np.concatenate([roles['selection'],roles['A'],roles['B'],roles['evaluation']]+roles['public'])
        tasks[name]={'evaluation_count':len(ey),'cells':cells}
    np.savez_compressed(TRANSFER_NPZ,**saved)
    TRANSFER.write_text(json.dumps({'kind':'Fresh-task transfer of pooled-frozen stacked constructor; no re-tuning.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'freeze_sha':digest(FREEZE),'unstacked_freeze_sha':digest(UNSTACKED_FREEZE),
        'draws':DRAWS_CONFIRM,'attack_draws':ATTACK_DRAWS,'tasks':tasks},indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=('freeze','transfer'),required=True); a=p.parse_args()
    freeze() if a.stage=='freeze' else transfer()
