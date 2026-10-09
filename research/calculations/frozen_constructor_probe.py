"""Pooled-freeze (development) and fresh-data confirmation (test split) of the limited-public protected one-step construction."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from datasets import Dataset
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import scores
from research.calculations.client_energy_one_release import local_query
from research.calculations.client_geometry_audit import features
from research.calculations.public_residual_probe import digest, public_geometry, public_hessian, bounded_public, public_scale, random_shares
from research.calculations.public_residual_signal_diagnostic import example_gradients
from research.calculations.class_conditional_probe import public_offset, client_queries
from research.calculations.headroom_probe import development_halves, public_subset, refine
from research.calculations.limited_public_probe import noise_total, step, mean_scores, UNCLIPPED, MODES, DIMS, CAPS, RISKS
from research.calculations.matched_cia_probe import auc_statistics

PROTOCOL=Path('research/proposals/2026-10-08_frozen_constructor_protocol.md')
CODE=Path('research/calculations/frozen_constructor_probe.py')
FREEZE=ROOT/'frozen_constructor_freeze.json'
CONFIRM=ROOT/'frozen_constructor_confirmation.json'
CONFIRM_NPZ=ROOT/'frozen_constructor_confirmation.npz'
HEAD=ROOT/'headroom_development.json'
BANK=ROOT/'headroom_development.npz'
ETAS=(.1,.3,1.,3.,10.,30.,100.,300.,1000.)
DRAWS_SELECTION=32
DRAWS_CONFIRM=512
ATTACK_DRAWS=2048


def development_data():
    source,ids,_pub,_priv,_y,dev,*_=prepare(False); labels=np.array(Dataset.from_file(str(source))['label'])
    sid,aid=development_halves(ids['development'],labels); look={int(v):i for i,v in enumerate(ids['development'])}
    sel=(dev[0][[look[int(v)] for v in sid]],labels[sid]); ass=(dev[0][[look[int(v)] for v in aid]],labels[aid])
    return source,sid,aid,sel,ass


def total_signal(query,cap):
    return bounded_public(query,cap or UNCLIPPED)[0].sum(0)


def construct(b,offset,decoder,query,cfg,draws=0,seed=0,noiseless=False,unclipped=False):
    total=total_signal(query,None if unclipped else cfg['cap'])
    if draws and not noiseless: total=total[None]+noise_total(draws,seed,cfg['dimension'],public_scale(cfg['cap'],cfg['risk']))
    return step(b,cfg['eta'],offset,total,decoder)


def freeze():
    source,sid,aid,sel,ass=development_data(); bank=np.load(BANK); head=json.loads(HEAD.read_text()); assert digest(BANK)==head['bank_sha']
    cells=[(s,c) for s in (42,43,44) for c in 'AB']; table={}
    def record(key,value): table.setdefault(key,[]).append(value)
    for seed,cohort in cells:
        prefix=f'budget32_seed{seed}'; b=bank[prefix+'_reference']; gamma=bank[prefix+'_gamma']; h=bank[prefix+'_hessian']; base=401000000+seed*100+'AB'.index(cohort)
        for mode in MODES:
            full=bank[f'{prefix}_cohort{cohort}_{mode}'].reshape(8,51)
            for d in DIMS:
                g=public_geometry(h,d,'euclidean'); query=full@g['encoder']; offset=public_offset(mode,gamma.mean(0),g)
                for eta in ETAS:
                    record(('public_zero',mode,d,None,eta,None),mean_scores(sel,step(b,eta,offset,np.zeros(d),g['decoder']))[0])
                    record(('unclipped',mode,d,None,eta,None),mean_scores(sel,step(b,eta,offset,total_signal(query,None),g['decoder']))[0])
                for cap in CAPS:
                    signal=total_signal(query,cap)
                    for eta in ETAS:
                        record(('clipped',mode,d,cap,eta,None),mean_scores(sel,step(b,eta,offset,signal,g['decoder']))[0])
                        for risk in RISKS:
                            total=signal[None]+noise_total(DRAWS_SELECTION,base,d,public_scale(cap,risk))
                            record(('noisy',mode,d,cap,eta,risk),mean_scores(sel,step(b,eta,offset,total,g['decoder']))[0])
    pooled=[{'stage':k[0],'mode':k[1],'dimension':k[2],'cap':k[3],'eta':k[4],'risk':k[5],'pooled_selection_ce':float(np.mean(v)),'cell_selection_ce':[float(x) for x in v]} for k,v in table.items()]
    def argmin(stage,risk=None):
        return min((p for p in pooled if p['stage']==stage and p['risk']==risk),key=lambda p:p['pooled_selection_ce'])
    frozen={'public_zero':argmin('public_zero')}
    for risk in RISKS: frozen[f'noisy_{int(risk*100)}']=argmin('noisy',risk)
    for name,cfg in frozen.items():
        print('frozen',name,{k:cfg[k] for k in ('mode','dimension','cap','eta','risk','pooled_selection_ce')},'edge' if cfg['eta']==ETAS[-1] else '',flush=True)
    # Descriptive pooled assessment check of the frozen configurations (reused development half; never used to choose).
    check={}
    for name,cfg in frozen.items():
        vals=[]
        for seed,cohort in cells:
            prefix=f'budget32_seed{seed}'; b=bank[prefix+'_reference']; gamma=bank[prefix+'_gamma']; h=bank[prefix+'_hessian']; base=401000000+seed*100+'AB'.index(cohort)+5000000
            g=public_geometry(h,cfg['dimension'],'euclidean'); query=bank[f"{prefix}_cohort{cohort}_{cfg['mode']}"].reshape(8,51)@g['encoder']; offset=public_offset(cfg['mode'],gamma.mean(0),g)
            if name=='public_zero': theta=step(b,cfg['eta'],offset,np.zeros(cfg['dimension']),g['decoder'])
            else: theta=construct(b,offset,g['decoder'],query,cfg,128,base)
            vals.append(mean_scores(ass,theta)[0])
        check[name]=vals
    FREEZE.write_text(json.dumps({'kind':'Pooled development freeze; selection CE only; no reserve/test access.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'headroom_sha':digest(HEAD),'bank_sha':digest(BANK),
        'eta_grid':ETAS,'frozen':frozen,'frozen_assessment_ce_by_cell':check,'cells':[f'{s}{c}' for s,c in cells],'pooled':pooled},indent=1,allow_nan=False)+'\n')


def fresh_roles(test):
    labels=np.array(test['label']); rng=np.random.default_rng(20261008); roles={'public':[],'cohort':[],'evaluation':[]}
    for k in range(4):
        ids=np.where(labels==k)[0].copy(); rng.shuffle(ids); assert len(ids)==1000
        roles['public'].append(ids[:24].reshape(3,8)); roles['cohort'].append(ids[24:24+512]); roles['evaluation'].append(ids[24+512:])
    pub=[np.concatenate([roles['public'][k][s] for k in range(4)]) for s in range(3)]
    cohort=np.concatenate(roles['cohort']); evaluation=np.concatenate(roles['evaluation'])
    assert len(cohort)==2048 and len(evaluation)==4*464
    allids=np.concatenate(pub+[cohort,evaluation]); assert len(set(allids.tolist()))==len(allids)
    return labels,pub,cohort,evaluation


def public_candidates(px,py,sel):
    candidates=[]
    for steps in (80,320,1280,5120):
        theta=local_query(px,py,steps)
        for mult in (.5,.75,1.,1.25,1.5,2.): candidates.append(('public_scratch',theta*mult))
    scratch=min(candidates,key=lambda c:mean_scores(sel,c[1][None])[0]); b=scratch[1]
    gamma=np.stack([example_gradients(px[py==k],py[py==k],b).mean(0) for k in range(4)]); h=public_hessian(px,b)
    for steps in (20,80,320,1280): candidates.append(('public_refine',refine(px,py,b,steps)))
    for d in (1,3,12,51):
        g=public_geometry(h,d,'euclidean'); grad=(gamma.mean(0).ravel()@g['basis']@g['basis'].T).reshape(3,17)
        for eta in (.1,.3,1.,3.,10.,30.): candidates.append(('public_step',b-eta*grad))
    strong=min(candidates,key=lambda c:mean_scores(sel,c[1][None])[0])
    return b,gamma,h,strong[0],strong[1]


def confirm():
    frozen_doc=json.loads(FREEZE.read_text()); assert digest(PROTOCOL)==frozen_doc['protocol_sha'] and digest(CODE)==frozen_doc['calculator_sha']
    frozen=frozen_doc['frozen']; _,_,_,sel,_=development_data()
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    test=Dataset.from_file(str(next(cache.rglob('fashion_mnist-test.arrow'))))
    labels,pub,cohort,evaluation=fresh_roles(test)
    ex=features(test,evaluation); ey=labels[evaluation]
    cx=features(test,cohort); cy=labels[cohort]
    counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
    parts=partition_by_class_counts(cy,counts,seed=20261008); assert all(len(p)==256 for p in parts) and len(set(np.concatenate(parts).tolist()))==2048
    saved={'public_original_indices':np.stack(pub),'cohort_original_indices':cohort,'evaluation_original_indices':evaluation}; rows=[]
    for s,ids in enumerate(pub):
        px=features(test,ids); py=labels[ids]; b,gamma,h,family,control=public_candidates(px,py,sel)
        control_ce,control_acc=mean_scores((ex,ey),control[None])
        queries,priors=client_queries(cx,cy,parts,b,gamma,np.ones(4)/4)
        saved[f'subset{s}_reference']=b; saved[f'subset{s}_priors']=priors
        z=frozen['public_zero']; g=public_geometry(h,z['dimension'],'euclidean'); zero_off=public_offset(z['mode'],gamma.mean(0),g)
        zero_ce,zero_acc=mean_scores((ex,ey),step(b,z['eta'],zero_off,np.zeros(z['dimension']),g['decoder']))
        row={'subset':s,'public_original_indices':ids.tolist(),'strong_public_control':{'family':family,'ce':control_ce,'accuracy':control_acc},'frozen_public_zero':{'ce':zero_ce,'accuracy':zero_acc},'risks':{}}
        for risk in RISKS:
            cfg=frozen[f'noisy_{int(risk*100)}']; d=cfg['dimension']; g=public_geometry(h,d,'euclidean'); query=queries[cfg['mode']].reshape(8,51)@g['encoder']; offset=public_offset(cfg['mode'],gamma.mean(0),g)
            seed=501000000+s*1000+int(risk*100); sigma=public_scale(cfg['cap'],risk)
            ces={}
            for arm,kw in (('unclipped_noiseless',dict(noiseless=True,unclipped=True)),('clipped_noiseless',dict(noiseless=True)),('noisy',dict(draws=DRAWS_CONFIRM,seed=seed))):
                theta=construct(b,offset,g['decoder'],query,cfg,**kw); ce,acc=scores(ex,ey,theta); ces[arm]={'ce':float(ce.mean()),'accuracy':float(acc.mean()),'ce_draw_std':float(ce.std(ddof=1)) if len(ce)>1 else 0.}
                if arm=='noisy': saved[f'subset{s}_risk{int(risk*100)}_noisy_ce']=ce
            signal,factors=bounded_public(query,cfg['cap']); attack=[]
            for target in range(8):
                peer=6 if target==7 else 7; out=signal.copy(); out[target]=0; shift=signal[target]; base=out.sum(0)-out[peer]; worlds=[]
                for world in (0,1):
                    active=signal if world else out; sh=random_shares(ATTACK_DRAWS,seed+10*target+world+7000000)[:,:,:d]
                    obj=active[None]+sh*sigma/np.sqrt(7); observed=obj.sum(1)-obj[:,peer]
                    worlds.append((observed-base-shift/2)@shift/sigma**2)
                expected=float(np.sqrt(np.sum(shift**2))/(np.sqrt(2)*sigma))
                a=auc_statistics(worlds[1],worlds[0]); a.update(target=target,expected_auc=float(ndtr(expected)),shift_norm=float(np.sqrt(np.sum(shift**2)))); attack.append(a)
            row['risks'][str(risk)]={'configuration':cfg,'sigma':float(sigma),'arms':ces,'clip_factors':factors.tolist(),'attack':attack,
                'gain_over_strong_public_control':control_ce-ces['noisy']['ce'],'gain_over_public_zero':zero_ce-ces['noisy']['ce']}
            print('confirm subset',s,'risk',risk,'gain vs control',round(control_ce-ces['noisy']['ce'],5),'vs zero',round(zero_ce-ces['noisy']['ce'],5),'max attack auc',round(max(a['auc'] for a in attack),3),flush=True)
        rows.append(row)
    np.savez_compressed(CONFIRM_NPZ,**saved)
    CONFIRM.write_text(json.dumps({'kind':'Fresh-data confirmation of pooled-frozen constructor; test-split roles disjoint; no re-tuning.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'freeze_sha':digest(FREEZE),
        'evaluation_count':len(evaluation),'draws':DRAWS_CONFIRM,'attack_draws':ATTACK_DRAWS,'rows':rows},indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=('freeze','confirm'),required=True); a=p.parse_args()
    freeze() if a.stage=='freeze' else confirm()
