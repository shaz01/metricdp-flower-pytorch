"""Offline conditional matched-CIA envelope; parameter selection is NOT private."""
from __future__ import annotations
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri
from sklearn.metrics import roc_auc_score,roc_curve
from research.calculations.private_descriptor_probe import ROOT,C,FAMILIES,encode,config_grid,decode
from research.calculations.private_prior_constructor_probe import prepare,TRANSFORMS,STEPS
from research.calculations.client_geometry_audit import features
from research.calculations.client_energy_filter_probe import scores,summarize
from research.calculations.client_energy_one_release import local_query

RISKS=(.55,.65,.8)


def auc_statistics(positive,negative):
    p=np.asarray(positive);n=np.asarray(negative);sp=np.sort(p);sn=np.sort(n)
    vp=(np.searchsorted(sn,p,'left')+np.searchsorted(sn,p,'right'))/(2*len(n))
    vn=1-(np.searchsorted(sp,n,'left')+np.searchsorted(sp,n,'right'))/(2*len(p))
    auc=float(vp.mean());se=float(np.sqrt(vp.var(ddof=1)/len(p)+vn.var(ddof=1)/len(n)))
    fpr,tpr,_=roc_curve(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n],drop_intermediate=False)
    return {'auc':auc,'se':se,'ci95':[max(0.,auc-1.96*se),min(1.,auc+1.96*se)],'tpr_at_fpr01':float(tpr[fpr<=.01].max()),'tpr_at_fpr05':float(tpr[fpr<=.05].max())}


def gaussian_std(magnitude,risk):
    if magnitude<=0 or not .5<risk<1:raise ValueError('Positive fixed shift and risk in(.5,1) required')
    return magnitude/(np.sqrt(2)*ndtri(risk))


def radial_ranks(z,scale,world):
    n2=np.sum(z*z,axis=1);n=np.sqrt(n2)
    if world:return np.sqrt(np.maximum(n2+2*z[:,0]/scale+1/scale**2,0))-n
    return n-np.sqrt(np.maximum(n2-2*z[:,0]/scale+1/scale**2,0))


@lru_cache(maxsize=None)
def radial_scale_ratio(d,risk):
    rng=np.random.default_rng(81000000+d*1000+int(risk*100))
    z=[rng.normal(size=(16384,d))*np.sqrt(rng.gamma((d+1)/2,2,size=(16384,1))) for _ in (0,1)]
    labels=np.r_[np.zeros(16384),np.ones(16384)]
    low=1e-6;high=1.
    def auc(s):return roc_auc_score(labels,np.r_[radial_ranks(z[0],s,0),radial_ranks(z[1],s,1)])
    while auc(high)>risk:high*=2
    for _ in range(36):
        middle=(low+high)/2
        if auc(middle)>risk:low=middle
        else:high=middle
    return high


def noise_objects(signal,law,scale,draws,seed,central=False):
    rng=np.random.default_rng(seed);d=signal.shape[1];z=rng.normal(size=(draws,8,d))
    divisor=8 if central else 7
    if law=='radial':z*=np.sqrt(rng.gamma((d+1)/2/divisor,2*scale**2,size=(draws,8,1)))
    else:z*=scale/np.sqrt(divisor)
    return signal[None]+z


def bounded(query,radius):
    q=query/8;norm=np.linalg.norm(q,axis=1)
    signal=q*np.minimum(1,(radius/8)/np.maximum(norm,1e-100))[:,None]
    return signal,float(np.linalg.norm(signal,axis=1).max())


def calibration(signal,law,risk):
    magnitude=float(np.linalg.norm(signal,axis=1).max());d=signal.shape[1]
    scale=gaussian_std(magnitude,risk) if law=='gaussian' else magnitude*radial_scale_ratio(d,risk)
    return {'law':law,'risk':risk,'dimension':d,'fixed_M':magnitude,'scale':scale,'unknown_variance':scale**2*(1 if law=='gaussian' else d+1)}


def development():
    original=json.loads((ROOT/'private_prior_constructor_development.json').read_text());stored=np.load(ROOT/'private_prior_constructor_development.npz')
    source,ids,public,x,y,dev,anchors,decoders,denoisers,_,_=prepare(False)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original['source_hash']
    rows=[];saved={}
    for seed in (42,43,44):
        teachers={(t,n):stored[f'seed{seed}_{t}_steps{n}_teachers'] for t in TRANSFORMS for n in STEPS}
        queries={(t,f,n,k):encode(teachers[t,n],f,anchors.get(k),active_mask=stored[f'seed{seed}_private_class_counts'].sum(axis=1)>0) for t in TRANSFORMS for f in FAMILIES for n in STEPS for k in ((0,) if f=='model_direct' else (4,16))}
        for risk in RISKS:
            for law in ('gaussian','radial'):
                candidates=[];selected={}
                for family in FAMILIES:
                    choices=[]
                    for transform in TRANSFORMS:
                        for j,config in enumerate(config_grid(family)):
                            signal,magnitude=bounded(queries[transform,family,config['steps'],config['prototypes']],config['radius'])
                            cal=calibration(signal,law,risk)
                            obj=noise_objects(signal,law,cal['scale'],32,82000000+seed*10000+int(risk*100))
                            theta=decode(obj,np.ones(8)/8,config,'aggregate',anchors,decoders,denoisers);ce,_=scores(*dev,theta)
                            key=f'seed{seed}_q{int(risk*100)}_{law}_{transform}_{family}_{j}'
                            saved[key+'_ce']=ce
                            record={'configuration':config,'transform':transform,'development_ce':float(ce.mean()),'calibration':cal,'archive_key':key}
                            choices.append(record);candidates.append(record)
                    selected[family]=min(choices,key=lambda v:v['development_ce'])
                control=min((f for f in FAMILIES if f!='votes'),key=lambda f:selected[f]['development_ce'])
                rows.append({'seed':seed,'risk':risk,'law':law,'candidates':candidates,'selected':selected,'control':control})
                print('development',seed,risk,law,'vote',round(selected['votes']['development_ce'],4),'control',control,round(selected[control]['development_ce'],4),flush=True)
    calibrations=[]
    for d in (12,48,51):
        for risk in RISKS:
            s=radial_scale_ratio(d,risk);rng=np.random.default_rng(81000000+d*1000+int(risk*100));z=[rng.normal(size=(16384,d))*np.sqrt(rng.gamma((d+1)/2,2,size=(16384,1))) for _ in (0,1)]
            ranks=[radial_ranks(z[w],s,w) for w in (0,1)]
            key=f'calibration_d{d}_q{int(risk*100)}'
            for w in (0,1):saved[key+f'_world{w}_score']=ranks[w]
            calibrations.append({'dimension':d,'risk':risk,'scale_ratio':s,'statistics':auc_statistics(ranks[1],ranks[0])})
    np.savez_compressed(ROOT/'matched_cia_development.npz',**saved)
    (ROOT/'matched_cia_development.json').write_text(json.dumps({'source_hash':original['source_hash'],'protocol_sha':hashlib.sha256(Path('research/proposals/2026-10-07_matched_cia_protocol.md').read_bytes()).hexdigest(),'rows':rows,'radial_calibrations':calibrations},indent=2,allow_nan=False)+'\n')


def evaluation():
    p=ROOT/'matched_cia_development.json';development=json.loads(p.read_text());stored=np.load(ROOT/'private_prior_constructor_development.npz')
    source,ids,public,x,y,dev,anchors,decoders,denoisers,_,_=prepare(False)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==development['source_hash']
    prior=np.load(ROOT/'private_prior_constructor_evaluation.npz');labels=np.array(__import__('datasets').Dataset.from_file(str(source))['label'])
    unused=np.array([i for i in ids['reserve'] if i not in set(prior['confirmation_original_indices'])])
    assert np.array_equal(np.sort(unused),np.sort(prior['remaining_reserve_original_indices']))
    new=np.concatenate([unused[labels[unused]==k][:256] for k in range(4)])
    assert len(new)==1024 and len(set(new))==1024
    utility=(features(__import__('datasets').Dataset.from_file(str(source)),new),labels[new])
    saved={'utility_original_indices':new,'remaining_reserve_original_indices':np.setdiff1d(unused,new)};rows=[]
    for record in development['rows']:
        seed=record['seed'];risk=record['risk'];law=record['law'];choices={f:dict(v) for f,v in record['selected'].items()}
        if law=='radial':
            for name in ('gaussian_same_query','gaussian_variance_match','central_same_query'):
                v=dict(record['selected']['votes']);v['calibration']=dict(v['calibration'])
                if name=='gaussian_same_query':v['calibration'].update(law='gaussian',scale=gaussian_std(v['calibration']['fixed_M'],risk))
                elif name=='gaussian_variance_match':v['calibration'].update(law='gaussian',scale=np.sqrt(v['calibration']['unknown_variance']))
                v['calibration']['unknown_variance']=v['calibration']['scale']**2*(1 if v['calibration']['law']=='gaussian' else v['calibration']['dimension']+1)
                choices[name]=v
        for arm,choice in choices.items():
            config=choice['configuration'];cal=choice['calibration'];d=cal['dimension'];scale=cal['scale'];channel=cal['law'];central=arm=='central_same_query'
            teacher=stored[f'seed{seed}_{choice["transform"]}_steps{config["steps"]}_teachers']
            query=encode(teacher,config['family'],anchors.get(config['prototypes']),active_mask=stored[f'seed{seed}_private_class_counts'].sum(axis=1)>0)
            signal,magnitude=bounded(query,config['radius']);assert np.isclose(magnitude,cal['fixed_M'])
            for target in range(8):
                peer=6 if target==7 else 7
                key=f'seed{seed}_q{int(risk*100)}_{law}_{arm}_target{target}';shift=signal[target];out=signal.copy();out[target]=0
                statscores=[];statutility={}
                for world in (0,1):
                    active=signal if world else out
                    obj=noise_objects(active,channel,scale,4096,83000000+seed*100000+int(risk*100)*1000+target*10+world,central)
                    observed=obj.sum(axis=1) if central else obj.sum(axis=1)-obj[:,peer]
                    base=out.sum(axis=0) if central else out.sum(axis=0)-out[peer]
                    if channel=='gaussian':rank=(observed-base-shift/2)@shift/scale**2
                    else:rank=(np.linalg.norm(observed-base,axis=1)-np.linalg.norm(observed-base-shift,axis=1))/scale
                    statscores.append(rank);saved[key+f'_world{world}_attack_score']=rank
                    obj=noise_objects(active,channel,scale,256,84000000+seed*100000+int(risk*100)*1000+target*10+world,central)
                    theta=decode(obj,np.ones(8)/8,config,'central' if central else 'aggregate',anchors,decoders,denoisers)
                    ce,acc=scores(*utility,theta);saved[key+f'_world{world}_utility_ce']=ce;saved[key+f'_world{world}_utility_accuracy']=acc
                    statutility[str(world)]={'ce':summarize(ce),'accuracy':summarize(acc)}
                rows.append({'seed':seed,'risk':risk,'law_selection':law,'arm':arm,'target':target,'peer':peer,'configuration':config,'transform':choice['transform'],'calibration':cal,'target_shift_norm':float(np.linalg.norm(shift)),
                             'attack':auc_statistics(statscores[1],statscores[0]),'gaussian_expected_auc':float(ndtr(np.linalg.norm(shift)/(np.sqrt(2)*scale))) if channel=='gaussian' else None,'utility':statutility})
        print('evaluation',seed,risk,law,'arms',len(choices),'targets8',flush=True)
    np.savez_compressed(ROOT/'matched_cia_evaluation.npz',**saved)
    public_teacher=local_query(public,labels[ids['public']],80)
    baseline_ce,baseline_acc=scores(*utility,public_teacher[None])
    (ROOT/'matched_cia_evaluation.json').write_text(json.dumps({'source_hash':development['source_hash'],'development_sha':hashlib.sha256(p.read_bytes()).hexdigest(),'calculator_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'utility_count':len(new),'remaining_reserve_count':len(saved['remaining_reserve_original_indices']),'uniform_control':{'ce':float(np.log(4)),'accuracy':.25},'labelled_public_control':{'ce':float(baseline_ce[0]),'accuracy':float(baseline_acc[0])},'rows':rows},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('development','evaluation'),default='development');args=parser.parse_args()
    if args.stage=='development':development()
    else:evaluation()
