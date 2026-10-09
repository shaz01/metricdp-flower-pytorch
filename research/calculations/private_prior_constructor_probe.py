"""Development-first prior-aligned constructor and untouched-reserve confirmation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from research.calculations.private_descriptor_probe import (ROOT,FAMILIES,STEPS,RIDGES,CONTRACTS,C,encode,config_grid,sanitize,decode,ridge_decoder,public_denoiser,probabilities)
from research.calculations.client_geometry_audit import features,partition,split
from research.calculations.client_energy_one_release import local_query
from research.calculations.client_energy_filter_probe import scores,summarize

TRANSFORMS=('raw','prior','balanced_loss')
LAWS=('analytic_gaussian','radial_laplace')

def balanced_teacher(x,y,steps):
    theta=np.zeros((3,17))
    if not len(y):return theta
    prior=(np.bincount(y,minlength=4)+1)/(len(y)+4)
    weights=1/prior[y];weights/=weights.mean()
    for _ in range(steps):
        p=probabilities(x@theta.T);p[np.arange(len(y)),y]-=1
        theta-=.5*C.T@((p*weights[:,None]).T@x/len(y))
    return theta

def prepare(include_reserve=False):
    original=np.load(ROOT/'private_descriptor_analytic_gaussian.npz')
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    source=next(cache.rglob('fashion_mnist-train.arrow'));data=Dataset.from_file(str(source));labels=np.array(data['label'])
    ids={name:original[name+'_original_indices'] for name in ('public','private','development','evaluation','reserve')}
    public=features(data,ids['public']);private=features(data,ids['private']);development=(features(data,ids['development']),labels[ids['development']])
    anchors={k:original['anchors'+str(k)] for k in (4,16)}
    decoders={(k,r):ridge_decoder(x,r) for k,x in anchors.items() for r in RIDGES};denoisers={r:public_denoiser(public,r) for r in RIDGES}
    reserve=None;confirmation_ids=None
    if include_reserve:
        confirmation_ids=np.concatenate([ids['reserve'][labels[ids['reserve']]==k][:512] for k in range(4)])
        assert len(confirmation_ids)==2048 and len(set(confirmation_ids))==2048
        assert all(not set(confirmation_ids)&set(ids[name]) for name in ('public','private','development','evaluation'))
        reserve=(features(data,confirmation_ids),labels[confirmation_ids])
    return source,ids,public,private,labels[ids['private']],development,anchors,decoders,denoisers,reserve,confirmation_ids

def development():
    source,ids,public,x,y,dev,anchors,decoders,denoisers,_,_=prepare(False)
    rows=[];saved={};diagnostics=[]
    for seed in (42,43,44):
        fit,_=split(partition(y,'label_stress',seed),y,seed);weights=np.ones(8)/8
        counts=np.stack([np.bincount(y[i],minlength=4) for i in fit]);priors=(counts+1)/(counts.sum(axis=1,keepdims=True)+4)
        saved[f'seed{seed}_private_class_counts']=counts
        teachers={}
        for steps in STEPS:
            raw=np.stack([local_query(x[i],y[i],steps) for i in fit]);aligned=raw.copy();aligned[:,:,-1]-=np.log(priors)@C
            balanced=np.stack([balanced_teacher(x[i],y[i],steps) for i in fit])
            for name,value in (('raw',raw),('prior',aligned),('balanced_loss',balanced)):
                teachers[(name,steps)]=value;saved[f'seed{seed}_{name}_steps{steps}_teachers']=value
                for k in (4,16):
                    pred=probabilities(np.einsum('md,icd->imc',anchors[k],value));votes=pred.argmax(axis=-1)
                    hist=np.stack([(votes==c).sum(axis=0) for c in range(4)],axis=-1)
                    saved[f'seed{seed}_{name}_steps{steps}_prototypes{k}_vote_histograms']=hist
                    diagnostics.append({'seed':seed,'transform':name,'steps':steps,'prototypes':k,'vote_histograms':hist.tolist(),'mean_vote_max_fraction':float(hist.max(axis=1).mean()/8)})
        query={(transform,family,steps,k):encode(teachers[(transform,steps)],family,anchors.get(k),active_mask=[len(i)>0 for i in fit]) for transform in TRANSFORMS for family in FAMILIES for steps in STEPS for k in ((0,) if family=='model_direct' else (4,16))}
        for law in LAWS:
            selected={};candidates=[];noise_seed=seed*100000+4100
            for contract in CONTRACTS:
                for transform in TRANSFORMS:
                    for family in FAMILIES:
                        choices=[]
                        for j,config in enumerate(config_grid(family)):
                            objects,calibration=sanitize(query[(transform,family,config['steps'],config['prototypes'])],weights,config['radius'],8,contract,32,noise_seed,law=law)
                            theta=decode(objects,weights,config,contract,anchors,decoders,denoisers);ce,_=scores(*dev,theta)
                            record={'contract':contract,'transform':transform,'configuration':config,'development_ce':float(ce.mean())}
                            choices.append(record);candidates.append(record)
                            saved[f'seed{seed}_{law}_{contract}_{transform}_{family}_dev{j}']=ce
                        selected[contract+'_'+transform+'_'+family]=min(choices,key=lambda c:c['development_ce'])
            chosen={}
            for contract in CONTRACTS:
                candidates_vote=[contract+'_'+t+'_votes' for t in ('prior','balanced_loss')]
                candidate=min(candidates_vote,key=lambda n:selected[n]['development_ce'])
                controls=[contract+'_'+t+'_'+f for t in TRANSFORMS for f in FAMILIES if f!='votes']+[contract+'_raw_votes']
                reference=min(controls,key=lambda n:selected[n]['development_ce'])
                chosen[contract]={'candidate':candidate,'control':reference,'development_gain':selected[reference]['development_ce']-selected[candidate]['development_ce']}
            rows.append({'seed':seed,'law':law,'selected':selected,'chosen':chosen,'development_candidates':candidates})
            print('development',seed,law,[(c,round(v['development_gain'],5),v['candidate'],v['control']) for c,v in chosen.items()],flush=True)
    np.savez_compressed(ROOT/'private_prior_constructor_development.npz',**saved)
    (ROOT/'private_prior_constructor_development.json').write_text(json.dumps({'kind':'development-only adaptive followup; reserve not accessed; no end-to-end DP research selection','protocol':'research/proposals/2026-10-06_prior_constructor_protocol.md','source_hash':hashlib.sha256(source.read_bytes()).hexdigest(),'draws':32,'rows':rows,'vote_diagnostics':diagnostics},indent=2,allow_nan=False)+'\n')

def evaluation():
    protocol=json.loads((ROOT/'private_prior_constructor_development.json').read_text())
    stored=np.load(ROOT/'private_prior_constructor_development.npz')
    source,ids,public,x,y,dev,anchors,decoders,denoisers,reserve,confirmation_ids=prepare(True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['source_hash']
    saved={'confirmation_original_indices':confirmation_ids,'remaining_reserve_original_indices':np.setdiff1d(ids['reserve'],confirmation_ids)};rows=[]
    for row in protocol['rows']:
        seed=row['seed'];law=row['law'];weights=np.ones(8)/8
        teachers={(t,steps):stored[f'seed{seed}_{t}_steps{steps}_teachers'] for t in TRANSFORMS for steps in STEPS}
        query={(t,f,steps,k):encode(teachers[(t,steps)],f,anchors.get(k),active_mask=stored[f'seed{seed}_private_class_counts'].sum(axis=1)>0) for t in TRANSFORMS for f in FAMILIES for steps in STEPS for k in ((0,) if f=='model_direct' else (4,16))}
        for epsilon in (4.,8.,16.):
            key=f'seed{seed}_{law}_epsilon{int(epsilon)}';arms={}
            for arm,choice in row['selected'].items():
                c=choice['configuration'];objects,calibration=sanitize(query[(choice['transform'],c['family'],c['steps'],c['prototypes'])],weights,c['radius'],epsilon,choice['contract'],512,seed*100000+16000000+int(epsilon),law=law)
                theta=decode(objects,weights,c,choice['contract'],anchors,decoders,denoisers);ce,acc=scores(*reserve,theta)
                saved[key+'_'+arm+'_ce']=ce;saved[key+'_'+arm+'_accuracy']=acc
                arms[arm]={'configuration':c,'transform':choice['transform'],'development_ce_epsilon8':choice['development_ce'],'calibration':calibration,'ce':summarize(ce),'accuracy':summarize(acc)}
            contrasts={}
            for contract,choice in row['chosen'].items():
                candidate=choice['candidate'];references={'same_contract':choice['control'],'primary_peer':row['chosen']['aggregate']['control'],'central_same_votes':row['chosen']['central']['candidate']}
                for label,reference in references.items():
                    d=saved[key+'_'+reference+'_ce']-saved[key+'_'+candidate+'_ce'];stat=summarize(d)
                    contrasts[contract+'_'+label]={'candidate':candidate,'reference':reference,'gain':stat,'ci95':[stat['mean']-1.96*stat['se'],stat['mean']+1.96*stat['se']]}
            rows.append({'seed':seed,'law':law,'epsilon':epsilon,'arms':arms,'contrasts':contrasts})
            print('reserve',seed,law,epsilon,'A gain',round(contrasts['aggregate_same_contract']['gain']['mean'],5),'B vs A',round(contrasts['individual_primary_peer']['gain']['mean'],5),flush=True)
    pub=local_query(public,np.array(Dataset.from_file(str(source))['label'])[ids['public']],80);public_ce,public_acc=scores(*reserve,pub[None,:,:])
    np.savez_compressed(ROOT/'private_prior_constructor_evaluation.npz',**saved)
    (ROOT/'private_prior_constructor_evaluation.json').write_text(json.dumps({'kind':'fresh reserve/noise confirmation after development selections frozen; same privatebank/publicsupport, not population replication; tuning not DP','source_hash':protocol['source_hash'],'draws':512,'evaluation_count':2048,'remaining_reserve_count':len(saved['remaining_reserve_original_indices']),'labelled_public_control':{'ce':float(public_ce[0]),'accuracy':float(public_acc[0])},'rows':rows},indent=2,allow_nan=False)+'\n')

def checks():
    rng=np.random.default_rng(91);x=rng.normal(size=(24,17));y=np.tile(np.arange(4),6)
    # Equal priors make weighted teacher training equal conventional training.
    assert np.allclose(balanced_teacher(x,y,20),local_query(x,y,20))
    assert np.all(balanced_teacher(np.empty((0,17)),np.empty(0,dtype=int),20)==0)
    p=np.array([.7,.1,.1,.1]);prior=np.array([.7,.1,.1,.1]);aligned=p/prior;aligned/=aligned.sum()
    assert np.allclose(aligned,.25)
    print('balanced-loss identity, empty teacher, label-prior correction algebra passed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('development','evaluation'),default='development');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    checks()
    if not args.check:
        if args.stage=='development':development()
        else:evaluation()
