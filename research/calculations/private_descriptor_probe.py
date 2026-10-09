"""One-time descriptor and reusable-distribution research pilot, not DP deployment."""
from __future__ import annotations
import argparse
import hashlib
import json
from functools import lru_cache
from scipy.special import log_ndtr
from pathlib import Path
import numpy as np
from datasets import Dataset
from sklearn.cluster import KMeans
from research.calculations.client_geometry_audit import features,partition,split
from research.calculations.client_energy_filter_probe import C,scores,summarize
from research.calculations.client_energy_one_release import local_query

RADII=(.1,.2,.4,.8,1.6,3.2)
RIDGES=(0.,.01,.1)
STEPS=(20,80)
FAMILIES=('model_direct','model_distribution','logits','probabilities','votes')
CONTRACTS=('aggregate','individual','central')
ROOT=Path('results/client_specific_noise')

def simplex(v):
    ordered=np.sort(v,axis=-1)[...,::-1]
    cumulative=np.cumsum(ordered,axis=-1)-1
    positive=ordered-cumulative/np.arange(1,5)>0
    count=positive.sum(axis=-1)
    threshold=np.take_along_axis(cumulative,(count-1)[...,None],axis=-1)[...,0]/count
    return np.maximum(v-threshold[...,None],0)

def probabilities(contrasts):
    logits=contrasts@C.T;logits-=logits.max(axis=-1,keepdims=True)
    p=np.exp(logits);return p/p.sum(axis=-1,keepdims=True)

def encode(teachers,family,anchors=None,active_mask=None):
    if active_mask is not None:
        result=encode(teachers,family,anchors)
        result[~np.asarray(active_mask,dtype=bool)]=0
        return result
    if family.startswith('model'):return teachers.reshape(8,51)
    contrasts=np.einsum('md,icd->imc',anchors,teachers)
    if family=='logits':return contrasts.reshape(8,-1)/np.sqrt(len(anchors))
    p=probabilities(contrasts)
    if family=='votes':p=np.eye(4)[p.argmax(axis=-1)]
    return (p@C).reshape(8,-1)/np.sqrt(len(anchors))

def ridge_decoder(x,ridge):
    gram=x.T@x/len(x)
    return np.linalg.pinv(gram+ridge*np.eye(17),rcond=1e-12)@x.T/len(x)

def public_denoiser(x,ridge):
    if ridge==0:return np.eye(17)
    gram=x.T@x/len(x);return np.linalg.solve(gram+ridge*np.eye(17),gram)

def config_grid(family):
    for steps in STEPS:
        for radius in RADII:
            for ridge in RIDGES:
                for k in ((0,) if family=='model_direct' else (4,16)):
                    yield {'family':family,'steps':steps,'radius':radius,'ridge':ridge,'prototypes':k}

@lru_cache(maxsize=None)
def analytic_gaussian_scale(epsilon):
    def delta(scale):
        a=log_ndtr(1/(2*scale)-epsilon*scale)
        b=epsilon+log_ndtr(-1/(2*scale)-epsilon*scale)
        return float(np.exp(a)*(-np.expm1(min(b-a,0))))
    low=1e-8;high=1.
    while delta(high)>1e-5:high*=2
    for _ in range(80):
        middle=(low+high)/2
        if delta(middle)>1e-5:low=middle
        else:high=middle
    return high

def sanitize(query,weights,radius,epsilon,contract,draws,seed,law='zcdp'):
    weighted=query*weights[:,None];norms=np.linalg.norm(weighted,axis=1);cap=radius/8
    factor=np.minimum(1,np.divide(cap,norms,out=np.ones_like(norms),where=norms>0))
    signal=weighted*factor[:,None]
    rho=(np.sqrt(np.log(1e5)+epsilon)-np.sqrt(np.log(1e5)))**2
    variance=cap*cap/(2*rho) if law=='zcdp' else cap*cap*analytic_gaussian_scale(epsilon)**2
    # Central uses eight SERVER-owned innovations; none belongs to the peer.
    divisor={'aggregate':7,'individual':1,'central':8}[contract]
    rng=np.random.default_rng(seed);d=query.shape[1]
    normals=rng.normal(size=(draws,8,d))
    if law=='radial_laplace':
        scale=cap/epsilon;alpha=(d+1)/2
        gamma=rng.gamma(alpha/divisor,2*scale*scale,size=(draws,8,1))
        noise=normals*np.sqrt(gamma);variance=(d+1)*scale*scale
    else:noise=normals*np.sqrt(variance/divisor)
    return signal[None,:,:]+noise,{'law':law,'dimension':d,'epsilon':epsilon,'delta':0. if law=='radial_laplace' else 1e-5,'rho_reference_only':rho,'weighted_cap':cap,'unknown_floor_variance':variance,
                                   'aggregate_variance':8*variance/divisor,'clip_factors':factor.tolist()}

def decode(objects,weights,config,contract,anchors,decoders,denoisers):
    family=config['family'];k=config['prototypes'];ridge=config['ridge']
    if family=='model_direct':
        return objects.sum(axis=1).reshape(-1,3,17)@denoisers[ridge]
    if family=='model_distribution':
        if contract=='individual':
            teachers=objects.reshape(-1,8,3,17)/weights[None,:,None,None]
            p=probabilities(np.einsum('md,nicd->nimc',anchors[k],teachers))
            targets=np.sum(p*weights[None,:,None,None],axis=1)
        else:
            teacher=objects.sum(axis=1).reshape(-1,3,17)
            targets=probabilities(np.einsum('md,ncd->nmc',anchors[k],teacher))
    else:
        if contract=='individual':
            latent=objects.reshape(-1,8,k,3)*np.sqrt(k)/weights[None,:,None,None]
            p=probabilities(latent) if family=='logits' else simplex(.25+latent@C.T)
            targets=np.sum(p*weights[None,:,None,None],axis=1)
        else:
            latent=objects.sum(axis=1).reshape(-1,k,3)*np.sqrt(k)
            targets=probabilities(latent) if family=='logits' else simplex(.25+latent@C.T)
    # Exact conditional-label mixture on public centroids; finite sampling omitted.
    log_targets=np.log(np.maximum(targets,.01))@C
    return np.einsum('dm,nmc->ncd',decoders[(k,ridge)],log_targets)

def data_split(train):
    previous=np.load(ROOT/'client_geometry_audit_matrices.npz')
    old=set()
    for seed in (42,43,44):
        old.update(previous[f'seed{seed}_pool_original_indices'].tolist())
        old.update(previous[f'seed{seed}_development_original_indices'].tolist())
    labels=np.array(train['label']);rng=np.random.default_rng(20261006)
    roles={name:[] for name in ('public','development','evaluation','private','reserve')}
    for klass in range(4):
        available=np.array([i for i in np.flatnonzero(labels==klass) if i not in old]);rng.shuffle(available)
        start=0
        for name,size in (('public',128),('development',256),('evaluation',512),('private',1024)):
            roles[name].extend(available[start:start+size]);start+=size
        roles['reserve'].extend(available[start:])
    roles={name:np.array(ids,dtype=int) for name,ids in roles.items()}
    used=set()
    for name,ids in roles.items():
        assert len(set(ids))==len(ids) and not set(ids)&old and not set(ids)&used
        used.update(ids.tolist())
    assert len(old)==12280 and len(used)==11720
    return roles,labels

def checks():
    rng=np.random.default_rng(17);v=rng.normal(size=(12,8,16,4));p=simplex(v)
    assert np.all(p>=0) and np.allclose(p.sum(axis=-1),1)
    valid=rng.dirichlet(np.ones(4),size=32)
    assert np.allclose(simplex(valid),valid)
    teachers=rng.normal(size=(8,3,17));anchors=rng.normal(size=(16,17))
    for family in ('probabilities','votes'):
        q=encode(teachers,family,anchors)
        assert np.max(np.linalg.norm(q,axis=1))<=np.sqrt(.75)+1e-12
    masked=encode(np.zeros((8,3,17)),'votes',anchors,active_mask=[False]+[True]*7)
    assert np.all(masked[0]==0)
    q=np.zeros((8,48));w=np.ones(8)/8
    objects,info=sanitize(q,w,.4,8,'individual',32,8)
    assert np.std(objects)>0 and np.allclose(info['clip_factors'],1)
    for contract in CONTRACTS:
        _,info=sanitize(encode(teachers,'model_direct'),w,.4,8,contract,32,8)
        assert np.isclose(info['rho_reference_only']+2*np.sqrt(info['rho_reference_only']*np.log(1e5)),8)
        assert np.all(np.array(info['clip_factors'])<=1)
    x=rng.normal(size=(30,17));theta=rng.normal(size=(3,17))*.01
    exact=(x@theta.T)@C.T
    recovered=ridge_decoder(x,0)@(exact@C)
    assert np.allclose(recovered,theta.T)
    assert np.allclose([analytic_gaussian_scale(e)**2 for e in (4.,8.,16.)],[1.168910944858024,.3602749391128144,.1184581022863902],rtol=1e-12)
    radial,_=sanitize(np.zeros((8,12)),w,.8,8,'individual',10000,31,law='radial_laplace')
    radii=np.linalg.norm(radial,axis=-1);s=.1/8
    assert abs(radii.mean()/(12*s)-1)<.01
    assert abs(np.mean(radial**2)/(13*s*s)-1)<.02
    return 'simplex feasibility/identity, whole probability/vote bound, dummy noise, joint clipping/accounting, exact identifiable logit interpolation, split disjointness'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--law',choices=('zcdp','analytic_gaussian','radial_laplace'),default='zcdp');args=parser.parse_args()
    verified=checks()
    if args.check:print(verified);return
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    source=next(cache.rglob('fashion_mnist-train.arrow'));train=Dataset.from_file(str(source))
    roles,labels=data_split(train);xs={name:features(train,ids) for name,ids in roles.items() if name!='reserve'}
    public=xs['public'];private=xs['private'];yp=labels[roles['private']]
    dev=(xs['development'],labels[roles['development']]);held=(xs['evaluation'],labels[roles['evaluation']])
    anchors={k:KMeans(n_clusters=k,n_init=10,random_state=20261006).fit(public).cluster_centers_ for k in (4,16)}
    decoders={(k,r):ridge_decoder(x,r) for k,x in anchors.items() for r in RIDGES}
    denoisers={r:public_denoiser(public,r) for r in RIDGES}
    saved={name+'_original_indices':ids for name,ids in roles.items()}
    saved.update({f'anchors{k}':x for k,x in anchors.items()})
    public_teacher=local_query(public,labels[roles['public']],80)
    public_ce,public_acc=scores(*held,public_teacher[None,:,:])
    rows=[]
    for seed in (42,43,44):
        for mode in ('balanced','quantity','label_stress'):
            fit,_=split(partition(yp,mode,seed),yp,seed)
            weights=np.arange(1,9)/36 if mode=='quantity' else np.ones(8)/8
            teachers={steps:np.stack([local_query(private[a],yp[a],steps) for a in fit]) for steps in STEPS}
            saved[f'{mode}_seed{seed}_fit_pool_indices']=np.concatenate(fit)
            queries={(family,steps,k):encode(teachers[steps],family,anchors.get(k),active_mask=[len(a)>0 for a in fit]) for family in FAMILIES for steps in STEPS for k in ((0,) if family=='model_direct' else (4,16))}
            development=[];selected={};noise_seed=seed*100000+{'balanced':100,'quantity':200,'label_stress':300}[mode]
            for contract in CONTRACTS:
                for family in FAMILIES:
                    candidates=[]
                    for j,config in enumerate(config_grid(family)):
                        objects,_=sanitize(queries[(family,config['steps'],config['prototypes'])],weights,config['radius'],8,contract,16,noise_seed,law=args.law)
                        theta=decode(objects,weights,config,contract,anchors,decoders,denoisers)
                        ce,_=scores(*dev,theta);record={'contract':contract,'configuration':config,'development_ce':float(ce.mean())}
                        candidates.append(record);development.append(record)
                        saved[f'{mode}_seed{seed}_{contract}_{family}_dev{j}']=ce
                    selected[contract+'_'+family]=min(candidates,key=lambda c:c['development_ce'])
            # Selection is frozen before any held-out/private-family utility evaluation.
            raw_configs=[]
            for family in FAMILIES:
                for steps in STEPS:
                    for ridge in RIDGES:
                        for k in ((0,) if family=='model_direct' else (4,16)):
                            config={'family':family,'steps':steps,'ridge':ridge,'prototypes':k}
                            objects=(queries[(family,steps,k)]*weights[:,None])[None,:,:]
                            theta=decode(objects,weights,config,'individual' if family!='model_direct' else 'aggregate',anchors,decoders,denoisers)
                            ce,_=scores(*dev,theta);raw_configs.append((float(ce[0]),config,theta))
            raw=min(raw_configs,key=lambda v:v[0]);raw_ce,raw_acc=scores(*held,raw[2])
            for epsilon in (4.,8.,16.):
                arms={};key=f'{mode}_seed{seed}_epsilon{int(epsilon)}'
                for arm,choice in selected.items():
                    config=choice['configuration'];contract=choice['contract']
                    objects,calibration=sanitize(queries[(config['family'],config['steps'],config['prototypes'])],weights,config['radius'],epsilon,contract,256,noise_seed+8000000+int(epsilon),law=args.law)
                    theta=decode(objects,weights,config,contract,anchors,decoders,denoisers)
                    ce,acc=scores(*held,theta);saved[key+'_'+arm+'_ce']=ce;saved[key+'_'+arm+'_accuracy']=acc
                    arms[arm]={'configuration':config,'development_ce_epsilon8':choice['development_ce'],'calibration':calibration,'ce':summarize(ce),'accuracy':summarize(acc)}
                control_families=('model_direct','model_distribution','logits')
                for contract in ('aggregate','individual'):
                    baseline=min((contract+'_'+f for f in control_families),key=lambda a:arms[a]['development_ce_epsilon8'])
                    for family in ('probabilities','votes'):
                        arm=contract+'_'+family
                        for label,reference in (('same_contract',baseline),('primary_peer',min(('aggregate_'+f for f in control_families),key=lambda a:arms[a]['development_ce_epsilon8']))):
                            gain=saved[key+'_'+reference+'_ce']-saved[key+'_'+arm+'_ce'];stat=summarize(gain)
                            arms[arm][label+'_contrast']={'reference':reference,'gain':stat,'ci95':[stat['mean']-1.96*stat['se'],stat['mean']+1.96*stat['se']]}
                rows.append({'partition':mode,'seed':seed,'epsilon':epsilon,'weights':weights.tolist(),'arms':arms,'development_candidates_epsilon8':development,
                             'raw_reference':{'configuration':raw[1],'ce':float(raw_ce[0]),'accuracy':float(raw_acc[0])}})
                print(mode,seed,epsilon,'A probability gain',round(arms['aggregate_probabilities']['same_contract_contrast']['gain']['mean'],5),'B probability vs A',round(arms['individual_probabilities']['primary_peer_contrast']['gain']['mean'],5),flush=True)
    filename='private_descriptor_probe' if args.law=='zcdp' else 'private_descriptor_'+args.law
    np.savez_compressed(ROOT/(filename+'.npz'),**saved)
    report={'law':args.law,'kind':'one-time descriptor/distribution utility pilot; fixed ideal channel certificates, offline selection/publication not DP; no CIA claim',
            'protocol':'research/proposals/2026-10-06_private_descriptor_protocol.md','source_hash':hashlib.sha256(source.read_bytes()).hexdigest(),'checks':verified,
            'data_counts':{name:len(ids) for name,ids in roles.items()},'development_draws':16,'evaluation_draws':256,
            'labelled_public_control':{'kind':'more auxiliary label information than core contract','steps':80,'ce':float(public_ce[0]),'accuracy':float(public_acc[0])},
            'uniform_control':{'ce':float(np.log(4)),'accuracy':.25},'rows':rows}
    (ROOT/(filename+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
