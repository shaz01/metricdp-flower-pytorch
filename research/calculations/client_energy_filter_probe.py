"""Coupled noisy real-data energy-filter pilot; offline tuning is not DP."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from research.calculations.client_geometry_audit import features, pool, partition, split

T=20
ENERGIES=(.02,.08,.32,1.28,5.12)
SCHEDULES=('exp_-4','exp_-2','exp_0','exp_2','exp_4','block_5_.75','block_10_.5','block_15_.25')
POLICIES=tuple(f'{family}_{schedule}' for family in ('public','carry') for schedule in SCHEDULES)+('greedy','remaining','forecast','ema')
EPSILONS=(4.,8.,16.)

def contrast():
    b=np.zeros((4,3))
    for j in range(3):
        b[:j+1,j]=1/np.sqrt((j+1)*(j+2));b[j+1,j]=-(j+1)/np.sqrt((j+1)*(j+2))
    return b

C=contrast()

def clip(u, cap):
    norms=np.linalg.norm(u,axis=(-2,-1))
    factor=np.minimum(1,np.divide(cap,norms,out=np.ones_like(norms),where=norms>0))
    return u*factor[...,None,None]

def scores(x,y,theta):
    w=np.einsum('ca,nad->ncd',C,theta)
    z=np.einsum('md,ncd->nmc',x,w)
    top=z.max(axis=2);logsum=top+np.log(np.exp(z-top[:,:,None]).sum(axis=2))
    ce=(logsum-z[:,np.arange(len(y)),y]).mean(axis=1)
    acc=(z.argmax(axis=2)==y).mean(axis=1)
    return ce,acc

def simulate(clients,weights,energy,epsilon,policy,draws,seed,eval_sets,audit_set=None):
    rng=np.random.default_rng(seed);theta=np.zeros((draws,3,17))
    cap=energy/64;used=np.zeros((draws,8));previous=np.zeros_like(used);trend=np.ones_like(used)
    rho=(np.sqrt(np.log(1e5)+epsilon)-np.sqrt(np.log(1e5)))**2
    variance=cap/(2*rho);std=np.sqrt(variance/7)
    allocations=np.ones(T)
    if policy.startswith(('public','carry')):
        fields=policy.split('_')
        if fields[1]=='exp':allocations=np.exp(float(fields[2])*np.linspace(0,1,T))
        else:
            k=int(fields[2]);fraction=float(fields[3]);allocations=np.r_[np.full(k,fraction/k),np.full(T-k,(1-fraction)/(T-k))]
    allocations/=allocations.sum();trace=[];zeros=np.zeros_like(used);ema=np.zeros_like(used);norm_trace=[];signal_trace=[];alignment_trace=[];linear_trace=[]
    for t in range(T):
        w=np.einsum('ca,nad->ncd',C,theta);updates=[]
        for x,y in clients:
            if not len(y):
                updates.append(np.zeros_like(theta));continue
            z=np.einsum('md,ncd->nmc',x,w);z-=z.max(axis=2,keepdims=True)
            p=np.exp(z);p/=p.sum(axis=2,keepdims=True);p[:,np.arange(len(y)),y]-=1
            g=np.einsum('nmc,md->ncd',p,x)/len(y)
            updates.append(-.5*np.einsum('ca,ncd->nad',C,g))
        proposed=np.stack(updates,axis=1)*weights[None,:,None,None]
        norms=np.linalg.norm(proposed,axis=(2,3));remaining=np.maximum(cap-used,0)
        if policy.startswith('public'):allow=np.full_like(used,cap*allocations[t])
        elif policy.startswith('carry'):allow=remaining*allocations[t]/allocations[t:].sum()
        elif policy=='greedy':allow=remaining
        elif policy=='remaining':allow=remaining/(T-t)
        elif policy=='ema':
            squared=norms*norms
            if t==0:ema=squared.copy()
            allow=remaining*np.divide(squared,squared+(T-t-1)*ema,out=np.zeros_like(squared),where=(squared+(T-t-1)*ema)>0)
            ema=.5*ema+.5*squared
        else:
            ratio=np.divide(norms,previous,out=np.ones_like(norms),where=previous>1e-15)
            if t:trend=np.clip(.5*trend+.5*ratio,.5,1.5)
            denominator=np.sum(trend[:,:,None]**(2*np.arange(T-t)),axis=2)
            allow=remaining/denominator
        signal=clip(proposed,np.sqrt(np.minimum(allow,remaining)))
        spent=np.sum(signal*signal,axis=(2,3));used+=spent;zeros+=spent<1e-24
        norm_trace.append(norms.copy());signal_trace.append(np.sqrt(spent))
        if audit_set is not None:
            ax,ay=audit_set;az=np.einsum('md,ncd->nmc',ax,w);az-=az.max(axis=2,keepdims=True)
            ap=np.exp(az);ap/=ap.sum(axis=2,keepdims=True);ap[:,np.arange(len(ay)),ay]-=1
            ag=np.einsum('ca,ncd->nad',C,np.einsum('nmc,md->ncd',ap,ax)/len(ay))
            product=-np.sum(proposed*ag[:,None,:,:],axis=(2,3))
            denominator=norms*np.linalg.norm(ag,axis=(1,2))[:,None]
            alignment_trace.append(np.divide(product,denominator,out=np.zeros_like(product),where=denominator>0))
            linear_trace.append(-np.sum(signal*ag[:,None,:,:],axis=(2,3)))
        assert np.all(used<=cap*(1+1e-12))
        # Explicit fixed-slot weighted shares, even when a signal is zero.
        noise=rng.normal(size=(draws,8,3,17))*std
        theta+=np.sum(signal+noise,axis=1)
        previous=norms;trace.append(used.mean(axis=0)/cap)
    out={name:scores(x,y,theta) for name,(x,y) in eval_sets.items()}
    return out,{'spent_fraction':used/cap,'zero_signal_rounds':zeros,'mean_spent_trace':np.array(trace),'proposed_norms':np.array(norm_trace),'signal_norms':np.array(signal_trace),'development_alignment':np.array(alignment_trace),'development_linear_descent':np.array(linear_trace)},theta

def summarize(a):
    return {'mean':float(np.mean(a)),'se':float(np.std(a,ddof=1)/np.sqrt(len(a)))}

def checks():
    rng=np.random.default_rng(10);u=rng.normal(size=(32,8,3,17));cap=rng.random((32,8))
    v=clip(u,cap);assert np.all(np.linalg.norm(v,axis=(2,3))<=cap+1e-12)
    assert np.array_equal(clip(u,np.zeros_like(cap)),np.zeros_like(u))
    assert np.allclose(C.T@C,np.eye(3));assert np.allclose(C.sum(axis=0),0)
    clients=[(rng.normal(size=(12,17)),rng.integers(0,4,12)) for _ in range(8)]
    for policy in POLICIES:
        _,diagnostic,model=simulate(clients,np.ones(8)/8,.02,8,policy,4,11,{})
        assert np.max(diagnostic['spent_fraction'])<=1+1e-12 and np.isfinite(model).all()
    clients[0]=(np.empty((0,17)),np.empty(0,dtype=int))
    _,diag,model=simulate(clients,np.ones(8)/8,.02,8,'greedy',4,11,{})
    assert np.all(diag['spent_fraction'][:,0]==0) and np.all(diag['zero_signal_rounds'][:,0]==T)
    # Fresh Gaussian shares continue despite the dummy's zero signal.
    empty=[(np.empty((0,17)),np.empty(0,dtype=int)) for _ in range(8)]
    _,all_dummy,model=simulate(empty,np.ones(8)/8,.02,8,'greedy',4,11,{})
    assert np.all(all_dummy['spent_fraction']==0) and np.linalg.norm(model)>0
    x=rng.normal(size=(23,17));y=rng.integers(0,4,23);theta=rng.normal(size=(1,3,17))*.1
    z=x@(C@theta[0]).T;z-=z.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True);p[np.arange(len(y)),y]-=1
    gradient=C.T@(p.T@x/len(y));direction=rng.normal(size=(1,3,17));h=1e-5
    finite_difference=(scores(x,y,theta+h*direction)[0]-scores(x,y,theta-h*direction)[0])/(2*h)
    assert np.allclose(finite_difference,np.sum(gradient*direction[0]),rtol=1e-7,atol=1e-8)
    # Algebraic unknown-noise floor and epsilon inversion.
    for epsilon in EPSILONS:
        rho=(np.sqrt(np.log(1e5)+epsilon)-np.sqrt(np.log(1e5)))**2
        assert np.isclose(rho+2*np.sqrt(rho*np.log(1e5)),epsilon)
    return 'radial/zero clipping, pathwise cap all policies, contrast gauge, finite causal trajectories, accounting inversion'

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    verified=checks()
    if args.check:print(verified);return
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    train_path=next(cache.rglob('fashion_mnist-train.arrow'));test_path=next(cache.rglob('fashion_mnist-test.arrow'))
    train=Dataset.from_file(str(train_path));test=Dataset.from_file(str(test_path))
    labels=np.array(test['label']);ids=np.concatenate([np.flatnonzero(labels==k)[768:1000] for k in range(4)])
    assert len(ids)==928
    held=(features(test,ids),labels[ids]);rows=[];saved={'heldout_original_indices':ids}
    for data_seed in (42,43,44):
        poolids,devids,yp,yd=pool(train,data_seed);xp=features(train,poolids);development=(features(train,devids),yd)
        for mode in ('balanced','quantity','label_stress'):
            fit,_=split(partition(yp,mode,data_seed),yp,data_seed)
            clients=[(xp[a],yp[a]) for a in fit]
            weights=np.arange(1,9)/36 if mode=='quantity' else np.ones(8)/8
            for epsilon in EPSILONS:
                seed=data_seed*10000+int(epsilon)*100
                candidates=[]
                for energy in ENERGIES:
                    for policy in POLICIES:
                        out,_,_=simulate(clients,weights,energy,epsilon,policy,16,seed,{'development':development})
                        candidates.append({'policy':policy,'energy':energy,'development_ce':float(out['development'][0].mean())})
                public=min([c for c in candidates if c['policy'].startswith(('public','carry'))],key=lambda c:c['development_ce'])
                selected={'public':public}
                for family in ('greedy','remaining','forecast','ema'):
                    selected[family+'_tuned']=min([c for c in candidates if c['policy']==family],key=lambda c:c['development_ce'])
                    selected[family+'_matched']=next(c for c in candidates if c['policy']==family and c['energy']==public['energy'])
                results={};diagnostics={}
                for arm,config in selected.items():
                    out,diag,theta=simulate(clients,weights,config['energy'],epsilon,config['policy'],128,seed+700000,{'heldout':held})
                    results[arm]=out['heldout'];diagnostics[arm]=diag
                    key=f'{mode}_seed{data_seed}_epsilon{int(epsilon)}_{arm}'
                    saved[key+'_ce']=out['heldout'][0];saved[key+'_accuracy']=out['heldout'][1]
                    saved[key+'_spent_fraction']=diag['spent_fraction'];saved[key+'_zero_signal_rounds']=diag['zero_signal_rounds'];saved[key+'_mean_spent_trace']=diag['mean_spent_trace']
                arms={}
                for arm,(ce,acc) in results.items():
                    gain=results['public'][0]-ce;stats=summarize(gain)
                    arms[arm]={'configuration':selected[arm],'ce':summarize(ce),'accuracy':summarize(acc),
                               'gain_vs_public':stats,'gain_ci95':[stats['mean']-1.96*stats['se'],stats['mean']+1.96*stats['se']],
                               'mean_spent_fraction':float(diagnostics[arm]['spent_fraction'].mean()),
                               'mean_zero_signal_rounds':float(diagnostics[arm]['zero_signal_rounds'].mean())}
                raw,_,_=simulate(clients,weights,1e6,1e20,'greedy',1,seed,{'heldout':held})
                rows.append({'partition':mode,'seed':data_seed,'epsilon':epsilon,'weights':weights.tolist(),'arms':arms,
                             'development_candidates':candidates,'near_noiseless_raw_reference_ce':float(raw['heldout'][0][0])})
                print(mode,data_seed,epsilon,'public',round(arms['public']['ce']['mean'],4),'matched forecast gain',round(arms['forecast_matched']['gain_vs_public']['mean'],5),flush=True)
    target=Path('results/client_specific_noise');np.savez_compressed(target/'client_energy_filter_probe.npz',**saved)
    report={'kind':'coupled noisy trajectory utility feasibility; offline tuning not end-to-end DP; no CIA claim',
            'protocol':'research/proposals/2026-10-06_client_energy_filter_protocol.md','checks':verified,
            'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path,test_path)},
            'rounds':T,'development_draws':16,'evaluation_draws':128,'heldout_count':len(ids),'gate':.001,'rows':rows}
    (target/'client_energy_filter_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
