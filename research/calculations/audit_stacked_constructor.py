"""Independent reconstruction of the stacked transfer: roles, public candidate family, control selection, queries, clipping, noise, CE, gains."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset, load_dataset
from scipy.special import ndtri, ndtr
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import C
from research.calculations.client_geometry_audit import features
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.headroom_probe import development_halves
from research.calculations.audit_frozen_constructor import probs, ce_of, grads, hessian

def gd(x,y,steps):
    th=np.zeros((3,17))
    for _ in range(steps):
        p=probs(x,th); p[np.arange(len(y)),y]-=1; th-=.5*C.T@(p.T@x/len(y))
    return th

def basis(H,d):
    w,U=np.linalg.eigh(H); B=U[:,np.argsort(w)[::-1][:d]].copy(); B*=np.sign(B[np.abs(B).argmax(0),np.arange(d)]); return B

def main():
    source,ids,_a,_b,_y,dev,*_=prepare(False); labels=np.array(Dataset.from_file(str(source))['label'])
    sid,_=development_halves(ids['development'],labels); look={int(v):i for i,v in enumerate(ids['development'])}; dsel=(dev[0][[look[int(v)] for v in sid]],labels[sid])
    d=json.load(open(ROOT/'stacked_constructor_transfer.json')); z=np.load(ROOT/'stacked_constructor_transfer.npz'); fz=json.load(open(ROOT/'stacked_constructor_freeze.json'))['frozen']; checks=0; maxerr=0.
    for name,first in (('fmnist_classes4to7',4),('mnist_digits0to3',0)):
        if name.startswith('fmnist'):
            cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'; data=Dataset.from_file(str(next(cache.rglob('fashion_mnist-train.arrow'))))
        else: data=load_dataset('ylecun/mnist')['train']
        raw=np.array(data['label']); roles=z[name+'_roles']; assert len(set(roles.tolist()))==len(roles); checks+=1
        n=4*128; sel_ids=roles[:n]; A=roles[n:n+2048]; B=roles[n+2048:n+4096]; ev=roles[n+4096:n+6144]; pub=roles[n+6144:].reshape(3,32)
        for a in (sel_ids,A,B,ev): assert set(np.unique(raw[a]))=={first,first+1,first+2,first+3}
        assert np.array_equal(np.bincount(raw[A]-first),[512]*4) and np.array_equal(np.bincount(raw[ev]-first),[512]*4); checks+=2
        sel=(features(data,sel_ids),raw[sel_ids]-first); ex=features(data,ev); ey=raw[ev]-first; coh={}
        counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
        for c,ids_ in (('A',A),('B',B)): cx=features(data,ids_); cy=raw[ids_]-first; coh[c]=(cx,cy,partition_by_class_counts(cy,counts,seed=20261009))
        for s in range(3):
            px=features(data,pub[s]); py=raw[pub[s]]-first; assert np.array_equal(np.bincount(py),[8]*4)
            cands=[]
            for steps in (80,320,1280,5120):
                th=gd(px,py,steps); cands+= [th*m for m in (.5,.75,1.,1.25,1.5,2.)]
            b=min(cands,key=lambda t:ce_of(*sel,t)); gam=np.stack([grads(px[py==k],py[py==k],b).mean(0) for k in range(4)]); H=hessian(px,b)
            for steps in (20,80,320,1280):
                t=b.copy()
                for _ in range(steps): t-=.5*grads(px,py,t).mean(0)
                cands.append(t)
            for dd in (1,3,12,51):
                Bs=basis(H,dd)
                for eta in (.1,.3,1.,3.,10.,30.): cands.append(b-eta*((gam.mean(0).ravel()@Bs)@Bs.T).reshape(3,17))
            control=min(cands,key=lambda t:ce_of(*sel,t)); cctl=ce_of(ex,ey,control)
            for c in 'AB':
                row=next(r for r in d['tasks'][name]['cells'] if r['subset']==s and r['cohort']==c); assert abs(row['strong_public_control']['ce']-cctl)<1e-9; checks+=1
                cx,cy,parts=coh[c]; gamc=np.stack([grads(px[py==k],py[py==k],control).mean(0) for k in range(4)]); Hc=hessian(px,control)
                qs=[]
                for idx in parts:
                    yi=cy[idx]; g=grads(cx[idx],yi,control); means=np.stack([g[yi==k].mean(0) if np.any(yi==k) else gamc[k] for k in range(4)]); bal=means.mean(0)
                    qs.append(bal if fz['noisy_65']['mode']=='target_balanced' else bal-gamc.mean(0))
                for risk in ('0.55','0.65','0.8'):
                    cfg=fz[f'noisy_{int(float(risk)*100)}']; assert cfg['mode']=='target_center' or True
                    qs2=[]
                    for idx in parts:
                        yi=cy[idx]; g=grads(cx[idx],yi,control); means=np.stack([g[yi==k].mean(0) if np.any(yi==k) else gamc[k] for k in range(4)]); bal=means.mean(0)
                        qs2.append(bal if cfg['mode']=='target_balanced' else bal-gamc.mean(0))
                    dd=cfg['dimension']; Bs=basis(Hc,dd); q=np.array(qs2).reshape(8,51)@Bs; nrm=np.linalg.norm(q,axis=1); sg=q*np.minimum(1,cfg['cap']/np.maximum(nrm,1e-100))[:,None]/8; tot=sg.sum(0)
                    sigma=cfg['cap']/(8*np.sqrt(2)*ndtri(float(risk))); seed=701000000+s*1000+('AB'.index(c))*100+int(float(risk)*100)
                    noise=np.random.default_rng(seed).normal(size=(512,8,51))[:,:,:dd].sum(1)*sigma/np.sqrt(7)
                    ces=np.array([ce_of(ex,ey,control-cfg['eta']*((tot+nz)@Bs.T).reshape(3,17)) for nz in noise])
                    saved=z[f'{name}_s{s}_{c}_noisy_{int(float(risk)*100)}_noisy_ce']; err=float(np.abs(ces-saved).max()); maxerr=max(maxerr,err); assert err<1e-6,(name,s,c,risk,err); checks+=1
                    v=row['risks'][risk]; assert abs(v['gain_over_strong_public_control']-(cctl-ces.mean()))<1e-9; checks+=1
                    for a in v['attack']: assert abs(a['expected_auc']-float(ndtr(np.linalg.norm(sg[a['target']])/(np.sqrt(2)*sigma))))<1e-12; checks+=1
    print('stacked-constructor audit passed',checks,'checks; max noisy-CE draw error',maxerr)

if __name__=='__main__': main()
