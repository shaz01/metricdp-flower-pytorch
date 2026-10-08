"""Independent reconstruction of the frozen-constructor confirmation: roles, queries, clipping, noise, CE, gains."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from scipy.special import ndtri
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import C
from research.calculations.client_geometry_audit import features

def probs(x,b):
    z=(x@b.T)@C.T; z=z-z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)

def ce_of(x,y,theta):
    z=x@(C@theta).T; z=z-z.max(1,keepdims=True); return float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(len(y)),y]))

def grads(x,y,b):
    r=probs(x,b)-np.eye(4)[y]; return np.einsum('na,nd->nad',r@C,x)

def hessian(x,b):
    p=probs(x,b); s=-p[:,:,None]*p[:,None,:]; s[:,np.arange(4),np.arange(4)]+=p
    ident=np.einsum('ca,ncd,db->nab',C,s,C); return np.einsum('nab,ni,nj->aibj',ident,x,x).reshape(51,51)/len(x)

def main():
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    test=Dataset.from_file(str(next(cache.rglob('fashion_mnist-test.arrow')))); labels=np.array(test['label'])
    d=json.load(open(ROOT/'frozen_constructor_confirmation.json')); z=np.load(ROOT/'frozen_constructor_confirmation.npz'); fz=json.load(open(ROOT/'frozen_constructor_freeze.json'))
    pub=z['public_original_indices']; cohort=z['cohort_original_indices']; ev=z['evaluation_original_indices']; checks=0
    allids=np.concatenate([pub.ravel(),cohort,ev]); assert len(set(allids.tolist()))==len(allids) and allids.max()<10000; checks+=1
    assert np.array_equal(np.bincount(labels[cohort],minlength=4),[512]*4) and np.array_equal(np.bincount(labels[ev],minlength=4),[464]*4); checks+=1
    for s in range(3): assert np.array_equal(np.bincount(labels[pub[s]],minlength=4),[8]*4)
    ex=features(test,ev); ey=labels[ev]; cx=features(test,cohort); cy=labels[cohort]
    counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205; parts=partition_by_class_counts(cy,counts,seed=20261008)
    maxerr=0.
    for r in d['rows']:
        s=r['subset']; assert r['public_original_indices']==pub[s].tolist(); px=features(test,pub[s]); py=labels[pub[s]]; b=z[f'subset{s}_reference']
        gamma=np.stack([grads(px[py==k],py[py==k],b).mean(0) for k in range(4)]); H=hessian(px,b)
        # control and zero arms
        w,U=np.linalg.eigh(H)
        for risk_s,v_ in r['risks'].items():
            cfg=v_['configuration']; dd=cfg['dimension']; B=U[:,np.argsort(w)[::-1][:dd]].copy(); B*=np.sign(B[np.abs(B).argmax(0),np.arange(dd)])
            prior=np.ones(4)/4; qs=[]
            for idx in parts:
                yi=cy[idx]; g=grads(cx[idx],yi,b); means=np.stack([g[yi==k].mean(0) if np.any(yi==k) else gamma[k] for k in range(4)])
                qs.append(np.einsum('k,kad->ad',prior,means) if cfg['mode']=='target_balanced' else g.mean(0)-np.einsum('k,kad->ad',np.bincount(yi,minlength=4)/len(yi),gamma))
            q=np.array(qs).reshape(8,51)@B; n=np.linalg.norm(q,axis=1); sig_=q*np.minimum(1,cfg['cap']/np.maximum(n,1e-100))[:,None]/8; tot=sig_.sum(0)
            sigma=cfg['cap']/(8*np.sqrt(2)*ndtri(float(risk_s))); assert abs(sigma-v_['sigma'])<1e-15; checks+=1
            gm=gamma.mean(0); proj=((gm.ravel()@B)@B.T).reshape(3,17); offset=gm-proj if cfg['mode']=='target_balanced' else gm
            noise=np.random.default_rng(501000000+s*1000+int(float(risk_s)*100)).normal(size=(512,8,51))[:,:,:dd].sum(1)*sigma/np.sqrt(7)
            ces=np.array([ce_of(ex,ey,b-cfg['eta']*(offset+((tot+nz)@B.T).reshape(3,17))) for nz in noise])
            assert np.allclose(ces,z[f'subset{s}_risk{int(float(risk_s)*100)}_noisy_ce'],atol=1e-10); checks+=1
            maxerr=max(maxerr,abs(ces.mean()-v_['arms']['noisy']['ce']),float(np.abs(ces-z[f'subset{s}_risk{int(float(risk_s)*100)}_noisy_ce']).max()))
            assert abs(v_['gain_over_strong_public_control']-(r['strong_public_control']['ce']-ces.mean()))<1e-9 and abs(v_['gain_over_public_zero']-(r['frozen_public_zero']['ce']-ces.mean()))<1e-9; checks+=2
            for a in v_['attack']: assert abs(a['expected_auc']-float(__import__('scipy.special',fromlist=['ndtr']).ndtr(np.linalg.norm(sig_[a['target']])/(np.sqrt(2)*sigma))))<1e-12; checks+=1
        zc=fz['frozen']['public_zero']; gm=gamma.mean(0)
        zero=ce_of(ex,ey,b-zc['eta']*gm); assert abs(zero-r['frozen_public_zero']['ce'])<1e-9; checks+=1
    print('frozen-constructor audit passed',checks,'checks; max noisy-CE error',maxerr)

if __name__=='__main__': main()
