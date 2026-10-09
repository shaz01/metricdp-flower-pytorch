"""Audit of the CNN-head port: retrains the CNN base (shared code, so determinism is what is checked) and independently recomputes roles, head queries, Hessian, clipping, noise, gating and CE."""
import json
import numpy as np
from scipy.special import ndtri
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import C
from research.calculations.gated_step_probe import load
from research.calculations.cnn_head_probe import images, train, embed, head_theta, EPOCHS, LRS, BUDGETS, MULTIPLIERS, SEED
from research.calculations.audit_frozen_constructor import probs, ce_of, grads
from research.calculations.audit_stacked_constructor import basis

def hess(x,b):
    p=probs(x,b); s=-p[:,:,None]*p[:,None,:]; s[:,np.arange(4),np.arange(4)]+=p
    ident=np.einsum('ca,ncd,db->nab',C,s,C); n=3*x.shape[1]; return np.einsum('nab,ni,nj->aibj',ident,x,x).reshape(n,n)/len(x)

def main():
    data,raw=load('kmnist'); d=json.load(open(ROOT/'cnn_head.json')); z=np.load(ROOT/'cnn_head.npz'); fz=json.load(open(ROOT/'stacked_constructor_freeze.json'))['frozen']; checks=0; maxerr=0.
    for name,first in (('kmnist_classes0to3',0),('kmnist_classes4to7',4)):
        roles=z[name+'_roles']; assert len(set(roles.tolist()))==len(roles); pos=0; pub={}
        for b in BUDGETS:
            for s in range(3): pub[(b,s)]=roles[pos:pos+b]; pos+=b
        val=roles[pos:pos+512]; pos+=512; A=roles[pos:pos+2048]; pos+=2048; B=roles[pos:pos+2048]; pos+=2048; ev=roles[pos:pos+2048]; assert pos+2048==len(roles)
        for ids in (val,A,B,ev,*pub.values()): assert set(np.unique(raw[ids]))<=set(range(first,first+4))
        assert np.array_equal(np.bincount(raw[A]-first),[512]*4) and np.array_equal(np.bincount(raw[ev]-first),[512]*4); checks+=2
        vx=images(data,val); vy=raw[val]-first; ex=images(data,ev); ey=raw[ev]-first; counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
        coh={c:(images(data,ids),raw[ids]-first,partition_by_class_counts(raw[ids]-first,counts,seed=SEED+first)) for c,ids in (('A',A),('B',B))}
        for b in BUDGETS:
            for s in range(3):
                ids=pub[(b,s)]; px=images(data,ids); py=raw[ids]-first; cands=[]
                for e in EPOCHS:
                    for lr in LRS:
                        net=train(px,py,e,lr,SEED+b+s+e); th=head_theta(net); cands.append((ce_of(embed(net,vx),vy,th),net,th,{'epochs':e,'lr':lr}))
                _,net,theta0,cfg0=min(cands,key=lambda c:c[0]); fv=embed(net,vx); fe=embed(net,ex); fp=embed(net,px); cce=ce_of(fe,ey,theta0)
                gc=np.stack([grads(fp[py==k],py[py==k],theta0).mean(0) for k in range(4)]); H=hess(fp,theta0)
                for c in 'AB':
                    row=next(r for r in d['tasks'][name] if r['budget']==b and r['subset']==s and r['cohort']==c); assert row['base']==cfg0 and abs(row['control']['ce']-cce)<1e-6; checks+=2
                    cx,cy,parts=coh[c]; fc=embed(net,cx); qs=[]
                    for idx in parts:
                        yi=cy[idx]; g=grads(fc[idx],yi,theta0); bal=np.stack([g[yi==k].mean(0) if np.any(yi==k) else gc[k] for k in range(4)]).mean(0); qs.append(bal-gc.mean(0))
                    for risk in ('0.65','0.8'):
                        cfg=fz[f'noisy_{int(float(risk)*100)}']; assert cfg['mode']=='target_center'; dd=cfg['dimension']; Bs=basis(H,dd); q=np.array(qs).reshape(8,-1)@Bs; nrm=np.linalg.norm(q,axis=1)
                        tot=(q*np.minimum(1,cfg['cap']/np.maximum(nrm,1e-100))[:,None]/8).sum(0); sigma=cfg['cap']/(8*np.sqrt(2)*ndtri(float(risk))); seed=SEED+1000*b+100*s+10*'AB'.index(c)+int(float(risk)*100)
                        noise=np.random.default_rng(seed).normal(size=(512,8,51))[:,:,:dd].sum(1)*sigma/np.sqrt(7); D=theta0.shape[1]
                        deltas=[(( tot+nz)@Bs.T).reshape(3,D) for nz in noise]
                        vce=np.array([[ce_of(fv,vy,theta0-cfg['eta']*m*dl) for dl in deltas] for m in MULTIPLIERS]); ece=np.array([[ce_of(fe,ey,theta0-cfg['eta']*m*dl) for dl in deltas] for m in MULTIPLIERS])
                        pick=vce.argmin(0); g=ece[pick,np.arange(512)]; saved=z[f'{name}_b{b}_s{s}_{c}_noisy_{int(float(risk)*100)}_gated_ce']; err=float(np.abs(g-saved).max()); maxerr=max(maxerr,err); assert err<1e-5,(name,b,s,c,risk,err); checks+=1
                        assert abs(row['risks'][risk]['gated_gain']-(cce-g.mean()))<1e-5; checks+=1
    print('cnn-head audit passed',checks,'checks; max gated-CE draw error',maxerr)

if __name__=='__main__': main()
