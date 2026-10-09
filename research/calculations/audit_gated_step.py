"""Independent reconstruction of the gated-step run on the fresh KMNIST tasks (budgets 32/128): roles, base selection, queries, noise, multiplier grid, gating."""
import json
import numpy as np
from datasets import load_dataset
from scipy.special import ndtri
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_geometry_audit import features
from research.calculations.audit_frozen_constructor import ce_of, grads, hessian
from research.calculations.audit_stacked_constructor import gd, basis

BUDGETS=(32,128,512); MULT=(0.,.25,.5,1.,2.); SEED=20261011

def main():
    data=load_dataset('tanganke/kmnist')['train']; raw=np.array(data['label'])
    d=json.load(open(ROOT/'gated_step.json')); z=np.load(ROOT/'gated_step.npz'); fz=json.load(open(ROOT/'stacked_constructor_freeze.json'))['frozen']; checks=0; maxerr=0.
    for name,first in (('kmnist_classes0to3',0),('kmnist_classes4to7',4)):
        roles=z[name+'_roles']; assert len(set(roles.tolist()))==len(roles); pos=0; pub={}
        for b in BUDGETS:
            for s in range(3): pub[(b,s)]=roles[pos:pos+b]; pos+=b
        val=roles[pos:pos+512]; pos+=512; A=roles[pos:pos+2048]; pos+=2048; B=roles[pos:pos+2048]; pos+=2048; ev=roles[pos:pos+2048]; assert pos+2048==len(roles)
        for ids in (val,A,B,ev): assert set(np.unique(raw[ids]))==set(range(first,first+4)); checks+=1
        for ids in (A,ev): assert np.array_equal(np.bincount(raw[ids]-first),[512]*4)
        vx=features(data,val); vy=raw[val]-first; ex=features(data,ev); ey=raw[ev]-first; coh={}
        counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
        for c,ids in (('A',A),('B',B)): cx=features(data,ids); cy=raw[ids]-first; coh[c]=(cx,cy,partition_by_class_counts(cy,counts,seed=SEED+first))
        sub=np.concatenate([np.where(vy==k)[0][:32] for k in range(4)])
        for b in (32,128):
            for s in range(3):
                ids=pub[(b,s)]; assert np.array_equal(np.bincount(raw[ids]-first),[b//4]*4); px=features(data,ids); py=raw[ids]-first
                cands=[]
                for steps in (80,320,1280,5120): th=gd(px,py,steps); cands+=[th*m for m in (.5,.75,1.,1.25,1.5,2.)]
                bb=min(cands,key=lambda t:ce_of(vx,vy,t)); gam=np.stack([grads(px[py==k],py[py==k],bb).mean(0) for k in range(4)]); H=hessian(px,bb)
                for steps in (20,80,320,1280):
                    t=bb.copy()
                    for _ in range(steps): t-=.5*grads(px,py,t).mean(0)
                    cands.append(t)
                for dd in (1,3,12,51):
                    Bs=basis(H,dd)
                    for eta in (.1,.3,1.,3.,10.,30.): cands.append(bb-eta*((gam.mean(0).ravel()@Bs)@Bs.T).reshape(3,17))
                control=min(cands,key=lambda t:ce_of(vx,vy,t)); cce=ce_of(ex,ey,control); gc=np.stack([grads(px[py==k],py[py==k],control).mean(0) for k in range(4)]); Hc=hessian(px,control)
                for c in 'AB':
                    row=next(r for r in d['tasks'][name] if r['budget']==b and r['subset']==s and r['cohort']==c); assert abs(row['control']['ce']-cce)<1e-9; checks+=1
                    cx,cy,parts=coh[c]; qs={m:[] for m in ('target_balanced','target_center')}
                    for idx in parts:
                        yi=cy[idx]; g=grads(cx[idx],yi,control); bal=np.stack([g[yi==k].mean(0) if np.any(yi==k) else gc[k] for k in range(4)]).mean(0); qs['target_balanced'].append(bal); qs['target_center'].append(bal-gc.mean(0))
                    for risk in ('0.55','0.65','0.8'):
                        cfg=fz[f'noisy_{int(float(risk)*100)}']; dd=cfg['dimension']; Bs=basis(Hc,dd); q=np.array(qs[cfg['mode']]).reshape(8,51)@Bs; nrm=np.linalg.norm(q,axis=1)
                        tot=(q*np.minimum(1,cfg['cap']/np.maximum(nrm,1e-100))[:,None]/8).sum(0); sigma=cfg['cap']/(8*np.sqrt(2)*ndtri(float(risk))); seed=901000000+b*1000+s*100+'AB'.index(c)*10+int(float(risk)*100)
                        noise=np.random.default_rng(seed).normal(size=(512,8,51))[:,:,:dd].sum(1)*sigma/np.sqrt(7)
                        thetas=[[control-cfg['eta']*m*((tot+nz)@Bs.T).reshape(3,17) for nz in noise] for m in MULT]
                        ece=np.array([[ce_of(ex,ey,t) for t in row_] for row_ in thetas]); saved=z[f'{name}_b{b}_s{s}_{c}_noisy_{int(float(risk)*100)}_eval_ce']; err=float(np.abs(ece-saved).max()); maxerr=max(maxerr,err); assert err<1e-6,(name,b,s,c,risk,err); checks+=1
                        for label,vs,sx in (('gated_val512',vx,vy),('gated_val128',vx[sub],vy[sub])):
                            vce=np.array([[ce_of(vs,sx,t) for t in row_] for row_ in thetas]); pick=vce.argmin(0); gated=ece[pick,np.arange(512)].mean()
                            assert abs(row['risks'][risk][label]['gain']-(cce-gated))<1e-6,(label,name,b,s,c,risk); checks+=1
                        assert abs(row['risks'][risk]['fixed_m1']['gain']-(cce-ece[3].mean()))<1e-6; checks+=1
    print('gated-step audit passed',checks,'checks; max eval-CE draw error',maxerr)

if __name__=='__main__': main()
