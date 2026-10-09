"""Independent recomputation of limited-public development rows (own CE, clip, noise, gains)."""
import json
import numpy as np
from datasets import Dataset
from scipy.special import ndtri
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import C
from research.calculations.headroom_probe import development_halves
from research.calculations.public_residual_probe import public_hessian

def ce(x,y,theta):
    z=x@np.einsum('ca,ad->cd',C,theta).T  # (m,4)
    z=z-z.max(1,keepdims=True); return float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(len(y)),y]))

def basis(h,d):
    v,u=np.linalg.eigh(h); b=u[:,np.argsort(v)[::-1][:d]].copy(); b*=np.sign(b[np.abs(b).argmax(0),np.arange(d)]); return b

def main():
    source,ids,_p,_q,_y,dev,*_=prepare(False); labels=np.array(Dataset.from_file(str(source))['label'])
    sid,aid=development_halves(ids['development'],labels); look={int(v):i for i,v in enumerate(ids['development'])}
    sel=(dev[0][[look[int(v)] for v in sid]],labels[sid]); ass=(dev[0][[look[int(v)] for v in aid]],labels[aid])
    d=json.load(open(ROOT/'limited_public_development.json')); bank=np.load(ROOT/'headroom_development.npz'); checks=0; maxerr=0.
    rng0=np.random.default_rng(7)
    for r in d['rows']:
        prefix=f"budget{r['budget']}_seed{r['subset_seed']}"; b=bank[prefix+'_reference']; gamma=bank[prefix+'_gamma']; h=bank[prefix+'_hessian']
        ci='AB'.index(r['cohort']); base=301000000+r['budget']*10000+r['subset_seed']*100+ci
        # selection minimality over every saved candidate, per (stage,risk,mode)
        groups={}
        for c in r['candidates']:
            k=c['configuration']; groups.setdefault((k['stage'],k['risk'],k['mode']),[]).append(c)
        for key,cs in groups.items():
            chosen=r['all_selected']['|'.join(str(x) for x in key)]
            assert chosen['selection_ce']==min(c['selection_ce'] for c in cs); checks+=1
        def total_and_theta(cfg,draws,seed):
            dd=cfg['dimension']; B=basis(h,dd); q=bank[f"{prefix}_cohort{r['cohort']}_{cfg['mode']}"].reshape(8,51)@B
            off=gamma.mean(0) if cfg['mode'] in ('class_center',) else gamma.mean(0)
            off=off if cfg['mode']=='class_center' else off-((gamma.mean(0).ravel()@B)@B.T).reshape(3,17)
            if cfg['stage']=='public_zero': tot=np.zeros(dd)
            else:
                cap=cfg['cap'] or 1e9; n=np.linalg.norm(q,axis=1); s=q*np.minimum(1,cap/np.maximum(n,1e-100))[:,None]/8; tot=s.sum(0)
            tot=np.atleast_2d(tot)
            if cfg['stage']=='noisy':
                sigma=cfg['cap']/(8*np.sqrt(2)*ndtri(cfg['risk'])); tot=tot+np.random.default_rng(seed).normal(size=(draws,8,51))[:,:,:dd].sum(1)*sigma/np.sqrt(7)
            return [b-cfg['eta']*(off+(t@B.T).reshape(3,17)) for t in tot]
        for name,v in r['all_selected'].items():
            cfg=v['configuration']; draws=32 if False else None
            for kind,data,dr,seed in (('selection',sel,32,base),('assessment',ass,128,base+5000000)):
                thetas=total_and_theta(cfg,dr,seed); val=float(np.mean([ce(*data,t) for t in thetas]))
                err=abs(val-v[kind+'_ce']); maxerr=max(maxerr,err); assert err<1e-9,(name,kind,err); checks+=1
        # recompute 25 random candidates' selection scores
        for c in [r['candidates'][i] for i in rng0.choice(len(r['candidates']),25,replace=False)]:
            thetas=total_and_theta(c['configuration'],32,base); val=float(np.mean([ce(*sel,t) for t in thetas])); err=abs(val-c['selection_ce']); maxerr=max(maxerr,err); assert err<1e-9; checks+=1
        for k,g in r['gain_over_public_control'].items(): assert abs(g-(r['public_control']['assessment_ce']-r['stages'][k]['assessment_ce']))<1e-15; checks+=1
        for k,g in r['gain_over_public_zero'].items(): assert abs(g-(r['public_zero']['assessment_ce']-r['stages'][k]['assessment_ce']))<1e-15; checks+=1
    print('limited_public audit passed',checks,'checks, max error',maxerr)

if __name__=='__main__': main()
