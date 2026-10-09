"""Protected-history bias routing on saved real-data updates, not private FL.

Fresh noise evaluations share a fixed development loss; no test data are read.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from research.calculations.client_geometry_audit import features, statistics
from research.calculations.broader_geometry_oracle import basis, clipped_bank, RADII, EPSILONS
from research.calculations.local_selector_cost import MIXTURES, moments

PROBE_RADII = (.02, .1, .3)
FRACTIONS = (.125, .25, .5)
FIT_DRAWS = 4096
EVAL_DRAWS = 16384
LABEL_MAP = np.array([0, 2, 0, 1])
CLASS_BASIS = basis(False)[[16, 33, 50, 67]]


def box_minimum(linear, matrix):
    """Convex two-coefficient quadratic, minimize on [0,1]^2 by faces."""
    assert np.linalg.eigvalsh(matrix).min() > -1e-10
    best = None
    for face in itertools.product((-1, 0, 1), repeat=2):
        free = np.flatnonzero(np.array(face)==-1)
        fixed = np.flatnonzero(np.array(face)!=-1)
        t = np.maximum(face, 0).astype(float)
        if len(free):
            rhs = -linear[free]-matrix[np.ix_(free,fixed)]@t[fixed]
            t[free] = np.linalg.lstsq(matrix[np.ix_(free,free)],rhs,rcond=None)[0]
            if np.max(np.abs(matrix[np.ix_(free,free)]@t[free]-rhs)) > 1e-9:continue
        if np.any(t < -1e-10) or np.any(t > 1+1e-10):continue
        t = np.clip(t,0,1)
        score = float(linear@t+.5*t@matrix@t)
        if best is None or score < best[0]:best = (score,t)
    assert best is not None and best[0] <= 1e-10
    return best


def probe(u, weights, c0, eps0, unit):
    clipped = clipped_bank(u,c0)[0][:,0]
    y = clipped[None]+unit*c0/eps0
    aggregate = np.sum(y*weights[None,:,None],axis=1)
    labels = np.argmax(np.einsum('tnd,cd->tnc',y,CLASS_BASIS),axis=2)
    return aggregate,LABEL_MAP[labels]


def second(u, weights, ids, c1, eps1, h):
    v,radii = clipped_bank(u,c1)
    selected = v[np.arange(len(u))[None],ids]
    mean = np.sum(selected*weights[None,:,None],axis=1)
    # Main noise scale r/eps1 for fixed-zero-dummy adjacency.
    trace = 2*np.sum(weights[None,:,None]**2*radii[ids]**2*np.diag(h),axis=(1,2))/eps1**2
    return mean,trace


def fit(A,B,noise_trace,g,h):
    linear = np.array([A.mean(axis=0)@g,B.mean(axis=0)@g])
    aa = np.mean(np.einsum('td,de,te->t',A,h,A))
    ab = np.mean(np.einsum('td,de,te->t',A,h,B))
    bb = np.mean(np.einsum('td,de,te->t',B,h,B)+noise_trace)
    matrix = np.array([[aa,ab],[ab,bb]])
    score,coeff = box_minimum(linear,matrix)
    return {'fit_score':score,'coefficients':coeff.tolist(),'linear':linear.tolist(),'quadratic_matrix':matrix.tolist()}


def values(A,B,noise_trace,coeff,g,h):
    a,b = coeff;delta = a*A+b*B
    return delta@g+.5*np.einsum('td,de,te->t',delta,h,delta)+.5*b*b*noise_trace


def configuration(u0,u1,weights,g,h,e,unit):
    best_adaptive = None;best_static = None
    for fraction in FRACTIONS:
        eps0 = fraction*e;eps1 = e-eps0
        for c0 in PROBE_RADII:
            A,ids = probe(u0,weights,c0,eps0,unit)
            for c1 in RADII:
                for profile in ('adaptive',0,1,2,3):
                    chosen = ids if profile=='adaptive' else np.full(ids.shape,profile)
                    B,trace = second(u1,weights,chosen,c1,eps1,h)
                    cfg = {'probe_fraction':fraction,'eps_probe':eps0,'eps_main':eps1,
                        'probe_radius':c0,'main_radius':c1,'main_profile':profile,
                        **fit(A,B,trace,g,h)}
                    if profile=='adaptive':
                        if best_adaptive is None or cfg['fit_score']<best_adaptive['fit_score']:best_adaptive=cfg
                    elif best_static is None or cfg['fit_score']<best_static['fit_score']:best_static=cfg
    # Static arm with exactly adaptive's probe and budget split; same tuning freedom for main.
    c = best_adaptive;A,ids = probe(u0,weights,c['probe_radius'],c['eps_probe'],unit)
    matched = None
    for c1 in RADII:
        for profile in range(4):
            chosen=np.full(ids.shape,profile)
            B,trace=second(u1,weights,chosen,c1,c['eps_main'],h)
            cfg={**c,'main_radius':c1,'main_profile':profile,**fit(A,B,trace,g,h)}
            if matched is None or cfg['fit_score']<matched['fit_score']:matched=cfg
    assert best_static['fit_score']<=matched['fit_score']+1e-12
    return {'adaptive':best_adaptive,'same_probe_static':matched,'optimized_static':best_static}


def one_release(u0,u1,weights,g,h,e):
    best = None
    probabilities = np.broadcast_to(MIXTURES[:,None,:],(len(MIXTURES),8,4))
    for blend in (0.,.5,1.):
        u=blend*u0+(1-blend)*u1
        for radius in RADII:
            v,r=clipped_bank(u,radius)
            linear,q,selection,noise=moments(v,r,weights,g,h,probabilities,2*e)
            # Up to2 matches the two-stage coefficient sum, rather than limiting its learning step.
            denom=2*(q+selection+noise)
            shrink=np.clip(np.divide(-linear,denom,out=np.zeros_like(linear),where=denom>1e-20),0,2)
            score=linear*shrink+(q+selection+noise)*shrink**2
            j=int(np.argmin(score))
            cfg={'fit_score':float(score[j]),'source_early_weight':blend,'radius':radius,
                 'shrink':float(shrink[j]),'common_profile_probabilities':MIXTURES[j].tolist()}
            if best is None or cfg['fit_score']<best['fit_score']:best=cfg
    assert best['fit_score']<=1e-12
    return best


def evaluate(cfg,u0,u1,weights,g,h,unit):
    A,ids=probe(u0,weights,cfg['probe_radius'],cfg['eps_probe'],unit)
    chosen=ids if cfg['main_profile']=='adaptive' else np.full(ids.shape,cfg['main_profile'])
    B,trace=second(u1,weights,chosen,cfg['main_radius'],cfg['eps_main'],h)
    x=values(A,B,trace,cfg['coefficients'],g,h)
    return x,{'mean_change':float(x.mean()),'noise_monte_carlo_standard_error':float(x.std(ddof=1)/np.sqrt(len(x))),
        'probe_route_frequencies_per_client':[(np.bincount(ids[:,i],minlength=4)/len(ids)).tolist() for i in range(8)]}


def contrast(a,b):
    d=a-b;se=float(d.std(ddof=1)/np.sqrt(len(d)));m=float(d.mean())
    return {'mean':m,'paired_noise_standard_error':se,'interval_95_normal':[m-1.96*se,m+1.96*se]}


def validate():
    rng=np.random.default_rng(166)
    linear=np.array([-.3,-.2]);matrix=np.array([[1.,.4],[.4,.7]])
    score,t=box_minimum(linear,matrix)
    grid=np.array(list(itertools.product(np.linspace(0,1,101),repeat=2)))
    assert score <= np.min(grid@linear+.5*np.einsum('ni,ij,nj->n',grid,matrix,grid))+1e-12
    assert box_minimum(np.array([1.,1.]),np.eye(2))[0]==0
    u0=rng.normal(size=(8,3));u1=rng.normal(size=(8,3));weights=np.full(8,1/8)
    unit=rng.laplace(size=(128,8,3));g=rng.normal(size=3);h=np.eye(3)
    A,ids=probe(u0,weights,.1,2,unit);B,trace=second(u1,weights,ids,.3,6,h)
    cfg=fit(A,B,trace,g,h)
    assert np.isclose(values(A,B,trace,cfg['coefficients'],g,h).mean(),cfg['fit_score'])
    order=np.roll(np.arange(8),-1)
    AA,jj=probe(u0[order],weights[order],.1,2,unit[:,order]);BB,tt=second(u1[order],weights[order],jj,.3,6,h)
    assert np.allclose(A,AA) and np.allclose(B,BB) and np.allclose(trace,tt)
    coeff=np.array([.3,.7]);delta=coeff[0]*A+coeff[1]*B
    noise=rng.laplace(size=(100000,8,3))
    # Independent sanity check of second-stage covariance for a fixed chosen assignment.
    chosen=ids[0];r=clipped_bank(u1,.3)[1][chosen]
    z=np.sum(noise*(r/6)[None]*weights[None,:,None],axis=1)
    expected=2*np.sum(weights[:,None]**2*r*r,axis=0)/36
    assert np.allclose(z.var(axis=0),expected,rtol=.03)
    assert np.isfinite(delta).all()
    print('Box optimization, correlated objective identity, matched-coin permutation and main-noise covariance checks passed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist')
    args=parser.parse_args();validate()
    if args.self_check:return
    root=Path('results/client_specific_noise');prior=json.loads((root/'client_geometry_audit.json').read_text())
    archive=np.load(root/'client_geometry_audit_matrices.npz')
    path=next(args.cache_dir.rglob('fashion_mnist-train.arrow'),None)
    if path is None:raise FileNotFoundError('Offline training Arrow cache required')
    sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==prior['source_hashes'][path.name]
    train=Dataset.from_file(str(path));labels=np.array(train['label']);p=basis(False);rows=[];saved={}
    for seed in (42,43,44):
        ids=archive[f'seed{seed}_development_original_indices'];x=features(train,ids);y=labels[ids]
        for mode in ('balanced','quantity','label_stress'):
            for early,late in ((0,5),(5,20)):
                key=f'{mode}_seed{seed}_path{early}_{late}'
                weights=archive[f'{mode}_seed{seed}_round{late}_weights']
                w=archive[f'{mode}_seed{seed}_round{late}_model']
                u0=-.5*archive[f'{mode}_seed{seed}_round{early}_mean_gradients'].reshape(8,68)@p
                u1=-.5*archive[f'{mode}_seed{seed}_round{late}_mean_gradients'].reshape(8,68)@p
                gd,_,hd,*_=statistics(x,y,w);g=p.T@gd.ravel();h=p.T@hd@p
                fit_unit=np.random.default_rng(seed*1000+late).laplace(size=(FIT_DRAWS,8,3))
                eval_unit=np.random.default_rng(seed*1000+late+100000).laplace(size=(EVAL_DRAWS,8,3))
                cases=[]
                for e in EPSILONS:
                    configs=configuration(u0,u1,weights,g,h,e,fit_unit);single=one_release(u0,u1,weights,g,h,e)
                    vals={};ev={}
                    for arm,cfg in configs.items():
                        vals[arm],ev[arm]=evaluate(cfg,u0,u1,weights,g,h,eval_unit)
                        saved[f'{key}_e{int(e)}_{arm}']=vals[arm]
                    contrasts={f'adaptive_minus_{arm}':contrast(vals['adaptive'],vals[arm]) for arm in ('same_probe_static','optimized_static')}
                    contrasts['adaptive_minus_one_release']=contrast(vals['adaptive'],np.full(EVAL_DRAWS,single['fit_score']))
                    cases.append({'dummy_budget_label':e,'configurations':configs,'one_release':single,
                        'noise_evaluation':ev,'contrasts':contrasts,
                        'material_gain_vs_one_release':contrasts['adaptive_minus_one_release']['mean']<=-.001})
                    print(key,e,contrasts['adaptive_minus_one_release']['mean'],flush=True)
                rows.append({'seed':seed,'partition':mode,'early_round':early,'late_round':late,'cases':cases})
    result={'kind':'protected-probe conditional bias replay, not private FL/CIA or heldout-data utility',
        'training_arrow_sha256':sha,'probe_radii':PROBE_RADII,'probe_fractions':FRACTIONS,'main_radii':RADII,
        'fit_noise_draws':FIT_DRAWS,'evaluation_noise_draws':EVAL_DRAWS,
        'selector':'argmax class reconstruction of protected earlybias3, fixedmap[0,2,0,1]; no rawlabels',
        'calibration':'dummy probeC0/eps0, conditionalmainr/eps1; eps0+eps1=E',
        'final_signal':'a*aggregateprobe+b*aggregatemain; a,b[0,1], empiricaljointmoments include probe-selection correlation',
        'one_release':'allbudget,early/current/averagequery,all4profiles+287commonmixtures,radiusgrid,shrink[0,2]',
        'limitations':['public/fixed history,banks,weights,calibration required for conditional privacy law; saved rawtrajectory not protected',
            'data-dependent config/coefficients fitted on rawdevelopment info and noise are unaccounted; optimistic feasibility only',
            'replay of two recorded updates, not model jointly trained from the protecting probe',
            'fresh protecting-noise evaluations use SAME development examples/objective; no heldoutCE/CIA evidence',
            'mainnoise analytically integrated for quadratic evaluation; intervals measure probe-noise uncertainty only',
            'finite bank/split/radius/query grids and exploratory multiple looks, not universal optimality'],
        'rows':rows}
    (root/'protected_real_history_probe.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(root/'protected_real_history_trials.npz',**saved)


if __name__=='__main__':main()
