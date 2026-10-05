"""Exact development-quadratic cost of a client-local categorical selector.

Partial accounting diagnostic only; raw histories/calibration are not private.
No fresh test examples, Monte Carlo selection or training are used.
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
from research.calculations.broader_geometry_oracle import RADII, EPSILONS, basis, clipped_bank, optimal_shrink

LABEL_MAP = np.array([0, 2, 0, 1])
PURITY = .6
FRACTIONS = (0., .05, .125, .25, .5)
MIXTURES = np.array([v for v in itertools.product(range(11), repeat=4) if sum(v)==10], dtype=float)/10
MIXTURES = np.vstack((MIXTURES, [1/3, 1/3, 1/3, 0]))


def routes(counts):
    counts = np.asarray(counts);totals = counts.sum(axis=1)
    purity = np.divide(counts.max(axis=1), totals, out=np.zeros(len(counts)), where=totals>0)
    return np.where(purity >= PURITY, LABEL_MAP[counts.argmax(axis=1)], 0)


def response(targets, xi):
    # Uniform at xi=0; all three positive probability routes, including dummy input.
    probabilities = np.zeros((len(targets),4))
    denominator = np.exp(xi)+2
    probabilities[:, :3] = 1/denominator
    probabilities[np.arange(len(targets)),targets] = np.exp(xi)/denominator
    assert np.allclose(probabilities.sum(axis=1),1)
    return probabilities


def moments(v, radii, weights, g, h, probabilities, eta):
    probabilities = np.asarray(probabilities)
    if probabilities.ndim==2:probabilities=probabilities[None]
    client_means = np.einsum('bnk,nkd->bnd',probabilities,v)
    mean = np.einsum('n,bnd->bd',weights,client_means)
    linear = mean@g
    mean_quadratic = .5*np.einsum('bd,de,be->b',mean,h,mean)
    per_profile_curvature = np.einsum('nkd,de,nke->nk',v,h,v)
    mean_curvature = np.einsum('bnd,de,bne->bn',client_means,h,client_means)
    selection = .5*np.sum(weights[None]**2*(np.einsum('bnk,nk->bn',probabilities,per_profile_curvature)-mean_curvature),axis=1)
    assert selection.min()>-1e-12
    selection = np.maximum(selection,0)
    profile_penalty = 4*(radii*radii @ np.diag(h))/eta**2
    upload = np.sum(weights[None]**2*np.einsum('bnk,k->bn',probabilities,profile_penalty),axis=1)
    return linear, mean_quadratic, selection, upload


def configurations(u, weights, g, h, probabilities, eta, fixed=False):
    candidates=[]
    for c in ((.3,) if fixed else RADII):
        v,radii=clipped_bank(u,c)
        linear,q,selection,noise=moments(v,radii,weights,g,h,probabilities,eta)
        if fixed:t=np.ones(len(linear));score=linear+q+selection+noise
        else:t,score=optimal_shrink(linear,q+selection+noise)
        j=int(np.argmin(score))
        candidates.append({'predicted_change':float(score[j]),'radius':c,'shrink':float(t[j]),
            'probability_batch_index':j,'linear_after_shrink':float(t[j]*linear[j]),
            'mean_quadratic_after_shrink':float(t[j]**2*q[j]),
            'selection_variance_penalty':float(t[j]**2*selection[j]),
            'upload_noise_penalty':float(t[j]**2*noise[j])})
    return min(candidates,key=lambda a:a['predicted_change'])


def calculate(u, weights, counts, g, h, e):
    targets=routes(counts)
    deterministic=np.eye(4)[targets]
    shared_probs=np.broadcast_to(np.eye(4)[:,None,:],(4,8,4))
    mixture_probs=np.broadcast_to(MIXTURES[:,None,:],(len(MIXTURES),8,4))
    shared=configurations(u,weights,g,h,shared_probs,e)
    mixture=configurations(u,weights,g,h,mixture_probs,e)
    mixture['common_probabilities']=MIXTURES[mixture['probability_batch_index']].tolist()
    assert mixture['predicted_change']<=shared['predicted_change']+1e-12
    free=configurations(u,weights,g,h,deterministic,e)
    grid=[]
    for fraction in FRACTIONS:
        xi=e*fraction;eta=e-xi;prob=response(targets,xi)
        tuned=configurations(u,weights,g,h,prob,eta)
        fixed=configurations(u,weights,g,h,prob,eta,True)
        grid.append({'selector_fraction':fraction,'selector_budget':xi,'upload_budget':eta,
            'correct_route_probability':float(np.exp(xi)/(np.exp(xi)+2)),
            'noise_inflation_at_fixed_profile':float((e/eta)**2),
            'optimistic_calibration':tuned,'fixed_radius_shrink':fixed})
    assert mixture['predicted_change']<=grid[0]['optimistic_calibration']['predicted_change']+1e-12
    best=min(grid,key=lambda a:a['optimistic_calibration']['predicted_change'])
    return {'total_label':e,'local_targets':targets.tolist(),'shared':shared,'public_mixture':mixture,
        'unaccounted_deterministic_route':free,'selector_grid':grid,
        'best_paid_fraction':best['selector_fraction'],
        'best_paid_minus_shared':best['optimistic_calibration']['predicted_change']-shared['predicted_change'],
        'best_paid_minus_public_mixture':best['optimistic_calibration']['predicted_change']-mixture['predicted_change'],
        'free_route_minus_shared':free['predicted_change']-shared['predicted_change'],
        'paid_material_0_001':best['optimistic_calibration']['predicted_change']-mixture['predicted_change']<=-.001}


def validate():
    rng=np.random.default_rng(131);u=rng.normal(scale=.1,size=(8,3));weights=np.arange(1,9,dtype=float);weights/=weights.sum()
    h=np.eye(3);g=rng.normal(size=3);targets=np.arange(8)%3;prob=response(targets,.7)
    v,r=clipped_bank(u,.2);linear,q,selection,noise=moments(v,r,weights,g,h,prob,3.3)
    assignment=np.array(list(itertools.product(range(3),repeat=8)))
    joint=np.prod(prob[np.arange(8)[None,:],assignment],axis=1)
    assert np.isclose(joint.sum(),1)
    means=np.sum(v[np.arange(8)[None,:],assignment]*weights[None,:,None],axis=1)
    penalties=4*np.sum(weights[None,:,None]**2*r[assignment]**2,axis=(1,2))/3.3**2
    exact=np.sum(joint*(means@g+.5*np.sum(means*means,axis=1)+penalties))
    assert np.isclose(exact,(linear+q+selection+noise)[0],atol=1e-12)
    order=np.array([1,2,3,4,5,6,7,0]);b=moments(v[order],r,weights[order],g,h,prob[order],3.3)
    assert all(np.allclose(a,c) for a,c in zip((linear,q,selection,noise),b))
    empty=np.zeros((1,4),dtype=int);assert routes(empty)[0]==0
    counts=np.array([[64]*4,[205,17,17,17],[17,205,17,17],[17,17,205,17],[17,17,17,205]])
    assert np.array_equal(routes(counts),[0,0,2,0,1])
    channels=np.stack([response([k],.7)[0,:3] for k in range(3)])
    assert np.max(channels[:,None,:]/channels[None,:,:])<=np.exp(.7)+1e-12
    print('Exact3^8 selector expectation, selection variance, permutation invariance, empty/tie routing and RR ratio checks passed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist')
    args=parser.parse_args();validate()
    if args.self_check:return
    root=Path('results/client_specific_noise');prior=json.loads((root/'client_geometry_audit.json').read_text())
    archive=np.load(root/'client_geometry_audit_matrices.npz')
    train_path=next(args.cache_dir.rglob('fashion_mnist-train.arrow'),None)
    if train_path is None:raise FileNotFoundError('Offline training Arrow cache required')
    digest=hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert digest==prior['source_hashes'][train_path.name]
    train=Dataset.from_file(str(train_path));labels=np.array(train['label']);p=basis(False);rows=[]
    for seed in (42,43,44):
        ids=archive[f'seed{seed}_development_original_indices'];x=features(train,ids);y=labels[ids]
        for mode in ('balanced','quantity','label_stress'):
            for round_ in (5,20):
                key=f'{mode}_seed{seed}_round{round_}';w=archive[key+'_model'];weights=archive[key+'_weights']
                counts=np.array(next(a for a in prior['rows'] if a['seed']==seed and a['partition']==mode and a['round']==round_)['client_class_counts'])
                u=-.5*archive[key+'_mean_gradients'].reshape(8,68)@p
                gd,_,hd,*_=statistics(x,y,w);g=p.T@gd.ravel();h=p.T@hd@p
                cases=[calculate(u,weights,counts,g,h,e) for e in EPSILONS]
                shifted=[calculate(u[np.roll(np.arange(8),-1)],weights[np.roll(np.arange(8),-1)],counts[np.roll(np.arange(8),-1)],g,h,e) for e in EPSILONS]
                for a,b in zip(cases,shifted):
                    for field in ('best_paid_minus_shared','best_paid_minus_public_mixture','free_route_minus_shared'):
                        assert np.isclose(a[field],b[field],atol=1e-12)
                rows.append({'seed':seed,'partition':mode,'round':round_,'cases':cases,'permutation_check':'passed'})
                print(seed,mode,round_,[(c['total_label'],c['best_paid_minus_public_mixture']) for c in cases],flush=True)
    result={'kind':'exact development-quadratic partial-accounting diagnostic; no heldout test or DP training',
        'training_arrow_sha256':digest,'label_map':LABEL_MAP.tolist(),'purity_threshold':PURITY,
        'selector':'3-category RR over profiles0/1/2; input counts remain local; empty/lowpurity route0',
        'radii':RADII,'total_labels':EPSILONS,'selector_fractions':FRACTIONS,
        'public_mixture_grid':'4-profile simplexstep.1 plus exact uniformprofiles0/1/2;287commonprobabilities',
        'objective':'exact expectation of development quadratic including clipping mean, selection covariance, upload noise and common shrink',
        'limitations':['development-only approximated utility, not independent predictive evidence',
            'raw checkpoints/global calibration/radius-shrink-budget tuning are not accounted; not actual total-budget DP mechanism',
            'local routing map learned from previousdiagnostics; publicstatus a conditional designassumption',
            'bias-only rule cannot certify exposed sensitive complementary updates or metadata',
            'finite categorical rule/profile/radius/budget/mixture grids, not all constructors or continuous optimum'],
        'rows':rows}
    (root/'local_selector_cost.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
