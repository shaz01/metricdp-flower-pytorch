"""Hidden-selector tail witnesses and fair contribution-to-dummy cost audit.

Numerical witnesses accompany an analytic bound; they do not prove a supremum.
All utility calculations are development-only partial-accounting diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from research.calculations.client_geometry_audit import features, statistics
from research.calculations.broader_geometry_oracle import basis, clipped_bank, EPSILONS
from research.calculations.local_selector_cost import routes, response, configurations, MIXTURES, FRACTIONS


def log_density(y, u, preferred, xi, eta, c=1.):
    v,r=clipped_bank(np.array(u)[None],c)
    probabilities=response([preferred],xi)[0,:3]
    component=np.log(probabilities)+np.log(eta/(4*r[:3])).sum(axis=1)
    component-=eta/2*np.sum(np.abs(y[None]-v[0,:3])/r[:3],axis=1)
    m=component.max()
    return float(m+np.log(np.exp(component-m).sum()))


def witnesses():
    rows=[]
    for xi,eta in ((1.,3.),(2.,6.),(4.,12.)):
        for distance in (1.,5.,20.,100.):
            y=np.full(3,distance)
            replacement=log_density(y,[1.,0,0],0,xi,eta)-log_density(y,[-1.,0,0],2,xi,eta)
            dummy=log_density(y,[0.,0,0],0,xi,eta)-log_density(y,[-1.,0,0],2,xi,eta)
            rows.append({'xi':xi,'eta':eta,'tail_distance':distance,
                'replacement_log_ratio':replacement,'replacement_bound':xi+eta,
                'dummy_reverse_log_ratio':dummy,'dummy_edge_bound':xi+eta/2})
    # Couple preferred class to an actually attainable bias gradient at a public
    # class-constant softmax model; vary public history toward two-class balance.
    coupled=[];pmap=basis(False)[[16,33,50,67]]
    xi,eta,c=2.,6.,.3
    for small in (.1,.01,.0001,1e-8):
        probabilities=np.array([.5-small,.5-small,small,small])
        u0=.5*(np.eye(4)[0]-probabilities)@pmap
        u1=.5*(np.eye(4)[1]-probabilities)@pmap
        y=np.full(3,100*c)
        replacement=log_density(y,u0,0,xi,eta,c)-log_density(y,u1,2,xi,eta,c)
        dummy=log_density(y,np.zeros(3),0,xi,eta,c)-log_density(y,u1,2,xi,eta,c)
        coupled.append({'public_small_class_probability':small,'replacement_log_ratio':replacement,
                        'dummy_reverse_log_ratio':dummy,'replacement_bound':8.,'dummy_edge_bound':5.})
    return rows,coupled


def dummy_cost(u,weights,counts,g,h,e):
    # For dummy-edge privacy: xi + eta/2 = e, so eta=2(e-xi).
    target=routes(counts)
    shared_probs=np.broadcast_to(np.eye(4)[:,None,:],(4,8,4))
    mixtures=np.broadcast_to(MIXTURES[:,None,:],(len(MIXTURES),8,4))
    shared=configurations(u,weights,g,h,shared_probs,2*e)
    mixture=configurations(u,weights,g,h,mixtures,2*e)
    mixture['common_probabilities']=MIXTURES[mixture['probability_batch_index']].tolist()
    free=configurations(u,weights,g,h,np.eye(4)[target],2*e)
    grid=[]
    for fraction in FRACTIONS:
        xi=e*fraction;eta=2*(e-xi)
        cfg=configurations(u,weights,g,h,response(target,xi),eta)
        grid.append({'selector_fraction':fraction,'xi':xi,'eta_kernel_parameter':eta,
            'dummy_edge_bound':xi+eta/2,'replacement_upper_bound':xi+eta,
            'correct_route_probability':float(np.exp(xi)/(np.exp(xi)+2)),
            'configuration':cfg})
    best=min(grid,key=lambda a:a['configuration']['predicted_change'])
    assert mixture['predicted_change']<=shared['predicted_change']+1e-12
    assert mixture['predicted_change']<=grid[0]['configuration']['predicted_change']+1e-12
    return {'contribution_dummy_label':e,'local_targets':target.tolist(),'shared':shared,
        'public_mixture':mixture,'unaccounted_deterministic_route':free,'selector_grid':grid,
        'best_selector_fraction':best['selector_fraction'],
        'best_selector_minus_public_mixture':best['configuration']['predicted_change']-mixture['predicted_change'],
        'material_0_001':best['configuration']['predicted_change']-mixture['predicted_change']<=-.001}


def validate():
    rows,coupled=witnesses()
    for a in rows:
        assert a['replacement_log_ratio']<=a['replacement_bound']+1e-10
        assert a['dummy_reverse_log_ratio']<=a['dummy_edge_bound']+1e-10
        if a['tail_distance']==100:
            assert np.isclose(a['replacement_log_ratio'],a['replacement_bound'],atol=1e-9)
            assert np.isclose(a['dummy_reverse_log_ratio'],a['dummy_edge_bound'],atol=1e-9)
    assert abs(coupled[-1]['replacement_log_ratio']-8)<1e-6
    assert abs(coupled[-1]['dummy_reverse_log_ratio']-5)<1e-6
    rng=np.random.default_rng(155);u=rng.normal(scale=.1,size=(8,3));w=np.full(8,1/8)
    counts=np.tile([205,17,17,17],(8,1));g=rng.normal(size=3);h=np.eye(3)
    a=dummy_cost(u,w,counts,g,h,8)
    order=np.arange(8)[::-1];b=dummy_cost(u[order],w[order],counts[order],g,h,8)
    assert np.isclose(a['best_selector_minus_public_mixture'],b['best_selector_minus_public_mixture'])
    print('Stable tail densities, attained limiting bounds, coupled-label witnesses, dummy-budget identity and permutation checks passed',flush=True)


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
    train=Dataset.from_file(str(path));labels=np.array(train['label']);p=basis(False);rows=[]
    for seed in (42,43,44):
        ids=archive[f'seed{seed}_development_original_indices'];x=features(train,ids);y=labels[ids]
        for mode in ('balanced','quantity','label_stress'):
            for round_ in (5,20):
                key=f'{mode}_seed{seed}_round{round_}';w=archive[key+'_model'];weights=archive[key+'_weights']
                counts=np.array(next(a for a in prior['rows'] if a['seed']==seed and a['partition']==mode and a['round']==round_)['client_class_counts'])
                u=-.5*archive[key+'_mean_gradients'].reshape(8,68)@p
                gd,_,hd,*_=statistics(x,y,w);g=p.T@gd.ravel();h=p.T@hd@p
                cases=[dummy_cost(u,weights,counts,g,h,e) for e in EPSILONS]
                rows.append({'seed':seed,'partition':mode,'round':round_,'cases':cases})
                print(seed,mode,round_,[c['best_selector_minus_public_mixture'] for c in cases],flush=True)
    tails,coupled=witnesses()
    report={'kind':'analytic-mixture tail witnesses and fair dummy-edge development cost; not private training/CIA',
        'training_arrow_sha256':sha,'tail_profiles':'three L1 profiles:equal axes,retainaxis0,retainaxis1;others.25',
        'uniform_bounds':{'replacement':'xi+eta','dataset_vs_fixed_zero_dummy':'xi+eta/2'},
        'calibration':'dummy label E uses eta=2(E-xi) for selector and eta=2E for all public controls',
        'tail_witnesses':tails,'coupled_label_route_witnesses':coupled,
        'limitations':['analytic proof required; numerical tailpoints alone cannot establish allinputs/alloutputs',
            'sharpness is uniform over arbitrary admissible privateupdates/preferences and publichistories, not certified fixedtrainedhead supremum',
            'dummy-edge bound is not same-budget arbitrary nonempty-to-nonempty replacement privacy',
            'raw history/radius/shrink/split calibration remains unaccounted; development approximation only',
            'no protected-history pilot, new sampler, heldout test or CIA experiment'],
        'rows':rows}
    (root/'hidden_mixture_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
