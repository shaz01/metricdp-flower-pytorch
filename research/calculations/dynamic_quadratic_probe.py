"""Bounded two-round quadratic learning spike; no Flower/image/CIA experiment.

Run through uv; tuning and evaluation federation draws are disjoint.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np

RADII = (0.1, 0.2, 0.4, 0.8, 1.2, 1.6)
FRACTIONS = (0.25, 0.5, 0.75)
PAIRS = tuple((a,b) for a in RADII for b in RADII if a>=b)


def draws(trials, clients, seed):
    rng=np.random.default_rng(seed)
    axes=rng.integers(0,2,size=(trials,clients))
    theta=np.eye(2)[axes]*0.8
    return theta, rng.laplace(size=theta.shape), rng.laplace(size=theta.shape)


def clip_diamond(update, radii):
    norm=np.sum(np.abs(update)/radii,axis=-1,keepdims=True)
    return update/np.maximum(1.0,norm)


def loss(model, theta):
    # Exact excess global objective relative to the realized federation optimum.
    return .5*np.sum((model-theta.mean(axis=1))**2,axis=-1)


def first_round(theta, noise, radius, epsilon):
    y=clip_diamond(theta,np.array([radius,radius]))+noise*(2*radius/epsilon)
    return y,y.mean(axis=1)


def second_round(theta, noise, first_upload, model, epsilon, pair, adaptive):
    long,short=pair
    if adaptive:
        axis=np.argmax(np.abs(first_upload),axis=-1)
        radii=np.where(axis[...,None]==np.arange(2),long,short)
    else:
        radii=np.array(pair)
    update=theta-model[:,None,:]
    y=clip_diamond(update,radii)+noise*(2*radii/epsilon)
    return model+y.mean(axis=1)


def objective(theta, n0, n1, total, config, adaptive):
    e0=total*config['fraction']
    y0,w1=first_round(theta,n0,config['first_radius'],e0)
    w2=second_round(theta,n1,y0,w1,total-e0,tuple(config['pair']),adaptive)
    return loss(w2,theta)


def tune(theta,n0,n1,total):
    # Exhaust the declared finite family; static controls include both fixed orientations.
    best={arm:{'development_mean':float('inf')} for arm in ('adaptive','static','one_release')}
    counts={arm:0 for arm in best}
    for fraction,radius in itertools.product(FRACTIONS,RADII):
        e0=total*fraction
        y0,w1=first_round(theta,n0,radius,e0)
        for pair in PAIRS:
            for adaptive,arm in ((True,'adaptive'),(False,'static')):
                oriented=(pair,) if adaptive or pair[0]==pair[1] else (pair,pair[::-1])
                for fixed_pair in oriented:
                    w2=second_round(theta,n1,y0,w1,total-e0,fixed_pair,adaptive)
                    value=float(loss(w2,theta).mean());counts[arm]+=1
                    if value<best[arm]['development_mean']:
                        best[arm]={'development_mean':value,'config':
                                   {'fraction':fraction,'first_radius':radius,'pair':list(fixed_pair)}}
    for radius in RADII:
        _,model=first_round(theta,n0,radius,total)
        value=float(loss(model,theta).mean());counts['one_release']+=1
        if value<best['one_release']['development_mean']:
            best['one_release']={'development_mean':value,'config':{'radius':radius}}
    return best,counts


def summaries(values):
    mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)))
    return {'mean_excess_loss':mean,'monte_carlo_standard_error':se,
            'mean_interval_95_normal':[mean-1.96*se,mean+1.96*se]}


def contrast(adaptive,reference):
    difference=adaptive-reference
    mean=float(difference.mean());se=float(difference.std(ddof=1)/np.sqrt(len(difference)))
    return {'difference_adaptive_minus_reference':mean,'paired_standard_error':se,
            'interval_95_normal':[mean-1.96*se,mean+1.96*se],
            'relative_reduction_percent':100*(float(reference.mean())-float(adaptive.mean()))/float(reference.mean())}


def validate():
    theta,n0,n1=draws(128,8,202610040)
    radii=np.array([.2,.4]);clipped=clip_diamond(theta,radii)
    assert np.max(np.sum(np.abs(clipped)/radii,axis=-1))<=1+1e-12
    # Both worlds use complete-vector gradient updates, not stationary copies of theta.
    y0,w1=first_round(theta,np.zeros_like(n0),100.,4.)
    w2=second_round(theta,np.zeros_like(n1),y0,w1,4.,(100.,100.),False)
    assert np.max(loss(w2,theta))<1e-25
    assert np.allclose((theta-w1[:,None,:]).mean(axis=1),0,atol=1e-12)
    # Equal-axis profiles must be exactly the same mechanism despite adaptive selection.
    y0,w1=first_round(theta,n0,.8,4.)
    a=second_round(theta,n1,y0,w1,4.,(.4,.4),True)
    b=second_round(theta,n1,y0,w1,4.,(.4,.4),False)
    assert np.array_equal(a,b)
    # Whole-client pairwise body sensitivity is <=2 after clipping arbitrary large vectors.
    rng=np.random.default_rng(4)
    x=clip_diamond(rng.normal(size=(1000,2))*100,radii)
    xp=clip_diamond(rng.normal(size=(1000,2))*100,radii)
    assert np.max(np.sum(np.abs(x-xp)/radii,axis=-1))<=2+1e-12


    # Independent conditional moment check: integrate fresh second noise only.
    # theta/probe/model are fixed; the shared-history bias is evaluated explicitly.
    long,short=.6,.2
    axis=np.argmax(np.abs(y0[:1]),axis=-1)
    body=np.where(axis[...,None]==np.arange(2),long,short)
    center=clip_diamond(theta[:1]-w1[:1,None,:],body)
    bias=w1[:1]+center.mean(axis=1)-theta[:1].mean(axis=1)
    expected=float(.5*np.sum(bias*bias)+4*np.sum(body*body)/(8**2*4**2))
    fresh=rng.laplace(size=(100_000,8,2))
    predicted=second_round(theta[:1],fresh,y0[:1],w1[:1],4.,(long,short),True)
    values=loss(predicted,theta[:1])
    observed=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)))
    assert abs(observed-expected)<6*se
    return {'conditional_noise_draws':len(values),'expected_excess':expected,
            'observed_excess':observed,'monte_carlo_standard_error':se,
            'tolerance':'six standard errors; numerical sanity only'}


def main():
    validation=validate()
    rows=[];arrays={}
    for cell,(total,clients) in enumerate(itertools.product((4.,8.,16.),(8,48))):
        dev_seed=2026100400+cell;eval_seed=2026100500+cell
        theta,n0,n1=draws(1024,clients,dev_seed)
        selected,counts=tune(theta,n0,n1,total)
        theta,n0,n1=draws(8192,clients,eval_seed)
        losses={arm:objective(theta,n0,n1,total,selected[arm]['config'],arm=='adaptive')
                for arm in ('adaptive','static')}
        config=selected['adaptive']['config']
        # Optimize the static second profile on development under the adaptive arm's exact probe.
        td,z0,z1=draws(1024,clients,dev_seed)
        y0,w1=first_round(td,z0,config['first_radius'],total*config['fraction'])
        matched=(float('inf'),None)
        for pair in itertools.product(RADII,repeat=2):
            wd=second_round(td,z1,y0,w1,total*(1-config['fraction']),pair,False)
            score=float(loss(wd,td).mean())
            if score<matched[0]:matched=(score,pair)
        matched_config={**config,'pair':list(matched[1])}
        losses['same_probe_static']=objective(theta,n0,n1,total,matched_config,False)
        _,single=first_round(theta,n0,selected['one_release']['config']['radius'],total)
        losses['one_release']=loss(single,theta)
        losses['known_population']=loss(np.full((len(theta),2),.4),theta)
        # Stronger free post-processing: shrink each final model toward public population mean.
        # Fit the scalar shrinkage on development only, freeze it for fresh evaluation.
        shrinkages={}
        for arm in ('adaptive','static','one_release'):
            conf=selected[arm]['config']
            if arm=='one_release':
                _,md=first_round(td,z0,conf['radius'],total)
                _,me=first_round(theta,n0,conf['radius'],total)
            else:
                yd,wd=first_round(td,z0,conf['first_radius'],total*conf['fraction'])
                md=second_round(td,z1,yd,wd,total*(1-conf['fraction']),tuple(conf['pair']),arm=='adaptive')
                ye,we=first_round(theta,n0,conf['first_radius'],total*conf['fraction'])
                me=second_round(theta,n1,ye,we,total*(1-conf['fraction']),tuple(conf['pair']),arm=='adaptive')
            xd=md-.4;target=td.mean(axis=1)-.4
            alpha=float(np.clip(np.sum(xd*target)/np.sum(xd*xd),0,1))
            shrinkages[arm]=alpha
            losses[arm+'_shrunk']=loss(.4+alpha*(me-.4),theta)
        for arm,values in losses.items():arrays[f'cell{cell}_{arm}']=values
        rows.append({'epsilon_total':total,'clients':clients,'development_trials':1024,
                     'evaluation_trials':8192,'development_seed':dev_seed,'evaluation_seed':eval_seed,
                     'candidate_counts':counts,'selected':selected,'same_probe_static_config':matched_config,
                     'shrinkage_fitted_on_development':shrinkages,
                     'evaluation':{arm:summaries(v) for arm,v in losses.items()},
                     'contrasts':{arm:contrast(losses['adaptive'],v) for arm,v in losses.items() if arm!='adaptive'},
                     'shrunk_contrasts':{arm:contrast(losses['adaptive_shrunk'],losses[arm+'_shrunk'])
                                         for arm in ('static','one_release')}})
        print(total,clients,{arm:round(float(v.mean()),7) for arm,v in losses.items()},flush=True)
    output=Path('results/client_specific_noise')
    np.savez_compressed(output/'dynamic_quadratic_trial_losses.npz',**arrays)
    report={'kind':'bounded numerical learning spike, not CIA/image-model experiment',
            'objective':'Fi(w)=.5||w-theta_i||²; exact excess=.5||w-mean(theta)||²',
            'protocol':'fixed equal-weight slots; w0=0; w1=mean(Y0); w2=w1+mean(Y1)',
            'private_inputs':'theta_i=.8 e1/e2 iid equiprobable; stationary theta, changing gradient theta-w',
            'privacy':'reference ideal densities; whole-input conditional clipping+Laplace; eta0+eta1=E',
            'grids':{'radii':RADII,'first_budget_fractions':FRACTIONS},
            'evaluation':'fresh seed per cell; same draws across arms for paired contrasts only',
            'intervals':'normal Monte Carlo mean/paired intervals, no multiplicity correction; exploratory',
            'validation':{'deterministic_checks':'zero-noise learnability, complete-gradient cancellation, clipping sensitivity, equal-profile identity','conditional_moment':validation},
            'limitations':['finite grid tuning, no optimality certificate','synthetic known population',
                           'postprocessing shrinkage is fitted after primary configuration selection, not jointly optimized',
                           'no contribution-inference attack run','ideal continuous density proof does not certify finite precision sampler'],
            'rows':rows}
    (output/'dynamic_quadratic_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
