"""Rotated-profile diagnostic; tuning integrates fresh second-round noise exactly."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
from dynamic_quadratic_probe import RADII,FRACTIONS,PAIRS,draws,first_round,loss,summaries,contrast

ANGLES=np.arange(4)*np.pi/4
ROTATIONS=np.array([[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]] for a in ANGLES])
ARMS=('static','stale','residual','corrected_residual','random')


def bank_coordinates(update):
    return np.einsum('tni,kij->tnkj',update,ROTATIONS)


def factors(coordinates,pair):
    norm=np.sum(np.abs(coordinates)/np.asarray(pair),axis=-1)
    return 1/np.maximum(1,norm)


def setup(theta,z0,radius,e0,random_indices):
    y0,w1=first_round(theta,z0,radius,e0)
    update=theta-w1[:,None,:]
    proxies={'stale':y0,'residual':y0-w1[:,None,:],
             'corrected_residual':max(1.,.8/radius)*y0-w1[:,None,:]}
    return {'update':update,'coordinates':bank_coordinates(update),'model':w1,
            'target':theta.mean(axis=1),'proxies':{k:bank_coordinates(v) for k,v in proxies.items()},
            'random':random_indices,'first_upload':y0}


def selected_indices(state,pair,arm,angle=0):
    if arm=='static':return np.full(state['update'].shape[:2],angle,dtype=int)
    if arm=='random':return state['random']
    # Radial clipping distortion has the same proxy norm across all bodies.
    # Max factor minimizes Euclidean distortion; public first-index tie rule.
    return np.argmax(factors(state['proxies'][arm],pair),axis=-1)


def evaluate(state,pair,eta1,indices,z1=None):
    fs=factors(state['coordinates'],pair)
    chosen=np.take_along_axis(fs,indices[...,None],axis=-1)[...,0]
    centers=state['update']*chosen[...,None]
    model=state['model']+centers.mean(axis=1)
    variance=4*sum(x*x for x in pair)/(state['update'].shape[1]*eta1**2)
    conditional=.5*np.sum((model-state['target'])**2,axis=-1)+variance
    if z1 is None:return conditional,None
    localnoise=z1*(2*np.array(pair)/eta1)
    rotations=ROTATIONS[indices]
    noise=np.einsum('tnij,tnj->tni',rotations,localnoise)
    sampled=.5*np.sum((model+noise.mean(axis=1)-state['target'])**2,axis=-1)
    return conditional,sampled


def tune(theta,z0,total,random_indices,first_override=None):
    best={arm:{'development_mean':float('inf')} for arm in ARMS};counts=dict.fromkeys(ARMS,0)
    probes=itertools.product(FRACTIONS,RADII) if first_override is None else [first_override]
    for fraction,radius in probes:
        state=setup(theta,z0,radius,total*fraction,random_indices)
        for pair in PAIRS:
            for arm in ARMS:
                for angle in (range(4) if arm=='static' else (0,)):
                    indices=selected_indices(state,pair,arm,angle)
                    values,_=evaluate(state,pair,total*(1-fraction),indices)
                    value=float(values.mean());counts[arm]+=1
                    if value<best[arm]['development_mean']:
                        best[arm]={'development_mean':value,'config':{'fraction':fraction,
                                    'first_radius':radius,'pair':list(pair),'angle_index':angle}}
    return best,counts


def validate():
    assert np.allclose(ROTATIONS@ROTATIONS.transpose(0,2,1),np.eye(2))
    theta,z0,z1=draws(128,8,43)
    random=np.zeros(theta.shape[:2],dtype=int)
    state=setup(theta,z0,.8,4.,random)
    indices=selected_indices(state,(.6,.2),'residual')
    # Exact body gauge after radial clipping in rotated coordinates.
    fs=factors(state['coordinates'],(.6,.2))
    clipped=state['coordinates']*fs[...,None]
    assert np.max(np.sum(np.abs(clipped)/np.array([.6,.2]),axis=-1))<=1+1e-12
    # Rotated noise's covariance trace is invariant; conditional risk identity is checked by draws.
    one={k:(v[:1] if isinstance(v,np.ndarray) else v) for k,v in state.items()}
    rng=np.random.default_rng(44);fresh=rng.laplace(size=(100000,8,2))
    expected,sampled=evaluate(one,(.6,.2),4.,indices[:1],fresh)
    se=float(sampled.std(ddof=1)/np.sqrt(len(sampled)))
    assert abs(float(sampled.mean())-expected[0])<6*se
    # Different rotation frames can differ even for equal radii: diamonds are not circles.
    u=np.array([[[.8,0.]]]);v=factors(bank_coordinates(u),(.8,.8))
    assert v[0,0,0]>v[0,0,1]
    return {'conditional_expected':float(expected[0]),'conditional_sampled':float(sampled.mean()),
            'conditional_standard_error':se,'samples':len(sampled),'diamond_rotation_distinction_checked':True}


def main():
    checks=validate();rows=[];saved={}
    for cell,(total,n) in enumerate(itertools.product((4.,8.,16.),(8,48))):
        dev_seed=2026100600+cell;eval_seed=2026100700+cell
        td,z0d,z1d=draws(1024,n,dev_seed)
        randomd=np.random.default_rng(dev_seed+100000).integers(0,4,size=td.shape[:2])
        best,counts=tune(td,z0d,total,randomd)
        te,z0e,z1e=draws(8192,n,eval_seed)
        randome=np.random.default_rng(eval_seed+100000).integers(0,4,size=te.shape[:2])
        conditional={};sampled={};matched={}
        for arm in ARMS:
            c=best[arm]['config'];state=setup(te,z0e,c['first_radius'],total*c['fraction'],randome)
            idx=selected_indices(state,tuple(c['pair']),arm,c['angle_index'])
            conditional[arm],sampled[arm]=evaluate(state,tuple(c['pair']),total*(1-c['fraction']),idx,z1e)
            if arm in ('stale','residual','corrected_residual'):
                mb,_=tune(td,z0d,total,randomd,(c['fraction'],c['first_radius']))
                mc=mb['static']['config']
                mi=selected_indices(state,tuple(mc['pair']),'static',mc['angle_index'])
                vals,_=evaluate(state,tuple(mc['pair']),total*(1-c['fraction']),mi)
                matched[arm]={'config':mc,'contrast':contrast(conditional[arm],vals)}
        # Public one-release tuning, same development trials and fresh held-out draws.
        one=(float('inf'),None)
        for radius in RADII:
            _,model=first_round(td,z0d,radius,total)
            value=float(loss(model,td).mean())
            if value<one[0]:one=(value,radius)
        _,model=first_round(te,z0e,one[1],total)
        conditional['one_release']=loss(model,te)
        sampled['one_release']=conditional['one_release']
        conditional['known_population']=loss(np.full((len(te),2),.4),te)
        sampled['known_population']=conditional['known_population']
        for arm,v in conditional.items():saved[f'cell{cell}_conditional_{arm}']=v
        for arm,v in sampled.items():saved[f'cell{cell}_sampled_{arm}']=v
        rows.append({'epsilon_total':total,'clients':n,'development_seed':dev_seed,'evaluation_seed':eval_seed,
                     'candidate_counts':counts,'selected':best,'one_release_radius':one[1],
                     'conditional_evaluation':{a:summaries(v) for a,v in conditional.items()},
                     'sampled_evaluation':{a:summaries(v) for a,v in sampled.items()},
                     'conditional_contrasts':{a:{b:contrast(conditional[a],conditional[b])
                                                 for b in ('static','one_release','random','stale') if b!=a}
                                              for a in ('residual','corrected_residual')},
                     'same_probe_static':matched})
        print(total,n,{a:round(float(v.mean()),7) for a,v in conditional.items()},flush=True)
    out=Path('results/client_specific_noise')
    np.savez_compressed(out/'rotated_residual_trial_losses.npz',**saved)
    report={'kind':'rotated/residual quadratic diagnostic; no CIA experiment',
            'development_trials':1024,'evaluation_trials':8192,
            'objective':'same changing-gradient quadratic as previous spike',
            'angles_degrees':[0,45,90,135],'radii':RADII,'budget_fractions':FRACTIONS,
            'selection':'minimize proxy Euclidean clipping distortion within public rotated bank',
            'corrected_proxy':'max(1,.8/a0)*Y0-w1; uses declared known synthetic input norm',
            'primary_endpoint':'conditional expected loss integrating independent second noise exactly',
            'secondary_endpoint':'sampled full two-round loss with common innovations for paired comparisons',
            'intervals':'normal paired Monte Carlo intervals; exploratory, unadjusted multiplicity',
            'checks':checks,'rows':rows,
            'limitations':['finite grids and known toy population','conditional endpoint is utility only, not attacker-accessible',
                           'no attack scores or novelty conclusion','ideal density does not certify finite precision sampler',
                           'same-probe static tuned with development only','one-release retains isotropic initial diamond family']}
    (out/'rotated_residual_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
