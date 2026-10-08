"""Bounded public-cap class-conditional query experiment; offline tuning is not DP."""
import argparse
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from scipy.special import ndtr
from research.calculations.public_residual_probe import public_geometry, bounded_public, public_scale, random_shares, digest
from research.calculations.public_residual_signal_diagnostic import example_gradients
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.client_geometry_audit import features, partition, split
from research.calculations.client_energy_filter_probe import C, scores, summarize
from research.calculations.client_energy_one_release import local_query
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.matched_cia_probe import auc_statistics

PROTOCOL=Path('research/proposals/2026-10-08_class_conditional_protocol.md')
DEV=ROOT/'class_conditional_development.json'
BANK=ROOT/'class_conditional_development.npz'
MODES=('raw','target_balanced','fixed_center','class_center','center_only','target_center')
CAPS=(.003,.01,.03,.1,.3)
ETAS=(.1,.3,1.,3.,10.,30.)
SCENARIOS=('baseline','prior','mask','both')
DEPS=tuple(Path(p) for p in (
 'research/calculations/public_residual_probe.py',
 'research/calculations/public_residual_signal_diagnostic.py',
 'research/calculations/private_prior_constructor_probe.py',
 'research/calculations/private_descriptor_probe.py',
 'research/calculations/client_geometry_audit.py',
 'research/calculations/client_energy_filter_probe.py',
 'research/calculations/client_energy_one_release.py',
 'research/calculations/matched_cia_probe.py',
 'results/client_specific_noise/public_residual_development.npz',
 'results/client_specific_noise/public_residual_evaluation.npz',
 'results/client_specific_noise/private_descriptor_analytic_gaussian.npz'))


def weighted_scores(x,y,theta,prior):
    ce=[]; acc=[]
    for k in range(4):
        a,b=scores(x[y==k],y[y==k],theta); ce.append(a); acc.append(b)
    return np.asarray(prior)@np.array(ce),np.asarray(prior)@np.array(acc)


def transform(x,mask):
    out=x.copy()
    if mask: out[:,[5,6,9,10]]=0
    return out


def prior_correct(b,prior):
    out=b.copy(); out[:,-1]+=np.log(np.asarray(prior)/.25)@C
    return out


def class_gradients(x,y,b):
    g=example_gradients(x,y,b)
    return np.stack([g[y==k].mean(0) for k in range(4)])


def client_queries(x,y,fits,b,gamma,target):
    result={m:[] for m in MODES}; priors=[]
    restored=np.einsum('k,kad->ad',target,gamma)
    for idx in fits:
        if not len(idx):
            for m in MODES: result[m].append(np.zeros_like(b))
            priors.append(np.zeros(4)); continue
        yi=y[idx]; g=example_gradients(x[idx],yi,b); prior=np.bincount(yi,minlength=4)/len(yi)
        raw=g.mean(0); means=np.stack([g[yi==k].mean(0) if np.any(yi==k) else gamma[k] for k in range(4)])
        balanced=np.einsum('k,kad->ad',target,means)
        center=raw-np.einsum('k,kad->ad',prior,gamma)
        values=(raw,balanced,raw-restored,center,center,balanced-restored)
        for m,v in zip(MODES,values): result[m].append(v)
        priors.append(prior)
    return {m:np.array(v) for m,v in result.items()},np.array(priors)


def public_offset(mode,gradient,geometry):
    if mode=='center_only': return np.zeros_like(gradient)
    if mode in ('raw','target_balanced'):
        return gradient-(gradient.ravel()@geometry['encoder']@geometry['decoder']).reshape(3,17)
    return gradient.copy()


def objective(scenario):
    return np.array([.4,.2,.2,.2]) if scenario in ('prior','both') else np.ones(4)/4


def scenario_fits(fits,y,scenario):
    if scenario not in ('prior','both'): return [a.copy() for a in fits]
    return [np.concatenate([a[y[a]==k][:len(a[y[a]==k]) if k==0 else len(a[y[a]==k])//2] for k in range(4)]) for a in fits]


def context():
    source,ids,pub,x,y,dev,*_=prepare(False)
    labels=np.array(Dataset.from_file(str(source))['label'])
    old=np.load(ROOT/'public_residual_development.npz')
    return source,ids,pub,labels[ids['public']],x,y,dev,old['reference'],old['public_hessian']


def public_controls(pub,py,dev,b,h,scenario):
    mask=scenario in ('mask','both'); prior=objective(scenario)
    px=transform(pub,mask); dx=transform(dev[0],mask); dy=dev[1]
    gamma=class_gradients(px,py,b); gradient=np.einsum('k,kad->ad',prior,gamma)
    candidates=[('reference',b),('prior_corrected_reference',prior_correct(b,prior))]
    oldg=public_geometry(h,3,'euclidean'); candidates.append(('previous_decoder_offset',b-.3*(b.ravel()@oldg['basis']@oldg['basis'].T).reshape(3,17)))
    for d in (1,3,12,51):
        g=public_geometry(h,d,'euclidean'); projected=(gradient.ravel()@g['basis']@g['basis'].T).reshape(3,17)
        for eta in (0.,)+ETAS:
            theta=b-eta*projected
            candidates.extend([(f'public_gradient_d{d}_eta{eta}',theta),(f'public_gradient_prior_d{d}_eta{eta}',prior_correct(theta,prior))])
    for steps in (80,320,1280):
        theta=local_query(px,py,steps)
        for mult in (.5,.75,1.,1.25,1.5,2.):
            candidates.extend([(f'public_refit_steps{steps}_mult{mult}',theta*mult),(f'public_refit_prior_steps{steps}_mult{mult}',prior_correct(theta*mult,prior))])
    values=[]
    for name,theta in candidates:
        ce,acc=weighted_scores(dx,dy,theta[None],prior)
        values.append({'name':name,'theta':theta.tolist(),'development_ce':float(ce[0]),'development_accuracy':float(acc[0])})
    return {'scenario':scenario,'prior':prior.tolist(),'candidates':values,'selected':min(values,key=lambda z:z['development_ce'])}


def development():
    source,ids,pub,py,x,y,dev,b,h=context()
    saved={'reference':b,'public_hessian':h}; gamma=class_gradients(pub,py,b); saved['gamma']=gamma
    controls=[public_controls(pub,py,dev,b,h,s) for s in SCENARIOS]
    for r in controls: print('public control',r['scenario'],r['selected']['name'],r['selected']['development_ce'],flush=True)
    rows=[]
    for seed in (42,43,44):
        fits,_=split(partition(y,'label_stress',seed),y,seed)
        q,priors=client_queries(x,y,fits,b,gamma,objective('baseline'))
        for risk in (.55,.65):
            innovations=np.stack([random_shares(32,101000000+seed*100000+int(risk*100)*1000+s).sum(1)/np.sqrt(7) for s in range(8)])
            candidates=[]; selected={}
            for mode in MODES:
                for d in (1,3,12,51):
                    for metric in ('euclidean','fisher'):
                        g=public_geometry(h,d,metric); query=q[mode].reshape(8,51)@g['encoder']
                        offset=public_offset(mode,gamma.mean(0),g)
                        for cap in CAPS:
                            signal,factors=bounded_public(query,cap); sigma=public_scale(cap,risk)
                            means=np.tile(signal.sum(0),(8,1))
                            for t in range(4): means[2*t]-=signal[t]
                            noisy=(means[:,None]+sigma*innovations[:,:,:d]).reshape(-1,d)
                            decoded=(noisy@g['decoder']).reshape(-1,3,17)+offset
                            for eta in ETAS:
                                ce,acc=weighted_scores(*dev,b[None]-eta*decoded,objective('baseline'))
                                config={'mode':mode,'dimension':d,'metric':metric,'cap':cap,'eta':eta}
                                key=f'seed{seed}_q{int(risk*100)}_candidate{len(candidates)}'
                                saved[key+'_ce']=ce.reshape(8,32)
                                record={'configuration':config,'development_ce':float(ce.mean()),'clip_factors':factors.tolist(),'archive_key':key}
                                candidates.append(record)
                                if mode not in selected or record['development_ce']<selected[mode]['development_ce']: selected[mode]=record
            rows.append({'seed':seed,'risk':risk,'candidates':candidates,'selected':selected})
            print('development',seed,risk,'class_center',selected['class_center']['development_ce'],'best',min(z['development_ce'] for z in selected.values()),flush=True)
    # All stress queries are constructed before opening fresh utility images.
    for seed in (42,43,44):
        fits,_=split(partition(y,'label_stress',seed),y,seed)
        for s in SCENARIOS:
            sf=scenario_fits(fits,y,s); sx=transform(x,s in ('mask','both')); sp=transform(pub,s in ('mask','both'))
            sources=('stale','refreshed') if s in ('mask','both') else ('stale',)
            for route in sources:
                sg=gamma if route=='stale' else class_gradients(sp,py,b)
                queries,priors=client_queries(sx,y,sf,b,sg,objective(s))
                prefix=f'seed{seed}_{s}_{route}'
                saved[prefix+'_gamma']=sg; saved[prefix+'_priors']=priors
                saved[prefix+'_counts']=np.array([np.bincount(y[a],minlength=4) for a in sf])
                saved[prefix+'_mean_client_prior']=priors.mean(0)
                counts=saved[prefix+'_counts']; saved[prefix+'_pooled_record_prior']=counts.sum(0)/counts.sum()
                for mode,query in queries.items(): saved[prefix+'_'+mode+'_query']=query
    np.savez_compressed(BANK,**saved)
    DEV.write_text(json.dumps({'source_hash':digest(source),'protocol_sha':digest(PROTOCOL),'calculator_sha':digest(__file__),
        'dependencies':{str(p):digest(p) for p in DEPS},'public_controls':controls,'rows':rows,
        'bank_sha':digest(BANK),'draws_per_stratum':32,'candidate_count':sum(len(r['candidates']) for r in rows)},indent=2,allow_nan=False)+'\n')


def evaluation():
    development=json.loads(DEV.read_text()); assert digest(BANK)==development['bank_sha']; bank=np.load(BANK)
    assert digest(PROTOCOL)==development['protocol_sha'] and digest(__file__)==development['calculator_sha']
    for path,sha in development['dependencies'].items(): assert digest(path)==sha
    source,ids,pub,py,x,y,dev,b,h=context(); assert digest(source)==development['source_hash']
    data=Dataset.from_file(str(source)); labels=np.array(data['label'])
    unused=np.load(ROOT/'public_residual_evaluation.npz')['remaining_reserve_original_indices']
    new=np.concatenate([unused[labels[unused]==k][:80] for k in range(4)])
    remaining=np.setdiff1d(unused,new)
    assert len(new)==320 and len(np.unique(new))==320 and len(remaining)==136
    for name in ('public','private','development','evaluation'): assert not set(new)&set(ids[name])
    for file,key in (('private_prior_constructor_evaluation.npz','confirmation_original_indices'),('matched_cia_evaluation.npz','utility_original_indices'),('public_residual_evaluation.npz','utility_original_indices')):
        assert not set(new)&set(np.load(ROOT/file)[key])
    ux=features(data,new); uy=labels[new]
    saved={'utility_original_indices':new,'remaining_reserve_original_indices':remaining}; rows=[]; controls=[]
    for control in development['public_controls']:
        s=control['scenario']; ce,acc=weighted_scores(transform(ux,s in ('mask','both')),uy,np.array(control['selected']['theta'])[None],objective(s))
        controls.append({'scenario':s,'selection':control['selected']['name'],'ce':float(ce[0]),'accuracy':float(acc[0])})
    for r in development['rows']:
        seed,risk=r['seed'],r['risk']; scenarios=SCENARIOS if risk==.55 else ('baseline',)
        for s in scenarios:
            routes=('stale','refreshed') if s in ('mask','both') else ('stale',)
            for route in routes:
                prefix=f'seed{seed}_{s}_{route}'; sg=bank[prefix+'_gamma']; gradient=np.einsum('k,kad->ad',objective(s),sg)
                tx=transform(ux,s in ('mask','both')); arms=dict(r['selected']); arms['central_class_center']=r['selected']['class_center']
                for arm,choice in arms.items():
                    config=choice['configuration']; d=config['dimension']; g=public_geometry(h,d,config['metric'])
                    query=bank[prefix+'_'+config['mode']+'_query'].reshape(8,51)@g['encoder']
                    signal,factors=bounded_public(query,config['cap']); sigma=public_scale(config['cap'],risk)
                    offset=public_offset(config['mode'],gradient,g); central=arm=='central_class_center'
                    saved[f'{prefix}_q{int(risk*100)}_{arm}_signal']=signal
                    saved[f'{prefix}_q{int(risk*100)}_{arm}_offset']=offset
                    for target in range(8):
                        peer=6 if target==7 else 7; out=signal.copy(); out[target]=0
                        shift=signal[target]; base=out.sum(0) if central else out.sum(0)-out[peer]
                        key=f'{prefix}_q{int(risk*100)}_{arm}_target{target}'
                        attack=[]; utility={}; noiseless={}
                        for world in (0,1):
                            active=signal if world else out
                            rng=102000000+seed*100000+int(risk*100)*1000+target*10+world
                            objects=active[None]+random_shares(2048,rng)[:,:,:d]*sigma/np.sqrt(8 if central else 7)
                            observed=objects.sum(1) if central else objects.sum(1)-objects[:,peer]
                            rank=(observed-base-shift/2)@shift/sigma**2
                            saved[key+f'_world{world}_attack_score']=rank; attack.append(rank)
                            noise=random_shares(128,rng+10000000)[:,:,:d].sum(1)*sigma/np.sqrt(8 if central else 7)
                            theta=b[None]-config['eta']*(offset[None]+((active.sum(0)[None]+noise)@g['decoder']).reshape(-1,3,17))
                            ce,acc=weighted_scores(tx,uy,theta,objective(s)); unce,unacc=scores(tx,uy,theta)
                            for field,v in (('ce',ce),('accuracy',acc),('unweighted_ce',unce),('unweighted_accuracy',unacc)): saved[key+f'_world{world}_utility_{field}']=v
                            utility[str(world)]={'ce':summarize(ce),'accuracy':summarize(acc)}
                            no=b-config['eta']*(offset+(active.sum(0)@g['decoder']).reshape(3,17))
                            a,z=weighted_scores(tx,uy,no[None],objective(s)); noiseless[str(world)]={'ce':float(a[0]),'accuracy':float(z[0])}
                        expected=float(ndtr(np.linalg.norm(shift)/(np.sqrt(2)*sigma))); assert expected<=risk+1e-12
                        rows.append({'seed':seed,'risk':risk,'scenario':s,'reference_source':route,'arm':arm,'target':target,'peer':peer,'archive_key':key,
                            'configuration':config,'sigma':sigma,'clip_factors':factors.tolist(),'target_shift_norm':float(np.linalg.norm(shift)),
                            'gaussian_expected_auc':expected,'attack':auc_statistics(attack[1],attack[0]),'utility':utility,'noiseless':noiseless})
                print('evaluation',seed,risk,s,route,len(rows),'rows',flush=True)
    np.savez_compressed(ROOT/'class_conditional_evaluation.npz',**saved)
    (ROOT/'class_conditional_evaluation.json').write_text(json.dumps({'development_sha':digest(DEV),'development_archive_sha':digest(BANK),
        'calculator_sha':digest(__file__),'source_hash':digest(source),'utility_count':len(new),'remaining_reserve_count':len(remaining),'public_controls':controls,'rows':rows},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=('development','evaluation'),default='development'); args=parser.parse_args()
    development() if args.stage=='development' else evaluation()
