"""Development-only raw private headroom; no reserve access or privacy claim."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_geometry_audit import partition,split
from research.calculations.client_energy_filter_probe import C,scores
from research.calculations.client_energy_one_release import local_query
from research.calculations.public_residual_probe import digest,public_hessian,public_geometry
from research.calculations.public_residual_signal_diagnostic import example_gradients
from research.calculations.class_conditional_probe import client_queries,public_offset

PROTOCOL=Path('research/proposals/2026-10-08_headroom_protocol.md')
CODE=Path('research/calculations/headroom_probe.py')
DEPS=tuple(Path('research/calculations')/(n+'.py') for n in ('private_prior_constructor_probe','private_descriptor_probe','client_geometry_audit','client_energy_filter_probe','client_energy_one_release','public_residual_probe','public_residual_signal_diagnostic','class_conditional_probe'))+(ROOT/'private_descriptor_analytic_gaussian.npz',ROOT/'class_conditional_development.json')


def public_subset(ids,labels,budget,seed):
    rng=np.random.default_rng(seed); groups=[]
    for k in range(4):
        a=np.asarray(ids)[labels[np.asarray(ids)]==k].copy();rng.shuffle(a);groups.extend(a[:budget//4])
    return np.array(groups,dtype=int)


def development_halves(ids,labels):
    return tuple(np.concatenate([np.asarray(ids)[labels[np.asarray(ids)]==k][start:start+128] for k in range(4)]) for start in (0,128))


def residual_moments(z):
    mean=z.mean(0);var=float(np.sum((z-mean)**2)/(len(z)-1)/len(z))
    return mean,var


def stratified_variance(z,y):
    return float(sum(np.sum((z[y==k]-z[y==k].mean(0))**2)/(np.sum(y==k)-1)*np.sum(y==k) for k in range(4) if np.sum(y==k)>1)/len(y)**2)


def refine(x,y,b,steps):
    theta=b.copy()
    for _ in range(steps): theta-=.5*example_gradients(x,y,theta).mean(0)
    return theta


def main():
    source,ids,public,private,y,dev,*_=prepare(False); labels=np.array(Dataset.from_file(str(source))['label'])
    selection_ids,assessment_ids=development_halves(ids['development'],labels)
    # Features already opened by prepare belong only to public/private/OLDdev roles.
    lookup={int(v):i for i,v in enumerate(ids['development'])}
    selection=(dev[0][[lookup[int(v)] for v in selection_ids]],labels[selection_ids]); assessment=(dev[0][[lookup[int(v)] for v in assessment_ids]],labels[assessment_ids])
    fits,checks=split(partition(y,'label_stress',42),y,42); cohorts={'A':fits,'B':checks}
    assert not set(np.concatenate(fits))&set(np.concatenate(checks))
    assert set(np.concatenate(fits))|set(np.concatenate(checks))==set(range(len(y)))
    saved={'selection_original_indices':selection_ids,'assessment_original_indices':assessment_ids}
    for name,parts in cohorts.items(): saved['cohort'+name+'_original_indices']=ids['private'][np.stack(parts)]
    public_lookup={int(v):i for i,v in enumerate(ids['public'])}; pubrows=[];rows=[];serial=0
    def record(theta,family,config):
        nonlocal serial
        key=f'model{serial}';serial+=1;saved[key]=theta
        sc,sa=scores(*selection,theta[None]);ac,aa=scores(*assessment,theta[None])
        return {'family':family,'configuration':config,'archive_key':key,'selection_ce':float(sc[0]),'selection_accuracy':float(sa[0]),'assessment_ce':float(ac[0]),'assessment_accuracy':float(aa[0])}
    for budget in (32,128,512):
        for seed in (42,43,44):
            pubids=public_subset(ids['public'],labels,budget,seed); px=public[[public_lookup[int(v)] for v in pubids]];py=labels[pubids];prefix=f'budget{budget}_seed{seed}';saved[prefix+'_public_original_indices']=pubids
            candidates=[]
            for steps in (80,320,1280,5120):
                theta=local_query(px,py,steps)
                for mult in (.5,.75,1.,1.25,1.5,2.): candidates.append(record(theta*mult,'public_scratch',{'steps':steps,'multiplier':mult}))
            reference=min(candidates,key=lambda r:r['selection_ce']); b=saved[reference['archive_key']]; saved[prefix+'_reference']=b
            gamma=np.stack([example_gradients(px[py==k],py[py==k],b).mean(0) for k in range(4)]); h=public_hessian(px,b);saved[prefix+'_gamma']=gamma;saved[prefix+'_hessian']=h
            for steps in (20,80,320,1280): candidates.append(record(refine(px,py,b,steps),'public_refine',{'steps':steps}))
            for d in (1,3,12,51):
                g=public_geometry(h,d,'euclidean');grad=(gamma.mean(0).ravel()@g['basis']@g['basis'].T).reshape(3,17)
                for eta in (.1,.3,1.,3.,10.,30.): candidates.append(record(b-eta*grad,'public_step',{'dimension':d,'eta':eta}))
            if budget==512:
                historical=json.loads((ROOT/'class_conditional_development.json').read_text())['public_controls'][0]['selected']['theta']; candidates.append(record(np.array(historical),'historical_public_anchor',{}))
            strong=min(candidates,key=lambda r:r['selection_ce']);pubrows.append({'budget':budget,'subset_seed':seed,'reference':reference,'candidates':candidates,'selected':strong})
            for name,parts in cohorts.items():
                pooled=np.concatenate(parts); xp=np.concatenate((px,private[pooled]));yp=np.concatenate((py,y[pooled]));private_candidates=[]
                for steps in (80,320,1280,5120):
                    theta=local_query(xp,yp,steps)
                    for mult in (.5,.75,1.,1.25,1.5,2.): private_candidates.append(record(theta*mult,'pooled_scratch',{'steps':steps,'multiplier':mult}))
                for steps in (20,80,320,1280):
                    private_candidates.append(record(refine(xp,yp,b,steps),'pooled_refine',{'steps':steps}))
                    private_candidates.append(record(refine(private[pooled],y[pooled],b,steps),'private_refine',{'steps':steps}))
                queries,priors=client_queries(private,y,parts,b,gamma,np.ones(4)/4)
                cp=prefix+'_cohort'+name;saved[cp+'_priors']=priors
                for mode,q in queries.items(): saved[cp+'_'+mode]=q
                variance=[];alignment=[];ag=example_gradients(*assessment,b).mean(0)
                for slot,indices in enumerate(parts):
                    yi=y[indices];raw=example_gradients(private[indices],yi,b);z=raw-gamma[yi];r,v=residual_moments(z)
                    variance.append({'slot':slot,'residual_norm_squared':float(np.sum(r*r)),'iid_mean_variance':v,'stratified_mean_variance':stratified_variance(z,yi),'raw_stratified_mean_variance':stratified_variance(raw,yi),'assessment_descent_inner_product':float(np.sum(ag*r))})
                for mode in ('raw','class_center','center_only','target_balanced'):
                    for d in (1,3,12,51):
                        g=public_geometry(h,d,'euclidean');gradient=public_offset(mode,gamma.mean(0),g)+(queries[mode].mean(0).ravel()@g['encoder']@g['decoder']).reshape(3,17)
                        for eta in (.1,.3,1.,3.,10.,30.): private_candidates.append(record(b-eta*gradient,'step_'+mode,{'dimension':d,'eta':eta}))
                selected={family:min((c for c in private_candidates if c['family']==family),key=lambda c:c['selection_ce']) for family in sorted({c['family'] for c in private_candidates})}
                oracle=min((v for k,v in selected.items() if not k.startswith('step_')),key=lambda c:c['selection_ce'])
                one=min((v for k,v in selected.items() if k.startswith('step_')),key=lambda c:c['selection_ce'])
                aggregate=queries['class_center'].mean(0).ravel(); norm2=float(aggregate@aggregate)
                fractions={str(d):float(np.sum((aggregate@public_geometry(h,d,'euclidean')['basis'])**2)/norm2) if norm2 else 0. for d in (1,3,12,51)}
                decomposition={'aggregate_residual_norm_squared':norm2,'aggregate_stratified_mean_variance':sum(v['stratified_mean_variance'] for v in variance)/64,'projected_mean_fractions':fractions}
                rows.append({'budget':budget,'subset_seed':seed,'cohort':name,'candidates':private_candidates,'selected':selected,'selected_oracle':oracle,'selected_one_step':one,'public_selected':strong,
                    'assessment_oracle_gain':strong['assessment_ce']-oracle['assessment_ce'],'assessment_one_step_gain':strong['assessment_ce']-one['assessment_ce'],'client_residual_diagnostics':variance,'residual_decomposition':decomposition})
                print('headroom',budget,seed,name,'oracle',rows[-1]['assessment_oracle_gain'],'one-step',rows[-1]['assessment_one_step_gain'],flush=True)
    bank=ROOT/'headroom_development.npz';np.savez_compressed(bank,**saved)
    (ROOT/'headroom_development.json').write_text(json.dumps({'kind':'Reused development-only unprotected headroom; no reserve, noise, CIA or privacy claim.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'bank_sha':digest(bank),'source_hash':digest(source),'dependencies':{str(p):digest(p) for p in DEPS},'public_rows':pubrows,'rows':rows,'model_count':serial,'unique_private_cohort_count':2,'selection_count':len(selection_ids),'assessment_count':len(assessment_ids)},indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
