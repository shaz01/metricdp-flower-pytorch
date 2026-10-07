"""Independent saved-score arithmetic audit, without importing CIA simulator."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr

ROOT=Path('results/client_specific_noise')


def stats(p,n):
    pn=np.sort(n);pp=np.sort(p)
    vp=(np.searchsorted(pn,p,side='left')+np.searchsorted(pn,p,side='right'))/(2*len(n))
    vn=1-(np.searchsorted(pp,n,side='left')+np.searchsorted(pp,n,side='right'))/(2*len(p))
    auc=vp.mean();se=np.sqrt(vp.var(ddof=1)/len(p)+vn.var(ddof=1)/len(n))
    thresholds=np.unique(np.r_[p,n])
    fpr=(len(n)-np.searchsorted(pn,thresholds,side='left'))/len(n)
    tpr=(len(p)-np.searchsorted(pp,thresholds,side='left'))/len(p)
    return {'auc':float(auc),'se':float(se),'ci95':[max(0.,auc-1.96*se),min(1.,auc+1.96*se)],
            'tpr_at_fpr01':float(tpr[fpr<=.01].max(initial=0)),'tpr_at_fpr05':float(tpr[fpr<=.05].max(initial=0))}


def main():
    data=json.loads((ROOT/'descriptor_cia_probe.json').read_text());a=np.load(ROOT/'descriptor_cia_probe.npz')
    dev=json.loads((ROOT/'private_prior_constructor_development.json').read_text())
    assert hashlib.sha256((ROOT/'private_prior_constructor_development.json').read_bytes()).hexdigest()==data['development_sha']
    assert hashlib.sha256(Path('research/proposals/2026-10-07_descriptor_cia_protocol.md').read_bytes()).hexdigest()==data['protocol_sha']
    assert hashlib.sha256(Path('research/calculations/descriptor_cia_probe.py').read_bytes()).hexdigest()==data['calculator_sha']
    assert data['targets']==[0,1,2,3] and len(data['rows'])==156
    expected={(seed,target,law,arm) for seed in (42,43,44) for target in (0,1,2,3) for law in ('analytic_gaussian','radial_laplace') for arm in ('aggregate_votes','aggregate_control','central_votes','individual_votes')}
    expected.update((seed,target,'metric_adaptation',f'metric_{multiplier}') for seed in (42,43,44) for target in (0,1,2,3) for multiplier in (.1,.3,1.,3.,10.))
    keys=[(r['seed'],r['target'],r['law'],r['arm']) for r in data['rows']]
    assert len(set(keys))==156 and set(keys)==expected
    errors=[];checks=0;g=[]
    def compare(actual,expected):
        nonlocal checks
        if isinstance(expected,dict):
            for k,v in expected.items():compare(actual[k],v)
        elif isinstance(expected,(tuple,list)):
            for x,y in zip(actual,expected):compare(x,y)
        else:
            error=abs(float(actual)-float(expected));errors.append(error);checks+=1;assert error<1e-11,(actual,expected,error)
    for r in data['rows']:
        key=f"seed{r['seed']}_target{r['target']}_{r['law']}_{r['arm']}"
        for view in ('descriptor','student'):
            p=a[key+'_world1_'+view+'_score'];n=a[key+'_world0_'+view+'_score']
            assert len(p)==len(n)==1024 and np.isfinite(p).all() and np.isfinite(n).all()
            compare(r[view],stats(p,n))
        values={}
        for name in r['validation_auc']:
            p=a[key+'_world1_validation_'+name];n=a[key+'_world0_validation_'+name]
            assert len(p)==len(n)==256;values[name]=stats(p,n)['auc'];compare(r['validation_auc'][name],values[name])
        assert max(values,key=values.get)==r['student_attack']
        for world in ('0','1'):
            for field in ('ce','accuracy'):
                v=a[key+'_world'+world+'_utility_'+field];assert len(v)==128
                compare(r['utility'][world][field],v.mean())
        selected=next(v for v in dev['rows'] if v['seed']==r['seed'] and v['law']==('analytic_gaussian' if r['law']=='metric_adaptation' else r['law']))
        if r['law']=='metric_adaptation':
            candidates=[v for v in selected['selected'].values() if v['contract']=='aggregate' and v['configuration']['family']=='model_direct']
            choice=min(candidates,key=lambda v:v['development_ce']);assert r['student_decoder_identity_verified'] and r['configuration']['ridge']==0
            compare(r['student_exact_llr'],r['descriptor'])
            c=r['metric_operator_calibration']
            for world in ('out','in'):compare(c[world+'_std'],c['multiplier']*r['configuration']['radius']/(8*c[world+'_distance']))
        else:
            contract=r['contract'];which='control' if r['arm']=='aggregate_control' else 'candidate'
            choice=selected['selected'][selected['chosen'][contract][which]]
            c=r['operator_calibration']['1'];compare(r['weighted_cap'],r['configuration']['radius']/8)
            compare(c['weighted_cap'],r['weighted_cap'])
            variance=r['weighted_cap']**2*(.3602749391128144 if r['law']=='analytic_gaussian' else (c['dimension']+1)/64)
            compare(c['unknown_floor_variance'],variance)
            compare(c['aggregate_variance'],variance*{'aggregate':8/7,'central':1,'individual':8}[contract])
            if r['law']=='analytic_gaussian':
                expected=ndtr(r['target_shift_norm']/np.sqrt(2*variance));compare(r['expected_gaussian_auc'],expected)
                deviation=abs(r['descriptor']['auc']-expected)/max(r['descriptor']['se'],1e-12);g.append(deviation);assert deviation<4
        assert r['configuration']==choice['configuration'] and r['transform']==choice['transform']
    prior=np.load(ROOT/'private_prior_constructor_evaluation.npz')
    assert np.array_equal(a['utility_original_indices'],prior['confirmation_original_indices'])
    result={'rows':156,'certified_channel_rows':96,'adapted_metric_rows':60,'numeric_checks':checks,'maximum_absolute_error':max(errors),
            'gaussian_expected_auc_checks':len(g),'maximum_gaussian_deviation_in_standard_errors':max(g),
            'target_classes_covered':4,'frozen_defense_and_validation_attack_choices_verified':True,'prior_utility_indices_unchanged':True}
    (ROOT/'descriptor_cia_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
