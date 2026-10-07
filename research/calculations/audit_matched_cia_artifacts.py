"""Independent saved-artifact audit and fixed-stratum contrast synthesis."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri

ROOT=Path('results/client_specific_noise')


def rank_statistics(p,n):
    # Count comparisons with independently sorted arrays; no simulator import.
    nsort=np.sort(n);psort=np.sort(p)
    positive=(np.searchsorted(nsort,p,'left')+.5*(np.searchsorted(nsort,p,'right')-np.searchsorted(nsort,p,'left')))/len(n)
    negative=(len(p)-np.searchsorted(psort,n,'right')+.5*(np.searchsorted(psort,n,'right')-np.searchsorted(psort,n,'left')))/len(p)
    auc=float(positive.mean());se=float(np.sqrt(positive.var(ddof=1)/len(p)+negative.var(ddof=1)/len(n)))
    cut=np.unique(np.r_[p,n]);fp=(len(n)-np.searchsorted(nsort,cut,'left'))/len(n);tp=(len(p)-np.searchsorted(psort,cut,'left'))/len(p)
    return {'auc':auc,'se':se,'ci95':[max(0.,auc-1.96*se),min(1.,auc+1.96*se)],'tpr_at_fpr01':float(tp[fp<=.01].max(initial=0)),'tpr_at_fpr05':float(tp[fp<=.05].max(initial=0))}


def main():
    d=json.loads((ROOT/'matched_cia_development.json').read_text());a=np.load(ROOT/'matched_cia_development.npz')
    e=json.loads((ROOT/'matched_cia_evaluation.json').read_text());b=np.load(ROOT/'matched_cia_evaluation.npz')
    assert hashlib.sha256((ROOT/'matched_cia_development.json').read_bytes()).hexdigest()==e['development_sha']
    assert hashlib.sha256(Path('research/proposals/2026-10-07_matched_cia_protocol.md').read_bytes()).hexdigest()==d['protocol_sha']
    assert hashlib.sha256(Path('research/calculations/matched_cia_probe.py').read_bytes()).hexdigest()==e['calculator_sha']
    assert len(d['rows'])==18 and len(e['rows'])==936
    numeric=0;maximum=0.
    def compare(actual,expected):
        nonlocal numeric,maximum
        if isinstance(expected,dict):
            for key,value in expected.items():compare(actual[key],value)
        elif isinstance(expected,list):
            for x,y in zip(actual,expected):compare(x,y)
        else:
            error=abs(float(actual)-float(expected));numeric+=1;maximum=max(maximum,error);assert error<1e-11,(actual,expected,error)
    selection={}
    for row in d['rows']:
        selection[(row['seed'],row['risk'],row['law'])]=row
        assert len(row['candidates'])==972
        for c in row['candidates']:
            v=a[c['archive_key']+'_ce'];assert len(v)==32;compare(c['development_ce'],v.mean())
        for family,c in row['selected'].items():assert c==min((v for v in row['candidates'] if v['configuration']['family']==family),key=lambda v:v['development_ce'])
        assert row['control']==min((f for f in row['selected'] if f!='votes'),key=lambda f:row['selected'][f]['development_ce'])
    for c in d['radial_calibrations']:
        key=f"calibration_d{c['dimension']}_q{int(c['risk']*100)}"
        compare(c['statistics'],rank_statistics(a[key+'_world1_score'],a[key+'_world0_score']))
        assert abs(c['statistics']['auc']-c['risk'])<1e-6
    seen=set();gaussian_z=[]
    for row in e['rows']:
        key=f"seed{row['seed']}_q{int(row['risk']*100)}_{row['law_selection']}_{row['arm']}_target{row['target']}"
        assert key not in seen;seen.add(key);assert row['target']!=row['peer']
        selected=selection[(row['seed'],row['risk'],row['law_selection'])]
        source=selected['selected'][row['arm'] if row['arm'] in selected['selected'] else 'votes']
        assert row['configuration']==source['configuration'] and row['transform']==source['transform']
        cal=row['calibration'];compare(cal['fixed_M'],source['calibration']['fixed_M'])
        compare(cal['unknown_variance'],cal['scale']**2*(1 if cal['law']=='gaussian' else cal['dimension']+1))
        if row['arm']=='gaussian_same_query':compare(cal['scale'],cal['fixed_M']/(np.sqrt(2)*ndtri(row['risk'])))
        elif row['arm']=='gaussian_variance_match':compare(cal['unknown_variance'],source['calibration']['unknown_variance'])
        else:assert cal==source['calibration']
        compare(row['attack'],rank_statistics(b[key+'_world1_attack_score'],b[key+'_world0_attack_score']))
        for world in (0,1):
            for field in ('ce','accuracy'):
                v=b[key+f'_world{world}_utility_{field}'];assert len(v)==256
                compare(row['utility'][str(world)][field]['mean'],v.mean());compare(row['utility'][str(world)][field]['se'],v.std(ddof=1)/np.sqrt(len(v)))
        assert row['target_shift_norm']<=cal['fixed_M']+1e-12
        if cal['law']=='gaussian':
            expected=float(ndtr(row['target_shift_norm']/(np.sqrt(2)*cal['scale'])));compare(row['gaussian_expected_auc'],expected)
            z=abs(row['attack']['auc']-expected)/row['attack']['se'];gaussian_z.append(z);assert z<4.5
    previous=np.load(ROOT/'private_prior_constructor_evaluation.npz')
    new=set(b['utility_original_indices']);remaining=set(b['remaining_reserve_original_indices'])
    assert len(new)==1024 and len(remaining)==968 and not new&remaining and new|remaining==set(previous['remaining_reserve_original_indices'])
    # Previous unused reserve was independently audited as disjoint from every role.
    original=np.load(ROOT/'private_descriptor_analytic_gaussian.npz')
    for role in ('public','private','development','evaluation'):assert not new&set(original[role+'_original_indices'])
    assert not new&set(previous['confirmation_original_indices'])
    contrasts=[]
    for seed in (42,43,44):
        for risk in (.55,.65,.8):
            control=selection[(seed,risk,'gaussian')]['control']
            for law,arm in [('radial','gaussian_same_query'),('radial','gaussian_variance_match'),('gaussian',control),('radial','central_same_query')]:
                strata=[]
                for target in range(4):
                    first=f'seed{seed}_q{int(risk*100)}_radial_votes_target{target}'
                    second=f'seed{seed}_q{int(risk*100)}_{law}_{arm}_target{target}'
                    for world in (0,1):strata.append(b[second+f'_world{world}_utility_ce']-b[first+f'_world{world}_utility_ce'])
                gain=float(sum(v.mean() for v in strata)/8);se=float(np.sqrt(sum(v.var(ddof=1)/len(v) for v in strata))/8)
                contrasts.append({'seed':seed,'risk':risk,'candidate':'radial_votes','reference_law':law,'reference_arm':arm,'strata':8,'gain':gain,'se':se,'ci95':[gain-1.96*se,gain+1.96*se]})
    (ROOT/'matched_cia_contrasts.json').write_text(json.dumps({'kind':'fixed primary targets0-3,equal-world stratified paired noise intervals;not population or simultaneous intervals','rows':contrasts},indent=2)+'\n')
    result={'development_configurations':17496,'development_rows':18,'attack_rows':936,'arms':117,'numeric_checks':numeric,'maximum_absolute_error':maximum,'gaussian_expected_auc_checks':len(gaussian_z),'maximum_gaussian_deviation_in_standard_errors':max(gaussian_z),'contrast_rows':len(contrasts),'fresh_utility_count':1024,'remaining_reserve_count':968,'indices_and_frozen_choices_verified':True}
    (ROOT/'matched_cia_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
