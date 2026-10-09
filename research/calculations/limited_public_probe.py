"""Development-only projection/clipping/noise decomposition of saved headroom queries; no reserve, CIA or privacy claim."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.client_energy_filter_probe import scores
from research.calculations.public_residual_probe import digest, public_geometry, bounded_public, public_scale, random_shares
from research.calculations.class_conditional_probe import public_offset
from research.calculations.headroom_probe import development_halves

PROTOCOL=Path('research/proposals/2026-10-08_limited_public_protocol.md')
CODE=Path('research/calculations/limited_public_probe.py')
HEAD=ROOT/'headroom_development.json'
BANK=ROOT/'headroom_development.npz'
OUT=ROOT/'limited_public_development.json'
MODES=('class_center','target_balanced')
DIMS=(1,3,12,51)
CAPS=(.003,.01,.03,.1,.3)
ETAS=(.1,.3,1.,3.,10.,30.)
RISKS=(.55,.65,.80)
UNCLIPPED=1e9
SELECTION_DRAWS=32
ASSESSMENT_DRAWS=128


def noise_total(draws,seed,d,sigma):
    """Peer-contract IN-world noise: eight shares summed, divided by sqrt(7), as in the evaluated CIA contract."""
    return random_shares(draws,seed)[:,:,:d].sum(1)*sigma/np.sqrt(7)


def step(b,eta,offset,total,decoder):
    return b[None]-eta*(offset[None]+(np.atleast_2d(total)@decoder).reshape(-1,3,17))


def mean_scores(data,theta):
    ce,acc=scores(*data,theta)
    return float(ce.mean()),float(acc.mean())


def best(candidates):
    return min(candidates,key=lambda c:c['selection_ce'])


def main():
    source,ids,_pub,_priv,_y,dev,*_rest=prepare(False); labels=np.array(Dataset.from_file(str(source))['label'])
    selection_ids,assessment_ids=development_halves(ids['development'],labels)
    lookup={int(v):i for i,v in enumerate(ids['development'])}
    selection=(dev[0][[lookup[int(v)] for v in selection_ids]],labels[selection_ids]); assessment=(dev[0][[lookup[int(v)] for v in assessment_ids]],labels[assessment_ids])
    head=json.loads(HEAD.read_text()); bank=np.load(BANK); assert digest(BANK)==head['bank_sha']
    assert np.array_equal(bank['selection_original_indices'],selection_ids) and np.array_equal(bank['assessment_original_indices'],assessment_ids)
    publics={(r['budget'],r['subset_seed']):r['selected'] for r in head['public_rows']}
    rows=[]
    for budget in (32,128,512):
        for seed in (42,43,44):
            prefix=f'budget{budget}_seed{seed}'; b=bank[prefix+'_reference']; gamma=bank[prefix+'_gamma']; h=bank[prefix+'_hessian']; public=publics[(budget,seed)]
            for ci,cohort in enumerate('AB'):
                cp=prefix+'_cohort'+cohort; base=301000000+budget*10000+seed*100+ci
                stages={}
                def add(stage,mode,d,eta,cap,risk,theta):
                    sel=mean_scores(selection,theta)
                    stages.setdefault((stage,risk,mode),[]).append({'configuration':{'stage':stage,'mode':mode,'dimension':d,'eta':eta,'cap':cap,'risk':risk},'selection_ce':sel[0],'selection_accuracy':sel[1]})
                    return stages[(stage,risk,mode)][-1]
                for mode in MODES:
                    query_all=bank[cp+'_'+mode].reshape(8,51)
                    for d in DIMS:
                        g=public_geometry(h,d,'euclidean'); query=query_all@g['encoder']; offset=public_offset(mode,gamma.mean(0),g)
                        for eta in ETAS:
                            add('public_zero',mode,d,eta,None,None,step(b,eta,offset,np.zeros(d),g['decoder']))
                            signal,_=bounded_public(query,UNCLIPPED)
                            add('unclipped',mode,d,eta,None,None,step(b,eta,offset,signal.sum(0),g['decoder']))
                        for cap in CAPS:
                            signal,factors=bounded_public(query,cap)
                            for eta in ETAS:
                                c=add('clipped',mode,d,eta,cap,None,step(b,eta,offset,signal.sum(0),g['decoder'])); c['clip_factors']=factors.tolist()
                                for risk in RISKS:
                                    sigma=public_scale(cap,risk); theta=step(b,eta,offset,signal.sum(0)[None]+noise_total(SELECTION_DRAWS,base,d,sigma),g['decoder'])
                                    add('noisy',mode,d,eta,cap,risk,theta)
                # Select by SELECTION CE only, then rescore assessment (fresh independent draws for noisy stages).
                selected={}
                for key,cands in stages.items():
                    c=best(cands); cfg=c['configuration']; d=cfg['dimension']; g=public_geometry(h,d,'euclidean'); mode=cfg['mode']
                    query=bank[cp+'_'+mode].reshape(8,51)@g['encoder']; offset=public_offset(mode,gamma.mean(0),g)
                    if cfg['stage']=='public_zero': total=np.zeros(d)
                    else: total=bounded_public(query,cfg['cap'] or UNCLIPPED)[0].sum(0)
                    if cfg['stage']=='noisy': total=total[None]+noise_total(ASSESSMENT_DRAWS,base+5000000,d,public_scale(cfg['cap'],cfg['risk']))
                    ace,aacc=mean_scores(assessment,step(b,cfg['eta'],offset,total,g['decoder']))
                    selected['|'.join(str(k) for k in key)]=dict(c,assessment_ce=ace,assessment_accuracy=aacc)
                def overall(stage,risk=None):
                    pool=[v for k,v in selected.items() if v['configuration']['stage']==stage and v['configuration']['risk']==risk]
                    return best(pool)
                zero=overall('public_zero'); res={'unclipped':overall('unclipped'),'clipped':overall('clipped')}
                for risk in RISKS: res[f'noisy_{int(risk*100)}']=overall('noisy',risk)
                row={'budget':budget,'subset_seed':seed,'cohort':cohort,'public_control':{'selection_ce':public['selection_ce'],'assessment_ce':public['assessment_ce']},'public_zero':zero,'stages':res,
                     'gain_over_public_control':{k:public['assessment_ce']-v['assessment_ce'] for k,v in res.items()},
                     'gain_over_public_zero':{k:zero['assessment_ce']-v['assessment_ce'] for k,v in res.items()},
                     'candidate_count':sum(len(v) for v in stages.values()),'all_selected':selected,
                     'candidates':[c for v in stages.values() for c in v]}
                rows.append(row)
                print('limited_public',budget,seed,cohort,{k:round(v,6) for k,v in row['gain_over_public_zero'].items()},flush=True)
    OUT.write_text(json.dumps({'kind':'Development-only protected-query decomposition; reused selection/assessment halves; no reserve, CIA or privacy claim.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),
        'headroom_sha':digest(HEAD),'bank_sha':digest(BANK),'source_hash':digest(source),'rows':rows},indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
