"""Independent saved-artifact audit. Does not import simulator or its statistics."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from datasets import Dataset

ROOT=Path('results/client_specific_noise')
count=0
maximum=0.

def equal(actual,expected):
    global count,maximum
    error=abs(float(actual)-float(expected));count+=1;maximum=max(maximum,error)
    assert error<5e-12,(actual,expected,error)

def stats(a,record):
    a=np.asarray(a);m=float(np.add.reduce(a)/a.size)
    se=math.sqrt(float(np.dot(a-m,a-m))/(a.size-1)/a.size)
    equal(record['mean'],m);equal(record['se'],se)
    return m,se

def contrast(a,b,record):
    m,se=stats(a-b,record['gain']);equal(record['ci95'][0],m-1.96*se);equal(record['ci95'][1],m+1.96*se)

def calibration(c,config,epsilon,law,contract):
    s=config['radius']/8;equal(c['weighted_cap'],s)
    if law=='radial_laplace':v=(c['dimension']+1)*(s/epsilon)**2
    elif law=='analytic_gaussian':v=s*s*{4.:1.168910944858024,8.:.3602749391128144,16.:.1184581022863902}[epsilon]
    else:
        rho=(math.sqrt(math.log(1e5)+epsilon)-math.sqrt(math.log(1e5)))**2;v=s*s/(2*rho)
    equal(c['unknown_floor_variance'],v)
    equal(c['aggregate_variance'],v*{'aggregate':8/7,'individual':8,'central':1}[contract])
    assert all(0<=v<=1 for v in c['clip_factors'])

def main():
    filenames={'zcdp':'private_descriptor_probe','analytic_gaussian':'private_descriptor_analytic_gaussian','radial_laplace':'private_descriptor_radial_laplace'}
    roles=None;sourcehash=None
    for law,name in filenames.items():
        data=json.loads((ROOT/(name+'.json')).read_text());archive=np.load(ROOT/(name+'.npz'))
        assert len(data['rows'])==27
        current={r:archive[r+'_original_indices'] for r in ('public','private','development','evaluation','reserve')}
        if roles is None:roles=current;sourcehash=data['source_hash']
        else:
            assert all(np.array_equal(roles[k],current[k]) for k in roles) and sourcehash==data['source_hash']
        for row in data['rows']:
            key=f"{row['partition']}_seed{row['seed']}_epsilon{int(row['epsilon'])}"
            candidates=row['development_candidates_epsilon8'];assert len(candidates)==972
            for arm,record in row['arms'].items():
                stats(archive[key+'_'+arm+'_ce'],record['ce']);stats(archive[key+'_'+arm+'_accuracy'],record['accuracy'])
                contract,family=arm.split('_',1);choices=[r for r in candidates if r['contract']==contract and r['configuration']['family']==family]
                selected=min(choices,key=lambda c:c['development_ce'])
                assert selected['configuration']==record['configuration'];equal(selected['development_ce'],record['development_ce_epsilon8'])
                calibration(record['calibration'],record['configuration'],row['epsilon'],law,contract)
                for label in ('same_contract','primary_peer'):
                    if label+'_contrast' in record:
                        c=record[label+'_contrast'];contrast(archive[key+'_'+c['reference']+'_ce'],archive[key+'_'+arm+'_ce'],c)
            if row['epsilon']==8:
                for contract in ('aggregate','individual','central'):
                    for family in ('model_direct','model_distribution','logits','probabilities','votes'):
                        choices=[r for r in candidates if r['contract']==contract and r['configuration']['family']==family]
                        for j,choice in enumerate(choices):equal(archive[f"{row['partition']}_seed{row['seed']}_{contract}_{family}_dev{j}"].mean(),choice['development_ce'])
    development=json.loads((ROOT/'private_prior_constructor_development.json').read_text());d=np.load(ROOT/'private_prior_constructor_development.npz')
    assert len(development['rows'])==6
    selection={}
    for row in development['rows']:
        selection[(row['seed'],row['law'])]=row
        for arm,choice in row['selected'].items():
            contract=choice['contract'];transform=choice['transform'];family=choice['configuration']['family']
            choices=[v for v in row['development_candidates'] if v['contract']==contract and v['transform']==transform and v['configuration']['family']==family]
            best=min(choices,key=lambda c:c['development_ce']);assert best==choice
            for j,value in enumerate(choices):equal(d[f"seed{row['seed']}_{row['law']}_{contract}_{transform}_{family}_dev{j}"].mean(),value['development_ce'])
        for contract,chosen in row['chosen'].items():
            available={name:v for name,v in row['selected'].items() if v['contract']==contract}
            votes=[n for n,v in available.items() if v['transform']!='raw' and v['configuration']['family']=='votes']
            controls=[n for n,v in available.items() if v['configuration']['family']!='votes' or v['transform']=='raw']
            assert chosen['candidate']==min(votes,key=lambda n:available[n]['development_ce'])
            assert chosen['control']==min(controls,key=lambda n:available[n]['development_ce'])
            equal(chosen['development_gain'],available[chosen['control']]['development_ce']-available[chosen['candidate']]['development_ce'])
    evaluation=json.loads((ROOT/'private_prior_constructor_evaluation.json').read_text());e=np.load(ROOT/'private_prior_constructor_evaluation.npz')
    assert len(evaluation['rows'])==18
    for row in evaluation['rows']:
        key=f"seed{row['seed']}_{row['law']}_epsilon{int(row['epsilon'])}"
        selected=selection[(row['seed'],row['law'])]['selected']
        assert len(row['arms'])==45 and len(row['contrasts'])==9
        chosen=selection[(row['seed'],row['law'])]['chosen']
        for contract,choice in chosen.items():
            for label,reference in [('same_contract',choice['control']),('primary_peer',chosen['aggregate']['control']),('central_same_votes',chosen['central']['candidate'])]:
                assert row['contrasts'][contract+'_'+label]['candidate']==choice['candidate']
                assert row['contrasts'][contract+'_'+label]['reference']==reference
        for arm,value in row['arms'].items():
            assert value['configuration']==selected[arm]['configuration'] and value['transform']==selected[arm]['transform']
            stats(e[key+'_'+arm+'_ce'],value['ce']);stats(e[key+'_'+arm+'_accuracy'],value['accuracy'])
            calibration(value['calibration'],value['configuration'],row['epsilon'],row['law'],selected[arm]['contract'])
        for c in row['contrasts'].values():contrast(e[key+'_'+c['reference']+'_ce'],e[key+'_'+c['candidate']+'_ce'],c)
    previous=np.load(ROOT/'client_geometry_audit_matrices.npz');old=set()
    for seed in (42,43,44):
        old.update(previous[f'seed{seed}_pool_original_indices']);old.update(previous[f'seed{seed}_development_original_indices'])
    seen=set()
    for ids in roles.values():
        assert len(set(ids))==len(ids) and not set(ids)&old and not set(ids)&seen;seen.update(ids)
    confirm=set(e['confirmation_original_indices']);remaining=set(e['remaining_reserve_original_indices'])
    assert len(confirm)==2048 and len(remaining)==1992 and not confirm&remaining and confirm|remaining==set(roles['reserve'])
    source=next((Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist').rglob('fashion_mnist-train.arrow'))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sourcehash==development['source_hash']==evaluation['source_hash']
    labels=np.array(Dataset.from_file(str(source))['label']);assert np.array_equal(np.bincount(labels[list(confirm)],minlength=4),np.full(4,512))
    # Reconstruct all six primary aggregate voting evaluations independently,
    # including features, bounded query, random law, simplex and student CE.
    dataset=Dataset.from_file(str(source));indices=e['confirmation_original_indices']
    images=np.stack([np.array(dataset[int(i)]['image'],dtype=float) for i in indices])/255
    pooled=images.reshape(-1,4,7,4,7).mean(axis=(2,4)).reshape(-1,16)-.5
    x=np.column_stack([pooled,np.ones(len(pooled))]);y=labels[indices]
    basis=np.zeros((4,3))
    for j in range(3):
        basis[:j+1,j]=1/math.sqrt((j+1)*(j+2));basis[j+1,j]=-(j+1)/math.sqrt((j+1)*(j+2))
    initial=np.load(ROOT/'private_descriptor_analytic_gaussian.npz');anchors=initial['anchors16']
    regression=np.linalg.pinv(anchors.T@anchors/16+.01*np.eye(17),rcond=1e-12)@anchors.T/16
    reconstructed=0
    for row in evaluation['rows']:
        if row['epsilon']!=8:continue
        seed=row['seed'];law=row['law'];key=f"seed{seed}_{law}_epsilon8_aggregate_balanced_loss_votes"
        configuration=row['arms']['aggregate_balanced_loss_votes']['configuration']
        assert configuration=={'family':'votes','steps':80,'radius':.8,'ridge':.01,'prototypes':16}
        teacher=d[f'seed{seed}_balanced_loss_steps80_teachers']
        predictions=np.array([(anchors@v.T@basis.T).argmax(axis=1) for v in teacher])
        query=(np.eye(4)[predictions]@basis).reshape(8,48)/4/8
        query[d[f'seed{seed}_private_class_counts'].sum(axis=1)==0]=0
        lengths=np.sqrt((query*query).sum(axis=1));query*=np.minimum(1,.1/np.maximum(lengths,1e-100))[:,None]
        rng=np.random.default_rng(seed*100000+16000000+8)
        z=rng.standard_normal((512,8,48))
        if law=='analytic_gaussian':z*=math.sqrt(.01*.3602749391128144/7)
        else:z*=np.sqrt(rng.gamma(24.5/7,2*(.1/8)**2,size=(512,8,1)))
        values=.25+((query[None]+z).sum(axis=1).reshape(512,16,3)*4)@basis.T
        ordered=np.sort(values,axis=-1)[...,::-1];cumulative=ordered.cumsum(axis=-1)-1
        active=(ordered-cumulative/np.arange(1,5)>0).sum(axis=-1)
        threshold=np.take_along_axis(cumulative,(active-1)[...,None],axis=-1)[...,0]/active
        targets=np.maximum(values-threshold[...,None],0)
        theta=np.array([regression@(np.log(np.maximum(v,.01))@basis) for v in targets])
        logits=np.einsum('md,nda,ca->nmc',x,theta,basis)
        lognormalizer=np.logaddexp.reduce(logits,axis=-1)
        ce=(lognormalizer-logits[:,np.arange(len(y)),y]).mean(axis=1)
        accuracy=(logits.argmax(axis=-1)==y).mean(axis=1)
        for actual,expected in zip(e[key+'_ce'],ce):equal(actual,expected)
        for actual,expected in zip(e[key+'_accuracy'],accuracy):equal(actual,expected)
        reconstructed+=1
    result={'numeric_comparisons':count,'maximum_absolute_error':maximum,'descriptor_cells':81,'descriptor_arms':1215,'prior_development_rows':6,'prior_confirmation_cells':18,'prior_confirmation_arms':810,'old_new_disjoint':True,'confirmation_count':2048,'remaining_reserve_count':1992,'source_sha_verified':True,'independently_reconstructed_primary_arms':reconstructed}
    (ROOT/'private_descriptor_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
