"""Conditional fixed-federation CIA audit; no population/deployment privacy claim."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from research.calculations.private_descriptor_probe import C, ROOT, encode, sanitize, decode
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.client_geometry_audit import partition, split
from research.calculations.client_energy_filter_probe import scores

TARGETS=(0,1,2,3)
SIZES={'train':256,'validation':256,'evaluation':1024}


def auc_statistics(positive,negative):
    p=np.asarray(positive);n=np.asarray(negative)
    wins=(p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])
    auc=float(wins.mean())
    se=float(np.sqrt(wins.mean(axis=1).var(ddof=1)/len(p)+wins.mean(axis=0).var(ddof=1)/len(n)))
    fpr,tpr,_=roc_curve(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n],drop_intermediate=False)
    return {'auc':auc,'se':se,'ci95':[max(0.,auc-1.96*se),min(1.,auc+1.96*se)],
            'tpr_at_fpr01':float(tpr[fpr<=.01].max()),'tpr_at_fpr05':float(tpr[fpr<=.05].max())}


def peer_residual(objects,peer):
    return objects.sum(axis=1)-objects[:,peer]


def gaussian_llr(observed,out_mean,in_mean,out_std,in_std):
    d=observed.shape[-1]
    return d*np.log(out_std/in_std)+np.sum((observed-out_mean)**2,axis=-1)/(2*out_std**2)-np.sum((observed-in_mean)**2,axis=-1)/(2*in_std**2)


def stream(seed,target,stage,world):
    return seed*100000+target*10000+{'train':50000000,'validation':60000000,'evaluation':70000000}[stage]+world*1000


def select_attack(training,validation,losses):
    # All preprocessing fits on attack training data, never evaluation.
    x=np.concatenate(training);labels=np.r_[np.zeros(len(training[0])),np.ones(len(training[1]))]
    center=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-10]=1
    models={};validation_scores={}
    for ridge in (.01,.1,1.):
        model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=ridge).fit((x-center)/scale,labels)
        name=f'lda_{ridge}';models[name]=model
        validation_scores[name]=[model.decision_function((v-center)/scale) for v in validation]
    for sign in (-1,1):validation_scores[f'loss_{sign}']=[sign*v for v in losses]
    best=max(validation_scores,key=lambda n:roc_auc_score(labels,np.r_[validation_scores[n][0],validation_scores[n][1]]))
    return best,models,center,scale,validation_scores


def metric_parameters(teachers,target,radius,multiplier):
    physical=np.einsum('ca,iad->icd',C,teachers)
    distance=max((np.linalg.norm(physical[i,:,:16]-physical[j,:,:16])+np.linalg.norm(physical[i,:,-1]-physical[j,:,-1]))/2 for i in range(8) for j in range(i+1,8))
    if not np.isfinite(distance) or distance<=0:distance=1.
    q=teachers.reshape(8,51);f=np.minimum(1,radius/np.maximum(np.linalg.norm(q,axis=1),1e-100))
    mean=(q*f[:,None]).sum(axis=0)/8
    return mean,float(multiplier*radius/(8*distance)),float(distance)


def run():
    protocol=json.loads((ROOT/'private_prior_constructor_development.json').read_text())
    stored=np.load(ROOT/'private_prior_constructor_development.npz')
    source,ids,public,x,y,dev,anchors,decoders,denoisers,reserve,confirmation_ids=prepare(True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['source_hash']
    archive={'utility_original_indices':confirmation_ids};rows=[]
    for seed in (42,43,44):
        fit,_=split(partition(y,'label_stress',seed),y,seed)
        for target in TARGETS:
            for law in ('analytic_gaussian','radial_laplace'):
                development=next(r for r in protocol['rows'] if r['seed']==seed and r['law']==law)
                choices={
                    'aggregate_votes':development['selected'][development['chosen']['aggregate']['candidate']],
                    'aggregate_control':development['selected'][development['chosen']['aggregate']['control']],
                    'central_votes':development['selected'][development['chosen']['central']['candidate']],
                    'individual_votes':development['selected'][development['chosen']['individual']['candidate']]}
                for name,choice in choices.items():
                    row=experiment(seed,target,law,name,choice,stored,fit,x,y,anchors,decoders,denoisers,reserve,archive)
                    rows.append(row)
                    print(seed,target,law,name,'descriptor',round(row['descriptor']['auc'],3),'student',round(row['student']['auc'],3),flush=True)
            development=next(r for r in protocol['rows'] if r['seed']==seed and r['law']=='analytic_gaussian')
            models=[v for v in development['selected'].values() if v['contract']=='aggregate' and v['configuration']['family']=='model_direct']
            choice=min(models,key=lambda v:v['development_ce'])
            for multiplier in (.1,.3,1.,3.,10.):
                name=f'metric_{multiplier}'
                row=experiment(seed,target,'metric_adaptation',name,choice,stored,fit,x,y,anchors,decoders,denoisers,reserve,archive,multiplier)
                rows.append(row)
                print(seed,target,name,'descriptor',round(row['descriptor']['auc'],3),'student',round(row['student']['auc'],3),flush=True)
    np.savez_compressed(ROOT/'descriptor_cia_probe.npz',**archive)
    result={'kind':'conditional known-alternative whole-dataset CIA; same private bank, not population replication; no private research pipeline',
            'source_hash':protocol['source_hash'],'development_sha':hashlib.sha256((ROOT/'private_prior_constructor_development.json').read_bytes()).hexdigest(),
            'protocol_sha':hashlib.sha256(Path('research/proposals/2026-10-07_descriptor_cia_protocol.md').read_bytes()).hexdigest(),
            'epsilon':8,'sizes_per_world':SIZES,'peer':7,'targets':TARGETS,'utility_draws_per_world':128,'rows':rows}
    (ROOT/'descriptor_cia_probe.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def experiment(seed,target,law,name,choice,stored,fit,x,y,anchors,decoders,denoisers,reserve,archive,multiplier=None):
    c=choice['configuration'];contract=choice['contract'];transform=choice['transform']
    teachers=stored[f'seed{seed}_{transform}_steps{c["steps"]}_teachers']
    query=encode(teachers,c['family'],anchors.get(c['prototypes']),active_mask=[len(v)>0 for v in fit])
    weights=np.ones(8)/8;cap=c['radius']/8
    weighted=query*weights[:,None];factors=np.minimum(1,cap/np.maximum(np.linalg.norm(weighted,axis=1),1e-100));signal=weighted*factors[:,None]
    shift=signal[target];out=signal.copy();out[target]=0
    metric=None
    if multiplier is not None:
        empty=teachers.copy();empty[target]=0
        metric=[metric_parameters(empty,target,c['radius'],multiplier),metric_parameters(teachers,target,c['radius'],multiplier)]
    key=f'seed{seed}_target{target}_{law}_{name}'
    features={};losses={};descriptor={};utility={};expected_auc=None;calibrations={}
    for stage,draws in SIZES.items():
        features[stage]=[];losses[stage]=[];descriptor[stage]=[]
        for world in (0,1):
            if metric is not None:
                mean,std,distance=metric[world]
                latent=mean+np.random.default_rng(stream(seed,target,stage,world)).normal(size=(draws,51))*std
                theta=latent.reshape(-1,3,17)@denoisers[c['ridge']]
                observation=latent;own=np.empty((draws,0))
                ranking=gaussian_llr(observation,metric[0][0],metric[1][0],metric[0][1],metric[1][1])
            else:
                q=query.copy()
                if not world:q[target]=0
                objects,calibration=sanitize(q,weights,c['radius'],8,contract,draws,stream(seed,target,stage,world),law=law)
                calibrations[str(world)]=calibration
                theta=decode(objects,weights,c,contract,anchors,decoders,denoisers)
                if contract=='aggregate':observation=peer_residual(objects,7);m0=out[:7].sum(axis=0);own=objects[:,7]
                elif contract=='central':observation=objects.sum(axis=1);m0=out.sum(axis=0);own=np.empty((draws,0))
                else:observation=objects[:,target];m0=np.zeros_like(shift);own=objects[:,7]
                m1=m0+shift
                if law=='analytic_gaussian':
                    std=np.sqrt(calibration['unknown_floor_variance']);ranking=gaussian_llr(observation,m0,m1,std,std)
                    expected_auc=float(ndtr(np.linalg.norm(shift)/(np.sqrt(2)*std)))
                else:ranking=(np.linalg.norm(observation-m0,axis=1)-np.linalg.norm(observation-m1,axis=1))/(cap/8)
            features[stage].append(np.column_stack([theta.reshape(draws,-1),own]))
            target_loss,_=scores(x[fit[target]],y[fit[target]],theta);losses[stage].append(target_loss)
            descriptor[stage].append(ranking)
            if stage=='evaluation':
                ce,accuracy=scores(*reserve,theta[:128]);utility[str(world)]={'ce':float(ce.mean()),'accuracy':float(accuracy.mean())}
                archive[key+f'_world{world}_utility_ce']=ce;archive[key+f'_world{world}_utility_accuracy']=accuracy
    best,models,center,scale,validation_scores=select_attack(features['train'],features['validation'],losses['validation'])
    student=[]
    for world in (0,1):
        if best.startswith('loss_'):student.append(int(best.split('_')[1])*losses['evaluation'][world])
        else:student.append(models[best].decision_function((features['evaluation'][world]-center)/scale))
        archive[key+f'_world{world}_descriptor_score']=descriptor['evaluation'][world]
        archive[key+f'_world{world}_student_score']=student[world]
        for attack,values in validation_scores.items():archive[key+f'_world{world}_validation_{attack}']=values[world]
    result={'seed':seed,'target':target,'law':law,'arm':name,'configuration':c,'transform':transform,'contract':contract,
            'descriptor':auc_statistics(descriptor['evaluation'][1],descriptor['evaluation'][0]),
            'student':auc_statistics(student[1],student[0]),'student_attack':best,
            'validation_auc':{n:float(roc_auc_score(np.r_[np.zeros(256),np.ones(256)],np.r_[v[0],v[1]])) for n,v in validation_scores.items()},
            'utility':utility,'target_shift_norm':float(np.linalg.norm(shift)),'weighted_cap':cap,'expected_gaussian_auc':expected_auc}
    if metric is None:result['operator_calibration']=calibrations
    if metric is not None:
        assert c['ridge']==0 and np.array_equal(denoisers[c['ridge']],np.eye(17))
        result['student_exact_llr']=result['descriptor'].copy()
        result['student_decoder_identity_verified']=True
        result['metric_operator_calibration']={'multiplier':multiplier,'out_std':metric[0][1],'in_std':metric[1][1],'out_distance':metric[0][2],'in_distance':metric[1][2]}
    return result


if __name__=='__main__':run()
