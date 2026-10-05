"""Offline real-data geometry audit; raw-information diagnostics are not private defenses."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from metricdp_pytorch.utils.split_data import balanced_stratified_partitions,quantity_skewed_partitions,partition_by_class_counts

ROUNDS=(0,5,20)
CLIP_RADII=(.001,.005,.02,.1,.3)
EPSILONS=(4.,8.,16.)
ANGLES=(0.,0.,np.pi/2,np.pi/4)
RATIOS=((1.,1.),(1.,.25),(1.,.25),(1.,.25))
ROT=np.array([[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]] for a in ANGLES])
ASSIGN=np.array(list(itertools.product(range(4),repeat=8)),dtype=int)


def softmax(x):
    e=np.exp(x-x.max(axis=1,keepdims=True));return e/e.sum(axis=1,keepdims=True)


def statistics(x,y,w):
    p=softmax(x@w.T);res=p-np.eye(4)[y]
    examples=(res[:,:,None]*x[:,None,:]).reshape(len(x),-1)
    moment=examples.T@examples/len(x)
    s=-p[:,:,None]*p[:,None,:]
    s[:,np.arange(4),np.arange(4)]+=p
    h=np.einsum('nab,ni,nj->aibj',s,x,x,optimize=True).reshape(68,68)/len(x)
    gradient=examples.mean(axis=0).reshape(4,17)
    eig,vec=np.linalg.eigh(moment);top=vec[:,-2:]
    covariance=moment-np.outer(gradient.ravel(),gradient.ravel())
    _,cv=np.linalg.eigh(covariance)
    mean_fraction=float(np.sum(gradient*gradient)/np.trace(moment))
    return gradient,moment,h,top,cv[:,-2:],covariance,mean_fraction


def evaluate(x,y,w):
    p=softmax(x@w.T)
    return {'cross_entropy':float(-np.log(p[np.arange(len(y)),y]).mean()),
            'accuracy':float(np.mean(p.argmax(axis=1)==y))}


def overlap(a,b):return float(np.sum((a.T@b)**2)/2)

def shape_cosine(a,b):
    return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def features(dataset,indices):
    images=np.stack([np.asarray(dataset[int(i)]['image'],dtype=float) for i in indices])/255
    pooled=images.reshape(-1,4,7,4,7).mean(axis=(2,4)).reshape(-1,16)-.5
    return np.column_stack((pooled,np.ones(len(pooled))))


def pool(dataset,seed):
    labels=np.array(dataset['label']);rng=np.random.default_rng(seed)
    train=[];dev=[]
    for k in range(4):
        indices=np.flatnonzero(labels==k);rng.shuffle(indices)
        train.extend(indices[:1024]);dev.extend(indices[1024:1280])
    return np.array(train),np.array(dev),labels[np.array(train)],labels[np.array(dev)]


def partition(labels,mode,seed):
    if mode=='balanced':return balanced_stratified_partitions(labels,8,seed=seed)
    if mode=='quantity':return quantity_skewed_partitions(len(labels),8,seed=seed,weights=(1,2,3,4,5,6,7,8))
    counts=np.full((8,4),34);counts[np.arange(8),np.arange(8)%4]=410
    return partition_by_class_counts(labels,counts,seed=seed)


def split(parts,labels,seed):
    rng=np.random.default_rng(seed);fit=[];check=[]
    for part in parts:
        a=[];b=[]
        for k in range(4):
            ids=np.array([i for i in part if labels[i]==k]);rng.shuffle(ids)
            a.extend(ids[:len(ids)//2]);b.extend(ids[len(ids)//2:])
        fit.append(np.array(a,dtype=int));check.append(np.array(b,dtype=int))
    return fit,check


def projected_gradient_hessian(x,y,w):
    # Public fixed subspace: biases of classes0 and1; all remaining parameters are frozen.
    p=softmax(x@w.T);g=(p-np.eye(4)[y]).mean(axis=0)[:2]
    h=np.diag(p[:,:2].mean(axis=0))-p[:,:2].T@p[:,:2]/len(x)
    return g,h


def bank_updates(u,c):
    coords=np.einsum('ni,kij->nkj',u,ROT)
    radii=c*np.array(RATIOS)
    gauge=np.sum(np.abs(coords)/radii,axis=-1)
    v=u[:,None,:]/np.maximum(1,gauge)[...,None]
    bodies=np.array([r@np.diag(a*a)@r.T for r,a in zip(ROT,radii)])
    return v,bodies


def oracle_test(u,weights,g,h,eval_g,eval_h,x_eval,y_eval,w,seed):
    best={e:{arm:(float('inf'),None) for arm in ('shared','oracle')} for e in EPSILONS}
    for c in CLIP_RADII:
        v,bodies=bank_updates(u,c)
        selected=v[np.arange(8)[None,:],ASSIGN]
        means=np.sum(selected*weights[None,:,None],axis=1)
        covcost=4*np.einsum('ij,kji->k',h,bodies)
        costs=np.sum(covcost[ASSIGN]*weights[None,:]**2,axis=1)
        changes=means@g+.5*np.einsum('ni,ij,nj->n',means,h,means)
        for e in EPSILONS:
            objective=changes+costs/e**2
            j=int(np.argmin(objective))
            if objective[j]<best[e]['oracle'][0]:best[e]['oracle']=(float(objective[j]),(c,ASSIGN[j].copy()))
            for k in range(4):
                ids=np.full(8,k);mean=np.sum(v[:,k,:]*weights[:,None],axis=0)
                value=float(mean@g+.5*mean@h@mean+np.sum(weights**2)*covcost[k]/e**2)
                if value<best[e]['shared'][0]:best[e]['shared']=(value,(c,ids))
    out=[];base=evaluate(x_eval,y_eval,w)['cross_entropy']
    raw_model=w.copy();raw_model[:2,-1]+=np.sum(u*weights[:,None],axis=0)
    raw_loss=evaluate(x_eval,y_eval,raw_model)['cross_entropy']
    rng=np.random.default_rng(seed);unit=rng.laplace(size=(2048,8,2))
    for e in EPSILONS:
        row={'epsilon_label':e,'development_optimum':{},'evaluation':{}}
        trial={}
        for arm in ('shared','oracle'):
            predicted,(c,ids)=best[e][arm]
            v,bodies=bank_updates(u,c);mean=np.sum(v[np.arange(8),ids]*weights[:,None],axis=0)
            aggregate_cov=8*np.sum(bodies[ids]*weights[:,None,None]**2,axis=0)/e**2
            change=float(eval_g@mean+.5*mean@eval_h@mean+.5*np.trace(eval_h@aggregate_cov))
            radii=c*np.array(RATIOS)[ids]
            localnoise=unit*(2*radii/e)
            noise=np.einsum('nij,tnj->tni',ROT[ids],localnoise)
            deltas=mean+np.sum(noise*weights[None,:,None],axis=1)
            logits=x_eval@w.T
            values=[]
            for offset in range(0,len(deltas),64):
                z=np.broadcast_to(logits,(len(deltas[offset:offset+64]),*logits.shape)).copy()
                z[:,:,:2]+=deltas[offset:offset+64,None,:]
                z-=z.max(axis=2,keepdims=True)
                logsum=np.log(np.exp(z).sum(axis=2))
                ce=np.mean(logsum-z[:,np.arange(len(y_eval)),y_eval],axis=1)
                values.extend(ce)
            values=np.array(values);trial[arm]=values
            row['development_optimum'][arm]={'predicted_change':predicted,'radius':c,'assignments':ids.tolist()}
            clipped_model=w.copy();clipped_model[:2,-1]+=mean
            clipped_loss=evaluate(x_eval,y_eval,clipped_model)['cross_entropy']
            row['evaluation'][arm]={'noiseless_clipped_loss':clipped_loss,'noise_quadratic_penalty':float(.5*np.trace(eval_h@aggregate_cov)),'second_order_change':change,'direct_noise_mean_loss':float(values.mean()),
                                     'mean_change_from_no_intervention':float(values.mean()-base),
                                     'monte_carlo_standard_error':float(values.std(ddof=1)/np.sqrt(len(values)))}
        difference=trial['oracle']-trial['shared'];se=float(difference.std(ddof=1)/np.sqrt(len(difference)))
        row['oracle_minus_shared']={'mean':float(difference.mean()),'paired_standard_error':se,
                                    'interval_95_normal':[float(difference.mean()-1.96*se),float(difference.mean()+1.96*se)]}
        row['no_intervention_loss']=base
        row['noiseless_raw_projected_step_loss']=raw_loss
        row['material_headroom_0_001']=float(difference.mean())<=-.001
        assert best[e]['oracle'][0]<=best[e]['shared'][0]+1e-12
        out.append(row)
    return out


def validate():
    rng=np.random.default_rng(5);x=np.column_stack((rng.normal(size=(32,16)),np.ones(32)))
    y=rng.integers(0,4,len(x));w=rng.normal(scale=.1,size=(4,17))
    g,m,h,*_=statistics(x,y,w);v=rng.normal(size=68);v/=np.linalg.norm(v);step=1e-5
    gp=statistics(x,y,w+step*v.reshape(4,17))[0].ravel()
    gm=statistics(x,y,w-step*v.reshape(4,17))[0].ravel()
    assert np.allclose((gp-gm)/(2*step),h@v,atol=1e-9)
    gp2,hp2=projected_gradient_hessian(x,y,w)
    idx=[16,33];assert np.allclose(gp2,g.ravel()[idx]);assert np.allclose(hp2,h[np.ix_(idx,idx)])
    assert np.linalg.eigvalsh(h).min()>-1e-12
    u=rng.normal(size=(8,2));v,b=bank_updates(u,.1)
    coords=np.einsum('nki,kij->nkj',v,ROT)
    assert np.max(np.sum(np.abs(coords)/(.1*np.array(RATIOS)),axis=-1))<=1+1e-12


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist')
    args=parser.parse_args()
    validate()
    if args.self_check:
        print('Softmax derivatives, projected Hessian, PSD and body checks passed');return
    cache=args.cache_dir
    train_path=next(cache.rglob('fashion_mnist-train.arrow'),None);test_path=next(cache.rglob('fashion_mnist-test.arrow'),None)
    if train_path is None or test_path is None:
        raise FileNotFoundError('Cached Fashion-MNIST Arrow files required. Supply --cache-dir; this audit never downloads data.')
    train=Dataset.from_file(str(train_path));test=Dataset.from_file(str(test_path))
    yl=np.array(test['label']);test_ids=np.concatenate([np.flatnonzero(yl==k)[:256] for k in range(4)])
    xe=features(test,test_ids);ye=yl[test_ids]
    rows=[];saved={}
    for seed in (42,43,44):
        poolids,devids,yp,yd=pool(train,seed);xp=features(train,poolids);xd=features(train,devids)
        assert not set(poolids)&set(devids)
        for mode in ('balanced','quantity','label_stress'):
            parts=partition(yp,mode,seed);fit,check=split(parts,yp,seed)
            assert len(set(np.concatenate(fit)))==sum(map(len,fit))
            assert not set(np.concatenate(fit))&set(np.concatenate(check))
            weights=np.array([len(a) for a in fit],dtype=float);weights/=weights.sum()
            w=np.zeros((4,17));previous=None;previous_h=None;previous_cov=None
            for round_ in range(21):
                gradients=[statistics(xp[a],yp[a],w)[0] for a in fit] if round_ in ROUNDS else None
                if round_ in ROUNDS:
                    fit_stats=[statistics(xp[a],yp[a],w) for a in fit]
                    hold_stats=[statistics(xp[a],yp[a],w) for a in check]
                    same=np.mean([overlap(a[3],b[3]) for a,b in zip(fit_stats,hold_stats)])
                    between=np.mean([overlap(fit_stats[i][3],fit_stats[j][3]) for i in range(8) for j in range(i)])
                    persistence=None if previous is None else float(np.mean([overlap(a[3],b) for a,b in zip(fit_stats,previous)]))
                    previous=[a[3] for a in fit_stats]
                    h_within=float(np.mean([shape_cosine(a[2],b[2]) for a,b in zip(fit_stats,hold_stats)]))
                    h_between=float(np.mean([shape_cosine(fit_stats[i][2],fit_stats[j][2]) for i in range(8) for j in range(i)]))
                    h_persistence=None if previous_h is None else float(np.mean([shape_cosine(a[2],b) for a,b in zip(fit_stats,previous_h)]))
                    cov_within=float(np.mean([overlap(a[4],b[4]) for a,b in zip(fit_stats,hold_stats)]))
                    cov_between=float(np.mean([overlap(fit_stats[i][4],fit_stats[j][4]) for i in range(8) for j in range(i)]))
                    cov_persistence=None if previous_cov is None else float(np.mean([overlap(a[4],b) for a,b in zip(fit_stats,previous_cov)]))
                    previous_h=[a[2] for a in fit_stats];previous_cov=[a[4] for a in fit_stats]
                    pooled_m=sum(alpha*a[1] for alpha,a in zip(weights,fit_stats))
                    pooled_c=sum(alpha*a[5] for alpha,a in zip(weights,fit_stats))
                    pooled_h=sum(alpha*a[2] for alpha,a in zip(weights,fit_stats))
                    _,pm=np.linalg.eigh(pooled_m);_,pc=np.linalg.eigh(pooled_c)
                    pooled_m_overlap=float(np.mean([overlap(pm[:,-2:],a[3]) for a in hold_stats]))
                    pooled_c_overlap=float(np.mean([overlap(pc[:,-2:],a[4]) for a in hold_stats]))
                    pooled_h_cosine=float(np.mean([shape_cosine(pooled_h,a[2]) for a in hold_stats]))
                    gaps=[]
                    for a in fit_stats:
                        eigenvalues=np.linalg.eigvalsh(a[1])
                        gaps.append(float((eigenvalues[-2]-eigenvalues[-3])/max(eigenvalues[-2],1e-12)))
                    u=-.5*np.array([a[0][:2,-1] for a in fit_stats])
                    g,h=projected_gradient_hessian(xd,yd,w);ge,he=projected_gradient_hessian(xe,ye,w)
                    oracles=oracle_test(u,weights,g,h,ge,he,xe,ye,w,seed*100+round_)
                    key=f'{mode}_seed{seed}_round{round_}'
                    saved[key+'_model']=w.copy();saved[key+'_gradient_moments']=np.stack([a[1] for a in fit_stats])
                    saved[key+'_local_hessians']=np.stack([a[2] for a in fit_stats])
                    saved[key+'_gradient_covariances']=np.stack([a[5] for a in fit_stats])
                    saved[key+'_mean_gradients']=np.stack([a[0] for a in fit_stats])
                    saved[key+'_projected_updates']=u;saved[key+'_weights']=weights
                    rows.append({'seed':seed,'partition':mode,'round':round_,'fit_counts':[len(a) for a in fit],
                                  'client_class_counts':[np.bincount(yp[a],minlength=4).tolist() for a in fit],
                                  'global_evaluation':evaluate(xe,ye,w),
                                  'gradient_second_moment_top2':{'within_client_disjoint_split_overlap':float(same),
                                                               'between_client_overlap':float(between),
                                                               'same_client_previous_snapshot_overlap':persistence},
                                  'centered_gradient_covariance_top2':{'within_client_disjoint_split_overlap':cov_within,'between_client_overlap':cov_between,'same_client_previous_snapshot_overlap':cov_persistence},
                                  'local_hessian_shape_cosine':{'within_client_disjoint_split':h_within,'between_clients':h_between,'same_client_previous_snapshot':h_persistence},
                                  'mean_gradient_energy_fraction':[a[6] for a in fit_stats],
                                  'pooled_geometry_prediction':{'moment_vs_client_holdout_overlap':pooled_m_overlap,'mean_within_client_covariance_vs_client_holdout_overlap':pooled_c_overlap,'hessian_vs_client_holdout_cosine':pooled_h_cosine},
                                  'moment_top2_eigengap_ratios':gaps,
                                  'oracle_interventions':oracles})
                    print(seed,mode,round_,round(same,3),round(between,3),round(evaluate(xe,ye,w)['accuracy'],3),flush=True)
                if round_<20:
                    if gradients is None:
                        gradients=[]
                        for a in fit:
                            p=softmax(xp[a]@w.T);gradients.append((p-np.eye(4)[yp[a]]).T@xp[a]/len(a))
                    w-=.5*np.sum(np.stack(gradients)*weights[:,None,None],axis=0)
            saved[f'{mode}_seed{seed}_fit_indices']=np.concatenate(fit)
            saved[f'{mode}_seed{seed}_check_indices']=np.concatenate(check)
        saved[f'seed{seed}_pool_original_indices']=poolids;saved[f'seed{seed}_development_original_indices']=devids
    saved['evaluation_original_indices']=test_ids
    out=Path('results/client_specific_noise')
    np.savez_compressed(out/'client_geometry_audit_matrices.npz',**saved)
    report={'kind':'real-data raw-information geometry diagnostic, not a DP defense/CIA result',
            'dataset':'Fashion-MNIST four repository-supported classes0..3; local Arrow offline',
            'source_hashes':{train_path.name:hashlib.sha256(train_path.read_bytes()).hexdigest(),
                             test_path.name:hashlib.sha256(test_path.read_bytes()).hexdigest()},
            'model':'public fixed4x4 average pixel features+bias, four-class softmax68parameters, no CNN transfer',
            'training':'20 sample-weighted fullbatch FedAvg gradient steps, lr.5; private head trajectory not DP',
            'regimes':'existing balanced/quantity partition helpers; separate80%labelstress',
            'utility_probe':'two fixed public class-bias coordinates only; complement frozen, raw snapshots are not protected',
            'oracle':'exact4^8 assignment enumeration per radius; same globaldevelopment gradient/Hessian for shared and personalized diagnostics',
            'radii':CLIP_RADII,'epsilon_labels':EPSILONS,'noise_samples':2048,
            'material_headroom_gate':'heldout oracle-minus-shared CE <=-.001 within this intervention family; not preregistered general criterion',
            'limitations':['three partitions reuse a corpus; seeds are not independent human populations',
                           'raw-oracle choice has no accounted privacy guarantee; epsilon labels are fixed-profile counterfactual calibrations',
                           'top2moment overlap is not Hessian/sensitivity/attack information',
                           'finite two-coordinate bank cannot upperbound all full-model mechanisms',
                           'noop and noiseless learning remain scientifically relevant controls',
                           'normal MonteCarlo intervals exploratory; no selection multiplicity adjustment'],
            'checks':'finite-difference softmax Hessian, projected Hessian identity, PSD, diamond containment, split disjointness, oracle includes shared choices',
            'rows':rows}
    (out/'client_geometry_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
