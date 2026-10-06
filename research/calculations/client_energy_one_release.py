"""Schedule-changing one-shot local-training reference for energy-filter pilot."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from research.calculations.client_energy_filter_probe import C,clip,scores,ENERGIES,EPSILONS,summarize
from research.calculations.client_geometry_audit import features,pool,partition,split

def local_query(x,y,steps):
    theta=np.zeros((3,17))
    if not len(y):return theta
    for _ in range(steps):
        z=x@(C@theta).T;z-=z.max(axis=1,keepdims=True)
        p=np.exp(z);p/=p.sum(axis=1,keepdims=True);p[np.arange(len(y)),y]-=1
        theta-=.5*C.T@(p.T@x/len(y))
    return theta

def release(queries,weights,energy,epsilon,draws,seed,eval_set):
    cap=energy/64;rho=(np.sqrt(np.log(1e5)+epsilon)-np.sqrt(np.log(1e5)))**2
    signal=clip(queries*weights[:,None,None],np.full(8,np.sqrt(cap))).sum(axis=0)
    noise=np.random.default_rng(seed).normal(size=(draws,8,3,17))*np.sqrt(cap/(2*rho*7))
    return scores(*eval_set,signal[None,:,:]+noise.sum(axis=1))

def main():
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    train=Dataset.from_file(str(next(cache.rglob('fashion_mnist-train.arrow'))));test=Dataset.from_file(str(next(cache.rglob('fashion_mnist-test.arrow'))))
    labels=np.array(test['label']);ids=np.concatenate([np.flatnonzero(labels==k)[768:1000] for k in range(4)])
    held=(features(test,ids),labels[ids]);rows=[];saved={}
    for data_seed in (42,43,44):
        poolids,devids,yp,yd=pool(train,data_seed);xp=features(train,poolids);dev=(features(train,devids),yd)
        for mode in ('balanced','quantity','label_stress'):
            fit,_=split(partition(yp,mode,data_seed),yp,data_seed);weights=np.arange(1,9)/36 if mode=='quantity' else np.ones(8)/8
            queries={steps:np.stack([local_query(xp[a],yp[a],steps) for a in fit]) for steps in (5,20,80)}
            for epsilon in EPSILONS:
                seed=data_seed*10000+int(epsilon)*100;choices=[]
                for steps,q in queries.items():
                    for energy in ENERGIES:
                        ce,_=release(q,weights,energy,epsilon,16,seed,dev)
                        choices.append({'steps':steps,'energy':energy,'development_ce':float(ce.mean())})
                config=min(choices,key=lambda c:c['development_ce'])
                ce,acc=release(queries[config['steps']],weights,config['energy'],epsilon,128,seed+700000,held)
                key=f'{mode}_seed{data_seed}_epsilon{int(epsilon)}';saved[key+'_ce']=ce;saved[key+'_accuracy']=acc
                rows.append({'partition':mode,'seed':data_seed,'epsilon':epsilon,'configuration':config,'ce':summarize(ce),'accuracy':summarize(acc),'development_candidates':choices})
    target=Path('results/client_specific_noise');np.savez_compressed(target/'client_energy_one_release.npz',**saved)
    (target/'client_energy_one_release.json').write_text(json.dumps({'kind':'schedule-changing one-release reference; offline tuning not end-to-end DP','heldout_count':len(ids),'rows':rows},indent=2,allow_nan=False)+'\n')
    print('one-release reference complete:',len(rows),'cells')
def compute_control():
    root=Path('results/client_specific_noise')
    pilot=json.loads((root/'client_energy_filter_probe.json').read_text())
    previous=json.loads((root/'client_energy_one_release.json').read_text())
    lookup={(r['partition'],r['seed'],r['epsilon']):r for r in previous['rows']}
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    train=Dataset.from_file(str(next(cache.rglob('fashion_mnist-train.arrow'))));test=Dataset.from_file(str(next(cache.rglob('fashion_mnist-test.arrow'))))
    labels=np.array(test['label']);ids=np.concatenate([np.flatnonzero(labels==k)[768:1000] for k in range(4)])
    held=(features(test,ids),labels[ids]);saved={};rows=[]
    for data_seed in (42,43,44):
        poolids,_,yp,_=pool(train,data_seed);xp=features(train,poolids)
        for mode in ('balanced','quantity','label_stress'):
            fit,_=split(partition(yp,mode,data_seed),yp,data_seed);weights=np.arange(1,9)/36 if mode=='quantity' else np.ones(8)/8
            queries=np.stack([local_query(xp[a],yp[a],20) for a in fit])
            for r in pilot['rows']:
                if r['seed']!=data_seed or r['partition']!=mode:continue
                epsilon=r['epsilon'];seed=data_seed*10000+int(epsilon)*100+700000
                key=f'{mode}_seed{data_seed}_epsilon{int(epsilon)}'
                best=min([c for c in lookup[(mode,data_seed,epsilon)]['development_candidates'] if c['steps']==20],key=lambda c:c['development_ce'])
                configs={'twenty_tuned':best,'twenty_matched':{'steps':20,'energy':r['arms']['public']['configuration']['energy']}}
                arms={}
                for arm,config in configs.items():
                    ce,acc=release(queries,weights,config['energy'],epsilon,128,seed,held)
                    saved[key+'_'+arm+'_ce']=ce;saved[key+'_'+arm+'_accuracy']=acc
                    arms[arm]={'configuration':config,'ce':summarize(ce),'accuracy':summarize(acc)}
                rows.append({'partition':mode,'seed':data_seed,'epsilon':epsilon,'arms':arms})
    np.savez_compressed(root/'client_energy_one_release_compute_control.npz',**saved)
    (root/'client_energy_one_release_compute_control.json').write_text(json.dumps({'kind':'post-hoc audit-directed20-step computation control using already frozen development grid; reused heldout data, not independent confirmation','rows':rows},indent=2,allow_nan=False)+'\n')
    print('20-step computation control complete:',len(rows),'cells')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--compute-control',action='store_true');args=parser.parse_args()
    if args.compute_control:compute_control()
    else:main()
