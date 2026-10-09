"""Development-only spending/usefulness diagnostic; no retuning or peer release."""
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from research.calculations.client_energy_filter_probe import simulate
from research.calculations.client_geometry_audit import features,pool,partition,split

def main():
    root=Path('results/client_specific_noise');report=json.loads((root/'client_energy_filter_probe.json').read_text())
    cache=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    train=Dataset.from_file(str(next(cache.rglob('fashion_mnist-train.arrow'))));rows=[];saved={}
    for r in report['rows']:
        if r['partition']!='label_stress' or r['epsilon']!=8:continue
        seed=r['seed'];poolids,devids,yp,yd=pool(train,seed);xp=features(train,poolids);dev=(features(train,devids),yd)
        fit,_=split(partition(yp,'label_stress',seed),yp,seed);clients=[(xp[a],yp[a]) for a in fit]
        for arm in ('public','greedy_matched','remaining_matched','forecast_matched','ema_matched'):
            config=r['arms'][arm]['configuration']
            out,diag,_=simulate(clients,np.ones(8)/8,config['energy'],8,config['policy'],128,seed*10000+800+900000,{'development':dev},audit_set=dev)
            key=f'seed{seed}_{arm}'
            saved[key+'_development_ce']=out['development'][0]
            for name in ('mean_spent_trace','proposed_norms','signal_norms','development_alignment','development_linear_descent'):saved[key+'_'+name]=diag[name]
            norms=diag['proposed_norms'];alignment=diag['development_alignment'];signal=diag['signal_norms']
            ratios=np.divide(signal,norms,out=np.ones_like(norms),where=norms>0)
            correlations=[]
            for t in range(20):
                x=(norms[t]**2).ravel();y=diag['development_linear_descent'][t].ravel()
                correlations.append(float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else 0.)
            rows.append({'seed':seed,'arm':arm,'configuration':config,'development_ce':float(out['development'][0].mean()),
                         'mean_proposed_norm':float(norms.mean()),'mean_signal_norm':float(signal.mean()),'mean_clipping_ratio':float(ratios.mean()),
                         'fraction_clipped':float(np.mean(ratios<1-1e-9)),'mean_alignment':float(alignment.mean()),
                         'fraction_non_descent_proposals':float(np.mean(alignment<0)),
                         'mean_signal_first_order_descent':float(diag['development_linear_descent'].mean()),
                         'per_round_norm_squared_vs_linear_descent_correlation':correlations,
                         'mean_spent_fraction':float(diag['spent_fraction'].mean())})
    np.savez_compressed(root/'client_energy_filter_diagnostic.npz',**saved)
    (root/'client_energy_filter_diagnostic.json').write_text(json.dumps({'kind':'development-only raw diagnostic; pooled gradients never used by the mechanism; not a new heldout evaluation','draws':128,'rows':rows},indent=2,allow_nan=False)+'\n')
    print('development-only diagnostic complete:',len(rows),'arms')
if __name__=='__main__':main()
