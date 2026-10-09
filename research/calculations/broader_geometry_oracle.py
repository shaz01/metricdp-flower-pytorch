"""Bounded raw-information oracle on all identifiable softmax-head directions.

Research diagnostic only: selection and source trajectories are not private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from research.calculations.client_geometry_audit import ASSIGN, features, statistics, softmax

RADII = (.001, .005, .02, .1, .3, 1., 3.)
EPSILONS = (4., 8., 16.)
DRAWS = 1024
GATE = .001


def basis(full):
    # Orthonormal, deterministic class contrasts; remove common-logit gauge only.
    b = np.zeros((4, 3))
    for j in range(3):
        b[:j+1, j] = 1 / np.sqrt((j+1)*(j+2))
        b[j+1, j] = -(j+1) / np.sqrt((j+1)*(j+2))
    if full:
        return np.kron(b, np.eye(17))
    p = np.zeros((68, 3))
    p[[16, 33, 50, 67]] = b
    return p


def ratios(d):
    r = np.full((4, d), .25)
    r[0] = 1
    for j in range(3):
        r[j+1, j*(d//3):(j+1)*(d//3)] = 1
    return r


def clipped_bank(u, c):
    radii = c * ratios(u.shape[1])
    gauges = np.sum(np.abs(u[:, None, :])/radii[None, :, :], axis=2)
    v = u[:, None, :] / np.maximum(1, gauges)[:, :, None]
    assert np.max(np.sum(np.abs(v)/radii[None, :, :], axis=2)) <= 1+1e-12
    return v, radii


def optimal_shrink(linear, quadratic):
    t = np.divide(-linear, 2*quadratic, out=np.zeros_like(linear), where=quadratic > 1e-20)
    t = np.where(quadratic <= 1e-20, np.where(linear < 0, 1., 0.), t)
    t = np.clip(t, 0, 1)
    return t, linear*t + quadratic*t*t


def select(u, weights, g, h):
    best = {e: {arm: None for arm in ('shared', 'oracle')} for e in EPSILONS}
    shared_indices = [np.flatnonzero(np.all(ASSIGN == k, axis=1))[0] for k in range(4)]
    for c in RADII:
        v, radii = clipped_bank(u, c)
        weighted = v*weights[:, None, None]
        local_linear = weighted @ g
        linear = np.sum(local_linear[np.arange(8)[None, :], ASSIGN], axis=1)
        pairs = np.einsum('ikd,de,jle->ikjl', weighted, h, weighted, optimize=True)
        q = np.zeros(len(ASSIGN))
        for i in range(8):
            for j in range(8):
                q += .5*pairs[i, ASSIGN[:, i], j, ASSIGN[:, j]]
        assert q.min() > -1e-10
        q = np.maximum(q, 0)
        local_penalty = 4 * (radii*radii @ np.diag(h))
        noise = np.sum(weights[None, :]**2 * local_penalty[ASSIGN], axis=1)
        for e in EPSILONS:
            t, score = optimal_shrink(linear, q+noise/e**2)
            for arm, ids in [('oracle', np.arange(len(ASSIGN))), ('shared', shared_indices)]:
                j = int(ids[np.argmin(score[ids])])
                if best[e][arm] is None or score[j] < best[e][arm]['predicted_change']:
                    best[e][arm] = {'predicted_change': float(score[j]), 'radius': c,
                                   'assignments': ASSIGN[j].tolist(), 'shrink': float(t[j])}
    for e in EPSILONS:
        assert best[e]['oracle']['predicted_change'] <= best[e]['shared']['predicted_change']+1e-12
        assert best[e]['shared']['predicted_change'] <= 1e-12  # includes no intervention
    return best


def loss_draws(x, y, w, p, deltas):
    values = []
    for start in range(0, len(deltas), 32):
        models = w[None, :, :] + (deltas[start:start+32] @ p.T).reshape(-1, 4, 17)
        logits = np.einsum('tcf,nf->tnc', models, x, optimize=True)
        logits -= logits.max(axis=2, keepdims=True)
        ce = np.log(np.exp(logits).sum(axis=2)) - logits[:, np.arange(len(y)), y]
        values.extend(ce.mean(axis=1))
    return np.array(values)


def compare(u, weights, g, h, x, y, w, p, seed):
    settings = select(u, weights, g, h)
    raw_mean = np.sum(u*weights[:, None], axis=0)
    _, _, full_h, *_ = statistics(x, y, w)
    eval_h = p.T @ full_h @ p
    base = float(loss_draws(x, y, w, p, np.zeros((1, p.shape[1])))[0])
    raw_loss = float(loss_draws(x, y, w, p, raw_mean[None])[0])
    unit = np.random.default_rng(seed).laplace(size=(DRAWS, 8, p.shape[1]))
    records, trials = [], {}
    for e in EPSILONS:
        record = {'epsilon_label': e, 'development_optimum': settings[e],
                  'no_intervention_loss': base, 'noiseless_raw_step_loss': raw_loss, 'evaluation': {}}
        vals = {}
        for arm, cfg in settings[e].items():
            ids = np.array(cfg['assignments']);t = cfg['shrink']
            v, radii = clipped_bank(u, cfg['radius'])
            clipped_mean = np.sum(v[np.arange(8), ids]*weights[:, None], axis=0)
            mean = t*clipped_mean
            noise = t*np.sum(unit*(2*radii[ids]/e)[None, :, :]*weights[None, :, None], axis=1)
            losses = loss_draws(x, y, w, p, mean[None]+noise)
            vals[arm] = losses;trials[f'e{int(e)}_{arm}'] = losses
            covariance_diag = t*t*8*np.sum(weights[:, None]**2*radii[ids]**2, axis=0)/e**2
            record['evaluation'][arm] = {
                'direct_noise_mean_loss': float(losses.mean()),
                'monte_carlo_standard_error': float(losses.std(ddof=1)/np.sqrt(DRAWS)),
                'noiseless_clipped_unshrunk_loss': float(loss_draws(x,y,w,p,clipped_mean[None])[0]),
                'noiseless_clipped_shrunk_loss': float(loss_draws(x,y,w,p,mean[None])[0]),
                'noise_quadratic_penalty': float(.5*np.diag(eval_h)@covariance_diag),
                'clipping_bias_norm_before_shrink': float(np.linalg.norm(clipped_mean-raw_mean)),
                'total_mean_deviation_from_raw_norm': float(np.linalg.norm(mean-raw_mean)),
                'aggregate_noise_trace': float(covariance_diag.sum())}
        diff = vals['oracle']-vals['shared'];se = float(diff.std(ddof=1)/np.sqrt(DRAWS))
        record['oracle_minus_shared'] = {'mean': float(diff.mean()), 'paired_standard_error': se,
            'interval_95_normal': [float(diff.mean()-1.96*se), float(diff.mean()+1.96*se)]}
        record['material_headroom_point_estimate'] = float(diff.mean()) <= -GATE
        record['noise_only_interval_beyond_gate'] = float(diff.mean()+1.96*se) <= -GATE
        records.append(record)
    return records, trials


def validate():
    rng = np.random.default_rng(91)
    for full in (False, True):
        p = basis(full);d = p.shape[1]
        assert np.allclose(p.T@p, np.eye(d))
        assert np.allclose(p.reshape(4,17,d).sum(axis=0),0)
        u = rng.normal(size=(8,d));clipped_bank(u,.1)
    p = basis(True);x = np.column_stack((rng.normal(size=(40,16)),np.ones(40)))
    y = rng.integers(0,4,len(x));w = rng.normal(scale=.1,size=(4,17))
    g,_,h,*_ = statistics(x,y,w);v = rng.normal(size=51);v /= np.linalg.norm(v);step=1e-5
    gp = statistics(x,y,w+(step*p@v).reshape(4,17))[0].ravel()
    gm = statistics(x,y,w-(step*p@v).reshape(4,17))[0].ravel()
    assert np.allclose(p.T@(gp-gm)/(2*step),(p.T@h@p)@v,atol=1e-9)
    d = 3;u = rng.normal(scale=.1,size=(8,d));weights=np.full(8,1/8)
    hh=np.eye(d);gg=rng.normal(size=d);cfg=select(u,weights,gg,hh)
    for e in EPSILONS:
        for arm in cfg[e].values():
            ids=np.array(arm['assignments']);v,r=clipped_bank(u,arm['radius']);t=arm['shrink']
            m=np.sum(v[np.arange(8),ids]*weights[:,None],axis=0)
            n=4*np.sum(weights[:,None]**2*r[ids]**2)/e**2
            score=t*(gg@m)+t*t*(.5*m@hh@m+n)
            assert np.isclose(score,arm['predicted_change'])
    assert np.allclose(loss_draws(x,y,w,p,np.zeros((2,51))),-np.log(softmax(x@w.T)[np.arange(len(y)),y]).mean())
    print('Contrast basis, projected derivatives, clipping, shrink score and direct loss checks passed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist')
    args=parser.parse_args();validate()
    if args.self_check:return
    root=Path('results/client_specific_noise')
    prior=json.loads((root/'client_geometry_audit.json').read_text())
    archive=np.load(root/'client_geometry_audit_matrices.npz')
    paths=[next(args.cache_dir.rglob(f'fashion_mnist-{s}.arrow'),None) for s in ('train','test')]
    if any(p is None for p in paths):raise FileNotFoundError('Offline Arrow cache required; use --cache-dir')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert hashes==prior['source_hashes'], 'Source dataset differs from saved trajectory'
    train,test=[Dataset.from_file(str(p)) for p in paths]
    labels=np.array(test['label']);eval_ids=np.concatenate([np.flatnonzero(labels==k)[256:512] for k in range(4)])
    assert not set(eval_ids)&set(archive['evaluation_original_indices'])
    xe=features(test,eval_ids);ye=labels[eval_ids]
    rows=[];saved={'evaluation_original_indices':eval_ids,'bias_basis':basis(False),'head_basis':basis(True)}
    for seed in (42,43,44):
        dev_ids=archive[f'seed{seed}_development_original_indices'];xd=features(train,dev_ids)
        yd=np.array(train['label'])[dev_ids]
        for mode in ('balanced','quantity','label_stress'):
            for round_ in (5,20):
                key=f'{mode}_seed{seed}_round{round_}';w=archive[key+'_model'];weights=archive[key+'_weights']
                gd,_,hd,*_=statistics(xd,yd,w)
                for full in (False,True):
                    space='full_head_51' if full else 'bias_contrasts_3';p=basis(full)
                    u=-.5*archive[key+'_mean_gradients'].reshape(8,68)@p
                    records,trials=compare(u,weights,p.T@gd.ravel(),p.T@hd@p,xe,ye,w,p,seed*1000+round_*10+int(full))
                    rows.append({'seed':seed,'partition':mode,'round':round_,'space':space,'comparisons':records})
                    for name,a in trials.items():saved[f'{key}_{space}_{name}']=a
                    print(seed,mode,round_,space,min(c['oracle_minus_shared']['mean'] for c in records),flush=True)
    report={'kind':'raw-information utility oracle, not a DP defense or CIA result',
            'source_hashes':hashes,'source_trajectory':'client_geometry_audit_matrices.npz',
            'spaces':['fixed Helmert class bias contrasts3','full identifiable fixed-feature softmax head51'],
            'profiles':'L1 diamond equal axes, or retain one class-contrast group and compress other groups to.25; no learned rotations',
            'radii':RADII,'epsilon_labels':EPSILONS,'noise_samples':DRAWS,
            'selection':'exact4^8 assignment enumeration per commonradius; development quadratic score; analytic shrink scales mean AND noise',
            'evaluation':'fresh official test indices256:512 perclass; disjoint previous audit evaluation0:256; same corpus, not new population',
            'gate_absolute_ce':GATE,'limitations':['unprotected raw checkpoints/updates, weights and selection; epsilon labels not accounted transcript privacy',
                'finite bank and common radii, not all noise laws or continuous best mechanism',
                'quadratic development selection can differ from nonlinear heldout ordering',
                'noise-only conditional exploratory intervals; no multiplicity or population uncertainty',
                'three seeds reuse training corpus and fixed test subset; not independent populations',
                '51-dimensional full head is reduced fixed-pixel classifier, not CNN/FL trajectory or attack evaluation'],
            'rows':rows}
    (root/'broader_geometry_oracle.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(root/'broader_geometry_oracle_trials.npz',**saved)


if __name__=='__main__':main()
