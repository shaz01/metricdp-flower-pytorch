"""Frozen slot-profile control; conditional utility, not a privacy experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from research.calculations.client_geometry_audit import features, statistics
from research.calculations.broader_geometry_oracle import (
    RADII, EPSILONS, basis, clipped_bank, optimal_shrink, select, loss_draws,
)

FROZEN_IDS = np.array([0, 2, 0, 1, 0, 2, 0, 1])
FROZEN_RADIUS = .3
DRAWS = 2048
SHIFT = np.array([1, 2, 3, 4, 5, 6, 7, 0])


def fixed_setting(u, weights, g, h, e, c, tune):
    v, radii = clipped_bank(u, c)
    mean = np.sum(v[np.arange(8), FROZEN_IDS]*weights[:, None], axis=0)
    linear = g@mean
    q = .5*mean@h@mean + 4*np.sum(weights[:, None]**2*radii[FROZEN_IDS]**2*np.diag(h))/e**2
    assert q >= -1e-12
    if tune:
        t, score = optimal_shrink(np.array([linear]), np.array([max(0, q)]))
        t, score = float(t[0]), float(score[0])
    else:
        t, score = 1., float(linear+q)
    return {'assignments': FROZEN_IDS.tolist(), 'radius': c, 'shrink': t, 'predicted_change': score}


def settings(u, weights, g, h):
    tuned = select(u, weights, g, h)
    result = {}
    for e in EPSILONS:
        fixed = fixed_setting(u, weights, g, h, e, FROZEN_RADIUS, False)
        candidates = [fixed_setting(u, weights, g, h, e, c, True) for c in RADII]
        assignment_tuned = min(candidates, key=lambda x: x['predicted_change'])
        result[e] = {'shared_tuned': tuned[e]['shared'], 'oracle_tuned': tuned[e]['oracle'],
                     'frozen_config': fixed, 'frozen_assignment_tuned': assignment_tuned}
        assert assignment_tuned['predicted_change'] <= fixed['predicted_change']+1e-12
        assert tuned[e]['oracle']['predicted_change'] <= assignment_tuned['predicted_change']+1e-12
    return result


def contrast(a, b):
    difference = a-b
    mean = float(difference.mean());se = float(difference.std(ddof=1)/np.sqrt(len(a)))
    return {'mean': mean, 'paired_standard_error': se,
            'interval_95_normal': [mean-1.96*se, mean+1.96*se]}


def compare(u, weights, g, h, x, y, w, seed):
    configs = settings(u, weights, g, h)
    p = basis(False)
    _, _, full_h, *_ = statistics(x, y, w)
    eval_h = p.T@full_h@p
    base = float(loss_draws(x, y, w, p, np.zeros((1, 3)))[0])
    unit = np.random.default_rng(seed).laplace(size=(DRAWS, 8, 3))
    records, trials = [], {}
    for e in EPSILONS:
        values = {};ev = {}
        for arm, cfg in configs[e].items():
            v, radii = clipped_bank(u, cfg['radius']);ids = np.array(cfg['assignments']);t = cfg['shrink']
            mean = t*np.sum(v[np.arange(8), ids]*weights[:, None], axis=0)
            noise = t*np.sum(unit*(2*radii[ids]/e)[None]*weights[None, :, None], axis=1)
            values[arm] = loss_draws(x, y, w, p, mean[None]+noise)
            trials[f'e{int(e)}_{arm}'] = values[arm]
            diag_cov = t*t*8*np.sum(weights[:, None]**2*radii[ids]**2, axis=0)/e**2
            ev[arm] = {'direct_noise_mean_loss': float(values[arm].mean()),
                'monte_carlo_standard_error': float(values[arm].std(ddof=1)/np.sqrt(DRAWS)),
                'noiseless_clipped_shrunk_loss': float(loss_draws(x, y, w, p, mean[None])[0]),
                'noise_quadratic_penalty': float(.5*np.diag(eval_h)@diag_cov)}
        contrasts = {f'{arm}_minus_shared': contrast(values[arm], values['shared_tuned'])
                     for arm in ('oracle_tuned', 'frozen_config', 'frozen_assignment_tuned')}
        for arm in ('frozen_config', 'frozen_assignment_tuned'):
            contrasts[f'{arm}_minus_oracle'] = contrast(values[arm], values['oracle_tuned'])
        records.append({'epsilon_label': e, 'settings': configs[e], 'no_intervention_loss': base,
            'evaluation': ev, 'contrasts': contrasts,
            'frozen_gain_0_001': contrasts['frozen_config_minus_shared']['mean'] <= -.001,
            'frozen_gap_to_oracle_0_0001': abs(contrasts['frozen_config_minus_oracle']['mean']) <= .0001})
    return records, trials


def validate():
    rng = np.random.default_rng(120);u = rng.normal(scale=.1, size=(8,3))
    weights = np.ones(8)/8;g = rng.normal(size=3);h = np.eye(3)
    a = settings(u, weights, g, h);b = settings(u[SHIFT], weights[SHIFT], g, h)
    for e in EPSILONS:
        for arm in ('shared_tuned','oracle_tuned'):
            assert np.isclose(a[e][arm]['predicted_change'], b[e][arm]['predicted_change'])
        assert a[e]['frozen_config']['assignments'] == FROZEN_IDS.tolist()
        assert a[e]['frozen_config']['radius'] == .3 and a[e]['frozen_config']['shrink'] == 1
    identical = np.ones(DRAWS);assert contrast(identical, identical)['paired_standard_error'] == 0
    print('Frozen configuration, oracle inclusion, roster permutation invariance and paired contrast checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--cache-dir', type=Path, default=Path.home()/'.cache/huggingface/datasets/zalando-datasets___fashion_mnist')
    args = parser.parse_args();validate()
    if args.self_check:return
    root = Path('results/client_specific_noise')
    prior = json.loads((root/'client_geometry_audit.json').read_text())
    archive = np.load(root/'client_geometry_audit_matrices.npz')
    broad = np.load(root/'broader_geometry_oracle_trials.npz')
    paths = [next(args.cache_dir.rglob(f'fashion_mnist-{s}.arrow'), None) for s in ('train','test')]
    if any(p is None for p in paths):raise FileNotFoundError('Offline Fashion-MNIST Arrow cache required')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert hashes == prior['source_hashes']
    train, test = [Dataset.from_file(str(p)) for p in paths]
    labels = np.array(test['label'])
    eval_ids = np.concatenate([np.flatnonzero(labels == k)[512:768] for k in range(4)])
    assert len(eval_ids) == 1024
    for old in (archive, broad):assert not set(eval_ids)&set(old['evaluation_original_indices'])
    x = features(test, eval_ids);y = labels[eval_ids];p = basis(False)
    rows = [];saved = {'evaluation_original_indices': eval_ids, 'frozen_ids': FROZEN_IDS, 'shift_order': SHIFT}
    for seed in (42,43,44):
        dev_ids = archive[f'seed{seed}_development_original_indices']
        xd = features(train, dev_ids);yd = np.array(train['label'])[dev_ids]
        for mode in ('balanced','quantity','label_stress'):
            for round_ in (5,20):
                key = f'{mode}_seed{seed}_round{round_}'
                w = archive[key+'_model'];weights = archive[key+'_weights']
                u = -.5*archive[key+'_mean_gradients'].reshape(8,68)@p
                gd, _, hd, *_ = statistics(xd, yd, w);g = p.T@gd.ravel();h = p.T@hd@p
                baseline = settings(u, weights, g, h)
                for roster in (('original','shifted') if mode == 'label_stress' else ('original',)):
                    order = np.arange(8) if roster == 'original' else SHIFT
                    cfg = settings(u[order], weights[order], g, h)
                    for e in EPSILONS:
                        for arm in ('shared_tuned','oracle_tuned'):
                            assert np.isclose(cfg[e][arm]['predicted_change'], baseline[e][arm]['predicted_change'])
                    records, trials = compare(u[order], weights[order], g, h, x, y, w, seed*1000+round_*10+int(roster=='shifted'))
                    rows.append({'seed': seed, 'partition': mode, 'round': round_, 'roster': roster,
                                 'client_order': order.tolist(), 'comparisons': records})
                    for name, a in trials.items():saved[f'{key}_{roster}_{name}'] = a
                    print(seed, mode, round_, roster, min(c['contrasts']['frozen_config_minus_shared']['mean'] for c in records), flush=True)
    report = {'kind': 'frozen-slot-profile raw-checkpoint utility diagnostic, not DP or CIA evidence',
        'source_hashes': hashes, 'source_trajectory': 'client_geometry_audit_matrices.npz',
        'frozen_assignments': FROZEN_IDS.tolist(), 'frozen_radius': .3, 'frozen_shrink': 1,
        'frozen_source': 'broader audit labelstress round20 epsilon-label8; frozen before this new test slice',
        'space': 'fixed3 class-bias contrasts; all feature coefficients frozen',
        'radii': RADII, 'epsilon_labels': EPSILONS, 'noise_samples': DRAWS,
        'evaluation': 'official test perclass positions512:768; disjoint two earlier test slices; same corpus/heads',
        'primary_cells': 'original roster,labelstress,round20,epsilon-label8,three seeds',
        'primary_gates': {'frozen_gain_over_shared': .001, 'absolute_frozen_minus_oracle_ce': .0001},
        'arms': {'frozen_config': 'profiles/radius/shrink frozen; no fresh tuning',
            'frozen_assignment_tuned': 'fixed slot profiles, development-optimized radius/shrink; unaccounted private tuning',
            'shared_tuned': 'development-optimized common profile/radius/shrink',
            'oracle_tuned': 'exact4^8 assignment oracle with development radius/shrink optimization'},
        'limitations': ['raw checkpoints/updates/development selectors not protected; epsilon labels not transcript privacy budgets',
            'frozen profiles learned from priorprivate raw diagnostics, not certified public calibration data',
            'cyclic roster shift preserves data but deliberately breaks slot-label alignment, not new population evidence',
            'paired intervals conditional on noise only, exploratory and uncorrected for multiple comparisons',
            'fixed-feature bias intervention only, no full FL/CNN/CIA run or distribution novelty'],
        'rows': rows}
    (root/'public_slot_profile_probe.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(root/'public_slot_profile_trials.npz', **saved)


if __name__ == '__main__':main()
