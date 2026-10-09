"""Post-hoc development-only private information and public control-variate audit."""
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from research.calculations.public_residual_probe import digest, DEV_JSON, DEV_NPZ
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.private_descriptor_probe import ROOT, C, probabilities
from research.calculations.client_geometry_audit import partition, split
from research.calculations.client_energy_one_release import local_query
from research.calculations.client_energy_filter_probe import scores

PROTOCOL = Path('research/proposals/2026-10-08_residual_signal_diagnostic_protocol.md')


def example_gradients(x, y, b):
    residual = probabilities(x @ b.T) - np.eye(4)[y]
    return np.einsum('na,nd->nad', residual @ C, x)


def main():
    source, ids, public, private, y, dev, _, _, _, _, _ = prepare(False)
    bank = np.load(DEV_NPZ); b = bank['reference']; h = bank['public_hessian']
    labels = np.array(Dataset.from_file(str(source))['label'])
    public_y = labels[ids['public']]
    gamma = np.stack([example_gradients(public[public_y == k], public_y[public_y == k], b).mean(axis=0)
                      for k in range(4)])
    public_prior = np.bincount(public_y, minlength=4) / len(public_y)
    public_g = np.einsum('k,kad->ad', public_prior, gamma)
    strong_public = json.loads((ROOT / 'public_residual_public_control_development.json').read_text())['selected']['development_ce']
    rows = []
    for seed in (42, 43, 44):
        fit, _ = split(partition(y, 'label_stress', seed), y, seed)
        pooled = np.concatenate(fit)
        xp = np.concatenate((public, private[pooled])); yp = np.concatenate((public_y, y[pooled]))
        oracle = []
        for steps in (80, 320, 1280):
            theta = local_query(xp, yp, steps)
            for multiplier in (.5, .75, 1., 1.25, 1.5, 2.):
                ce, acc = scores(*dev, (theta * multiplier)[None])
                oracle.append({'steps': steps, 'logit_multiplier': multiplier,
                               'ce': float(ce[0]), 'accuracy': float(acc[0])})
        raw = []; centered = []; balanced = []; priors = []
        for indices in fit:
            yi = y[indices]; gradients = example_gradients(private[indices], yi, b)
            counts = np.bincount(yi, minlength=4); prior = counts / len(yi)
            raw.append(gradients.mean(axis=0)); priors.append(prior)
            centered.append(raw[-1] - np.einsum('k,kad->ad', prior, gamma))
            smoothed = (counts + 1) / (len(yi) + 4)
            weights = 1 / smoothed[yi]; weights /= weights.mean()
            balanced.append(np.einsum('n,nad->ad', weights, gradients) / len(yi))
        arrays = {'raw': np.array(raw), 'centered': np.array(centered), 'balanced': np.array(balanced)}
        pooled_prior = np.mean(priors, axis=0)
        identity_error = float(np.max(np.abs(public_g + arrays['centered'].mean(axis=0) - arrays['raw'].mean(axis=0))))
        expected_difference = np.einsum('k,kad->ad', public_prior - pooled_prior, gamma)
        assert np.allclose(public_g + arrays['centered'].mean(axis=0) - arrays['raw'].mean(axis=0), expected_difference, atol=1e-14)
        directions = {'public_only': public_g, 'raw_mean': arrays['raw'].mean(axis=0),
                      'balanced_mean': arrays['balanced'].mean(axis=0),
                      'centered_with_public': public_g + arrays['centered'].mean(axis=0),
                      'centered_only': arrays['centered'].mean(axis=0)}
        steps = []
        for name, gradient in directions.items():
            for eta in (0., .1, .3, 1., 3., 10., 30.):
                ce, acc = scores(*dev, (b - eta * gradient)[None])
                steps.append({'direction': name, 'eta': eta, 'ce': float(ce[0]), 'accuracy': float(acc[0])})
        norm_stats = {}
        for name, value in arrays.items():
            flat = value.reshape(8, 51)
            norm_stats[name] = {'euclidean': np.linalg.norm(flat, axis=1).tolist(),
                                'public_fisher': np.sqrt(np.einsum('ni,ij,nj->n', flat, h + .01 * np.eye(51), flat)).tolist()}
        raw_norm = np.linalg.norm(arrays['raw'].reshape(8, 51), axis=1)
        centered_norm = np.linalg.norm(arrays['centered'].reshape(8, 51), axis=1)
        best = min(oracle, key=lambda r: r['ce'])
        row = {'seed': seed, 'private_fit_count': len(pooled), 'public_count': len(public),
               'unprotected_pooled_choices': oracle, 'unprotected_pooled_selected': best,
               'development_ce_gain_over_strong_public': strong_public - best['ce'],
               'public_prior': public_prior.tolist(), 'mean_client_prior': pooled_prior.tolist(),
               'class_prior_centering_identity_error': identity_error,
               'client_norms': norm_stats, 'centered_to_raw_norm_ratios': (centered_norm / raw_norm).tolist(),
               'noiseless_gradient_choices': steps,
               'selected_direction_choices': {name: min((v for v in steps if v['direction'] == name), key=lambda v: v['ce']) for name in directions}}
        rows.append(row)
        print('development-only signal', seed, 'pooled CE gain', row['development_ce_gain_over_strong_public'],
              'centered/raw norms', min(row['centered_to_raw_norm_ratios']), max(row['centered_to_raw_norm_ratios']), flush=True)
    (ROOT / 'public_residual_signal_diagnostic.json').write_text(json.dumps({
        'kind': 'Post-hoc development-only raw-information diagnostic; no reserve/noise/DP claim.',
        'source_hash': digest(source), 'development_sha': digest(DEV_JSON), 'development_archive_sha': digest(DEV_NPZ),
        'protocol_sha': digest(PROTOCOL), 'calculator_sha': digest(__file__),
        'strong_public_development_ce': strong_public, 'public_conditional_gradients': gamma.tolist(),
        'rows': rows}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
