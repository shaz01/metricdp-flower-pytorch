"""Independently reconstruct the post-hoc DEV-ONLY signal diagnostic.

Only imports helpers from the independent artifact auditor. No diagnostic or
mechanism simulator imports; no reserve data reads, model selection on fresh
data, or protected-mechanism experiments. Writes a receipt only after passing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.calculations.audit_public_residual_artifacts import (
    C, ROOT, Checks, cached_data, fit_indices, metrics, sha, softmax, train,
)

ARTIFACT = ROOT / 'public_residual_signal_diagnostic.json'
CODE = Path('research/calculations/public_residual_signal_diagnostic.py')
PROTOCOL = Path('research/proposals/2026-10-08_residual_signal_diagnostic_protocol.md')
AUDITOR = Path('research/calculations/audit_public_residual_signal.py')
HELPERS = Path('research/calculations/audit_public_residual_artifacts.py')
DEV_JSON = ROOT / 'public_residual_development.json'
DEV_NPZ = ROOT / 'public_residual_development.npz'
PUBLIC_CONTROL = ROOT / 'public_residual_public_control_development.json'
ETAS = (0., .1, .3, 1., 3., 10., 30.)
MULTIPLIERS = (.5, .75, 1., 1.25, 1.5, 2.)


def gradients(x, y, reference):
    probabilities = softmax(x @ (C @ reference).T)
    residual = probabilities - np.eye(4)[y]
    return (residual @ C)[:, :, None] * x[:, None, :]


def main():
    required = (ARTIFACT, CODE, PROTOCOL, DEV_JSON, DEV_NPZ, PUBLIC_CONTROL)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit('No receipt written: missing ' + ', '.join(missing))
    result = json.loads(ARTIFACT.read_text())
    original = np.load(ROOT / 'private_descriptor_analytic_gaussian.npz')
    bank = np.load(DEV_NPZ)
    source, labels, features = cached_data(original)
    assert result['source_hash'] == sha(source)
    assert result['development_sha'] == sha(DEV_JSON)
    assert result['development_archive_sha'] == sha(DEV_NPZ)
    assert result['calculator_sha'] == sha(CODE)
    assert result['protocol_sha'] == sha(PROTOCOL)
    assert len(result['rows']) == 3
    assert {r['seed'] for r in result['rows']} == {42, 43, 44}
    check = Checks()
    reference, hessian = bank['reference'], bank['public_hessian']
    public_ids = original['public_original_indices']
    private_ids = original['private_original_indices']
    dev_ids = original['development_original_indices']
    public, public_y = features(public_ids), labels[public_ids]
    private, private_y = features(private_ids), labels[private_ids]
    dev_x, dev_y = features(dev_ids), labels[dev_ids]
    # Do not read reserve/evaluation indices or images in this diagnostic.
    assert len(public_ids) == 512 and len(private_ids) == 4096 and len(dev_ids) == 1024
    assert not set(public_ids) & set(private_ids)
    assert not set(dev_ids) & (set(public_ids) | set(private_ids))
    gamma = np.stack([gradients(public[public_y == k], public_y[public_y == k], reference).mean(axis=0)
                      for k in range(4)])
    check.equal(result['public_conditional_gradients'], gamma, 'class conditional public gradients')
    public_prior = np.bincount(public_y, minlength=4) / len(public_y)
    public_gradient = np.tensordot(public_prior, gamma, axes=1)
    strong_public = json.loads(PUBLIC_CONTROL.read_text())['selected']['development_ce']
    check.equal(result['strong_public_development_ce'], strong_public, 'frozen public offset CE')
    rows = []
    for row in result['rows']:
        seed = row['seed']
        fit = fit_indices(private_y, seed)
        pooled = np.concatenate(fit)
        assert len(pooled) == len(set(pooled)) == row['private_fit_count'] == 2048
        assert row['public_count'] == 512
        raw, centered, balanced, priors = [], [], [], []
        for indices in fit:
            yi = private_y[indices]
            example = gradients(private[indices], yi, reference)
            counts = np.bincount(yi, minlength=4)
            prior = counts / len(yi)
            g = example.mean(axis=0)
            raw.append(g)
            centered.append(g - np.tensordot(prior, gamma, axes=1))
            smoothed = (counts + 1) / (len(yi) + 4)
            weights = 1 / smoothed[yi]
            weights /= weights.mean()
            balanced.append((example * weights[:, None, None]).mean(axis=0))
            priors.append(prior)
        arrays = {'raw': np.asarray(raw), 'centered': np.asarray(centered), 'balanced': np.asarray(balanced)}
        pooled_prior = np.mean(priors, axis=0)
        check.equal(row['public_prior'], public_prior, 'public prior')
        check.equal(row['mean_client_prior'], pooled_prior, 'mean private prior')
        identity_difference = public_gradient + arrays['centered'].mean(axis=0) - arrays['raw'].mean(axis=0)
        expected_difference = np.tensordot(public_prior - pooled_prior, gamma, axes=1)
        check.equal(identity_difference, expected_difference, 'general prior identity', tolerance=1e-14)
        check.equal(row['class_prior_centering_identity_error'], np.max(np.abs(identity_difference)), 'saved identity error')
        for name, value in arrays.items():
            flat = value.reshape(8, 51)
            euclidean = np.sqrt(np.sum(flat * flat, axis=1))
            fisher = np.sqrt(np.sum((flat @ (hessian + .01 * np.eye(51))) * flat, axis=1))
            check.equal(row['client_norms'][name]['euclidean'], euclidean, name + ' vector norm')
            check.equal(row['client_norms'][name]['public_fisher'], fisher, name + ' public Fisher norm')
        raw_norm = np.linalg.norm(arrays['raw'].reshape(8, 51), axis=1)
        centered_norm = np.linalg.norm(arrays['centered'].reshape(8, 51), axis=1)
        check.equal(row['centered_to_raw_norm_ratios'], centered_norm / raw_norm, 'norm ratios')
        directions = {'public_only': public_gradient, 'raw_mean': arrays['raw'].mean(axis=0),
                      'balanced_mean': arrays['balanced'].mean(axis=0),
                      'centered_with_public': public_gradient + arrays['centered'].mean(axis=0),
                      'centered_only': arrays['centered'].mean(axis=0)}
        choices = row['noiseless_gradient_choices']
        assert len(choices) == 35
        assert {(r['direction'], r['eta']) for r in choices} == {(d, e) for d in directions for e in ETAS}
        for choice in choices:
            theta = reference - choice['eta'] * directions[choice['direction']]
            expected = metrics(dev_x, dev_y, theta)
            check.equal({k: choice[k] for k in ('ce', 'accuracy')}, expected, 'noiseless gradient scores')
        assert set(row['selected_direction_choices']) == set(directions)
        for name in directions:
            assert row['selected_direction_choices'][name] == min((c for c in choices if c['direction'] == name), key=lambda c: c['ce'])
        oracle = row['unprotected_pooled_choices']
        assert len(oracle) == 18
        assert {(c['steps'], c['logit_multiplier']) for c in oracle} == {(n, m) for n in (80, 320, 1280) for m in MULTIPLIERS}
        pooled_x = np.concatenate((public, private[pooled]))
        pooled_y = np.concatenate((public_y, private_y[pooled]))
        for steps in (80, 320, 1280):
            theta = train(pooled_x, pooled_y, steps)
            for choice in (c for c in oracle if c['steps'] == steps):
                expected = metrics(dev_x, dev_y, theta * choice['logit_multiplier'])
                check.equal({k: choice[k] for k in ('ce', 'accuracy')}, expected, 'unprotected pooled scores')
        assert row['unprotected_pooled_selected'] == min(oracle, key=lambda c: c['ce'])
        check.equal(row['development_ce_gain_over_strong_public'], strong_public - row['unprotected_pooled_selected']['ce'], 'pooled oracle offset gain')
        public_choice = row['selected_direction_choices']['public_only']
        best_public_ce = min(public_choice['ce'], strong_public)
        rows.append({'seed': seed, 'raw_mean_norm': float(raw_norm.mean()),
                     'centered_mean_norm': float(centered_norm.mean()),
                     'mean_relative_norm_reduction': float(1 - (centered_norm / raw_norm).mean()),
                     'ratio_of_mean_norms_reduction': float(1 - centered_norm.mean() / raw_norm.mean()),
                     'identity_error': float(np.max(np.abs(identity_difference))),
                     'saved_identity_error': row['class_prior_centering_identity_error'],
                     'best_public_gradient_eta': public_choice['eta'], 'best_public_gradient_ce': public_choice['ce'],
                     'strong_public_offset_ce': strong_public,
                     'unprotected_pooled_gain_vs_offset': row['development_ce_gain_over_strong_public'],
                     'unprotected_pooled_gain_vs_best_public_diagnostic': best_public_ce - row['unprotected_pooled_selected']['ce'],
                     'eta3_private_direction_gains_vs_public_gradient_eta3': {
                         name: next(v['ce'] for v in choices if v['direction'] == 'public_only' and v['eta'] == 3.)
                         - next(v['ce'] for v in choices if v['direction'] == name and v['eta'] == 3.)
                         for name in directions if name != 'public_only'},
                     'selected_private_direction_gains_vs_best_public_diagnostic': {
                         name: best_public_ce - value['ce'] for name, value in row['selected_direction_choices'].items()
                         if name != 'public_only'}})
    receipt = {'kind': 'Independent post-hoc raw-information DEVELOPMENT diagnostic audit; no protected/fresh confirmation claim.',
               'seeds': 3, 'client_gradient_vectors': 72, 'norm_values': 144, 'ratio_values': 24,
               'gradient_direction_step_records': 105, 'unprotected_pooled_records': 54,
               'selected_direction_records': 15, 'numeric_checks': check.count,
               'maximum_absolute_error': check.maximum, 'cached_source_sha256': sha(source),
               'no_reserve_indices_or_images_read': True,
               'hashes': {str(p): sha(p) for p in (*required, AUDITOR, HELPERS)}, 'rows': rows}
    target = ROOT / 'public_residual_signal_artifact_audit.json'
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
