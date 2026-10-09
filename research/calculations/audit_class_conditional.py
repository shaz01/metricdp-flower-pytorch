"""Independent class-conditional saved-artifact audit; no simulator imports.

Development-only execution never opens reserve features. Full execution requires
both completed evaluation artifacts before reading their fresh image indices.
Reconstructs public controls, raw queries, clipped channels and weighted utility.
Only writes class_conditional_artifact_audit.json after a full passing audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from research.calculations.audit_public_residual_artifacts import (
    C, ROOT, Checks, cached_data, fit_indices, geometry, rank_statistics,
    sha, softmax, summary, train,
)

CODE = Path('research/calculations/class_conditional_probe.py')
PROTOCOL = Path('research/proposals/2026-10-08_class_conditional_protocol.md')
AUDITOR = Path('research/calculations/audit_class_conditional.py')
HELPERS = Path('research/calculations/audit_public_residual_artifacts.py')
DEV = ROOT / 'class_conditional_development.json'
BANK = ROOT / 'class_conditional_development.npz'
EVAL = ROOT / 'class_conditional_evaluation.json'
EVAL_BANK = ROOT / 'class_conditional_evaluation.npz'
MODES = ('raw', 'target_balanced', 'fixed_center', 'class_center', 'center_only', 'target_center')
SCENARIOS = ('baseline', 'prior', 'mask', 'both')
CAPS = (.003, .01, .03, .1, .3)
ETAS = (.1, .3, 1., 3., 10., 30.)
DEPS = [Path('research/calculations') / (name + '.py') for name in (
    'public_residual_probe', 'public_residual_signal_diagnostic', 'private_prior_constructor_probe',
    'private_descriptor_probe', 'client_geometry_audit', 'client_energy_filter_probe',
    'client_energy_one_release', 'matched_cia_probe')]
DEPS += [ROOT / name for name in ('public_residual_development.npz',
                                'public_residual_evaluation.npz', 'private_descriptor_analytic_gaussian.npz')]


def target_prior(scenario):
    return np.asarray([.4, .2, .2, .2]) if scenario in ('prior', 'both') else np.full(4, .25)


def domain(x, scenario):
    value = x.copy()
    if scenario in ('mask', 'both'):
        value[:, [5, 6, 9, 10]] = 0
    return value


def weighted_metrics(x, y, theta, prior):
    """Per-class mean followed by target-prior sum, not record reweighting."""
    theta = np.asarray(theta)
    if theta.ndim == 2:
        theta = theta[None]
    weights = np.einsum('ca,nad->ncd', C, theta)
    z = np.einsum('md,ncd->nmc', x, weights)
    top = z.max(axis=2)
    losses = top + np.log(np.exp(z - top[:, :, None]).sum(axis=2)) - z[:, np.arange(len(y)), y]
    correct = z.argmax(axis=2) == y
    class_ce = np.stack([losses[:, y == k].mean(axis=1) for k in range(4)])
    class_acc = np.stack([correct[:, y == k].mean(axis=1) for k in range(4)])
    return {'ce': prior @ class_ce, 'accuracy': prior @ class_acc,
            'unweighted_ce': losses.mean(axis=1), 'unweighted_accuracy': correct.mean(axis=1)}


def examples(x, y, b):
    residual = softmax(x @ b.T @ C.T) - np.eye(4)[y]
    return (residual @ C)[:, :, None] * x[:, None, :]


def public_gamma(x, y, b):
    g = examples(x, y, b)
    return np.stack([g[y == k].mean(axis=0) for k in range(4)])


def thin(fits, y, scenario):
    if scenario not in ('prior', 'both'):
        return [a.copy() for a in fits]
    result = []
    for a in fits:
        kept = []
        for k in range(4):
            items = a[y[a] == k]
            kept.extend(items if k == 0 else items[:len(items) // 2])
        result.append(np.asarray(kept, dtype=int))
    return result


def queries(x, y, fits, b, gamma, prior):
    out = {mode: [] for mode in MODES}
    proportions, counts = [], []
    restoration = np.tensordot(prior, gamma, axes=1)
    for indices in fits:
        count = np.bincount(y[indices], minlength=4)
        counts.append(count)
        if not len(indices):
            proportions.append(np.zeros(4))
            for mode in MODES:
                out[mode].append(np.zeros_like(b))
            continue
        p = count / len(indices)
        g = examples(x[indices], y[indices], b)
        raw = g.mean(axis=0)
        class_means = np.stack([g[y[indices] == k].mean(axis=0) if count[k] else gamma[k] for k in range(4)])
        balanced = np.tensordot(prior, class_means, axes=1)
        centered = raw - np.tensordot(p, gamma, axes=1)
        values = (raw, balanced, raw - restoration, centered, centered, balanced - restoration)
        for mode, value in zip(MODES, values):
            out[mode].append(value)
        proportions.append(p)
    return {mode: np.asarray(value) for mode, value in out.items()}, np.asarray(proportions), np.asarray(counts)


def offset(mode, gradient, encoder, decoder):
    if mode == 'center_only':
        return np.zeros_like(gradient)
    if mode in ('raw', 'target_balanced'):
        return gradient - (gradient.ravel() @ encoder @ decoder).reshape(3, 17)
    return gradient.copy()


def clipped(query, cap):
    factors = np.minimum(1., cap / np.maximum(np.linalg.norm(query, axis=1), 1e-100))
    return query * factors[:, None] / 8, factors


def corrected(theta, prior):
    value = theta.copy()
    value[:, -1] += np.log(prior / .25) @ C
    return value


def public_models(public, py, b, h, scenario):
    x = domain(public, scenario)
    prior = target_prior(scenario)
    gradient = np.tensordot(prior, public_gamma(x, py, b), axes=1)
    result = {'reference': b, 'prior_corrected_reference': corrected(b, prior)}
    basis, _, _ = geometry(h, 3, 'euclidean')
    result['previous_decoder_offset'] = b - .3 * (b.ravel() @ basis @ basis.T).reshape(3, 17)
    for d in (1, 3, 12, 51):
        basis, _, _ = geometry(h, d, 'euclidean')
        projected = (gradient.ravel() @ basis @ basis.T).reshape(3, 17)
        for eta in (0., *ETAS):
            theta = b - eta * projected
            result[f'public_gradient_d{d}_eta{eta}'] = theta
            result[f'public_gradient_prior_d{d}_eta{eta}'] = corrected(theta, prior)
    for steps in (80, 320, 1280):
        theta = train(x, py, steps)
        for multiplier in (.5, .75, 1., 1.25, 1.5, 2.):
            result[f'public_refit_steps{steps}_mult{multiplier}'] = theta * multiplier
            result[f'public_refit_prior_steps{steps}_mult{multiplier}'] = corrected(theta * multiplier, prior)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    required = [DEV, BANK] + ([] if args.development_only else [EVAL, EVAL_BANK])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit('No receipt written: missing ' + ', '.join(missing))
    check = Checks()
    d = json.loads(DEV.read_text())
    assert d['bank_sha'] == sha(BANK), 'Frozen development archive hash'
    bank = np.load(BANK)
    assert sha(CODE) == d['calculator_sha'] and sha(PROTOCOL) == d['protocol_sha']
    assert set(d['dependencies']) == {str(p) for p in DEPS}
    for path, expected in d['dependencies'].items():
        assert sha(path) == expected, path
    original = np.load(ROOT / 'private_descriptor_analytic_gaussian.npz')
    source, labels, features = cached_data(original)
    assert sha(source) == d['source_hash']
    public_ids, private_ids, dev_ids = [original[name + '_original_indices'] for name in ('public', 'private', 'development')]
    public, py = features(public_ids), labels[public_ids]
    private, y = features(private_ids), labels[private_ids]
    dev_x, dev_y = features(dev_ids), labels[dev_ids]
    previous_bank = np.load(ROOT / 'public_residual_development.npz')
    b, h = bank['reference'], bank['public_hessian']
    check.equal(b, previous_bank['reference'], 'unchanged public reference')
    check.equal(h, previous_bank['public_hessian'], 'unchanged Hessian')
    gamma = public_gamma(public, py, b)
    check.equal(bank['gamma'], gamma, 'original gamma')
    assert len(d['public_controls']) == 4
    controls = {}
    for control in d['public_controls']:
        scenario = control['scenario']
        assert scenario not in controls
        controls[scenario] = control
        prior = target_prior(scenario)
        check.equal(control['prior'], prior, 'public objective')
        models = public_models(public, py, b, h, scenario)
        assert len(control['candidates']) == 95 and {c['name'] for c in control['candidates']} == set(models)
        for record in control['candidates']:
            check.equal(record['theta'], models[record['name']], 'public model reconstruction')
            # Score saved floats after validating the independently fitted model.
            calculated = weighted_metrics(domain(dev_x, scenario), dev_y, np.asarray(record['theta']), prior)
            check.equal(record['development_ce'], calculated['ce'][0], 'public weighted CE')
            check.equal(record['development_accuracy'], calculated['accuracy'][0], 'public weighted accuracy')
        assert control['selected'] == min(control['candidates'], key=lambda c: c['development_ce'])
    assert set(controls) == set(SCENARIOS)
    route_info = []
    for seed in (42, 43, 44):
        fits = fit_indices(y, seed)
        for scenario in SCENARIOS:
            selected_fits = thin(fits, y, scenario)
            x = domain(private, scenario)
            for route in (('stale', 'refreshed') if scenario in ('mask', 'both') else ('stale',)):
                sg = gamma if route == 'stale' else public_gamma(domain(public, scenario), py, b)
                q, priors, counts = queries(x, y, selected_fits, b, sg, target_prior(scenario))
                prefix = f'seed{seed}_{scenario}_{route}'
                check.equal(bank[prefix + '_gamma'], sg, 'route gamma')
                check.equal(bank[prefix + '_priors'], priors, 'internal client priors')
                check.equal(bank[prefix + '_counts'], counts, 'thinned counts')
                for mode in MODES:
                    check.equal(bank[prefix + '_' + mode + '_query'], q[mode], 'full client query/' + mode)
                mean_prior = priors.mean(axis=0)
                pooled_prior = counts.sum(axis=0) / counts.sum()
                for suffix, expected in (('_mean_client_prior', mean_prior), ('_pooled_record_prior', pooled_prior)):
                    check.equal(bank[prefix + suffix], expected, 'saved prior diagnostic')
                raw_identity = np.tensordot(target_prior(scenario), sg, axes=1) + q['class_center'].mean(axis=0) - q['raw'].mean(axis=0)
                expected_difference = np.tensordot(target_prior(scenario) - mean_prior, sg, axes=1)
                check.equal(raw_identity, expected_difference, 'all-IN unbounded prior identity', tolerance=1e-14)
                route_info.append({'seed': seed, 'scenario': scenario, 'reference_source': route,
                                   'fit_count': int(counts.sum()), 'mean_client_prior': mean_prior.tolist(),
                                   'pooled_record_prior': pooled_prior.tolist(),
                                   'target_objective_prior': target_prior(scenario).tolist()})
    selection = {}
    assert len(d['rows']) == 6 and d['candidate_count'] == 8640 and d['draws_per_stratum'] == 32
    for row in d['rows']:
        seed, risk = row['seed'], row['risk']
        assert (seed, risk) not in selection
        selection[seed, risk] = row
        assert len(row['candidates']) == 1440 and set(row['selected']) == set(MODES)
        configurations = set()
        for record in row['candidates']:
            c = record['configuration']
            token = (c['mode'], c['dimension'], c['metric'], c['cap'], c['eta'])
            assert token not in configurations
            configurations.add(token)
            values = bank[record['archive_key'] + '_ce']
            assert values.shape == (8, 32)
            check.equal(record['development_ce'], values.mean(), 'candidate eight-stratum mean')
            _, encoder, _ = geometry(h, c['dimension'], c['metric'])
            query = bank[f'seed{seed}_baseline_stale_{c["mode"]}_query'].reshape(8, 51) @ encoder
            _, factors = clipped(query, c['cap'])
            check.equal(record['clip_factors'], factors, 'development clipping')
        assert configurations == {(m, dim, metric, cap, eta) for m in MODES for dim in (1, 3, 12, 51)
                                  for metric in ('euclidean', 'fisher') for cap in CAPS for eta in ETAS}
        for mode in MODES:
            assert row['selected'][mode] == min((c for c in row['candidates'] if c['configuration']['mode'] == mode), key=lambda c: c['development_ce'])
    assert set(selection) == {(s, q) for s in (42, 43, 44) for q in (.55, .65)}
    if args.development_only:
        print(json.dumps({'kind': 'Development-only audit; no receipt or reserve features opened',
                          'numeric_checks': check.count, 'maximum_absolute_error': check.maximum,
                          'public_control_candidates': 380, 'private_candidates': 8640}))
        return
    e = json.loads(EVAL.read_text())
    archive = np.load(EVAL_BANK)
    assert e['development_sha'] == sha(DEV) and e['development_archive_sha'] == sha(BANK)
    assert e['calculator_sha'] == sha(CODE) and e['source_hash'] == sha(source)
    assert len(e['rows']) == 1176 and e['utility_count'] == 320 and e['remaining_reserve_count'] == 136
    prior_archive = np.load(ROOT / 'public_residual_evaluation.npz')
    previous = prior_archive['remaining_reserve_original_indices']
    new, remaining = archive['utility_original_indices'], archive['remaining_reserve_original_indices']
    expected = np.concatenate([previous[labels[previous] == k][:80] for k in range(4)])
    assert np.array_equal(new, expected)
    assert len(new) == len(set(new)) == 320 and len(remaining) == len(set(remaining)) == 136
    assert not set(new) & set(remaining) and set(new) | set(remaining) == set(previous) and len(previous) == 456
    for role in ('public', 'private', 'development', 'evaluation'):
        assert not set(new) & set(original[role + '_original_indices'])
    for file, key in (('private_prior_constructor_evaluation.npz', 'confirmation_original_indices'),
                      ('descriptor_cia_probe.npz', 'utility_original_indices'),
                      ('matched_cia_evaluation.npz', 'utility_original_indices'),
                      ('public_residual_evaluation.npz', 'utility_original_indices')):
        assert not set(new) & set(np.load(ROOT / file)[key])
    ancient = np.load(ROOT / 'client_geometry_audit_matrices.npz')
    for seed in (42, 43, 44):
        for role in ('pool', 'development'):
            assert not set(new) & set(ancient[f'seed{seed}_{role}_original_indices'])
    ux, uy = features(new), labels[new]
    check.equal(np.bincount(uy, minlength=4), [80] * 4, 'fresh class counts')
    assert len(e['public_controls']) == 4
    evaluated_controls = {}
    for control in e['public_controls']:
        scenario = control['scenario']
        assert scenario not in evaluated_controls
        evaluated_controls[scenario] = control
        selected = controls[scenario]['selected']
        assert control['selection'] == selected['name']
        calculated = weighted_metrics(domain(ux, scenario), uy, np.asarray(selected['theta']), target_prior(scenario))
        check.equal(control['ce'], calculated['ce'][0], 'frozen public weighted reserve CE')
        check.equal(control['accuracy'], calculated['accuracy'][0], 'frozen public weighted reserve accuracy')
    seen = set()
    auc_deviations = []
    channel_cache = {}
    attack_innovations = {}
    innovation_group = None
    for row in e['rows']:
        seed, risk, scenario, route, arm, target = [row[k] for k in ('seed', 'risk', 'scenario', 'reference_source', 'arm', 'target')]
        identifier = (seed, risk, scenario, route, arm, target)
        assert identifier not in seen
        seen.add(identifier)
        assert target in range(8) and row['peer'] == (6 if target == 7 else 7) and row['peer'] != target
        central = arm == 'central_class_center'
        mode = 'class_center' if central else arm
        config = selection[seed, risk]['selected'][mode]['configuration']
        assert row['configuration'] == config
        prefix = f'seed{seed}_{scenario}_{route}'
        key = f'{prefix}_q{int(risk*100)}_{arm}_target{target}'
        assert row['archive_key'] == key
        channel = (seed, risk, scenario, route, arm)
        if channel not in channel_cache:
            _, encoder, decoder = geometry(h, config['dimension'], config['metric'])
            query = bank[prefix + '_' + mode + '_query'].reshape(8, 51) @ encoder
            signal, factors = clipped(query, config['cap'])
            gradient = np.tensordot(target_prior(scenario), bank[prefix + '_gamma'], axes=1)
            restoration = offset(mode, gradient, encoder, decoder)
            check.equal(archive[f'{prefix}_q{int(risk*100)}_{arm}_signal'], signal, 'saved bounded channel')
            check.equal(archive[f'{prefix}_q{int(risk*100)}_{arm}_offset'], restoration, 'public restoration')
            assert np.linalg.norm(signal, axis=1).max() <= config['cap'] / 8 + 1e-12
            channel_cache[channel] = (signal, factors, decoder, restoration)
        signal, factors, decoder, restoration = channel_cache[channel]
        sigma = config['cap'] / (8 * np.sqrt(2) * ndtri(risk))
        check.equal(row['sigma'], sigma, 'public cap sigma')
        check.equal(row['clip_factors'], factors, 'evaluation clipping')
        norm = float(np.linalg.norm(signal[target]))
        check.equal(row['target_shift_norm'], norm, 'weighted target shift')
        expected_auc = float(ndtr(norm / (np.sqrt(2) * sigma)))
        check.equal(row['gaussian_expected_auc'], expected_auc, 'exact Gaussian AUC')
        assert expected_auc <= risk + 1e-12
        ranks = [archive[key + f'_world{w}_attack_score'] for w in (0, 1)]
        assert all(v.shape == (2048,) for v in ranks)
        statistics = rank_statistics(ranks[1], ranks[0])
        check.equal(row['attack'], statistics, 'ordinary oriented ROC/SE/low-FPR')
        auc_deviations.append(abs(statistics['auc'] - expected_auc) / statistics['se'] if statistics['se'] else 0.)
        for world in (0, 1):
            active = signal.copy()
            if world == 0:
                active[target] = 0
            rng_seed = 102000000 + seed * 100000 + int(risk * 100) * 1000 + target * 10 + world
            # Reconstruct the actual observer, including attacker-owned share
            # subtraction, rather than relying on an expected-AUC calculation.
            # Retain at most one seed/risk's16 fresh streams (~107MB).
            if innovation_group != (seed, risk):
                attack_innovations.clear()
                innovation_group = (seed, risk)
            if (target, world) not in attack_innovations:
                attack_innovations[target, world] = np.random.default_rng(rng_seed).normal(size=(2048, 8, 51))
            objects = active[None] + attack_innovations[target, world][:, :, :config['dimension']] * sigma / np.sqrt(8 if central else 7)
            absent = signal.copy()
            absent[target] = 0
            base = absent.sum(axis=0)
            observed = objects.sum(axis=1)
            if not central:
                base -= absent[row['peer']]
                observed -= objects[:, row['peer']]
            reconstructed_rank = (observed - base - signal[target] / 2) @ signal[target] / sigma**2
            check.equal(ranks[world], reconstructed_rank, 'fresh peer-conditioned exact LR vector')
            # Fixed 51-coordinate stream matters for shared-prefix CRN controls.
            z = np.random.default_rng(rng_seed + 10000000).normal(size=(128, 8, 51))[:, :, :config['dimension']]
            noise = z.sum(axis=1) * sigma / np.sqrt(8 if central else 7)
            theta = b[None] - config['eta'] * (restoration[None] + ((active.sum(axis=0)[None] + noise) @ decoder).reshape(-1, 3, 17))
            calculated = weighted_metrics(domain(ux, scenario), uy, theta, target_prior(scenario))
            for field, value in calculated.items():
                saved = archive[key + f'_world{world}_utility_{field}']
                assert saved.shape == (128,)
                check.equal(saved, value, 'independently reconstructed weighted/unweighted utility')
                if field in ('ce', 'accuracy'):
                    check.equal(row['utility'][str(world)][field], summary(saved), 'utility summary')
            noiseless = b - config['eta'] * (restoration + (active.sum(axis=0) @ decoder).reshape(3, 17))
            calculated_no = weighted_metrics(domain(ux, scenario), uy, noiseless, target_prior(scenario))
            check.equal(row['noiseless'][str(world)], {f: calculated_no[f][0] for f in ('ce', 'accuracy')}, 'noiseless utility')
    expected_cells = set()
    for seed in (42, 43, 44):
        for risk in (.55, .65):
            for scenario in (SCENARIOS if risk == .55 else ('baseline',)):
                for route in (('stale', 'refreshed') if scenario in ('mask', 'both') else ('stale',)):
                    for arm in (*MODES, 'central_class_center'):
                        for target in range(8):
                            expected_cells.add((seed, risk, scenario, route, arm, target))
    assert seen == expected_cells
    contrasts = []
    for seed, risk, scenario, route, _, target in sorted(seen):
        if target != 0:
            continue
        # Iterating unique routes, not repeated arms or targets.
        token = (seed, risk, scenario, route)
        if any((r['seed'], r['risk'], r['scenario'], r['reference_source']) == token for r in contrasts):
            continue
        for reference_mode in ('public_only', 'raw', 'target_balanced', 'fixed_center', 'center_only', 'target_center', 'central_class_center'):
            strata = []
            for t in range(4):
                for world in (0, 1):
                    candidate = archive[f'seed{seed}_{scenario}_{route}_q{int(risk*100)}_class_center_target{t}_world{world}_utility_ce']
                    other = evaluated_controls[scenario]['ce'] if reference_mode == 'public_only' else archive[f'seed{seed}_{scenario}_{route}_q{int(risk*100)}_{reference_mode}_target{t}_world{world}_utility_ce']
                    strata.append(other - candidate)
            gain = float(np.mean([v.mean() for v in strata]))
            se = float(np.sqrt(sum(v.var(ddof=1) / len(v) for v in strata)) / 8)
            contrasts.append({'seed': seed, 'risk': risk, 'scenario': scenario, 'reference_source': route,
                              'reference_arm': reference_mode, 'candidate': 'class_center', 'gain': gain,
                              'fixed_development_strongest_nonprimary': min(
                                  (m for m in MODES if m != 'class_center'),
                                  key=lambda m: selection[seed, risk]['selected'][m]['development_ce']),
                              'se': se, 'ci95': [gain - 1.96 * se, gain + 1.96 * se]})
    receipt = {'kind': 'Independent fixed-client, conditional Gaussian channel artifact audit; offline tuning is not end-to-end private.',
               'development_rows': 6, 'development_configurations': 8640, 'public_candidates': 380,
               'query_routes': len(route_info), 'evaluation_rows': 1176, 'gaussian_ceiling_checks': len(auc_deviations),
               'maximum_gaussian_auc_deviation_in_standard_errors': max(auc_deviations),
               'numeric_checks': check.count, 'maximum_absolute_error': check.maximum,
               'weighted_and_unweighted_utility_reconstructed': True, 'fresh_utility_count': 320,
               'peer_conditioned_attack_score_vectors_reconstructed': 2352,
               'remaining_reserve_count': 136, 'route_priors': route_info, 'contrast_rows': len(contrasts),
               'contrasts': contrasts,
               'hashes': {str(p): sha(p) for p in (*required, CODE, PROTOCOL, AUDITOR, HELPERS)}}
    (ROOT / 'class_conditional_artifact_audit.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('contrasts', 'route_priors', 'hashes')}, indent=2))


if __name__ == '__main__':
    main()
