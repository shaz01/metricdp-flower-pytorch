"""Public-cap residual/compression research envelope, not privately tuned deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from scipy.special import ndtr, ndtri

from research.calculations.client_energy_filter_probe import C, scores, summarize
from research.calculations.client_energy_one_release import local_query
from research.calculations.client_geometry_audit import features, partition, split
from research.calculations.matched_cia_probe import auc_statistics
from research.calculations.private_descriptor_probe import ROOT, encode, decode, probabilities
from research.calculations.private_prior_constructor_probe import prepare

CAPS = (.025, .05, .1, .2, .4, .8, 1.6)
GAINS = (.1, .3, 1.)
DIMENSIONS = (1, 3, 12, 51)
RISKS = (.55, .65)
PROTOCOL = Path('research/proposals/2026-10-08_public_residual_protocol.md')
DEV_JSON = ROOT / 'public_residual_development.json'
DEV_NPZ = ROOT / 'public_residual_development.npz'
DEPENDENCIES = (
    ROOT / 'private_descriptor_analytic_gaussian.npz',
    Path('research/calculations/private_descriptor_probe.py'),
    Path('research/calculations/private_prior_constructor_probe.py'),
    Path('research/calculations/client_geometry_audit.py'),
    Path('research/calculations/client_energy_filter_probe.py'),
    Path('research/calculations/client_energy_one_release.py'),
    Path('research/calculations/matched_cia_probe.py'),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_hessian(x, reference):
    p = probabilities(x @ reference.T)
    s = -p[:, :, None] * p[:, None, :]
    s[:, np.arange(4), np.arange(4)] += p
    identifiable = np.einsum('ca,ncd,db->nab', C, s, C)
    return np.einsum('nab,ni,nj->aibj', identifiable, x, x, optimize=True).reshape(51, 51) / len(x)


def public_geometry(hessian, dimension, metric):
    values, vectors = np.linalg.eigh(hessian)
    order = np.argsort(values)[::-1][:dimension]
    basis = vectors[:, order].copy()
    # A deterministic sign convention binds saved public operators across stages.
    signs = np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(dimension)])
    basis *= signs
    values = np.maximum(values[order], 0)
    scaling = np.sqrt(values + .01) if metric == 'fisher' else np.ones(dimension)
    return {'basis': basis, 'eigenvalues': values, 'encoder': basis * scaling,
            'decoder': (basis / scaling).T}


def local_residual(x, y, reference, basis, steps, balanced):
    if not len(y):
        return np.zeros_like(reference)
    theta = reference.copy()
    prior = (np.bincount(y, minlength=4) + 1) / (len(y) + 4)
    weights = 1 / prior[y] if balanced else np.ones(len(y))
    weights /= weights.mean()
    for _ in range(steps):
        p = probabilities(x @ theta.T)
        p[np.arange(len(y)), y] -= 1
        gradient = C.T @ ((p * weights[:, None]).T @ x / len(y))
        gradient = (gradient.ravel() @ basis @ basis.T).reshape(3, 17)
        theta -= .5 * gradient
    return theta - reference


def projected_query(teachers, reference, geometry, kind, active_mask=None):
    source = teachers - reference if kind == 'centered' else teachers
    result = source.reshape(len(source), 51) @ geometry['encoder']
    if active_mask is not None:
        result[~np.asarray(active_mask, dtype=bool)] = 0
    return result


def query_offset(reference, geometry, kind, gain):
    if kind == 'absolute':
        projection = reference.ravel() @ geometry['encoder'] @ geometry['decoder']
        return reference - gain * projection.reshape(3, 17)
    return reference


def bounded_public(query, cap):
    norms = np.linalg.norm(query, axis=1)
    factors = np.minimum(1., cap / np.maximum(norms, 1e-100))
    return query * factors[:, None] / 8, factors


def public_scale(cap, risk):
    if cap <= 0 or not .5 < risk < 1:
        raise ValueError('Positive public cap and risk in(.5,1) required')
    return cap / (8 * np.sqrt(2) * ndtri(risk))


def decode_total(total, config, reference, hessian, anchors, decoders, denoisers):
    gain = config['gain']
    if config['kind'] == 'legacy':
        old = decode(total[:, None, :], np.ones(8) / 8, config['legacy_configuration'],
                     'aggregate', anchors, decoders, denoisers)
        return reference[None] + gain * (old - reference[None])
    geometry = public_geometry(hessian, config['dimension'], config['metric'])
    offset = query_offset(reference, geometry, config['kind'], gain)
    return offset[None] + gain * (total @ geometry['decoder']).reshape(-1, 3, 17)


def strata_means(signal):
    means = np.tile(signal.sum(axis=0), (8, 1))
    # Four targets, OUT then IN. Every stratum has separate innovations.
    for target in range(4):
        means[2 * target] -= signal[target]
    return means


def random_shares(draws, seed):
    return np.random.default_rng(seed).normal(size=(draws, 8, 51))


def development():
    stored = np.load(ROOT / 'private_prior_constructor_development.npz')
    old = json.loads((ROOT / 'matched_cia_development.json').read_text())
    source, ids, public, x, y, dev, anchors, decoders, denoisers, _, _ = prepare(False)
    assert digest(source) == old['source_hash']
    labels = np.array(Dataset.from_file(str(source))['label'])
    references = []
    for steps in (80, 320, 1280):
        theta = local_query(public, labels[ids['public']], steps)
        for temperature in (.5, .75, 1., 1.25, 1.5, 2.):
            b = theta * temperature
            ce, acc = scores(*dev, b[None])
            references.append({'steps': steps, 'temperature': temperature,
                               'ce': float(ce[0]), 'accuracy': float(acc[0]), 'reference': b.tolist()})
    public_choice = min(references, key=lambda v: v['ce'])
    reference = np.array(public_choice['reference'])
    hessian = public_hessian(public, reference)
    saved = {'reference': reference, 'public_hessian': hessian}
    rows = []
    print('public-only development', public_choice['steps'], public_choice['temperature'],
          public_choice['ce'], public_choice['accuracy'], flush=True)
    for seed in (42, 43, 44):
        fit, _ = split(partition(y, 'label_stress', seed), y, seed)
        fine = {}
        for dimension in DIMENSIONS:
            basis = public_geometry(hessian, dimension, 'euclidean')['basis']
            for transform in ('raw', 'balanced_loss'):
                for steps in (20, 80):
                    value = np.stack([local_residual(x[a], y[a], reference, basis, steps,
                                                    transform == 'balanced_loss') for a in fit])
                    fine[dimension, transform, steps] = value
                    saved[f'seed{seed}_fine_d{dimension}_{transform}_steps{steps}'] = value
        for risk in RISKS:
            old_row = next(r for r in old['rows'] if r['seed'] == seed and r['risk'] == risk and r['law'] == 'gaussian')
            # Prefix coordinates share innovations across dimensions/caps/gains.
            innovations = np.stack([random_shares(32, 91000000 + seed * 100000 + int(risk * 100) * 1000 + s)
                                    .sum(axis=1) / np.sqrt(7) for s in range(8)])
            choices = []; selected = {}
            templates = []
            for kind in ('absolute', 'centered', 'fine'):
                transforms = ('raw', 'balanced_loss') if kind == 'fine' else ('raw', 'prior', 'balanced_loss')
                for transform in transforms:
                    for steps in (20, 80):
                        for dimension in DIMENSIONS:
                            for metric in ('euclidean', 'fisher'):
                                g = public_geometry(hessian, dimension, metric)
                                teacher = fine[dimension, transform, steps] if kind == 'fine' else stored[f'seed{seed}_{transform}_steps{steps}_teachers']
                                query = projected_query(teacher, reference, g, kind,
                                                        active_mask=[len(a) > 0 for a in fit])
                                config = {'kind': kind, 'transform': transform, 'steps': steps,
                                          'dimension': dimension, 'metric': metric}
                                templates.append((kind, config, query))
            for family, prior in old_row['selected'].items():
                c = prior['configuration']
                teacher = stored[f'seed{seed}_{prior["transform"]}_steps{c["steps"]}_teachers']
                query = encode(teacher, family, anchors.get(c['prototypes']),
                               active_mask=[len(a) > 0 for a in fit])
                templates.append(('legacy_' + family, {'kind': 'legacy', 'transform': prior['transform'],
                                 'dimension': query.shape[1], 'legacy_configuration': c}, query))
            for group, template, query in templates:
                for cap in CAPS:
                    signal, factors = bounded_public(query, cap)
                    sigma = public_scale(cap, risk)
                    d = signal.shape[1]
                    total = (strata_means(signal)[:, None] + innovations[:, :, :d] * sigma).reshape(-1, d)
                    for gain in GAINS:
                        config = dict(template, cap=cap, gain=gain)
                        theta = decode_total(total, config, reference, hessian, anchors, decoders, denoisers)
                        ce, _ = scores(*dev, theta)
                        key = f'seed{seed}_q{int(risk*100)}_choice{len(choices)}'
                        saved[key + '_ce'] = ce.reshape(8, 32)
                        record = {'group': group, 'configuration': config, 'development_ce': float(ce.mean()),
                                  'archive_key': key, 'clip_factors': factors.tolist(),
                                  'max_shift_norm': float(np.linalg.norm(signal, axis=1).max()), 'sigma': sigma}
                        choices.append(record)
                        if group not in selected or record['development_ce'] < selected[group]['development_ce']:
                            selected[group] = record
            for group, record in selected.items():
                saved[f'seed{seed}_q{int(risk*100)}_{group}_signal'] = construct_signal(record['configuration'], seed, reference, hessian, stored, fine, anchors)[0]
            strongest = min((v for k, v in selected.items() if k != 'fine'), key=lambda v: v['development_ce'])
            reference_name = 'public_only' if public_choice['ce'] <= strongest['development_ce'] else strongest['group']
            rows.append({'seed': seed, 'risk': risk, 'candidates': choices, 'selected': selected,
                         'strongest_alternative': reference_name})
            print('development', seed, risk, 'fine', selected['fine']['development_ce'],
                  'alternative', reference_name, min(public_choice['ce'], strongest['development_ce']), flush=True)
    np.savez_compressed(DEV_NPZ, **saved)
    DEV_JSON.write_text(json.dumps({'source_hash': digest(source), 'protocol_sha': digest(PROTOCOL),
                        'preceding_development_sha': digest(ROOT / 'matched_cia_development.json'),
                        'teacher_archive_sha': digest(ROOT / 'private_prior_constructor_development.npz'),
                        'calculator_sha': digest(__file__),
                        'dependency_sha': {str(p): digest(p) for p in DEPENDENCIES},
                        'public_choices': references, 'public_choice': public_choice,
                        'public_role_count': len(public), 'draws_per_stratum': 32, 'rows': rows}, indent=2, allow_nan=False) + '\n')


def construct_signal(config, seed, reference, hessian, stored, fine, anchors):
    if config['kind'] == 'legacy':
        c = config['legacy_configuration']
        teacher = stored[f'seed{seed}_{config["transform"]}_steps{c["steps"]}_teachers']
        active = stored[f'seed{seed}_private_class_counts'].sum(axis=1) > 0
        query = encode(teacher, c['family'], anchors.get(c['prototypes']), active_mask=active)
    else:
        g = public_geometry(hessian, config['dimension'], config['metric'])
        teacher = fine[config['dimension'], config['transform'], config['steps']] if config['kind'] == 'fine' else stored[f'seed{seed}_{config["transform"]}_steps{config["steps"]}_teachers']
        active = stored[f'seed{seed}_private_class_counts'].sum(axis=1) > 0
        query = projected_query(teacher, reference, g, config['kind'], active_mask=active)
    return bounded_public(query, config['cap'])


def evaluation():
    development = json.loads(DEV_JSON.read_text()); bank = np.load(DEV_NPZ)
    assert digest(PROTOCOL) == development['protocol_sha']
    assert digest(__file__) == development['calculator_sha']
    assert digest(ROOT / 'matched_cia_development.json') == development['preceding_development_sha']
    assert digest(ROOT / 'private_prior_constructor_development.npz') == development['teacher_archive_sha']
    for dependency, expected in development['dependency_sha'].items():
        assert digest(dependency) == expected
    source, ids, public, x, y, dev, anchors, decoders, denoisers, _, _ = prepare(False)
    assert digest(source) == development['source_hash']
    data = Dataset.from_file(str(source)); labels = np.array(data['label'])
    unused = np.load(ROOT / 'matched_cia_evaluation.npz')['remaining_reserve_original_indices']
    new = np.concatenate([unused[labels[unused] == k][:128] for k in range(4)])
    assert len(new) == 512 and len(set(new)) == 512
    remaining = np.setdiff1d(unused, new)
    assert len(remaining) == 456 and not set(new) & set(remaining)
    assert set(new) | set(remaining) == set(unused)
    for role in ('public', 'private', 'development', 'evaluation'):
        assert not set(new) & set(ids[role])
    previous = np.load(ROOT / 'private_prior_constructor_evaluation.npz')
    assert not set(new) & set(previous['confirmation_original_indices'])
    preceding = np.load(ROOT / 'matched_cia_evaluation.npz')
    assert not set(new) & set(preceding['utility_original_indices'])
    utility = (features(data, new), labels[new])
    reference, hessian = bank['reference'], bank['public_hessian']
    baseline_ce, baseline_acc = scores(*utility, reference[None])
    saved = {'utility_original_indices': new, 'remaining_reserve_original_indices': remaining}
    rows = []
    for record in development['rows']:
        seed, risk = record['seed'], record['risk']
        arms = dict(record['selected']); arms['central_fine'] = record['selected']['fine']
        for arm, choice in arms.items():
            config = choice['configuration']; sigma = public_scale(config['cap'], risk)
            d = config['dimension']; signal = bank[f'seed{seed}_q{int(risk*100)}_{choice["group"]}_signal']
            central = arm == 'central_fine'
            for target in range(8):
                peer = 6 if target == 7 else 7
                key = f'seed{seed}_q{int(risk*100)}_{arm}_target{target}'
                out = signal.copy(); out[target] = 0
                shift = signal[target]; base = out.sum(axis=0) if central else out.sum(axis=0) - out[peer]
                utility_stats = {}; attack_scores = []; noiseless_stats = {}
                for world in (0, 1):
                    active = signal if world else out
                    rng_seed = 92000000 + seed * 100000 + int(risk * 100) * 1000 + target * 10 + world
                    objects = active[None] + random_shares(4096, rng_seed)[:, :, :d] * sigma / np.sqrt(8 if central else 7)
                    observed = objects.sum(axis=1) if central else objects.sum(axis=1) - objects[:, peer]
                    rank = (observed - base - shift / 2) @ shift / sigma**2
                    saved[key + f'_world{world}_attack_score'] = rank; attack_scores.append(rank)
                    noise = random_shares(256, rng_seed + 10000000)[:, :, :d].sum(axis=1) * sigma / np.sqrt(8 if central else 7)
                    theta = decode_total(active.sum(axis=0)[None] + noise, config, reference, hessian, anchors, decoders, denoisers)
                    ce, acc = scores(*utility, theta)
                    for field, value in (('ce', ce), ('accuracy', acc)):
                        saved[key + f'_world{world}_utility_{field}'] = value
                    utility_stats[str(world)] = {'ce': summarize(ce), 'accuracy': summarize(acc)}
                    no_noise_theta = decode_total(active.sum(axis=0)[None], config, reference, hessian, anchors, decoders, denoisers)
                    raw_ce, raw_acc = scores(*utility, no_noise_theta)
                    noiseless_stats[str(world)] = {'ce': float(raw_ce[0]), 'accuracy': float(raw_acc[0])}
                expected = float(ndtr(np.linalg.norm(shift) / (np.sqrt(2) * sigma)))
                assert expected <= risk + 1e-12
                rows.append({'seed': seed, 'risk': risk, 'arm': arm, 'target': target, 'peer': peer,
                             'configuration': config, 'sigma': sigma, 'public_weighted_cap': config['cap'] / 8,
                             'target_shift_norm': float(np.linalg.norm(shift)), 'gaussian_expected_auc': expected,
                             'attack': auc_statistics(attack_scores[1], attack_scores[0]),
                             'utility': utility_stats, 'noiseless_diagnostic': noiseless_stats})
        print('evaluation', seed, risk, len(arms), 'arms', flush=True)
    np.savez_compressed(ROOT / 'public_residual_evaluation.npz', **saved)
    (ROOT / 'public_residual_evaluation.json').write_text(json.dumps({
        'source_hash': development['source_hash'], 'development_sha': digest(DEV_JSON),
        'development_archive_sha': digest(DEV_NPZ), 'calculator_sha': digest(__file__),
        'preceding_evaluation_archive_sha': digest(ROOT / 'matched_cia_evaluation.npz'),
        'utility_count': len(new), 'remaining_reserve_count': len(saved['remaining_reserve_original_indices']),
        'public_only': {'ce': float(baseline_ce[0]), 'accuracy': float(baseline_acc[0])},
        'rows': rows}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('development', 'evaluation'), default='development')
    args = parser.parse_args()
    development() if args.stage == 'development' else evaluation()
