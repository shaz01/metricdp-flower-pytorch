"""Independent public-residual saved-artifact audit (no experiment imports).

Run from the repository root with ``uv run python -m
research.calculations.audit_public_residual_artifacts``. This only reads saved
results/cached images and writes the contrast synthesis and audit receipt after
all checks pass. --development-only checks frozen development without a receipt.
The optional --skip-image-reconstruction omits teacher/Hessian/data-label checks
and explicitly records that limitation. Neither route performs selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

ROOT = Path('results/client_specific_noise')
PROTOCOL = Path('research/proposals/2026-10-08_public_residual_protocol.md')
SIMULATOR = Path('research/calculations/public_residual_probe.py')
AUDITOR = Path('research/calculations/audit_public_residual_artifacts.py')
PUBLIC_CONTROL_CODE = Path('research/calculations/public_residual_public_control.py')
PUBLIC_CONTROL_PROTOCOL = Path('research/proposals/2026-10-08_public_residual_control_addendum.md')
PUBLIC_CONTROL_DEV = ROOT / 'public_residual_public_control_development.json'
PUBLIC_CONTROL_EVAL = ROOT / 'public_residual_public_control_evaluation.json'
DEPENDENCIES = (ROOT / 'private_descriptor_analytic_gaussian.npz', *[
    Path('research/calculations') / (name + '.py') for name in (
        'private_descriptor_probe', 'private_prior_constructor_probe',
        'client_geometry_audit', 'client_energy_filter_probe',
        'client_energy_one_release', 'matched_cia_probe')])
CAPS = (.025, .05, .1, .2, .4, .8, 1.6)
GAINS = (.1, .3, 1.)
GROUPS = ('absolute', 'centered', 'fine', 'legacy_model_direct',
          'legacy_model_distribution', 'legacy_logits', 'legacy_probabilities', 'legacy_votes')
C = np.zeros((4, 3))
for j in range(3):
    C[:j + 1, j] = 1 / np.sqrt((j + 1) * (j + 2))
    C[j + 1, j] = -(j + 1) / np.sqrt((j + 1) * (j + 2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0
        self.maximum = 0.

    def equal(self, actual, expected, name='', tolerance=1e-10):
        if isinstance(expected, dict):
            for key, value in expected.items():
                self.equal(actual[key], value, name + '/' + key, tolerance)
            return
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), name
        error = float(np.max(np.abs(a - b), initial=0))
        self.count += a.size
        self.maximum = max(self.maximum, error)
        assert error <= tolerance, (name, error, tolerance)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def metrics(x, y, theta):
    logits = x @ (C @ theta).T
    top = logits.max(axis=1)
    ce = np.mean(top + np.log(np.exp(logits - top[:, None]).sum(axis=1))
                 - logits[np.arange(len(y)), y])
    return {'ce': float(ce), 'accuracy': float(np.mean(logits.argmax(axis=1) == y))}


def geometry(h, dimension, metric):
    values, vectors = np.linalg.eigh(h)
    order = np.argsort(values)[::-1][:dimension]
    basis = vectors[:, order]
    basis *= np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(dimension)])
    scales = np.sqrt(np.maximum(values[order], 0) + .01) if metric == 'fisher' else np.ones(dimension)
    return basis, basis * scales, (basis / scales).T


def train(x, y, steps, initial=None, basis=None, balanced=False):
    theta = np.zeros((3, 17)) if initial is None else initial.copy()
    if not len(y):
        return theta
    weights = np.ones(len(y))
    if balanced:
        prior = (np.bincount(y, minlength=4) + 1) / (len(y) + 4)
        weights = 1 / prior[y]
        weights /= weights.mean()
    for _ in range(steps):
        residual = softmax(x @ (C @ theta).T)
        residual[np.arange(len(y)), y] -= 1
        gradient = C.T @ ((residual * weights[:, None]).T @ x) / len(y)
        if basis is not None:
            gradient = ((gradient.ravel() @ basis) @ basis.T).reshape(3, 17)
        theta -= .5 * gradient
    return theta


def fit_indices(y, seed):
    # Independently reproduce the documented class-count and stratified halves.
    counts = np.full((8, 4), 34)
    counts[np.arange(8), np.arange(8) % 4] = 410
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(8)]
    for klass in range(4):
        ids = np.flatnonzero(y == klass)
        rng.shuffle(ids)
        start = 0
        for client in range(8):
            stop = start + counts[client, klass]
            parts[client].extend(ids[start:stop])
            start = stop
    for part in parts:
        rng.shuffle(part)
    rng = np.random.default_rng(seed)
    fit = []
    for part in parts:
        keep = []
        for klass in range(4):
            ids = np.asarray([i for i in part if y[i] == klass], dtype=int)
            rng.shuffle(ids)
            keep.extend(ids[:len(ids) // 2])
        fit.append(np.asarray(keep, dtype=int))
    return fit


def encoded(teachers, family, anchors):
    if family.startswith('model'):
        return teachers.reshape(8, 51)
    contrasts = np.einsum('md,icd->imc', anchors, teachers)
    if family == 'logits':
        return contrasts.reshape(8, -1) / np.sqrt(len(anchors))
    p = softmax(contrasts @ C.T)
    if family == 'votes':
        p = np.eye(4)[p.argmax(axis=-1)]
    return (p @ C).reshape(8, -1) / np.sqrt(len(anchors))


def signal(config, seed, reference, hessian, bank, teachers, original):
    if config['kind'] == 'legacy':
        legacy = config['legacy_configuration']
        theta = teachers[f'seed{seed}_{config["transform"]}_steps{legacy["steps"]}_teachers']
        anchors = original['anchors' + str(legacy['prototypes'])] if legacy['prototypes'] else None
        query = encoded(theta, legacy['family'], anchors)
    else:
        _, encoder, _ = geometry(hessian, config['dimension'], config['metric'])
        if config['kind'] == 'fine':
            theta = bank[f'seed{seed}_fine_d{config["dimension"]}_{config["transform"]}_steps{config["steps"]}']
        else:
            theta = teachers[f'seed{seed}_{config["transform"]}_steps{config["steps"]}_teachers']
            if config['kind'] == 'centered':
                theta = theta - reference
        query = theta.reshape(8, 51) @ encoder
    active = teachers[f'seed{seed}_private_class_counts'].sum(axis=1) > 0
    query[~active] = 0
    norms = np.linalg.norm(query, axis=1)
    factors = np.minimum(1., config['cap'] / np.maximum(norms, 1e-100))
    return query * factors[:, None] / 8, factors


def rank_statistics(positive, negative):
    """Ordinary oriented Mann-Whitney AUC; DeLong two-sample influence SE."""
    p, n = np.asarray(positive), np.asarray(negative)
    sp, sn = np.sort(p), np.sort(n)
    vp = (np.searchsorted(sn, p, 'left') + np.searchsorted(sn, p, 'right')) / (2 * len(n))
    vn = (2 * len(p) - np.searchsorted(sp, n, 'left') - np.searchsorted(sp, n, 'right')) / (2 * len(p))
    auc = float(vp.mean())
    se = float(np.sqrt(vp.var(ddof=1) / len(p) + vn.var(ddof=1) / len(n)))
    # Evaluate EVERY attainable threshold, keeping all ties together. The
    # +infinity threshold supplies the zero-positive operating point.
    thresholds = np.unique(np.r_[p, n])
    fp = (len(n) - np.searchsorted(sn, thresholds, 'left')) / len(n)
    tp = (len(p) - np.searchsorted(sp, thresholds, 'left')) / len(p)
    return {'auc': auc, 'se': se, 'ci95': [max(0., auc - 1.96 * se), min(1., auc + 1.96 * se)],
            'tpr_at_fpr01': float(tp[fp <= .01].max(initial=0)),
            'tpr_at_fpr05': float(tp[fp <= .05].max(initial=0))}


def summary(values):
    return {'mean': float(np.mean(values)), 'se': float(np.std(values, ddof=1) / np.sqrt(len(values)))}


def cached_data(original):
    from datasets import Dataset
    cache = Path.home() / '.cache/huggingface/datasets/zalando-datasets___fashion_mnist'
    source = next(cache.rglob('fashion_mnist-train.arrow'))
    data = Dataset.from_file(str(source))
    labels = np.asarray(data['label'])

    def features(indices):
        images = np.stack([np.asarray(data[int(i)]['image'], dtype=float) for i in indices]) / 255
        pooled = images.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4)).reshape(-1, 16) - .5
        return np.column_stack((pooled, np.ones(len(indices))))

    return source, labels, features


def decode_noiseless(total, config, reference, hessian, original, public):
    """Reconstruct saved noiseless diagnostics only; never select on reserve."""
    if config['kind'] != 'legacy':
        _, encoder, decoder = geometry(hessian, config['dimension'], config['metric'])
        offset = reference.copy()
        if config['kind'] == 'absolute':
            offset -= config['gain'] * (reference.ravel() @ encoder @ decoder).reshape(3, 17)
        return offset + config['gain'] * (total @ decoder).reshape(3, 17)
    legacy = config['legacy_configuration']
    family, ridge, k = legacy['family'], legacy['ridge'], legacy['prototypes']
    if family == 'model_direct':
        gram = public.T @ public / len(public)
        denoiser = np.eye(17) if ridge == 0 else np.linalg.solve(gram + ridge * np.eye(17), gram)
        old = total.reshape(3, 17) @ denoiser
    else:
        anchor = original['anchors' + str(k)]
        if family == 'model_distribution':
            targets = softmax(anchor @ total.reshape(3, 17).T @ C.T)
        else:
            latent = total.reshape(k, 3) * np.sqrt(k)
            if family == 'logits':
                targets = softmax(latent @ C.T)
            else:
                raw = .25 + latent @ C.T
                ordered = np.sort(raw, axis=1)[:, ::-1]
                cumulative = np.cumsum(ordered, axis=1) - 1
                count = (ordered - cumulative / np.arange(1, 5) > 0).sum(axis=1)
                threshold = cumulative[np.arange(k), count - 1] / count
                targets = np.maximum(raw - threshold[:, None], 0)
        gram = anchor.T @ anchor / k
        decoder = np.linalg.pinv(gram + ridge * np.eye(17), rcond=1e-12) @ anchor.T / k
        old = (decoder @ (np.log(np.maximum(targets, .01)) @ C)).T
    return reference + config['gain'] * (old - reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-only', action='store_true')
    parser.add_argument('--skip-image-reconstruction', action='store_true')
    parser.add_argument('--require-public-control', action='store_true',
                        help='Require the pre-reserve supplement, including evaluation for a full audit.')
    args = parser.parse_args()
    required = [ROOT / ('public_residual_development' + suffix) for suffix in ('.json', '.npz')]
    if not args.development_only:
        required += [ROOT / ('public_residual_evaluation' + suffix) for suffix in ('.json', '.npz')]
    missing = [str(p) for p in required if not p.exists()]
    if args.require_public_control:
        supplement_required = [PUBLIC_CONTROL_DEV]
        if not args.development_only:
            supplement_required.append(PUBLIC_CONTROL_EVAL)
        missing += [str(p) for p in supplement_required if not p.exists()]
    if missing:
        raise SystemExit('No receipt written: missing ' + ', '.join(missing))
    check = Checks()
    development = json.loads(required[0].read_text())
    bank = np.load(required[1])
    original = np.load(ROOT / 'private_descriptor_analytic_gaussian.npz')
    teachers = np.load(ROOT / 'private_prior_constructor_development.npz')
    preceding = json.loads((ROOT / 'matched_cia_development.json').read_text())
    assert sha(PROTOCOL) == development['protocol_sha']
    assert sha(ROOT / 'matched_cia_development.json') == development['preceding_development_sha']
    assert sha(ROOT / 'private_prior_constructor_development.npz') == development['teacher_archive_sha']
    assert sha(SIMULATOR) == development['calculator_sha']
    assert set(development['dependency_sha']) == {str(p) for p in DEPENDENCIES}
    for dependency, expected_sha in development['dependency_sha'].items():
        assert sha(dependency) == expected_sha, dependency
    assert len(development['rows']) == 6 and development['draws_per_stratum'] == 32
    assert development['public_role_count'] == 512
    assert len(development['public_choices']) == 18
    assert {(c['steps'], c['temperature']) for c in development['public_choices']} == {
        (n, t) for n in (80, 320, 1280) for t in (.5, .75, 1., 1.25, 1.5, 2.)}
    assert development['public_choice'] == min(development['public_choices'], key=lambda c: c['ce'])
    reference, hessian = bank['reference'], bank['public_hessian']
    check.equal(reference, development['public_choice']['reference'], 'public reference')
    check.equal(hessian, hessian.T, 'H symmetry')
    assert np.linalg.eigvalsh(hessian).min() >= -1e-12
    source_sha = None
    image_checks = not args.skip_image_reconstruction
    if image_checks:
        source, labels, feature = cached_data(original)
        source_sha = sha(source)
        assert source_sha == development['source_hash']
        public = feature(original['public_original_indices'])
        dev_ids = original['development_original_indices']
        dev_x, dev_y = feature(dev_ids), labels[dev_ids]
        for steps in (80, 320, 1280):
            theta = train(public, labels[original['public_original_indices']], steps)
            for choice in (c for c in development['public_choices'] if c['steps'] == steps):
                value = theta * choice['temperature']
                check.equal(choice['reference'], value, 'public trained reference')
                check.equal({k: choice[k] for k in ('ce', 'accuracy')}, metrics(dev_x, dev_y, value), 'public dev scores')
        p = softmax(public @ (C @ reference).T)
        covariance = -p[:, :, None] * p[:, None, :]
        covariance[:, np.arange(4), np.arange(4)] += p
        contrast_cov = np.einsum('ca,ncd,db->nab', C, covariance, C)
        rebuilt_h = np.einsum('nab,ni,nj->aibj', contrast_cov, public, public, optimize=True).reshape(51, 51) / len(public)
        check.equal(hessian, rebuilt_h, 'public Hessian')
        private_ids = original['private_original_indices']
        private_x, private_y = feature(private_ids), labels[private_ids]
        for seed in (42, 43, 44):
            fit = fit_indices(private_y, seed)
            counts = np.stack([np.bincount(private_y[i], minlength=4) for i in fit])
            check.equal(teachers[f'seed{seed}_private_class_counts'], counts, 'private split counts')
            for dimension in (1, 3, 12, 51):
                basis, _, _ = geometry(hessian, dimension, 'euclidean')
                check.equal(basis.T @ basis, np.eye(dimension), 'orthonormal basis')
                for transform in ('raw', 'balanced_loss'):
                    for steps in (20, 80):
                        rebuilt = np.stack([train(private_x[i], private_y[i], steps, reference, basis,
                                                  transform == 'balanced_loss') - reference for i in fit])
                        key = f'seed{seed}_fine_d{dimension}_{transform}_steps{steps}'
                        check.equal(bank[key], rebuilt, key)
                        projected = (bank[key].reshape(8, 51) @ basis) @ basis.T
                        check.equal(bank[key].reshape(8, 51), projected, 'constrained fine teacher')
    selection = {}
    for row in development['rows']:
        seed, risk = row['seed'], row['risk']
        assert (seed, risk) not in selection
        selection[seed, risk] = row
        assert len(row['candidates']) == 2793 and set(row['selected']) == set(GROUPS)
        old = next(r for r in preceding['rows'] if r['seed'] == seed and r['risk'] == risk and r['law'] == 'gaussian')
        counts = {g: 0 for g in GROUPS}
        configs = set()
        query_cache = {}
        for candidate in row['candidates']:
            config, group = candidate['configuration'], candidate['group']
            assert config['cap'] in CAPS and config['gain'] in GAINS
            token = json.dumps(config, sort_keys=True)
            assert token not in configs
            configs.add(token)
            counts[group] += 1
            values = bank[candidate['archive_key'] + '_ce']
            assert values.shape == (8, 32)
            check.equal(candidate['development_ce'], values.mean(), 'development mean')
            if config['kind'] == 'legacy':
                prior = old['selected'][group.removeprefix('legacy_')]
                assert config['legacy_configuration'] == prior['configuration'] and config['transform'] == prior['transform']
            else:
                assert config['dimension'] in (1, 3, 12, 51) and config['metric'] in ('euclidean', 'fisher')
                assert config['steps'] in (20, 80)
                assert config['transform'] in (('raw', 'balanced_loss') if group == 'fine' else ('raw', 'prior', 'balanced_loss'))
            base_config = dict(config)
            base_config.pop('gain')
            cache_key = json.dumps(base_config, sort_keys=True)
            if cache_key not in query_cache:
                query_cache[cache_key] = signal(config, seed, reference, hessian, bank, teachers, original)
            bounded, factors = query_cache[cache_key]
            assert bounded.shape == (8, config['dimension'])
            norm = np.linalg.norm(bounded, axis=1)
            assert norm.max() <= config['cap'] / 8 + 1e-12
            check.equal(candidate['clip_factors'], factors, 'clip factors')
            check.equal(candidate['max_shift_norm'], norm.max(), 'candidate norm')
            check.equal(candidate['sigma'], config['cap'] / (8 * np.sqrt(2) * ndtri(risk)), 'public sigma')
        assert counts == dict(absolute=1008, centered=1008, fine=672, **{g: 21 for g in GROUPS if g.startswith('legacy_')})
        for group, selected in row['selected'].items():
            assert selected == min((c for c in row['candidates'] if c['group'] == group), key=lambda c: c['development_ce'])
            rebuilt, _ = signal(selected['configuration'], seed, reference, hessian, bank, teachers, original)
            check.equal(bank[f'seed{seed}_q{int(risk*100)}_{group}_signal'], rebuilt, 'frozen selected signal')
        alternative = min((c for g, c in row['selected'].items() if g != 'fine'), key=lambda c: c['development_ce'])
        name = 'public_only' if development['public_choice']['ce'] <= alternative['development_ce'] else alternative['group']
        assert row['strongest_alternative'] == name
    assert set(selection) == {(s, q) for s in (42, 43, 44) for q in (.55, .65)}
    # This supplement was motivated by development output. Reconstruct only
    # its frozen finite grid here; it never changes any original selection.
    public_control = None
    if PUBLIC_CONTROL_DEV.exists():
        public_control = json.loads(PUBLIC_CONTROL_DEV.read_text())
        assert sha(required[0]) == public_control['development_sha']
        assert sha(required[1]) == public_control['development_archive_sha']
        assert sha(PUBLIC_CONTROL_CODE) == public_control['calculator_sha']
        assert sha(PUBLIC_CONTROL_PROTOCOL) == public_control['addendum_sha']
        assert public_control['source_sha'] == development['source_hash']
        models = {'reference': reference}
        for dimension in (1, 3, 12, 51):
            basis, _, _ = geometry(hessian, dimension, 'euclidean')
            projected = (reference.ravel() @ basis @ basis.T).reshape(3, 17)
            for gain in GAINS:
                models[f'projection_offset_d{dimension}_gain{gain}'] = reference - gain * projected
        for gain in GAINS:
            models[f'legacy_offset_gain{gain}'] = (1 - gain) * reference
        records = public_control['records']
        assert len(records) == 16 and {r['name'] for r in records} == set(models)
        for record in records:
            model = models[record['name']]
            check.equal(record['model'], model, 'public supplemental model')
            if image_checks:
                # Full-dimensional subtraction can leave roundoff-size logits.
                # Verify the reconstruction above, then score the actual saved
                # floats: argmax at an approximately zero model is discontinuous.
                calculated = metrics(dev_x, dev_y, np.asarray(record['model']))
                check.equal(record['development_ce'], calculated['ce'], 'supplement development CE')
                check.equal(record['development_accuracy'], calculated['accuracy'], 'supplement development accuracy')
        assert public_control['selected'] == min(records, key=lambda r: r['development_ce'])
    if args.development_only:
        print(json.dumps({'stage': 'development-only; no completed receipt written', 'numeric_checks': check.count,
                          'maximum_absolute_error': check.maximum, 'image_reconstruction': image_checks,
                          'public_control_development_records_verified': 16 if public_control else 0}))
        return
    evaluation = json.loads(required[2].read_text())
    archive = np.load(required[3])
    assert sha(required[0]) == evaluation['development_sha']
    assert sha(required[1]) == evaluation['development_archive_sha']
    assert sha(SIMULATOR) == evaluation['calculator_sha']
    assert sha(ROOT / 'matched_cia_evaluation.npz') == evaluation['preceding_evaluation_archive_sha']
    assert evaluation['source_hash'] == development['source_hash']
    assert len(evaluation['rows']) == 432
    prior = np.load(ROOT / 'matched_cia_evaluation.npz')
    new = archive['utility_original_indices']
    remaining = archive['remaining_reserve_original_indices']
    previous = prior['remaining_reserve_original_indices']
    assert len(new) == len(set(new)) == evaluation['utility_count'] == 512
    assert len(remaining) == len(set(remaining)) == evaluation['remaining_reserve_count'] == 456
    assert len(previous) == 968 and not set(new) & set(remaining)
    assert set(new) | set(remaining) == set(previous)
    for role in ('public', 'private', 'development', 'evaluation'):
        assert not set(new) & set(original[role + '_original_indices'])
    for filename, key in (('private_prior_constructor_evaluation.npz', 'confirmation_original_indices'),
                          ('descriptor_cia_probe.npz', 'utility_original_indices'),
                          ('matched_cia_evaluation.npz', 'utility_original_indices')):
        earlier = np.load(ROOT / filename)
        assert not set(new) & set(earlier[key])
    ancient = np.load(ROOT / 'client_geometry_audit_matrices.npz')
    for seed in (42, 43, 44):
        for role in ('pool', 'development'):
            assert not set(new) & set(ancient[f'seed{seed}_{role}_original_indices'])
    if image_checks:
        expected_ids = np.concatenate([previous[labels[previous] == klass][:128] for klass in range(4)])
        assert np.array_equal(new, expected_ids)
        check.equal(np.bincount(labels[new], minlength=4), [128] * 4, 'fresh class balance')
        utility_x, utility_y = feature(new), labels[new]
        check.equal(evaluation['public_only'], metrics(utility_x, utility_y, reference), 'public-only reserve utility')
    observed = set()
    gaussian_deviations = []
    score_moment_deviations = []
    score_variance_deviations = []
    for row in evaluation['rows']:
        seed, risk, arm, target = row['seed'], row['risk'], row['arm'], row['target']
        identifier = (seed, risk, arm, target)
        assert identifier not in observed
        observed.add(identifier)
        assert target in range(8) and row['peer'] == (6 if target == 7 else 7) and target != row['peer']
        selected = selection[seed, risk]['selected']['fine' if arm == 'central_fine' else arm]
        config = selected['configuration']
        assert row['configuration'] == config
        key = f'seed{seed}_q{int(risk*100)}_{arm}_target{target}'
        bounded = bank[f'seed{seed}_q{int(risk*100)}_{selected["group"]}_signal']
        norm = float(np.linalg.norm(bounded[target]))
        sigma = config['cap'] / (8 * np.sqrt(2) * ndtri(risk))
        check.equal(row['sigma'], sigma, 'evaluation sigma')
        check.equal(row['public_weighted_cap'], config['cap'] / 8, 'weighted cap')
        check.equal(row['target_shift_norm'], norm, 'target shift')
        expected_auc = float(ndtr(norm / (np.sqrt(2) * sigma)))
        check.equal(row['gaussian_expected_auc'], expected_auc, 'Gaussian exact AUC')
        assert expected_auc <= risk + 1e-12
        attacks = [archive[key + f'_world{world}_attack_score'] for world in (0, 1)]
        assert all(v.shape == (4096,) for v in attacks)
        stats = rank_statistics(attacks[1], attacks[0])
        check.equal(row['attack'], stats, 'ordinary ROC statistics')
        # Monte Carlo departures are reported, never demanded to equal zero.
        if stats['se']:
            gaussian_deviations.append(abs(stats['auc'] - expected_auc) / stats['se'])
        else:
            assert abs(stats['auc'] - expected_auc) <= 1e-12
            gaussian_deviations.append(0.)
        variance = (norm / sigma) ** 2
        for world in (0, 1):
            if variance:
                expected_mean = (world - .5) * variance
                score_moment_deviations.append(abs(attacks[world].mean() - expected_mean) / np.sqrt(variance / 4096))
                score_variance_deviations.append(abs(attacks[world].var(ddof=1) - variance)
                                                 / (variance * np.sqrt(2 / 4095)))
            else:
                check.equal(attacks[world], np.zeros(4096), 'zero shift attack')
            for field in ('ce', 'accuracy'):
                v = archive[key + f'_world{world}_utility_{field}']
                assert v.shape == (256,)
                assert np.all(v >= 0) and (field != 'accuracy' or np.all(v <= 1))
                check.equal(row['utility'][str(world)][field], summary(v), 'utility moments')
            if image_checks:
                active = bounded.copy()
                if world == 0:
                    active[target] = 0
                theta = decode_noiseless(active.sum(axis=0), config, reference, hessian, original, public)
                check.equal(row['noiseless_diagnostic'][str(world)], metrics(utility_x, utility_y, theta), 'noiseless decoding')
    assert observed == {(s, q, a, t) for s in (42, 43, 44) for q in (.55, .65)
                        for a in (*GROUPS, 'central_fine') for t in range(8)}
    contrasts = []
    for seed, risk in sorted(selection):
        for reference_arm in ('public_only', 'absolute', 'centered', *GROUPS[3:], 'central_fine'):
            contrast = {'seed': seed, 'risk': risk, 'candidate': 'fine', 'reference_arm': reference_arm,
                        'primary_targets': list(range(4)), 'strata': 8, 'draws_per_stratum': 256}
            for field in ('ce', 'accuracy'):
                strata = []
                for target in range(4):
                    candidate_key = f'seed{seed}_q{int(risk*100)}_fine_target{target}'
                    other_key = f'seed{seed}_q{int(risk*100)}_{reference_arm}_target{target}'
                    for world in (0, 1):
                        candidate_values = archive[candidate_key + f'_world{world}_utility_{field}']
                        reference_values = evaluation['public_only'][field] if reference_arm == 'public_only' else archive[other_key + f'_world{world}_utility_{field}']
                        # Positive means candidate improvement for both metrics.
                        strata.append(reference_values - candidate_values if field == 'ce' else candidate_values - reference_values)
                gain = float(np.mean([v.mean() for v in strata]))
                se = float(np.sqrt(sum(v.var(ddof=1) / len(v) for v in strata)) / 8)
                contrast[field] = {'gain': gain, 'se': se, 'ci95': [gain - 1.96 * se, gain + 1.96 * se]}
            contrast['ce_lower_ci_exceeds_001'] = contrast['ce']['ci95'][0] > .001
            contrasts.append(contrast)
    supplement_contrasts = 0
    supplementary_files = []
    if PUBLIC_CONTROL_EVAL.exists():
        assert public_control is not None, 'Supplement evaluation requires its frozen development file'
        supplement = json.loads(PUBLIC_CONTROL_EVAL.read_text())
        assert sha(PUBLIC_CONTROL_DEV) == supplement['freeze_sha']
        assert sha(required[3]) == supplement['evaluation_archive_sha']
        assert supplement['selected_name'] == public_control['selected']['name']
        assert supplement['utility_count'] == 512 and supplement['public_only']['auc'] == .5
        if image_checks:
            calculated = metrics(utility_x, utility_y, np.asarray(public_control['selected']['model']))
            check.equal({k: supplement['public_only'][k] for k in ('ce', 'accuracy')}, calculated,
                        'supplement frozen reserve utility')
        assert len(supplement['contrasts']) == 54
        identifiers = set()
        for row in supplement['contrasts']:
            seed, risk, arm = row['seed'], row['risk'], row['arm']
            assert (seed, risk, arm) not in identifiers
            identifiers.add((seed, risk, arm))
            strata = []
            for target in range(4):
                key = f'seed{seed}_q{int(risk*100)}_{arm}_target{target}'
                for world in (0, 1):
                    strata.append(supplement['public_only']['ce'] - archive[key + f'_world{world}_utility_ce'])
            gain = float(np.mean([v.mean() for v in strata]))
            se = float(np.sqrt(sum(v.var(ddof=1) / len(v) for v in strata)) / 8)
            check.equal(row['gain_vs_public_offset'], gain, 'supplement contrast gain')
            check.equal(row['se'], se, 'supplement contrast SE')
            check.equal(row['ci95'], [gain - 1.96 * se, gain + 1.96 * se], 'supplement contrast CI')
        assert identifiers == {(s, q, a) for s in (42, 43, 44) for q in (.55, .65)
                               for a in (*GROUPS, 'central_fine')}
        supplement_contrasts = len(identifiers)
        supplementary_files = [PUBLIC_CONTROL_DEV, PUBLIC_CONTROL_EVAL, PUBLIC_CONTROL_CODE, PUBLIC_CONTROL_PROTOCOL]
    receipt = {'development_rows': 6, 'development_configurations': 16758,
               'public_reference_choices': 18, 'frozen_group_choices': 48,
               'evaluation_rows': 432, 'arms': 54, 'contrast_rows': len(contrasts),
               'numeric_checks': check.count, 'maximum_absolute_error': check.maximum,
               'gaussian_expected_auc_checks': len(gaussian_deviations),
               'maximum_gaussian_auc_deviation_in_standard_errors': max(gaussian_deviations),
               'maximum_gaussian_score_mean_deviation_in_standard_errors': max(score_moment_deviations, default=0),
               'maximum_gaussian_score_variance_deviation_in_standard_errors': max(score_variance_deviations, default=0),
               'fresh_utility_count': 512, 'remaining_reserve_count': 456,
               'indices_and_frozen_choices_verified': True, 'image_teacher_hessian_reconstruction': image_checks,
               'public_control_development_records_verified': 16 if public_control else 0,
               'public_control_evaluation_contrasts_verified': supplement_contrasts,
               'limitations': ['Fixed-target noise intervals are not population or selection-adjusted intervals.',
                               'Raw artifacts and offline research selection are outside end-to-end privacy.',
                               'No statistical equality requirement is imposed on finite attack draws.'],
               'hashes': {str(p): sha(p) for p in (*required, PROTOCOL, SIMULATOR, AUDITOR,
                                                  *DEPENDENCIES, *supplementary_files)}}
    if not supplement_contrasts:
        receipt['limitations'].append('Supplementary public-only evaluation not yet available; its54contrasts not checked.')
    if source_sha:
        receipt['cached_source_sha256'] = source_sha
    if not image_checks:
        receipt['limitations'].append('Cached-image source hash, class order, teacher/Hessian and noiseless utility reconstruction skipped.')
    (ROOT / 'public_residual_contrasts.json').write_text(json.dumps({
        'kind': 'Fixed primary targets0-3, equally weighted OUT/IN strata; paired noise variance; public baseline constant.',
        'evaluation_sha': sha(required[2]), 'rows': contrasts}, indent=2, allow_nan=False) + '\n')
    (ROOT / 'public_residual_artifact_audit.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
