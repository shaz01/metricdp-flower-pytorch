"""Independent same-image attribution audit; imports no experiment simulator.

Only the 320 already consumed evaluation image IDs are extracted.  No defense
selection, fresh draws or additional reserve features are constructed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.calculations.audit_class_conditional import (
    ROOT, Checks, cached_data, domain, geometry, offset, sha,
    target_prior, weighted_metrics,
)

PROTOCOL = Path('research/proposals/2026-10-08_class_conditional_attribution_protocol.md')
CALCULATOR = Path('research/calculations/class_conditional_attribution.py')
AUDITOR = Path('research/calculations/audit_class_conditional_attribution.py')
OUTPUT = ROOT / 'class_conditional_attribution.json'
DEVELOPMENT = ROOT / 'class_conditional_development.json'
DEVELOPMENT_BANK = DEVELOPMENT.with_suffix('.npz')
EVALUATION = ROOT / 'class_conditional_evaluation.json'
EVALUATION_BANK = EVALUATION.with_suffix('.npz')


def main():
    check = Checks()
    report = json.loads(OUTPUT.read_text())
    development = json.loads(DEVELOPMENT.read_text())
    evaluation = json.loads(EVALUATION.read_text())
    dbank, ebank = np.load(DEVELOPMENT_BANK), np.load(EVALUATION_BANK)
    expected_hash_paths = (PROTOCOL, CALCULATOR, EVALUATION, EVALUATION_BANK,
                           DEVELOPMENT, DEVELOPMENT_BANK,
                           Path('research/calculations/class_conditional_probe.py'))
    hashes = {Path(p).resolve(): value for p, value in report['hashes'].items()}
    assert set(hashes) == {p.resolve() for p in expected_hash_paths}
    for path in expected_hash_paths:
        assert hashes[path.resolve()] == sha(path), path
    assert development['bank_sha'] == sha(DEVELOPMENT_BANK)
    assert evaluation['development_sha'] == sha(DEVELOPMENT)
    assert evaluation['development_archive_sha'] == sha(DEVELOPMENT_BANK)

    original = np.load(ROOT / 'private_descriptor_analytic_gaussian.npz')
    source, labels, extract = cached_data(original)
    assert sha(source) == report['source_hash'] == evaluation['source_hash']
    ids = ebank['utility_original_indices']
    assert len(ids) == len(set(ids.tolist())) == report['utility_count'] == 320
    assert len(ebank['remaining_reserve_original_indices']) == 136
    assert not set(ids) & set(ebank['remaining_reserve_original_indices'])
    previous = np.load(ROOT / 'public_residual_evaluation.npz')
    assert set(ids) | set(ebank['remaining_reserve_original_indices']) == set(previous['remaining_reserve_original_indices'])
    for role in ('public', 'private', 'development', 'evaluation'):
        assert not set(ids) & set(original[role + '_original_indices'])
    for name, key in (('private_prior_constructor_evaluation', 'confirmation_original_indices'),
                      ('matched_cia_evaluation', 'utility_original_indices'),
                      ('public_residual_evaluation', 'utility_original_indices')):
        assert not set(ids) & set(np.load(ROOT / (name + '.npz'))[key])
    # This is the only image extraction: exactly the consumed 320 IDs.
    x, y = extract(ids), labels[ids]
    assert np.array_equal(np.bincount(y, minlength=4), [80] * 4)
    b, h = dbank['reference'], dbank['public_hessian']
    groups = {}
    for row in evaluation['rows']:
        if row['target'] < 4:
            token = tuple(row[k] for k in ('seed', 'risk', 'scenario', 'reference_source', 'arm'))
            groups.setdefault(token, []).append(row)
    assert len(groups) == len(report['rows']) == 147
    seen = set()
    for record in report['rows']:
        token = tuple(record[k] for k in ('seed', 'risk', 'scenario', 'reference_source', 'arm'))
        assert token in groups and token not in seen
        seen.add(token)
        seed, risk, scenario, route, arm = token
        cells = sorted(groups[token], key=lambda r: r['target'])
        assert [r['target'] for r in cells] == list(range(4))
        c = record['configuration']
        assert all(r['configuration'] == c for r in cells)
        prefix = f'seed{seed}_{scenario}_{route}'
        name = f'{prefix}_q{int(risk*100)}_{arm}'
        _, encoder, decoder = geometry(h, c['dimension'], c['metric'])
        gradient = np.tensordot(target_prior(scenario), dbank[prefix + '_gamma'], axes=1)
        public_offset = offset(c['mode'], gradient, encoder, decoder)
        check.equal(ebank[name + '_offset'], public_offset, 'public offset reconstruction')
        metrics = weighted_metrics(domain(x, scenario), y, b - c['eta'] * public_offset,
                                   target_prior(scenario))
        public_ce = float(metrics['ce'][0])
        check.equal(record['matched_public_offset_ce'], public_ce, 'matched offset CE')
        check.equal(record['matched_public_offset_accuracy'], metrics['accuracy'][0], 'matched offset accuracy')
        q = dbank[prefix + '_' + c['mode'] + '_query'].reshape(8, 51) @ encoder
        factors = np.minimum(1., c['cap'] / np.maximum(np.linalg.norm(q, axis=1), 1e-100))
        signal = q * factors[:, None] / 8
        check.equal(ebank[name + '_signal'], signal, 'bounded selected signal')
        check.equal(record['clip_factors'], factors, 'clip factors')
        strata, noiseless = [], []
        for cell in cells:
            for world in (0, 1):
                a = ebank[cell['archive_key'] + f'_world{world}_utility_ce']
                assert a.shape == (128,)
                strata.append(a)
                active = signal.copy()
                if not world:
                    active[cell['target']] = 0
                theta = b - c['eta'] * (public_offset + (active.sum(axis=0) @ decoder).reshape(3, 17))
                no = weighted_metrics(domain(x, scenario), y, theta, target_prior(scenario))['ce'][0]
                check.equal(cell['noiseless'][str(world)]['ce'], no, 'noiseless stratum CE')
                noiseless.append(no)
        private_ce = float(np.mean([a.mean() for a in strata]))
        se = float(np.sqrt(sum(a.var(ddof=1) / len(a) for a in strata)) / 8)
        gain = public_ce - private_ce
        check.equal(record['protected_ce'], private_ce, 'private CE')
        check.equal(record['protected_gain_over_matched_offset'], gain, 'private gain')
        check.equal(record['gain_se'], se, 'fixed-stratum SE')
        check.equal(record['gain_ci95'], [gain - 1.96*se, gain + 1.96*se], 'gain interval')
        check.equal(record['noiseless_ce'], np.mean(noiseless), 'noiseless mean')
        check.equal(record['noiseless_gain_over_matched_offset'], public_ce - np.mean(noiseless), 'noiseless gain')
        for mode, field in (('raw', 'encoded_raw_client_norms'),
                            ('class_center', 'encoded_class_center_client_norms')):
            vectors = dbank[prefix + '_' + mode + '_query'].reshape(8, 51) @ encoder
            check.equal(record[field], np.linalg.norm(vectors, axis=1), 'encoded norms/' + mode)
    assert seen == set(groups)
    receipt = {'kind': 'Independent post-hoc matched-offset arithmetic; same320images, no extra reserve or selection.',
               'records': 147, 'numeric_checks': check.count, 'maximum_absolute_error': check.maximum,
               'utility_image_count': 320, 'remaining_reserve_count': 136,
               'noiseless_strata_reconstructed': 1176,
               'hashes': {str(p): sha(p) for p in (*expected_hash_paths, OUTPUT, AUDITOR,
                   Path('research/calculations/audit_class_conditional.py'),
                   Path('research/calculations/audit_public_residual_artifacts.py'))}}
    (ROOT / 'class_conditional_attribution_audit.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'hashes'}, indent=2))


if __name__ == '__main__':
    main()
