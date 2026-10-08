"""Independently retrain and audit development-only headroom artifacts.

Imports independent audit helpers only, never the headroom simulator. Uses old
public/private/development roles; no reserve indices or images are accessed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.calculations.audit_class_conditional import examples, public_gamma, queries, offset
from research.calculations.audit_public_residual_artifacts import (
    C, ROOT, Checks, cached_data, geometry, metrics, sha, softmax, train,
)

ARTIFACT = ROOT / 'headroom_development.json'
BANK = ROOT / 'headroom_development.npz'
CODE = Path('research/calculations/headroom_probe.py')
PROTOCOL = Path('research/proposals/2026-10-08_headroom_protocol.md')
AUDITOR = Path('research/calculations/audit_headroom.py')
DEPS = [Path('research/calculations') / (name + '.py') for name in (
    'private_prior_constructor_probe', 'private_descriptor_probe', 'client_geometry_audit',
    'client_energy_filter_probe', 'client_energy_one_release', 'public_residual_probe',
    'public_residual_signal_diagnostic', 'class_conditional_probe')]
DEPS += [ROOT / 'private_descriptor_analytic_gaussian.npz', ROOT / 'class_conditional_development.json']
STEPS = (80, 320, 1280, 5120)
REFINES = (20, 80, 320, 1280)
MULTIPLIERS = (.5, .75, 1., 1.25, 1.5, 2.)
ETAS = (.1, .3, 1., 3., 10., 30.)
DIMS = (1, 3, 12, 51)


def allocation(y):
    """Independent seed42 fixed class-count partition, then stratified halves."""
    counts = np.full((8, 4), 34)
    counts[np.arange(8), np.arange(8) % 4] = 410
    rng = np.random.default_rng(42)
    parts = [[] for _ in range(8)]
    for k in range(4):
        ids = np.flatnonzero(y == k)
        rng.shuffle(ids)
        start = 0
        for i in range(8):
            stop = start + counts[i, k]
            parts[i].extend(ids[start:stop])
            start = stop
    for part in parts:
        rng.shuffle(part)
    rng = np.random.default_rng(42)
    fit, check = [], []
    for part in parts:
        first, second = [], []
        for k in range(4):
            ids = np.asarray([i for i in part if y[i] == k], dtype=int)
            rng.shuffle(ids)
            first.extend(ids[:len(ids) // 2])
            second.extend(ids[len(ids) // 2:])
        fit.append(np.asarray(first, dtype=int))
        check.append(np.asarray(second, dtype=int))
    return {'A': fit, 'B': check}


def subset(public_ids, labels, budget, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for k in range(4):
        ids = public_ids[labels[public_ids] == k].copy()
        rng.shuffle(ids)
        selected.extend(ids[:budget // 4])
    return np.asarray(selected, dtype=int)


def public_hessian(x, b):
    p = softmax(x @ b.T @ C.T)
    variance = -p[:, :, None] * p[:, None, :]
    variance[:, np.arange(4), np.arange(4)] += p
    identifiable = np.einsum('ca,ncd,db->nab', C, variance, C)
    return np.einsum('nab,ni,nj->aibj', identifiable, x, x, optimize=True).reshape(51, 51) / len(x)


def stratified_variance(values, y):
    # Fixed class-count variance of the empirical mean, summed across51 axes.
    total = 0.
    for k in range(4):
        group = values[y == k]
        if len(group) > 1:
            total += len(group) * np.sum((group - group.mean(axis=0)) ** 2) / (len(group) - 1)
    return float(total / len(y) ** 2)


def main():
    missing = [str(p) for p in (ARTIFACT, BANK) if not p.exists()]
    if missing:
        raise SystemExit('No receipt written: missing ' + ', '.join(missing))
    j = json.loads(ARTIFACT.read_text())
    assert j['protocol_sha'] == sha(PROTOCOL) and j['calculator_sha'] == sha(CODE)
    assert j['bank_sha'] == sha(BANK)
    assert set(j['dependencies']) == {str(p) for p in DEPS}
    for path, expected in j['dependencies'].items():
        assert sha(path) == expected, path
    bank = np.load(BANK)
    original = np.load(ROOT / 'private_descriptor_analytic_gaussian.npz')
    source, labels, features = cached_data(original)
    assert j['source_hash'] == sha(source)
    public_ids = original['public_original_indices']
    private_ids = original['private_original_indices']
    dev_ids = original['development_original_indices']
    selection_ids = np.concatenate([dev_ids[labels[dev_ids] == k][:128] for k in range(4)])
    assessment_ids = np.concatenate([dev_ids[labels[dev_ids] == k][128:256] for k in range(4)])
    assert j['selection_count'] == j['assessment_count'] == 512
    assert np.array_equal(selection_ids, bank['selection_original_indices'])
    assert np.array_equal(assessment_ids, bank['assessment_original_indices'])
    assert not set(selection_ids) & set(assessment_ids)
    assert set(selection_ids) | set(assessment_ids) == set(dev_ids)
    assert not (set(selection_ids) | set(assessment_ids)) & (set(public_ids) | set(private_ids))
    sx, sy = features(selection_ids), labels[selection_ids]
    ax, ay = features(assessment_ids), labels[assessment_ids]
    private, y = features(private_ids), labels[private_ids]
    cohorts = allocation(y)
    assert j['unique_private_cohort_count'] == 2
    for name, parts in cohorts.items():
        allocation_ids = private_ids[np.stack(parts)]
        assert allocation_ids.shape == (8, 256)
        assert np.array_equal(allocation_ids, bank['cohort' + name + '_original_indices'])
        for slot, part in enumerate(parts):
            counts = np.full(4, 17)
            counts[slot % 4] = 205
            assert np.array_equal(np.bincount(y[part], minlength=4), counts)
    a_set = set(np.concatenate(cohorts['A']))
    b_set = set(np.concatenate(cohorts['B']))
    assert len(a_set) == len(b_set) == 2048 and not a_set & b_set
    assert a_set | b_set == set(range(4096))
    assert len(j['public_rows']) == 9 and len(j['rows']) == 18 and j['model_count'] == 2775
    check = Checks()
    model_keys = set()

    def audit_record(record, expected):
        key = record['archive_key']
        assert key not in model_keys, key
        model_keys.add(key)
        saved = bank[key]
        check.equal(saved, expected, 'independently fitted model', tolerance=1e-10)
        for prefix, x, yy in (('selection', sx, sy), ('assessment', ax, ay)):
            score = metrics(x, yy, saved)
            for field in ('ce', 'accuracy'):
                check.equal(record[prefix + '_' + field], score[field], prefix + ' model score')

    public_map = {}
    public_subsets = {}
    for row in j['public_rows']:
        budget, seed = row['budget'], row['subset_seed']
        assert (budget, seed) not in public_map
        public_map[budget, seed] = row
        prefix = f'budget{budget}_seed{seed}'
        selected_ids = subset(public_ids, labels, budget, seed)
        public_subsets[budget, seed] = selected_ids
        assert np.array_equal(bank[prefix + '_public_original_indices'], selected_ids)
        assert len(selected_ids) == len(set(selected_ids)) == budget
        assert set(selected_ids) <= set(public_ids) and not set(selected_ids) & set(private_ids)
        assert np.array_equal(np.bincount(labels[selected_ids], minlength=4), np.full(4, budget // 4))
        px, py = features(selected_ids), labels[selected_ids]
        candidates = row['candidates']
        assert len(candidates) == (53 if budget == 512 else 52)
        scratch = [r for r in candidates if r['family'] == 'public_scratch']
        assert len(scratch) == 24
        assert {(r['configuration']['steps'], r['configuration']['multiplier']) for r in scratch} == {(n, m) for n in STEPS for m in MULTIPLIERS}
        scratch_models = {n: train(px, py, n) for n in STEPS}
        for record in scratch:
            c = record['configuration']
            audit_record(record, scratch_models[c['steps']] * c['multiplier'])
        assert row['reference'] == min(scratch, key=lambda r: r['selection_ce'])
        b = bank[row['reference']['archive_key']]
        check.equal(bank[prefix + '_reference'], b, 'selected scratch reference')
        gamma = public_gamma(px, py, b)
        h = public_hessian(px, b)
        check.equal(bank[prefix + '_gamma'], gamma, 'public class gradients')
        check.equal(bank[prefix + '_hessian'], h, 'public Hessian')
        h = bank[prefix + '_hessian']
        gamma = bank[prefix + '_gamma']
        public_refines = [r for r in candidates if r['family'] == 'public_refine']
        assert len(public_refines) == 4 and {r['configuration']['steps'] for r in public_refines} == set(REFINES)
        for record in public_refines:
            audit_record(record, train(px, py, record['configuration']['steps'], initial=b))
        public_steps = [r for r in candidates if r['family'] == 'public_step']
        assert len(public_steps) == 24
        assert {(r['configuration']['dimension'], r['configuration']['eta']) for r in public_steps} == {(dim, eta) for dim in DIMS for eta in ETAS}
        for record in public_steps:
            c = record['configuration']
            basis, _, _ = geometry(h, c['dimension'], 'euclidean')
            gradient = (gamma.mean(axis=0).ravel() @ basis @ basis.T).reshape(3, 17)
            audit_record(record, b - c['eta'] * gradient)
        anchors = [r for r in candidates if r['family'] == 'historical_public_anchor']
        assert len(anchors) == (1 if budget == 512 else 0)
        for record in anchors:
            historical = json.loads((ROOT / 'class_conditional_development.json').read_text())['public_controls'][0]['selected']['theta']
            audit_record(record, np.asarray(historical))
        assert row['selected'] == min(candidates, key=lambda r: r['selection_ce'])
    assert set(public_map) == {(b, s) for b in (32, 128, 512) for s in (42, 43, 44)}
    for seed in (42, 43, 44):
        assert set(public_subsets[32, seed]) <= set(public_subsets[128, seed]) <= set(public_subsets[512, seed])
        assert set(public_subsets[512, seed]) == set(public_ids)
    summaries = []
    cells = set()
    for row in j['rows']:
        budget, seed, name = row['budget'], row['subset_seed'], row['cohort']
        assert (budget, seed, name) not in cells
        cells.add((budget, seed, name))
        prefix = f'budget{budget}_seed{seed}'
        cp = prefix + '_cohort' + name
        public_row = public_map[budget, seed]
        assert row['public_selected'] == public_row['selected']
        b, h, gamma = bank[prefix + '_reference'], bank[prefix + '_hessian'], bank[prefix + '_gamma']
        public_subset = public_subsets[budget, seed]
        px, py = features(public_subset), labels[public_subset]
        parts = cohorts[name]
        pooled = np.concatenate(parts)
        xp, yp = np.concatenate((px, private[pooled])), np.concatenate((py, y[pooled]))
        q, priors, _ = queries(private, y, parts, b, gamma, np.full(4, .25))
        check.equal(bank[cp + '_priors'], priors, 'internal client priors')
        for mode, value in q.items():
            check.equal(bank[cp + '_' + mode], value, 'unclipped full gradient queries')
        check.equal(gamma.mean(axis=0) + q['class_center'].mean(axis=0), q['raw'].mean(axis=0), 'unbounded all-IN identity', tolerance=1e-14)
        assessment_gradient = examples(ax, ay, b).mean(axis=0)
        assert len(row['client_residual_diagnostics']) == 8
        for slot, indices in enumerate(parts):
            yi = y[indices]
            raw = examples(private[indices], yi, b)
            z = raw - gamma[yi]
            r = z.mean(axis=0)
            iid_var = float(np.sum((z - r) ** 2) / (len(z) * (len(z) - 1)))
            stratum_var = stratified_variance(z, yi)
            raw_var = stratified_variance(raw, yi)
            check.equal(stratum_var, raw_var, 'class-constant centering preserves stratified variance', tolerance=1e-14)
            record = row['client_residual_diagnostics'][slot]
            assert record['slot'] == slot
            check.equal({key: record[key] for key in ('residual_norm_squared', 'iid_mean_variance', 'stratified_mean_variance', 'raw_stratified_mean_variance', 'assessment_descent_inner_product')},
                        {'residual_norm_squared': float(np.sum(r * r)), 'iid_mean_variance': iid_var,
                         'stratified_mean_variance': stratum_var, 'raw_stratified_mean_variance': raw_var,
                         'assessment_descent_inner_product': float(np.sum(assessment_gradient * r))}, 'residual diagnostics')
        candidates = row['candidates']
        assert len(candidates) == 128
        scratch = [r for r in candidates if r['family'] == 'pooled_scratch']
        assert len(scratch) == 24
        assert {(r['configuration']['steps'], r['configuration']['multiplier']) for r in scratch} == {(n, m) for n in STEPS for m in MULTIPLIERS}
        scratch_models = {n: train(xp, yp, n) for n in STEPS}
        for record in scratch:
            c = record['configuration']
            audit_record(record, scratch_models[c['steps']] * c['multiplier'])
        for family, x, yy in (('pooled_refine', xp, yp), ('private_refine', private[pooled], y[pooled])):
            records = [r for r in candidates if r['family'] == family]
            assert len(records) == 4 and {r['configuration']['steps'] for r in records} == set(REFINES)
            for record in records:
                audit_record(record, train(x, yy, record['configuration']['steps'], initial=b))
        for mode in ('raw', 'class_center', 'center_only', 'target_balanced'):
            records = [r for r in candidates if r['family'] == 'step_' + mode]
            assert len(records) == 24
            assert {(r['configuration']['dimension'], r['configuration']['eta']) for r in records} == {(dim, eta) for dim in DIMS for eta in ETAS}
            for record in records:
                c = record['configuration']
                _, encoder, decoder = geometry(h, c['dimension'], 'euclidean')
                gradient = offset(mode, gamma.mean(axis=0), encoder, decoder) + (q[mode].mean(axis=0).ravel() @ encoder @ decoder).reshape(3, 17)
                audit_record(record, b - c['eta'] * gradient)
        families = {r['family'] for r in candidates}
        assert len(families) == 7 and set(row['selected']) == families
        for family in families:
            assert row['selected'][family] == min((r for r in candidates if r['family'] == family), key=lambda r: r['selection_ce'])
        assert row['selected_oracle'] == min((r for family, r in row['selected'].items() if not family.startswith('step_')), key=lambda r: r['selection_ce'])
        assert row['selected_one_step'] == min((r for family, r in row['selected'].items() if family.startswith('step_')), key=lambda r: r['selection_ce'])
        public_ce = row['public_selected']['assessment_ce']
        oracle_gain = public_ce - row['selected_oracle']['assessment_ce']
        step_gain = public_ce - row['selected_one_step']['assessment_ce']
        check.equal(row['assessment_oracle_gain'], oracle_gain, 'oracle assessment gain')
        check.equal(row['assessment_one_step_gain'], step_gain, 'one-step assessment gain')
        aggregate = q['class_center'].mean(axis=0).ravel()
        norm_squared = float(aggregate @ aggregate)
        fractions = {}
        for dimension in DIMS:
            basis, _, _ = geometry(h, dimension, 'euclidean')
            fractions[str(dimension)] = float(np.sum((aggregate @ basis) ** 2) / norm_squared) if norm_squared else 0.
        decomposition = {'aggregate_residual_norm_squared': norm_squared,
                         'aggregate_stratified_mean_variance': float(sum(
                             r['stratified_mean_variance'] for r in row['client_residual_diagnostics']) / 64),
                         'projected_mean_fractions': fractions}
        check.equal(row['residual_decomposition'], decomposition, 'aggregate residual decomposition')
        assert all(-1e-12 <= f <= 1 + 1e-12 for f in fractions.values())
        assert all(fractions[str(a)] <= fractions[str(b)] + 1e-12 for a, b in zip(DIMS[:-1], DIMS[1:]))
        summaries.append({'budget': budget, 'subset_seed': seed, 'cohort': name,
                          'public_family': row['public_selected']['family'], 'oracle_family': row['selected_oracle']['family'],
                          'one_step_family': row['selected_one_step']['family'],
                          'assessment_oracle_gain': oracle_gain, 'assessment_one_step_gain': step_gain,
                          'selection_frozen_family_assessment_gains': {
                              family: public_ce - record['assessment_ce'] for family, record in row['selected'].items()},
                          'aggregate_residual_norm_squared': float(np.sum(q['class_center'].mean(axis=0) ** 2)),
                          'conditional_aggregate_stratified_mean_variance_proxy': float(
                              sum(r['stratified_mean_variance'] for r in row['client_residual_diagnostics']) / 64),
                          'conditional_aggregate_iid_mean_variance_proxy': float(
                              sum(r['iid_mean_variance'] for r in row['client_residual_diagnostics']) / 64),
                          'projected_residual_mean_fractions': fractions})
    assert cells == {(b, s, n) for b in (32, 128, 512) for s in (42, 43, 44) for n in ('A', 'B')}
    assert len(model_keys) == 2775
    assert {k for k in bank.files if k.startswith('model')} == model_keys
    receipt = {'kind': 'Independent reconstruction of reused development-only raw/noiseless headroom; no protected/fresh result.',
               'model_count': len(model_keys), 'public_cells': 9, 'private_cells': 18,
               'public_models': 471, 'private_models': 2304, 'client_variance_records': 144,
               'aggregate_residual_decompositions': 18,
               'numeric_checks': check.count, 'maximum_absolute_error': check.maximum,
               'exact_splits_and_nested_public_subsets_verified': True, 'no_reserve_indices_or_images_read': True,
               'selection_count': 512, 'assessment_count': 512, 'private_cohort_counts': [2048, 2048],
               'all_models_retrained_independently': True, 'rows': summaries,
               'limitations': ['Both development halves were used in previous phases; neither is fresh confirmation.',
                               'Budget512 orderings repeat the same public set, not three independent subsets.',
                               'Historical512 public anchor was previously tuned using both development halves.',
                               'Variance estimates assume IID within client or fixed-class-count strata; no privacy implication.',
                               'Aggregate variance proxies condition on fixed public gamma and assume independent client mean errors.'],
               'hashes': {str(p): sha(p) for p in (ARTIFACT, BANK, CODE, PROTOCOL, AUDITOR,
                    Path('research/calculations/audit_class_conditional.py'),
                    Path('research/calculations/audit_public_residual_artifacts.py'))}}
    (ROOT / 'headroom_artifact_audit.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('rows', 'hashes')}, indent=2))


if __name__ == '__main__':
    main()
