"""Supplementary decoder-only public control, frozen before reserve access."""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from research.calculations.public_residual_probe import digest, public_geometry, DEV_JSON, DEV_NPZ
from research.calculations.private_descriptor_probe import ROOT
from research.calculations.private_prior_constructor_probe import prepare
from research.calculations.client_geometry_audit import features
from research.calculations.client_energy_filter_probe import scores

ADDENDUM = Path('research/proposals/2026-10-08_public_residual_control_addendum.md')
FREEZE = ROOT / 'public_residual_public_control_development.json'


def development():
    bank = np.load(DEV_NPZ)
    source, _, _, _, _, dev, _, _, _, _, _ = prepare(False)
    b, h = bank['reference'], bank['public_hessian']
    models = [('reference', b)]
    for d in (1, 3, 12, 51):
        basis = public_geometry(h, d, 'euclidean')['basis']
        projected = (b.ravel() @ basis @ basis.T).reshape(3, 17)
        for a in (.1, .3, 1.):
            models.append((f'projection_offset_d{d}_gain{a}', b - a * projected))
    for a in (.1, .3, 1.):
        models.append((f'legacy_offset_gain{a}', (1 - a) * b))
    rows = []
    for name, theta in models:
        ce, acc = scores(*dev, theta[None])
        rows.append({'name': name, 'model': theta.tolist(), 'development_ce': float(ce[0]),
                     'development_accuracy': float(acc[0])})
    selected = min(rows, key=lambda r: r['development_ce'])
    FREEZE.write_text(json.dumps({'development_sha': digest(DEV_JSON), 'development_archive_sha': digest(DEV_NPZ),
                      'source_sha': digest(source), 'addendum_sha': digest(ADDENDUM),
                      'calculator_sha': digest(__file__), 'records': rows, 'selected': selected}, indent=2) + '\n')
    print('supplementary public selection', selected['name'], selected['development_ce'], flush=True)


def evaluation():
    freeze = json.loads(FREEZE.read_text())
    assert digest(ADDENDUM) == freeze['addendum_sha'] and digest(__file__) == freeze['calculator_sha']
    assert digest(DEV_JSON) == freeze['development_sha'] and digest(DEV_NPZ) == freeze['development_archive_sha']
    source, _, _, _, _, _, _, _, _, _, _ = prepare(False)
    assert digest(source) == freeze['source_sha']
    data = Dataset.from_file(str(source)); labels = np.array(data['label'])
    archive = np.load(ROOT / 'public_residual_evaluation.npz')
    indices = archive['utility_original_indices']
    theta = np.array(freeze['selected']['model'])
    ce, acc = scores(features(data, indices), labels[indices], theta[None])
    dev = json.loads(DEV_JSON.read_text())
    contrasts = []
    for record in dev['rows']:
        seed, risk = record['seed'], record['risk']
        for arm in (*record['selected'], 'central_fine'):
            strata = [ce[0] - archive[f'seed{seed}_q{int(risk*100)}_{arm}_target{t}_world{w}_utility_ce']
                      for t in range(4) for w in (0, 1)]
            gain = float(np.mean([v.mean() for v in strata]))
            se = float(np.sqrt(sum(v.var(ddof=1) / len(v) for v in strata)) / 8)
            contrasts.append({'seed': seed, 'risk': risk, 'arm': arm, 'gain_vs_public_offset': gain,
                              'se': se, 'ci95': [gain - 1.96 * se, gain + 1.96 * se]})
    out = {'freeze_sha': digest(FREEZE), 'evaluation_archive_sha': digest(ROOT / 'public_residual_evaluation.npz'),
           'utility_count': len(indices), 'selected_name': freeze['selected']['name'],
           'public_only': {'ce': float(ce[0]), 'accuracy': float(acc[0]), 'auc': .5}, 'contrasts': contrasts}
    (ROOT / 'public_residual_public_control_evaluation.json').write_text(json.dumps(out, indent=2) + '\n')
    print('supplementary public confirmation', out['public_only'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('development', 'evaluation'), default='development')
    args = parser.parse_args()
    development() if args.stage == 'development' else evaluation()
