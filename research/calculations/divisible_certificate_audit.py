"""Deterministic scalar certificate arithmetic; no learning or privacy sampler."""
from __future__ import annotations
import json
import math
from pathlib import Path


def variance(epsilon: float, m: int, smooth_epsilon: float) -> float:
    """Sensitivity-one smoothed MSDLap variance, exact real-arithmetic formula."""
    assert m >= 1 and 0 < smooth_epsilon < epsilon
    discrete = epsilon - smooth_epsilon
    denominator = 2 * math.sinh(discrete / 2) ** 2
    return (m * (m + 1) * (2 * m + 1) / (6 * denominator)
            + 1 / (2 * smooth_epsilon ** 2)) / m ** 2


def main() -> None:
    scalars = []
    for epsilon in (2., 4., 8., 16.):
        m = math.ceil(math.exp(epsilon / 3))
        v = variance(epsilon, m, 1.)
        lap = 2 / epsilon ** 2
        scalars.append({'epsilon': epsilon, 'integer_sensitivity': m,
                        'smoother_epsilon': 1., 'msdlap_variance': v,
                        'laplace_variance': lap, 'variance_ratio': v / lap,
                        'eight_slot_peer_variance': v * 8 / 7,
                        'gdl_substitution_prerequisite': epsilon - 1 > 2 + math.log(m)})
    cases = []
    for total in (4., 8., 16.):
        for dimensions in (1, 3, 51):
            for rounds in (1, 2, 20):
                coordinate = total / (dimensions * rounds)
                m = math.ceil(math.exp(coordinate / 3))
                # Always-valid split extension of the displayed remainder proof.
                smoothing = 1. if coordinate >= 2 else coordinate / 2
                v = variance(coordinate, m, smoothing)
                cases.append({'total_epsilon': total, 'dimensions': dimensions,
                              'rounds': rounds, 'equal_coordinate_epsilon': coordinate,
                              'theorem21_advertised_regime': coordinate >= 2,
                              'smoother_epsilon': smoothing, 'integer_sensitivity': m,
                              'scalar_variance_at_unit_coordinate_sensitivity': v,
                              'laplace_variance_same_coordinate_budget': 2 / coordinate ** 2,
                              'variance_ratio': v / (2 / coordinate ** 2)})
    assert math.isclose(variance(4., 4, 1),
                        (30 / (math.cosh(3) - 1) + .5) / 16)
    assert all(x['msdlap_variance'] > 0 for x in scalars)
    assert not (4.6 - 1 > 2 + math.log(math.ceil(math.exp(4.6 / 3))))
    result = {'kind': 'deterministic certificate arithmetic, not sampled utility or CIA',
              'sensitivity': 1,
              'scalar_reference': scalars, 'equal_coordinate_round_allocation': cases,
              'limitations': ['real-arithmetic proof formula, not implemented DP sampler',
                              'independent coordinate composition is sufficient, not universal optimality',
                              'unit coordinate sensitivity does not encode actual model geometry',
                              'ratios compare pure-DP scalar Laplace, not epsilon-delta Gaussian',
                              'no private distribution construction or FL trajectory'],
              'source': 'FORC2025 Harrison-Manurangsi Theorem16/21; flexible split is own proof extension'}
    path = Path('results/client_specific_noise/divisible_certificate_audit.json')
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for row in scalars:
        print(f"epsilon={row['epsilon']:g}, m={row['integer_sensitivity']}, "
              f"MSDLap/Laplace variance={row['variance_ratio']:.6f}")
    print(f'{len(cases)} allocation cases saved; no simulation performed')


if __name__ == '__main__':
    main()
