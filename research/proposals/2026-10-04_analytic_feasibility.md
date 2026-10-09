# First analytical feasibility check: charge the constructor

2026-10-04. Own closed-form calculation for the [joint-profile proposal](2026-10-04_mechanism_proposal.md). This is not a training experiment, a CIA result or a privacy certification. No result artifacts were created under `results/`.

## Fully specified diagnostic

One release, dimension d=2, C=1, rho=1/2. The unnoised input update is equally likely to be (0.8,0) or (0,0.8). The public bank has isotropic A_0=I and two joint profiles A_1=diag(1,1/4), A_2=diag(1/4,1). The private selector chooses only between A_1 and A_2; binary RR has selection budget xi=2. Its preferred profile preserves the input's axis. With these parameters the proposal's reconstruction-plus-noise score indeed favors that profile.

The endpoint is expected squared Euclidean reconstruction error E||Y-u||² at a common total pure-DP budget E. It is a utility proxy, not predictive accuracy or attack leakage. Fresh noise is centered and independent of the clipped update conditional on the chosen profile, so squared clipping bias and noise trace add without a cross term.

The isotropic profile has no clipping bias and noise trace 24/E². A public fixed shaped profile chooses the wrong axis for half the population; clipping then maps magnitude 0.8 to 0.5, creating squared bias 0.09. Its average risk is 15/E²+0.045. Binary RR picks the wrong profile with probability 1/(1+exp(2)); the update receives budget E-2. Its risk is

    R_private(E) = 15/(E-2)² + 0.09/(1+exp(2)).

These constants follow from Cov(Z)=4(d+1)C²A/eta² in the math review. The comparisons below are exact formulas evaluated numerically, with displayed rounding. E denotes the one-release privacy budget here, not a claim that a many-round run has the same noise.

| Total epsilon | Public isotropic | Public fixed shaped | Private profile selection | Change versus best listed public |
|---|---:|---:|---:|---:|
| 4 | 1.500000 | 0.982500 | 3.760728 | 282.77% worse |
| 8 | 0.375000 | 0.279375 | 0.427395 | 52.98% worse |
| 16 | 0.093750 | 0.103594 | 0.087259 | 6.92% better |

## Stronger public control: optimize the clipping radius

The small epsilon-16 advantage disappears under a simple stronger public control. Keep isotropic noise and choose a common public clipping radius a <= 0.8. In this explicitly known toy population its risk is

    R_iso_tuned(a,E) = (0.8-a)² + 24a²/E².
    a*(E) = 0.8 E²/(E²+24).
    R_iso_tuned(E) = 0.64 * 24/(E²+24).

The optimized radius is public and common to all clients; it does not require a private selector. This is a population-known mathematical control, not tuning on private confirmation data.

| Total epsilon | Optimized public isotropic risk | Private profile-selection risk |
|---|---:|---:|
| 4 | 0.384000 | 3.760728 |
| 8 | 0.174545 | 0.427395 |
| 16 | 0.054857 | 0.087259 |

**The proposed private selector loses at all three budgets against this optimized public control.** This calculation rules out a benefit for this toy and these allocations; it does not rule out all heterogeneous populations, profile banks, objectives or accounted constructors. The next analytic question needs genuinely different client utility geometries and optimized public competitors, rather than repeating this toy as a positive result.

## Interpretation and limits

The construction cost matters enough to reverse the apparent benefit. This particular private selector fails at budgets 4 and 8, and gives only a small proxy benefit at 16. A whole-input epsilon of 16 permits a very large worst-case likelihood ratio; that row does not establish practically strong protection. Do not promote it as a successful CIA defense.

“Best listed public” means these three fixed profiles. It is not the best possible public clipping radius, geometry, law, privacy accountant, estimator or population-specific mechanism. Optimizing the public isotropic radius already erases the small benefit, as shown above. Giving each client an independent public random profile gives the same expected risk as a fixed shaped profile in this symmetric population. An unprivate oracle would have risk 15/E² but cannot be claimed at the stated whole-input budget when its profile choice leaks private orientation.

This example supplies a falsifiable starting point and a negative result under two budgets. It supports continued inexpensive analytical work, not a neural-network sweep. Next vary population heterogeneity, selection allocation and rho, while maintaining optimized public geometry/clipping controls, before considering a complete-vector learning pilot. Multi-round composition, stale profile utility, independent CIA trials and model accuracy remain untested.

## Reproduce arithmetic

Run from the repository root with `uv run python`:

```python
import math
for total in (4, 8, 16):
    isotropic = 24 / total**2
    shaped = 15 / total**2 + 0.045
    private = 15 / (total - 2)**2 + 0.09 / (1 + math.exp(2))
    tuned_public = 0.64 * 24 / (total**2 + 24)
    print(total, isotropic, shaped, private, tuned_public)
```

The displayed formulas and outputs were checked locally on 2026-10-04. No randomness, trained model, fitted distribution or fabricated accuracy/attack measurement is involved.
