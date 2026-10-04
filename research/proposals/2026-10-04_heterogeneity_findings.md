# Heterogeneity check and the next construction route

2026-10-04. Own deterministic analytical calculations, not a completed FL experiment or CIA result. The owner requested the next step after the initial two-profile toy failed an optimized public clipping control.

## What was actually checked

We optimized the clipping geometry as well as the privacy allocation, instead of keeping the earlier profile radii fixed. Two synthetic population families:

1. **Direction and utility geometry:** equiprobable updates u_1=0.8e_1 and u_2=0.8e_2. Their quadratic utility weights are H_1=diag(1,h), H_2=diag(h,1), with h in {0,0.01,0.1,0.5,1}. Small h makes errors in the client's other direction less costly. h=1 is the earlier common Euclidean objective.
2. **Magnitude:** equiprobable updates 0 and e_1, evaluated using a common squared Euclidean objective. A zero update can describe a nonempty client's stationary update; it does not imply that physical absence is hidden.

For each, compare total one-release whole-input pure-DP budgets E in {1,2,4,8,16}. Both public and privately selected profiles use valid bounded clipping plus fresh K-norm noise. In addition to ellipsoids, include **diamond clipping bodies** with their own matching non-Gaussian noise law. For these axis-aligned inputs, a diamond provides the same clipping behavior with smaller noise variance.

Amplitude profiles are restricted to isotropic radii; the directional family optimizes both aligned semiaxes. An additional public scalar-projection control is included for the amplitude population. At each fixed binary-RR selection budget xi, the two clipping radii/semiaxes have closed-form risk minima. Search xi on uniform grids of 4096 and 8192 intervals in [0,E), including xi=0. At zero selection budget the two optimal profiles coincide, so the method reduces to the optimized public reference. These are bounded grid comparisons, not certified global continuous optimization.

## Derivations and fair controls

For an ellipsoid with semiaxes a,b, the two-dimensional noise covariance is 12 diag(a²,b²)/eta². For the diamond {|z_1|/a+|z_2|/b <= 1}, it is 8 diag(a²,b²)/eta². Both use density proportional to exp[-eta ||z||_B/2], and both protect arbitrary whole-client input replacement after clipping into B. These moment and risk calculations are our own adaptation of the existing [K-norm mechanism](https://arxiv.org/pdf/0907.3754); the binary selector is existing [randomized response](https://proceedings.mlr.press/v48/kairouz16.pdf), not a new privacy primitive.

Write c=12 for ellipsoids or c=8 for diamonds. In the direction/curvature population the optimum public aligned profile is isotropic within its body family:

    a=b=0.8 E²/[E²+c(1+h)],
    R_public=0.64 c(1+h)/[E²+c(1+h)].

With p=exp(xi)/(1+exp(xi)), q=1-p, eta=E-xi, t=c/eta², privately selected swapped profiles have optimum

    L=0.8p/[p+t(p+qh)],
    S=0.8q/[q+t(ph+q)],
    R_private=p(0.8-L)²+q(0.8-S)²+t[(p+qh)L²+(ph+q)S²].

For the magnitude population, public radius a=E²/(E²+4c) gives risk 0.5*4c/(E²+4c). Private large/small radii are p/[p+2c/eta²] and q/[q+2c/eta²], with the corresponding clipping/noise costs in the reproducible calculator. Because this toy has signal only on the first axis, a stronger public control projects onto that axis, clips at a=E²/(E²+16), adds scalar Laplace noise of scale 2a/E, and releases second coordinate zero. Its risk is 8/(E²+16). Projection is fixed and public; the mechanism remains valid for arbitrary inputs, but discards all second-coordinate signal, which this particular toy does not penalize.

Read the [independent mathematical review](2026-10-04_heterogeneity_math_review.md): it verifies these calculations and shows that rotated centered ellipsoids cannot improve the public ellipsoid control for this symmetric toy. It does not prove optimality among all possible distributions, bodies or estimators.

## Observed calculation outputs

**No private-selector configuration beats the strongest listed public control across these 30 population/budget cases on the searched grids.** Within each tested private-law bank, the best option is the xi=0, nonadaptive boundary, except the ellipsoidal magnitude case at E=16. The extra public projection beats both magnitude-case banks. Every law and grid outcome is retained, including unfavorable results.

Selected cases, from the saved artifact:

| Population | Total epsilon | Best public risk | Best searched private risk |
|---|---:|---:|---:|
| Direction/curvature, h=0 | 4 | 0.213333 | 0.213333 |
| Direction/curvature, h=0 | 8 | 0.071111 | 0.071111 |
| Direction/curvature, h=0 | 16 | 0.019394 | 0.019394 |
| Common Euclidean loss, h=1 | 4 | 0.320000 | 0.320000 |
| Common Euclidean loss, h=1 | 8 | 0.128000 | 0.128000 |
| Common Euclidean loss, h=1 | 16 | 0.037647 | 0.037647 |
| Magnitudes zero/one | 4 | 0.250000 | 0.333333 |
| Magnitudes zero/one | 8 | 0.100000 | 0.166667 |
| Magnitudes zero/one | 16 | 0.029412 | 0.055556 |

In the directional rows, equality means selecting the nonadaptive boundary, not a positive-cost private constructor matching its performance. There is a small apparent magnitude-case benefit against **ellipsoids alone** at E=16 (0.078609 versus 0.078947); the public isotropic diamond's 0.055556 removes that claim, and projection improves the public risk further to 0.029412. This is why public law-family controls matter as well as tuning the clipping radius.

The maximum risk change from doubling grid resolution was 5.365e-9. Mathematical checks covered public-boundary formulas, the h=0 identity and finite-difference stationarity of the optimized geometry. No trained model or sampled attack score enters these outputs.

## Research decision

Deprioritize **separately paid binary-RR profile selection** as the first FL implementation candidate. These bounded results do not prove it never helps; the independent review gives asymptotic high-epsilon examples where orientation adaptation can help. They do show that the tested stronger controls leave no reason to spend deep-model compute on this version now.

Client-specific quadratic local reconstruction risk is not global FL predictive utility. Averaging updates and considering the global task may change the objective; the zero mean cross-term of an independent noise draw does not eliminate aggregation/clipping bias. This check does not establish protection against CIA, acceptable privacy strength, or a novel distribution.

## Next candidate: reuse already protected information

Investigate construction from a client's **previously privatized uploads**, rather than paying a separate categorical-selection cost. The concrete reference is:

1. At round 0, upload Y_i,0 using a fixed public clipping/noise mechanism at budget eta_0. This upload participates in learning and also serves as a protected probe.
2. Choose a profile K_i=f(Y_i,0) using a public, fixed function and a public bank. Do not use an additional raw private statistic inside f.
3. At later rounds, clip the new raw local update in B_K_i and draw fresh matching noise. Each fixed conditional profile must protect every client input at every history.
4. Conservatively analyze the hypothetical full upload transcript, including Y_i,0, even when the curious peer sees only global models. Adaptive composition costs sum_t eta_t. There is no extra RR charge for deterministic post-processing of the protected upload, but the probe's eta_0 and its utility/noise cost remain fully counted.

This is **not free private estimation**: the probe can be too noisy to identify useful geometry, and a stale profile can distort later updates. A public baseline must receive the same initial upload, total budget and number of learning steps. Any raw-data-dependent bank fitting or extra local history used to choose a profile reopens the construction proof obligation. Fresh protecting randomness remains essential.

The next bounded task is a two-round, low-dimensional comparison with a public baseline given the same probe, followed by an explicit global-objective check. Protected-history adaptation already has predecessors; novelty must come from a well-founded construction or demonstrated tradeoff, not this post-processing argument alone. No neural training or CIA sweep is started by this note.

## Reproduce and agent pickup

- Calculator: [heterogeneous_profile_risk.py](../calculations/heterogeneous_profile_risk.py).
- Raw arithmetic artifact: [analytical_heterogeneity.json](../../results/client_specific_noise/analytical_heterogeneity.json).
- Independent review: [heterogeneity_math_review.md](2026-10-04_heterogeneity_math_review.md).
- Original proposal: [mechanism_proposal.md](2026-10-04_mechanism_proposal.md).

Run `uv run python research/calculations/heterogeneous_profile_risk.py` from the repository root. The output is deterministic and includes both grid resolutions' discrepancies. Preserve the unfavorable cases and distinguish the later protected-history candidate from the separately paid selector evaluated here.
