# Changing-update quadratic probe: held-out utility findings

2026-10-04. Bounded numerical research spike authorized by the owner's “lets move to that”. These are real synthetic learning measurements, not CIA scores, image-model training or a declaration that the branch's experiment is finished.

## Finding and research decision

In all six tested budget/client-count settings, development tuning selected **equal second-round radii** for the adaptive bank. Those profiles are isotropic, so their choice no longer depends on the protected history. The selected adaptive and static mechanisms have identical configurations and bitwise-identical held-out trial losses. A public one-release mechanism performs better in every cell. Free public shrinkage improves both families but does not reverse that ordering.

The earlier stationary two-round advantage does not survive this learning problem and finite profile family. Deprioritize the current initial-axis selector for an FL/CIA implementation. This is not evidence that all client-specific distributions fail. The next bounded diagnostic should distinguish a stale selector from a missing useful profile geometry, with stronger public controls before adding complexity.

## Actual learning problem

Each registered client has a private quadratic loss F_i(w)=0.5||w-theta_i||². Draw theta_i as 0.8e_1 or 0.8e_2 independently with equal probability. The shared model starts at w_0=0. Its raw update is theta_i-w_t, so it changes after the first round.

Use fixed equal weights 1/N. Round 0 clips the full two-coordinate update inside a public isotropic diamond and adds independent Laplace coordinates at eta_0. The server sets w_1 to the mean upload. Round 1 recomputes the **full vector** theta_i-w_1, clips in its selected body, adds fresh noise at eta_1, then sets w_2=w_1+mean(Y_i,1). This is a unit gradient step, not the arithmetic average of the two stationary uploads used in the preceding calculation.

Adaptive profile choice is the larger absolute coordinate of the client's first privatized upload. Its fixed public bank contains swapped diamond semiaxes (L,S)/(S,L). Static controls use fixed public semiaxes, including both orientations. Radial clipping uses the L1 body norm |u_1|/a+|u_2|/b; it does not clip each coordinate independently or ignore the new other-coordinate gradient.

The exact target is the realized federation optimum theta_bar. The reported excess global loss is 0.5||w_2-theta_bar||². With no noise and nonbinding clipping, one unit step already reaches theta_bar; the second average raw gradient is zero. This is deliberately a cheap, learnable quadratic test, and remains a weak justification for requiring multiple rounds.

## Privacy scope and conditioning

For each fixed diamond, clipped outputs differ by at most 2 in its body norm. Laplace coordinate scales are 2a/eta and 2b/eta, giving the ideal joint density proportional to exp[-eta ||z||_B/2]. The conditional upload mechanism protects arbitrary whole-client replacement, not just changing one quadratic coefficient.

The bank index is a fixed function of the already protected first upload. The hypothetical full upload transcript charges eta_0+eta_1=E; profile selection adds no separate RR cost. Global-model dependence does not permit an unprotected raw statistic in the constructor. Conditional second-round bounds must hold for every observed history, including models far from the optimum. These are reference continuous-density arguments, not certification of NumPy's finite-precision/PRNG implementation.

The primary scientific secret remains dataset contribution among fixed registered slots. This utility probe does not run IN/OUT attacks. A future empty-input slot must use a raw update of zero at every model while retaining noise/profile traffic; setting theta=0 would produce gradient -w and is a different learning contribution.

## Controls, tuning and fresh evaluation

Per cell, development uses 1,024 independent federation realizations; evaluation uses 8,192 newly drawn realizations with a distinct seed. The same underlying client types and independent standardized noise arrays are shared across arms only to form paired contrasts. No configuration or shrinkage coefficient is fit on evaluation losses.

Public tuning grid: radii {0.1,0.2,0.4,0.8,1.2,1.6}; first-round budget fractions {0.25,0.5,0.75}. Evaluate 378 adaptive candidates, 648 static candidates (both fixed orientations), and six one-release radii per cell. These are finite family optima, not continuous optimality or an exhaustive distribution comparison. The smallest radius and endpoint budget fractions may restrict the family; arbitrary zero-output/skipped rounds and learning-rate tuning were not jointly optimized.

Controls:

- Optimized static two-round mechanism under the same budget and dynamics.
- Same-probe static mechanism: fix the adaptive arm's exact first release/budget, tune only the static second profile on development.
- Public one-release diamond using all E. It is a different schedule, scientifically relevant because one noiseless step solves this particular objective.
- Known population mean (0.4,0.4), costing no privacy. Its expected excess against the realized optimum is 0.16/N. This public prior is available under the declared known synthetic population.
- Public scalar shrinkage of each selected final model toward (0.4,0.4). Fit its coefficient on development, apply it unchanged on evaluation. This is post-processing and adds no privacy cost; it is not jointly optimized with the main configurations.

Audit seeds permit reproduction and are not intended as attacker-visible protecting randomness in a later attack benchmark.

## Held-out measurements

Values below are mean **excess quadratic loss**, not accuracy or AUC.

| Total epsilon | Clients | Adaptive = static | Public one-release | Adaptive, public shrinkage | One-release, public shrinkage |
|---|---:|---:|---:|---:|---:|
| 4 | 8 | 0.0587524 | 0.0400140 | 0.0179968 | 0.0136455 |
| 4 | 48 | 0.0122541 | 0.0067324 | 0.0025060 | 0.0022543 |
| 8 | 8 | 0.0172418 | 0.0099495 | 0.0082953 | 0.0066541 |
| 8 | 48 | 0.0030756 | 0.0016549 | 0.0015143 | 0.0011097 |
| 16 | 8 | 0.0046555 | 0.0025013 | 0.0036326 | 0.0022328 |
| 16 | 48 | 0.0007621 | 0.0004218 | 0.0005644 | 0.0003710 |

Example E=8,N=8: both two-round families choose first budget fraction 0.75, first radius 0.8, second radii (0.1,0.1). The held-out adaptive-minus-one-release difference is 0.0072922, with paired normal Monte Carlo 95% interval [0.0070573,0.0075271]. All six unshrunk one-release contrast intervals favor one release. Intervals are exploratory, unadjusted for the six cells/multiple contrasts, and quantify simulation variation under this toy population, not a universal scientific claim. The artifact retains all means, standard errors, intervals, selected configurations and per-trial losses.

## Verification and interpretation

Independent mathematical review derives the correct conditional risk over fresh second-round noise:

    E[excess | theta, first uploads]
       =0.5||w_1+mean(v_i)-theta_bar||²
        +4 sum_i(a_i²+b_i²)/(N² eta_1²).

A 100,000-draw conditional sampler check gave 0.0150152 versus formula 0.0149625, Monte Carlo standard error 0.0000521, within the prespecified six-standard-error tolerance. Deterministic checks cover zero-noise learnability, full-gradient cancellation, body sensitivity <=2, and exact adaptive/static identity for equal profiles. The stationary IID error formula was not reused after gradients became coupled through w_1.

A likely reason to investigate is the mismatch between initial parameters and current update directions: near the public population mean, the two raw gradients point along opposite directions of the same diagonal, rather than their initial coordinate axes. This is an inference from the task geometry, not an isolated causal test of the selector. A richer public rotated profile can also exploit that shared direction, so merely adding rotations to a private bank would not establish a client-specific benefit.

## Next research gate

Before extending to images or CIA:

1. Add public rotated profiles and a residual-aware protected-history selector using only the previous private upload and the released global model, e.g. a function of Y_i,0-w_1. Keep static and adaptive banks/total budgets explicit.
2. Compare stale-axis, residual-aware and public-geometry controls to isolate bank limitations from construction limitations. Retain one-release, population-prior and post-processing controls.
3. If useful geometry is shared by all clients, favor the public law. Investigate controlled heterogeneous local curvature only with a stated reason and realistic global objective, rather than designing a population solely to manufacture an adaptive win.

A distribution-construction contribution needs information that helps after aggregation and changing history, a valid complete-law privacy argument, and evidence against strong public controls. No attack-protection conclusion is inferred here. Experiment completion remains the owner's decision.

## Reproduce and pick up

Run `uv run python research/calculations/dynamic_quadratic_probe.py` from the repository root.

- [Calculator](../calculations/dynamic_quadratic_probe.py)
- [Settings, means and uncertainty](../../results/client_specific_noise/dynamic_quadratic_probe.json)
- [Paired held-out trial losses](../../results/client_specific_noise/dynamic_quadratic_trial_losses.npz)
- [Independent math review](2026-10-04_dynamic_quadratic_math_review.md)
- [Preceding stationary findings](2026-10-04_protected_history_findings.md)

The earlier systematic-review counts remain unchanged. This is a mechanism feasibility follow-up, not new literature screening or a final experiment report.
