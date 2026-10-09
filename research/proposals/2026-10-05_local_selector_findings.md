# Client-local profile routing: construction law and selection-cost decision

2026-10-05. Owner: “lets move with that”. Approved work: define a rule that follows client characteristics, then calculate whether its privacy cost leaves benefit. This is a bounded construction/accounting design review with a deterministic development calculation, not a deployed private estimator or a declaration that the overall experiment is finished.

The rule removes the slot-order failure: each client privately computes a coarse label statistic, chooses a preferred profile, randomizes that preference, then clips and noises the released vector using that selected profile. A conditional one-upload privacy certificate is available when all calibration/history assumptions hold. However, even the **optimistic partial-accounting calculation** gains at most **0.000850** in development quadratic loss; no case reaches 0.001. This does not justify implementing a paid private selector in the current family.

## What the client would do

1. Count its own examples in four classes. If the largest class fraction is at least 0.6, map its dominant class through the fixed table below; otherwise prefer profile 0. Empty input prefers profile 0; tied dominant counts use the smallest label. The rule uses client content, not a client ID or slot.
2. Privatize that three-category preference with randomized response. This deliberately sometimes chooses a different profile, so seeing a selected law does not deterministically reveal the preference.
3. Clip the complete vector being released into that profile's public weighted L1 body. Add independent coordinate Laplace noise calibrated to the **remaining** upload budget. Apply any public/common shrink to both the signal and noise.

| Local dominant class | Preferred profile |
|---|---:|
| 0 | 0 |
| 1 | 2 |
| 2 | 0 |
| 3 | 1 |
| No class fraction ≥0.6, or empty input | 0 |

In the three-dimensional bias-contrast bank, profile 0 has equal axes; profiles 1 and 2 preserve the first or second Helmert contrast and compress the other directions to one quarter. The retained class-to-profile map comes from the prior controlled label-skew diagnostics. It is a frozen heuristic, not a proven sufficient statistic, a generally optimal routing rule or a new distribution. A nominal “class” here is one of the repository's four image-label classes, not a client identity.

The same client moved to another slot retains the same preference and conditional law. Independent random choices need not yield bitwise-identical samples after relabeling; the assignment **distribution and aggregate expected objective** are invariant when updates, weights and counts follow the clients. The calculator checks this.

## Conditional output law and privacy derivation

Fix a registered slot, a genuinely public conditional history h, positive public semiaxes r_k, a public mapping/threshold, budgets ξ≥0 and η>0, and a public shrink t. Let the whole private client input be D. Its preferred category s(D;h) is in {0,1,2}. For category A=k,

    p_D(k) = exp(ξ)/(exp(ξ)+2) if k=s(D;h),
             1/(exp(ξ)+2) otherwise.

This is existing k-ary randomized response, specialized to three outcomes. The source is [Kairouz, Bonawitz and Ramage, ICML 2016, §4.1, Eq.4](https://proceedings.mlr.press/v48/kairouz16.pdf). The formula was refreshed from the primary paper; its distribution-estimation results are not a proof of FL utility or novelty for this project.

For arbitrary local vector u(D;h), define profile clipping

    v_k(D;h) = u / max(1, sum_j |u_j|/r_kj).

Thus both neighboring inputs clip into the same profile body, including when randomized response selects a nonpreferred profile. Set the empty-input update to zero, **but still run the same randomized-response channel and noise sampler**. Draw Z_kj independently from Laplace(scale=2r_kj/η). The unshrunk joint output (A,V), V=v_A(D;h)+Z_A, has density

    f_D(k,v) = p_D(k) product_j [η/(4r_kj)]
               exp[−(η/2) sum_j |v_j−v_kj(D;h)|/r_kj].

For any two whole-client inputs D,D' in this slot, including a dataset-versus-dummy pair, the RR ratio is at most exp(ξ). The weighted L1 distance between their clipped vectors is at most 2. The Laplace likelihood ratio is therefore at most exp(η). Multiplying gives

    f_D(k,v)/f_D'(k,v) ≤ exp(ξ+η).

With E=ξ+η, this is a **conditional whole-input E-LDP reference certificate for this complete vector release**. Hiding A and applying fixed/public shrink t are postprocessing, including t=0. The calculation does not rely on a record-level class-count sensitivity or stability of the dominant class: arbitrary preferred categories are protected by the RR channel. This is our explicit reconstruction of standard categorical privacy plus bounded-vector composition, not a new theorem or a certified training protocol.

The assumptions matter operationally:

- The profile bank, mapping, radius, budget split, shrink and conditional history must be public/fixed or obtained through already accounted protected releases. Choosing them from raw target gradients is additional private construction.
- Registered slots send the same message schema in dataset and dummy worlds. Private counts, participation flags, raw preferences, timing branches and random coins cannot be exposed outside the output law.
- Public fixed aggregation weights and normalization must not change with the hidden event. Diagnostic fit counts are held fixed for this calculation; they are not certified public metadata for deployment.
- For several adaptive uploads, charge conditional costs across rounds and charge protected history formation too. There is no automatic whole-client privacy guarantee for the saved raw model checkpoint, no automatic amplification from summing clients, and no protection for sensitive complementary parameters left outside the release.

The existing diagnostic perturbs only three bias coordinates of a **raw, unprotected** 68-parameter classifier. Its other coefficients are frozen during the intervention but were learned privately. Hence the lemma does **not** certify the saved model or its full transcript. Raw-development fitting of radius/shrink/budget also remains unaccounted. These obligations must be resolved before calling an implementation a defense against client-contribution inference.

## Exact utility calculation, including selection mistakes

Read the [pre-calculation protocol](2026-10-05_local_selector_protocol.md), [independent math/result review](2026-10-05_local_selector_math_review.md), [calculator](../calculations/local_selector_cost.py), and [all cost curves](../../results/client_specific_noise/local_selector_cost.json).

Reuse saved Fashion-MNIST heads, mean gradients and fit class counts: three seeds, three partition regimes, rounds 5 and 20 = **18 records**, each at E labels 4/8/16 = **54 cells**. No test Arrow is read, no fresh evaluation set is opened, no selector/noise simulation or model training is performed. Existing development examples supply the common loss gradient g and PSD Hessian H.

For independent client selections p_ik and profile-clipped vectors v_ik, define

    μ_i = sum_k p_ik v_ik,
    S_i = sum_k p_ik v_ik v_ikᵀ − μ_i μ_iᵀ,
    μ = sum_i α_i μ_i,
    S = sum_i α_i² S_i,
    N = (8/η²) sum_i α_i² sum_k p_ik diag(r_k²).

The exact expectation of the quadratic development loss change is

    t gᵀμ + (t²/2) [μᵀHμ + tr(HS) + tr(HN)].

Here S is the extra variation from randomized profile selection; N is conditional upload noise averaged over profiles. Omitting S or using the full E upload noise after spending ξ would overstate benefit. The quadratic expectation is exact for this approximation, not exact nonlinear predictive loss. We validate it against full enumeration of all 3⁸ selector assignments on a separate synthetic identity-curvature case.

Optimize common radius on the preceding seven-point grid and public/common shrink over [0,1] for each frozen split ξ/E in {0,0.05,0.125,0.25,0.5}. The ξ=0 case is a data-independent uniform mixture of the three profiles, not a paid informative selector. Record fixed radius 0.3/shrink 1 as well as optimistic raw-development calibration.

Controls receive the full E upload label: tuned common profile; independent public profile draws from a common four-profile categorical distribution, searched on a 0.1 simplex grid with the exact uniform-three distribution added; and an explicitly **unaccounted** deterministic local route without selection cost as a headroom reference. The public mixture contains every deterministic common profile and ξ=0 candidate. Its optimum happens to tie the best shared profile in all 54 cells; that does not prove mixtures universally useless.

Even the optimized selector split is selected using raw development information. This is a **partial** construction ledger: RR and upload costs are included, other calibration/history costs are not. It tests an optimistic route rather than demonstrating an actual equal-total-DP-budget trained mechanism.

## Results and why the budget matters

Positive gain means the best shared/public-mixture quadratic score minus the best selector-grid score. For label stress, each row below spans three seeds and two checkpoints:

| E label | Best selector fraction ξ/E | Correct-route probability | Noise inflation at the same fixed profile | Predicted gain range |
|---|---:|---:|---:|---:|
| 4 | 0.25 | 0.576117 | 1.777778× | −0.000121086 to −0.000071949 |
| 8 | 0.25 | 0.786986 | 1.777778× | +0.000239430 to +0.000367826 |
| 16 | 0.25 | 0.964663 | 1.777778× | +0.000658117 to +0.000849841 |

At E=8, spending ξ=2 leaves η=6. The preferred profile is chosen about 78.7% of the time and its fixed-profile noise variance increases by (8/6)²≈1.78. At E=16, ξ=4 leaves η=12 and the choice is more reliable. This balances errors against noise better, but still does not clear 0.001. The selected split is a finite-grid observation, not the globally optimal continuous allocation.

The route beats the listed baseline in **12/54** cells (label stress at 8/16), loses in the other **42**, and reaches the development 0.001 gate in **0/54**. In balanced/quantity regimes the purity gate routes all clients to profile 0; paid randomization and the restricted profile subset cannot improve the stronger common-profile control. Some best grid allocations are ξ=0, so “best paid” fields in JSON must be interpreted as the best **selector-grid** allocation, not always positive-cost selection.

For seed 43, label stress, round 20, E=8:

| Diagnostic | Development quadratic change |
|---|---:|
| Shared/public mixture | −0.000715005 |
| Unaccounted deterministic local route | −0.001771648 |
| RR route, ξ=2 and η=6 | −0.001044767 |

The apparent free-routing gain of 0.001056643 shrinks to **0.000329762** after the displayed selector/upload costs. The selected RR configuration has radius 0.3 and shrink 1. Its score includes selection variance penalty 0.000042571 and upload-noise penalty 0.000597819. This is a concrete example of construction cost consuming the oracle opportunity.

The strongest remaining advantage, 0.000849841 at E=16, is still an approximate development result with optimistically free global calibration/history. It must not be substituted for the preceding held-out CE results, interpreted as an attack score, or described as a certified privacy improvement over metric privacy.

## Decision and next research question

We now have a specified local rule, a conditional output law, and a reproducible cost calculation. **Do not implement this separately paid RR selector as the next FL mechanism.** The observed headroom is small, no bounded gate passes even with missing calibration costs, and the route remains specialized to manufactured label skew. This is not a universal impossibility result for client-side noise construction.

The next proposed design question is whether the same coarse routing statistic can be derived from **already protected client history**, or whether the selected mixture admits a uniformly tighter direct privacy analysis when its category is hidden. Either route must analyze the complete law and match probe/history costs; neither makes data-dependent noise free. Existing protected-history results also lost to strong one-release controls, so revisiting that direction requires a concrete structural difference rather than another sampler sweep. No such mechanism, tighter bound or follow-up experiment is authorized or claimed by this note.

A genuine distribution-construction contribution still requires a privacy-accounted estimator/kernel and enough utility headroom to survive strong controls. The present result provides a defined target and an unfavorable cost benchmark for future proposals.

Reproduce with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m research.calculations.local_selector_cost --self-check`, then without `--self-check`; use `--cache-dir PATH` for the offline training Arrow. Checks cover exact categorical expectation, selector variance, arbitrary slot relabeling, empty/tie rules, RR likelihood ratios, common-control inclusion and source hash identity. No experiment-completion decision is inferred.
