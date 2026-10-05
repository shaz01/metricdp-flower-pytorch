# Hidden selectors: tail-tight accounting and a fair dummy-adjacency comparison

2026-10-05. Owner: “move on”, approving the next construction/accounting investigation. This follow-up analyzes the hidden selector before a new protected-history design. It is a mathematical audit plus development-cost calculation, not a trained defense, new sampler or declaration that the overall project experiment is finished.

**Finding:** hiding the selected category does not uniformly remove its cost from the current three-profile RR/Laplace law. The widest profile dominates generic tails; a likelihood ratio can approach the additive bound. A separate useful correction comes from the adjacency: dataset-contribution versus a fixed dummy is weaker than arbitrary dataset replacement. Applying that correction **to both the selector and all public controls** leaves a maximum optimistic development gain of **0.000655**, with zero 0.001 gate passes.

## What hiding the profile actually releases

The preceding [local routing law](2026-10-05_local_selector_findings.md) used three-category randomized response with parameter ξ, then profile clipping and coordinate Laplace noise scale 2 r/η. If the observer sees only the vector Y, its density is

    f_D(y) = sum_k p_D(k) q_k(y − v_k(D)),

where q_k is the full normalized noise density and v_k is the profile-clipped vector. Removing the category from the message invokes postprocessing, which preserves the existing upper bound; it does not automatically improve it. Evaluating only typical outputs, covariance matrices or one fitted attack cannot establish a uniform privacy bound.

For this bank the semiaxes are

    r_0 = C(1,1,1),
    r_1 = C(1,1/4,1/4),
    r_2 = C(1/4,1,1/4).

We analyze one complete three-dimensional vector upload conditional on genuinely public history/calibration. This is stronger information access than a model-only observer after aggregation. The conclusion is not an impossibility theorem for aggregate-only CIA defenses, approximate DP, other distributions or mechanisms exploiting a restricted input domain.

## Analytic generic-tail argument

Take y=T(1,1,1) with T→∞. Since every profile-clipped center is bounded, eventually every coordinate y_j−v_kj is positive. The log noise density is

    log q_k(y−v_k) = constant_k − (η/2) sum_j (y_j−v_kj)/r_kj.

The tail slopes are 3/C for profile 0 and 9/C for profiles 1/2. Their density ratio to profile 0 decays as exp(−3ηT/C) times a bounded center/normalizer factor. RR assigns strictly positive probability to all three profiles at finite ξ, so profile 0 dominates the hidden mixture in both neighboring worlds:

    f_D(y)/f_D'(y) →
      [p_D(0)/p_D'(0)] exp[(η/2 C) sum_j(v_0 j(D)−v_0 j(D'))].

For arbitrary admissible private preference/update pairs, choose preferred category 0 and u=+Ce_0 in one world, category 2 and u=−Ce_0 in the other. Their profile 0 centers are already on opposite body boundaries; the RR weight ratio is exp(ξ) and the noise ratio tends to exp(η). Therefore the hidden-vector loss approaches **ξ+η**, the same upper bound as joint category/vector release.

This supplies both a uniform upper bound (the preceding componentwise proof) and an analytic matching supremum witness for the unrestricted bounded-vector/preference family. A finite output grid or numerical optimizer alone would not prove the supremum; the saved finite-tail values only validate the algebra and stable density implementation. Any universal discount below ξ+η is contradicted by this family of outputs/inputs. Pure DP counts rare tails even if typical outputs reveal much less.

Two qualifications preserve the scope:

- Preferred category and update are coupled in the actual dominant-label rule. We also construct a feasible limiting example: a class-constant public softmax with probabilities p=(0.5−a,0.5−a,a,a), pure class 0/class 1 datasets, learning rate 0.5, and C=0.3. Their three bias-contrast gradients approach opposite first-axis directions as a→0, and their preferred categories are 0 and 2. This approaches the same bound over admissible public histories. It does **not** prove that each saved trained head, every radius in the grid or a tighter restricted-data-domain certificate has exactly that supremum.
- A fixed/public positive shrink is an invertible scalar map of Y, so it preserves these ratios. With shrink 0, hidden-vector output is constant and its privacy loss is 0; the tail-tight statement excludes that degenerate intervention. Revealing the category would still carry its own cost at shrink 0.

## Dataset-contribution/dummy adjacency is a separate change

The owner chose secrecy of a registered client's dataset contribution, not physical connection. For a **fixed zero-update dummy**, define edges between any nonempty dataset D and empty input∅ in the same slot; all slots continue sending the same schema, and the dummy uses preferred category 0 through the same RR channel.

For every profile, the weighted L 1 distance between v_k(D) and v_k(∅)=0 is at most 1, rather than the at-most 2 bound for arbitrary dataset replacement. Thus even the **joint** category/vector law has the dummy-edge bound

    E_dummy ≤ ξ + η/2.

This is not a benefit from hiding the profile. It follows from the smaller shift domain. In the hidden mixture the bound is also uniformly sharp in the unrestricted family: reverse the likelihood ratio, place the nonempty input at u=−Ce_0 with preferred category 2, and compare the category 0 dummy in the same positive generic tail. The RR factor contributesξ and the center shift contributesη/2. The coupled class 1 example above approaches this as well.

An E_dummy guarantee on these edges does not give arbitrary nonempty-to-nonempty replacement at E_dummy. Chaining through the dummy gives at most 2 E_dummy on that larger relation. Keep the guarantees separately labeled, including after multi-round composition. Raw participation metadata or changing denominators still fall outside the conditional vector law.

For a target dummy-edge label E, use

    selector: η = 2(E−ξ), hence coordinate scale r/(E−ξ),
    fixed/public controls: η = 2 E, hence coordinate scale r/E.

Both sides get half the noise amplitude of their preceding replacement-calibrated versions at the same numerical label, and one quarter of its fixed-profile variance. Recalibrating only the selector while leaving the shared baseline at scale 2 r/E would create an unfair advantage. The relative variance inflation from spending ξ remains [E/(E−ξ)]².

## Fair development-cost results

[Pre-calculation protocol](2026-10-05_hidden_mixture_protocol.md), [independent mathematical/code/result review](2026-10-05_hidden_mixture_math_review.md), [calculator](../calculations/hidden_mixture_audit.py), [tail witnesses and every cost curve](../../results/client_specific_noise/hidden_mixture_audit.json).

Keep the previously fixed local routing rule, radius/shrink grids, five selector fractions and 287 public common-mixture probabilities. Reuse three seeds × three regimes × two snapshots =18 records, with 54 dummy-label cases. Exact quadratic expectation includes clipped means, independent selector covariance and conditional upload noise. Shared profiles and all public mixtures use the fair dummy calibration above. No test Arrow, new test slice, new training or attack is used.

Positive gain below means public-mixture/shared score minus best selector-grid score. Each label-stress row spans three seeds and two snapshots:

| Dummy label E | Best selector fraction(s) ξ/E | Predicted gain range |
|---|---|---:|
| 4 | 0.25 or 0.5 | −0.000151665 to −0.000114422 |
| 8 | 0.5 | +0.000460306 to +0.000654549 |
| 16 | 0.5 | +0.000418444 to +0.000623879 |

The selector wins 12/54 cases and loses 42/54; all wins remain in label stress at 8/16. Balanced and quantity-skew cases remain unfavorable. **Zero of 54 cases reaches 0.001**, even under optimistically free raw history/global calibration. The finite-grid split optimum is not a continuous-budget optimum. At labels 8/16 the chosen fraction 0.5 is the largest allowed fraction, so the interior/continuous optimum remains unresolved. The zero-pass finding applies to the listed grid; no numerical grid was extended after inspecting outcomes.

Example, seed 43, label stress, round 20, dummy label 8:

| Control | Development quadratic change |
|---|---:|
| Fair shared/public mixture | −0.001081960 |
| Unaccounted deterministic local route | −0.002036978 |
| RR selector, ξ=4 and η=8 | −0.001696178 |

The remaining predicted gain is 0.000614217. Its dummy-edge upper bound is 8; its displayed arbitrary-replacement upper bound is 12. Calling both bounds 8 would silently change the claim. The shared control has η=16 and the same dummy-edge bound 8. Mixture and shared optima tie in these cells; this observation is not a theorem about all public distributions.

All numbers are **development quadratic proxies**, not held-out CE or CIA outcomes. The current saved history, privately discovered routing map and raw-development fitting of radius/shrink/budget are still unaccounted. The fair adjacency correction does not fix those missing construction costs or certify the raw 68-parameter classifier's complementary outputs.

## Research decision

Do not implement the current paid selector by assuming its category can be hidden for free. The generic-tail argument blocks a uniform discount for this law; the owner-aligned dummy-edge calibration is real but applies equally to the controls and does not pass the bounded headroom gate. This leaves private distribution construction open, rather than proving that client-side protection cannot work.

The next proposed investigation is the remaining **protected-history route on the measured label-skew setting**: determine whether an already protected low-dimensional release contains enough information to route later profiles, while counting its cost and keeping a matched one-release control. The structural reason to reconsider history is now explicit: a coarse, slot-invariant local statistic and persistent real-data geometry have been identified. Prior synthetic history tests still argue against assuming a benefit. A design/proof and probe-cost comparison must precede implementation; no new history sampler or FL/CIA sweep is launched here.

Alternatively, a different mixture family with common tail behavior or a proved restricted-domain/approximate-privacy guarantee would need a new uniform analysis and literature comparison; it cannot inherit a discount from this calculation.

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m research.calculations.hidden_mixture_audit --self-check`, then without `--self-check`; `--cache-dir PATH` overrides the offline training cache. Stable logsumexp avoids tail underflow. Checks cover finite witnesses against analytic bounds, feasible label-gradient limiting witnesses, dummy budget identity, public-control inclusion, source hashes and permutation invariance. No experiment-completion decision is inferred.
