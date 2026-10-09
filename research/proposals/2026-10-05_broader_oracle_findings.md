# Broader oracle: narrow bias-calibration headroom, no full-head gain

2026-10-05. Owner approved the next step (“lets move to that”, then “continue”). This diagnostic answers the approved feasibility question; it does not declare the project experiment finished.

Expanding from two biases to **all three class-bias contrasts** reveals a small, repeatable point-estimate benefit under the explicit label-skew stress partition. At round 20 and noise label 8, all three seeds improve cross-entropy by slightly more than the 0.001 gate; two noise-only intervals lie entirely beyond it. Expanding to the **entire 51-dimensional identifiable classifier head** produces no gate pass. The positive result concerns how client-specific clipping changes the global learning signal, rather than lower noise cost or a successful learned covariance distribution.

## What was frozen and run

Read the [pre-evaluation protocol](2026-10-05_broader_oracle_protocol.md) and [independent mathematical review](2026-10-05_broader_oracle_math_review.md). Code: [broader_geometry_oracle.py](../calculations/broader_geometry_oracle.py). Artifacts: [all settings and measurements](../../results/client_specific_noise/broader_geometry_oracle.json), [paired trial losses, bases and fresh test indices](../../results/client_specific_noise/broader_geometry_oracle_trials.npz).

We reused the saved unprotected Fashion-MNIST fixed-feature classifier checkpoints and client mean gradients, not new training or a CNN. Three seeds × three partitions × rounds 5/20 × two spaces give **36 records**, with three noise labels each: **108 comparisons**. The primary evaluation uses the next 256 official test images per class (indices 256–511 in per-class order), disjoint from the preceding audit's first 256. Training corpus and evaluation subset are still shared across seeds; this is not an independent-client-population replication.

The public Helmert class-contrast basis is fixed independently of the data. Bias space has three directions and freezes feature coefficients. Full-head space is 3 class contrasts × 17 feature/bias coordinates. Removing the common-logit gauge preserves every predictive head change in this **reduced fixed-pixel classifier**; it does not represent a full CNN.

Each space uses four fixed weighted L1 bodies: equal axes or retain one contrast group while compressing the other two to one quarter. This extends the diamond idea but is not the identical earlier rotated two-dimensional bank. Common radius grid 0.001, 0.005, 0.02, 0.1, 0.3, 1, 3; noise labels 4, 8, 16. Fixed-profile independent Laplace noise has coordinate scale 2r/label. Shared and oracle choices use the same raw client updates, global development gradient/Hessian and radius grid. Oracle enumerates all 4⁸ assignments; shared receives all four common assignments. Both optimize a common step shrink in [0,1], scaling **mean and noise together**, and include no intervention at shrink zero. Selection uses development data only; fresh test loss is not used to tune settings.

The same numerical 0.001 absolute-CE gate was retained before fresh evaluation. It is a local feasibility decision, not a formal preregistration, privacy bound or significance standard. Direct CE uses 1,024 paired noise draws per comparison. Counts and settings were not extended after inspecting results.

## Results across the complete frozen family

Positive “gain” means shared noisy loss minus oracle noisy loss. Each table row contains 18 comparisons (three seeds, two checkpoints, three labels). Best gains are descriptive maxima, not corrected hypothesis tests.

| Space | Partition | Largest CE gain | Point-estimate gate passes |
|---|---|---:|---:|
| Three bias contrasts | Balanced | 0.000029561 | 0/18 |
| Three bias contrasts | Quantity skew | 0.000161121 | 0/18 |
| Three bias contrasts | Label stress | 0.001126164 | 3/18 |
| Full head, 51 directions | Balanced | 0 (exact shared configuration) | 0/18 |
| Full head, 51 directions | Quantity skew | 0.000153738 | 0/18 |
| Full head, 51 directions | Label stress | 0.000236471 | 0/18 |

Exactly **3/108 point estimates** and **2/108 conditional noise-only intervals** clear the gate. All three positive cases are bias space, label stress, round 20, noise label 8:

| Seed | Shared noisy CE | Oracle noisy CE | CE gain | 95% noise-only gain interval | Interval clears 0.001? |
|---|---:|---:|---:|---|---|
| 42 | 1.229458561 | 1.228387120 | 0.001071441 | [0.001029197, 0.001113685] | Yes |
| 43 | 1.226790396 | 1.225664231 | 0.001126164 | [0.001079150, 0.001173179] | Yes |
| 44 | 1.231008628 | 1.229972388 | 0.001036240 | [0.000994775, 0.001077705] | No |

These gains are approximately 0.084–0.092% of shared CE. At round 5, label 8 gains are 0.0009228–0.0009832, below the gate. Other bias label-stress budgets also remain below it. The signal is consistent but marginal and localized, not evidence of a broad effect. Confidence intervals condition on the fixed examples, checkpoints and settings, omit data/client/estimation uncertainty, and are exploratory without multiplicity correction.

All three passing cells choose radius 0.3 and shrink 1 for both arms. Shared chooses profile 2 for every client. Oracle selects [0,2,0,1,0,2,0,1], repeating across seeds. Thus the positive result was not obtained by withholding shrinkage optimization from the shared control. A post-run descriptive check of development scores confirms that in these three cells bias-space shared choice also beats the full-head shared choice; the positive result is not erased by selecting between the two already frozen shared spaces on development data. This check adds no settings or test-driven selection and is not a new primary endpoint.

## What generates the gain?

The strongest case (seed 43) separates the controls:

| Quantity | Shared | Oracle |
|---|---:|---:|
| Noiseless clipped/shrunk CE | 1.226647980 | 1.225335263 |
| Quadratic noise penalty | 0.000178428 | 0.000353801 |
| Actual mean noisy CE | 1.226790396 | 1.225664231 |
| Clipped aggregate deviation from raw step, Euclidean norm | 0.010183533 | 0.027727776 |

No intervention has CE 1.227564745; the noiseless raw projected step has CE 1.226393861. The oracle's noiseless advantage is about 0.001313, despite almost double the quadratic noise penalty and a **larger** deviation from the raw aggregate update. The personalized clipping changes which client directions cancel, producing a global bias correction favored by the development loss. It is not simply more faithful reconstruction of the raw update, lower covariance noise or proof that local gradient-covariance estimation is useful.

The repeated assignment aligns with the manufactured label structure. Because the stress partition assigns dominant classes by fixed client slot, a **frozen publicly heterogeneous slot-profile control** may reproduce these outputs without private client estimation. This control was not part of the frozen comparison, so superiority over it is unestablished. Freezing the observed assignment now and evaluating again on the same examples would be hindsight, not confirmation; assess it on genuinely fresh data in a subsequent approved comparison. No causal ablation has established which client statistic is sufficient to select it. It uses globally optimized raw information. A client cannot just reproduce that assignment without a defined local rule, shared utility signal and accounted construction process.

In the 51-dimensional family, many balanced/quantity cases select identical shared and oracle configurations. Label-stress cases often shrink much more strongly: the wider L1 clipping/noise tradeoff produces little personalization gain. This result depends on the fixed bank, common radii and coordinate convention; it does not rule out different bodies, projections or covariance-aware full-head mechanisms.

## Research decision and handoff

This is **limited positive headroom for bias calibration**, alongside an unfavorable full-head result. It warrants a narrowly scoped construction/proof design review, rather than implementing a private full-gradient covariance estimator or launching a CIA sweep. Existing Laplace/L1 sampling is a reference law; no distribution novelty is established.

The next proposed question is: can a client-local, protected statistic choose a low-dimensional joint clipping/noise profile that reproduces a meaningful fraction of this bias-calibration gain at the same total protection cost? Before implementing that selector:

1. Define a local statistic and how clients obtain a legitimate shared utility signal. The present oracle sees private global development derivatives and all raw client updates; neither may be silently available for free in the actual threat model.
2. Specify the entire output law for dataset-contribution versus dummy-slot adjacency. Profile choice, radius/shrink choice, weights and all complementary model coordinates belong in the observer/accounting argument. Bias-only protection cannot leave sensitive feature updates exposed.
3. Compare against a frozen publicly heterogeneous per-slot profile assignment, stronger shared calibration, and ordinary optimization corrections that could explain the same improvement. The repeated assignment makes the per-slot control a priority before spending privacy budget on a selector. This could be a learning-calibration opportunity, not a noise-density opportunity.
4. Bound the extra private construction cost against the small 0.001-scale oracle benefit before sampling/training. State the gate needed for repeated seeds and genuinely fresh evaluation; two of three conditional intervals is not a population-level confirmation.

Do not infer authorization for another experiment from these recommendations. No final mechanism has been chosen, no privacy guarantee has been proved for the diagnostic, and no CIA score was measured. Keep the frontier as the later model-utility/leakage comparator with the evaluation limitations already documented.

## Reproduction and checks

Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m research.calculations.broader_geometry_oracle --self-check`, then the same command without `--self-check`. `--cache-dir PATH` overrides the offline Fashion-MNIST Arrow location. Source hashes must match the saved audit. The run reads no external dataset and retrains no model. Checks cover contrast orthonormality/gauge removal, projected finite-difference derivatives, L1 containment, reconstructed objective, no-op/oracle inclusion, direct CE identity, source hashes and fresh test separation. Committed trial losses allow independent reconstruction of every mean/paired interval.
