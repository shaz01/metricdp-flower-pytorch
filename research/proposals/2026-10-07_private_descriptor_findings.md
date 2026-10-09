> Subsequent attack evidence: [fixed-slot CIA findings](2026-10-07_descriptor_cia_findings.md). The useful epsilon8 descriptor remains distinguishable. The next recommendation is now matched attack–utility calibration; this earlier utility result is not a CIA superiority claim.

# One-time client descriptors: bounded feasibility findings and research handoff

2026-10-07; runs on 2026-10-06, verification on 2026-10-07. Active branch: `feature/client-specific-noise`.

A promising utility construction survived a fresh held-out comparison: **class-balanced local teachers, jointly clipped public-prototype votes, client noise shares added once, and learning from the sanitized aggregate**. At epsilon 8, this beats the strongest development-selected corrected/reweighted model, logit and probability controls in all three label-stress seeds under both analytic Gaussian and radial L2-Laplace noise. Individually reusable private client distributions do not survive the stronger comparison. Central noise can use the same successful constructor. This is a bounded reduced-model feasibility result, not CIA protection, superiority over metric privacy, novelty certification or completion of the overall experiment.

## What a client constructs, in plain language

A client has its own skewed dataset: a locally frequent class can dominate a teacher even on examples from other classes. Asking that teacher for probabilities or votes can therefore preserve the skew rather than useful class distinctions. The follow-up tested two established remedies before applying privacy:

1. **Prior correction:** estimate smoothed local frequencies `pi_c=(count_c+1)/(n+4)`, train normally, then subtract `log(pi_c)` from the prediction logits. Under label shift and a calibrated local posterior, this removes the local class-prior advantage and targets the public uniform prior. It does not solve arbitrary class-conditional feature shift or learn missing classes.
2. **Class-balanced loss:** train the local teacher with inverse-frequency example weights `1/pi_y`, normalized to mean one. This encourages learning minority classes before any predictions are constructed. It is a separate training rule; do not apply prior correction again to this arm or equate it with FedLC's margin formula.

The useful candidate uses the second remedy. The local teacher predicts one class at each of 16 public, unlabelled feature prototypes. Those predictions become a single centered vote vector, represented in three identifiable class contrasts per prototype: **48 coordinates total**. Divide by the square root of the prototype count, apply the fixed slot weight, and clip the entire vector to one public norm cap. Add one random vector locally and upload the bounded noisy vector. Registered dummy clients upload zero signal plus the same prescribed noise share.

The server first **linearly sums** the noisy vectors. Only then does it project predictions onto probability simplices and fit a student on public prototypes using a ridge decoder. Reusing this aggregate or student is postprocessing; it does not revisit private client data. This pilot computes the distribution exactly rather than introducing synthetic sampling noise.

There are two distributions to distinguish. A **predictive distribution** describes class probabilities at public inputs, which can guide subsequent learning. An **additive-noise distribution** describes the random vector used to protect the released descriptor. This positive result improves construction of the private signal/predictive distribution. Its noise law and calibration are public and fixed; it does not yet demonstrate a useful privately learned, client-specific noise density. Internal private frequencies are permitted by the conditional certificate because the entire resulting query is clipped and sanitized once. Publishing the frequencies, counts, raw teachers or tuning diagnostics would be a separate release.

## Protocol chronology and information boundaries

The [initial descriptor protocol](2026-10-06_private_descriptor_protocol.md) preceded utility inspection. The non-Gaussian/analytic-Gaussian extension was frozen before inspecting utility, although the initial zCDP pilot was already running. This is a locally frozen research protocol, not a formal external preregistration. All three law pilots reuse the same data and cannot be counted as independent population confirmations.

The initial pilot compared model parameters, model-derived distributions, logits, probabilities and hard votes. It used balanced, quantity-skewed and explicit label-stress partitions, three seeds, and aggregate, individual and central privacy contracts. Each law searched 972 settings across the contracts on development data; 27 evaluation cells and 405 arms per law were saved. Configurations were selected at epsilon 8 and frozen for evaluations at 4, 8 and 16. Each evaluation used 256 new paired noise draws. The initial labels-stress result was negative.

A subsequent development-only diagnosis found that raw 80-step teachers produced the same aggregate histogram `[2,2,2,2]` at every one of the 16 prototypes, in all three label-stress seeds. A query with no varying class signal cannot be rescued just by improving its noise. This motivated the separate [prior/balanced constructor protocol](2026-10-06_prior_constructor_protocol.md), not a retrospective change to the original gate.

The second stage compared raw, prior-corrected and balanced-loss teachers for every descriptor family and every contract. Both analytic Gaussian and radial L2-Laplace were retained. Development used 32 paired draws and 17,496 configurations over six seed/law rows. Candidates and strongest controls were saved before extracting reserve features. For each contract the voting candidate was the better development-selected prior/balanced route; controls included all three transforms of model, model-distribution, logits, probabilities and raw votes. Thus the positive result is not merely comparison against an uncorrected teacher.

The source is cached Fashion-MNIST classes 0–3, a 16 pooled-pixel-feature plus bias classifier with 51 identifiable parameters, eight fixed client slots, local full-batch training at learning rate 0.5. All 12,280 records used by the earlier geometry study were excluded. The 11,720 untouched records were split into 512 public, 1,024 development, 2,048 initial evaluation, 4,096 private-bank and 4,040 reserve examples. Public prototypes use features only. Client private-fit/check splits and repeated partition seeds reuse the private bank; they are not independent sampled client populations.

Confirmation used the first preordered 512 reserve examples per class: **2,048 fresh evaluation examples**, disjoint from the original evaluation and development roles. There are **1,992 reserve examples still unused**. It evaluated the previously saved epsilon-8 choices at epsilon 4/8/16, with **512 new paired noise draws**, producing 18 cells and 810 arms. The successful aggregate voting settings were identical across all six seed/law rows: 80 local steps, radius 0.8, 16 prototypes and ridge 0.01. No confirmation-driven retuning was performed.

## Fresh-reserve primary result at epsilon 8

Positive gain means `control CE - candidate CE`; lower cross entropy (CE) is better. Intervals are paired-noise mean plus/minus 1.96 standard errors on the fixed held-out examples. They quantify noise variability, not population uncertainty or a simultaneous confidence guarantee after searching many alternatives. The retained feasibility gate requires the lower interval endpoint to exceed 0.001 in each of the three seeds. Both laws pass this bounded gate.

| Noise law | Seed | Voting CE | Accuracy | Corrected control CE | Paired CE gain [95% interval] |
| --- | ---: | ---: | ---: | ---: | --- |
| analytic_gaussian | 42 | 0.898331 | 66.40% | 1.075642 | 0.177312 [0.172327, 0.182296] |
| radial_laplace | 42 | 0.950665 | 63.21% | 1.082478 | 0.131813 [0.122285, 0.141342] |
| analytic_gaussian | 43 | 0.956956 | 64.34% | 1.072514 | 0.115558 [0.110066, 0.121049] |
| radial_laplace | 43 | 0.991605 | 62.26% | 1.088914 | 0.097309 [0.087083, 0.107535] |
| analytic_gaussian | 44 | 0.966584 | 63.14% | 1.080559 | 0.113975 [0.108744, 0.119206] |
| radial_laplace | 44 | 1.012031 | 60.37% | 1.112460 | 0.100429 [0.094356, 0.106502] |

For analytic Gaussian, the strongest selected controls are balanced-loss probabilities in every seed. For radial Laplace, seeds 42/43 use balanced-loss logits with four prototypes, while seed 44 uses balanced-loss probabilities. As a descriptive check, radial seed 44's best actual held-out selected non-voting control is the balanced-loss logits arm: the voting gain remains 0.084996, with interval [0.075378, 0.094614]. This check is not a new test-selected configuration.

The richer auxiliary control trains on the labels of the 512 public examples; the core constructor does not use those labels. On this reserve its CE is **0.989902** and accuracy **71.14%**, above the private voting candidate's accuracy. A lower CE for some voting arms does not establish superiority on accuracy or under equal auxiliary information.

The initial stage is also informative: at epsilon 8, aggregate votes cleared the balanced/quantity gate in all six such cells per law, but lost in all label-stress cells. Aggregate Gaussian probability descriptors also passed those six cells; radial probabilities did not. The second stage therefore addresses a specific failure with a separate fresh-image confirmation, not a blanket win across every dataset or partition regime.

## Stronger privacy and central controls

| Noise law | Seed | Individual candidate gain vs aggregate control | Aggregate candidate gain vs central same-vote control |
| --- | ---: | ---: | ---: |
| analytic_gaussian | 42 | -0.141845 | -0.007128 |
| radial_laplace | 42 | -0.082982 | -0.011842 |
| analytic_gaussian | 43 | -0.151586 | -0.004202 |
| radial_laplace | 43 | -0.080593 | -0.009336 |
| analytic_gaussian | 44 | -0.133165 | -0.005956 |
| radial_laplace | 44 | -0.046743 | 0.002365 |

The individually reusable candidate has negative gain against the strongest aggregate control in all six primary cells. Against controls with the same stronger individual contract, no complete three-seed pass survives (radial seed 44 alone passes). Separately private client distributions remain a research problem, not a successful mechanism here.

Analytic-Gaussian aggregate voting also loses to central noise on the same voting query in every seed. Radial aggregate voting loses in two seeds; its small positive third difference has an interval crossing zero. Central placement can exploit the same representation. Representation/training, not relocation of noise, explains the promising result.

## What the privacy certificate covers

The hidden event is whether a registered client's whole dataset contributes to learning, versus its fixed-slot empty-data dummy. Weights and roster are public and fixed; no private counts, flags, absent-slot renormalization, or connection metadata are released. Quantity weights are the public sequence 1 through 8 divided by 36, not private sample-count weights. Empty queries must explicitly map to zero: a zero teacher's argmax otherwise votes for class zero. The implementation now carries an explicit activity mask for this reason.

Let `x_i=clip_S(a_i d_i)` and `S=radius/8`. The dummy is zero, so its star-adjacency shift is at most S. Arbitrary dataset replacement can have shift up to 2S and needs corresponding recalibration. Three contracts stay separate:

| Contract | Observer/release | Noise relation to certified unknown floor |
| --- | --- | --- |
| Aggregate A | Curious peer sees aggregate/student and its own data/update/coins; honest server withholds individual uploads | Seven unknown shares protect a target; all eight shares give 8/7 times the residual variance |
| Individual B | Each client descriptor may be reused individually, including client-wise nonlinear distribution construction | Every upload has the full certified floor; linear aggregate variance is eight times the floor |
| Central C | Trusted server applies one certified noise vector to the same bounded aggregate query | Aggregate variance equals the floor |

In A, separate nonlinear transformations of the low-noise uploads do not inherit the aggregate certificate. In B, learning from the collection of individually sanitized objects is postprocessing for the one target client; unchanged clients do not consume extra target budgets. A is not protection against a server inspecting raw uploads or colluding peers; that would require a separate residual-noise/accounting contract.

The conditional fixed-mechanism calculation covers the released query under the stated adjacency and observer assumptions. Offline raw-data tuning, archive publication and experiment-selection metadata are unaccounted. The committed archives deliberately contain operator-only raw teachers/counts and diagnostic objects; they are **not** the modeled private transcript. NumPy seeds and finite-precision noise sampling are research tools, not an audited deployment implementation. There is no end-to-end private selection/publication guarantee.

## Why test non-Gaussian laws and what happened

The [math review](2026-10-06_private_descriptor_math_review.md) and [noise/descriptor methods cards](../literature_review/one_time_descriptor_methods.md) explain the primary sources and complete sampler.

- **zCDP Gaussian:** a conservative reference uses `rho=(sqrt(log(1/delta)+epsilon)-sqrt(log(1/delta)))^2` and variance `S^2/(2rho)`. It supplies a rho-zCDP certificate converted to the stated epsilon/delta target.
- **Analytic Gaussian:** calibrate directly with the exact Gaussian hockey-stick expression at delta `1e-5`. At epsilon 4/8/16, variance divided by S squared is 1.168910944858024 / 0.3602749391128144 / 0.1184581022863902. This is an epsilon/delta comparison, not equality of the earlier rho values.
- **Radial L2-Laplace:** the established spherical density is proportional to `exp(-norm(z)/s)`, with `s=S/epsilon`. Draw one scalar `G ~ Gamma((d+1)/2, scale=2s^2)` and a d-dimensional standard normal vector; output `sqrt(G)*N`. The same G scales the entire vector. Its characteristic function is `(1+s^2*norm(t)^2)^(-(d+1)/2)`, permitting client shares with Gamma shape divided by seven for A. Seven unknown shares recover the certified radial law. Keep scales/coins secret. This is pure epsilon protection for dummy adjacency, not a new density.

Radial coordinate variance is `(d+1)*(S/epsilon)^2`. At epsilon 8 and dimension 12 it is 0.203125 times S squared, about 44% below analytic Gaussian's 0.360275. At dimension 48 it rises to 0.765625 times S squared. Both selected voting constructors use dimension 48; analytic Gaussian has better held-out CE in all three matched primary rows. Non-Gaussian noise remains a legitimate candidate, but scalar or low-dimensional advantage cannot be extrapolated to this joint vector.

Individual low-noise radial shares in A can have a singular density and lack a finite pure translation-DP bound. Only the unknown aggregate residual has the intended radial certificate. Publishing the Gamma scale or incorrectly applying a coordinate-independent mixture invalidates the stated law. The full eight-share marginal differs from the seven-share residual; equal central marginal laws can replicate it. Do not describe client shares as individually certified uploads.

## Literature interpretation and novelty boundaries

The [prior-alignment card](../literature_review/prior_aligned_constructor.md) explains Menon's posthoc logit adjustment versus training adjustment, and FedLC's different count-based margin rule. The [descriptor card](../literature_review/one_time_descriptor_methods.md) separates FedMD's public-labelled repeated distillation, FedDF's unlabelled fusion, PATE's record-based teacher voting, NFDP's sampled-subset guarantee and privately released input distributions. Our balanced-loss arm is not a replication of Menon's training adjustment or FedLC. Whole-client dummy adjacency is not inherited from a record-level PATE/NFDP theorem.

The defensible insight is that **constructing a useful, bounded client query before a one-time release can matter much more than changing covariance or adaptive noise spending** in this reduced label-skew regime. Familiar imbalance correction, voting, distillation and radial noise ingredients prevent a novelty claim based only on their combination. A paper needs a precise closest-work comparison and demonstrable attack/utility advantage under its complete threat model. The frozen 31-family systematic corpus is unchanged; these scoped method reads and follow-up query executions are recorded separately and overlap earlier families.

## Verification and reproducible artifacts

The independent saved-artifact auditor imports neither simulator nor simulator statistics. It recomputes every saved utility mean, paired contrast and interval, development selection, calibrated variance, source hash and role overlap. It also independently reconstructs all six primary aggregate voting evaluations from cached images and saved teachers, including bounded query, noise generation, simplex projection, student fitting, CE and accuracy. Its [receipt](../../results/client_specific_noise/private_descriptor_artifact_audit.json) records **68,532 numerical comparisons**, maximum absolute error **4.440892098500626e-16**, all 81 initial cells/1,215 arms and 18 confirmation cells/810 arms, source verification, disjoint roles and reserve counts. Earlier independent agent reviews checked the privacy math/sampler and the stage-1/development arithmetic; the final reserve arithmetic is checked by this separate auditor, not claimed as a completed agent review.

Kernel checks cover projection, contrast geometry, analytic calibration, radial moments, cap/dummy behavior, balanced-loss identity and prior-correction algebra. The full default suite passed **277 tests, 5 deselected** on 2026-10-07. The five legacy-environment reproducibility tests are outside this change. Reproduction commands and archive descriptions are in the [result README](../../results/client_specific_noise/README.md). No jobs remain running.

## Research roadmap after this bounded result

The next direction is the successful aggregate balanced-vote constructor, with individually reusable distributions retained as a harder extension. Keep analytic Gaussian as the primary control and radial Laplace as a genuine alternative. This is a recommendation supported by utility evidence, not a declared final defense.

1. **Define and freeze the CIA evaluation before reading new attack results.** Use fixed registered slots and identical public denominator; whole-dataset IN versus empty-data dummy; the same noise law in both worlds. The peer gets its own state and coins and all released global information. Preserve background-client identities/randomness. Separate attack-development clients/data from attack evaluation. Report ordinary oriented ROC-AUC, uncertainty and attacker feature selection; retain historical folded concordance only as a separate reproduction metric. Do not compare fixed-slot dummy results directly with the old renormalized removal game.
2. **Test whether utility survives matched leakage.** Include aggregate balanced votes, corrected/reweighted model/logit/probability controls, the central same-vote constructor, and individual distributions. Include metric-inspired server calibration under the same observation/event/utility protocol, without assigning it the new DP certificate. Verify whether the candidate moves the attack–utility frontier rather than simply improves its student. Epsilon 8 utility feasibility alone says nothing about chance-level CIA.
3. **Probe the failure conditions.** Independently sampled client populations, missing classes, class-conditional feature shift, varying prototype dimension and truly unseen support must precede a broad CNN sweep. The current three seeds reuse the same private bank. Use the remaining reserve only under a newly frozen bounded protocol; it is not another tuning set.
4. **Complete the contribution test and deployment boundary.** Resolve closest one-shot private ensemble/distillation work and remaining method gaps. Decide whether the contribution is a query construction/certificate or a new noise law. If claiming individually constructed private noise shapes, specify the full conditional law and private estimation cost first; this pilot does not solve it. A deployed system needs hidden uploads where A requires them, a secure sampler, explicit collusion assumptions and accounted tuning/publication.

This note documents an authorized research chunk. It does not decide that the overall experiment is finished, authorize merging the active branch or replace the owner's completion decision.
