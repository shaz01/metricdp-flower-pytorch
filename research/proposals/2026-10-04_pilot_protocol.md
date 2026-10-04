# Proposed pilot: client-specific noise for whole-client contribution inference

2026-10-04. Branch: `feature/client-specific-noise`; baseline inspected: `a03d637`. **Planning proposal only. The owner confirmed that the hidden event is dataset contribution, rather than physical connection. The detailed numerical pilot remains proposed and no experiments have been launched.** Numerical values below are starting defaults for feasibility/power planning, not validated sample sizes or a selected mechanism.

The [concrete mechanism proposal](2026-10-04_mechanism_proposal.md) and [first analytical feasibility check](2026-10-04_analytic_feasibility.md) precede any learning pilot.

Read with [threat specification](../threat_specification_review.md), [evaluation requirements](../literature_review/evaluation_requirements.md), [construction obligations](../noise_construction_proof_obligations.md), and [non-Gaussian research direction](../non_gaussian_mechanism_research.md). The owner wants a potentially non-Gaussian client-specific construction; Gaussian remains a mandatory reference.

## 1. Proposed game and exact release

The server is trusted and follows the protocol. A passive honest-but-curious peer A and target T are distinct enrolled parties. A public roster of N slots is identical in both worlds. For each round t, all slots send identically scheduled, shaped messages. IN uses T's whole local dataset; OUT replaces T's *learning contribution* with zero. The target's slot still supplies the prescribed randomized profile selection/noise/dummy traffic. No other dataset is redistributed.

This hides whether an enrolled client's data contributed. It does **not** hide physical enrollment, connection, or message existence. Physical-absence secrecy would require a different protocol. This pilot does not inherit an absent physical client's noise magically.

Use public fixed weights `a_i=1/N` and fixed denominator N in the first pilot. Train and clip an update u_i before adding noise; `||u_i||_2 <= C`, and `u_T=0` in OUT. Release

\[
w_{t+1}=w_t+N^{-1}\sum_{i=1}^N(u_{i,t}+Z_{i,t}).
\]

Weights, the public bank of joint clipping/noise profiles, optimizer/noise position, N, roster, public law families, privacy/accounting parameters and release schedule are known to A. The displayed Euclidean norm bound is the isotropic reference. The main candidate selects a joint clipping/noise profile; each fixed profile needs its own same-history sensitivity bound, with an explicit public bound on its clipping set. For the norm-C reference, the zero-slot target mean shift is bounded by `C/N`; arbitrary replacement requires `2C/N`. Background updates may change later because the global history changed.

The primary attacker view includes every released global checkpoint plus A's dataset, local optimizer state, clipped/unclipped update, uploaded message, sampled noise/profile, known seeds, and the allowed configuration. It does not contain other parties' raw updates, noise draws, private summaries, oracle labels, or counterpart trajectories. Public law parameters need not be secret; their realized honest randomness must remain unknown to A. For independent noise, residual covariance is `N^-2 sum_{i != A} Cov(Z_i)`; correlated laws require the full observer-conditioned joint law.

The simulator/oracle manifest retains IN/OUT labels, actual private datasets, private reproducibility seeds, and diagnostic pairing IDs. Export a separate attacker manifest. Never hide actual peer-visible information merely by excluding it from a Python feature list. Logging, failure/timing behavior, message shape and any profile-related metadata must obey this contract.

## 2. Controls and candidate slots

| Arm | Purpose and comparability |
|---|---|
| V: zero-noise fixed-roster reference | Establish learnability and participation signal under the new denominator. |
| G-local: fixed public isotropic Gaussian shares | Establish the local fixed-noise baseline. |
| G-central-peer: matched hybrid central reference | A uses/knows its own draw exactly as in G-local; server samples the sum-law of the other parties' draws. Isolate honest-noise placement. |
| Q-public: fixed public non-Gaussian law | Test law shape before paying private construction cost; density and aggregate law must pass the gates below. |
| Q-selected: randomized private selection from a finite public **joint clipping/noise** profile bank | Main viable construction hypothesis; account the selector and every conditional update law. Launch remains gated. |
| Common-Euclidean-clip/shaped-Gaussian negative control | With unrestricted Euclidean sensitivity and the same Gaussian bound, feasible shaped covariance is PSD-dominated by the minimal isotropic reference. Diagnose purported geometry gains caused by changed assumptions. |
| Legacy server metric/global-DP | Historical separate sample-weighted/active-removal protocol; useful benchmark, not a matched placement arm. |

No non-Gaussian family or profile bank is selected here. The public law can be a calibrated known family or a new justified template; novelty and suitability remain research questions. Within a fixed profile, hold clipping geometry fixed when comparing density shape. The main profile construction varies clipping geometry jointly with noise; compare it against public fixed profiles and a data-independent selector. Common Euclidean clipping with shaped Gaussian noise is a negative control under the full sensitivity-ball model, not the main expected gain. Compare formal budget, distribution shape, geometry, and total noise as distinct factors.

**Exact matched peer control.** Let H be all slots other than A. In a local arm, unknown aggregate noise has law

\[
Q_H=\mathcal L(N^{-1}\sum_{i\in H}Z_i).
\]

In its central peer control, the trusted server draws `E_H ~ Q_H`, keeps A's own weighted noisy contribution unchanged, and releases `N^-1 sum_i u_i + N^-1 Z_A + E_H`. For Gaussian shares, `E_H` is Gaussian with covariance `N^-2 sum_H Sigma_i`. For a non-Gaussian law, sample the **exact scaled convolution** or use a certified approximation; matching its variance or applying the Gaussian square-root-N scaling does not establish law equality. The server can generate the individual public-law draws and sum them when no closed-form sampler is known.

If conditional update/noise laws and A's information match in every round, the peer trajectory laws match. The server sees different information by design. A placement advantage in this control indicates a mismatch needing diagnosis, not an expected scientific gain. Privately selected profiles require observer-conditioned joint matching of selection and draw laws; a central arm substituting a public average profile is not an equivalent control.

**Legacy boundary.** Current inspected code uses `num-examples` weights and an active sample-count denominator, with server standard deviation calculated using client count. Historical OUT removes the slot and renormalizes. Re-run only as a separately labeled protocol if later authorized; retain existing results as historical artifacts. The [frontier](../../reports/auc_frontier.html) uses folded paired checkpoint concordance, not conventional independent-trial ROC-AUC. Its values cannot set the pilot's leakage or power thresholds directly.

## 3. Privately selected finite profiles: launch gates

Consider a bank of K publicly specified **joint clipping/noise** profiles with a local classifier `j=f(D_i)`. A possible selector is K-ary randomized response:

\[
P(J=j)=\frac{e^{\varepsilon_s}}{e^{\varepsilon_s}+K-1},\quad
P(J=k\ne j)=\frac1{e^{\varepsilon_s}+K-1}.
\]

This is the **main candidate construction route**, not a final selected mechanism. Profiles must jointly specify a public clipping set/operation and a valid accompanying noise law; arbitrary data-dependent covariance after unchanged Euclidean clipping is not the main proposal. An initial feasibility default is `K=2`, once-per-training selection, `epsilon_s=0.5`. The empty target slot evaluates an explicitly defined `f(empty)` and runs the same randomized selector. It does not suppress selection or send a distinguishable deterministic marker.

Every fixed selected profile must independently satisfy the required update-release bound for every relevant neighboring input, including the empty slot. A conservative hypothetical release of J costs epsilon_s even when J stays local; then add/combine the uniformly valid conditional update cost. Per-round reselection would incur additional selector cost and is not the initial default. Raw selector inputs, scores and exact statistics must not leak through messages or logs. Selecting a profile privately does not fix an invalid per-profile law.

Before adding Q-selected, document: whole-client adjacency of the selector; K profile probabilities; empty behavior; per-profile mean/noise law; estimator/selection cost; composition; and the complete peer-observable mixture. Include a public fixed-profile control with the same bank and a data-independent profile-selection control to determine whether benefit comes from private local information rather than a mixture or changed noise budget.

## 4. Mathematical and numerical feasibility gates

1. Lock the detailed threat/metadata contract and release equation. The owner confirmed dataset-contribution secrecy; numerical protocol, observer details and experiments still need review.
2. Derive privacy bounds for the **aggregate law remaining unknown to the peer** and all selected profiles. If using a stronger upload-level guarantee, state it explicitly. A variance plot or sampled loss histogram is not an accountant.
3. For public non-Gaussian densities, bound relevant shift likelihood ratios/excess mass or Rényi divergence, tails, support, discretization and sampling approximation. Uncontrolled quadrature or Monte Carlo is exploratory evidence, not a certified bound. Include both neighboring directions. If profile-dependent laws differ across neighbors, analyze both laws, not a fixed-density shift alone.
4. Verify zero-slot behavior and fixed weights/noise-before-or-after-clipping. Do not clip after adding noise unless the actual composed mechanism and matched control include that operation.
5. Account multi-round releases and selector cost under a stated total budget. A planning comparison point is whole-client `(epsilon,delta)=(8,10^-6)` over the full transcript, with epsilon_s reserved when applicable. This is neither a recommendation of acceptable privacy nor evidence that useful learning at that budget is feasible. Pure-DP profiles use their own compatible composition; do not relabel an RDP/approximate guarantee as pure DP.
6. Start with the root proposal's two-dimensional analytic release and two joint profiles; dimensions `d=4,16,64` are later feasibility extensions before expensive full-network training. These diagnose shape/sampling behavior, not image-model CIA performance. A projected real-model experiment must freeze unreleased/nonprotected parameters or protect their updates too; releasing deterministic complementary private updates would defeat the restricted-direction proof.
7. Estimate the induced perturbation magnitude and compute cost before deep training. If a valid budget collapses utility, record that result and revise the scientific question/candidate. Do not lower noise without relabeling the formal budget.

## 5. Deferred real-model numerical defaults

Deferred real-model engineering task, after the analytic/profile gates and an explicit execution instruction: CIFAR-10, existing `cifar10_cnn`/Adam implementation, `N=8`, target canonical ID 0, peer canonical ID 1, `10` global rounds, `1` local epoch, batch size `64`, learning rate `0.001`, clipping norm `C=5`, and `500` training records per nonempty slot. These are economical starting choices; they differ from the 20-round/5-epoch frontier and do not establish faithful reproduction or adequate attack sensitivity. Keep the standard test set for utility only. All slots communicate at every round; no sampling amplification or dropout guarantee is claimed.

Use a balanced background-client population and a preregistered target-distribution stress: 50% class 0, remaining 50% spread over the other nine classes. Add an all-balanced target diagnostic. Target skew is imposed in both candidate worlds and does not depend on the membership label. This stress model is synthetic and must be labeled as such. Controlled quantity/label/domain heterogeneity and larger N are later robustness stages, not hidden additions to the first pilot.

Keep an overlapping shadow set of 10% of the candidate target dataset (50 records under these defaults) for a strong-knowledge primary setting. It is available to A in both worlds. Evaluate disjoint same-domain shadow data as a separately declared access setting; do not change access to make a mechanism look safer.

Canonical IDs must determine diagnostic random-stream coupling. Current active-ID seeding changes surviving clients' streams upon removal; the proposed fixed roster avoids that remapping, and any later removal implementation must use canonical IDs. Shared seed values are private audit controls, not attacker knowledge of honest perturbation draws.

## 6. Exact split, independent unit and attack procedure

**Unit:** one independently generated federation/target realization, one randomized membership label, and its complete training transcript. Every checkpoint and attack feature from that transcript is one grouped example. If mechanisms share a realization for paired comparison, all arms of that realization remain one inference cluster. A diagnostic counterfactual IN/OUT pair stays in one split and is never two independent trials.

Proposed CIFAR construction uses 5,000 disjoint public design records and three disjoint 15,000-record banks for attack development, calibration and final confirmation, stratified once using an audit-only partition seed. No private training/target records cross these banks. Within a bank, independently draw each federation with disjoint slot datasets; trial draws may overlap between realizations. Independence is conditional on this fixed finite corpus and independently drawn trial RNGs; do not claim independent sampled human organizations. Reused persistent target identities or shared trained models require additional clustering.

Allocate membership labels in a randomized, balanced schedule independent of other trial randomness. Provisional counts **per arm**: development `12 IN + 12 OUT`, calibration `8 IN + 8 OUT`, untouched confirmation initially planned as `40 IN + 40 OUT`. These counts are starting compute allocations, not adequate power claims. Confirmation count is finalized from development/calibration variability before any confirmation models/scores are examined. Three engineering paired frames per arm belong to a fourth debugging-only group.

Proposed first four core arms are V, G-local, G-central-peer, and Q-selected **only if its joint profile bank and selector pass all gates**. If that construction is not ready, do not fill its slot with an unanalyzed private adaptation; public-law checks remain the preparatory work. Public fixed-profile, data-independent selector, common-Euclidean/shaped-Gaussian negative-control and non-Gaussian shape arms are required targeted ablations but need their own compute allocation before a construction-gain claim. Across the four core arms, development plus calibration totals 160 trained trajectories; three diagnostic IN/OUT pairs per arm add 24. Set an initial engineering/development ceiling of 200 full training trajectories. Estimate actual hardware runtime/memory and stop for review at that ceiling; unused allowance is not permission to run. The 320 planned confirmation trajectories, and additional ablation trajectories, are separately gated by power and compute review. An incomplete ablation set cannot establish that private profile choice caused an observed gain.

Train defense-aware attacks on development only: (a) scalar negative target-shadow loss, (b) a regularized whole-transcript loss/own-update classifier or likelihood model, and (c) a covariance/magnitude-aware transcript classifier. Use only observer-available features. A cannot see hidden noise residual samples directly; it can construct residual features only from known weights, its own upload, and visible model increments. Tuning occurs through grouped inner splits of development. Calibration fixes direction, thresholds, feature normalization and attack selection; it does not retune the noise mechanism against confirmation data.

Release one score per transcript. Final conventional ROC-AUC compares independent labeled trial scores, not IN/OUT checkpoints. A below-chance result is reported with its frozen direction; do not fold final-test AUC to select direction. Report all prespecified attacks. If using maximum test AUC across the frozen family as an endpoint, provide simultaneous uncertainty and acknowledge maximization bias; do not interpret its null expectation as exactly 0.5.

## 7. Effect criteria, uncertainty and escalation

Planning primary utility is IN-model test accuracy; report OUT utility separately. A candidate is worth scaling only if it improves leakage at comparable useful accuracy, or improves accuracy at comparable measured leakage, with independent uncertainty. Proposed practical margins: `0.02` AUC reduction and at most `0.03` absolute accuracy loss versus the relevant fixed-noise reference. These are reviewable scientific effect sizes, not current evidence or guaranteed attainable targets.

For a claim of measured near-chance behavior, require simultaneous 95% intervals for the frozen attack family inside `[0.45,0.55]`; an interval merely crossing 0.5 is inconclusive. This establishes equivalence only for that attack family/population. It does not establish a universal privacy guarantee. A superiority claim needs a preregistered contrast interval supporting the selected effect margin; “no significant difference” does not establish matching leakage. Placement controls should have equivalent laws analytically; finite numerical disagreements trigger a diagnostic check.

Use grouped/hierarchical uncertainty over independent federation realizations, preserving shared mechanism arms and counterfactual siblings. Before allocating confirmation, simulate the intended contrasts using development/calibration variability to choose sample size for target precision/power (planning targets: 95% interval and 80% power for the agreed effect). Include attack-family multiplicity, dependencies, and utility/leakage joint comparisons. If the required count exceeds the budget, label the pilot underpowered and limit conclusions to feasibility. Do not count 10 checkpoints as ten trials or assume 40+40 is sufficient.

Low-FPR reporting starts at a prespecified FPR of 0.1 with uncertainty if negative-trial counts support it. FPR 0.01/0.001 claims need substantially more independent OUT trials; the initial allocations do not resolve them credibly. Report count-based resolution and intervals instead of interpolating a dramatic low-FPR headline.

Confirmation is evaluated once after the mechanism, sampler, accountant, attacks, split manifest, endpoints and sample size are locked. No noise search, score-sign selection, profile redesign, or extra sampling based on confirmation outcomes. A revised design gets a new version and fresh confirmation bank/realizations; prior test data becomes development evidence. Do not declare an experiment finished: the owner makes that decision.

## 8. Agent pickup deliverables before launch

- Owner-confirmed dataset-contribution secret recorded; detailed observer/metadata contract and privacy target reviewed.
- Fully specified public law(s), finite profile bank if applicable, empty-slot selector/draw procedure, and conditional release equation.
- Peer-residual density/calibration/composition analysis, approximation errors, and private selector cost.
- Exact matched-control equations and sampler equivalence checks; historical protocol labels kept separate.
- Immutable split/role/seed manifest with private oracle fields excluded from the declared attack view.
- Independent-trial attack recipe, uncertainty/power plan, endpoints/effect margins, compute allocation and no-reuse rule.

This proposal makes the next research step concrete. Dataset-contribution secrecy is confirmed; complete profile/law choice, detailed pilot selection, implementation and execution remain pending.
