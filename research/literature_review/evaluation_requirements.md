# Evaluation requirements for a client-specific noise mechanism

Date: 2026-10-01. This document distinguishes verified repository observations from proposed evaluation requirements and mathematical deductions. It is an agent handoff and planning document; it does not declare an experiment finished or a technique novel.

## 1. Verified evidence baseline

Sources: [project audit](../project_evidence_audit.md), [scorer](../../experiments/contest_at_scale/auc_target_sweep/score_stage.py), [search controller](../../experiments/contest_at_scale/auc_target_sweep/auc_target_search.py), [frontier generator](../../experiments/contest_at_scale/auc_target_sweep/reports/build_auc_frontier.py), [raw states](../../experiments/contest_at_scale/auc_target_sweep/results/), and [initial paper](../../papers/Initial-Paper.pdf), Section 7.4.2/Table 13.

An independent arithmetic check on 2026-10-01 recomputed all 110 recorded anchor/search/confirmation entries from `runs/cia.json` and each trajectory's final-round server accuracy. All matched their state values to 1e-12; IN and OUT checkpoint indices matched. This verifies the recorded arithmetic, not the training or statistical interpretation.

For negative clean-shadow loss scores, the scorer estimates

\[
q=K^{-1}\sum_t\{1[s_t^{IN}>s_t^{OUT}]+\tfrac12 1[s_t^{IN}=s_t^{OUT}]\},\qquad q^*=\max(q,1-q).
\]

This is **folded paired checkpoint concordance**, conditional on a target client, training seed, pairing/coupling of random streams, shadow sample, and selected checkpoints. It asks how often IN has the higher score at the *same* checkpoint. It does not estimate ordinary ROC-AUC over independent client-participation trials. Checkpoints share trained parameters and shadow records, so their count is not an independent trial count.

The initial paper instead computes pooled ROC-AUC over the 20 IN and 20 OUT checkpoint scores: comparisons can cross training rounds. Its Table 13 reports vanilla 0.890, global-DP 0.397, and metric calibration 0.493, with respective 95% intervals (0.767, 0.977), (0.224, 0.585), and (0.296, 0.689). This is a different estimand. Its inference that intervals overlapping 0.5 confirm inability to infer participation is too strong: failure to reject a difference from chance is not evidence of near-chance equivalence. A reversed global-DP loss ranking may also be useful to an attacker; its sign must be calibrated independently before evaluation.

The frontier's seed 42 selects the noise level; seeds 43/44 supply exploratory confirmation. Averaging all three includes the selection seed. The plotting script does not compute a validated Pareto boundary. Utility averages the IN and OUT final-round accuracies. These are observations about the current protocol, not defects in the underlying stored arithmetic.

## 2. Define the protected event and observable transcript first

**Recommendation.** Primary CIA target: whether a specified client's whole dataset contributed to a training transcript. Define add/remove adjacency, permitted public information, population of targets, federation construction, sampling policy, and rounds before choosing noise. Evaluate replacement adjacency separately when maintaining a fixed cohort by substituting a background client; it answers a different question.

Specify separate observers: (a) peer client observing released global models and its own updates, (b) observer of a final model, (c) server observing individual uploads, (d) server observing only secure aggregates, and (e) colluding peers/server with their own messages and randomness. The original trusted-server CIA question is primarily (a); protection against (c) is an additional requirement, not an automatic benefit of moving noise to devices.

The complete observable transcript includes messages, global checkpoints, cohort counts, identities if visible, aggregation weights, sampling/dropout flags, release schedules, clipping thresholds, estimated covariance/scale parameters, and protocol failures. If the adversary directly sees the target's participation identity, content perturbation cannot hide that event. Experiments that hide identities and counts should say so explicitly; the server may still know them, and a data-content guarantee need not hide network enrollment.

## 3. Measure ordinary CIA ROC-AUC on independent trials

**Recommendation.** A trial is one observable training transcript and one candidate target with an independently generated membership label. Generate many independently randomized federation/training realizations from a specified population. Keep evaluation records and target-shadow assumptions fixed by protocol. Compute one prespecified attack score per trial: final-round loss for the simplest attack, or a learned transcript score for a stronger multi-round attack. Do not count each checkpoint as another client-participation trial.

For an independently selected IN trial and OUT trial, ordinary ROC-AUC estimates

\[
A=P(S_{IN}>S_{OUT})+\tfrac12 P(S_{IN}=S_{OUT}),
\]

using all cross-class comparisons of the held-out trial scores. This remains a property of the specified attacker and trial population, not a universal guarantee. If target populations differ in score scale, report conditional target/domain AUCs or use an independently trained conditional attacker; arbitrary pooling can measure target identity or difficulty rather than participation.

Split **whole federation realizations and target identities** into attack-development/calibration and final-test groups according to the intended generalization claim. An IN/OUT sibling pair and all of its checkpoints belong to the same group. Shared targets, partitions, initialization seeds, or shadow samples can couple trials; track these dependencies and keep coupled units together. An attacker trained on the same target identity can be a deliberate target-specific threat setting, but must have independent development and final-test realizations.

The attacker sees one real transcript, not both counterfactual IN and OUT versions. Paired counterfactual training remains useful as an auxiliary influence experiment or variance-reduction device. Do not give the attack the unavailable counterpart, membership label, matching identifier, or counterpart-derived score normalization. Freeze score sign, thresholds, checkpoint selection, model architecture, and hyperparameters on development data. Never choose the stronger sign by taking `max(A, 1-A)` on final-test labels; report signed test AUC of the frozen attacker and explain reversals.

Evaluate clean and noisy shadow access separately, including overlap vs independently acquired target-domain samples and shadow sample size. Include a defense-aware attacker retrained on each mechanism, loss-based baseline, own-update-aware peer attack, and a trajectory classifier when multiple checkpoints are observable. Failure of one fixed loss score is not failure of all CIA.

## 4. Uncertainty, search, and practical protection criteria

**Recommendation.** Independent inference units are independent federation/training realizations; resample or model whole trajectories and their grouped targets. If targets share a model, they share a cluster. If IN/OUT realizations are paired, preserve pair structure in uncertainty calculations. A simple checkpoint bootstrap or binomial interval on the 11/20 paired comparisons would overstate available independence. Repeated target identities require a stated conditional inference target or hierarchical/cluster analysis.

Preregister the attack family, primary utility endpoint, primary leakage endpoint, meaningful near-chance tolerance, confidence level, selection procedure, and test-set reuse rules. Assess practical protection with an upper confidence bound on leakage or a defined equivalence test; an interval merely containing 0.5 is insufficient. Include TPR at prespecified low FPRs when enough independent OUT trials support them, plus thresholded accuracy/advantage under a declared membership prior. AUC alone does not characterize rare-event operational risk.

Choose sample size through a pilot of independent realizations and simulation/resampling under the planned analysis. Specify the precision or detectable superiority margin, clustering, number of targets, multiple mechanisms, and compute budget. Do not assert that three seeds, 20 checkpoints, or an arbitrary large number provides adequate power. Low-FPR claims require enough negative trials to resolve that FPR with meaningful uncertainty.

Search noise/covariance parameters on development realizations only, lock candidates, and evaluate on new realizations. Report all searched settings and failure/utility-collapse outcomes. Use matched-leakage comparisons (utility with comparable attack strength) and matched-utility comparisons (leakage with comparable usefulness), accompanied by joint uncertainty. Also compare matched formal privacy budgets only when all mechanisms have valid guarantees under compatible adjacency and accounting.

## 5. Client-side noise changes the observation model and covariance

**Deduction, assuming independent, centered client noise and fixed aggregation weights.** If client i sends `u_i + z_i`, `Cov(z_i)=Sigma_i`, then

\[
\operatorname{Cov}(\textstyle\sum_i a_i z_i)=\sum_i a_i^2\Sigma_i.
\]

With equal-weight averaging and identical covariance, aggregate covariance is `Sigma/n`, not `Sigma/n²`. Matching a desired central aggregate standard deviation therefore requires careful local scaling. The conditional covariance seen by a colluding observer changes after its known noise contributions are subtracted. Secure aggregation hides messages, but the guaranteed residual noise depends on honest contributors, dropout, and the protocol's visibility assumptions.

**Deduction.** Removing a client changes both the update mean and the noise law: one `a_i² Sigma_i` term disappears, weights may renormalize, and client-specific covariance estimation may depend on prior released models. Therefore a noise-based CIA can distinguish variance/covariance or higher moments even when target-domain loss is equalized. Matching marginal per-coordinate variance is not sufficient when covariance, non-Gaussian shape, or temporal correlation changes. Correlated client noise needs the full covariance including cross-client terms; cancellation that improves utility can also remove the randomness needed for a privacy guarantee.

**Recommendation.** Audit IN/OUT aggregate covariance and repeated-round noise correlations, then include defense-aware variance/trajectory attacks. Separately compare fixed public scalar noise, client-specific scalar variance, data-dependent covariance, and any coordinated noise. Use matched aggregate noise where feasible as an ablation to isolate placement from distribution construction, while preserving the stated threat model. Do not retune with the true IN/OUT label using a rule unavailable at deployment. Publish any deliberately public participant-count dependence; analyze its metadata disclosure separately.

Client-specific distribution construction is itself part of the privacy mechanism. If `Sigma_i` or its estimate depends on private data, analyze the entire mapping from neighboring client datasets to released messages, including estimator randomness and metadata. Clipping and adding Gaussian draws after an unprotected private covariance estimate does not automatically establish DP. An empirical CIA reduction and a generic DP proof are complementary evidence: attacks test the measured adversary; a valid client-level proof bounds a specified release over all neighboring datasets and adversaries. Record-level DP does not automatically give a useful whole-client bound.

## 6. Utility and imbalance

**Recommendation.** Report IN deployment utility and OUT utility separately, retaining their mean only as a clearly labeled auxiliary measure. For Alzheimer, use balanced accuracy, macro-F1, class-wise recall/precision, confusion matrices, and class prevalence alongside ordinary accuracy and weighted F1. Compare to majority-class and stratified-random baselines computed on the actual test population. The frontier's `1/classes + 0.02` collapse guard cannot diagnose a high-accuracy majority-class predictor. Check patient/group independence if identifiers exist; document when unavailable. Report per-client and worst-group utility so a mechanism cannot conceal targeted utility loss behind its global average.

## 7. Minimal agent pickup checklist

- State client event, adjacency, transcript visibility, collusion/dropout assumptions, and metadata.
- Distinguish central DP, local upload protection, and distributed aggregate protection.
- Freeze distribution-estimation and release rules; specify privacy accounting and residual honest noise.
- Define independent CIA trials and split coupled groups before noise/attack tuning.
- Use a frozen defense-aware attacker, independent score direction, ordinary trial-level ROC-AUC, and grouped uncertainty.
- Prespecify leakage tolerance, utility thresholds, power/precision planning, and matched-comparison criteria.
- Preserve current frontier as exploratory evidence; retain paired checkpoint concordance as a separately named diagnostic.

These are design requirements and open checks, not a selected mechanism or an experiment-completion decision.
