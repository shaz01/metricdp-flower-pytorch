# Provisional research plan grounded in the review

2026-10-01. A new research proposal, not a selected mechanism or authorization to launch experiments. The owner's question is how each client constructs an individual additive-noise distribution that improves participation privacy at useful accuracy. The [frontier](../../results/contest_at_scale/auc_target_sweep/reports/auc_frontier.html) and [evidence audit](../project_evidence_audit.md) define the empirical starting point; the [integrative review](client_specific_noise_review.md) explains the prior-art constraints.

## 1. Resolve closest prior art and define the contribution

Acquire FACP and FedFR-ADP full methods, inspect private-statistic acquisition, adjacency, distribution law, accounting and threat model. Resolve the remaining pending sources in the ledger according to relevance. Do not describe category/source inference as participation inference. Compare against PAC/PAC-Private, Residual-PAC and time-adaptive spending before claiming novelty.

**Deliverable:** a one-page contribution statement specifying which construction, guarantee and evaluation is new relative to each nearest method. If the proposal duplicates a known estimator/sampler, change the contribution to the missing client-participation analysis rather than rename the mechanism.

## 2. Fix the CIA game and success criterion

Keep the initial trusted-server/curious-peer model as the primary game. Define whole-client add/remove adjacency, target distribution knowledge, observer's own data/update/noise, global-model checkpoints, client sampling metadata, weights and denominator. Use dummy/absent-client noise shares or explicitly analyze covariance changes; do not silently fix them only for IN worlds. Server-visible-upload protection is a separate extension.

**Deliverable:** threat specification plus a formal release equation. Establish whether the intended guarantee is worst-case user-level DP or a distributional/empirical participation guarantee. Show why that guarantee applies to the designated observer.

## 3. Prove the relocation control before adapting geometry

Compare fixed server isotropic noise against independent client isotropic shares matched in conditional aggregate law, including peer subtraction. Match clipping, weights, noise position and optimizer. Derive unknown residual covariance for honest clients and any dropout/collusion assumptions. Public weighting or renormalization must be in the sensitivity analysis.

**Deliverable:** an algebraic equivalence statement and a small sampler/aggregation verification when implementation starts. If a measured advantage appears under supposedly equal laws, diagnose the implementation or threat difference before treating it as a discovery.

## 4. Investigate three distribution-construction routes separately

| Route | Construction | Guarantee task | Why start here |
|---|---|---|---|
| A: public or protected-history geometry | Public basis or directions derived from an already protected transcript; shrinkage and isotropic floor | Uniform conditional sensitivity bound and composition | Establish benefit of shaping without introducing raw private estimation |
| B: private local estimation | Per-layer/diagonal or low-rank projected gradient statistics; account the estimator | Whole-client treatment of estimate and upload, including covariance changes | Directly addresses the proposed local construction question |
| C: modeled client distribution | Simulate client perturbations/populations; PAC-style covariance or learned sampler | Explicit population and attacker conditioning; empirical audit of approximation | Allows useful distribution-dependent claims if worst-case utility is prohibitive |

For each route write an executable mathematical recipe: estimator input/sample count; basis; regularization; clipping; covariance-to-sampler factors; minimum eigenvalue; update frequency; and temporal/accounting rule. Avoid full dense covariance initially. Gradient variance, utility curvature and participation sensitivity are candidate signals to compare, not interchangeable definitions.

**Deliverable:** two or three fully specified candidates with computational costs and failure cases. Reject candidates whose guarantee relies on a record-level estimator while claiming whole-client DP, or whose private calibration law is unanalyzed.

## 5. Rebuild inference evaluation around independent trials

Follow [evaluation_requirements.md](evaluation_requirements.md). Preserve the original frontier scorer as a historical diagnostic; report conventional held-out ROC-AUC, attack advantage and low-FPR performance separately. Train defense-aware whole-client IN/OUT shadows. Include loss-based, likelihood-ratio/transcript and covariance-aware attacks. Fit direction/calibration on attack-training data; evaluate independent target clients and retraining realizations. Checkpoint count is not the sample size. Plan power from pilot variability, not an arbitrary seed count or the old 0.55 cutoff.

Start with a small controlled federation where independent repeats are affordable; then use representative utility-surviving frontier settings. Alzheimer imbalance and tiny shadow sets need explicit collapse and uncertainty checks. Add controlled label/domain skew: existing scalable “nonIID” is principally quantity skew. Keep a final untouched confirmation set.

**Deliverable:** preregistered internal evaluation protocol and seed/target split before searching hyperparameters.

## 6. Measure a frontier with matched comparisons and ablations

Controls: vanilla; original metric-inspired server mechanism; fixed client-level server Gaussian; matched local isotropic; schedule-only adaptation; shaped public geometry; private estimated geometry; a nearest implementable prior method. Report actual aggregate/residual covariance and formally valid privacy parameters when available. Compare at matched task utility and, separately, at matched formal privacy target. Measure minority/per-client utility, not only aggregate accuracy.

Ablations: covariance signal, estimator sample size, shrinkage, floor, rank/layer grouping, recalibration frequency, privacy cost of estimation, client heterogeneity, count/weighting and temporal dependence. Adaptive attacks must be retrained for the mechanism. Include compute/memory/communication overhead and failure rates.

**Deliverable:** a Pareto comparison with uncertainty and explanation of gains. A weakened fixed attack, altered total noise or collapsed learning does not satisfy the goal.

## 7. Decide the paper claim from the evidence

Advance if a specified construction gives reproducible held-out protection improvements with useful learning, and its stated guarantee/assumptions withstand scrutiny. Narrow to an empirical CIA defense if only empirical evidence is supported; a distributional guarantee must state its model. A formal user-DP claim requires a complete client-level mechanism/accountant.

If geometry adds no benefit under the relevant sensitivity model, investigate that negative result rather than force an anisotropic mechanism. If estimation cost erases the gain, test public/protected-history routes. If a stronger attack recovers leakage, revise the construction before scaling.

**Deliverable:** a supervisor-reviewable proposal with mechanism equations, nearest-prior-art table, proof obligations and validated pilot protocol. Only the owner decides when an experiment is finished and when its formal report should be written.
