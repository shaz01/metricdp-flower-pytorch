# Comparison with the paper's server-side mechanisms on the same base, data and releases

Owner (2026-10-09): "add the missing comparison". Every earlier utility number was measured against public-only models; none was measured against the original server-side mechanisms. This protocol adds that, with rules fixed before any baseline cell touches evaluation data.

## Mechanisms (all one-shot releases from the same public base, same eight clients' records, same public validation gate, same held-out evaluation)

- **OURS**: the stacked construction (existing cells; frozen q.65/q.80 configuration; V0 gate).
- **GDP**: Flower's fixed-clipping global DP through the repository's own `make_strategy("fedavg", "global-dp", ...)`: each client trains the linear head locally on its records, the server clips the update to norm C, averages, adds Gaussian noise with std z*C/n (n = 8).
- **MDP**: the paper's metric-privacy calibration through the same repository code (`MetricPrivacyServerSideFixedClipping`): identical to GDP but the noise multiplier is z = nm / d with d the maximum pairwise client-model distance (mean over layers; the federated model is the two head arrays).
- **VAN**: vanilla FedAvg of the local head updates, no clipping, no noise: the non-private reference.

Baseline clients use the same public body (frozen), start from the same base head, and run `local-steps` full-batch softmax-CE gradient steps with step size `local-lr` on their records. The public validation gate then chooses a multiplier m in (0, 1/8, 1/4, 1/2, 1, 2, 4, 8) on the aggregated update (post-processing, the same gate as ours). The ungated plain release (m = 1, the paper as published) is reported too.

## Matching privacy

- **Calibrated worst-case risk (GDP vs OURS).** For a client whose clipped update saturates C, the optimal known-alternative attacker on the server-side Gaussian noise has AUC Phi(1/(sqrt(2) z)); GDP therefore uses z_r = 1 / (sqrt(2) Phi^-1(r)) so its calibrated risk equals ours, r in {.55, .65, .80, .90, .95}. For OURS the same calibration holds (earlier protocols).
- **Realized oracle AUC (MDP).** MDP has no such calibration (noise depends on d). Its nm_r is chosen on development data so that the mean realized known-alternative AUC, Phi(min(||update||, C) / n / (sqrt(2) sigma)) averaged over clients and releases, equals r. The realized oracle AUC of every baseline run is recorded and reported.
- **Practical attacker.** The model-only calibrated own-records attack of the frontier protocol is applied identically to every mechanism (IN world: all clients; OUT world: the target sends a dummy: ours a noise share only, baselines the base head plus a 1e-6 perturbation because Flower's clipping rejects an exactly zero update).

## Fair tuning (development only; Fashion-MNIST classes 0-3, public sets 0-4, cohort A, 32 images, 32 independent releases per cell)

Fashion-MNIST 0-3 is the dataset on which OURS was tuned. GDP gets a CNN-head grid: local-lr {0.01, 0.03, 0.1} x local-steps {5, 20} x clip norm C {0.01, 0.03, 0.1, 0.3, 1.0} = 30 configurations per risk, r in {.65, .80} (OURS had 360 configurations per risk in its freeze). For each r the configuration with the lowest pooled mean PUBLIC-VALIDATION CE of the gated release is frozen; no evaluation image is used. Risks .55, .90 and .95 reuse the q.65 configuration (as OURS reuses its q.65 configuration at every frontier risk). MDP reuses GDP's frozen local training and C and only calibrates nm_r (from the recorded update norms and distances of a development run). VAN's local-lr/steps are chosen the same way (6 configurations).

**Addendum (before any evaluation cell was run).** The first freeze selected local-steps = 20, the largest grid value, for every baseline (and local-lr = 0.1, the largest, for GDP at q.80), i.e. a grid-edge selection that would handicap the baseline. The grid is therefore extended to local-lr {0.01, 0.03, 0.1, 0.3} x local-steps {5, 20, 50} (60 configurations per risk; vanilla 12) and the freeze is redone on the same development cells; if an edge is still selected it is reported as such and not extended further.

## Part 1: utility at matched risk (fresh data)

Cells: exactly the confirmation cells of the gate protocol (KMNIST 0-3 and 4-7 sets 30-59; MNIST 0-3, MNIST 4-7, Fashion-MNIST 4-7 sets 10-19; cohort A; held-out CE/accuracy on the fixed 2,000-image test-split subset; 256 independent releases), risks .65 and .80, mechanisms GDP and MDP (VAN once per set). None of these cells or images informed any baseline hyperparameter.

Per group (task, risk) and per public set, paired difference D = gain(OURS) - gain(baseline) in held-out CE over the control (gated, V0). A group is "ours better" if mean D > 0 and the bootstrap 95% interval of the mean (resampling public sets, 2,000 draws) excludes 0, "baseline better" if mean D < 0 with the interval excluding 0, otherwise "no clear difference".

**Pre-stated verdict per baseline over the 10 groups:** "ours better" iff at least 8 groups are "ours better"; "baseline better" iff at least 8 are "baseline better"; otherwise "mixed". Also reported: gains as a fraction of the non-private VAN gain, the ungated (m = 1) comparison, accuracy deltas, and the realized oracle AUC of each baseline group.

## Part 2: practical-attack frontier (KMNIST, 32 images, sets 0-4, targets 0-3, risks .55, .65, .80, .90, .95, noise-free = VAN)

Same grid as the frontier protocol for the baselines. For each task and baseline, a piecewise-linear utility-versus-practical-AUC curve is built from the baseline's five risk points plus its noise-free point; **claim "ours above the baseline curve" iff at least 4 of OURS' 5 risk points lie above that curve in both tasks.** Practical-AUC differences smaller than the bootstrap interval are reported as such.

## Limits

One-shot head update only (the paper's mechanism is a multi-round full-model protocol; this tests the NOISE CALIBRATION and CLIPPING design in a matched setting, not the paper's full training pipeline); baselines are tuned on a different dataset than they are evaluated on, exactly as OURS was; the baseline grid is smaller than OURS' (disclosed above); MDP's nm is calibrated to realized oracle AUC rather than to a formal guarantee because none exists; KMNIST/MNIST/Fashion-MNIST 4-class tasks, small CNN body, Gaussian noise; the practical attacker is weak (earlier findings) so Part 2 differences may be small.
