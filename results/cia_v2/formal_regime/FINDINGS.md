# B4 findings: formal regime with frozen public features and a small shared head

Roadmap v2, package B4 (Agent C), 2026-10-17. Protocol: [PROTOCOL.md](PROTOCOL.md) @ `44fe21f9` (committed before
any run). Analysis: [analysis.json](analysis.json), built by `python -m experiments.frozen_head.runner analyze` from
the run JSONs in `results/` (1,440 tuning runs, 160 evaluation runs). Calculation tables:
`research/cia_roadmap_v2/theory/tables/formal_regime_{nsr,predicted_accuracy,crowd}.csv`, from
`research/cia_roadmap_v2/theory/formal_regime_calcs.py`. Compute: 1.76 CPU-hours of run time (feature extraction and the calculation stage not timed), no GPU.

## Gate

**G-B4: fail.** No setting with `mu_total <= 8` reaches 5 pp above the public-only model; every setting is listed in
the evaluation table below.

- The best DP setting (`n = 480`, `T = 20`, `mu_total = 8`) is 3.49 pp above the public-only model on the server test
  set, 95% interval [2.37, 4.64].
- This is outside the protocol's 4-6 pp extension band, so no extension was run.
- Per the roadmap, the paper statement is: **"no non-vacuous client-level regime at this scale"**. That statement has
  to carry the nuance in "What the result does and does not say" below: there is a non-vacuous regime, but it buys
  about 2 pp, not 5.

## Setting (from the protocol)

- **Model:** a linear head (`d = 5,130`) on frozen ResNet-18 ImageNet features of EuroSAT.
- **Mechanism:** client-level DP-FedAvg with full participation, equal weights and a public slot count `n`; exact
  `mu_total = sqrt(T) / z`.
- **Federations:**
  - `n = 48`: the seed-42 Dirichlet 0.3 layout.
  - `n = 480`: each of those clients split into 10 pseudo-clients.
- **Baseline:** the public-only head, logistic regression on the 1,350-image public set with L2 chosen by public
  cross-validation (inverse L2 0.03). It also initializes every federated run.
- **Tuning:** on layout 43 by pooled client local-test accuracy (36 grid points x 2 seeds per setting).
- **Evaluation:** layout 42, `R = 8` replicates with independent `client_seed` and `noise_seed` per replicate.
- **Metric:** final server-test accuracy (1,350 images), as a paired difference from public-only; two-way bootstrap
  over replicates and test images.

## Evaluation results (server test set; public-only head 0.9044)

| `n` | `T` | `mu_total` | `epsilon` at 1e-5 | DP head accuracy (sd over replicates) | Gain over public-only, pp [95% CI] | Per-client local-test p10 |
|---|---|---|---|---|---|---|
| 48 | 10 | no noise | n/a | 0.9389 (0.0036) | 3.44 [2.19, 4.70] | 0.901 |
| 48 | 10 | 1 | 4.38 | 0.9111 (0.0023) | 0.67 [-0.10, 1.48] | 0.858 |
| 48 | 10 | 2 | 10.00 | 0.9204 (0.0018) | 1.59 [0.85, 2.43] | 0.869 |
| 48 | 10 | 4 | 24.38 | 0.9281 (0.0035) | 2.36 [1.32, 3.42] | 0.877 |
| 48 | 10 | 8 | 65.32 | 0.9354 (0.0045) | 3.09 [1.95, 4.20] | 0.892 |
| 48 | 20 | no noise | n/a | 0.9452 (0.0007) | 4.07 [2.66, 5.42] | 0.907 |
| 48 | 20 | 1 | 4.38 | 0.9012 (0.0067) | -0.32 [-1.37, 0.74] | 0.829 |
| 48 | 20 | 2 | 10.00 | 0.9213 (0.0037) | 1.69 [0.79, 2.61] | 0.871 |
| 48 | 20 | 4 | 24.38 | 0.9304 (0.0047) | 2.59 [1.50, 3.69] | 0.882 |
| 48 | 20 | 8 | 65.32 | 0.9375 (0.0040) | 3.31 [2.15, 4.41] | 0.893 |
| 480 | 10 | no noise | n/a | 0.9309 (0.0008) | 2.65 [1.60, 3.76] | 0.800 |
| 480 | 10 | 1 | 4.38 | 0.9252 (0.0017) | 2.07 [1.20, 3.04] | 0.794 |
| 480 | 10 | 2 | 10.00 | 0.9271 (0.0018) | 2.27 [1.29, 3.27] | 0.800 |
| 480 | 10 | 4 | 24.38 | 0.9269 (0.0012) | 2.24 [1.26, 3.22] | 0.800 |
| 480 | 10 | 8 | 65.32 | 0.9309 (0.0012) | 2.65 [1.63, 3.70] | 0.800 |
| 480 | 20 | no noise | n/a | 0.9341 (0.0009) | 2.96 [1.90, 4.06] | 0.800 |
| 480 | 20 | 1 | 4.38 | 0.9284 (0.0027) | 2.40 [1.39, 3.40] | 0.797 |
| 480 | 20 | 2 | 10.00 | 0.9334 (0.0030) | 2.90 [1.81, 4.06] | 0.800 |
| 480 | 20 | 4 | 24.38 | 0.9339 (0.0024) | 2.94 [1.83, 4.08] | 0.800 |
| 480 | 20 | 8 | 65.32 | 0.9394 (0.0024) | 3.49 [2.37, 4.64] | 0.800 |

- **Per-client minimum.** At `n = 48` the minimum local-test accuracy is 0.80 in most settings (0.75 and 0.79 at
  `T = 20`, `mu` = 1 and 2). At `n = 480` it is 0 in every setting, because many local test sets hold 1-3 records, so the 10th
  percentile is shown instead.
- **Interval widths.** The intervals are 1.6-2.8 pp wide (half-widths 0.8-1.4 pp), in line with the protocol's
  estimate of about ±1.4 pp. Test images dominate them: replicate standard deviations are 0.07-0.67 pp.

## What the result does and does not say

1. **The gap is mostly the ceiling, not the noise.**
   - A non-private centralized linear probe on all clients' training data reaches at most 0.9548 on the server test set,
     the best of three L2 values chosen on that test set, so it is optimistic (`STATE.md` Log; PROTOCOL Section 8).
     The headroom over public-only is therefore at most 5.04 pp in this setup, whatever the privacy mechanism.
   - Even without noise, 10-20 rounds of FedAvg on the non-IID clients reach 2.65-4.07 pp.
   - So the 5 pp gate is out of reach mainly because 1,350 public images already give a strong linear probe on ImageNet
     features. The v1 stacked-head lesson repeats: the public-only model is the real competitor.

2. **A non-vacuous guarantee does buy a small, significant gain at 480 clients.**
   - At `mu_total = 1` (`epsilon` 4.38 at `delta = 1e-5`; any membership attack has AUC at most 0.760 and TPR at 5% FPR
     at most 0.26), the 480-client federation gains 2.07 [1.20, 3.04] pp (`T = 10`) and 2.40 [1.39, 3.40] pp (`T = 20`)
     over public-only. Both intervals exclude 0.
   - The cost against no noise is 0.6 pp.
   - At 48 clients, `mu = 1` gives nothing (0.67 [-0.10, 1.48] and -0.32 [-1.37, 0.74] pp).
   - With 480 clients the per-coordinate noise on the average is 10 times smaller, which is what the noise-to-signal
     table predicts: 0.47-0.67 at 480 clients against 4.7-6.7 at 48 (`formal_regime_nsr.csv`).

3. **The upper part of the gate's range carries no protection.** `mu = 4` bounds the attack AUC only at 0.998 and TPR
   at 5% FPR at 0.99, and `mu = 8` at 1.000 (`formal_regime_nsr.csv`). Even a pass at `mu = 8` would not have shown a
   meaningful formal regime. Among the settings with a meaningful bound (`mu <= 2`), the best gain is 2.90 [1.81, 4.06]
   pp (`n = 480`, `T = 20`, `mu = 2`, AUC bound 0.921).

4. **The calculation stage was pessimistic but pointed the right way.**
   - The untuned first-order prediction gave at most 1.6 pp, and a destroyed head at `n = 48`, `mu <= 4`.
   - Tuning mainly found smaller clipping norms and, at `n = 480`, larger server steps, which recovered 1-2 pp over the
     prediction.

## Crowd bound at head dimensions (the D6 question)

WP3's Propositions 1-3 (`research/cia_review_2026-10/theory/f_cmip.md`) evaluated per round on noise-free head
trajectories.

**Inputs:**

- Calculation config, evaluation layout 42.
- Crowd covariance from the other clients' clipped updates in that round, applied with the Woodbury identity.
- Statistic direction: the target's local-test loss gradient.
- Rounds composed as independent.
- Targets: all 48 clients at `n = 48`; at `n = 480`, the top decile (48) plus 48 spread by size.

**Results** (`formal_regime_crowd.csv`), as a ratio to the GDP bound:

| `n` | `T` | `mu_total` | Full release, top decile (median) | Full release, all (median) | Loss statistic `rho`, top decile (median) |
|---|---|---|---|---|---|
| 48 | 10 | 1 | 0.985 | 0.778 | 0.974 |
| 48 | 10 | 2 | 0.958 | 0.746 | 0.907 |
| 48 | 10 | 8 | 0.891 | 0.681 | 0.474 |
| 48 | 20 | 1 | 0.952 | 0.731 | 0.988 |
| 480 | 10 | 1 | 0.924 | 0.889 | 0.892 |
| 480 | 10 | 2 | 0.849 | 0.806 | 0.703 |
| 480 | 10 | 8 | 0.706 | 0.673 | 0.241 |
| 480 | 20 | 1 | 0.954 | 0.909 | 0.946 |
| 480 | 20 | 2 | 0.892 | 0.838 | 0.826 |

The full table (`mu` = 4 included, maxima and minima) is in the CSV.

**Reading:**

- **Large clients:** where the guarantee means something (`mu <= 2`), an attacker who does not know the other
  clients is at most 15% weaker than the GDP worst case against the large clients (ratio 0.85-0.99). Against the loss
  statistic it is at most 30% weaker (`rho` 0.70-0.99).
- **"All" medians:** the lower ratios for all clients (0.65-0.91) come mostly from clients whose update norm is below
  `C`, since Proposition 1 scales with `||u_i|| / C`. They are not a crowd effect.
- **Weak guarantees:** the crowd helps substantially only at `mu = 8`, where the bound is vacuous anyway (loss `rho`
  0.24-0.74).
- **D6 answer:** WP3's full-model conclusion carries over to `d ~ n`. The crowd term matters only while the DP noise
  is small against the crowd's own variation, which is exactly the regime without a useful guarantee. D6 does not
  give a materially smaller bound in the formal regime of this setup.

**Limits:**

- Gaussian crowd model with the population known to the attacker.
- Noise-free trajectories, whereas the noise changes the trajectory.
- The calculation config, not the tuned one.
- Independent rounds (Proposition 3.2: without that assumption the worst case is GDP).
- Sample covariance from 47 or 479 other clients.

## Caveats

- **Tuning:** it used private records of a separate layout (layout 43, the same pool of training images) and is not
  counted in the privacy budget, as in v1. Every selected DP configuration lies on at least one grid edge, most often
  the smaller clip multiplier (0.5) and, at `n = 480`, the largest server step (10). A wider grid might add some
  accuracy. It cannot exceed the optimistic centralized ceiling (5.04 pp), and the no-noise references reach at most
  4.07 pp.
- **Fixed choices:**
  - One extractor (ResNet-18 ImageNet). A stronger extractor would likely raise the public-only model as well.
  - One public set size (the setup's 1,350 images).
  - One dataset (EuroSAT).
  - One partition (Dirichlet 0.3).
- **Simulator:** an in-process NumPy simulator of the mechanism (unit-tested against the closed forms and the
  frontier layouts), not the Flower message path. A formal-regime comparison opened by the owner would need the real
  path.
- **No attack measurement here.** The secondary empirical audit in the protocol needs Agent A's attackers after G-M0
  and was not run. The privacy statements above are the formal bounds.

## Options for the owner (not decided here)

1. Accept the fail and record the statement, with the nuance from points 2 and 3 of "What the result does and does
   not say": about 2 pp at `mu = 1` with 480 clients.
2. Probe where the formal regime would pay off. Shrinking the public set (as in the v1 stacked-head sweep) would
   widen the headroom, but it changes the comparison and needs a new protocol. Client subsampling with Agent B's B1
   accountant would amplify `mu` at `n = 480`.
3. Run the secondary empirical audit once G-M0 passes. It would show how far below the `mu = 1` bound the realistic
   attacker sits in this regime, which is cheap here (head-only runs).
