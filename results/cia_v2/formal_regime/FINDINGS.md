# B4 findings: formal regime with frozen public features and a small shared head

Roadmap v2, package B4 (Agent C), 2026-10-17. Protocol: [PROTOCOL.md](PROTOCOL.md) @ `44fe21f9` (committed before
any run). Analysis: [analysis.json](analysis.json), built by `python -m experiments.frozen_head.runner analyze` from
the run JSONs in `results/` (1,440 tuning runs, 160 evaluation runs). Calculation tables:
`research/cia_roadmap_v2/theory/tables/formal_regime_{nsr,predicted_accuracy,crowd}.csv`, from
`research/cia_roadmap_v2/theory/formal_regime_calcs.py`. Compute: 1.76 CPU-hours of run time (feature extraction and the calculation stage not timed), no GPU.

**Owner follow-up:** Amendments 1 and 2 (PROTOCOL.md, @ `3ec476ee` and `6c23c129`, each committed before its runs)
add three things:

- A public-set sweep, with a check on the strength of its baseline.
- An extended-grid sensitivity run.
- The secondary empirical audit, at the full public set and at 20 public images.

They are reported in "Owner follow-up" below. They do not change the verdict above for this setup's 1,350-image public
set. Analyses:

- `analysis_public_sweep.json`, `analysis_extended_grid.json` and `analysis_audit.json`, from the `runner` and `audit`
  `analyze` stages.
- `analysis_audit_public20.json`, from `audit analyze --public-size 20`.
- `analysis_baseline_check.json`, from `python -m experiments.frozen_head.baseline_check`.

The follow-up took 21.3 CPU-hours of run time (record `seconds` summed; the baseline check is not timed). No GPU was
used.

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

## Owner follow-up (Amendments 1 and 2)

The owner asked: "try a smaller public dataset and run the secondary attack check. before we accept the fail we should
analyze carefully". The amendments were written after the original result was known. They are exploration, and each
reading was fixed before its runs.

### Summary

1. **The fail does not depend on the grid.** An extended grid of 150 points per setting raises the best DP gain from
   3.49 to 3.81 pp (`n = 480`, `T = 20`, `mu = 8`, [2.68, 5.05]). The best gain at `mu <= 2` stays at 2.90 pp. The
   no-noise references reach at most 3.90 pp on this grid.
2. **The formal regime pays off when public data is scarce.**
   - Against the protocol's public-only head, the gate passes at 20, 50, 100 and 200 public images (2-20 per class)
     and fails at 500 (`analysis_public_sweep.json`).
   - The pass survives the strongest public-only baseline we could fit (`analysis_baseline_check.json`).
   - At 200 images it holds only through `mu = 4`-`8`, whose bounds are vacuous: no setting with `mu <= 2` is robust
     there.
3. **The DP model's accuracy hardly depends on the public set; the baseline's does.**
   - At `n = 480` and `mu <= 2`, the DP head reaches 0.909-0.924 starting from 20 public images, and 0.925-0.933 from
     the full set.
   - The public-only head goes from 0.660 (20 images) to 0.9044 (1,350).
   - So private federated data replaces about 1,300 public images. It cannot add much once those exist.
4. **The realistic attacker stays far below the bound and is not weak.**
   - Without noise it reaches AUC 0.84-0.98.
   - At `mu = 1` it reaches AUC 0.52-0.61 against the bound 0.760, which is an implied `mu` of 0.09-0.41.
   - No calibrated AUC's jackknife lower end exceeds its bound, in any of the 24 audited settings with noise.
   - A weak initial head (20 public images) does not increase the measured leakage.
5. **Caveat for `n = 480`: the protected unit is a pseudo-client.**
   - Each of the 48 real clients is 10 pseudo-clients. By GDP group privacy (Dong, Roth and Su 2022), the guarantee
     for a real client of this layout is only `10 mu`, which is vacuous.
   - The `n = 480` rows describe a federation of 480 genuine owners of 5-115 records each, not more protection for
     these 48.
   - At the real `n = 48`, the `mu <= 2` passes are:
     - 20 images: all four settings, robust except `T = 20`, `mu = 1` against the strongest baseline.
     - 50 images: `mu = 2`.
     - 100 images: `mu = 2`, robust only against the protocol baseline.

### A. Public-set sweep (Amendment 1) and baseline strength (Amendment 2, E)

**Public-only accuracy on the server test set,** mean over the 5 evaluation subsets:

- `best_legit` is, per subset, the most accurate of four legitimate fits: the protocol head, a wide L2 grid, L2-normalized
  raw features, and nearest class mean. It is chosen by test accuracy, which favours the baseline.
- `oracle_c` chooses L2 on the test set. It is an optimistic ceiling, not a legitimate baseline.

| public images | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|
| public-only, protocol | 0.660 | 0.782 | 0.832 | 0.864 | 0.890 |
| public-only, best_legit | 0.688 | 0.800 | 0.843 | 0.871 | 0.891 |
| public-only, oracle_c | 0.676 | 0.785 | 0.838 | 0.868 | 0.893 |

- The protocol head is recomputed exactly (checked against every one of the 2,000 records).
- L2-normalized raw features are the strongest legitimate fit on 17 of 25 subsets.
- Neither stronger baseline changes any per-size verdict.

**Gains at `mu <= 2`** (the settings with a meaningful bound), in pp:

- Each cell is "gain [lower 95% end]", against the protocol baseline / against best_legit.
- **Bold** means the lower end is at least 5.0 (robust).
- The intervals resample the 5 subsets, the replicates and the test images.

| `n` | `T` | `mu` | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| 48 | 10 | 1 | **14.9 [10.3]** / **12.2 [6.9]** | 6.7 [4.7] / 5.0 [2.5] | 2.8 [1.5] / 1.6 [0.1] | 2.2 [1.0] / 1.5 [0.4] | -0.1 [-1.1] / -0.3 [-1.2] |
| 48 | 10 | 2 | **22.7 [17.8]** / **20.0 [14.8]** | **11.6 [9.8]** / **9.8 [7.6]** | **6.8 [5.3]** / 5.7 [4.3] | 4.2 [2.9] / 3.5 [2.3] | 2.0 [1.0] / 1.9 [0.9] |
| 48 | 20 | 1 | **12.1 [7.5]** / 9.4 [4.1] | 4.2 [2.2] / 2.5 [-0.1] | 0.9 [-0.7] / -0.2 [-1.8] | 0.4 [-1.1] / -0.3 [-1.6] | -2.6 [-3.9] / -2.8 [-4.0] |
| 48 | 20 | 2 | **21.8 [17.4]** / **19.1 [14.0]** | **11.3 [9.4]** / **9.5 [7.1]** | **6.7 [5.1]** / 5.6 [4.1] | 4.3 [2.9] / 3.7 [2.3] | 2.0 [1.0] / 1.8 [0.8] |
| 480 | 10 | 1 | **24.9 [20.2]** / **22.2 [16.9]** | **13.3 [11.3]** / **11.5 [9.2]** | **8.5 [6.8]** / **7.3 [5.8]** | 5.6 [4.1] / 4.9 [3.7] | 3.1 [2.1] / 3.0 [2.0] |
| 480 | 10 | 2 | **26.2 [21.7]** / **23.5 [18.5]** | **14.3 [12.4]** / **12.6 [10.3]** | **9.2 [7.5]** / **8.0 [6.5]** | 6.0 [4.4] / 5.3 [3.9] | 3.8 [2.8] / 3.7 [2.6] |
| 480 | 20 | 1 | **25.6 [21.0]** / **22.9 [17.5]** | **13.7 [11.8]** / **11.9 [9.6]** | **8.8 [7.1]** / **7.7 [6.1]** | 5.9 [4.3] / 5.2 [3.8] | 3.4 [2.4] / 3.3 [2.2] |
| 480 | 20 | 2 | **26.4 [21.5]** / **23.7 [18.0]** | **14.6 [12.8]** / **12.9 [10.7]** | **9.8 [7.9]** / **8.6 [7.1]** | 6.7 [5.0] / 6.0 [4.5] | 4.1 [3.0] / 4.0 [2.8] |

The full tables, with `mu` = 4 and 8, the no-noise references, `oracle_c` and per-subset gains, are in the two JSON
files.

**Reading:**

- **Interval precision.** With 5 subsets, the subset term dominates the intervals: the public-only accuracy varies
  0.618-0.725 between subsets at 20 images.
- **Where the pass is solid.** At 20-100 images and `n = 480`, every `mu <= 2` gain is at least 7.3 pp against
  best_legit, and robust.
- **At 200 images it is marginal.** At `n = 480` the gains at `mu <= 2` are 4.9-6.0 pp against best_legit, with no
  lower end at 5. At `n = 48` they are 1.5-3.7 pp.
- **At 500 images it fails everywhere.** The best gain is 4.8 pp (`n = 480`, `T = 20`, `mu = 8`, protocol baseline).
- **Cost of privacy at 20 images.** Against no noise at the same public size, the cost is:
  - 0.3-1.6 pp at `n = 480`, `mu <= 2` (DP 0.909-0.924, no noise 0.925-0.932);
  - 4.5-15.6 pp at `n = 48` (DP 0.781-0.887, no noise 0.932-0.937).
- **Hyperparameters.** As in the base grid, `n = 48`, `mu = 1` selects `E = 1`, local lr 0.05 and clip multiplier 0.5
  at every size. The noise at 48 slots, not tuning, limits it.

### B. Extended-grid sensitivity (Amendment 1)

- **Grid:** `E` in {1, 5}, local lr 0.01-1.0, `eta_s` 0.3-30 and clip multiplier 0.25-1, so 150 points per setting.
  Selection used the Section 5 rule on the tuning layout.
- **Verdict:** fail. The best gain is 3.81 [2.68, 5.05] pp (`n = 480`, `T = 20`, `mu = 8`, clip multiplier 0.25,
  `eta_s` 10).
- **`mu <= 2`:** at best 2.90 [1.81, 4.06] pp, unchanged.
- **Grid edges:** several selections moved to the new edges (clip 0.25, `eta_s` 0.3 or 30, lr 0.01). The gains
  moved by -0.5 to +1.3 pp. The largest move is `n = 48`, `T = 20`, `mu = 1`, from -0.32 to 0.99.
- **Reading:** the fail did not depend on the grid. The no-noise references cap it (at most 3.90 pp here, 4.07 on the
  base grid).

### C and D. Secondary empirical audit (Amendment 1, C; Amendment 2, D)

**Design:** M0's resolved design, with Agent A's attackers imported unchanged.

- 32 candidates and `R = 32` balanced replicates, 16 candidates included per replicate.
- `n - 16` active clients. At 48 clients the noise on the average is therefore 1.5x B4's, and the audit's DP accuracy
  is lower than the evaluation's.
- Calibrated loss attacker, with the 97.5% pair jackknife.

**Full public set, every setting** (`analysis_audit.json`):

| `n` | `T` | `mu` | AUC bound | AUC [97.5% jackknife] | TPR at 5% FPR (bound) | TPR at 1% FPR (bound) | top-5 AUC | threshold AUC | implied mu |
|---|---|---|---|---|---|---|---|---|---|
| 48 | 10 | no noise | n/a | 0.935 [0.916, 0.954] | 0.717 (n/a) | 0.494 (n/a) | 0.992 | 0.627 | 2.14 |
| 48 | 10 | 1 | 0.760 | 0.614 [0.578, 0.651] | 0.078 (0.26) | 0.016 (0.09) | 0.497 | 0.543 | 0.41 |
| 48 | 10 | 2 | 0.921 | 0.676 [0.640, 0.713] | 0.158 (0.64) | 0.021 (0.37) | 0.624 | 0.546 | 0.65 |
| 48 | 10 | 4 | 0.998 | 0.747 [0.714, 0.780] | 0.178 (0.99) | 0.033 (0.95) | 0.706 | 0.590 | 0.94 |
| 48 | 10 | 8 | 1.000 | 0.861 [0.835, 0.888] | 0.359 (1.00) | 0.119 (1.00) | 0.836 | 0.616 | 1.54 |
| 48 | 20 | no noise | n/a | 0.977 [0.969, 0.984] | 0.828 (n/a) | 0.686 (n/a) | 0.993 | 0.724 | 2.81 |
| 48 | 20 | 1 | 0.760 | 0.595 [0.550, 0.639] | 0.098 (0.26) | 0.020 (0.09) | 0.551 | 0.543 | 0.34 |
| 48 | 20 | 2 | 0.921 | 0.664 [0.622, 0.706] | 0.127 (0.64) | 0.023 (0.37) | 0.617 | 0.557 | 0.60 |
| 48 | 20 | 4 | 0.998 | 0.759 [0.722, 0.796] | 0.182 (0.99) | 0.057 (0.95) | 0.740 | 0.592 | 0.99 |
| 48 | 20 | 8 | 1.000 | 0.851 [0.823, 0.879] | 0.320 (1.00) | 0.111 (1.00) | 0.841 | 0.627 | 1.47 |
| 480 | 10 | no noise | n/a | 0.842 [0.810, 0.874] | 0.393 (n/a) | 0.232 (n/a) | 0.948 | 0.523 | 1.42 |
| 480 | 10 | 1 | 0.760 | 0.524 [0.498, 0.551] | 0.051 (0.26) | 0.014 (0.09) | 0.589 | 0.507 | 0.09 |
| 480 | 10 | 2 | 0.921 | 0.576 [0.538, 0.615] | 0.086 (0.64) | 0.021 (0.37) | 0.686 | 0.511 | 0.27 |
| 480 | 10 | 4 | 0.998 | 0.657 [0.615, 0.698] | 0.156 (0.99) | 0.043 (0.95) | 0.775 | 0.515 | 0.57 |
| 480 | 10 | 8 | 1.000 | 0.718 [0.689, 0.747] | 0.178 (1.00) | 0.080 (1.00) | 0.836 | 0.519 | 0.82 |
| 480 | 20 | no noise | n/a | 0.881 [0.851, 0.912] | 0.547 (n/a) | 0.332 (n/a) | 0.953 | 0.524 | 1.67 |
| 480 | 20 | 1 | 0.760 | 0.539 [0.502, 0.575] | 0.066 (0.26) | 0.014 (0.09) | 0.608 | 0.514 | 0.14 |
| 480 | 20 | 2 | 0.921 | 0.577 [0.537, 0.617] | 0.096 (0.64) | 0.025 (0.37) | 0.686 | 0.516 | 0.28 |
| 480 | 20 | 4 | 0.998 | 0.656 [0.612, 0.700] | 0.166 (0.99) | 0.053 (0.95) | 0.801 | 0.521 | 0.57 |
| 480 | 20 | 8 | 1.000 | 0.703 [0.670, 0.735] | 0.207 (1.00) | 0.037 (1.00) | 0.887 | 0.528 | 0.75 |

"Implied mu" is `sqrt(2) Phi^-1(AUC)`, which is exact only for Gaussian trade-off curves.

**20 public images, `mu <= 2` and no noise** (`analysis_audit_public20.json`), next to the full public set:

| `n` | `T` | `mu` | AUC bound | AUC, full public [97.5% jackknife] | AUC, 20 public [97.5% jackknife] | TPR at 5% FPR, full / 20 (bound) | implied mu, full / 20 | DP accuracy, full / 20 |
|---|---|---|---|---|---|---|---|---|
| 48 | 10 | no noise | n/a | 0.935 [0.916, 0.954] | 0.921 [0.898, 0.945] | 0.717 / 0.627 (n/a) | 2.14 / 2.00 | 0.930 / 0.928 |
| 48 | 10 | 1 | 0.760 | 0.614 [0.578, 0.651] | 0.599 [0.546, 0.651] | 0.078 / 0.088 (0.26) | 0.41 / 0.35 | 0.899 / 0.714 |
| 48 | 10 | 2 | 0.921 | 0.676 [0.640, 0.713] | 0.638 [0.587, 0.689] | 0.158 / 0.121 (0.64) | 0.65 / 0.50 | 0.917 / 0.843 |
| 48 | 20 | no noise | n/a | 0.977 [0.969, 0.984] | 0.947 [0.928, 0.967] | 0.828 / 0.740 (n/a) | 2.81 / 2.29 | 0.939 / 0.932 |
| 48 | 20 | 1 | 0.760 | 0.595 [0.550, 0.639] | 0.590 [0.549, 0.631] | 0.098 / 0.086 (0.26) | 0.34 / 0.32 | 0.860 / 0.678 |
| 48 | 20 | 2 | 0.921 | 0.664 [0.622, 0.706] | 0.659 [0.617, 0.702] | 0.127 / 0.094 (0.64) | 0.60 / 0.58 | 0.909 / 0.822 |
| 480 | 10 | no noise | n/a | 0.842 [0.810, 0.874] | 0.860 [0.837, 0.882] | 0.393 / 0.469 (n/a) | 1.42 / 1.53 | 0.931 / 0.926 |
| 480 | 10 | 1 | 0.760 | 0.524 [0.498, 0.551] | 0.545 [0.493, 0.598] | 0.051 / 0.051 (0.26) | 0.09 / 0.16 | 0.926 / 0.910 |
| 480 | 10 | 2 | 0.921 | 0.576 [0.538, 0.615] | 0.547 [0.496, 0.599] | 0.086 / 0.059 (0.64) | 0.27 / 0.17 | 0.928 / 0.921 |
| 480 | 20 | no noise | n/a | 0.881 [0.851, 0.912] | 0.897 [0.873, 0.922] | 0.547 / 0.580 (n/a) | 1.67 / 1.79 | 0.934 / 0.931 |
| 480 | 20 | 1 | 0.760 | 0.539 [0.502, 0.575] | 0.536 [0.500, 0.573] | 0.066 / 0.061 (0.26) | 0.14 / 0.13 | 0.926 / 0.913 |
| 480 | 20 | 2 | 0.921 | 0.577 [0.537, 0.617] | 0.594 [0.550, 0.637] | 0.096 / 0.078 (0.64) | 0.28 / 0.34 | 0.932 / 0.923 |

**Readings (pre-registered, no gate):**

1. **Validity.**
   - No calibrated-AUC jackknife lower end exceeds its formal bound, in all 24 settings with noise (`bound_violated`
     false throughout).
   - The point estimates of TPR at 5% and 1% FPR are also below their bounds everywhere.
   - No re-check or escalation is triggered.
2. **Gap.**
   - Where the bound means something (`mu <= 2`), the attacker's AUC sits 0.15-0.24 below it at `mu = 1`, and
     0.24-0.37 below it at `mu = 2`.
   - The implied `mu` is 0.09-0.41 at `mu = 1`. At `mu = 8` it is 0.75-1.54, below the no-noise levels.
   - Against this attacker the bound is loose by a factor of 2.4-12 in `mu`, over both audits.
3. **Leakage without DP.**
   - The no-noise configurations reach AUC 0.84-0.98, and 0.93-0.99 on the 5 largest clients (both audits).
   - So the attacker is strong where leakage exists, and the low numbers under noise reflect the mechanism rather than
     the attacker. The threshold attacker is much weaker throughout (AUC at most 0.72).
4. **Weaker initial head (D).**
   - The differences (20 public minus full) run from -0.038 to +0.021.
   - In 10 of the 12 settings each AUC lies within the other's 97.5% jackknife interval. The two exceptions are both
     lower at 20 images:
     - `n = 48`, `T = 10`, `mu = 2`: 0.638 against [0.640, 0.713].
     - `n = 48`, `T = 20`, no noise: 0.947 against [0.969, 0.984].
   - So starting from a worse public head does not increase the measured leakage, and at 48 clients it lowers it
     slightly. That holds even though the 48-client DP head improves much less from it (accuracy 0.678-0.843 at 32
     active clients).
   - The top-5 subset cannot be resolved (M0's power table), so it is shown but not read.

**Limits:**

- Two attackers on one panel and one layout.
- The audit's 48-client runs have 1.5x B4's noise.
- An empirical audit bounds nothing from above. It shows what this attacker achieves, and the formal bound stays the
  privacy statement.

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
  - One public set size for the gate (the setup's 1,350 images). Amendment 1 sweeps 20-500 as exploration.
  - One dataset (EuroSAT).
  - One partition (Dirichlet 0.3).
- **Simulator:** an in-process NumPy simulator of the mechanism (unit-tested against the closed forms and the
  frontier layouts), not the Flower message path. A formal-regime comparison opened by the owner would need the real
  path.
- **Attack measurement:** added by Amendments 1-2 (see "Owner follow-up"). The privacy statements remain the formal
  bounds.
- **`n = 480` is a pseudo-client split.** Its guarantee is per pseudo-client. For the 48 real clients it implies
  `10 mu` by group privacy, so the `n = 480` rows stand in for a federation of 480 genuine small owners.
- **Public-sweep precision:** 5 public subsets per size. The subset term dominates the intervals.

## Options for the owner (not decided here)

The original options are superseded by the follow-up. As of Amendment 2:

1. **Accept the G-B4 fail for this setup.**
   - Record: "no non-vacuous client-level regime at this scale with a 1,350-image public set".
   - Add the nuance: about 2 pp at `mu = 1` with 480 (pseudo-)clients, and a realistic attacker at implied `mu` 0.09-0.41.
2. **Also record the conditional positive.**
   - Statement: "with at most 10-20 public images per class, head-only DP-FedAvg at `mu <= 2` beats the public-only
     head by 5-26 pp".
   - Against the strongest legitimate baseline it is robust:
     - at 20-100 images for `n = 480`;
     - at 20 images for `n = 48`, except `T = 20`, `mu = 1`.
   - This is exploration after the fact, not a pre-registered gate result. A confirmatory run on a fresh layout and
     fresh subsets (for example layout 44, 20 and 50 images) would make it a result. It costs about 1 CPU-hour and
     would need a new protocol.
3. **Client subsampling at `n = 480`** with the accountant, as listed before. It is not pursued here.
