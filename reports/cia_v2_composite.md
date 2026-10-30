# CIA roadmap v2, C1: the composite (subsampling + logit adjustment) against metric privacy at matched accuracy

Package C1 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9, 13, 14, 15 and 16). Branch
`feature/cia-v2-composite`. The owner assigned Agent B's packages to the agent that ran Agent A's, opened C1 on
2026-10-29 and closed it on 2026-10-30.

All numbers below come from committed artifacts in `results/cia_v2/composite/`:
- `STATE.md`: design and log.
- `PROTOCOL.md`: fixed 2026-10-29, after the accuracy pilot and before any leakage run.
- `FINDINGS.md`.
- `analysis.json`: built from the 128 runs in `results/`, paired with A1's, M2's and B1's committed runs on the same design.
- The accuracy pilot in `pilot/`, never analyzed for leakage.

## 1. Question

Two mechanisms passed a B gate, and both act on the same channel, the client's label mix (B5's diagnostic):
- **Logit adjustment** (B5) removes the client's label prior from local training: -0.100 AUC at 2x noise.
- **Client subsampling at `q = 0.25`** (B1) dilutes that prior across rounds: about -0.10 at 0.740 accuracy against
  global DP.

C1 asks whether their combination leaks at least 0.05 AUC less than the paper's metric privacy at the same accuracy. It
also asks whether that reduction survives attackers who know the defenses.

## 2. Mechanism and design

- **Composite** (no new mechanism code). B1's Poisson-sampled global DP:
  - Each round, every active client joins with probability `q = 0.25`.
  - The release divides by `q N`.
  - It adds noise `r C` to the average.

  Combined with B5's logit-adjusted local loss at `tau = 1`. Runner flags:
  `--privacy subsampled-global-dp --sampling-rate 0.25 --logit-adjustment-tau 1.0 --surrogates`.
- **Regime: M2's.**
  - Randomized inclusion: 32 candidates, 16 IN per replicate, 32 active of 48.
  - EuroSAT, Dirichlet alpha 0.3, layout seed 42.
  - 50 rounds, equal weights.
  - `R = 32`, `design_seed = 2`.

  Every C1 run pairs with the A1, M2 and B1 runs of the same replicate.
- **Levels.**
  - The noise levels are `r` = 0.000773, 0.003092, 0.006184 and 0.012368, chosen from an accuracy pilot (Section 3).
  - The rule was A1's replacement rule, applied automatically after stage 1 (replicates 0-15). It required no replacement.
- **Owner decisions** (decision 15):
  - EuroSAT only: on Alzheimer, leakage without noise is already near chance (AUC 0.564, AZ).
  - The formal clause is reported, not gated, as decision 13 did for B1.
  - Two adaptive attackers are added and reported.
- **Gated attacker.** The calibrated loss attacker on the 11-checkpoint shadow-loss trajectory, as everywhere since M0.
- **Adaptive attacker (a), label channel.** Label-matched public surrogates, scored like the shadow samples. This
  attacker holds no candidate record, only the candidate's label mix. Only M2's global-DP runs also scored surrogates,
  so (a) is compared against global DP under the same attacker.
- **Adaptive attacker (b), defense-aware.** Each candidate's shadow sample is scored with the logit-adjusted loss
  `CE(softmax(log p + tau log pi_c))`:
  - `pi_c` is the candidate's label prior, estimated from its shadow sample with add-one smoothing.
  - The attacker therefore evaluates the loss the defended clients trained on.
  - Computed for the composite only, and compared against metric privacy under the standard attacker.
- **Matched-accuracy statistic** (A1's):
  - The statistic is `AUC_composite(a*) - AUC_other(a*)`.
  - Each AUC curve is linear in mean accuracy between levels.
  - The gate interval is a pair jackknife at 97.5%, in which every resample draws the same pairs in all configurations.
  - A two-way bootstrap at 95% is reported alongside.
  - Targets: `a*` = 0.740 and 0.660.
- **G-C1.** At both targets, the composite's all-candidate AUC must be at least 0.05 below metric privacy's, with the
  gate interval below 0.
- **Safety.**
  - `--require-all-clients` stops a run at the first failed or missing sampled client.
  - The run check refuses rounds that aggregated fewer clients than were sampled.

## 3. Results

**Pilot** (accuracy only, `design_seed = 3`, 2 runs per level), mean final accuracy:

| `r` | 0.0001 | 0.000773 | 0.003092 | 0.006184 | 0.012368 | 0.024736 |
|---|---|---|---|---|---|---|
| Composite | 0.776 | 0.780 | 0.713 | 0.701 | 0.594 | 0.228 |
| B1, `q = 0.25` alone (main runs) | 0.746 | 0.736 | 0.698 | 0.602 | | |

**Main runs.**
- All 128 runs pass the run check: every sampled client trained in every round.
- There were no out-of-memory failures and no runs stopped by `--require-all-clients`.
- Stage 1's accuracies bracketed both targets.

**Per level** (all candidates, gated attacker unless noted):

| `r` | Accuracy | Lowest client acc. | AUC [95%] | TPR@5% | TPR@1% | Top-decile AUC | Surrogate AUC (a) | Defense-aware AUC (b) | Amplified `mu` |
|---|---|---|---|---|---|---|---|---|---|
| 0.000773 | 0.772 | 0.497 | 0.592 [0.538, 0.646] | 0.072 | 0.020 | 0.659 | 0.563 | 0.619 | not computable |
| 0.003092 | 0.723 | 0.415 | 0.614 [0.570, 0.658] | 0.088 | 0.008 | 0.691 | 0.599 | 0.598 | not computable |
| 0.006184 | 0.661 | 0.327 | 0.574 [0.520, 0.627] | 0.127 | 0.053 | 0.696 | 0.587 | 0.588 | not computable |
| 0.012368 | 0.559 | 0.200 | 0.537 [0.505, 0.569] | 0.092 | 0.016 | 0.619 | 0.532 | 0.530 | 37.1 |

**G-C1 (gate): passes at both targets, under both intervals.**

| `a*` | Composite AUC | Metric privacy AUC | Difference [97.5% jackknife] | Two-way bootstrap [95%] | G-C1 |
|---|---|---|---|---|---|
| 0.740 | 0.606 | 0.702 | **-0.095 [-0.137, -0.053]** | [-0.154, -0.036] | **pass** |
| 0.660 | 0.573 | 0.681 | **-0.108 [-0.163, -0.053]** | [-0.183, -0.035] | **pass** |

**Reported comparisons** (all candidates, matched accuracy):

| Composite against | `a*` = 0.740 | `a*` = 0.660 |
|---|---|---|
| Private median (A1) | -0.090 [-0.139, -0.041]; two-way [-0.149, -0.025] | -0.083 [-0.131, -0.034]; two-way [-0.150, -0.018] |
| Global DP (M2) | -0.096 [-0.148, -0.043]; two-way [-0.159, -0.027] | -0.093 [-0.157, -0.030]; two-way [-0.170, -0.018] |
| Subsampling alone, `q = 0.25` (B1) | +0.006 [-0.048, 0.061]; two-way [-0.061, 0.078] | -0.034 [-0.084, 0.016]; two-way [-0.109, 0.042] |
| Metric privacy, top decile | -0.050 [-0.138, 0.037]; two-way [-0.174, 0.084] | -0.055 [-0.141, 0.030]; two-way [-0.201, 0.057] |

**Adaptive attackers** (reported):

| Attacker | `a*` = 0.740 | `a*` = 0.660 |
|---|---|---|
| (b) Defense-aware on the composite, against metric privacy (standard attacker) | -0.096 [-0.136, -0.057]; two-way [-0.154, -0.038] | -0.094 [-0.143, -0.044]; two-way [-0.170, -0.024] |
| (a) Surrogates, composite against global DP (same attacker) | -0.093 [-0.134, -0.053]; two-way [-0.161, -0.026] | -0.066 [-0.122, -0.010]; two-way [-0.135, -0.002] |

At matched accuracy, the defense-aware attacker scores the composite at 0.605 and 0.587, against 0.606 and 0.573 for the
standard attacker.

## 4. Reading

- **The composite beats every full-participation mechanism measured on this design** (metric privacy, private median,
  global DP), by about 0.08-0.11 AUC at matched accuracy, under both intervals.
- **Knowing the defenses does not help the attacker.**
  - Scoring with the loss the clients trained on recovers no leakage.
  - An attacker who holds only the candidate's label mix stays well below global DP under the same attacker.
- **The reduction comes from subsampling.** At matched accuracy the composite sits on B1's `q = 0.25` curve: no
  detectable difference at either target. Logit adjustment moves the composite along that curve instead:
  - It gives 3-10 pp more accuracy at the same noise.
  - It extends the reachable accuracy from B1's best of 0.746 to 0.772.
  - It adds no privacy that can be measured at matched accuracy.
- **B5 is not contradicted.** B5's -0.100 was measured against unadjusted training at the same noise, not against
  subsampling at matched accuracy. The two effects do not add on this design.
- **Top decile:** the point estimates are near -0.05, but the intervals include 0; that subset has a tenth of the pairs.
- **The formal guarantee is no better than B1's:**
  - At three of the four levels, the noise multiplier is below the accountant's supported range.
  - Where it computes, the amplified `mu` is 37.1, which is vacuous.

## 5. Anomalies and corrections

- **Out-of-memory in the pilot (2026-10-29 17:11).**
  - Cause: GPU 1 was shared with another user's job holding about 14 GB, and two workers ran 8 parallel clients each.
  - Effect: 166 client failures in 10 minutes. `--require-all-clients` stopped every affected run at once, so nothing
    corrupted was saved.
  - Fix: the pilot was relaunched as one worker with 4 parallel clients. The main runs used two workers with 4 each.
- **Non-monotone AUC between the two lowest levels** (0.592 then 0.614): the intervals overlap fully.
- **Bootstrap bracketing:**
  - Against subsampling alone at 0.740, 1,720 of 2,000 two-way resamples bracketed the target. B1's best level is just
    above it.
  - Against private median at 0.740, 1,962 of 2,000 did.
  - All other comparisons had 1,996 or more.
- **Scaler.** At the owner's request, a scaler was ready to add two workers whenever GPU 1 had no other user's process.
  The other user's job stayed on GPU 1 throughout, so it never added any.

## 6. Caveats

- EuroSAT only, one layout, one design (`R = 32`, alpha 0.3).
- The comparison against subsampling alone cannot detect differences below about 0.05.
- The formal guarantee is weaker than global DP's at the same noise on the average, and mostly not computable.
  Amplification also assumes the per-round sample stays secret.
- Fairness cost: the lowest client accuracy is 0.33 at the level nearest 0.660 and 0.20 at the noisiest level. It is not
  gated.
- Adaptive attacker (a) can only be compared against global DP, since A1's and B1's runs did not score surrogates.

## 7. Cost

- **Total:** about 11.5 GPU-h of the 15 GPU-h cap (decision 16), measured as GPU-1 occupancy as for B1.
- **Pilot:** 12 runs, 1.3 run-hours.
- **Main runs:** two workers on GPU 1, shared with another user's job by owner approval. They ran 2026-10-29 20:15 to
  2026-10-30 06:24, about 10.2 h of GPU-1 time.
- **Summed per-run time:** 16.9 h, because the two workers overlapped on one GPU. That is under the 18.75 stop line.
- GPU 0 was left to other users.

## 8. Conclusions and owner decisions (2026-10-30)

1. **G-C1 passes at both targets:** -0.095 at 0.740 and -0.108 at 0.660 AUC against metric privacy, under both
   intervals. The reduction holds against both adaptive attackers.
2. **The gain is attributable to client subsampling.** Logit adjustment adds accuracy, not measured privacy, at matched
   accuracy.
3. **Closed by the owner** on 2026-10-30; `feature/cia-v2-composite` merges into master. Roadmap v2 has no remaining
   package. The owner will run the literature and novelty check separately, from
   `research/cia_roadmap_v2/RESULTS_HANDOFF.md`.
