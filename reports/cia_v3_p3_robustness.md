# CIA roadmap v3, P3: every-round attacker and IN-count interference check

Package P3 of `research/cia_roadmap_v3/ROADMAP.md` (direction F4), Agent P, branch `feature/cia-v3-robustness`, on the
compute server, with ROADMAP decisions 7, 8 and 9. Closed and merged by the owner on 2026-11-09.

All numbers come from committed artifacts in `results/cia_v3/robustness/`:
- `STATE.md` (design and log), `PROTOCOL.md` (Sections 1-4 fixed before any run; Sections 5 and 6 before their runs)
  and `ESCALATION.md` (G-P3(b));
- `analysis.json`, `analysis_follow_up.json` and `analysis_follow_up_r64.json`, from `analyze.py`;
- `FINDINGS.md` and `figures/`; the 288 runs in `results/`.

## 1. Question

- **(a)** Does a sampling-aware attacker that sees every round, not 5-round windows, recover the composite's leakage, and
  does D-4 survive it on EuroSAT?
- **(b)** Does the attack AUC depend on how many candidates are IN per replicate? Every v2 and v3 result uses 16 of 32.

## 2. Design

- EuroSAT in M2's regime, `design_seed = 2`, `R = 32`: the composite at `r` 0.000773 and 0.003092 and metric privacy at
  `r` 0.001546 and 0.003092 (the levels bracketing 0.740), scored at every round; global DP 2x with 8 and with 24
  candidates IN (192 runs).
- **Every-round attacker:** the calibrated attacker on the 11 standard checkpoints plus `z = sum_t (s_t - q)(d_t -
  dbar_{c,t})`, the target's per-round schedule against the per-round drop in its shadow loss (real schedule for IN
  trials, simulated i.i.d. schedule for OUT trials, as in measurement standard v3's sampling-aware attacker).
- **IN-count statistic:** `AUC(8 IN) - AUC(24 IN)`, standard attacker; complementary pairs do not exist in these
  designs, so the interval is a leave-one-block-out jackknife (blocks of 4 replicates, 97.5%).
- **Follow-ups** (decisions 7 and 8): the same comparison at a fixed active count of 24 (32 runs with 24 IN and no
  always-in slot, through the new default-off runner option `--always-in-count`), then extended to `R = 64` (64 runs).
- **Reproduction first:** every analysis reproduced C1's composite minus metric privacy at 0.740, -0.095 [-0.137,
  -0.053].

## 3. Gate

- "(a) The every-round sampling-aware attacker: G-C1's rule for the composite against metric privacy at 0.740 and
  0.660, where bracketed."
- "(b) |AUC(8 IN) - AUC(24 IN)| at most 0.03, by the pair jackknife at 97.5%. If not, escalate before P2b's analysis:
  measurement standard v3 then needs an IN-count correction."

## 4. Results

**G-P3(a) fails.** Composite minus metric privacy at 0.740:

| Composite attacker | Difference [97.5%] | Two-way |
|---|---|---|
| Standard, 11 checkpoints | -0.122 [-0.158, -0.086] | [-0.187, -0.062] |
| All 51 checkpoints | -0.132 [-0.186, -0.079] | [-0.197, -0.069] |
| Sampling-aware, 5-round windows | -0.065 [-0.120, -0.009] | [-0.128, 0.003] |
| 51 checkpoints plus `z` | +0.082 [0.024, 0.140] | [0.020, 0.147] |
| **Every-round sampling-aware (gated)** | **+0.097 [0.042, 0.151]** | [0.035, 0.161] |

0.660 is not testable (the levels stop at 0.724 and 0.678). Per level, the every-round attacker scores the composite
0.820 and 0.788 against 0.582 and 0.578 for the standard attacker, and `z` alone gives 0.807 and 0.779; metric
privacy scores 0.706 and 0.679. The owner put an attacker who knows the target's schedule in scope (decision 8), so D-4
does not hold on EuroSAT against it.

**G-P3(b) fails and is resolved by decision 9:**

| Configuration | `R` | AUC [97.5%] | Accuracy |
|---|---|---|---|
| 8 IN, 16 always in (24 active) | 32 | 0.694 [0.647, 0.742] | 0.731 |
| 24 IN, 16 always in (40 active) | 32 | 0.654 [0.624, 0.684] | 0.749 |
| 24 IN, none always in (24 active) | 32 | 0.682 [0.637, 0.727] | 0.761 |
| 8 IN, 16 always in (24 active) | 64 | 0.729 [0.703, 0.754] | 0.732 |
| 24 IN, none always in (24 active) | 64 | 0.715 [0.692, 0.738] | 0.761 |

- G-P3(b): +0.040 [0.009, 0.072], outside +-0.03; escalated.
- At a fixed active count: +0.012 [-0.025, 0.050] (`R = 32`), +0.014 [-0.016, 0.044] (`R = 64`), inconclusive by the
  protocol but not distinguishable from 0. The active count at a fixed IN count: +0.028 [-0.008, 0.064].
- AUC rises by about 0.035 from `R = 32` to `R = 64`, because the calibrated attacker gets twice the references.
- Decision 9: no IN-count correction; absolute AUC is stated as conditional on the design and on `R`, and
  matched-accuracy comparisons, which hold both fixed, stand.

## 5. Anomalies and deviations

- **Order:** P2b's analysis ran before this package's interference check, which the roadmap puts first; G-P2b stayed
  provisional until decision 9.
- **Option 2 as built differs from ESCALATION.md:** holding the active count at 32 needs 24 always-in slots, and
  EuroSAT has 16, so the follow-up held it at 24 (decision 7 records the change).
- **Out-of-memory retries:** 51 tries failed on 2026-11-07 between 14:00 and 16:00, when Agent P ran five workers on the
  two shared GPUs; all were retried, and the extension ran one worker per GPU.
- **Evaluation summaries committed by mistake** (174 files) in 827c7478, untracked in b9b66955; they remain in the history.
- **Rounding:** the `R = 64` upper bound was first reported as 0.045 after rounding twice; it is 0.044 (0.04445).

## 6. Conclusions

- Against an attacker who knows when the target is sampled, client subsampling does not protect on EuroSAT: the
  composite leaks more than metric privacy at matched accuracy. The 5-round attacker of measurement standard v3
  under-states this.
- Randomized inclusion measures an absolute AUC that depends on the active count and on `R`; within one design and one
  `R`, as every gate uses it, it remains a valid comparison instrument.

## 7. Cost

Summed run time 75.6 run-hours: the six configurations 52.8, the fixed-active follow-up 7.6, the `R = 64` extension 15.2.
The original design ran 2026-11-06 21:43 to 2026-11-07 16:11 on both GPUs, at most about 37 GPU-h of occupancy against
the 35 GPU-h cap (escalation 43.75); the follow-ups were added by decisions 7 and 8. Failed tries are not counted.
