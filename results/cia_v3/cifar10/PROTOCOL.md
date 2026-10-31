# P2 protocol: CIFAR-10 replication of A-3 and D-4

Fixed 2026-10-31, before any P2 run. Hypothesis, design summary and gates: `STATE.md`. Roadmap:
`research/cia_roadmap_v3/ROADMAP.md` Section 3 (P2).

## 1. Runs

Runner: `results/cia_v2/randomized_inclusion/runner.py --dataset cifar10` (P0). Common arguments:
`--dataset cifar10 --replicates 32 --execute --max-parallel-clients 4`. Output `results/cia_v3/cifar10/results/`
(pilot: `results/cia_v3/cifar10/pilot/`). Every run passes the runner's every-client check; a failed run is rerun
with the same seeds, at most three passes.

| Stage | Config | Arguments | Design seed | Replicates |
|---|---|---|---|---|
| a | `vanilla` | `--privacy vanilla` | 2 | 0-7 |
| pilot | `pilot-gdp-r<k>x` | `--privacy global-dp --noise-ratio <k * 0.001546>`, k = 1, 2, 4, 8 | 3 | 0-1 |
| pilot | `pilot-mp-r<k>x` | `--privacy metric-privacy --noise-ratio <k * 0.001546>`, k = 1, 2, 4, 8 | 3 | 0-1 |
| pilot | `pilot-comp-r<r>` | `--privacy subsampled-global-dp --sampling-rate 0.25 --logit-adjustment-tau 1.0 --noise-ratio <r>`, r = 0.0001, 0.000773, 0.003092, 0.006184, 0.012368 | 3 | 0-1 |
| b | `vanilla` | as stage a | 2 | 8-31 |
| b | `gdp-r1x`, `gdp-r2x`, `gdp-r4x` | `--privacy global-dp --noise-ratio` 0.001546, 0.003092, 0.006184 | 2 | 0-31 |
| b | `mp-<level>` x 3 | `--privacy metric-privacy`, levels by Section 3 | 2 | 0-31 |
| b | `comp-r<r>` x 3 | composite as in the pilot, plus `--checkpoint-every 1`, levels by Section 3 | 2 | 0-31 |

Stage b runs in two halves: replicates 0-15 (vanilla 8-15), then 16-31.

## 2. Stage a and G-P2a

After the 8 vanilla runs, and before any other P2 run is analyzed: the calibrated attacker
(`experiments.cia.audit_attacks.calibrated_scores`) on the 11-checkpoint shadow losses of replicates 0-7, membership
checked against columns 0-7 of the `design_seed = 2` design, `roc_auc` over the 256 trials.
- **Pass:** point estimate at least 0.65. The pilot starts.
- **Fail:** stop, write `ESCALATION.md`, propose FEMNIST (decision 2).
The pair-jackknife interval over the 4 pairs, the top-decile AUC and the mean final accuracy are reported with it.

## 3. Pilot and level choice (accuracy only)

Mean final server-test accuracy of the two pilot runs per level.
- Pilot targets: `t2` and `t4` = global DP's pilot accuracy at 2x and 4x.
- **Safeguard:** if `t4` is below 0.30 or `t2 - t4` is below 0.02, the targets are uninformative on CIFAR-10. P2 then
  stops before stage b and escalates.
- **Level choice for metric privacy and the composite**, each separately, from its pilot grid sorted by `r`:
  among windows of three consecutive grid levels whose first level's accuracy is at least `t2` and whose last level's
  is at most `t4`, take the one with the smallest `r`. If no window qualifies, take the window with the smallest
  shortfall `max(0, t2 - acc_first) + max(0, acc_last - t4)`, ties to the smaller `r`. A target that this window cannot
  bracket may become not testable (Section 5).
- The chosen levels are recorded in `STATE.md` before stage b starts.

## 4. Replacement rule (A1's, accuracy only)

After the first half of stage b, the targets are re-estimated from global DP's half-stage accuracy at 2x and 4x. If a
mechanism's three levels do not bracket both, its outermost level on the wrong side is replaced once, by the next
level outward in its pilot grid, extended by doubling or halving `r`. The replacement runs both halves. No leakage
statistic is computed before the second half of every configuration is complete.

## 5. Analysis (`analyze.py`), after stage b

1. **Reproduction check first:** with A1's committed runs, the code reproduces metric privacy against global DP at
   0.740, -0.000 [-0.019, 0.018], to 3 decimals. Otherwise it stops.
2. Per configuration (M2's `config_summary`): AUC with the 97.5% pair jackknife and the two-way bootstrap, top decile,
   TPR at 5% and 1% FPR, accuracy, lowest-client accuracy. The composite is also scored by the sampling-aware attacker
   (11 checkpoints) and by the calibrated attacker on all 51 checkpoints (reported only).
3. **Targets:** `a2`, `a4` = global DP's mean final accuracy at 2x and 4x over the 32 replicates.
4. **Matched accuracy** (A1's statistic, `curve_value` linear in mean accuracy, pair jackknife at 97.5% with the same
   pairs in every configuration, two-way bootstrap 2,000 resamples, seed 2026):
   - G-A1: metric privacy minus global DP at `a2` and `a4`;
   - G-C1: composite minus metric privacy at `a2` and `a4`, standard attacker for both; and composite
     (sampling-aware) minus metric privacy (standard).
   A target is **not testable** for a comparison if either curve does not reach it, or if the jackknife cannot be
   formed because a leave-one-pair-out curve does not reach it. In that case the two-way bootstrap is reported.
5. **Decisions** as in `STATE.md` (Gate). Fairness (lowest-client accuracy at the targets) is reported, not gated.

## 6. Cost and machine

Compute server, one worker per GPU under ROADMAP decision 4, 4 parallel clients. CIFAR-10 workers peaked at 2.5 GB in P0, so a shared GPU qualifies for P2 with at least 4 GB free (about 1.5 GB of headroom for the other user). Projected about 196 run-hours
(`STATE.md`, Cost); stage a about 4.3, pilot about 12.4. Stop and escalate at 150 GPU-h measured.
