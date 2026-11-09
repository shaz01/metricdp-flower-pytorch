# CIA roadmap v3, P2: CIFAR-10 replication of A-3 and D-4

Package P2 of `research/cia_roadmap_v3/ROADMAP.md` (direction F2), Agent P, branch `feature/cia-v3-cifar10`, on the
compute server, using P0's `cifar10` profile (`feature/cia-v3-infra`). Closed and merged by the owner on 2026-11-09.

All numbers come from committed artifacts in `results/cia_v3/cifar10/`:
- `STATE.md` (design and log) and `PROTOCOL.md` (runs, level rule, analysis; fixed before any run);
- `analysis_stage_a.json` (G-P2a), `pilot_levels.json` (level choice) and `analysis.json` (G-P2b), from `analyze.py`;
- `FINDINGS.md` and `figures/`; the runs in `results/` (320) and `pilot/` (26).

## 1. Question

Do the two central EuroSAT results hold on a second label-skewed dataset with real leakage?
- **A-3:** at matched accuracy, metric privacy leaks like global DP.
- **D-4:** the composite (Poisson client subsampling at `q = 0.25` with logit adjustment, `tau = 1`) leaks at least
  0.05 AUC less than metric privacy, under the standard and the sampling-aware attacker.

## 2. Design

- **Regime:** M2's on CIFAR-10: label-Dirichlet alpha 0.3, 48 slots, layout seed 42, 32 candidates (top decile plus 27
  by size quantile), 16 IN per replicate, 16 always in, 32 active; 50 rounds, 5 local epochs, clipping 5, equal weights;
  `R = 32`, `design_seed = 2`; 1,024 trials per configuration.
- **Stage a:** vanilla, replicates 0-7; G-P2a needs an all-candidate AUC of at least 0.65.
- **Accuracy pilot** (`design_seed = 3`, 2 replicates): global DP and metric privacy at 1x-8x of `r = 0.001546`, the
  composite at five ratios. Targets `t2`, `t4` from global DP at 2x and 4x; three consecutive levels chosen per
  mechanism by the protocol's window rule.
- **Stage b:** vanilla, global DP 1x/2x/4x, metric privacy and the composite at three levels each, `R = 32`, in two
  halves with A1's replacement rule between them. The composite is scored at every round (51 checkpoints); the gated
  attackers use the standard 11.
- **Statistics:** A1's matched-accuracy difference (curves linear in mean final accuracy, pair jackknife at 97.5%,
  two-way bootstrap); targets `a2`, `a4` = global DP's mean final accuracy at 2x and 4x. Untestable targets do not count
  as passes.
- **Reproduction first:** A1's metric privacy minus global DP at 0.740, -0.000 [-0.019, 0.018], to 3 decimals.

## 3. Results

**G-P2a passed:** vanilla AUC 0.998 [0.995, 1.001] at `R = 8` (EuroSAT's M2 vanilla 0.723), accuracy 0.633.

**Pilot and levels:** targets `t2` 0.554, `t4` 0.326; safeguard passed. Metric privacy 2x, 4x, 8x; composite `r`
0.000773, 0.003092, 0.006184 (no window qualified; smallest shortfall 0.0004). The replacement check after the first
half (`a2` 0.546, `a4` 0.319 on replicates 0-15) bracketed both targets for both mechanisms; no replacement.

**Per level (all candidates):**

| Configuration | Accuracy | Lowest client | AUC [97.5%] | Top decile | TPR@1% |
|---|---|---|---|---|---|
| Vanilla | 0.630 | 0.440 | 0.999 [0.998, 1.000] | 1.000 | 0.971 |
| Global DP 1x / 2x / 4x | 0.610 / 0.547 / 0.319 | 0.401 / 0.355 / 0.160 | 0.997 / 0.985 / 0.770 | 0.996 / 0.976 / 0.692 | 0.945 / 0.791 / 0.076 |
| Metric privacy 2x / 4x / 8x | 0.598 / 0.516 / 0.298 | 0.384 / 0.319 / 0.137 | 0.997 / 0.975 / 0.743 | 0.999 / 0.952 / 0.680 | 0.959 / 0.629 / 0.061 |
| Composite `r` 0.000773 / 0.003092 / 0.006184 | 0.607 / 0.538 / 0.316 | 0.398 / 0.346 / 0.161 | 0.978 / 0.965 / 0.707 | 0.977 / 0.964 / 0.675 | 0.805 / 0.592 / 0.053 |

**G-P2b** (targets `a2` 0.547, `a4` 0.319):

| Comparison | `a2` | `a4` | Outcome |
|---|---|---|---|
| G-A1: metric privacy minus global DP | -0.002 [-0.008, 0.005], equivalent | not testable | **A-3 replicates** |
| G-C1, standard: composite minus metric privacy | -0.017 [-0.027, -0.007], fail | -0.055 [-0.086, -0.025], pass | **D-4 does not replicate** |
| G-C1, sampling-aware | -0.007 [-0.013, -0.001], fail | -0.040 [-0.076, -0.003], fail | **D-4 does not replicate** |
| Reported: 51-checkpoint attacker | -0.011 [-0.019, -0.004] | -0.051 [-0.075, -0.026] | |

At `a4`, global DP's 4x level is the curve's lowest point and defines the target, so leave-one-pair-out curves do not
reach it; the two-way bootstrap for G-A1 there is [-0.041, 0.026] (984 of 2,000 resamples bracketed). Lowest-client
accuracy at `a2` / `a4`: global DP 0.355 / 0.160, metric privacy 0.344 / 0.155, composite 0.353 / 0.163.

ROADMAP Section 4 for this outcome: "the positive section is stated for EuroSAT only, with CIFAR-10 as a bound; the owner
reconsiders decision 1".

## 4. Anomalies and deviations

- **Cost above the original cap:** P0's cost runs projected P2 above its 85 GPU-h cap; the owner raised it to 120
  (decision 5) and, after the pilot measured about 254 run-hours for stage b, approved stage b in full at about 290
  (decision 6).
- **Shared-machine failures:** out of memory in stage a when another user's job grew (2026-10-31); Ray host-memory kills
  (2026-10-31, and 2026-11-04 23:00 to 2026-11-05 05:00); two queues once picked the same GPU (fixed with a lock).
- **Launcher failure, 2026-11-02:** the stage-b launcher checked host RAM only at start; for 9.5 hours runs were killed
  within seconds and only 9 completed. The runner exits 0 on a failed run, so the failures did not stop the loop. The
  script was rewritten to wait for RAM and GPU memory before every run and to read success from `complete.json`.
- **Owner's F0.7 job:** 27 of its units on GPU 1 failed with CUDA out of memory around 2026-11-05 02:30, while two P2
  workers shared that GPU with it and another user's job. Agent P now runs at most one worker on a GPU with other jobs.
- **Order deviation:** P2b's analysis ran before P3's interference check, which the roadmap puts first. P3 found no
  IN-count effect distinguishable from 0 at a fixed active count, and the owner decided on no correction (decision 9).
- **Parallel clients:** 2 or 4 per run (PROTOCOL.md says 4); this changes scheduling, not trajectories.

## 5. Conclusions

- A-3 replicates on CIFAR-10: metric privacy is no more private than global DP at matched accuracy.
- D-4 does not: the composite's reduction (every interval below 0) reaches 0.05 only once, and the sampling-aware
  attacker removes most of it. With P3's every-round result on EuroSAT, the composite's advantage is not robust.
- CIFAR-10 leaks near-perfectly at every useful accuracy (AUC 0.97-0.99 at `a2`); no mechanism changes the lowest
  client's accuracy at matched accuracy.
- Absolute AUC depends on the design and on `R` (P3, decision 9); all comparisons here hold both fixed.

## 6. Cost

Summed run time 247.7 run-hours: stage a 8.0, pilot 21.3, stage b 218.4. Failed tries kept nothing and are not counted.
GPU occupancy is lower than summed run time (shared GPUs, up to four workers). Cap about 290 GPU-h (decision 6).
