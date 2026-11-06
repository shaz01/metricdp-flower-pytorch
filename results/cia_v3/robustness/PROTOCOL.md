# P3 protocol: every-round attacker and interference check

Fixed 2026-11-06, before any P3 run. Hypothesis, design summary and gate: `STATE.md`. Roadmap:
`research/cia_roadmap_v3/ROADMAP.md` Section 3 (P3).

## 1. Runs

Runner: `results/cia_v2/randomized_inclusion/runner.py` (EuroSAT, M2's regime; P0's options). Common arguments:
`--replicates 32 --design-seed 2 --execute`. Output `results/cia_v3/robustness/results/`. Every run passes the runner's
every-client check; a failed run is retrained from scratch with the same seeds, at most 6 tries.

| Config | Arguments | Runs |
|---|---|---|
| `p3-comp-r0.000773` | `--privacy subsampled-global-dp --sampling-rate 0.25 --logit-adjustment-tau 1.0 --noise-ratio 0.000773 --checkpoint-every 1` | 32 |
| `p3-comp-r0.003092` | as above, `--noise-ratio 0.003092` | 32 |
| `p3-mp-r1x` | `--privacy metric-privacy --noise-ratio 0.001546 --checkpoint-every 1` | 32 |
| `p3-mp-r2x` | `--privacy metric-privacy --noise-ratio 0.003092 --checkpoint-every 1` | 32 |
| `p3-gdp-r2x-in8` | `--privacy global-dp --noise-ratio 0.003092 --in-count 8` | 32 |
| `p3-gdp-r2x-in24` | `--privacy global-dp --noise-ratio 0.003092 --in-count 24` | 32 |

**Levels.** The composite's two levels are the two that bracket 0.740 on EuroSAT (C1: 0.772 and 0.723). Metric
privacy's are A1's two that bracket it (0.752 and 0.678). Replicates share `design_seed = 2`, so the composite and
metric-privacy runs share the same seeds and inclusion as C1 and A1, and the differences are paired.

**IN-count design.** `design_matrix(32, design_seed=2, in_count=8)`: 8 blocks of 4 replicates, each block splitting the
32 candidates into groups of 8. The 24-IN design is its complement with the same seed. 16 slots always take part, so
8 IN has 24 active clients and 24 IN has 40. The noise std on the released average is `r C` in both.

## 2. Analysis (`analyze.py`)

1. **Reproduction check first:** the code reproduces C1's composite minus metric privacy at 0.740,
   -0.095 [-0.137, -0.053], to 3 decimals from the committed runs, with `rz.matched` (`rz` =
   `research/cia_roadmap_v3/reanalysis/reanalyze.py`). Otherwise it stops.
2. **Every-round statistic:** `rz.schedule_statistic` with a one-round window on the 51-checkpoint losses,
   `z = sum_t (s_t - q) (d_t - dbar_{c,t})`, where `s_t` is the round's schedule (real for IN trials; i.i.d.
   Bernoulli(`q`) from `rz.simulated_schedules` for OUT trials), `d_t = x_{t-1} - x_t`, and `dbar_{c,t}` is the
   candidate's mean drop over its replicates. `z` is divided by its standard deviation, one constant per configuration.
3. **G-P3(a) attacker:** `calibrated_scores` on the 11 standard checkpoints plus `z`, the same construction as
   `rz.sampling_aware` with only the statistic changed. Metric privacy: the standard attacker on 11 checkpoints.
4. **G-P3(a) statistic:** `rz.matched` (curves linear in mean final accuracy, pair jackknife at 97.5%, two-way bootstrap
   2,000 resamples, seed 2026), at 0.740 and 0.660.
5. **G-P3(b) statistic:** `AUC(8 IN) - AUC(24 IN)`, standard attacker, all candidates. Complementary pairs do not exist
   in these designs, so the jackknife unit is the design's block of 4 replicates, left out of both configurations
   (8 blocks, t with 7 df, 97.5%). This is the same leave-one-balanced-unit-out interval as the pair jackknife.
6. **Reported, not gated:** per configuration, M2's `config_summary` at 11 checkpoints; the composite under the 11-checkpoint
   sampling-aware attacker (`rz.sampling_aware` on standard checkpoints), under 51 checkpoints alone and under 51
   checkpoints plus `z`; `z` alone; mean accuracy of the 8-IN and 24-IN runs; the lowest-client accuracy at 0.740.

## 3. Decisions

- **G-P3(a):** G-C1's rule at each target: pass if the difference is at most -0.05 and the 97.5% interval is below 0.
  A target outside either curve is "not testable" and does not count as a pass. The levels above stop at 0.723 and
  0.678, so 0.660 is expected to be not testable.
- **G-P3(b):** pass if the whole 97.5% interval lies within +-0.03. Otherwise escalate: P2b's analysis stays provisional
  and measurement standard v3 needs an IN-count correction.

## 4. Cost and machine

47 run-hours projected (P0's cost runs: composite 15.8, metric privacy 19.2, IN-count 12.3). The cap is 35 GPU-h of
measured occupancy (ROADMAP decision 5), escalation at 43.75. Several workers share a GPU, so occupancy is below summed
run time; both are reported. GPUs are shared with other jobs under ROADMAP decision 4, each run waiting for available
host RAM and GPU memory first.
