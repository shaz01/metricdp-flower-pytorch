# CIA roadmap v3, P1: the paper's 3-client setting under measurement standard v3

Package P1 of `research/cia_roadmap_v3/ROADMAP.md` (direction F3), Agent P, branch `feature/cia-v3-paper-setting`, on
the compute server. Closed and merged by the owner on 2026-11-09.

All numbers come from committed artifacts in `results/cia_v3/paper_setting/`:
- `STATE.md` (design, written before any code change, and the full log) and `PROTOCOL.md` (fixed before any run);
- `analysis.json`, from `analyze.py --main vanilla gdp-z0.01 mp-z0.01 --extension gdp-z0.024 mp-z0.024`;
- `FINDINGS.md` and `figures/` (from `make_figures.py`); the runs in `results/` and `pilot/`.

## 1. Question

The paper (Sáinz-Pardo Díaz et al. 2026, Table 13) reports client-inference AUCs of 0.397 (global DP) and 0.493
(metric privacy) in its 3-client Alzheimer setting and reads them as protection. Each comes from one IN and one OUT
trajectory, pooling 20 dependent checkpoint scores. P1 replays that setting under measurement standard v3: independent
private randomness in every world, 20 pairs per mechanism, and a calibrated attacker.

## 2. Design

- **Setting (the paper's Section 7.4.2):** Alzheimer MRI, 3 clients by Table 9 (target client 3), FedAvg weighted by
  examples, 20 rounds, 5 local epochs, batch 32, lr 0.001, clipping 5. Noise std `z C / n_t` on the aggregate, divided
  by the round's maximum client distance for metric privacy. Shadow sample: 10% of the target's records with 20% pixel
  noise.
- **Runs:** vanilla, global DP and metric privacy at the paper's `z = 0.01`, 20 IN/OUT pairs each (120 runs); a
  declared extension at `z_acc`, where global DP matches the paper's accuracy (80 runs); an accuracy pilot for `z_acc`
  (20 runs). Each world draws its own client and noise seeds from `SeedSequence([design_seed, pair, world])`.
- **Statistics:** the paper's pooled ROC per pair (its distribution over the 20 pairs), and v2's calibrated
  likelihood-ratio attacker over the 40 trajectories with a pair-jackknife 95% interval.
- **Reproduction first:** the analysis code reproduced M2's published vanilla AUC, 0.723, before reporting anything.

## 3. Gate (G-P1, report only)

- Both DP lower bounds above 0.5: "the published protection claim fails in its own setting".
- The paper's statistic spanning 0.5 across pairs (5th percentile below, 95th above): "Table 13 is a single draw from
  a wide distribution".
- Otherwise: "consistent with the paper".

## 4. Results

| Configuration | IN accuracy | Calibrated AUC [95%] | Clean-shadow AUC | Paper's statistic, median (p05-p95) | Pairs below 0.5 |
|---|---|---|---|---|---|
| Vanilla | 0.970 | 0.637 [0.474, 0.801] | 1.000 | 0.749 (0.213-0.913) | 3 |
| Global DP, `z = 0.01` | 0.947 | **1.000 [1.000, 1.000]** | 1.000 | 0.853 (0.732-0.988) | 0 |
| Metric privacy, `z = 0.01` | 0.957 | **1.000 [1.000, 1.000]** | 1.000 | 0.887 (0.822-0.955) | 0 |
| Global DP, `z = 0.024` | 0.672 | **0.973 [0.927, 1.018]** | 0.985 | **0.497 (0.366-0.646)** | **10** |
| Metric privacy, `z = 0.024` | 0.915 | **0.985 [0.956, 1.014]** | 1.000 | 0.812 (0.673-0.928) | 0 |

**G-P1:** main runs, "the published protection claim fails in its own setting"; extension, the same and "Table 13 is a
single draw from a wide distribution".

**Accuracy pilot (global DP, mean IN accuracy over rounds 16-20):** `z` 0.0075 0.961, 0.015 0.893, 0.03 0.501, 0.06
0.447, 0.12 0.501. Log-linear interpolation between 0.015 and 0.03 gave `z_acc = 0.024`, recorded before any extension
run; the extension measured 0.672 against the paper's 0.637.

**Why an uncalibrated statistic falls below 0.5:** on global DP at `z = 0.024`, the IN world's final noisy-shadow loss is
higher than the OUT world's in 19 of 20 pairs (12.6 against 4.3), while its clean-shadow loss is lower (2.25 against
3.59). The final-round threshold attacker therefore scores 0.113 [-0.019, 0.244], near-perfect separation with the sign
reversed. The calibrated attacker learns the direction from reference trajectories and is not fooled.

## 5. Anomalies and deviations

- **Host memory and GPU sharing:** on 2026-10-31 other users' jobs filled the node's RAM to 96.5%, and Ray killed every
  worker; the queue was rebuilt to wait for 25 GB of available RAM. Extension attempts on 2026-11-01 failed three times
  with CUDA out of memory on GPU 0 (another user's 9 GB job) and were moved to GPU 1. No partial run was ever kept.
- **Pilot grid extended:** 0.637 was not bracketed by `z` 0.03-0.12 (accuracy collapses near 0.5), so the grid was
  halved twice (0.015, 0.0075), as PROTOCOL.md allowed.
- **Extension delayed 18 hours** by a queueing bug: `tmux has-session -t p1` matched its own session `p1-ext` by prefix.
- **Evaluation summaries committed by mistake** (220 `*.evaluation.json`, about 87 MB) in 7cc4df8a, untracked in
  bdd4ec1c; they remain in the history.

## 6. Conclusions

- In the paper's own setting, the target client is identifiable under both DP mechanisms, at the paper's noise and at
  its accuracy. The target's records are memorized (clean-shadow IN loss 0.008-0.043 against 1.08-1.65 OUT at
  `z = 0.01`).
- The paper's AUCs below 0.5 are ordinary draws of a high-variance, fixed-direction statistic; an AUC below 0.5 is not
  evidence of protection.

## 7. Cost

Compute server, one worker, GPUs shared with other jobs (ROADMAP decision 4). Summed training time 14.0 run-hours plus
about 14 s of evaluation per run: about 15 GPU-h against the 40 GPU-h cap.
