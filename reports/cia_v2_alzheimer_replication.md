# CIA roadmap v2, AZ: Alzheimer replication of M2 and A1

Owner decision 8 (`research/cia_roadmap_v2/ROADMAP.md` Section 10, 2026-10-20), with the owner's later choices recorded
in `results/cia_v2/alzheimer/STATE.md` and `ESCALATION.md`. Branch `feature/cia-v2-alzheimer`. The owner closed the
package on 2026-10-26.

All numbers below come from committed artifacts in `results/cia_v2/alzheimer/`:

- `STATE.md` (design, the log of every decision and incident), `ESCALATION.md` (the setup that did not train),
  `PROTOCOL.md` (part 1: M2, fixed 2026-10-22; part 2: A1, fixed 2026-10-23; each before its runs).
- `FINDINGS.md` (both parts), `analysis.json` (M2 on Alzheimer, 128 runs in `results/`) and `a1/analysis.json`
  (A1 on Alzheimer, 192 runs in `a1/results/`).
- `calibration/`, `diagnosis/`, `calibration_a1/`: accuracy-only runs, never analyzed for leakage.

## 1. Question

Do the EuroSAT conclusions replicate on a second dataset and model?

1. M2: global DP has no useful empirical regime (the attacker stays above AUC 0.6 wherever accuracy is at least 0.70).
2. A1: at matched accuracy, neither the paper's metric privacy nor private-median metric privacy leaks detectably less
   than global DP.

## 2. Setup, and how it was reached

**The decided setup did not train.** At 48 slots and label-Dirichlet alpha 0.3, Alzheimer MRI clients hold 14-372
records (median 90). Vanilla reached only 0.554 accuracy after 50 rounds, against a majority-class rate of 0.501, so
G-M2's 0.70 accuracy floor was out of reach even without noise (`ESCALATION.md`). WP1's Alzheimer module also needed a
lower minimum client size (10 records instead of its 125, which is infeasible at 48 clients).

**Diagnosis (owner's choice, accuracy only, vanilla, 2 runs each):** alpha 1.0 with 50 rounds 0.740; alpha 100
(near-IID) with 50 rounds 0.780; alpha 0.3 with 100 rounds 0.623. **The owner chose alpha 1.0 with 100 rounds**, which
keeps label skew (the channel B5 found to carry most of EuroSAT's leakage).

**Final setup:**

| Item | Value |
|---|---|
| Data | Alzheimer MRI (5,120 training, 1,280 test images, 4 classes), label-Dirichlet alpha 1.0, layout seed 42, at least 10 records per client, 48 slots |
| Model and training | `paper_cnn`, v1's Alzheimer hyperparameters (clip 5, 5 local epochs, batch 32, lr 0.001), 100 rounds, FedAvg with equal weights |
| Design | M0's: 32 candidates by M0's rule from this layout (top decile 4, 29, 31, 22, 15), 16 IN per replicate, 32 active; `R = 32`, `design_seed = 2`; checkpoints every 10 rounds (11, as on EuroSAT) |
| Shadow samples | 25% of each candidate's training records (at 10% the smallest candidates get one image); a stronger attacker than EuroSAT's, so absolute AUCs do not compare across datasets |
| Private median | `q_0 = 0.4` in `[0.04, 1.3]`, scaled from EuroSAT's settings by the logged client distances |

**Calibration (accuracy only, 2 runs per level):** global DP 0.807 / 0.825 / 0.782 / 0.731 / 0.688 / 0.627 / 0.483 at
`r` = 0.000625 / 0.00125 / 0.0025 / 0.003 / 0.0035 / 0.004 / 0.005; vanilla 0.805. Accuracy is flat to 0.0025 and
collapses by 0.005.

## 3. Part 1: M2 on Alzheimer

Vanilla and global DP at 0.0025, 0.003 and 0.0035 (128 runs). Calibrated attacker, pair jackknife at 97.5%.

| Config | `mu_total` | AUC, all candidates (97.5%) | Two-way | Top-decile AUC | TPR at 5% FPR | Mean final accuracy (95%) |
|---|---|---|---|---|---|---|
| Vanilla | none | 0.564 (0.507-0.620) | 0.492-0.634 | 0.576 | 0.078 | 0.815 (0.809-0.822) |
| Global DP 0.0025 | 125.0 | 0.540 (0.512-0.568) | 0.476-0.599 | 0.586 | 0.041 | 0.784 (0.777-0.791) |
| Global DP 0.003 | 104.2 | 0.523 (0.490-0.556) | 0.463-0.583 | 0.502 | 0.061 | 0.732 (0.724-0.741) |
| Global DP 0.0035 | 89.3 | 0.546 (0.510-0.582) | 0.471-0.616 | 0.472 | 0.086 | 0.670 (0.658-0.681) |

**G-M2: useful empirical regime at `r = 0.0025`** (upper end 0.568, top decile 0.586, accuracy lower end 0.777). This
does not replicate EuroSAT's "no useful regime", and the reason is the baseline: vanilla's AUC is already 0.564.
Global DP's regime is useful because there is little leakage to remove; no difference between levels is
distinguishable.

## 4. Part 2: A1 on Alzheimer

Metric privacy at 0.00125, 0.0025, 0.003 and private median at 0.000625, 0.00125, 0.0016 (192 runs), paired with part
1's design. Targets: global DP's accuracy at `r_0.6` = 0.0025 and at 0.003, **0.784 and 0.732**; global-DP curve:
vanilla and the three global-DP levels. The replacement check ran automatically after stage 1 (metric privacy
0.829 / 0.771 / 0.639, private median 0.829 / 0.741 / 0.525): both targets bracketed, nothing replaced.

| Mechanism | `a* = 0.784` | `a* = 0.732` |
|---|---|---|
| Metric privacy | -0.051 [-0.100, -0.002]: "better" by the gate interval; two-way [-0.125, 0.016] | -0.026 [-0.076, 0.025]: no detectable difference |
| Private median | -0.019 [-0.064, 0.027]: no detectable difference | -0.023 [-0.078, 0.033]: no detectable difference |

| Config | `mu_total` | AUC, all candidates (97.5%) | Top-decile AUC | Mean final accuracy (95%) |
|---|---|---|---|---|
| Metric privacy 0.00125 | none | 0.506 (0.472-0.540) | 0.594 | 0.829 (0.821-0.836) |
| Metric privacy 0.0025 | none | 0.485 (0.436-0.533) | 0.532 | 0.773 (0.762-0.783) |
| Metric privacy 0.003 | none | 0.526 (0.472-0.580) | 0.545 | 0.639 (0.597-0.680) |
| Private median 0.000625 | 685.2 | 0.540 (0.494-0.585) | 0.590 | 0.825 (0.819-0.832) |
| Private median 0.00125 | 342.6 | 0.499 (0.451-0.548) | 0.458 | 0.736 (0.721-0.752) |
| Private median 0.0016 | 267.6 | 0.545 (0.508-0.581) | 0.624 | 0.518 (0.484-0.552) |

**G-A1 on Alzheimer: no robust difference.** The one "better" outcome is fragile: its gate interval excludes 0 by
0.002 and the two-way bootstrap includes 0; the mechanism's AUC there is 0.489, below chance, where the calibrated
attacker's leave-pair-out references bias the null slightly downward; and with four gated outcomes, one false
"better" has a chance of about 5%.

## 5. Reading

- **No-useful-regime is setup-dependent.** With strong label skew (EuroSAT, alpha 0.3) global DP cannot silence the
  attacker at usable accuracy. With milder skew (Alzheimer, alpha 1.0) the attacker is weak to begin with (TPR at 5%
  FPR 0.04-0.09 against 0.17-0.23 on EuroSAT), and DP's added protection is not measurable. That fits B5's finding that
  EuroSAT's leakage is mostly label-distribution inference; this package did not test it directly (no surrogates).
- **The audit claim survives on Alzheimer only weakly:** no mechanism is robustly better than global DP at matched
  accuracy, but near-chance leakage everywhere means the comparison has little to resolve.
- **Formal guarantees stay vacuous** in every usable configuration (`mu_total` 89-685; the paper's mechanism none).

## 6. Anomalies and corrections

- **Setup did not train** (alpha 0.3): escalated, diagnosed, setup changed by the owner (Section 2).
- **Out-of-memory on a shared GPU (2026-10-22).** Three workers with 8 parallel clients each shared GPU 1 with another
  user's growing process. About 1,000 client trainings per worker failed, and Flower aggregated the remaining clients
  silently: 18 of 25 completed runs had rounds with missing clients. They were quarantined outside the repository and
  rerun. Fixes, at the owner's request: the server stops a run at the first round with a failed or missing client
  (`--require-all-clients`, always on in the randomized-inclusion runner), and the runner refuses any run with a short
  round. A scan of every earlier v2 run (M2, A1, M1, A2, B5, calibration) found no such rounds. Workers now train 4
  clients in parallel.
- **Private-median collapse.** At high noise every client update can reach zero norm; the private-median strategy had
  no guard and crashed (one calibration run). It now keeps the previous model for that round, as global DP and metric
  privacy do.
- **False positive in the run check (2026-10-24).** A collapsed round logs no client list, and the check first read it
  as missing clients (one metric-privacy run failed three times). Collapsed rounds are now accepted; the run was redone.
- **GPU sharing.** The owner asked to keep one GPU free for other users; part 1 was relaunched on GPU 1 only, sharing
  it with another user's process by owner approval.
- **Late replacement check avoided.** Unlike A1 on EuroSAT, the stage-1 replacement check ran automatically inside the
  worker chains before stage 2.

## 7. Caveats

- One layout, one skew level, the calibrated loss attacker only; a different model, rounds, skew and shadow fraction
  from EuroSAT.
- `r_0.6` has no lower global-DP level in the grid, so A1's targets are `r_0.6` and the next level, 0.003.
- Private median's third level (0.0016) was not calibrated before the runs.

## 8. Cost

| Step | GPU-h (occupancy) |
|---|---|
| Diagnosis and calibration (including the stopped alpha-0.3 batch) | about 10 |
| Part 1, first launch (corrupted runs quarantined) | about 2.7 |
| Part 1, relaunch (two workers on GPU 1) | about 31.1 |
| Part 2 (three workers on GPU 1, plus one rerun) | about 40 |
| Total | about 84 (the decided cap was 40; part 2 ran by the owner's decision with the overrun known) |

## 9. Conclusions and owner decisions

1. **M2 on Alzheimer:** G-M2 useful regime at `r = 0.0025`, because leakage is near chance even without noise.
2. **A1 on Alzheimer:** no robust difference between the mechanisms and global DP; one fragile nominal "better".
3. **For the paper:** EuroSAT's audit result stands; Alzheimer shows that in a low-leakage setting no mechanism is
   measurably better, and that "no useful regime" depends on how strongly client data is skewed.
4. **Closed by the owner** on 2026-10-26.
