package: B5, exposure of large clients (owner decision 10: B5 first, then B1)
status: protocol fixed; smoke run, then stage A when a GPU is idle and unclaimed
gate: G-B5 (amended by owner decision 9) = not evaluated yet
protocol: results/cia_v2/mechanisms/b5_large_clients/PROTOCOL.md (fixed 2026-10-20, before any B5 run)
machine: compute server
cost so far: 0 GPU-h of the 15 GPU-h cap (stop and escalate at 18.75)
next: stage A at 2x (cap 326, cap 182, logit adjustment; 96 runs)
needs from others: none (M2's runs are the global-DP baseline; no other agent session is running)

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, B5, as amended by Section 10 (decisions 9 and 10, 2026-10-20).
Branch `feature/cia-v2-mechanisms` (Agent B's), master merged in at 1298b5c1. The owner assigned Agent B's packages to
the agent that ran Agent A's, since no other agent session is running.

## Item 1: diagnostic (done, no training)

M2's runs scored every candidate with a label-matched public surrogate set as well as its true shadow sample
(`results/cia_v2/frontier/analysis.json`, `auc_surrogate`). The roadmap's rule: if the surrogate attacker's AUC is
within 0.05 of the true one, the leakage is mostly label-distribution inference (Wainakh et al. 2022), and the
prior-neutral defenses are worth one screen.

| Config | All candidates: true | surrogate | difference | Top decile: true | surrogate | difference |
|---|---|---|---|---|---|---|
| Vanilla | 0.723 | 0.714 | +0.009 | 0.738 | 0.700 | +0.037 |
| Global DP 1x | 0.720 | 0.697 | +0.024 | 0.741 | 0.726 | +0.015 |
| Global DP 2x | 0.702 | 0.680 | +0.022 | 0.746 | 0.726 | +0.020 |
| Global DP 4x | 0.666 | 0.652 | +0.014 | 0.741 | 0.697 | +0.044 |
| Global DP 8x | 0.671 | 0.646 | +0.025 | 0.724 | 0.677 | +0.047 |
| Global DP 16x | 0.593 | 0.584 | +0.009 | 0.603 | 0.608 | -0.005 |

**Within 0.05 everywhere: the measured leakage is mostly label-distribution inference.** An attacker holding only
public images with a candidate's label histogram, and none of its records, does nearly as well. That fits M2 (noise
barely moves leakage, since the class prior is a strong low-dimensional signal) and A1 (equal accuracy, equal leakage).
Item 3 therefore runs.

## Hypothesis

A defense that removes a client's label prior from what it contributes (item 3), or that limits how much a large
client contributes per round (item 2), lowers the attacker's AUC at the same noise without costing accuracy.

## Design

**Regime: M2's.** Randomized inclusion (32 candidates, 16 IN per replicate, 32 active of 48), EuroSAT alpha 0.3, the
seed-42 layout, 50 rounds, equal weights, WP2 hyperparameters, `R = 32`, `design_seed = 2`. Replicate `j` has M2's
inclusion and `(client_seed, noise_seed)`, so every B5 configuration is paired with M2's global-DP run at the same
noise. Global DP at M2's operating points (decision 10): 2x (`r = 0.003092`, accuracy 0.740) and 4x (0.006184,
0.660).

**Interventions** (client-side, default off; the server and the privacy accounting are unchanged):

- **Item 2, contribution caps.** Each round a client trains on at most `m` of its training records, drawn uniformly
  without replacement with its round seed (a fresh draw every round). `m` at the median and the 25th percentile of the
  48 clients' training-set sizes (seed-42 layout): **326** (caps 24 clients, 17 of 32 candidates, 29% of the records
  per round) and **182** (36 clients, 25 candidates, 53%). The top decile holds 691-920 training records.
- **Item 3, prior-neutral local loss: logit adjustment** (Menon et al. 2020). The local loss is the cross-entropy of
  `softmax(log p + tau log pi_k)`, where `p` is the model's output and `pi_k` client `k`'s own label prior with add-one
  smoothing, `tau = 1`. The model then learns a prior-free posterior; prediction uses `p` unchanged. One screen, as the
  roadmap allows (label calibration, Zhang et al. 2022, is not run).

**Stages.**

- **Stage A, at 2x:** cap 326, cap 182 and logit adjustment: 3 configurations x 32 runs = 96 runs, about 10.6 GPU-h at
  M2's 0.11 GPU-h per run.
- **Stage B, at 4x:** for each stage-A intervention with a point AUC reduction of at least 0.03 and an accuracy loss of
  at most 1 pp at 2x, decided after stage A is complete. At most 96 runs; above the cap, stage B escalates first.

## Attacker, metrics, gate

- Calibrated loss attacker on the 11-checkpoint trajectory, all candidates and top decile (as M2).
- **Paired difference** to M2's global-DP run at the same noise: `AUC_B5 - AUC_GDP`, pair jackknife at 97.5% (the same
  pairs left out of both); two-way difference bootstrap alongside.
- Accuracy difference with a paired bootstrap over replicates; the lowest client accuracy; the surrogate attacker is
  not scored (no `--surrogates`).
- **G-B5 (decision 9):** all-candidate AUC at least 0.05 below global DP at the same noise, with the gate interval of
  the difference excluding 0, at no more than 1 pp accuracy loss (point estimate). The top decile is reported with its
  interval but not gated.
- **Power:** A1's estimate from M2's runs puts the detectable all-candidate difference at about 0.03 for paired runs,
  so a true 0.05 reduction should be detected.

## Files

| File | Change |
|---|---|
| `experiments/reproduce/client.py` | Default-off `contribution-cap` (per-round subsample) and `logit-adjustment-tau` (local prior from the client's labels) |
| `experiments/reproduce/paper_training.py` | `train_with_adam(..., log_prior=None, logit_tau=0.0)` |
| `experiments/reproduce/runner.py` | `--contribution-cap`, `--logit-adjustment-tau` into the run config |
| `results/cia_v2/randomized_inclusion/runner.py` | Extra local-training options per configuration (`--contribution-cap`, `--logit-adjustment-tau`), recorded in the manifest only when set, so M2's manifests stay unchanged |
| `results/cia_v2/mechanisms/b5_large_clients/` | STATE, PROTOCOL, FINDINGS, `analyze.py` with tests, results |

## Log

| When | What |
|---|---|
| 2026-10-20 | Master merged into the branch (1298b5c1). Item 1 diagnostic from M2's committed analysis: surrogate within 0.009-0.025 of the true AUC (all candidates). Cap values computed from the layout. STATE written before any code. |
| 2026-10-20 11:20 | Code: default-off contribution cap and logit-adjusted loss (client, training loop, runner flags, randomized-inclusion options; manifests record them only when set, and M2's 437 manifests regenerate unchanged); 7 tests. `analyze.py` (paired differences to M2, G-B5) with 3 tests. 517 related tests pass. PROTOCOL.md fixed. |
