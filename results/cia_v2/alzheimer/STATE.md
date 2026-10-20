package: AZ, Alzheimer replication of M2 and A1 (owner decision 8)
status: calibration waiting for an idle, unclaimed GPU
gate: G-M2 and G-A1 as applied on EuroSAT = not evaluated yet
protocol: results/cia_v2/alzheimer/PROTOCOL.md, fixed after the accuracy calibration and before any leakage run
machine: compute server
cost so far: 0 GPU-h of the 40 GPU-h cap (decision 8); stop and escalate at 50
next: calibration batch 1 (18 runs), private-median settings, batch 2; then PROTOCOL.md
needs from others: none

Owner decision 8 (`research/cia_roadmap_v2/ROADMAP.md` Section 10, 2026-10-20): M2 and A1 repeated on Alzheimer MRI,
48 slots, alpha 0.3, randomized inclusion as in M0, cap 40 GPU-h, gates as applied on EuroSAT. Branch
`feature/cia-v2-alzheimer` from master 2bce957e.

## Hypothesis

The EuroSAT conclusions carry over to a second dataset and model:

1. **M2:** no useful empirical regime. Global DP's calibrated-attacker AUC stays above 0.6 wherever accuracy is 0.70 or
   higher.
2. **A1:** at matched accuracy, neither the paper's metric privacy nor private-median metric privacy leaks detectably
   less than global DP.

v1 makes (1) plausible but not certain. On Alzheimer at 48 clients (v1's non-IID mode, 100 rounds) global DP held
accuracy near 0.91 up to a noise ratio of 0.00125 and then fell off a cliff (0.888 at 0.0025, 0.50 at 0.005), so the
usable band between "no noise effect" and "collapse" is narrow.

## Design

**Federation** (as M2, with Alzheimer's data and model):

- Alzheimer MRI (`Falah/Alzheimer_MRI`; 5,120 training and 1,280 test images; 4 classes, Non_Demented 50.1% of the
  training split). Label-Dirichlet alpha 0.3, layout seed 42, 48 slots.
- **Minimum client size 10 records** (the EuroSAT frontier's rule). WP1's Alzheimer module redraws until every client
  holds 125 records, which is infeasible at 48 clients (about 107 records per client on average). At seed 42 the
  sizes are 14-372 records, median 90.
- `experiments.reproduce.paper_cnn` with v1's Alzheimer hyperparameters
  (`results/contest_at_scale/auc_frontier/alzheimer_remove.py`: clip 5, 5 local epochs, batch 32, lr 0.001), 50
  rounds, FedAvg with equal weights.
- **Candidates:** M0's rule on this layout: the 5 largest clients, then 27 by size quantile; the other 16 always take
  part, so 32 active clients. Fixed from the layout before any run.
- **Shadow samples: 25% of each candidate's training records** (stratified, without augmentation), scored at rounds
  0, 5, ..., 50. EuroSAT used 10%. At 10% a median Alzheimer client would give about 7 images and the smallest
  candidates 1-2, so the loss feature would be mostly sampling noise. 25% gives about 18 for a median client. This
  is a stronger attacker than EuroSAT's, so absolute AUCs are not comparable across datasets; the replication
  compares conclusions (gate outcomes), not numbers.
- No surrogate scoring (B5's diagnostic is on EuroSAT).

**Replicates.** `R = 32`, `design_seed = 2`, two stages (replicates 0-15, then 16-31, always run). Replicate `j` shares
`(client_seed, noise_seed)` and inclusion across configurations.

**Accuracy calibration** (accuracy only, `design_seed = 3`, 2 runs per level, never analyzed), as in A1:

1. Vanilla, global DP and the paper's metric privacy at several noise ratios around v1's cliff.
2. The private-median quantile settings (initial value and bounds) are set from the metric-privacy runs' logged
   client distances, since EuroSAT's (0.3 in [0.03, 1.0]) come from a different model.
3. Private median at several ratios.

**M2-AZ.** Vanilla plus global DP at levels chosen from the calibration to span accuracy from near vanilla to below
0.70, with M2's fill-in rule between adjacent levels (add the midpoint level if their stage-1 mean accuracies differ by
more than 0.05). Fixed in PROTOCOL.md.

**A1-AZ.** Targets: M2-AZ's two global-DP levels around 0.70 accuracy (M2's rule for "no useful regime"), or, if M2-AZ
finds a useful regime, `r_0.6` and the next-lower level. Levels per mechanism from the calibration so that both targets
are bracketed, with A1's replacement rule.

## Attacker, metrics, gates

As on EuroSAT, with the same code (`results/cia_v2/frontier/analyze.py`, `results/cia_v2/frontier/a1/analyze.py`,
generalised to read the candidate panel from the run manifests):

- Calibrated loss attacker; all candidates and the top decile; pair jackknife at 97.5%; two-way bootstrap alongside.
- **G-M2 as applied:** all-candidate AUC upper gate end at most 0.6, top-decile point at most 0.6, lower accuracy end
  at least 0.70.
- **G-A1:** better / equivalent (within ±0.03) / worse or inconclusive, with "no detectable difference at this power"
  for an interval inside ±0.10 that contains 0.

## Files

| File | Change |
|---|---|
| `results/cia_v2/randomized_inclusion/runner.py`, `data.py` | Dataset profiles (`--dataset eurosat`, the default, or `alzheimer`); EuroSAT plans and manifests unchanged |
| `results/cia_v2/randomized_inclusion/analysis.py` | Candidate panel read from the manifests |
| `results/cia_v2/frontier/analyze.py`, `a1/analyze.py` | Top decile read from the audit instead of the EuroSAT constant |
| `results/cia_frontier/side_channel_audit/data.py` | `DirichletAlzheimer` gets a `min_records` argument (default unchanged, 125) |
| `results/cia_v2/alzheimer/` | This package: STATE, PROTOCOL, FINDINGS, calibration, results, analyses |

## Log

| When | What |
|---|---|
| 2026-10-20 | Branch and sparse worktree opened. STATE written before any code. |
| 2026-10-20 10:30 | Code: dataset profiles (`--dataset alzheimer`), panel read from manifests, `DirichletAlzheimer(min_records)`. All 437 committed EuroSAT manifests regenerate identically, and the M2 and A1 analyses reproduce byte for byte. Alzheimer candidates fixed from the layout (top decile 33, 37, 22, 43, 28; 27 more of 15-197 records); shadow samples at 25% hold 3-74 images (median 21). 86 cia_v2 tests pass. |
| 2026-10-20 10:40 | 1-round CPU smoke run (metric privacy, r = 0.00125, scratch output): 32 active clients, accuracy 0.011 at round 0 and 0.359 after round 1, noise std = r C / max pairwise distance (0.00453 = 0.00625 / 1.378), median pairwise client distance 0.89 (EuroSAT about 0.4-0.5). 119 s training and 17 s scoring on the CPU. |
| 2026-10-20 10:45 | Calibration batch 1 fixed (accuracy only, design seed 3, 2 runs per level, output `calibration/`): vanilla; global DP and metric privacy at r = 0.000625, 0.00125, 0.0025 and 0.005 (around v1's cliff). Batch 2 (private median, 4 levels) follows once batch 1's metric-privacy distances fix its quantile settings. Both GPUs are busy with another user's jobs, so a launcher waits for an idle, unclaimed GPU. |
