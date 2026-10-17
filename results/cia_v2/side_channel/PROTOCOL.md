# A2 protocol: metric privacy's noise-level side channel at strong noise, small federations

Package A2 of [research/cia_roadmap_v2/ROADMAP.md](../../../research/cia_roadmap_v2/ROADMAP.md) (Agent A, branch
`feature/cia-v2-side-channel`). Fixed before any A2 run. Hypothesis, design rationale, interval calibration and gate
are in [STATE.md](STATE.md).

## 1. Runs

| Item | Value |
|---|---|
| Runner | `results/cia_v2/side_channel/runner.py`: WP1's small-federation trajectories, layout seed 42, target-only OUT, per-trajectory seeds, `--log-noise-diagnostics` |
| Cells and targets | Alzheimer 3 (client 2), 5 (client 1), 8 (client 5); EuroSAT 3 (client 1) |
| Mechanisms | global DP; the paper's metric privacy. Noise std independent of the active count |
| Noise grid | 1, 4, 16, 64 x WP1's base ratio (EuroSAT 0.001546, Alzheimer 0.01/3) |
| Pilot | global DP only, 4 replicates per world (pilot seeds) |
| Main | global DP and metric privacy at the cell's chosen level, `R = 8` per world (main seeds) |
| Output | `results/cia_v2/side_channel/results/{pilot,main}/<dataset>/clients-<n>/<mechanism>-x<multiple>/` |

## 2. Order

1. **Pilot, per cell.** Run 1x. Compute the calibrated loss attacker's AUC (point estimate, `analyze.cell_summary`
   on the pilot directory). If it is above 0.6, run the next multiple, and so on up to 64x. The cell's level is the
   first multiple with AUC 0.6 or below.
   - If 64x does not qualify, the cell is reported as "no qualifying noise in the grid" and gets no main runs.
   - Pilot trajectories are never reused in the main runs.
2. **Main, per qualifying cell.** Global DP and metric privacy at the chosen level, 8 replicates per world.
3. **Analysis.** `python -m results.cia_v2.side_channel.analyze --out results/cia_v2/side_channel/analysis.json`
   after all main runs.

## 3. Cost

Planning figures from v1 WP1: about 0.07 GPU-h per Alzheimer trajectory and 0.35 per EuroSAT 3-client trajectory
(50 rounds). Each step costs:

- **Pilot:** at most 8 trajectories per cell and level.
- **Main:** 32 trajectories per cell.

The worst case, all four pilot levels in every cell, is about 36 GPU-h; a typical case is about 25 GPU-h.

**Stop rule.** The cost per trajectory is re-measured after the first pilot level. If the projected total exceeds
the 35 GPU-h cap by more than 25%, the package stops and escalates before the main runs.

## 4. Gate

G-A2 as written in ROADMAP.md (verbatim in STATE.md), with the interval and the reading of the global-DP condition
stated there:

- The pair jackknife at 97.5% for the noise-level attacker's lower end.
- The point estimate for global DP's loss AUC.

The gate's false-claim probability is about 3.4% at the worst per-cell rate (STATE.md). FINDINGS reports every cell's
noise-level AUC (blind, 50-round, oracle), loss AUC and update-norm channel, whatever the gate's outcome.

## 5. Machine

Compute server. Runs use a GPU only when one is idle and unclaimed, or with the owner's approval to share.
