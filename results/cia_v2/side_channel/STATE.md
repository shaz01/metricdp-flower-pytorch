package: A2
status: planning (G-M0 passed; protocol and code being prepared; no run yet)
gate: G-A2 = not evaluated yet
protocol: results/cia_v2/side_channel/PROTOCOL.md (to be fixed before the pilot)
machine: compute server (the owner PC is allowed for the Alzheimer cells; this agent runs on the compute server)
cost so far: 0 GPU-h of the 35 GPU-h cap
next: runner (WP1's small-federation combos plus per-replicate seeds), analysis, and a null simulation of the A2-size interval; then the GDP pilot
needs from others: none (GPU time shared with M1/M2 on this machine)

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, A2 (Agent A, branch `feature/cia-v2-side-channel`, from
`feature/cia-v2-measurement` at 539153d8).

## Hypothesis

In small federations, metric privacy's noise level still separates the worlds with and without the largest client
at noise strong enough that the calibrated loss attacker's AUC under global DP is 0.6 or below. The reason is that
the noise std can be estimated from one release to about `1/sqrt(2D)` relative precision, whatever its size
(REVIEW.md Section 5).

## Design

- **Cells.** Alzheimer MRI with 3, 5 and 8 clients, and EuroSAT with 3 clients. Label-Dirichlet alpha 0.3, layout
  seed 42 (fixed, `partition_seed = 42`), 50 rounds, equal weights, WP1's data modules and hyperparameters
  (`results/cia_frontier/side_channel_audit/runner.py`).
- **Targets.** The largest client of each cell in the seed-42 layout, fixed now:

  | Cell | Client sizes | Target |
  |---|---|---|
  | Alzheimer, 3 clients | 1,121 / 1,333 / 2,666 | client 2 |
  | Alzheimer, 5 clients | 302 / 2,328 / 1,622 / 138 / 730 | client 1 |
  | Alzheimer, 8 clients | 236 / 253 / 407 / 307 / 440 / 1,891 / 1,040 / 546 | client 5 |
  | EuroSAT, 3 clients | 3,582 / 9,573 / 8,445 | client 1 |

- **Worlds and replicates.** Removal adjacency: IN (all clients) and OUT (target removed), `R = 8` independent
  replicates per world.
  - Each trajectory has its own `client_seed` and `noise_seed`.
  - Replicate `j` of IN and replicate `j` of OUT are stored as a pair (columns `2j`, `2j + 1`), the layout the pair
    jackknife expects.
  - Seeds come from `SeedSequence([3, cell, world, replicate])`. They are shared across mechanisms and noise levels
    (across configurations), never between worlds.
- **Mechanisms.** Global DP and the paper's metric privacy. Under the frontier convention the noise std is
  independent of the active count: noise multiplier = ratio x active clients.
- **Noise level per cell, from a pilot.**
  - Global DP, 4 replicates per world (pilot seeds; not reused in the main runs).
  - Grid: 1, 4, 16 and 64 times WP1's base ratio (EuroSAT 0.001546, Alzheimer 0.01/3), tried in increasing order.
  - The cell's level is the lowest at which the calibrated loss attacker's point AUC is 0.6 or below. Higher levels
    are not run once one qualifies.
  - If even 64x does not qualify, the cell is reported as "no qualifying noise in the grid" and its main runs are
    skipped.
- **Main runs.** Per cell, global DP and metric privacy at the chosen level, `R = 8` per world: 32 trajectories per
  cell, 128 in total.
- **Cost estimate.** v1 WP1 measured Alzheimer at about 0.2 of EuroSAT per round; small federations cost about as
  much per round as 48 clients. That puts the main runs at about 13 GPU-h and the pilot at at most about 16 GPU-h.
  Both are re-estimated after the first pilot trajectories.

## Attacker and metrics

- **Calibrated loss attacker.** `calibrated_scores` on the target's shadow-loss trajectory at rounds 0, 5, ..., 50.
  It uses balanced references: the trial's own world minus the trial, and the other world minus its nearest replicate.
  Trials are the 16 trajectories of a cell.
- **Noise-level attacker** (blind). Per round, `sigma_hat_t = ||Delta_t|| / sqrt(D)` from the released-update norm
  (`--log-noise-diagnostics`).
  - **Primary feature:** the mean of `sigma_hat_t` over the 50 rounds, a scalar fed to the calibrated likelihood
    ratio. With one target, a cell has only 6 references per world, so a 50-dimensional ratio is not estimable.
  - **Sensitivity check:** the 50-round vector.
  - Also reported: the oracle with the strategy's logged true noise std, and the update-norm channel under global DP.
- **Metrics.** AUC and TPR at 5% FPR per cell, mechanism and attacker; final accuracy.
- **Interval: the pair jackknife at `GATE_LEVEL` (97.5%),** decided by simulation at the A2 size (one target, 8
  pairs, scalar Gaussian feature, calibrated attacker; 400-800 simulated cells per point).

  P(lower end >= 0.8) by expected attacker AUC:

  | Expected AUC | Pair jackknife 97.5% | Pair jackknife 99% | Pair bootstrap | Hanley-McNeil |
  |---|---|---|---|---|
  | 0.657 | 0.013 | | 0.018 | 0.018 |
  | 0.749 | 0.037 | | 0.060 | 0.052 |
  | 0.787 | 0.079 | 0.068 | | |
  | 0.810 | 0.090 | 0.062 | | |
  | 0.879 | 0.223 | | 0.300 | 0.307 |
  | 0.966-0.968 | 0.73-0.76 | 0.68 | 0.785 | 0.805 |

  With 16 trials the AUC is too discrete for any per-cell interval to stay at 5% just below 0.8. G-A2 needs the claim
  in at least two of four cells, though. At the worst per-cell rate (0.079) that gives a gate-level false-claim
  probability of 1 - (0.921^4 + 4 x 0.079 x 0.921^3) = 3.4%. The jackknife at 97.5% is therefore kept, and per-cell
  statements carry their ~8% boundary error in FINDINGS.

## Gate (copied verbatim from ROADMAP.md Section 3, A2)

**G-A2:** a strong claim holds if metric privacy's noise-level AUC is 0.8 or higher at the lower interval end while
global DP's loss AUC is 0.6 or below in at least two cells. Otherwise record the size of the channel at the tested
noise.

"Global DP's loss AUC is 0.6 or below" is read on the main runs' point estimate, the quantity the pilot selects on.
It is reported with its interval.

## Files

| File | Change |
|---|---|
| `results/cia_v2/side_channel/runner.py` | Plans unless `--execute`. WP1's `build_audit_combos` with `partition_seed = 42`, target-only OUT, per-replicate seeds, `--log-noise-diagnostics`; stage `pilot` or `main`. |
| `results/cia_v2/side_channel/analyze.py` | Loss and noise-level attackers per cell, intervals, G-A2 |
| `results/cia_v2/side_channel/tests/` | Planning and analysis tests (no training) |
| `results/cia_v2/side_channel/{PROTOCOL,STATE,FINDINGS}.md`, `analysis.json`, `results/` | This package |

## Log

| When | What |
|---|---|
| 2026-10-17 | Branch and sparse worktree opened. Targets fixed from the seed-42 layouts. STATE written before any A2 code. |
| 2026-10-17 | Runner (`runner.py`, 8 tests): WP1 combos with layout seed 42, target-only OUT, per-trajectory seeds, pilot/main stages. A2-size interval simulation: pair jackknife at 97.5% chosen (table above); the noise-level attacker's primary feature is the scalar mean of `sigma_hat`. |
