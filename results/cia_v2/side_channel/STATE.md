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
  (`--log-noise-diagnostics`). The 50-round vector goes into the same calibrated likelihood ratio. Also reported: the
  oracle with the strategy's logged true noise std, and the update-norm channel under global DP.
- **Metrics.** AUC and TPR at 5% FPR per cell, mechanism and attacker; final accuracy.
- **Interval.** The pair jackknife (`pair_jackknife`) at `GATE_LEVEL` is calibrated for 32 candidates and `R = 32`,
  not for one target and 8 pairs. Before the protocol is fixed, a null and effect simulation at the A2 size decides
  between that interval, a wider level and an exact permutation bound for the AUC. The chosen method must give at
  most 5% false positives for the claim "lower end at least 0.8".

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
