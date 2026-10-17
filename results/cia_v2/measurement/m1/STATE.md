package: M1
status: planning (blocked by G-M0)
gate: G-M1 = not evaluated yet
protocol: this file (Design section); M1 repeats v1's consensus-KL step 3 global-DP arm with one change
machine: compute server
cost so far: 0 GPU-h of the 6 GPU-h cap
next: runner and analysis code while M0 finishes; launch after G-M0 passes and a GPU is idle and unclaimed
needs from others: none

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, M1 (Agent A, branch `feature/cia-v2-measurement`).

## Hypothesis

The flat noise response of v1's consensus-KL step 3 is an artifact of shared noise. In v1, global DP at 1, 2 and 4 times
the frontier noise gave paired clean-view scores of 0.788, 0.808 and 0.792. In that design the IN and OUT trajectories
drew identical server noise, and shared the training seed too. With independent noise and client seeds per world,
the 4x paired score falls to 0.70 or below (REVIEW.md predicts 0.621 or lower).

## Design

The design is v1's consensus-KL step 3 global-DP arm (`results/cia_frontier/consensus_kl/runner_dp.py`, mechanism
`gdp`), changed only in its private randomness:

- **Setup.** EuroSAT, alpha 0.3, seed-42 layout, 48 clients, 50 rounds, equal weights, WP2 hyperparameters, WP0
  diagnostics. Global DP with noise std `r C`, fixed across removal (noise multiplier `r` times the active clients),
  at `r = 0.001546` (1x) and `r = 0.006184` (4x).
- **Worlds and targets.** Removal adjacency with targets 23, 7, 1, 0, 3. Per noise level there is one IN trajectory
  and five OUT trajectories, one per target. That is 12 trajectories, estimated at about 4 GPU-h; v1 measured 0.27-0.42
  GPU-h per trajectory.
- **Private randomness.** Each of the 6 worlds (IN, OUT-23, OUT-7, OUT-1, OUT-0, OUT-3) gets its own `client_seed`
  and `noise_seed`, drawn from `SeedSequence([1, world])` (world 0 = IN, worlds 1-5 = OUT in target order).
  No two worlds share private randomness.
- **Shared randomness.** The same world at 1x and 4x reuses its seeds, which is allowed across configurations.
  Public randomness is shared as in v1: `seed` = 42, so the same initial model, layout, server test split and
  shadow samples.
- **Evaluation.** Unchanged from v1: frontier evaluation path, shadow samples drawn by the layout seed, both views.

## Attacker and metrics

M1 deliberately keeps v1's statistic (ROADMAP.md: "M1 deliberately keeps v1's statistic and changes only the noise
coupling"):

- **Paired clean-view score.** Per target, the fraction of rounds 1-50 at which the OUT trajectory's clean shadow
  loss exceeds the IN trajectory's. Pooled over the five targets with v1's `cell_statistics`
  (`results/cia_frontier/signature_suppression/analyze.py`, read through its `load`).
- **Interval.** v1's 2,000-resample cluster bootstrap over targets. With one seed this gives five clusters.
- **Also reported:** the oracle-midpoint AUC, the top-decile score (targets 23 and 7), final accuracy, and the
  realized-noise-energy correlation between the IN and each OUT trajectory. The correlation verifies the
  decoupling in the real runs; it should be near 0.
- **Label.** Following Section 2 item 6 of the measurement standard, the paired score is reported as an effect
  size, not as attack success.
- **Coupled reference values:** 0.788 (1x) and 0.792 (4x), from `results/cia_frontier/consensus_kl/analysis_step3.json`.

## Gate (copied verbatim from ROADMAP.md Section 3, M1)

**G-M1** (fixed now):

| 4x decoupled paired score | Verdict | Consequence |
|---|---|---|
| 0.70 or below | Confirmed (prediction: 0.62 or below) | v1 DP leakage numbers flagged; M2 proceeds |
| Between 0.70 and 0.75 | Partial | M2 proceeds; the 1x result is reported alongside |
| 0.75 or above | Not material | The flat noise response is real; escalate, since B4 then becomes the main line |

The verdict uses the point estimate of the 4x pooled score, as the table states. The interval is reported next to it.

## Files

| File | Change |
|---|---|
| `results/cia_v2/measurement/m1/runner.py` | Plans unless `--execute`. Builds v1's step-3 `gdp` combos, adds the per-world seeds, and executes through WP2's stall-guarded executor. |
| `results/cia_v2/measurement/m1/analyze.py` | The v1 statistic through v1's `load` and `cell_statistics`, plus the noise-energy correlation check. |
| `results/cia_v2/measurement/m1/tests/` | Planning and seed-independence tests (no training) |
| `results/cia_v2/measurement/m1/{STATE,FINDINGS}.md`, `analysis.json`, `results/` | This package |

## Log

| When | What |
|---|---|
| 2026-10-17 | STATE written before any M1 code; M1 waits for G-M0. |
