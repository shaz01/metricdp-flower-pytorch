package: M2
status: planning (blocked by G-M1; protocol and analysis code being prepared)
gate: G-M2 = not evaluated yet
protocol: results/cia_v2/frontier/PROTOCOL.md (to be fixed after the M0 cost run and G-M1, before any M2 run)
machine: compute server
cost so far: 0 GPU-h of the 64 GPU-h cap (raised from 30 by the ESCALATION.md resolution of 2026-10-17)
next: analysis code and tests; protocol once the M0 cost run gives the cost per run; launch only after G-M1
needs from others: M1 (G-M1 verdict), M0 cost run (GPU-h per run)

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, M2 (Agent A, branch `feature/cia-v2-frontier`, branched from
`feature/cia-v2-measurement` at 158b2380). The design follows the measurement standard as amended by
`results/cia_v2/measurement/ESCALATION.md` (resolution, 2026-10-17).

## Hypothesis

With independent private randomness between worlds, global DP's attacker-realistic leakage falls with noise. There
is a noise level `r_0.6` at which the calibrated attacker's AUC upper gate-interval end is 0.6 or below (all
candidates, top-decile point estimate 0.6 or below) while mean accuracy stays at 0.70 or above: a useful empirical
regime. The alternative is that leakage stays above 0.6 until accuracy collapses.

## Design

- **Federation.** Randomized inclusion (`results/cia_v2/randomized_inclusion/runner.py`):
  - EuroSAT, alpha 0.3, seed-42 layout, 48 slots, 50 rounds, FedAvg with equal weights, WP2 hyperparameters.
  - 32 candidates (top decile 23, 7, 40, 18, 33, plus 27 by size quantile); 16 of them are IN per replicate, so 32
    active clients.
  - Checkpoints at rounds 0, 5, ..., 50, with surrogate sets scored too (for B5).
- **Replicates.** `R = 32` per configuration with `design_seed = 2`, fixed now. Every configuration uses the same
  design, so replicate `j` has the same inclusion and the same `(client_seed, noise_seed)` in every configuration.
  That pairs the configurations, which is allowed across configurations. Within a configuration no two runs share
  private randomness.
- **Configurations.**
  - Vanilla, and global DP with noise std `r C` at `r` = 1x, 4x and 16x the frontier ratio 0.001546 (0.001546,
    0.006184, 0.024736).
  - Noise multiplier `r x 32`.
  - **Fill-in rule** (fixed now, uses accuracy only): after stage 1, add 2x if the stage-1 mean accuracy at 1x
    exceeds that at 4x by more than 0.05; add 8x on the same rule for 4x vs 16x.
- **Two stages per configuration.**
  - Stage 1: replicates 0-15 (8 complementary pairs).
  - Stage 2: replicates 16-31, always run.
  - No leakage statistic is computed before both stages are complete.
- **Cost.** 4 configurations x 32 runs = 128 runs, plus 32 per fill-in level. The cap of 64 GPU-h assumes 0.5 GPU-h
  per run and is re-checked against the M0 cost run before the protocol is fixed.

## Attacker and metrics

- **Calibrated loss attacker (primary).** `experiments/cia/audit_attacks.py:calibrated_scores` on the 11-checkpoint
  shadow-loss trajectory.
- **Also reported:** the final-round-only calibrated attacker and the uncalibrated final-round threshold.
- **Metrics.**
  - AUC, and TPR at 5% and 1% FPR, for all candidates and for the top decile.
  - Gate interval: `pair_jackknife` at 97.5% (`GATE_LEVEL`). The two-way bootstrap (2,000 resamples) is reported
    alongside.
  - Mean final-round accuracy, with a 95% interval from a bootstrap over replicates; also the per-client minimum.
  - **Formal `mu`.** Equal weights over 32 active clients give `mu_total = sqrt(50) / (32 r)`. Vanilla has no
    guarantee.
  - `r_0.6` and the accuracy at `r_0.6`.
- **Power** (copied from `results/cia_v2/measurement/tables/power_table.csv`, `R = 32`, 32 candidates): the difference
  detectable at 80% power is 0.087 (all candidates) at a vanilla-sized baseline and 0.088 at half of it. For the top
  decile it is 0.196.
- **Shared-noise check.** The realized-noise-energy correlation between configurations at equal replicate seeds is
  reported as a check of the pairing.

## Gate

ROADMAP.md text (verbatim):

> **G-M2:**
>
> - **Useful empirical regime:** at some noise the top-decile AUC's upper end is 0.6 or below while the lower interval
>   end of mean accuracy (bootstrap over replicates) is 0.70 or higher. Then A1, B1-B3 and B5 use `r_0.6` and the next-lower noise as operating points.
> - **No useful regime:** leakage stays above 0.6 until accuracy falls below 0.70. The roadmap then escalates with the
>   recommendation to make B4 the main line and frame the paper around the audit claims.

As applied (ESCALATION.md resolution, item 4), at a noise level the useful regime holds when all three conditions are
met:

1. The all-candidate AUC's upper gate-interval end is 0.6 or below.
2. The top-decile point estimate is 0.6 or below.
3. The lower interval end of mean accuracy is 0.70 or higher.

The top-decile interval is reported but not gated.

## Files

| File | Change |
|---|---|
| `results/cia_v2/frontier/analyze.py` | M2 and A1 analysis: attackers, intervals, accuracy, formal `mu`, `r_0.6`, G-M2 |
| `results/cia_v2/frontier/tests/` | Synthetic tests of the analysis |
| `results/cia_v2/frontier/{PROTOCOL,STATE,FINDINGS}.md`, `analysis.json`, `figures/`, `results/` | This package |

## Log

| When | What |
|---|---|
| 2026-10-17 | Branch `feature/cia-v2-frontier` from `feature/cia-v2-measurement` at 158b2380; sparse worktree opened. STATE written before any M2 code. M1 and the M0 cost run are still waiting for an idle GPU. |
