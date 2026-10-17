# Escalation (Agent A, M0, 2026-10-17): the v2 design cannot resolve the roadmap's 0.05 AUC thresholds

Status: **open, waiting for an owner decision.** Raised under ROADMAP.md Section 6 (v1 escalation rules). G-M0 itself
passed and is not affected.

## What stops, what continues

- **Continues:**
  - **M1:** its gate uses v1's paired statistic, not the v2 attacker.
  - **The M0 cost run:** it is a cost measurement.
  - **Code work** by every agent.
- **Waits for the owner:**
  - Fixing the replicate count, candidate count or interval method in any protocol that compares configurations or
    applies an AUC threshold: M2, A1, A2, B1-B5, C1.
  - M2 cannot start before G-M1 anyway.

## Finding

M0 item 6 asks for the minimum detectable AUC difference at `R` = 8, 12 and 16, with the rule "if `R = 8` cannot
detect 0.05, screens use `R = 12`". The simulation (`power.py`) uses:

- **Model:** WP2's fixed-layout vanilla runs at seeds 42-45, for per-target IN/OUT means and the replicate covariance.
- **Design:** the specified one: 16 candidates, balanced inclusion, the primary calibrated attacker, and the paired
  two-way bootstrap of Section 2 item 5.

It finds that no `R` in the roadmap's range comes close to 0.05.

**Minimum detectable difference at 80% power** (`tables/power_table.csv`; 200 simulations per point):

| R | Baseline effect (attacker AUC) | All candidates | Top decile |
|---|---|---|---|
| 8 | vanilla-sized (0.76) | not reached, even for the full 0.26 gap down to no leakage | not reached |
| 12 | vanilla-sized (0.78) | 0.24 | not reached |
| 16 | vanilla-sized (0.80) | 0.21 | not reached |
| 8-16 | half of vanilla (0.59-0.63) | not reached | not reached |

**Why** (`tables/power_diagnosis.csv`, `tables/power_options.csv`; `power_options.py`). Two causes add up:

1. **The paired two-way bootstrap is too wide for this design.** Its half-width is 2.4-3.7 times the true
   replicate-to-replicate SD of the difference; 1.96 would be nominal. It resamples the 16 candidates as if they were
   a random sample, but they are a fixed panel. Equal configurations are declared different in 0-0.7% of simulations,
   against a nominal 5%.
2. **Even an exact interval needs more trials.** The true SD of the difference is 0.071-0.085 at `R = 8`,
   0.044-0.047 at `R = 16` and 0.023-0.030 at `R = 32`. An exact interval would therefore detect 0.20-0.24, 0.12-0.13
   and 0.07-0.08 at 80% power.

**What the levers buy** (power to detect a difference of about 0.055; 150 simulations per cell):

| Candidates x replicates | False positives: two-way / replicate-only bootstrap | Power: two-way / replicate-only |
|---|---|---|
| 16 x 16 | 0.7% / 6.7% | 0.07 / 0.37 |
| 16 x 32 | 0.0% / 10.0% | 0.10 / 0.57 |
| 32 x 16 | 0.0% / 8.7% | 0.15 / 0.59 |
| 32 x 32 | 0.7% / 8.0% | 0.31 / 0.78 |

The replicate-only bootstrap conditions on the fixed candidate panel. It is slightly liberal (7-10% false positives),
so a gate would need a calibrated version, for example a studentized interval checked against the null simulation.

**The model is probably pessimistic, by an unknown amount:**

- WP2's seeds also changed the initial model, which v2 shares between replicates.
- v2 configurations share replicate seeds. Their noise is then correlated, which shrinks the variance of differences.
  The simulation draws the two configurations independently.

The first real M2 replicates would re-estimate the variance.

**Single-configuration criteria are affected too.** G-M2 asks for the top-decile AUC's upper interval end to be 0.6 or
below. The top decile is 5 candidates, so at `R = 8` that interval rests on 40 trials. Its two-way half-width is about
0.17-0.2 even for all 16 candidates (STATE.md, finding 3). G-M2 can then pass only if the point estimate is near or
below 0.45.

## Options for the owner

| # | Option | Effect | Cost |
|---|---|---|---|
| A | Keep the design (16 candidates, R = 8/16, two-way bootstrap) and raise the thresholds | Gates at about 0.2 AUC; most B gates and A1's +-0.03 equivalence become undecidable | none |
| B | **Recommended.** Three changes: 32 candidates per run (16 clients always in, 16 of the 32 candidates per replicate, so 32 active of 48 slots); `R = 32` for every comparison with a gate; and a calibrated replicate-only interval for gates, with the two-way interval reported alongside for generalisation beyond the panel | Power about 0.8 at a 0.05 difference (simulation); intervals describe this panel of clients | About 4x the specified runs. M2's 6 configurations at R = 32 are 192 runs, about 96 GPU-h at the estimated 0.5 GPU-h per run (cap 30); the cost run will firm this up |
| C | Middle: 32 candidates, `R = 16`, calibrated replicate-only interval | Power about 0.6 at 0.05; detectable difference about 0.08 | About 2x the specified runs for M2 (R = 16 instead of 8) |
| D | Option B or C plus a smaller M2 grid (vanilla, 1x, 4x, 16x), filling in only if needed | Cuts B's M2 cost to about 64 GPU-h | |

Questions that need a decision whichever option is chosen:

1. Should gate intervals condition on the candidate panel (replicate-only) or generalise over clients (two-way)?
2. Should G-M2 use the top decile (5 clients) or all candidates?
3. Should the compute budget change (Section 1, decision 2)?

The candidate list for 32 candidates would be fixed before any run by the same rule as now: top decile, then size
quantiles.

## Evidence

- `results/cia_v2/measurement/power.py` and `power_options.py`: simulation code. Model inputs are committed WP2
  trajectories only; no training.
- `results/cia_v2/measurement/tables/power_table.csv`, `power_grid.csv`, `power_model.json`, `power_diagnosis.csv` and
  `power_options.csv`.
- STATE.md, findings 1-3 (synthetic checks of the attacker and the bootstrap).
