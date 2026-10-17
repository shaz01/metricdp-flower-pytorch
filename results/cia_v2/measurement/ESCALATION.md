# Escalation (Agent A, M0, 2026-10-17): the v2 design cannot resolve the roadmap's 0.05 AUC thresholds

Status: **resolved 2026-10-17.** The owner delegated the decisions to Agent A ("pick what comes logical to you"); the
resolution is in the last section, and the owner may override any part of it. Raised under ROADMAP.md Section 6 (v1
escalation rules). G-M0 itself passed and is not affected.

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

## Resolution (2026-10-17, decided by Agent A under the owner's delegation)

The owner replied "pick what comes logical to you for the questions you asked to me". The choices below replace the
corresponding parts of the measurement standard (ROADMAP.md Section 2, items 3, 5 and 8) for Agent A's packages, and are
offered to Agent B for theirs. Everything else in Section 2 stands.

1. **Design: option D.**
   - 32 candidates per run, fixed by the same rule as before: the top decile (23, 7, 40, 18, 33), then 27 clients by
     size quantile. The other 16 clients always take part. The original 16-candidate panel is a subset.
   - Each replicate includes 16 candidates, so every run has 32 active clients of the 48 slots.
   - `R = 32` for every comparison with a gate. M2's grid starts with vanilla, 1x, 4x and 16x; 2x and 8x are added
     only if the accuracy curve needs them.
2. **Gate interval: conditional on the fixed candidate panel.**
   - Leave-one-pair-out jackknife over the complementary replicate pairs, with `t` quantiles on `P - 1` degrees of
     freedom (`pair_jackknife`, `pair_jackknife_difference`).
   - Simulation (32 candidates, 200 simulations): at `R = 32`, 4.5% false positives and power 0.745 at a 0.053
     difference. At `R = 16` it is liberal (9%), so **no gate is applied at `R = 16`**.
   - The two-way bootstrap is reported alongside every gate interval as the conservative interval that generalises
     beyond the panel.
3. **Two stages instead of `R = 16` screens.** A package may run replicates 0-15 first (8 complementary pairs, a
   balanced design on its own) to inspect accuracy, for example to choose M2's fill-in noise levels. It then always
   completes replicates 16-31 before any gate statistic is computed. The second stage does not depend on the first
   stage's leakage, so this is not optional stopping.
4. **G-M2, as applied by Agent A.** The useful empirical regime holds at a noise level when two conditions are met:
   - The all-candidate AUC's upper gate-interval end is 0.6 or below, and the top-decile point estimate is 0.6 or
     below. The top-decile interval is reported but not gated: 5 clients cannot narrow it.
   - The lower interval end of mean accuracy is 0.70 or higher.
5. **Budget.** M2's cap rises from 30 to 64 GPU-h: 4 configurations x 32 runs at the estimated 0.5 GPU-h, to be
   re-estimated from the cost run before M2's protocol is fixed. Caps for A1, A2 and the B packages are set when their
   protocols are written, from the measured cost. The roadmap total will rise accordingly; each new cap is recorded in
   that package's STATE.md and in STATUS.md, so the owner can stop any of them.

**Open for later protocols, not decided here.** At `R = 32` the gate interval's half-width is about 0.05. G-A1's
"equivalent: within +-0.03" outcome is therefore close to unreachable. A1's protocol will state this and report the
interval rather than widen the band silently.
