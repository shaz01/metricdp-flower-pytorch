# WP6 (D4 item 2): residual-clipping screen

Branch `feature/residual-clipping`. Written 2026-10-15, before any protocol run; the commit that adds this file is
recorded in `STATE.md`. Specification: `research/cia_review_2026-10/agent_roadmap.md` WP6 (binding) and
`research_directions.md` D4 item 2. Mechanism and proof: `THEORY.md`. Changes after the first protocol run go into the
dated Amendments section, committed before the affected runs; no amendment may change the gate.

## 1. Question

Does clipping each client's residual around the previous released update, `c_t + (1/n) sum_i clip(u_i - c_t, rho)`,
let the server add less noise at the same guarantee than plain clipping, without costing more than 2 pp accuracy and
while lowering CIA leakage for the largest clients? D1 is not alive (WP2), so no D1 composition.

## 2. Design (fixed)

| Item | Value |
|---|---|
| Data and model | EuroSAT, EurosatCNN, label-Dirichlet alpha 0.3, 48 slots, layout seed 42 (frontier data module) |
| Training | 50 rounds, FedAvg update rule, 5 local epochs, batch 32, Adam lr 0.001; equal weights; public slot count 48 as the denominator |
| Seed | 42; seed 43 with the seed-42 layout only as the extension (Section 5) |
| Targets | The top decile of the seed-42 partition, fixed now: clients 23 (1,150 records), 7 (1,089), 40 (917), 18 (873), 33 (864) |
| Adjacency | Removal: one IN trajectory and one OUT trajectory per target and mechanism |
| Guarantee | Every mechanism is `(1/z_plain)`-GDP per round with `z_plain = 0.001546 x 48 = 0.0742` (the committed frontier noise: plain clipping `sigma = 0.001546 x C`), i.e. 13.48 per round, 95.3 over 50 rounds (vacuous, as every committed DP run in this repository) |
| Logging | WP0 diagnostics (shadow losses, increments, gradient norms; leave-one-out residual norms) plus per round `rho_t`, noise std, residual norms, fraction clipped, centre norm |

Mechanisms (all equal weights, slot denominator 48; `THEORY.md` Section 1):

| Name | Centre | `rho` | `z` | Role |
|---|---|---|---|---|
| `plain` | zero | fixed `C` = 5 | 0.0742 | baseline (plain clipping at the same guarantee) |
| `residual-fixed` | previous | fixed `C` = 5 | 0.0742 | diagnostic: centring alone, same noise as plain |
| `residual-median` | previous | private median of residual norms (initial 5, bounds [0.05, 20]) | 0.0782 | candidate |
| `residual-q90` | previous | private 0.9-quantile (same settings) | 0.0782 | candidate |

The private `rho` uses 10% of each round's `mu^2` (bit noise std 0.2347 on the count), the Andrew et al. step 0.2.

Trajectories: 4 x 6 = 24.

## 3. Metrics

As in WP2 (`results/cia_frontier/signature_suppression/PROTOCOL.md` Section 4), computed by `analyze.py`: per target and
round `g = L_OUT - L_IN` of the clean shadow loss; **clean-view score** = fraction of rounds with `g > 0`, pooled over
targets; AUC with the per-round midpoint reference; TPR at 5% and 1% FPR; noisy view secondary. Utility: server test
accuracy at round 50, mean over a mechanism's six trajectories. Intervals: 2,000-resample cluster bootstrap (seed
1729), cluster = target-seed pair, paired across mechanisms. All five targets are top-decile targets. Mechanism
diagnostics per round: median `rho_t`, noise std, fraction of residuals clipped.

## 4. Gate G6 (copied from agent_roadmap.md Section 3)

| Gate | Pass | On pass | On fail |
|---|---|---|---|
| G6 | Residual clipping costs at most 2 pp accuracy vs plain clipping at the same guarantee and reduces CIA for top-decile targets | Record; owner decides on composition with D1 | D4 item 2 stops |

How the statistics are formed (fixed now), for each candidate `v` in {`residual-median`, `residual-q90`}:

- Accuracy condition: `acc(v) >= acc(plain) - 0.02`.
- Reduction: `R(v) = S(plain) - S(v)` (clean-view score over the top-decile targets) with its paired bootstrap interval.
- `v` **holds** if the accuracy condition holds and the lower interval end of `R(v)` is above 0; `v` **fails** if the
  accuracy condition fails or the upper interval end is at most 0; otherwise `v` is open.
- **G6 passes** if some candidate holds, **fails** if both candidates fail, and is otherwise **inconclusive**.

## 5. Inconclusive results and the extension

If G6 is inconclusive: seed 43 with the seed-42 layout for the open candidates and `plain` (at most 18 trajectories,
about 6.7 t-h), then recompute with 10 clusters. If it is still inconclusive, stop and escalate. No further runs.

## 6. Prediction (stated before the runs)

`THEORY.md` Section 3 predicts a fail: residuals in this federation are at least as large as updates (pre-analysis),
and the centre carries the previous round's noise (`kappa` = 0.876), so a quantile-tracked `rho` settles near 7 (above
`C`), giving about 1.5 times plain clipping's noise. The screen tests this.

## 7. Compute and running

24 trajectories x 50 rounds, 8.9 t-h planned (`runner.py` prints it) of the 60 t-h cap. Measured WP2 trajectories on
the compute server took 0.23-0.30 h. Running: as WP2 (`results/cia_frontier/signature_suppression/PROTOCOL.md`
Section 8) with `python -m results.cia_frontier.residual_clipping.runner --execute`; workers share the plan, stalled
trajectories are retried, results go to `results/stage_a/<mechanism>/<run>/` with the mechanism pinned in
`residual_config.json`. GPU claims in the shared claims file; worker logs outside the repository.

## 8. Smoke tests (compute server, at most 5 rounds, scratch output)

Every mechanism's IN trajectory for 2 rounds, and `plain` plus `residual-median` end to end through `analyze.py`.

## 9. Known limitations

- One seed, five clusters (the extension adds a second seed).
- The shared guarantee is vacuous (`mu` 95 over 50 rounds); the comparison is about noise and accuracy at matched
  formal parameters, as the gate asks, not about a meaningful privacy level.
- The private quantile tracks the residual norm including the inherited noise; this is the mechanism as specified.

## Amendments

None.
