package: M0
status: planning
gate: G-M0 = not evaluated yet
protocol: results/cia_v2/measurement/PROTOCOL.md (to be written before the cost run; this STATE fixes the design)
machine: compute server
cost so far: 0 GPU-h (M0 allows no training beyond the 2-round checks and one cost-measurement run)
next: seed plumbing and noise generators, then the attack library, the two pipeline checks, the randomized-inclusion runner, the cost run, the power table
needs from others: none (Agent B: see "Ownership exception" below)

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, M0 (Agent A, branch `feature/cia-v2-measurement`, worktree
`metricdp-pytorch-feature-cia-v2-measurement`). M1 follows on the same branch, in `results/cia_v2/measurement/m1/`, with
its own STATE file.

## Hypothesis

M0 is infrastructure. It tests two claims:

1. The v2 analysis code recovers a known privacy level. On synthetic trajectories with a known GDP parameter `mu`, the
   leave-one-replicate-out attack pipeline returns the trade-off `TPR(alpha) = 1 - G_mu(alpha) = Phi(mu - Phi^-1(1 - alpha))`
   within its bootstrap interval.
2. With the new seed plumbing, worlds with different `noise_seed` draw independent server noise, and worlds with the
   same `noise_seed` reproduce the v1 coupling.

A secondary, measured quantity: the cost of one randomized-inclusion run, which the roadmap estimates at about 0.5 GPU-h.

## Design

**Seeds** (Section 2, items 1-2). There are four run options, `layout-seed`, `init-seed`, `client-seed` and
`noise-seed`, each defaulting to today's `seed`.

- `layout-seed` draws the partition, each client's train/test split, the server evaluation split and the shadow
  samples. In the EuroSAT frontier data module it is the existing `partition_seed`.
- `init-seed` sets the initial model.
- `client-seed` sets the client round seeds (batches, augmentation, dropout): `client_seed + 100000 * id + round`, as today.
- `noise-seed` seeds a dedicated `np.random.Generator` that draws every server noise vector of global DP and metric
  privacy (a wrapper around Flower's `add_gaussian_noise_inplace`). For the private-median and residual-clipping
  strategies, `noise_seed` replaces `seed`.
- **Committed runs stay reproducible.** Without an explicit `noise-seed`, global DP and metric privacy keep drawing from
  NumPy's global state exactly as now, so committed runs replay bit for bit. The dedicated generator is used only when
  `noise-seed` is set.

**Randomized-inclusion design** (Section 2, item 3; runner `results/cia_v2/randomized_inclusion/runner.py`, plans
unless `--execute`):

- EuroSAT, alpha 0.3, seed-42 layout, 48 slots, 50 rounds, FedAvg with equal weights, WP2 hyperparameters.
- Sixteen candidates, fixed before any run:
  - The five top-decile clients by size: 23, 7, 40, 18, 33.
  - Eleven more by size quantile: sort the other 43 clients by record count and take ranks `round((k + 0.5) * 43 / 11)`
    for k = 0, ..., 10. They are computed once from the seed-42 partition summary and listed in `PROTOCOL.md` before the cost run.
- The other 32 clients always take part.
- **Design matrix** (16 candidates x `R` replicates, drawn once from a recorded `design_seed`):
  - Each candidate is IN in exactly `R/2` replicates, as the roadmap requires.
  - I add one refinement: each replicate includes exactly 8 candidates, so every run has exactly 40 active clients.
    The denominator is then a public constant (40), and the update norm cannot reveal how many candidates are in.
  - The noise std is fixed at `r C` per coordinate, as in v1's frontier convention.
  - If the owner prefers a variable count with a 48-slot denominator (`sum / 48`), the runner supports it as an option.
    It is not the default.
- **Worlds.** A replicate is one run. Candidate `c`'s IN world is the replicates that include it, and its OUT world is
  the replicates that do not.
  - Every replicate has its own `client_seed` and `noise_seed`, derived from `(design_seed, replicate)`.
  - No two runs of one configuration share private randomness. That covers noise, client training seeds, and the
    inclusion of the other candidates.
- **What is shared.** Public randomness only: `init_seed`, `layout_seed`, the server evaluation set and the shadow samples.
  Across configurations (for example two noise levels), replicate `j` may reuse the same `(client_seed, noise_seed,
  inclusion column)` to reduce the variance of differences. It never does so within one configuration.
- **Shadow samples.** The v1 convention: 10% of the candidate's records, stratified, drawn by `layout_seed` (clean view;
  the noisy view is kept for comparability). Each candidate's shadow sample is scored at rounds 0, 5, ..., 50
  (11 checkpoints) in every run.
- **Surrogate sets** (optional, for B5): public test images resampled to each candidate's label histogram, scored at
  the same checkpoints.
- **Logged per round.** The released-update norm, the clipped-signal norm and the expected noise norm, on the server
  side. This avoids per-round checkpoints just for the noise-level attacker.

**Cost run** (the only training in M0): one full randomized-inclusion run, global DP at the frontier noise
(`r = 0.001546`), recorded here as GPU-h, split into training and evaluation.

**Power table.** Simulate the whole analysis with the following inputs:

- **Between-replicate variance:** WP2's fixed-layout vanilla seeds 42-45 (5 targets, no DP), and M1's runs once available.
- **Per-candidate loss gaps:** the same runs.
- **Tests:** two independent configurations, two-way bootstrap, alpha 0.05 two-sided, 80% power.

The output is the minimum detectable AUC difference (all candidates and top decile) for `R` = 8, 12 and 16. Later
protocols copy the matching row. If `R = 8` cannot detect 0.05, screens use `R = 12`.

## Attacker and metrics

All attackers score `(candidate, replicate)` trials and are fitted with leave-one-replicate-out references.

- **Calibrated loss attacker (primary).** A Gaussian likelihood ratio on the candidate's 11-point shadow-loss
  trajectory, in the style of LiRA with difficulty calibration.
  - **Model:** IN and OUT means are per candidate. The per-checkpoint variances are pooled over candidates (the
    fixed-variance LiRA variant), because `R/2 - 1` references per world cannot support per-candidate variances. The
    covariance is diagonal.
  - **References:** for trial `(c, j)`, the other replicates of candidate `c`.
  - **Sensitivity check:** a final-round-only version is reported alongside.
- **Uncalibrated loss threshold (weak baseline).** The final-round shadow loss with one global threshold.
- **Noise-level attacker.** For mechanisms with a data-dependent noise level only (metric privacy). It makes a blind
  per-round std estimate from the released-update norm and fits a likelihood ratio on it with the same references.
- **Metrics.**
  - AUC, and TPR at 5% and 1% FPR. At `R = 8`, 1% FPR rests on 64 OUT trials and is flagged as low resolution.
  - 95% intervals: two-way bootstrap (2,000 resamples) that draws replicates and candidates independently, with
    replacement, and recomputes each metric on the product set.
  - Final-round server test accuracy, and the per-client minimum accuracy.
  - All candidates and the top-decile five, reported separately.
  - The v1 paired clean-view score is computed only where a package asks for it, labelled "effect size", never attack
    success.
- **Pipeline checks** (Section 2, item 7; both are tests run by `uv run pytest`):
  1. **Gaussian calibration.** Synthetic trajectories for 16 candidates and `R` replicates:
     - Per-candidate difficulty offsets, which calibration must remove.
     - IN shifted by `mu / sqrt(11)` per checkpoint (independent unit-variance coordinates, so the total shift is `mu`).
     - `mu` = 0.5, 1 and 2.

     The test checks AUC against `Phi(mu / sqrt 2)` and TPR at 5% and 10% FPR against `1 - G_mu`, each inside the
     bootstrap interval. It also checks that `mu = 0` gives an interval containing AUC 0.5.
  2. **Noise independence.** Real 2-round Flower runs (small EuroSAT federation, capped client data):
     - Case (i): different `noise_seed`.
     - Case (ii): equal seeds.
     - Case (iii): equal `noise_seed` with different `client_seed` (the v1 coupling).

     The statistic is the realized noise energy `||Delta||^2 - ||s||^2 - E||xi||^2` of
     `research/cia_roadmap_v2/scripts/measurement_audit.py`. I evaluate it on 64 fixed coordinate blocks per round,
     because two whole-model values per run would make every correlation +-1. That gives 128 points per run, so the
     null standard deviation of the correlation is about 0.09.

     Pass: (i) has correlation in [-0.3, 0.3]; (ii) and (iii) have correlation above 0.9.

## Gate (copied verbatim from ROADMAP.md Section 3)

**G-M0** passes when the tests are green, the calibration test recovers the Gaussian trade-off and the
noise-independence test separates equal from different seeds. On fail, fix before any other package runs.

On pass: push `feature/cia-v2-measurement` and set this file's status to `ready` (Agent B branches from it).

## Files

| File | Change |
|---|---|
| `experiments/reproduce/server.py`, `experiments/reproduce/runner.py` | The four seed options (default `seed`), recorded in run metadata. The noise generator is attached to the strategy after construction, so `strategy_factory.py` (Agent B) stays untouched. |
| `metricdp_pytorch/globaldp_strategy.py`, `metricdp_strategy.py`, `private_metric_strategy.py`, `residual_clipping_strategy.py` | Optional dedicated noise generator; per-round released-update norm and per-block noise-energy diagnostics (default off) |
| `experiments/cia/` (new `audit_attacks.py`, `bootstrap.py`, `inclusion.py`; tests in `experiments/cia/tests/`) | Calibrated loss, threshold and noise-level attackers; two-way bootstrap; balanced design matrix; subset-inclusion partition view; calibration test |
| `results/cia_v2/randomized_inclusion/` (`runner.py`, `data.py`, `tests/`) | Randomized-inclusion runner (plan by default, resumable, stall guard), surrogate sets |
| `results/cia_v2/measurement/` (`STATE.md`, `PROTOCOL.md`, `power.py`, `tables/`, `tests/`) | This package: noise-independence test, cost run record, power table |
| `experiments/reproduce/client.py` | **Ownership exception** (see below) |
| `STATUS.md` | Agent A's rows only |

**Ownership exception.** M0 item 1 requires client round seeds derived from `client_seed`, and that code lives in
`experiments/reproduce/client.py`, which Section 6 assigns to Agent B. The change is two reads with fallback:

- `client-seed` (default `seed`) for the round seed.
- `layout-seed` (default `seed`) for `client_loaders`.

Nothing else in that file changes. Agent B branches from this branch after G-M0, so there is no conflict. If the owner
objects, the fallback is to pass `client_seed` through the run config's `seed` key and keep `client.py` unchanged.

## Log

| When | What |
|---|---|
| 2026-10-17 | Branch `feature/cia-v2-measurement` from master 7fda7242; sparse worktree opened. Read HANDOFF, REVIEW, ROADMAP, SYNTHESIS and the code paths for seeds and noise (`seed_training`, `_client_round_seed`, `DirichletEuroSAT.partition_seed`, Flower's global-RNG `add_gaussian_noise_inplace`, `default_rng(seed)` in the private-median and residual-clipping strategies). This STATE written before any code change. |
