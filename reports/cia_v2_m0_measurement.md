# CIA roadmap v2, M0: measurement infrastructure

Package M0 of `research/cia_roadmap_v2/ROADMAP.md` (Agent A, branch `feature/cia-v2-measurement`, 2026-10-17). M0
built the measurement standard that every later v2 package uses. It trains only one model, the cost run. The owner
closed the package on 2026-10-20.

All numbers below come from committed artifacts in `results/cia_v2/measurement/`:

- `STATE.md`: design, the synthetic checks, the gate result and the log.
- `PROTOCOL.md`: the candidate list and the cost-run command, written before the cost run.
- `ESCALATION.md`: the power problem and its resolution.
- `tables/`: `power_table.csv`, `power_grid.csv`, `power_model.json`, `power_diagnosis.csv`, `power_options.csv`,
  `interval_calibration.csv`, and the original design's tables in `tables/spec16/`.
- `cost_run/`: the cost run's JSONs.

## 1. Question

The v2 review (`research/cia_roadmap_v2/REVIEW.md`) found that v1's leakage measurements shared private randomness
between the worlds they compared. The IN and OUT trajectories drew identical server noise and identical client
training seeds. The review's hypothesis was that this coupling explains why v1's leakage barely responded to noise.

M0 had to deliver a measurement in which no two compared runs share private randomness, and to show that the code
does what it claims:

1. The attack pipeline recovers a known privacy level on synthetic data.
2. Worlds with different noise seeds draw independent server noise, and worlds with equal seeds reproduce v1's
   coupling.

It also had to measure the cost of one run and the design's statistical power.

## 2. What was built

**Seed separation** (`experiments/reproduce/seeds.py`, `server.py`, `runner.py`).

- Four run options, each defaulting to the old single `seed`:
  - `layout-seed`: partition, train/test splits, server evaluation split, shadow samples.
  - `init-seed`: the initial model.
  - `client-seed`: client round seeds (batches, augmentation, dropout).
  - `noise-seed`: a dedicated generator for every server noise vector (`metricdp_pytorch/seeded_noise.py`), used by
    global DP, metric privacy, private median and residual clipping.
- **Committed runs replay unchanged.** Without `noise-seed`, noise is still drawn from NumPy's global state.
- `experiments/reproduce/client.py` belongs to Agent B. It changed only by two reads with fallback (`client-seed`,
  `layout-seed`), approved by the owner as an ownership exception.

**Randomized-inclusion design** (`results/cia_v2/randomized_inclusion/`).

- EuroSAT, label-Dirichlet alpha 0.3, the seed-42 layout, 48 slots, 50 rounds, FedAvg with equal weights, WP2
  hyperparameters.
- **Final design** (after the escalation, Section 5):
  - 32 candidates, fixed before any run: the top decile by size (23, 7, 40, 18, 33), then 27 clients by size quantile.
  - Each replicate includes 16 candidates; the other 16 clients always take part. Every run therefore has 32 active
    clients of 48, a public constant, so the update norm cannot reveal how many candidates are in.
  - Each candidate is IN in exactly half of the replicates, built from complementary pairs
    (`experiments/cia/inclusion.py`).
  - Every replicate has its own `client_seed` and `noise_seed`, derived from `(design_seed, replicate)`. Within a
    configuration no two runs share private randomness. Across configurations replicate `j` reuses its seeds, which
    makes differences between configurations more precise.
- **Shadow samples** are 10% of each candidate's training records, scored at rounds 0, 5, ..., 50 without the
  training augmentation. v1 read them through random crops and flips.
- **Surrogate sets** (optional, for B5): public test images resampled to each candidate's label histogram.
- **Per-round server logs:** released-update norm, clipped-signal norm and expected noise norm.

**Attack library** (`experiments/cia/audit_attacks.py`).

- **Calibrated loss attacker (primary).** A Gaussian likelihood ratio on the 11-checkpoint shadow-loss trajectory, in
  the style of LiRA with per-candidate difficulty calibration.
  - References for trial `(c, j)` are candidate `c`'s other replicates, with equal IN and OUT counts
    (`balanced_references`).
  - One covariance is pooled over candidates and both worlds, with Ledoit-Wolf shrinkage. The direction
    `w = S^-1 dbar` is also pooled, and each candidate gets a one-dimensional ratio on the projection.
- **Also:** an uncalibrated final-round loss threshold, and a noise-level attacker for mechanisms whose noise depends
  on the data.
- **Metrics:** AUC, TPR at 5% and 1% FPR, all candidates and the top decile separately.
- **Intervals:** the two-way bootstrap over candidates and replicates, and (added in Section 5) the leave-one-pair-out
  jackknife used for gates.

## 3. Gate G-M0: pass

ROADMAP.md: "G-M0 passes when the tests are green, the calibration test recovers the Gaussian trade-off and the
noise-independence test separates equal from different seeds."

| Check | Result |
|---|---|
| Test suite | 720 passed, 2 skipped (MPS-only tests that were already skipped) |
| Gaussian calibration (`test_calibration_recovers_the_gaussian_tradeoff`; `R = 1000`, `mu` 1 and 2) | AUC and TPR at 5% and 10% FPR inside the bootstrap interval |
| Noise independence (real 2-round Flower runs; realized noise energy on 64 coordinate blocks per round) | Correlation 1.0000 for equal seeds, 1.0000 for equal noise seed with different client seed (v1's coupling), -0.107 for different seeds |

Pass rule: different seeds within [-0.3, 0.3], equal seeds above 0.9. With 128 points per run the null standard
deviation of the correlation is about 0.09.

A 1-round CPU smoke run of the randomized-inclusion runner matched the design: the 8 candidates of that early design
were included, 40 active clients, noise std 0.00773 (0.001546 x 5), all four seeds in the metadata, and round-1
accuracy 0.117, as in v1's WP2.

## 4. Synthetic checks of the attacker (before any run)

Synthetic setup: 16 candidates, AR(1) noise over 11 checkpoints (rho 0.8), a shift that grows over training, and
per-candidate difficulty offsets; 100-200 simulated audits per cell (`STATE.md`, findings).

1. **Leave-one-replicate-out alone is biased.** It leaves `R/2 - 1` references in the trial's own world and `R/2` in
   the other, which gives a mean null AUC of 0.416 at `R = 8`. Equal reference counts remove the bias (null AUC
   0.49-0.50), so `balanced_references` became the rule.
2. **Pooling the direction matters at these sizes.**

   | True `mu` (optimal AUC) | `R` | Per candidate, 11-dim | Projected (primary) | Final round only |
   |---|---|---|---|---|
   | 0 (0.500) | 8 | 0.494 | 0.491 | 0.487 |
   | 1 (0.760) | 8 | 0.594 | 0.677 | 0.655 |
   | 1 (0.760) | 16 | 0.629 | 0.714 | 0.698 |
   | 2 (0.921) | 8 | 0.788 | 0.873 | 0.852 |
   | 2 (0.921) | 16 | 0.847 | 0.899 | 0.876 |

   Even the projected attacker stays below the optimal trade-off, so measured AUCs are lower bounds on what a
   better-referenced attacker achieves.
3. **The two-way bootstrap is conservative:** 96-99% coverage of the attacker's expected AUC, about 1.3 times the
   width the replicate-to-replicate spread implies.
4. **On real WP2 covariances** the primary attacker's null mean AUC is 0.496 (`R = 8`) and 0.493 (`R = 16`), over
   300 simulations.

## 5. Power, and the escalation it caused

The roadmap asked for the smallest detectable AUC difference at `R` = 8, 12 and 16, with the rule "if `R = 8` cannot
detect 0.05, screens use `R = 12`". The simulation (`power.py`, WP2's vanilla seeds 42-45 as the model, 200
simulations per point) found the specified design far from that:

| `R` | Baseline (attacker AUC) | All candidates | Top decile |
|---|---|---|---|
| 8 | vanilla-sized (0.76) | not reached | not reached |
| 12 | vanilla-sized (0.78) | 0.24 | not reached |
| 16 | vanilla-sized (0.80) | 0.21 | not reached |

Two causes (`power_diagnosis.csv`, `power_options.csv`):

- The paired two-way bootstrap was 2.4-3.7 standard deviations wide, against a nominal 1.96 (false positives 0-0.7%),
  because it resamples a fixed panel of 16 candidates as if it were random.
- Even an exact interval would need about `R = 32`: the true standard deviation of a difference is 0.071-0.085 at
  `R = 8`, 0.044-0.047 at 16 and 0.023-0.030 at 32.

`ESCALATION.md` was raised. The owner delegated the choice ("pick what comes logical to you for the questions you
asked to me"), and Agent A resolved it as follows:

1. **Design:** 32 candidates, 16 included per replicate, `R = 32` for every gated comparison. M2's grid starts at
   vanilla, 1x, 4x and 16x and adds levels only if accuracy needs them.
2. **Gate interval:** the leave-one-pair-out jackknife with `t` quantiles, conditional on the fixed candidate panel,
   at **97.5%**. At the nominal 95% it gave 6-7% false positives; at 97.5% it gives 2.2-3.0%
   (`interval_calibration.csv`, 400 simulated pairs per cell). At `R = 16` every interval tried was liberal, so no
   gate is applied there. The two-way bootstrap is reported alongside every gate interval.
3. **Two stages:** replicates 0-15 may be run first to inspect accuracy, and replicates 16-31 always follow before
   any gate statistic.
4. **G-M2 as applied:** the all-candidate AUC's upper gate end at 0.6 or below, the top-decile point estimate at 0.6
   or below (5 clients cannot narrow its interval), and the lower end of mean accuracy at 0.70 or above.
5. **Budget:** M2's cap raised from 30 to 64 GPU-h, to be re-checked against the cost run.

Resulting power at `R = 32` (`power_table.csv`):

| Baseline | Subset | Detectable difference at 80% power | False positives |
|---|---|---|---|
| vanilla-sized (AUC 0.82) | all candidates | 0.087 | 3.5% |
| half (AUC 0.66) | all candidates | 0.088 | 5.0% |
| vanilla-sized (AUC 0.81) | top decile | 0.196 | 1.0% |
| half (AUC 0.65) | top decile | not reached | 7.5% |

A true 0.05 difference is detected 67% of the time at the vanilla-sized baseline and 27% at half of it. These
simulations draw configurations independently. Later packages found that runs sharing replicate seeds make
differences much more precise (A1's power estimate from M2's runs: 0.027-0.037 for all candidates), so the table is
pessimistic for paired comparisons.

## 6. Cost run

One randomized-inclusion run: 32 active clients, global DP at the frontier noise (`r = 0.001546`), 32 candidates
scored with surrogates at 11 checkpoints, on an NVIDIA RTX 5000 Ada shared with one M1 worker and another user's idle
process (owner approval).

| Item | Value |
|---|---|
| Training | 672.6 s |
| Scoring | 43.2 s |
| Wall time | 0.20 h |
| Final accuracy | 0.810 |

The roadmap had estimated 0.5 GPU-h per run. M2's 128 base runs came to about 26 GPU-h at the measured cost, within
the original 30 GPU-h cap. M2 later measured about 0.11 GPU-h per run as GPU occupancy divided by its two concurrent
workers.

## 7. Anomalies and corrections

- **Biased references** (Section 4, finding 1) were found and fixed in synthetic checks, before any run.
- **Liberal jackknife at 95%** was found by calibration and fixed by the 97.5% gate level before any gated statistic.
- **The cost-run launcher was stopped** while the runner moved to 32 candidates, and restarted with the amended
  command.
- **GPU sharing.** Both GPUs were busy with another user's jobs at first. The smoke run moved to the CPU, and the cost
  run waited until the owner approved sharing GPU 1.

## 8. Caveats

- The power model comes from WP2's vanilla runs, whose seeds also changed the initial model; v2 shares it.
- The gate interval describes this panel of 32 clients, not clients in general. The two-way bootstrap is the
  interval that generalises.
- Only the loss and noise-level attackers exist; no attacker uses gradients or released models directly.
- One dataset and one layout define every v2 measurement.

## 9. Conclusions and owner decisions

1. **G-M0 passed** on 2026-10-17; Agent B could branch from `feature/cia-v2-measurement`.
2. **The v2 measurement standard is in place:** separate seeds, randomized inclusion with 32 candidates, `R = 32`,
   the calibrated attacker, and the 97.5% pair-jackknife gate interval with the two-way bootstrap alongside.
3. **The original design could not resolve 0.05 AUC.** The owner delegated the escalation, and the resolution above
   governs M2, A1 and A2.
4. **Closed by the owner** on 2026-10-20 together with M1, M2, A1 and A2.
