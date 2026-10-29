# Research roadmap v2: CIA defenses after the 2026-10 programme

Version 1, 2026-10-16. Status: **approved by the owner on 2026-10-17 (Section 1 defaults accepted as written); agents
may start**. Built on
[REVIEW.md](REVIEW.md), which re-examines the 2026-10 programme (`research/cia_review_2026-10/SYNTHESIS.md`) against
its committed trajectories. Audience: the owner and Claude Code agents working autonomously. Process rules from
`research/cia_review_2026-10/agent_roadmap.md` (protocol before runs, fixed gates, one extension, escalation, agents never
merge into `master`, no `Co-Authored-By`) carry over unless changed here.

## 0. Why a new roadmap

The programme answered several questions cleanly:

- Metric privacy's formal repair cannot beat global DP at a matched guarantee.
- Local-training signature suppression does not cut leakage.
- Residual clipping does not lower sensitivity.

Its leakage numbers under DP rest on a measurement in which the IN and OUT worlds draw the same noise (REVIEW.md
Section 3). Without the noise, the attacker statistics do not respond to noise, and the one-round bound puts the
corrected score at 4x noise at 0.62 or lower, against 0.79 measured. Before any defense can be ranked, the measurement
has to be rebuilt; after that, the question "what beats metric privacy" can be asked in its empirical and formal forms
(REVIEW.md Section 8).

The roadmap therefore runs in three tracks:

| Track | Question | Packages |
|---|---|---|
| M: measurement | What leakage does an attacker who sees one world actually achieve, and how does it fall with noise? | M0, M1, M2 |
| A: audit | Is metric privacy better than global DP at matched accuracy, and how strong is its side channel at strong noise? | A1, A2 |
| B: mechanisms | What lowers measured CIA below metric privacy at equal accuracy with a guarantee no weaker than global DP; is there a regime with a non-vacuous guarantee? | B1-B5, C1 |

## 1. Owner decisions (approved 2026-10-17, binding)

| # | Decision | Owner decision (the proposed defaults, accepted unchanged) |
|---|---|---|
| 1 | Measurement standard | Adopt Section 2 for every new package; cite v1 DP leakage numbers only as "paired, shared noise" |
| 2 | Compute | About 175 GPU-h over 4-5 weeks on the compute server's two shared GPUs; per-package caps in Section 3; escalation at 1.25x as before |
| 3 | Agents | Three: A (measurement and audit), B (mechanisms), C (theory, accounting, formal regime) |
| 4 | Carried-over decisions | Participation secret (public slot count, noise independent of who participates), equal weights, top-decile targets by dataset size, EuroSAT alpha 0.3 with 48 slots primary, Alzheimer MRI second |
| 5 | Definition of "beats metric privacy" | At matched accuracy (within 1 pp), top-decile AUC under the v2 attacker at least 0.05 lower with the 95% interval excluding 0, on EuroSAT and on Alzheimer, with a formal client-level guarantee no weaker than global DP at the same noise. Also reported at matched AUC |
| 6 | Paper | The owner decides after M2, A1 and A2 which of the three claim forms (REVIEW.md Section 8) the paper makes |

## 2. Measurement standard v2 (binding for all new packages)

1. **Independent private randomness between worlds.** Trajectories that an attacker must tell apart never share
   private randomness:
   - Server noise draws.
   - Client training seeds (batches, dropout).
   - Inclusion of other clients.

   Public randomness may be shared: the initial model, the partition layout and the evaluation sets. Each trajectory
   gets explicit `noise_seed` and `client_seed` values, independent of the world. Randomness may be shared *across
   configurations* (to reduce the variance of differences between mechanisms), never between worlds of one
   configuration.
2. **Fixed layout.** A separate `layout_seed` draws the partition and the shadow samples; replicates vary
   `client_seed` and `noise_seed` only. The default layout is the seed-42 EuroSAT layout used by WP2 and WP6 (top-decile
   clients 23, 7, 40, 18, 33).
3. **Randomized-inclusion design (default).** Each run has 48 slots and the public slot count as denominator. Sixteen
   candidate clients are each included with probability 1/2 in a balanced design (every candidate IN in exactly half of
   the replicates); the other 32 clients always take part. A run with `R` replicates yields `16 R` membership trials, and
   every candidate has `R/2` IN and `R/2` OUT references with independent randomness. The candidates are fixed before
   any run: the five top-decile clients plus eleven chosen by size quantile. This follows the one-run auditing idea of
   Steinke et al. 2023 and its FL use by Andrew et al. 2023 (both in `research/cia_review_2026-10/paper_table.csv`).
   Removal adjacency (IN plus OUT-`t`) remains allowed where a package needs it, with `R` independent replicates per
   world.
4. **Attackers.** All three are fitted with leave-one-replicate-out references:
   - **Calibrated loss attacker (primary):** a per-candidate Gaussian likelihood ratio on the candidate's shadow-loss
     trajectory, sampled every 5 rounds plus the last round (LiRA, Carlini et al. 2022; difficulty calibration, Watson
     et al. 2021).
   - **Uncalibrated loss threshold (weak baseline).**
   - **Noise-level attacker:** blind std estimate per round; mechanisms with a data-dependent noise level only.
5. **Metrics.**
   - AUC and TPR at 5% and 1% FPR over (candidate, replicate) trials.
   - 95% intervals from a two-way bootstrap over replicates and candidates.
   - Accuracy at the final round, with the per-client minimum reported for fairness.
   - All targets and top-decile targets separately.
6. **Paired statistics.** The v1 clean-view score may still be computed, but only as an *effect size* (how much the
   target changes its distribution's loss), labelled as such, and never as attack success.
7. **Pipeline checks.** Two must pass before any experiment (gate G-M0):
   - **Gaussian calibration test:** synthetic "trajectories" with a known GDP parameter `mu`; the analysis code must
     recover the trade-off `1 - G_mu(alpha)` within its bootstrap interval.
   - **Noise-independence regression test:** two 2-round Flower runs with different `noise_seed` give realized-noise-energy
     correlation within [-0.3, 0.3], and with equal seeds above 0.9. Use the statistic of `scripts/measurement_audit.py`.
8. **Power.** Every protocol states its minimum detectable AUC difference from the replicate count. Defaults: `R = 8`
   for screens and `R = 16` for confirmatory comparisons.

## 3. Work packages

Cost unit: GPU-hour (GPU-h) of measured wall time. v1 measured 0.27-0.42 GPU-h per 50-round, 48-client EuroSAT
trajectory. A randomized-inclusion run is estimated at about 0.5 GPU-h (16 candidates scored at 11 checkpoints); M0
re-measures it. Small federations cost about as much per round as 48 clients, because every round trains on the whole
dataset (WP1 lesson); Alzheimer is about 0.2 times EuroSAT.

| WP | Agent | Branch | Depends on | Cap (GPU-h) | Gate |
|---|---|---|---|---|---|
| M0 Measurement infrastructure | A | `feature/cia-v2-measurement` | none | no training beyond 2-round checks | G-M0 |
| M1 Shared-noise confirmation | A | `feature/cia-v2-measurement` | M0 | 6 | G-M1 |
| M2 Corrected leakage-accuracy curves (GDP, vanilla) | A | `feature/cia-v2-frontier` | G-M0, G-M1 | 30 | G-M2 |
| A1 Metric privacy vs GDP at matched accuracy | A | `feature/cia-v2-frontier` | M2 | 30 | G-A1 (report) |
| A2 Side channel at strong noise, small federations | A | `feature/cia-v2-side-channel` | G-M0 | 35 | G-A2 |
| B1 Client subsampling with amplification | B | `feature/cia-v2-mechanisms` | G-M0, C's accountant, M2 operating point | 15 | G-B1 |
| B2 Leakage-aware temporal allocation (reopened D5) | B | `feature/cia-v2-mechanisms` | M2 shows leakage falls with noise | 15 | G-B2 |
| B3 Public-validation step-size gate | B | `feature/cia-v2-mechanisms` | G-M0 | 10 | G-B3 |
| B4 Formal regime: frozen public features + small head | C | `feature/cia-v2-formal-regime` | none (calculation first) | 10 | G-B4 |
| B5 Exposure of large clients | B | `feature/cia-v2-mechanisms` | M2 runs (diagnostic part) | 15 | G-B5 |
| C1 Composite candidate vs metric privacy, EuroSAT + Alzheimer | B | `feature/cia-v2-composite` | A1 and at least one passed B gate | 45 | G-C1 |

Total cap: about 211 GPU-h, with an expected spend of about 175. Results directories: `results/cia_v2/<package>/`, each
with `PROTOCOL.md`, `STATE.md` (format of the v1 roadmap Section 3.4), `FINDINGS.md`, analysis JSON and figures. Use
`tables/` rather than `data/` for derived tables: the repository-wide `data/` ignore rule would hide them.

### M0 Measurement infrastructure (Agent A)

1. Seed plumbing: `layout_seed`, `init_seed`, `client_seed` and `noise_seed` as separate run options in
   `experiments/reproduce/server.py` and `runner.py`, defaulting to today's single `seed` so that committed runs stay
   reproducible.
   - Global DP and metric privacy: draw from a `np.random.Generator` seeded by `noise_seed` instead of NumPy's global
     state. Wrap Flower's `add_gaussian_noise_inplace`.
   - Private-median and residual-clipping strategies: `noise_seed` replaces `seed`.
   - Client round seeds: derived from `client_seed`.
2. Randomized-inclusion runner, `results/cia_v2/randomized_inclusion/runner.py`:
   - Balanced inclusion design, public slot count.
   - Scoring of every candidate's shadow sample at rounds 0, 5, ..., 50.
   - Optional label-matched surrogate sets: public test images resampled to each candidate's label histogram (B5).
   - Plans unless `--execute`.
3. Attack and analysis library in `experiments/cia/` (calibrated loss, threshold, noise-level), with the two-way
   bootstrap.
4. The two pipeline checks (Section 2, item 7) as tests; `uv run pytest` green.
5. **Cost measurement.** One full randomized-inclusion run, global DP at frontier noise, recorded in `STATE.md`. This is
   the only training in M0.
6. **Power table.** Simulate the analysis with between-replicate variance taken from WP2's fixed-layout seeds 42-45
   (no DP) and, once available, M1's runs; write the minimum detectable AUC difference for `R` = 8, 12 and 16. Every later protocol copies its
   row; if `R = 8` cannot detect 0.05, screens use `R = 12`.

**G-M0** passes when the tests are green, the calibration test recovers the Gaussian trade-off and the
noise-independence test separates equal from different seeds. On fail, fix before any other package runs.

### M1 Shared-noise confirmation (Agent A)

Repeat consensus-KL step 3's global-DP arm with independent noise per world: 1x and 4x the frontier noise, seed-42
layout, targets 23, 7, 1, 0, 3. Per noise level that is one IN trajectory and five OUT trajectories (one per target),
each with its own `noise_seed` and `client_seed`: 12 trajectories, about 4 GPU-h. Compute the paired clean-view score
exactly as in v1.

M1 deliberately keeps v1's statistic and changes only the noise coupling. It isolates whether coupling caused the flat
noise response; attacker-realistic leakage is M2's job.

**G-M1** (fixed now):

| 4x decoupled paired score | Verdict | Consequence |
|---|---|---|
| 0.70 or below | Confirmed (prediction: 0.62 or below) | v1 DP leakage numbers flagged; M2 proceeds |
| Between 0.70 and 0.75 | Partial | M2 proceeds; the 1x result is reported alongside |
| 0.75 or above | Not material | The flat noise response is real; escalate, since B4 then becomes the main line |

The coupled reference values are 0.792 (4x) and 0.788 (1x) (`results/cia_frontier/consensus_kl/analysis_step3.json`).

### M2 Corrected leakage-accuracy curves (Agent A)

Randomized-inclusion design, `R = 8`, global DP at 1, 2, 4, 8 and 16 times the frontier noise, plus vanilla. Run 1x, 4x
and 16x first and fill in the rest after inspecting accuracy. This is 48 runs, about 24 GPU-h. Report:

- The calibrated-attacker AUC and TPR at 1%/5% FPR against accuracy, for all and for top-decile targets.
- The noise `r_0.6` at which the top-decile AUC's upper interval end first drops to 0.6 or below.
- Accuracy at `r_0.6`.
- The formal `mu` at each point.

**G-M2:**

- **Useful empirical regime:** at some noise the top-decile AUC's upper end is 0.6 or below while the lower interval
  end of mean accuracy (bootstrap over replicates) is 0.70 or higher. Then A1, B1-B3 and B5 use `r_0.6` and the next-lower noise as operating points.
- **No useful regime:** leakage stays above 0.6 until accuracy falls below 0.70. The roadmap then escalates with the
  recommendation to make B4 the main line and frame the paper around the audit claims.

### A1 Metric privacy vs global DP at matched accuracy (Agent A)

The paper's metric privacy and WP3's private-median metric privacy, each at three noise multipliers chosen to bracket
the accuracies of M2's operating points: 6 configurations, `R = 8`, 48 runs, about 24 GPU-h. Leakage is compared at
matched accuracy by interpolating along each mechanism's curve, with intervals from the two-way bootstrap over runs.
The formal guarantee is stated for global DP and private-median metric privacy; the paper's mechanism has none.

**G-A1** (report, no kill), one of three outcomes:

- Metric privacy is better than global DP at matched accuracy (interval excludes 0).
- Equivalent: the interval lies within ±0.03 AUC.
- Worse, or inconclusive.

This settles the audit claim with a valid measurement.

### A2 Side channel at strong noise, small federations (Agent A)

Hypothesis: at noise strong enough that the calibrated loss attacker has AUC 0.6 or below under global DP, metric
privacy's noise-level attack still separates the worlds, because the noise std is estimable to `1/sqrt(2D)` whatever its
size.

Design:

- Cells: Alzheimer with 3, 5 and 8 clients, and EuroSAT with 3 clients.
- Fixed layout; the largest client is the target; removal adjacency with `R = 8` independent replicates per world.
- Global DP and metric privacy.
- One noise level per cell, chosen by a 4-replicate pilot as the lowest at which the global-DP loss attacker's AUC is 0.6
  or below.

The owner PC can run the Alzheimer cells (MPS fixed in v1); EuroSAT runs on the server. About 28 GPU-h.

**G-A2:** a strong claim holds if metric privacy's noise-level AUC is 0.8 or higher at the lower interval end while
global DP's loss AUC is 0.6 or below in at least two cells. Otherwise record the size of the channel at the tested
noise.

### B1 Client subsampling with amplification (Agent B; accounting by Agent C)

Poisson client sampling per round with rate `q`, with the expected count `q N` as the public denominator, so that the
noise is independent of who participates. Agent C supplies the accountant (privacy loss distributions for the
subsampled Gaussian mechanism, via an established library).

Design: `q` = 0.25 and 0.5, each at two noise levels chosen to match global DP's accuracy at M2's operating points
(within 1 pp, from a pilot), randomized inclusion with `R = 8`, 50 rounds plus a 100-round variant for `q = 0.25`.
About 12 GPU-h; per-round cost falls roughly with `q`.

**G-B1:** at matched accuracy, top-decile AUC at least 0.05 below full-participation global DP (interval excludes 0)
and an amplified formal `mu` no larger.

### B2 Leakage-aware temporal allocation (Agent B; reopened D5)

Start only if M2 shows the AUC falling with noise. Three schedules share one total `sum_t mu_t^2`: uniform,
front-loaded (more noise in rounds 1-25, where v1 saw the largest clean-view gap) and back-loaded. Global DP at M2's
operating point, `R = 8`, 24 runs, about 12 GPU-h.

**G-B2:** a schedule lowers top-decile AUC by at least 0.05 at no more than 1 pp accuracy loss and the same total `mu`.
On fail, D5 closes.

### B3 Public-validation step-size gate (Agent B)

The post-processing gate from the closed client-specific-noise line (`experiments/stacked_head/gates.py`). It uses the
server's public validation set, which is already used to train the initial model, and changes no guarantee. Global DP
with and without the gate at two noise levels; the runs without the gate are reused from M2. Sixteen new runs, about
8 GPU-h.

**G-B3:** at least 2 pp accuracy gain at equal noise, with top-decile AUC rising by no more than 0.02.

### B4 Formal regime: frozen public features and a small head (Agent C)

Calculation first. Fix a public feature extractor (for example ImageNet-pretrained, declared in the protocol) and
share only a linear head (about 5,000 parameters on EuroSAT). For the following settings, compute the noise-to-signal
ratio and the predicted accuracy loss:

- `n` = 48, and `n` = 480 by splitting the EuroSAT training data into more clients.
- `T` = 10 or 20 rounds.
- `mu_total` = 1, 2, 4 or 8.

Recompute the D3 crowd bound for these dimensions, which is the D6 question. Then run head-only training for the
promising settings; it is cheap, and the `experiments/stacked_head/` infrastructure can be reused. Always include the
public-only model as a baseline: v1's stacked-head line showed it is the real competitor.

**G-B4:** some setting with `mu_total` of 8 or less reaches at least 5 pp above the public-only model on the server test
set. If it passes, the owner may open a formal comparison there: global DP vs private-median metric privacy vs the
best B mechanism, each re-run with the same frozen-head architecture (full-network results do not transfer). If it fails, record "no non-vacuous client-level regime at this scale" as a paper statement.

### B5 Exposure of large clients (Agent B)

1. **Diagnostic, at no training cost.** In M2's runs, score each candidate with a public label-matched surrogate set as
   well as its true shadow sample.
   - If the surrogate attacker's AUC is close to the true one (within 0.05), the leakage is mostly label-distribution
     inference (Wainakh et al. 2022). The prior-neutral defenses below are then worth one screen.
   - Otherwise it is data-specific.
2. **Contribution caps.** Each client trains on at most `m` randomly drawn records per round, with `m` at the median and
   the 25th percentile of client sizes. Global DP at M2's operating point, `R = 8`, about 8 GPU-h.
3. **Conditional on item 1.** A prior-neutral local loss: logit adjustment (Menon et al. 2020) or label calibration
   (Zhang et al. 2022), screened with `R = 8`.

**G-B5:** top-decile AUC at least 0.05 lower than global DP at the same noise, at no more than 1 pp accuracy loss.

### C1 Composite candidate vs metric privacy (Agent B)

Combine the passed B mechanisms, for example subsampling plus the gate plus a cap. Compare the combination with metric
privacy, private-median metric privacy and global DP at matched accuracy, at `R = 16`, on EuroSAT (48 slots) and
Alzheimer MRI (48-client homogeneous setting, noise below the collapse points seen in v1's AUC sweep). About 40 GPU-h.

**G-C1:** the Section 1 definition of "beats metric privacy" holds on both datasets. On fail, report per dataset.

## 4. Decision flow

| After | If | Then |
|---|---|---|
| G-M0 | fail | Nothing else runs |
| G-M1 | confirmed or partial | M2; v1 DP leakage numbers labelled "paired, shared noise" in every new document |
| G-M1 | not material | Escalate: the flat response is real; B4 and the audit paper become the main line |
| G-M2 | useful regime | A1, B1, B3 and B5 at `r_0.6`; B2 opens |
| G-M2 | no useful regime | Escalate with the recommendation that B4 lead; A1 still runs (the audit claim needs it) |
| G-A1 | any outcome | Recorded; the outcome decides the audit claim |
| B gates | at least one pass | C1 |
| B gates | none pass | No composite; the paper reports the corrected frontier and the audit |
| G-B4 | pass | The owner may open a formal-regime comparison (a new package) |

## 5. Calendar (indicative)

| Week | Agent A | Agent B | Agent C |
|---|---|---|---|
| 1 | M0, then M1 | Code for B1-B3 and the B5 caps on top of M0's branch | B4 calculations; subsampling accountant for B1 |
| 2 | M2 | B3 at frontier noise; pilots for B1 | B4 head-only runs |
| 3 | A1; A2 pilots | B1, B5 | B4 findings; theory notes for the paper |
| 4 | A2 | B2 (if open), C1 design | Accounting checks for C1 |
| 5 | Findings | C1 | Synthesis draft with all agents' `FINDINGS.md` (owner request) |

## 6. Agent organisation

| Agent | Owns (edit only these) | Packages |
|---|---|---|
| A | `experiments/cia/`, seed plumbing in `experiments/reproduce/server.py` and `runner.py`, `metricdp_pytorch/metricdp_strategy.py`, `globaldp_strategy.py`, `private_metric_strategy.py`, `residual_clipping_strategy.py`, `results/cia_v2/{measurement,randomized_inclusion,frontier,side_channel}/` | M0, M1, M2, A1, A2 |
| B | `experiments/reproduce/paper_training.py`, `client.py`, `metricdp_pytorch/strategy_factory.py`, new strategy files for subsampling, schedules and caps, `results/cia_v2/{mechanisms,composite}/` | B1, B2, B3, B5, C1 |
| C | new `metricdp_pytorch/accounting.py`, `research/cia_roadmap_v2/theory/`, `experiments/stacked_head/` (reuse) or a new `experiments/frozen_head/`, `results/cia_v2/formal_regime/` | B4; accountant for B1 and C1 |

Agent B branches from `feature/cia-v2-measurement` once G-M0 passes, because its mechanisms use the new seed plumbing
and runner. The v1 coordination rules apply: `STATE.md` per package read with `git show`, an `ESCALATION.md` to stop,
each agent editing only its own `STATUS.md` rows.

New rules on top of v1:

1. **Independence (Section 2, item 1).** A protocol that compares worlds sharing private randomness is rejected.
2. **The run is the unit of replication.** Rounds and target-round cells are never pooled as independent trials.
3. **Power stated before running.** Each protocol states its minimum detectable effect.
4. **Cost estimates include evaluation.** They are re-measured after the first run; small federations are not
   assumed cheap.
5. **v1 citations.** v1 DP leakage numbers are cited only with the label "paired, shared noise".
6. **Two sources of participation randomness.** Every protocol states its participation regime separately for the
   audit and the mechanism:
   - The audit's randomized inclusion of candidates is fixed for a whole run.
   - A mechanism's per-round client sampling (B1) applies to every included client.

   Matched-accuracy comparisons use baselines run under the same audit regime.

## 7. Agent prompts

**Shared opening (paste first in every session)**

```
You are an autonomous research agent on this repository. Follow AGENTS.md. Then read, in order:
HANDOFF.md; research/cia_roadmap_v2/REVIEW.md; research/cia_roadmap_v2/ROADMAP.md (Sections 0-4 and 6, and your
packages in Section 3); research/cia_review_2026-10/SYNTHESIS.md (what v1 established).
The owner decisions in ROADMAP.md Section 1 and the measurement standard in Section 2 are binding. Work through your
packages in order, obey gates, caps, file ownership and escalation rules, and never merge into master.
Before editing anything, open your own sparse worktree for your branch (.agents/skills/worktrees/SKILL.md).
Before your first code change, write your package's STATE.md and reply with five bullets: hypothesis, design
(worlds, replicates, what randomness is shared), attacker and metrics, the gate rule copied verbatim, files you will touch.
```

**Agent A (measurement and audit)**

```
You are Agent A. Packages: M0 and M1 (feature/cia-v2-measurement), then M2 and A1 (feature/cia-v2-frontier), and A2
(feature/cia-v2-side-channel). M0 blocks every other package except Agent C's calculations: push it as soon as G-M0
passes and set its STATE.md to ready. Machine: compute server for EuroSAT runs; owner PC allowed for the Alzheimer
cells of A2.
```

**Agent B (mechanisms)**

```
You are Agent B. Packages: B3, B1, B5, then B2 if M2 opens it, then C1 if at least one B gate passes; branches
feature/cia-v2-mechanisms and feature/cia-v2-composite, both started from feature/cia-v2-measurement after G-M0 passes.
Write code and protocols while waiting for M0 and M2; launch nothing before G-M0. Machine: compute server.
```

**Agent C (theory, accounting, formal regime)**

```
You are Agent C. Package: B4 on feature/cia-v2-formal-regime, plus the subsampled-Gaussian accountant in
metricdp_pytorch/accounting.py that Agent B needs for B1 and C1 (deliver it first, with tests against published
values). Start with calculations; head-only runs are cheap and may use the owner PC. Do not edit files other agents own.
```

## 8. Risks

| Risk | Mitigation |
|---|---|
| M1 shows the shared-noise effect is not material | Pre-declared branch in Section 4 (B4 and the audit become the main line) |
| Randomized inclusion changes the federation size by up to 16 clients, so the update norm reveals the count | This is the participation-secret threat model; the norm channel is part of what a defense must handle, and it is reported separately (noise-level attacker) |
| Calibrated attacker needs many references | Balanced design gives `R/2` references per world per candidate; `R = 16` for confirmatory packages |
| Accuracy collapses before leakage falls (no useful empirical regime) | G-M2 branch; B4 explores the regime where formal guarantees are possible |
| Alzheimer at 48 clients collapsed under DP in v1's sweep | C1 uses the homogeneous partition and noise below the recorded collapse points, with accuracy checked before any attack |
| Agents' cost estimates drift (v1 WP1 was 3.4x its cap) | Cost measured in M0 before any package protocol is fixed |

## 9. References added in v2

Verified via OpenAlex on 2026-10-16:

- Aerni, Zhang and Tramèr 2024, evaluation pitfalls of privacy defenses:
  [10.48550/arxiv.2404.17399](https://doi.org/10.48550/arxiv.2404.17399)
- Watson et al. 2021, difficulty calibration in membership inference:
  [10.48550/arxiv.2111.08440](https://doi.org/10.48550/arxiv.2111.08440)
- Zanella-Béguelin et al. 2022, Bayesian estimation of DP from attacks:
  [10.48550/arxiv.2206.05199](https://doi.org/10.48550/arxiv.2206.05199)
- Ponomareva et al. 2023, practical DP regimes:
  [10.48550/arxiv.2303.00654](https://doi.org/10.48550/arxiv.2303.00654)
- Wainakh et al. 2022, user-level label leakage in FL:
  [10.48550/arxiv.2105.09369](https://doi.org/10.48550/arxiv.2105.09369)
- Menon et al. 2020, logit adjustment:
  [10.48550/arxiv.2007.07314](https://doi.org/10.48550/arxiv.2007.07314)
- Zhang et al. 2022, FL with label-distribution skew via logits calibration:
  [10.48550/arxiv.2209.00189](https://doi.org/10.48550/arxiv.2209.00189)

Already in the v1 corpus: Steinke et al. 2023 (one-run auditing), Andrew et al. 2023 (one-shot FL auditing), Carlini et
al. 2022 (LiRA), Balle et al. 2018 (subsampling amplification), McMahan et al. 2018 (client-level DP-FedAvg).

## 10. Owner decisions after Agent A's packages (2026-10-20, binding)

Taken after M0-M2, A1 and A2 were closed and merged (reports `reports/cia_v2_m0_measurement.md` to
`cia_v2_a2_side_channel.md`), resolving M2's escalation (`results/cia_v2/frontier/ESCALATION.md`). They amend Sections
1, 3 and 4 where they conflict.

| # | Decision |
|---|---|
| 7 | **Paper.** The audit form (REVIEW.md Section 8, form 1) is the main line: metric privacy's apparent advantage is a measurement artifact; at matched accuracy it leaks like global DP (A1), it has a participation side channel in small federations (A2), and its formal guarantee is absent or weaker. The empirical form is a secondary line. |
| 8 | **Alzheimer replication.** M2 and A1 are repeated on Alzheimer MRI (48 slots, alpha 0.3, randomized inclusion as in M0), Agent A, branch `feature/cia-v2-alzheimer`, cap 40 GPU-h. The gates are G-M2 and G-A1 as applied on EuroSAT. |
| 9 | **Decision 5 amended.** "Beats" is gated on the all-candidate AUC: at least 0.05 lower, with the 97.5% pair-jackknife interval of the paired difference excluding 0, at `R = 32` per configuration on M2's design. The top decile is reported with its interval but not gated (its detectable difference is about 0.09 at `R = 32`). |
| 10 | **Agent B, selectively.** B5 first, then B1, at M2's operating points 2x and 4x (M2 found no `r_0.6`), `R = 32`, gated as in decision 9. B2 is deprioritized (leakage barely moves with noise in M2) and does not run unless the owner reopens it; B3 and C1 wait. B5's item 1 diagnostic uses M2's surrogate scores. |
| 11 | **B4 confirmatory run.** The scarce-public positive (20-100 public images, `mu <= 2`) is re-run on a fresh layout under a new protocol fixed before the run (CPU only, about 1 CPU-hour), branch `feature/cia-v2-formal-confirm`. |
| 12 | **2026-10-23:** B5 closed (G-B5 passes for logit adjustment at 2x, fails at 4x; caps fail; report `reports/cia_v2_b5_large_clients.md` on `feature/cia-v2-mechanisms`, merged with B1). The B4 confirmatory run closed and merged (confirmed at 20 and 50 public images; `reports/cia_v2_formal_confirm.md`). **C1 stays waiting** although B5 passed a B gate (Section 4). |
| 13 | **2026-10-23: G-B1's formal condition is reported, not gated.** At B1's noise levels both mechanisms' `mu` are vacuous (about 40-140) and the accountant overflows for some settings; G-B1 is decided on the all-candidate AUC reduction (decision 9). |
| 14 | **2026-10-29:** B1 closed (client subsampling at `q = 0.25` passes G-B1 at both targets, about -0.10 and -0.06 AUC; `q = 0.5` fails; report `reports/cia_v2_b1_subsampling.md`); `feature/cia-v2-mechanisms` (B5, B1, B3's on-hold STATE) merged into master. **C1 opened** (`feature/cia-v2-composite`). |
| 15 | **2026-10-29, C1 design:** EuroSAT only (Alzheimer's leakage is near chance even without noise, so no mechanism can be measurably better there); G-C1's formal clause is reported, not gated (as decision 13); two adaptive attackers are added and reported (label-matched surrogates; logit-adjusted shadow losses). |
| 16 | **2026-10-29: C1's cap is 15 GPU-h** (stop and escalate at 18.75), beyond the roadmap's original total (about 178 of 175 GPU-h spent before C1). |

No other agent session is running; the owner assigned all of the above to the agent that ran Agent A's packages,
which works in each package's own branch and worktree and keeps each package's STATE, PROTOCOL and FINDINGS files.
