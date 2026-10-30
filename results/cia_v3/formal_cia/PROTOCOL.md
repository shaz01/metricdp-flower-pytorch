# P5 protocol: client inference in the formal regime

Roadmap v3, package P5 (Agent F), branch `feature/cia-v3-formal-cia`. Written 2026-10-30 before any code change to the
mechanisms and before any run. Its commit hash is recorded in `STATE.md`. The design is the one in `STATE.md`
(@ `8d988c97`); this file adds the exact definitions, the run order and the analysis. Code:
`experiments/frozen_head/formal_cia.py`, mechanism options in `experiments/frozen_head/fedavg.py`.

## 1. Question and gate

On B4's frozen-feature linear head, in the scarce-public regime where a non-vacuous client-level guarantee is useful,
how much does the calibrated client-inference attacker get against global DP, the paper's metric-privacy
calibration and Poisson-subsampled DP, and how do the three compare at matched accuracy?

**G-P5 (copied verbatim from ROADMAP.md Section 3):**

> **G-P5 (report).**
> - Global DP at `mu = 2` with an AUC upper bound of at most 0.60: "protective regime confirmed".
> - The matched-accuracy differences between the three mechanisms are reported with the pair jackknife.

The reading (STATE.md, Gate) is fixed:

1. "AUC upper bound" = the upper end of the 97.5% pair-jackknife interval of the calibrated attacker's all-candidate
   AUC, global DP at `mu_total = 2`. Per federation; the headline is `n = 48`. At most 0.60: "protective regime
   confirmed"; otherwise "not confirmed". Always testable.
2. Matched-accuracy differences carry decision 9's label ("beats": estimate at most -0.05 and the 97.5% interval
   below 0) and A1's label (`results/cia_v2/frontier/a1/analyze.py` `outcome`).
3. Untestable targets: outside either curve's accuracy range: "not testable", not a pass. Inside both, but some
   leave-one-pair-out curve does not reach it: "not testable by the gate interval", two-way bootstrap reported. No
   testable target in a comparison: the comparison is "not testable".

## 2. Fixed setting (all public; shared by every run)

- Features: B4's (ResNet-18 `IMAGENET1K_V1`, 512-d pooled, cache fingerprints train `30ee5d8828e2339c`, test
  `cdd8f364245a012b`; a mismatch stops everything).
- Layout 42 (`runner.EVALUATION_LAYOUT`), label-Dirichlet 0.3, `n` in {48, 480} (`runner.SPLIT_FACTOR`).
- Public set: B4 Part A's evaluation subset 0 of 20 images (`runner.federation(n, 42, 20, 0)`); standardizer,
  public-only initial head, shadow-loss scaling from it (`audit._scale(n, 20)`).
- `T = 10`, batch 32.
- Hyperparameters: `audit._chosen(20)[(n, 10, mu)]` (B4 Part A's tuning at 20 public images). Level `mu = 0.5` takes
  `mu = 1`'s; vanilla takes the no-noise selection; metric privacy and subsampling at nominal level `mu` take global
  DP's selection at `mu`. Clip `C = clip_multiplier x runner.base_clip(n, E, lr, 20)`.

## 3. Design

- Candidates: `audit.panel(n)` (32 candidates; at `n = 48` M0's fixed panel, checked at run time; top 5 = 23, 7, 40,
  18, 33 at `n = 48`).
- Membership: `balanced_design(32, 32, seed=P5_DESIGN_SEED)`, `P5_DESIGN_SEED = 2026103001`. Column `j` is replicate
  `j`; replicates `2p` and `2p + 1` are complementary (checked).
- Active clients: every non-candidate plus the included candidates, `n - 16`, the public denominator `N`.
- Seeds: `(client_seed, noise_seed, sampling_seed) = SeedSequence([P5_DESIGN_SEED, j]).generate_state(3)`. Batch order
  `default_rng([client_seed, client, round])`; server noise from `default_rng(noise_seed)`; Poisson sample from
  `default_rng(sampling_seed)`. Independent between replicates; shared across configurations (pairing).

## 4. Mechanisms

All three clip each client update `u_i = theta_i - theta` to norm `C` and release
`theta' = theta + eta_s (S / D + xi)`.

| Key | `S` | `D` | `xi` | Guarantee |
|---|---|---|---|---|
| `vanilla` | all active clipped updates | `N` | 0 | none |
| `gdp-mu{m}` | all active clipped updates | `N` | `N(0, (z C / N)^2 I)`, `z = sqrt(T) / m` | exact `m`-GDP |
| `mp-mu{m}` | all active clipped updates | `N` | `N(0, ((z / d_t) C / N)^2 I)`, same `z` as `gdp-mu{m}` | none |
| `ss-mu{m}` | clipped updates of a Poisson(`q = 0.25`) sample of the active clients | `q N` | `N(0, (z_q C / (q N))^2 I)`, `z_q = accounting.noise_multiplier_for_mu(m, sampling_rate=0.25, rounds=10)` | `m`-GDP up to `delta_floor` 1e-10 (`dominating_mu`) |

`m` in {0.5, 1, 2, 4}. Metric privacy's `d_t` is `metricdp_strategy.py`'s: the maximum over pairs of active clients of
the mean of the per-layer Euclidean distances between their local models `theta + u_i` (unclipped), the layers being
`W` (10 x 512) and `b` (10). A non-finite or non-positive `d_t` falls back to the previous valid value, or 1.0 if none
(as the strategy). An empty subsampling round releases `theta + eta_s xi`.

Formal values recorded per configuration: `mu` (`gdp_mu` or `dominating_mu`), `epsilon` at `delta = 1e-5`
(`accounting.report`), the AUC bound `Phi(mu / sqrt 2)`.

26 configurations (13 per federation), `R = 32`, 832 runs. Declared extension (STATE.md): once, on accuracy alone,
before any leakage statistic.

## 5. What each run records

`results/cia_v3/formal_cia/results/runs/<config>_n<n>_r<jj>.json`: the job; seeds; membership column; active count;
config (clip, `z`, `eta_s`, ...); shadow losses of the 32 candidates after rounds `0..T` (round 0 identical in all
runs); final server-test accuracy and per round; per-client local-test accuracy (minimum, 10th percentile, median) of
the final head; for `mp` the per-round `d_t` and noise std; for `ss` the per-round sampled active clients (canonical
ids) and their count; feature fingerprints; seconds.

## 6. Order of work

1. Mechanism options in `fedavg.run` (`calibration="metric"`, `sampling_rate`, `sampling_seed`) with defaults that
   leave the existing code path bit-identical; synthetic unit tests; `uv run pytest` green.
2. **Reproduction check (a):** re-run B4 Part D's `n48_T10_mu2_pub20_r00` through `audit.execute`'s path with the
   modified `fedavg` and compare with the committed JSON: final accuracy equal, shadow losses equal to 1e-12.
3. **Reproduction check (b):** P5's analysis code on the committed `audit_public20` records reproduces B4's
   `n = 48`, `T = 10`, `mu = 2` calibrated AUC 0.638 [0.587, 0.689] to 3 decimals.
4. Runs (`formal_cia run --execute`), at most 6 single-thread workers on the owner PC.
5. Accuracy table; apply the extension rule if needed (and record it in STATE.md before step 6).
6. Analysis (`formal_cia analyze`) to `analysis.json`; figures and tables; FINDINGS.md.

## 7. Analysis

- **Per configuration:** calibrated attacker on losses at rounds `1..T` (`calibrated_scores`), AUC, TPR at 5% and 1%
  FPR; all candidates and top 5; 97.5% pair jackknife and two-way bootstrap (2,000 resamples, seed
  `P5_DESIGN_SEED`); threshold attacker; implied `mu = sqrt 2 Phi^-1(AUC)`; validity check (jackknife lower end above
  the formal AUC bound = pipeline error, escalate).
- **Sampling-aware attacker** (`ss` only): IN trials use the realized schedule `s_t` of the candidate; OUT trials an
  i.i.d. Bernoulli(`q`) schedule from `default_rng([20261030, candidate, j])` (`reanalyze.py`'s rule and seed). With
  one round per checkpoint window, the statistic is `reanalyze.schedule_statistic` with window 1:
  `z = sum_{t=1..T} (s_t - q)(d_t - dbar_t)`, `d_t = x_{t-1} - x_t` the shadow-loss drop, `dbar_t` its mean over the
  candidate's replicates. `z / std(z)` (one constant per configuration) is appended to the 10 loss features of the
  calibrated attacker. Realized IN sampling fraction checked within 0.02 of `q`.
- **Matched accuracy:** A1's statistic. Each mechanism's points are (mean final accuracy, AUC) over its four levels
  plus vanilla (the noise-free end of every curve), linear in accuracy, no extrapolation (`a1.analyze.curve_value`).
  Every resample draws the same pairs, candidates and replicates in all configurations. Targets: `n = 48`: 0.843 and
  0.714; `n = 480`: 0.921 and 0.910 (B4 Part D's global-DP mean accuracy at `mu` 2 and 1). Comparisons, all
  candidates (labelled) and top 5 (reported, not labelled):
  1. `mp - gdp`;
  2. `ss - gdp`, standard and sampling-aware;
  3. `ss - mp`, standard and sampling-aware.
- **Matched guarantee (descriptive):** at each `m`, `ss-mu{m}` against `gdp-mu{m}` (paired AUC difference with the pair
  jackknife, and the accuracy difference).
- **Fairness alongside:** the per-replicate minimum client local-test accuracy, averaged, interpolated at the targets
  like the AUC (point estimate). At `n = 480` many local test sets hold 1-3 records, so the 10th percentile is also
  given.

## 8. Escalation, cost

- `ESCALATION.md` and stop if: the reproduction checks fail; a validity check fails; a fingerprint differs.
- Cost: about 832 runs of 2-15 s (B4 Part D's records), 2-3 CPU-hours; owner PC; no GPU.
