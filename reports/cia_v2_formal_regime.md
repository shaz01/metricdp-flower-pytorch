# CIA roadmap v2, B4: formal client-level DP regime with frozen features and a small head (and the accountant)

Package B4 of `research/cia_roadmap_v2/ROADMAP.md` (Agent C, branch `feature/cia-v2-formal-regime`, 2026-10-17). This
report also covers the subsampled-Gaussian accountant that Agent C delivered for B1 and C1. The owner closed the
package on 2026-10-17 and accepted the G-B4 fail. The scarce-public positive is recorded as exploration.

All numbers below come from committed artifacts:

- **B4:** `results/cia_v2/formal_regime/`. That is `FINDINGS.md`, `PROTOCOL.md` and `STATE.md`, the analyses
  `analysis.json`, `analysis_extended_grid.json`, `analysis_public_sweep.json`, `analysis_baseline_check.json`,
  `analysis_audit.json` and `analysis_audit_public20.json`, and the run JSONs in `results/`.
- **Calculation tables:** `research/cia_roadmap_v2/theory/tables/`.
- **Accountant:** `research/cia_roadmap_v2/theory/accounting.md`.

`FINDINGS.md` holds the full tables; this report gives the protocol, the main tables, the anomalies and the
conclusions.

## 1. Question

The v2 review (`research/cia_roadmap_v2/REVIEW.md` Section 6) found that the full network allows no non-vacuous
client-level guarantee. At 48 clients and 50 rounds, `mu_total = 1` needs a noise norm of about 396 against a signal
norm of at most 5.

B4 asks whether sharing only a linear head on frozen public features closes that gap. That head has `d = 5,130`
parameters instead of 289,194, and runs 10-20 rounds. Per release the noise-to-signal ratio is `z sqrt(d) / n` with
`z = sqrt(T) / mu_total`. At `d = 5,130` it is 0.06-6.7, against 9.9-79 for the full network (`formal_regime_nsr.csv`).

**Gate G-B4** (ROADMAP Section 3, copied in `STATE.md`): some setting with `mu_total <= 8` reaches at least 5 pp above
the public-only model on the server test set.

## 2. Protocol

The protocol is `PROTOCOL.md` @ `44fe21f9`, committed before any tuning or evaluation run. Two amendments, requested
by the owner, follow it (Section 5).

**Model.**

- Torchvision ResNet-18 (`IMAGENET1K_V1`), frozen.
- EuroSAT 64x64 images resized to 224, with 512-dimensional pooled features cached once (train fingerprint
  `30ee5d8828e2339c`, test `cdd8f364245a012b`).
- A linear softmax head of 512 x 10 weights plus 10 biases.

**Federations.**

- `n = 48`: the seed-42 Dirichlet-0.3 layout of the frontier data module.
- `n = 480`: each of those 48 clients split uniformly at random into 10 pseudo-clients (5-115 records, median 41). A
  direct 480-client Dirichlet layout was infeasible, and this was amended before any run.
- **Server split:** the 2,700-image test split halved into a public set (1,350 images) and a server test set (1,350).

**Mechanism.** Client-level DP-FedAvg on the head.

- Full participation, `E` local SGD epochs, and clipping to `C`.
- The sum of clipped updates is divided by the public slot count `n`, plus `N(0, (zC/n)^2 I)`, then a server step
  `eta_s`.
- The initial head is the public-only head.
- Exact GDP: `mu_total = sqrt(T) / z` (add/remove-one-client adjacency).
- An in-process NumPy simulator, unit-tested against the closed forms, not the Flower message path.

**Settings.**

- `n` in {48, 480}, `T` in {10, 20}, `mu_total` in {1, 2, 4, 8}, plus a no-noise reference per `(n, T)`.
- `R = 8` evaluation replicates. Each has its own `client_seed` and `noise_seed`, and replicate `j`'s seeds are
  shared across settings.

**Baseline.** Logistic regression on the 1,350 public images, with L2 chosen by public 5-fold cross-validation (inverse
L2 0.03), at 0.9044 on the server test set.

**Tuning.**

- 36 grid points x 2 seeds per setting: `E` {1, 5}, local lr {0.05, 0.2, 0.5}, `eta_s` {1, 3, 10}, clip multiplier
  {0.5, 1} x the median round-0 update norm.
- Run on a separate tuning layout (seed 43) and scored by pooled client local-test accuracy. This was amended before
  any run: the public set cannot select, because the initial head is fit on it.
- Tuning is not counted in the privacy budget, as in v1.

**Metric.** Final server-test accuracy minus public-only accuracy, as a paired difference. 95% interval: two-way
bootstrap over replicates and test images (2,000 resamples).

**Compute.**

- 1.76 CPU-hours for the original protocol and 21.3 for the amendments: record `seconds` summed, about 23.1 in all.
- No GPU.
- Compute server, 8 single-thread workers.

## 3. Calculation stage (before any DP run)

- **Headroom** (exploratory, `STATE.md` Log).
  - A non-private centralized linear probe on all 48 clients' training data reaches at most 0.9548, the best of three
    L2 values chosen on the test set.
  - So the headroom over public-only is at most 5.04 pp, whatever the mechanism.
- **Predicted accuracy loss** (`formal_regime_predicted_accuracy.csv`). An untuned first-order prediction gave at most
  a 1.6 pp gain, and a destroyed head at `n = 48`, `mu <= 4`.
- **Attack bounds implied by `mu`-GDP.**

  | `mu` | AUC bound | TPR bound at 5% FPR |
  |---|---|---|
  | 1 | 0.760 | 0.26 |
  | 2 | 0.921 | 0.64 |
  | 4 | 0.998 | 0.99 |
  | 8 | 1.000 | 1.00 |

  The gate's range `mu <= 8` therefore includes settings whose guarantee constrains no attacker. Results are reported
  per `mu`.

## 4. Main result: G-B4 fails

Server test set, public-only head 0.9044 (`analysis.json`; selections in its `selected` field):

| `n` | `T` | `mu_total` | `epsilon` at 1e-5 | DP head accuracy (sd over replicates) | Gain over public-only, pp [95% CI] |
|---|---|---|---|---|---|
| 48 | 10 | no noise | n/a | 0.9389 (0.0036) | 3.44 [2.19, 4.70] |
| 48 | 10 | 1 | 4.38 | 0.9111 (0.0023) | 0.67 [-0.10, 1.48] |
| 48 | 10 | 2 | 10.00 | 0.9204 (0.0018) | 1.59 [0.85, 2.43] |
| 48 | 10 | 4 | 24.38 | 0.9281 (0.0035) | 2.36 [1.32, 3.42] |
| 48 | 10 | 8 | 65.32 | 0.9354 (0.0045) | 3.09 [1.95, 4.20] |
| 48 | 20 | no noise | n/a | 0.9452 (0.0007) | 4.07 [2.66, 5.42] |
| 48 | 20 | 1 | 4.38 | 0.9012 (0.0067) | -0.32 [-1.37, 0.74] |
| 48 | 20 | 2 | 10.00 | 0.9213 (0.0037) | 1.69 [0.79, 2.61] |
| 48 | 20 | 4 | 24.38 | 0.9304 (0.0047) | 2.59 [1.50, 3.69] |
| 48 | 20 | 8 | 65.32 | 0.9375 (0.0040) | 3.31 [2.15, 4.41] |
| 480 | 10 | no noise | n/a | 0.9309 (0.0008) | 2.65 [1.60, 3.76] |
| 480 | 10 | 1 | 4.38 | 0.9252 (0.0017) | 2.07 [1.20, 3.04] |
| 480 | 10 | 2 | 10.00 | 0.9271 (0.0018) | 2.27 [1.29, 3.27] |
| 480 | 10 | 4 | 24.38 | 0.9269 (0.0012) | 2.24 [1.26, 3.22] |
| 480 | 10 | 8 | 65.32 | 0.9309 (0.0012) | 2.65 [1.63, 3.70] |
| 480 | 20 | no noise | n/a | 0.9341 (0.0009) | 2.96 [1.90, 4.06] |
| 480 | 20 | 1 | 4.38 | 0.9284 (0.0027) | 2.40 [1.39, 3.40] |
| 480 | 20 | 2 | 10.00 | 0.9334 (0.0030) | 2.90 [1.81, 4.06] |
| 480 | 20 | 4 | 24.38 | 0.9339 (0.0024) | 2.94 [1.83, 4.08] |
| 480 | 20 | 8 | 65.32 | 0.9394 (0.0024) | 3.49 [2.37, 4.64] |

- **Verdict: fail.** The best DP gain is 3.49 [2.37, 4.64] pp (`n = 480`, `T = 20`, `mu = 8`). That is outside the
  protocol's 4-6 pp extension band, so no extension was run.
- **The gap is the ceiling, not the noise.**
  - The headroom is at most 5.04 pp.
  - Even without noise, 10-20 FedAvg rounds on the non-IID clients reach only 2.65-4.07 pp.
  - 1,350 public images already give a strong linear probe on ImageNet features. The v1 stacked-head lesson repeats:
    the public-only model is the real competitor.
- **A non-vacuous guarantee buys a small, significant gain at 480 clients.**
  - At `mu = 1` (`epsilon` 4.38 at `delta = 1e-5`, AUC bound 0.760), the gain is 2.07 [1.20, 3.04] pp at `T = 10`
    and 2.40 [1.39, 3.40] pp at `T = 20`.
  - At 48 clients `mu = 1` gives nothing.
  - The 10x lower per-coordinate noise at 480 clients matches the noise-to-signal table: 0.47-0.67 against 4.7-6.7.
- **Per-client accuracy.** At `n = 48` the minimum local-test accuracy is 0.80 in most settings. At `n = 480` it is 0,
  because local test sets hold 1-3 records, so the 10th percentile is reported in FINDINGS instead.

**Crowd bound at head dimensions (the D6 question).** WP3's Propositions 1-3 were evaluated per round on noise-free
head trajectories (`formal_regime_crowd.csv`).

- Where `mu <= 2`, an attacker who does not know the other clients is at most 15% weaker than the GDP worst case
  against the top-decile clients: the ratio is 0.85-0.99 for the full release and 0.70-0.99 for the loss statistic.
- The crowd helps materially only at `mu = 8`, where the bound is vacuous anyway.
- So D6 does not give a materially smaller bound in this formal regime.

## 5. Owner follow-up (Amendments 1 and 2)

After reading the fail, the owner asked: "try a smaller public dataset and run the secondary attack check. before we
accept the fail we should analyze carefully". Two amendments followed. Each was committed before its runs.

- **Amendment 1** @ `3ec476ee`: the public-set sweep, the extended grid and the full-public audit. Results @
  `97c51b18`.
- **Amendment 2** @ `6c23c129`: the audit at 20 public images and a baseline-strength check. Code `d96cc23b`, results
  `21bbb20b`.

Both are exploration after the original result was known. The original verdict stays in `analysis.json`.

### 5.1 Extended-grid sensitivity

- **Grid:** 150 points per setting: local lr 0.01-1.0, `eta_s` 0.3-30, clip multiplier 0.25-1; 4,560 extra tuning runs
  plus 160 evaluation runs.
- **Verdict:** fail. The best gain rises from 3.49 to 3.81 [2.68, 5.05] pp (`n = 480`, `T = 20`, `mu = 8`).
- **`mu <= 2`:** the best is 2.90 [1.81, 4.06] pp, unchanged.
- **Gains moved** by -0.5 to +1.3 pp, the largest at `n = 48`, `T = 20`, `mu = 1` (-0.32 to 0.99).
- **Reading:** no-noise references cap the result (at most 3.90 pp here), so the fail did not depend on the grid.

### 5.2 Public-set sweep and baseline strength

**Design.**

- Public sets of 20, 50, 100, 200 and 500 images (2-50 per class), 5 evaluation subsets each, `R = 4` replicates.
- Per-size tuning on the tuning layout.
- 7,200 tuning runs plus 2,000 evaluation runs.
- The gain is the mean over subsets of the DP accuracy minus that subset's public-only accuracy. The bootstrap
  resamples subsets, replicates and test images.

**Baselines.** A small public set could make the public-only baseline weak, so Amendment 2 refit it four legitimate
ways:

- the protocol head;
- a wide L2 grid;
- L2-normalized raw features;
- nearest class mean.

`best_legit` takes, per subset, whichever of these is most accurate on the test set, which favours the baseline.
`oracle_c` tunes L2 on the test set and is a ceiling. The protocol head was recomputed exactly for all 2,000 records.

| public images | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|
| public-only, protocol | 0.660 | 0.782 | 0.832 | 0.864 | 0.890 |
| public-only, best_legit | 0.688 | 0.800 | 0.843 | 0.871 | 0.891 |
| public-only, oracle_c | 0.676 | 0.785 | 0.838 | 0.868 | 0.893 |
| verdict (protocol, best_legit and oracle_c alike) | pass | pass | pass | pass | fail |

**Gains at `mu <= 2`,** in pp:

- Each cell is "gain [lower 95% end]", against the protocol baseline / against best_legit.
- **Bold** means the lower end is at least 5.0.

| `n` | `T` | `mu` | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| 48 | 10 | 1 | **14.9 [10.3]** / **12.2 [6.9]** | 6.7 [4.7] / 5.0 [2.5] | 2.8 [1.5] / 1.6 [0.1] | 2.2 [1.0] / 1.5 [0.4] | -0.1 [-1.1] / -0.3 [-1.2] |
| 48 | 10 | 2 | **22.7 [17.8]** / **20.0 [14.8]** | **11.6 [9.8]** / **9.8 [7.6]** | **6.8 [5.3]** / 5.7 [4.3] | 4.2 [2.9] / 3.5 [2.3] | 2.0 [1.0] / 1.9 [0.9] |
| 48 | 20 | 1 | **12.1 [7.5]** / 9.4 [4.1] | 4.2 [2.2] / 2.5 [-0.1] | 0.9 [-0.7] / -0.2 [-1.8] | 0.4 [-1.1] / -0.3 [-1.6] | -2.6 [-3.9] / -2.8 [-4.0] |
| 48 | 20 | 2 | **21.8 [17.4]** / **19.1 [14.0]** | **11.3 [9.4]** / **9.5 [7.1]** | **6.7 [5.1]** / 5.6 [4.1] | 4.3 [2.9] / 3.7 [2.3] | 2.0 [1.0] / 1.8 [0.8] |
| 480 | 10 | 1 | **24.9 [20.2]** / **22.2 [16.9]** | **13.3 [11.3]** / **11.5 [9.2]** | **8.5 [6.8]** / **7.3 [5.8]** | 5.6 [4.1] / 4.9 [3.7] | 3.1 [2.1] / 3.0 [2.0] |
| 480 | 10 | 2 | **26.2 [21.7]** / **23.5 [18.5]** | **14.3 [12.4]** / **12.6 [10.3]** | **9.2 [7.5]** / **8.0 [6.5]** | 6.0 [4.4] / 5.3 [3.9] | 3.8 [2.8] / 3.7 [2.6] |
| 480 | 20 | 1 | **25.6 [21.0]** / **22.9 [17.5]** | **13.7 [11.8]** / **11.9 [9.6]** | **8.8 [7.1]** / **7.7 [6.1]** | 5.9 [4.3] / 5.2 [3.8] | 3.4 [2.4] / 3.3 [2.2] |
| 480 | 20 | 2 | **26.4 [21.5]** / **23.7 [18.0]** | **14.6 [12.8]** / **12.9 [10.7]** | **9.8 [7.9]** / **8.6 [7.1]** | 6.7 [5.0] / 6.0 [4.5] | 4.1 [3.0] / 4.0 [2.8] |

**Reading.**

- **The DP model's accuracy barely depends on the public set; the baseline's does.**
  - At `n = 480` and `mu <= 2`, the DP head reaches 0.909-0.924 starting from 20 public images, against 0.925-0.933
    from the full set.
  - Private federated data replaces about 1,300 public images, but adds little once they exist.
- **Solid at 20-100 images.** At `n = 480`, `mu <= 2`, every gain is at least 7.3 pp against best_legit, and robust.
- **Marginal at 200 images.** At `n = 480` the gains are 4.9-6.0 pp and none is robust; the per-size pass rests on
  `mu = 4`-`8`. At `n = 48` they are 1.5-3.7 pp.
- **Fails at 500 images.** The best gain is 4.8 pp.
- **Cost of privacy at 20 images,** against no noise at the same size:
  - 0.3-1.6 pp at `n = 480`;
  - 4.5-15.6 pp at `n = 48`.

### 5.3 Secondary empirical audit

**Design:** Agent A's M0 resolved design, with its attackers imported unchanged (merged from
`feature/cia-v2-measurement` @ `b8122d9d`).

- Randomized inclusion of 32 candidates over `R = 32` balanced replicates, 16 included per replicate, with `n - 16`
  active clients. At 48 clients that is 1.5x B4's noise.
- Independent seeds per replicate.
- A calibrated loss attacker (primary) and an uncalibrated threshold, with the 97.5% pair jackknife.
- 640 runs at the full public set, and 384 at 20 public images (`mu <= 2` and no noise, from Part A's selections).

**Calibrated attacker, all candidates** (`analysis_audit.json`; `analysis_audit_public20.json` for the "20 public"
column):

| `n` | `T` | `mu` | AUC bound | AUC, full public [97.5% jackknife] | AUC, 20 public | TPR at 5% FPR, full (bound) | implied `mu`, full |
|---|---|---|---|---|---|---|---|
| 48 | 10 | no noise | n/a | 0.935 [0.916, 0.954] | 0.921 | 0.717 (n/a) | 2.14 |
| 48 | 10 | 1 | 0.760 | 0.614 [0.578, 0.651] | 0.599 | 0.078 (0.26) | 0.41 |
| 48 | 10 | 2 | 0.921 | 0.676 [0.640, 0.713] | 0.638 | 0.158 (0.64) | 0.65 |
| 48 | 10 | 8 | 1.000 | 0.861 [0.835, 0.888] | n/a | 0.359 (1.00) | 1.54 |
| 48 | 20 | no noise | n/a | 0.977 [0.969, 0.984] | 0.947 | 0.828 (n/a) | 2.81 |
| 48 | 20 | 1 | 0.760 | 0.595 [0.550, 0.639] | 0.590 | 0.098 (0.26) | 0.34 |
| 48 | 20 | 2 | 0.921 | 0.664 [0.622, 0.706] | 0.659 | 0.127 (0.64) | 0.60 |
| 480 | 10 | no noise | n/a | 0.842 [0.810, 0.874] | 0.860 | 0.393 (n/a) | 1.42 |
| 480 | 10 | 1 | 0.760 | 0.524 [0.498, 0.551] | 0.545 | 0.051 (0.26) | 0.09 |
| 480 | 10 | 2 | 0.921 | 0.576 [0.538, 0.615] | 0.547 | 0.086 (0.64) | 0.27 |
| 480 | 20 | no noise | n/a | 0.881 [0.851, 0.912] | 0.897 | 0.547 (n/a) | 1.67 |
| 480 | 20 | 1 | 0.760 | 0.539 [0.502, 0.575] | 0.536 | 0.066 (0.26) | 0.14 |
| 480 | 20 | 2 | 0.921 | 0.577 [0.537, 0.617] | 0.594 | 0.096 (0.64) | 0.28 |
| 480 | 20 | 8 | 1.000 | 0.703 [0.670, 0.735] | n/a | 0.207 (1.00) | 0.75 |

Every setting (`mu` = 4 and 8 included), the TPR at 1% FPR, the top-5 subset and the threshold attacker are in
FINDINGS.

**Reading.**

- **Validity.** In all 24 audited settings with noise, no calibrated-AUC jackknife lower end exceeds its formal bound,
  and no TPR point estimate exceeds its bound.
- **Leakage without DP.** With no noise the attacker reaches AUC 0.84-0.98, so the low numbers under noise come from
  the mechanism, not a weak attacker.
- **Gap.** At `mu = 1` the attacker reaches AUC 0.52-0.61 against the bound 0.760, an implied `mu` of 0.09-0.41.
  Against this attacker the bound is loose by a factor of 2.4-12 in `mu`.
- **Weak initial head.** Starting from 20 public images does not increase the measured leakage. The differences from
  the full-public audit run from -0.038 to +0.021. In the two settings outside the other audit's jackknife interval,
  the 20-image AUC is lower.

## 6. Anomalies and corrections

- **Grid edges.** Every selected base-grid configuration lay on a grid edge. The extended grid (Section 5.1) showed
  that this did not cause the fail.
- **Tuning metric.** Selecting on the public set was impossible, because the initial head scores 0.98 there. The metric
  was amended before any run to the tuning layout's pooled local-test accuracy.
- **480-client layout.** A direct Dirichlet layout was infeasible (the partitioner's 10-record minimum failed 1,000
  attempts), so the 480 clients are a pseudo-client split. This was amended before any run.
- **Exploratory log correction.** On the protocol's full CV grid, the public-only head selects inverse L2 0.03, not
  the 0.1 of an exploratory note. Its accuracy is unchanged (0.9044).
- **Shared-machine incident.** An early sklearn job without thread limits took about 30 cores under load 76. It was
  killed by PID, and every later CPU job sets the OMP/OPENBLAS/MKL thread limits.
- **Small-public cross-validation.** At 20 images the CV uses 2 folds. Two of five subsets chose the smallest grid L2
  (0.001), and at 200 images one chose the largest (10). This is the reason for the baseline-strength check, which
  did not change any verdict.

## 7. Caveats

- **`n = 480` is a pseudo-client split.**
  - Its guarantee is per pseudo-client. By group privacy, an original client of this layout gets only `10 mu`, which
    is vacuous.
  - The `n = 480` rows describe a federation of 480 genuine small owners.
  - With the real 48 clients, the `mu <= 2` passes are:
    - 20 images: all four settings, with `T = 20`, `mu = 1` not robust against best_legit.
    - 50 images: `mu = 2`.
    - 100 images: `mu = 2`, robust only against the protocol baseline.
- **Tuning on private data of a separate layout** (the same image pool) is not counted in the privacy budget.
- **Fixed choices:** one extractor (ResNet-18 ImageNet), one dataset (EuroSAT) and one partition (Dirichlet 0.3).
- **Simulator:** in process, not the Flower message path.
- **Few public subsets:** 5 per size, and the subset term dominates the intervals.
- **Audit scope:** one panel and layout, and two attackers. An empirical audit bounds nothing from above; the formal
  bound stays the privacy statement.
- **Crowd-bound limits:** a Gaussian crowd model with the population known, noise-free trajectories, independent
  rounds, and the calculation config.

## 8. The accountant (delivered first, for Agent B's B1 and C1)

**Files:** `metricdp_pytorch/accounting.py`, tests `tests/test_accounting.py` (60 tests), definitions and checks in
`research/cia_roadmap_v2/theory/accounting.md`.

**What it computes:** client-level DP-FedAvg with Poisson client sampling (rate `q`, public denominator).

- Exact GDP `sqrt(T) / z` at `q = 1`.
- PLD `(epsilon, delta)` via Google's `dp-accounting` 0.6.0: pessimistic discretization 1e-4, the maximum of the add
  and remove directions.
- RDP with the improved and the classical (Mironov 2017, Proposition 3) conversions.
- `dominating_mu`: the smallest `mu` whose GDP curve dominates the mechanism's, above an explicit delta floor
  (default 1e-10).
- Inverse searches for the noise multiplier.

**Published values reproduced** (`tables/accounting_published.csv`, `epsilon` at the published `delta`):

| Source | `q` | `z` | steps | published | RDP classical | RDP improved | PLD |
|---|---|---|---|---|---|---|---|
| Abadi et al. 2016 Sec. 5.1, E=100 | 0.01 | 4.0 | 10,000 | 1.26 | 1.2586 | 1.0355 | 0.947 |
| Abadi et al. 2016 Sec. 5.1, E=400 | 0.01 | 4.0 | 40,000 | 2.55 | 2.5737 | 2.2097 | 2.0334 |
| TF Privacy MNIST, noise 1.1, 60 epochs | 0.004267 | 1.1 | 14,063 | 3.01 | 3.0084 | 2.5967 | 2.3818 |
| TF Privacy MNIST, noise 1.3, 15 epochs | 0.004267 | 1.3 | 3,516 | 1.19 | 1.1923 | 0.9546 | 0.8646 |
| TF Privacy MNIST, noise 0.7, 45 epochs | 0.004267 | 0.7 | 10,547 | 7.10 | 7.1016 | 6.3197 | 5.6397 |

- The classical conversion reproduces the TF Privacy values and Abadi's 1.26 to two decimals, and Abadi's 2.55 to
  within 0.024.
- The tighter methods are ordered PLD <= improved RDP <= classical RDP, as the tests require.

**Findings for users of the accountant.**

- **The amplified `mu` depends on the delta floor.** The subsampled Gaussian is not exactly GDP, so "`mu`-GDP for all
  `epsilon`" depends on the floor. The value changes by 4-15% between floors of 1e-4 and 1e-12
  (`tables/accounting_floor.csv`).
- **`mu_CLT` is not a bound.** The CLT value (Bu et al. 2020, Eq. 5) is no bound in either direction.
- **Calibration** (`accounting.md` Section 4, `tables/accounting_calibration.csv`). At matched `mu`, sampling with
  denominator `qN` puts 1.04-3.06x more noise on each released average than full participation.
- **Threat model.** Amplification holds only against observers of the released models while each round's sample
  stays secret. It does not hold against the sampling server or a participant-count channel.

**Fix during the package.** The noise-multiplier search once jumped to `z = 0.01`, where a PLD has a huge support, and
crashed a calibration pool. The bracket now doubles or halves from `z = 1` and refuses `z < 0.05`, with a regression
test (`1567624e`).

**Dependency.** `dp-accounting==0.6.0` became a project dependency on 2026-10-17 (owner decision).

- `uv sync` added `absl-py`, `dm-tree` and `wrapt`, and moved `attrs` from 26.1.0 to 23.2.0.
- Full suite on the branch: 804 passed, 4 skipped, 5 deselected.

## 9. Conclusions and owner decisions (2026-10-17)

1. **G-B4 fails for this setup (accepted).** There is no non-vacuous client-level regime at this scale with a
   1,350-image public set. The nuance:
   - about 2 pp at `mu = 1` with 480 (pseudo-)clients;
   - a realistic attacker at an implied `mu` of 0.09-0.41.

   The obstacle is the strong public-only probe, not the noise.
2. **Conditional positive, recorded as exploration.**
   - With at most 10-20 public images per class, head-only DP-FedAvg at `mu <= 2` beats the public-only head by 5-26
     pp.
   - It is robust against the strongest legitimate baseline at 20-100 images for `n = 480`, and at 20 images for
     `n = 48` (except `T = 20`, `mu = 1`).
   - It is not a pre-registered result. A confirmatory run on a fresh layout (about 1 CPU-hour, new protocol) is parked.
3. **Client subsampling at `n = 480`** with the accountant's amplified `mu` is recorded as an open direction. It is
   not pursued in B4.
