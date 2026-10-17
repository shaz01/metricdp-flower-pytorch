# B4 protocol: formal regime with frozen public features and a small shared head

Roadmap v2, package B4 (Agent C), branch `feature/cia-v2-formal-regime`. Written 2026-10-17 after the calculation
stage and before any tuning or evaluation run; committed before the first run, and its commit hash is recorded in
`STATE.md`. The design is the one fixed in `STATE.md`, with the amendments listed in its Log (n = 480 layout, tuning
metric). Code: `experiments/frozen_head/`. Calculations: `research/cia_roadmap_v2/theory/formal_regime_calcs.py`
(tables in `research/cia_roadmap_v2/theory/tables/formal_regime_*.csv`). Accountant: `metricdp_pytorch/accounting.py`.

## 1. Question and gate

Is there a setting in which client-level DP-FedAvg with a non-vacuous formal guarantee beats the public-only model by
a useful margin, once only a small head on frozen public features is shared?

**G-B4 (copied verbatim from ROADMAP.md Section 3):** some setting with `mu_total` of 8 or less reaches at least 5 pp
above the public-only model on the server test set. If it passes, the owner may open a formal comparison there:
global DP vs private-median metric privacy vs the best B mechanism, each re-run with the same frozen-head architecture
(full-network results do not transfer). If it fails, record "no non-vacuous client-level regime at this scale" as a
paper statement.

**How the gate is read** (fixed here; this does not change the threshold):

- "Reaches" means that, for some setting with `mu_total <= 8`, the replicate mean of (DP head accuracy minus
  public-only accuracy) on the server test set is at least 5.0 pp.
- The 95% interval is the two-way bootstrap in Section 6. A setting that reaches 5.0 pp while its interval's lower end
  is below 5.0 pp is reported as "pass, not robust".
- Every setting is reported. The verdict is also given separately for each `mu_total`, because the attack bounds
  implied by `mu`-GDP differ widely across the gate's range (`tables/formal_regime_nsr.csv`): AUC at most 0.760,
  0.921, 0.998 and 1.000 at `mu` = 1, 2, 4 and 8, and TPR at 5% FPR at most 0.26, 0.64, 0.99 and 1.00.
- The no-noise reference settings never count toward the gate.

## 2. Fixed architecture and data (all public; shared by every run)

- **Extractor:** torchvision ResNet-18 with `IMAGENET1K_V1` weights, frozen and in eval mode. Inputs are 64x64
  EuroSAT images upsampled to 224 (bilinear) with ImageNet normalization and no augmentation. The feature is the
  512-dimensional pooled output, cached once (fingerprints: train `30ee5d8828e2339c`, test `cdd8f364245a012b`).
- **Head:** linear, 512 x 10 plus 10 biases, `d = 5,130`.
- **Standardization:** features are standardized with the mean and standard deviation of the layout's public set.
- **Layouts** (`experiments/frozen_head/data.py`, identical to the frontier data module):
  - `n = 48`: Dirichlet alpha 0.3, 80/20 local train/test split per client.
  - `n = 480`: each of the 48 clients split uniformly at random into 10 pseudo-clients.
  - Evaluation layout: `layout_seed = 42`, top decile 23, 7, 40, 18, 33 at `n = 48`.
  - Tuning layout: `layout_seed = 43`.
- **Server split:** the EuroSAT test split (2,700 images) is halved by `layout_seed`. The public set has 1,350 images
  and the server test set has 1,350.

## 3. Public-only baseline

Multinomial logistic regression on the standardized public features (scikit-learn, L-BFGS). The inverse L2 strength
is chosen from {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10} by stratified 5-fold cross-validation on the public set
only (ties go to the stronger regularization), and the model is then refit on the whole public set. Its parameters
are the gate's baseline and also the initial head of every federated run, which is public randomness.

## 4. Mechanism

Client-level DP-FedAvg on the head (`experiments/frozen_head/fedavg.py`):

- Full participation.
- Each client runs `E` local epochs of minibatch SGD (batch 32) on mean cross-entropy, starting from the broadcast
  head.
- Each update is clipped to norm `C`.
- The server forms the sum divided by the public slot count `n`, adds noise `N(0, (z C / n)^2 I)` from a dedicated
  generator seeded by `noise_seed`, and takes a server step `eta_s`.
- Batch order comes from `default_rng([client_seed, client, round])`.
- The guarantee is exact `mu_total = sqrt(T) / z` (add/remove one client), so `z = sqrt(T) / mu_total`.

**Settings:**

- `n` in {48, 480}, `T` in {10, 20} and `mu_total` in {1, 2, 4, 8}: 16 DP settings.
- One no-noise reference per `(n, T)`: 4 references.

## 5. Tuning (before evaluation; not counted in the privacy budget)

**Grid,** for every DP and reference setting (36 grid points):

- `E` in {1, 5}.
- Local learning rate in {0.05, 0.2, 0.5}.
- `eta_s` in {1, 3, 10}.
- `C` = multiplier x the median round-0 update norm, with the multiplier in {0.5, 1}. The norm is measured once per
  `(n, E, lr)` on the tuning layout from the public-only head, with `client_seed = 0`. The absolute `C` is then a
  fixed public hyperparameter, reused unchanged on the evaluation layout.

**Tuning runs:**

- Layout `layout_seed = 43`.
- Two seeds per grid point (`client_seed` 1000 + j and `noise_seed` 2000 + j, j = 0, 1).
- 20 settings x 36 grid points x 2 seeds = 1,440 runs.

**Selection metric:** final-round accuracy on the pooled local test splits of the tuning layout's clients (records of
the EuroSAT training split).

- The public set cannot serve, because the initial head was fit on it (0.98 accuracy there).
- The test split cannot serve, because layout 43's halves overlap layout 42's server test set.

**Selection rule:** the grid point with the highest mean over the two seeds. Ties go to the smaller `eta_s`, then
fewer epochs, then the smaller learning rate, then the smaller multiplier. The grid is not extended, and a selection
on its edge is reported as such.

The tuning runs use private records of a separate layout. Their privacy cost is not accounted, as in v1 and most of
the DP literature; FINDINGS states this.

## 6. Evaluation

**Runs:**

- `layout_seed = 42` with the selected grid point.
- `R = 8` replicates per setting, with `client_seed` 10000 + j and `noise_seed` 20000 + j, j = 0..7.
- Each replicate has its own independent private randomness, and replicate j shares its seeds across settings (Section
  2 item 1 of the roadmap allows sharing across configurations).
- 20 x 8 = 160 runs.

**Primary metric:** final-round server-test accuracy, and its paired difference from the public-only head on the same
1,350 images.

**Interval:** two-way bootstrap with 2,000 resamples (seed 20261017) that resamples replicates and test images
independently. Each resample computes `mean_r acc_r - acc_public` on the resampled images.

**Also reported:**

- Accuracy at every round.
- Pooled local-test accuracy.
- The per-client local-test minimum and 10th percentile (at `n = 480` many local test sets hold 1-3 records, so the
  minimum is coarse).
- Clipped fraction and median update norm.
- `mu_total` and `epsilon` at `delta = 1e-5` from the accountant.

## 7. Power

The calculation stage gives the spread of accuracy over noise draws (`formal_regime_predicted_accuracy.csv`). It is
at most 0.6 pp in every setting whose predicted accuracy is within 5 pp of the ceiling, and up to 6.6 pp where noise
destroys the head. Two sources of uncertainty combine for the paired difference:

- Replicate-to-replicate spread: about 0.6/sqrt(8), or 0.2 pp.
- Test-image sampling: about sqrt(0.06 / 1,350), or 0.7 pp, assuming the two heads disagree on about 6% of images.

The 95% interval half-width is therefore about 1.4 pp near the gate, and the minimum detectable gain (80% power,
two-sided 0.05) is about 2 pp. The gate's 5 pp is resolved to about ±1.4 pp. The measured interval replaces this
estimate in FINDINGS.

## 8. Calculation-stage expectations (recorded before any run, not results)

- **Headroom** (exploratory check, `STATE.md` Log): the public-only head gives 0.9044 on the server test set. A
  non-private centralized probe on all clients' training data gives at most 0.9548, the best of three L2 values chosen
  on the test set, so it is optimistic. A pass therefore needs DP-FedAvg to come within about 0.04 pp of that ceiling.
- **Prediction** (tuning layout, calculation config `E = 5`, lr 0.2, `eta_s = 1`):
  - The best predicted gain over public-only is 1.6 pp.
  - At `n = 48` the noise destroys the head for `mu <= 4`.
  - At `n = 480` the noise costs at most 0.6 pp, but noise-free FedAvg gains only 1-1.6 pp in 10-20 rounds.

  The tuning grid includes larger server steps for this reason.
- **Expected outcome:** G-B4 fails. The runs are cheap, so they are made anyway: the prediction rests on a
  first-order noise model and an untuned configuration.

## 9. Extension, escalation, cost

- **Extension (once):** if the best DP setting's gain lies between 4.0 and 6.0 pp, add 8 replicates
  (`j` = 8..15) to that setting and re-evaluate. No other extension is allowed.
- **Escalation:** as in ROADMAP Section 6 and v1 Section 4.4. A pass would contradict the calculation stage, so the
  code path is re-checked before it is reported.
- **Cost:** CPU only, about 1-2 CPU-hours for tuning and under 0.5 for evaluation, at one thread per worker and at most
  8 workers on the compute server. No GPU is claimed, so the 10 GPU-h cap is untouched.
- **Secondary empirical audit** (STATE.md): it needs Agent A's attackers after G-M0, so it is specified in a dated
  amendment here before it runs. It is not part of the gate.

## 10. Outputs

- Run JSONs: `results/cia_v2/formal_regime/results/{tune,evaluate}/`.
- Analysis: `analysis.json`.
- Findings: `FINDINGS.md`, which carries the crowd-bound results (`formal_regime_crowd.csv`) and the D6 answer.

## Amendment 1 (2026-10-17, owner request, after the original evaluation)

The owner asked, after reading FINDINGS.md: "try a smaller public dataset and run the secondary attack check. before we
accept the fail we should analyze carefully". This amendment is written after the original G-B4 result was known
(fail, best gain 3.49 pp). It is exploration requested by the owner. It does not replace that verdict, which stays
recorded in `analysis.json`. Every new reading below is fixed here, before any of its runs. Code:
`experiments/frozen_head/runner.py` (stages `*_public`, `*_extended`) and `experiments/frozen_head/audit.py`.

### A. Public-set sweep

**Question:** with less public data, does the formal regime beat the public-only model by 5 pp? The public data budget
is a property of the deployment, so a pass here is a statement about that smaller-public regime. The verdict for this
setup's 1,350-image public set is unchanged.

**Public sizes:** 20, 50, 100, 200 and 500 images, stratified (2, 5, 10, 20 and 50 per class). They are drawn from
the layout's public half by `default_rng([layout_seed, size, subset])`. The standardizer and the public-only head use
only the drawn subset. Cross-validation uses `min(5, per-class count)` folds, so 2 folds at 20 images.

**Tuning, per size:**

- Tuning layout (43), one public subset (`subset = 100`).
- The base grid, all 20 settings, 2 seeds: 1,440 runs per size and 7,200 in all.
- The clip base norm is re-measured from that size's public-only head.
- Selection as in Section 5.

**Evaluation, per size:**

- Evaluation layout (42), public subsets 0-4, `R = 4` replicates each (seeds 10000 + j and 20000 + j, shared across
  subsets and settings).
- 20 settings x 5 subsets x 4 = 400 runs per size, 2,000 in all.

**Metric and interval:**

- The metric is the mean over subsets of (mean DP-head accuracy minus that subset's public-only accuracy) on the
  1,350-image server test set.
- The interval is a bootstrap with 2,000 resamples. It resamples subsets, replicates within each drawn subset, and test
  images, with one image draw shared by all subsets.

**Reading:** Section 1's reading of G-B4, applied per size: gain at least 5.0 pp at `mu_total <= 8`, and "pass, not
robust" when the lower interval end is below 5.0. The settings that reach 5.0 pp with `mu_total <= 2` are listed
separately, since only those carry a meaningful bound (AUC at most 0.921).

**Smoke check (not a result):** at 20 public images the public-only head reaches 0.66 on the tuning layout's pooled
local tests, against 0.91 for the full public set.

### B. Extended-grid sensitivity (full public set)

**Question:** did the fail depend on the base grid? Every selected configuration lay on an edge of it.

**Grid:** `E` in {1, 5}, local learning rate in {0.01, 0.05, 0.2, 0.5, 1.0}, `eta_s` in {0.3, 1, 3, 10, 30} and clip
multiplier in {0.25, 0.5, 1}. Only the 114 grid points the base grid lacks are run: 20 settings x 114 x 2 seeds =
4,560 tuning runs. Selection then runs over all 150 points with the Section 5 rule, followed by 160 evaluation runs
with Section 6's seeds.

**Reading:** if the extended selection reaches 5 pp where the base grid did not, FINDINGS reports that the fail
depended on the grid. The protocol's verdict stays as computed.

### C. Secondary empirical audit (full public set)

**Design** (Agent A's M0 resolution, imported unchanged from `experiments/cia/`, merged from
`feature/cia-v2-measurement` @ `b8122d9d`):

- Randomized inclusion with 32 candidates by M0's rule (5 largest clients, then 27 by size quantile). At `n = 48` this
  is M0's fixed panel, checked at run time.
- `R = 32` replicates of `balanced_design(32, 32, seed=2026101801)`: each replicate includes 16 candidates.
- Every run has `n - 16` active clients (32 or 464), and that public constant is the denominator.
- Noise multiplier `z = sqrt(T) / mu_total`, so the formal `mu_total` is unchanged. The noise on the average is
  `n / (n - 16)` times B4's: 1.5x at 48 clients and 1.03x at 480.
- Replicate seeds come from `SeedSequence([2026101801, j])`. They are independent between the runs of a configuration
  and shared across configurations.
- Hyperparameters are B4's base selections.

**Shadow sample:** a stratified 10% of the candidate's training records with at least 2 (M0's convention). The shadow
loss is scored after every round, 1..T.

**Settings:** all 20 (`n` x `T` x `mu` and no noise): 640 runs.

**Attackers:** the calibrated loss attacker (primary) and the uncalibrated threshold.

**Metrics:**

- AUC, and TPR at 5% and 1% FPR.
- All 32 candidates and the 5 largest separately.
- Intervals: the leave-one-pair-out jackknife at `GATE_LEVEL` = 97.5%, and the two-way bootstrap at 95%.
- Compared with the formal bounds of `mu_total`-GDP: AUC `Phi(mu / sqrt 2)`, and TPR `Phi(mu - Phi^-1(1 - alpha))`.
- The GDP parameter implied by the measured AUC, `sqrt(2) Phi^-1(AUC)`.

**Readings (no gate):**

1. **Validity check:** a jackknife lower end of the calibrated AUC above the formal AUC bound means an error in the
   pipeline. It triggers a re-check and an escalation.
2. **Gap:** how far below the bound the realistic attacker sits, per setting.
3. **Leakage without DP:** the no-noise configurations measure leakage when no noise is added. Low leakage there
   would mean the attacker, not the mechanism, limits what the audit can show.

**Power:** M0's power table gives a jackknife half-width of about 0.05 AUC (all candidates, `R = 32`). The 5-client
subset is reported but cannot be resolved.

**Conditional:** if Part A finds a setting with `mu_total <= 2` that reaches 5 pp, the audit is repeated for that
setting at the smallest such public size (subset 0, same design). This is specified in a further dated amendment
before it runs.

### Cost and order

CPU only, at most 8 single-thread workers: C, then B, then A. About 13 CPU-hours, about 2.5 h wall time. No GPU.
