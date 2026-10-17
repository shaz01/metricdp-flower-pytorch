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
