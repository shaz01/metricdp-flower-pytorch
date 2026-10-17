package: B4 (plus the subsampled-Gaussian accountant for B1 and C1)
status: planning
gate: G-B4 = not evaluated yet
protocol: results/cia_v2/formal_regime/PROTOCOL.md (to be written before the first head-only run; this STATE fixes the design)
machine: compute server (CPU only; no GPU claimed)
cost so far: 0 GPU-h of 10
next: accountant `metricdp_pytorch/accounting.py` with tests against published values, then the B4 calculations (noise-to-signal table, predicted accuracy loss, crowd bound)
needs from others: none. Agent B: read the accountant line below before using it.

accountant: planning. When it is ready, this line reads `ready @ <commit>`, and Agent B may merge `feature/cia-v2-formal-regime` into its own branch.

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, B4 (Agent C, branch `feature/cia-v2-formal-regime`, worktree
`metricdp-pytorch-feature-cia-v2-formal-regime`). B4 depends on no other package; the calculations come first.

## Hypothesis

The full-network regime has no non-vacuous client-level guarantee: at 48 clients and 50 rounds, `mu_total = 1` needs a
noise norm of about 396 against a signal norm of at most 5 (REVIEW.md Section 6). The hypothesis is that sharing only a
linear head on frozen public features closes that gap. The head is about 5,000 parameters instead of 289,194, and only
10-20 rounds are run. Per release the noise-to-signal ratio is

    NSR = noise norm / max signal norm = z sqrt(d) / n,   with   z = sqrt(T) / mu_total

(full participation, equal weights, public slot count `n`, so one round is `(1/z)`-GDP). For `d = 5,130`:

- `n = 48`, `T = 10`, `mu_total = 8`: NSR 0.59.
- `n = 48`, `T = 10`, `mu_total = 1`: NSR 4.7.
- `n = 480`, `T = 10`, `mu_total = 1`: NSR 0.47.
- The full model at `n = 48`, `T = 50`, `mu_total = 1`: NSR 79.

The question is whether some setting with `mu_total <= 8` leaves enough private signal to beat the public-only head by
at least 5 pp. The v1 stacked-head line found that the public-only model is the real competitor. Here the public set is
the server's public half of the EuroSAT test split, 1,350 images (135 per class), which is large for a linear probe on
ImageNet features. The headroom over public-only is therefore expected to be small; the gate decides.

## Design

**Architecture (public, fixed).**

- Extractor: torchvision ResNet-18 with ImageNet-1k weights (`IMAGENET1K_V1`, cached locally), frozen, in eval mode.
- Input: EuroSAT 64x64 RGB, resized to 224 with ImageNet normalization, no augmentation.
- Features: the 512-dimensional global-average-pool output, extracted once for all images and cached outside the
  repository.
- Shared object: a linear head of 512 x 10 weights plus 10 biases, `d = 5,130`.

**Data (public layout, shared by every run).**

- `n = 48`: the seed-42 EuroSAT layout, built by the frontier data module
  (`results/cia_frontier/eurosat_frontier/data.py`, Dirichlet alpha 0.3, `layout_seed = 42`, client train/test split
  as there). This is the layout of WP2, WP6 and M0, with top-decile clients 23, 7, 40, 18, 33.
- `n = 480`: the same 21,600 training images split into 480 clients by the same partition function, alpha 0.3,
  `layout_seed = 42`.
- Server: the frontier module's server split, which halves the 2,700-image test split. The public half (1,350 images)
  trains the public-only head. The other half (1,350 images) is the server test set on which the gate is evaluated.
- Tuning layout: `layout_seed = 43` with the same `n`, used only for hyperparameter selection.

**Mechanism.** Client-level DP-FedAvg on the head.

- Every client takes part in every round (`q = 1`).
- Each client runs `E` local epochs of minibatch SGD on cross-entropy over its training features, starting from the
  broadcast head.
- Each client update is clipped to L2 norm `C`.
- The server sums the clipped updates, divides by the public slot count `n` (equal weights), adds
  `N(0, (z C / n)^2 I_d)` from a dedicated generator, and takes a server step `eta_s`.
- The initial head is the public-only head, which is public randomness. Training therefore starts at the baseline.
- The formal guarantee is `mu_total = sqrt(T) / z` under add/remove-one-client adjacency. It is exact GDP (Dong, Roth
  and Su 2022), and the accountant reproduces it at `q = 1`.

**Settings.** The roadmap grid, 16 settings:

- `n` in {48, 480}.
- `T` in {10, 20}.
- `mu_total` in {1, 2, 4, 8}.
- A no-noise reference for each `(n, T)`, which does not count toward the gate.

**Replicates and randomness.**

- Each setting gets `R = 8` replicates.
- Replicate `j` has its own `client_seed` (local minibatch order) and `noise_seed`, both derived from
  `(design_seed, j)`.
- **Shared:** public randomness only. That is the extractor, the features, the layout, the server split and the
  initial head.
- Across settings, replicate `j` reuses the same `(client_seed, noise_seed)`. Section 2 item 1 allows this across
  configurations, and it reduces the variance of differences between settings.
- No attack worlds enter the gate, which is a utility gate. The secondary audit below has worlds, and they follow
  Section 2.

**Hyperparameters.** Chosen before any evaluation run and frozen in `PROTOCOL.md`; the tuning itself is not counted
in the privacy budget, and the protocol says so.

- **Public-only head:** L2 strength and epochs by 5-fold cross-validation on the public set only.
- **DP-FedAvg (`C`, `E`, local learning rate, `eta_s`):** a small grid per `(n, T, mu_total)`, run on the tuning
  layout (`layout_seed = 43`) with separate seeds, and selected by accuracy on the public set. The server test set is
  never used for selection.

**Calculation stage** (before any head-only DP run):

1. NSR table for the 16 settings, with the full-network regime alongside.
2. Predicted accuracy loss. Run noise-free head-only FedAvg for each `(n, T)` on the tuning layout. Then perturb its
   final head by Gaussian noise with the accumulated server-noise std `eta_s sqrt(T) z C / n` per coordinate (first
   order, ignoring the contraction from later rounds, so pessimistic). The prediction is the Monte Carlo test accuracy
   of the perturbed head on the public set.
3. The D3 crowd bound at these dimensions (the D6 question). Propositions 1 and 2 of
   `research/cia_review_2026-10/theory/f_cmip.md` are evaluated per round on noise-free head trajectories at
   `layout_seed = 42`, then composed over independent rounds (Proposition 3.1):
   - Full release: `mu_c / mu_GDP`, with `Sigma_B` estimated from the other clients' head updates in that round
     (rank at most `n - 1`, applied with the Woodbury identity).
   - The loss statistic (`k = 1`): `rho`.
   - Targets: all clients, with the top decile by size reported separately.

## Attacker and metrics

**Gate metric.**

- Final-round accuracy on the 1,350-image server test set, as the mean over the `R = 8` replicates.
- It is compared with the public-only head on the same images, as a paired difference.
- 95% interval: a two-way bootstrap over replicates and test images (2,000 resamples).
- Also reported: the per-client minimum accuracy on the clients' local test splits (Section 2 item 5), and
  accuracy at every round.

**Formal privacy.**

- `mu_total` for every setting from `metricdp_pytorch/accounting.py`.
- `epsilon` at `delta = 1e-5` alongside.
- The crowd-bound ratios from the calculation stage.

**Secondary empirical audit (not part of the gate; runs only after G-M0 passes).** Training a head is cheap, so for
the best setting at `n = 48` and its `mu_total = 8` neighbour I can afford a full audit:

- Section 2 randomized inclusion with M0's 16 candidates and design matrix, at `R = 16`.
- Independent `client_seed` and `noise_seed` per replicate.
- Agent A's attackers, imported from `experiments/cia/` and not edited: the calibrated loss attacker (primary) and the
  uncalibrated threshold.
- Reported: AUC, TPR at 5% and 1% FPR, for all and for top-decile targets, against the formal trade-off
  `1 - G_mu(alpha)` and the crowd bound.

The noise-level attacker does not apply, because the noise level here is public.

**Power.** The protocol states the minimum detectable accuracy difference at `R = 8`, using the replicate variance
measured on the tuning layout.

## Gate (copied verbatim from ROADMAP.md Section 3)

**G-B4:** some setting with `mu_total` of 8 or less reaches at least 5 pp above the public-only model on the server test
set. If it passes, the owner may open a formal comparison there: global DP vs private-median metric privacy vs the best
B mechanism, each re-run with the same frozen-head architecture (full-network results do not transfer). If it fails,
record "no non-vacuous client-level regime at this scale" as a paper statement.

**How the gate is read** (this fixes the reading; it does not change the threshold):

- "Reaches" means the replicate mean of (DP head accuracy minus public-only accuracy) is at least 5.0 pp.
- The 95% interval is reported with it.
- A pass whose interval's lower end is below 5.0 pp is labelled "pass, not robust".
- Every setting is reported, not only the best one.

## Accountant (delivered first; Agent B uses it for B1 and C1)

`metricdp_pytorch/accounting.py` wraps Google's `dp-accounting` library (version 0.6.0, privacy loss distributions),
with add/remove-one adjacency. It provides:

- `epsilon` at a given `delta`, and `delta` at a given `epsilon`, for `T` rounds of the Poisson-subsampled Gaussian
  mechanism with noise multiplier `z` (noise std `z C` on the sum) and sampling rate `q`. At `q = 1` this is the
  exact Gaussian.
- The amplified formal `mu` that G-B1 compares, defined as the smallest `mu` for which the mechanism is `mu`-GDP. That
  holds exactly when its privacy profile satisfies `delta(eps) <= delta_mu(eps)` for every `eps >= 0` (Dong, Roth and
  Su 2022, Corollary 2.13). The profile is checked on a dense `eps` grid, with the PLD's pessimistic discretization.
- The central-limit `mu_CLT = q sqrt(T (exp(1/z^2) - 1))` (Dong, Roth and Su 2022, Section 5). It is labelled
  asymptotic, not a guarantee.
- The inverse: the noise multiplier `z` for a target `mu` or `(epsilon, delta)`, given `q` and `T`.
- An RDP value from the same library, used as a cross-check.

Tests in `tests/test_accounting.py`:

- Closed forms: at `q = 1`, `mu = sqrt(T) / z` and the analytic Gaussian `delta(eps)` of Balle and Wang 2018.
- Published DP-SGD values, each verified against its source before it enters a test, for example Abadi et al. 2016's
  `q = 0.01, z = 4, T = 10,000, delta = 1e-5`.
- Ordering: the PLD `epsilon` never exceeds the RDP `epsilon`, and the dominating `mu` is at least `mu_CLT` in the
  small-`q` regime.
- Monotonicity in `q`, `z` and `T`.

## Files

| File | Change |
|---|---|
| `metricdp_pytorch/accounting.py` (new), `tests/test_accounting.py` (new) | The accountant and its tests |
| `pyproject.toml`, `uv.lock` | Add `dp-accounting==0.6.0`. These files have no owner in Section 6, so the owner is asked first. Fallback: a lazy import used through `uv run --with dp-accounting`. |
| `experiments/frozen_head/` (new: `features.py`, `data.py`, `fedavg.py`, `public_only.py`, `crowd.py`, `runner.py`, `tests/`) | Feature cache, layouts, head-only DP-FedAvg simulator (in process, NumPy/torch on CPU), crowd bound, runner (plans unless `--execute`, resumable, atomic JSON) |
| `research/cia_roadmap_v2/theory/` (new: `accounting.md`, `formal_regime.md`, `formal_regime_calcs.py`) | The accountant's definitions and checks; the B4 calculations |
| `results/cia_v2/formal_regime/` (`STATE.md`, `PROTOCOL.md`, `FINDINGS.md`, `tables/`, `figures/`, `results/`) | This package |
| `STATUS.md` | Agent C's bullet and rows only |

Not touched: every file Section 6 assigns to Agents A or B, and `experiments/stacked_head/`. Its code is imported only
if it is reused unchanged.

## Log

| When | What |
|---|---|
| 2026-10-17 | Branch `feature/cia-v2-formal-regime` from master 7fda7242; sparse worktree opened. Read HANDOFF, REVIEW, ROADMAP, SYNTHESIS, WP3's `f_cmip.md`, the stacked-head README, the EuroSAT and frontier data modules (server split 1,350 public plus 1,350 test), and Agent A's M0 STATE. Checked that `dp-accounting` 0.6.0 installs on Python 3.13 and that the ResNet-18 ImageNet weights are cached. This STATE written before any code change. |
