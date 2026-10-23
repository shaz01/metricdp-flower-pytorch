# B4 confirmatory run: the scarce-public positive on a fresh layout

Owner decision 11 (`research/cia_roadmap_v2/ROADMAP.md` Section 10, 2026-10-20). Branch
`feature/cia-v2-formal-confirm`. Fixed 2026-10-20, before any run. B4's design, code and selections are unchanged
(`results/cia_v2/formal_regime/PROTOCOL.md`, Amendments 1 and 2); only the evaluation layout, the public subsets and
the replicate seeds are new.

## 1. What is being confirmed

B4's exploration result (FINDINGS, owner decision 2 of 2026-10-17): "with at most 10-20 public images per class,
head-only DP-FedAvg at `mu <= 2` beats the public-only head by 5-26 pp", robust against the strongest legitimate
public-only baseline (`best_legit`) in these 13 settings (`analysis_baseline_check.json`):

| Public images | `(n, T, mu)` robust against best_legit (lower end at least 5 pp) |
|---|---|
| 20 | (48, 10, 1), (48, 10, 2), (48, 20, 2), (480, 10, 1), (480, 10, 2), (480, 20, 1), (480, 20, 2) |
| 50 | (48, 10, 2), (48, 20, 2), (480, 10, 1), (480, 10, 2), (480, 20, 1), (480, 20, 2) |

It was found after B4's pre-registered result was known, on evaluation layout 42 with public subsets 0-4.

## 2. Runs

| Item | Value |
|---|---|
| Evaluation layout | `layout_seed = 44` (fresh: B4 used 42 for evaluation and 43 for tuning) |
| Public sizes | 20 and 50 images (2 and 5 per class) |
| Public subsets | 0-4 of layout 44, drawn by `default_rng([44, size, subset])` from layout 44's public half (fresh subsets) |
| Settings | the 8 settings with `mu_total <= 2` plus the 4 no-noise references, per size |
| Hyperparameters | B4's Part A tuning selections for that size (tuning layout 43, `tune_public`), unchanged; clip base norm as in Part A |
| Replicates | `R = 4` per subset, seeds `client_seed = 30000 + j`, `noise_seed = 40000 + j` (fresh; shared across subsets and settings) |
| Count | 2 sizes x 12 settings x 5 subsets x 4 replicates = 480 runs |
| Code | `experiments/frozen_head/runner.py` stage `confirm_public`; analysis `experiments/frozen_head/confirm.py` |
| Output | `results/cia_v2/formal_regime/results/confirm_public/`, `results/cia_v2/formal_regime/confirm/analysis.json` |

CPU only, at most 8 single-thread workers; B4's runs took 1-15 s each, so under 1 CPU-hour.

## 3. Analysis

- **Gain** per size and setting: the mean over subsets of (mean DP-head accuracy minus that subset's public-only
  accuracy) on layout 44's 1,350-image server test set; 95% interval from B4's bootstrap (subsets, replicates within
  subsets, test images).
- **Baselines** (Amendment 2, part E, refit on layout 44's subsets): `protocol` and `best_legit` (per subset, the most
  accurate of `protocol`, `wide_cv`, `l2norm_cv`, `nearest_mean` on the server test, which favours the baseline).
  `oracle_c` is reported as a ceiling.

## 4. Confirmation rule

Applied against `best_legit`, per size, to the settings of Section 1:

- **Confirmed:** every listed setting has a lower interval end of at least 5 pp.
- **Partly confirmed:** some but not all do; the ones that do are listed.
- **Not confirmed:** none does.

All 16 settings with `mu_total <= 2` and the no-noise references are reported, whatever the outcome.
