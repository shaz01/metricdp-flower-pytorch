# CIA roadmap v3, P5: client inference in the formal regime

Package P5 of `research/cia_roadmap_v3/ROADMAP.md` (direction F7), Agent F, branch `feature/cia-v3-formal-cia`, on the
owner PC (CPU only). The owner approved v3 on 2026-10-30 (decision 1) and closed P5 on 2026-10-30.

All numbers below come from committed artifacts in `results/cia_v3/formal_cia/`:
- `STATE.md`: design (written before any code change) and log.
- `PROTOCOL.md`: fixed @ `45f3af54` before any run. Amendment 1 (@ `a8273d8a`), also before any run, records the
  owner's decision on `ESCALATION.md`.
- `reproduction.json`: the two reproduction checks.
- `analysis.json`: built by `python -m experiments.frozen_head.formal_cia analyze` from the 832 runs in `results/runs/`.
- `tables/configs.csv`, `tables/matched.csv` and `figures/auc_vs_accuracy.png`, from `make_figures.py`.
- `FINDINGS.md`.

## 1. Question

Every useful setting in roadmap v2 had a vacuous client-level guarantee. The one exception was B4's linear head on
frozen public ResNet-18 features. With 20 public images, DP-FedAvg at `mu <= 2` beats the best public-only head there
by 14-26 pp (B4 confirmation). Client inference against competing mechanisms had never been measured in that regime.

P5 asks two things:
- Is global DP actually protective there?
- At matched accuracy, do metric privacy's distance calibration or Poisson subsampling protect better than global DP?

That would give the hybrid paper a setting where a "beats" claim comes with a real guarantee.

## 2. Mechanism and design

**Setting:**
- B4's head (`d = 5,130`) on frozen ResNet-18 ImageNet features of EuroSAT.
- Layout 42 (label-Dirichlet 0.3), public subset 0 of 20 images, `T = 10` rounds.
- Two federations:
  - `n = 48`, the real clients (headline);
  - `n = 480`, each client split into 10 pseudo-clients, so the guarantee is `10 mu` per real client.
- Hyperparameters: B4 Part A's global-DP tuning at 20 images, reused for every mechanism.

**Randomized inclusion** (measurement standard v2 and v3, as B4's audit):
- 32 candidates (M0's panel at `n = 48`), `R = 32` balanced replicates, 16 IN per replicate.
- `n - 16` active clients, a public denominator.
- 1,024 trials per configuration.

**Seeds:**
- A fresh design seed, 2026103001.
- Each replicate draws its own client, noise and sampling seeds, so no two runs of a configuration share private
  randomness.
- Replicates are shared across configurations, which pairs them.

**Mechanisms** (13 configurations per federation, 832 runs):

| Mechanism | Levels | Noise on the released average |
|---|---|---|
| Vanilla | one | none |
| Global DP | `mu_total` 0.5, 1, 2, 4 (exact GDP) | `z C / n_active`, `z = sqrt(T) / mu` |
| Metric privacy (the paper's calibration) | the same four `z` | `(z / d_t) C / n_active`, `d_t` the maximum pairwise mean layer-wise distance between local models |
| Poisson subsampling, `q = 0.25` | dominating `mu` 0.5, 1, 2, 4 (`delta_floor` 1e-10) | `z_q C / (q n_active)` |

**Attackers and statistic:**
- **Attackers:**
  - the calibrated loss attacker on shadow losses after every round (primary);
  - for subsampling, also the sampling-aware attacker of the v3 re-analysis (window 1).
- **Statistic:** A1's matched-accuracy difference, with the pair jackknife at 97.5% (gate interval) and the two-way
  bootstrap.
- **Targets**, fixed before the runs from B4's committed global-DP accuracies:
  - 0.843 and 0.714 at `n = 48`;
  - 0.921 and 0.910 at `n = 480`.

**Gate G-P5 (report):**
- "Global DP at `mu = 2` with an AUC upper bound of at most 0.60: 'protective regime confirmed'."
- Matched-accuracy differences reported with the pair jackknife.
- "Upper bound" is read as the 97.5% jackknife upper end, fixed in STATE.md.

## 3. Reproduction and escalation

- **The cache.** The owner PC had no feature cache. The locally extracted one has different fingerprints from B4's.
  The protocol required a stop, so `ESCALATION.md` was written.
- **The size of the difference.** Five committed B4 runs, re-run here, reproduce their accuracy exactly. Their shadow
  losses differ by at most 1.8e-5, which is float32 rounding across CPUs.
- **The resolution.** The owner chose to accept a tolerance (Amendment 1: identical accuracy, losses within 1e-4).
- **Both checks then pass** (`reproduction.json`):
  - (a) the five runs within 1.8e-5;
  - (b) P5's analysis code recovers B4's published 0.638 [0.587, 0.689] to 3 decimals.

## 4. Results

**G-P5:**

| Federation | Global DP `mu = 2`: AUC [97.5% jackknife] | Accuracy | Verdict |
|---|---|---|---|
| **48 clients** | **0.639 [0.598, 0.680]** | 0.842 | **not confirmed** |
| 480 pseudo-clients | 0.480 [0.438, 0.523] | 0.920 | protective regime confirmed |

**Per configuration** (all candidates; from `tables/configs.csv`):

| `n` | Config | `mu` | `epsilon` (1e-5) | Accuracy | AUC [97.5%] | Sampling-aware AUC | Lowest client |
|---|---|---|---|---|---|---|---|
| 48 | vanilla | — | — | 0.929 | 0.921 [0.905, 0.938] | — | 0.799 |
| 48 | gdp-mu1 | 1.0 | 4.38 | 0.711 | 0.576 [0.528, 0.624] | — | 0.462 |
| 48 | gdp-mu2 | 2.0 | 10.0 | 0.842 | 0.639 [0.598, 0.680] | — | 0.652 |
| 48 | gdp-mu4 | 4.0 | 24.4 | 0.900 | 0.726 [0.690, 0.762] | — | 0.745 |
| 48 | mp-mu1 | none | none | 0.635 | 0.532 [0.484, 0.580] | — | 0.372 |
| 48 | mp-mu2 | none | none | 0.830 | 0.631 [0.591, 0.672] | — | 0.630 |
| 48 | mp-mu4 | none | none | 0.888 | 0.708 [0.673, 0.743] | — | 0.724 |
| 48 | ss-mu2 | 2.0 | 8.21 | 0.657 | 0.554 [0.511, 0.598] | 0.648 | 0.408 |
| 48 | ss-mu4 | 4.0 | 19.9 | 0.788 | 0.592 [0.558, 0.627] | 0.693 | 0.567 |
| 480 | vanilla | — | — | 0.926 | 0.852 [0.825, 0.880] | — | 0.0 |
| 480 | gdp-mu2 | 2.0 | 10.0 | 0.920 | 0.480 [0.438, 0.523] | — | 0.0 |
| 480 | mp-mu2 | none | none | 0.926 | 0.734 [0.701, 0.766] | — | 0.0 |
| 480 | ss-mu4 | 4.0 | 19.9 | 0.914 | 0.526 [0.491, 0.560] | 0.531 | 0.0 |

Every level, including `mu` 0.5, is in `tables/configs.csv`.

**Matched accuracy** (all candidates; difference [97.5% jackknife]; none beats its reference):

| `n` | Comparison | Target | Difference | Label |
|---|---|---|---|---|
| 48 | metric privacy − global DP | 0.843 | +0.008 [-0.001, 0.017] | equivalent |
| 48 | metric privacy − global DP | 0.714 | -0.005 [-0.015, 0.004] | equivalent |
| 48 | subsampling − global DP | 0.714 | -0.007 [-0.050, 0.036] | no detectable difference |
| 48 | subsampling (sampling-aware) − global DP | 0.714 | +0.090 [0.040, 0.140] | worse |
| 48 | subsampling (sampling-aware) − metric privacy | 0.714 | +0.096 [0.052, 0.139] | worse |
| 480 | metric privacy − global DP | 0.910 | -0.004 [-0.043, 0.035] | no detectable difference |
| 480 | subsampling − global DP | 0.910 | +0.007 [-0.042, 0.055] | no detectable difference |

## 5. Reading

1. **The formal regime is protective only for many small owners.**
   - At the real 48 clients, `mu = 2` (`epsilon` 10 at `delta = 1e-5`; formal AUC bound 0.921) still lets the
     attacker reach 0.639.
   - At 480 pseudo-clients the attacker is at chance (0.480).
   - B4's audit had found the same on other seeds (0.638 and 0.547).
2. **Metric privacy is global DP in this regime, at matched accuracy.**
   - It is equivalent within ±0.03 at both 48-client targets, with jackknife half-widths of about 0.01.
   - Mechanically, its distance `d_t` sits at 0.79-1.10 (median) at 48 clients, so its noise is 0.91-1.26 times
     global DP's.
   - At 480 clients and `mu >= 2`, `d_t` rises to about 5.9. It then adds about 6 times less noise and leaks like
     vanilla (0.734-0.797 against 0.852). That is the paper's calibration trading leakage for accuracy, with no
     guarantee.
   - A-3 therefore holds in the one regime where global DP has a meaningful guarantee.
3. **Subsampling's full-network advantage does not carry over.**
   - On the EuroSAT CNN, subsampling beat global DP by 0.10 at matched accuracy (D-3), and the composite survived a
     sampling-aware attacker (v3 re-analysis).
   - Here, at 48 clients, subsampling shows no detectable difference under the standard attacker, and leaks 0.09 AUC
     more once the schedule is known.
   - At equal formal `mu` it lowers the standard attacker's AUC (-0.053 to -0.134) only by losing 11-20 pp of accuracy.
   - With about 8 participants per round, each sampled round is a large share of a candidate's contribution, which
     the schedule-aware attacker exploits (+0.08 to +0.10 AUC over the standard attacker).
4. **Fairness is unchanged.** At matched accuracy the lowest client accuracy is 0.466-0.477 at 0.714 for all three
   mechanisms. That differs from the full network, where subsampling cost 4-9 pp.

## 6. Anomalies and corrections

1. **The protocol drew subsampling's curve through vanilla.**
   - Vanilla was declared the noise-free end of every curve. That is right for global DP and metric privacy. For
     subsampling it is wrong: its noise-free limit (subsampling without noise) was not run.
   - The pre-registered numbers call subsampling "worse" at the upper targets (+0.081 at 0.843, +0.218 at 0.921).
     They are interpolated through vanilla, so they are not a measurement.
   - **Corrected reading (post hoc, stated in FINDINGS):** those targets are not testable for subsampling. The
     lower-target results do not involve vanilla.
   - `analysis.json` keeps the pre-registered numbers; the figure shows the corrected curves.
2. **Metric privacy's 480-client curve is vertical.** Its accuracy equals vanilla's at `mu` 2 and 4 (0.926). The
   0.921 comparison (+0.239 [-0.137, 0.615]) is not interpretable.
3. **The feature cache** differs from B4's (Section 3). This was resolved by owner decision before any run.
4. **The full test suite** hung on a Flower integration test, idle for over an hour. The suite was re-run without the
   six Flower-launching test files, none of which imports the changed code: 852 passed, 2 skipped. The re-analysis
   tests (3) pass in the main checkout.

## 7. Caveats

- One extractor, one dataset, one layout, one public subset of 20 images, `T = 10`.
- Hyperparameters were tuned for global DP only. Per-mechanism tuning could change subsampling's accuracy.
- The 480-client rows are a pseudo-client split.
- The in-process NumPy simulator was used, not the Flower path, as in B4.
- The top-5 subset is reported in the tables but cannot be resolved at `R = 32`.
- An empirical audit bounds nothing from above. The formal `mu` remains the privacy statement. No jackknife lower end
  exceeds its formal AUC bound.

## 8. Cost

- 832 runs, 0.32 CPU-hours of recorded run time, 6 single-thread workers on the owner PC, 2026-10-30.
- Analysis about 2 minutes.
- One feature extraction (not timed). No GPU; the v3 GPU cap is untouched.

## 9. Conclusions and owner decisions (2026-10-30)

1. **G-P5 is reported as:**
   - "not confirmed" at the real 48 clients (0.639 [0.598, 0.680]);
   - "protective regime confirmed" at 480 pseudo-clients (0.480 [0.438, 0.523]).
2. **For the paper's formal-regime paragraph:**
   - metric privacy leaks like global DP at matched accuracy even where the guarantee means something;
   - subsampling gives no advantage there, and loses to an attacker who knows the schedule.
3. **The owner accepted the feature-cache tolerance** (Amendment 1) and closed P5 on 2026-10-30.
