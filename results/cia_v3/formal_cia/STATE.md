package: P5, client inference in the formal regime (roadmap v3, F7)
status: ESCALATED 2026-10-30 (feature-cache fingerprint differs from B4's; ESCALATION.md); no P5 run launched
gate: G-P5 (report, no kill), not yet evaluated
protocol: results/cia_v3/formal_cia/PROTOCOL.md @ 45f3af54 (fixed before any run)
machine: owner PC, CPU only (no GPU claimed; the v3 GPU cap is untouched)
cost: estimate 2-3 CPU-hours of run time (832 runs of about 2-15 s, from B4 Part D's recorded seconds), plus a
  one-off ResNet-18 feature extraction on this machine (the cache is absent here)
next: owner decision on ESCALATION.md; then reproduction check (a) under the chosen rule, runs, FINDINGS.md
needs from others: owner decision (ESCALATION.md options 1-3)

Roadmap: `research/cia_roadmap_v3/ROADMAP.md` Section 3, P5, under Section 1 (decisions 1-8) and measurement standard
v3 (Section 2). Agent F. Branch `feature/cia-v3-formal-cia` from master `103d26b8`. Agent F owns
`results/cia_v3/{formal_cia,label_channel}/`, `research/cia_roadmap_v3/label_channel/` and head-only code under
`experiments/`; nothing Agent P owns is edited.

## What is already known (stated before the design, so it cannot be tuned to it)

B4's secondary audit (`results/cia_v2/formal_regime/FINDINGS.md`, Amendment 2 Part D) already measured **global DP**
on this head with the same randomized-inclusion design at 20 public images, `T = 10`, calibrated attacker, 97.5% pair
jackknife:

| `n` | `mu_total` | AUC [97.5%] | accuracy |
|---|---|---|---|
| 48 | 1 | 0.599 [0.546, 0.651] | 0.714 |
| 48 | 2 | 0.638 [0.587, 0.689] | 0.843 |
| 480 | 1 | 0.545 [0.493, 0.598] | 0.910 |
| 480 | 2 | 0.547 [0.496, 0.599] | 0.921 |

So the first G-P5 item is predictable: at `n = 48` the upper end at `mu = 2` was 0.689 (above 0.60), at `n = 480`
0.599 (at the threshold). P5 re-measures global DP on fresh replicates and adds what B4 never ran: metric privacy's
distance calibration and Poisson subsampling on the same head, with matched-accuracy comparisons.

## Hypothesis

(DIRECTIONS.md F7.) Where the guarantee is non-vacuous, global DP and subsampled DP both keep the calibrated
attacker's AUC near 0.5, and metric privacy's distance calibration gives no extra protection at matched accuracy
(the full-network A-3 result carries over to the formal regime). Prediction from B4: the regime is protective at
`n = 480` (pseudo-clients) and not, by the 0.60 rule, at the real `n = 48`.

## Design

**Regime: B4's confirmed scarce-public formal regime**, where head-only DP-FedAvg at `mu <= 2` beats the best
public-only head by 14-26 pp (B4 confirmation). At the full 1,350-image public set DP buys only about 2 pp, so that
setting is not a useful regime and is not run.

- Frozen ResNet-18 ImageNet features of EuroSAT, linear head (`d = 5,130`), `experiments/frozen_head/`.
- Evaluation layout 42 (label-Dirichlet 0.3), public set = B4 Part A's evaluation subset 0 of **20 images**
  (standardizer, initial head and clip base from that subset, as Part D). Server test set: the other 1,350 images.
- `T = 10` rounds. Two federations: **`n = 48` (primary; real clients)** and `n = 480` (secondary; each real client
  split into 10 pseudo-clients, so its guarantee is per pseudo-client, `10 mu` per real client by group privacy).
- Hyperparameters: B4 Part A's tuning selections at 20 public images (tuned on layout 43, not accounted, as in B4).
  `mu = 0.5` uses `mu = 1`'s selection. Metric privacy and subsampling at a level use global DP's selection at the
  same nominal `mu`. No per-mechanism tuning.

**Randomized inclusion (measurement standard v2 and v3 item 1, as B4 Part C/D):** 32 candidates by M0's rule (the 5
largest clients 23, 7, 40, 18, 33 plus 27 by size quantile; at `n = 48` M0's fixed panel, checked at run time),
`R = 32` replicates of `balanced_design(32, 32, seed)`: each replicate includes 16 candidates, each candidate is IN in
exactly 16 of 32 replicates, and the other clients always take part. Active clients per run: `n - 16` (32 or 464),
a public constant and the denominator. 1,024 trials per configuration.

**Mechanisms and levels** (10 rounds; `z` = noise multiplier; noise std on the average `z C / n_active` unless
stated):

| Mechanism | Levels | Noise | Formal guarantee |
|---|---|---|---|
| Vanilla | one | none | none |
| Global DP | `mu_total` 0.5, 1, 2, 4 (`z` = 6.325, 3.162, 1.581, 0.791) | `z C / n_active` | exact `sqrt(T)/z` |
| Metric privacy (the paper's calibration on the head) | the same four `z` | `(z / d_t) C / n_active`, `d_t` the maximum pairwise mean layer-wise (W, b) Euclidean distance between the active clients' local models that round, as `metricdp_strategy.py` (same fallback for a non-finite or zero distance) | none |
| Poisson subsampling, `q = 0.25` | dominating `mu` 0.5, 1, 2, 4 at `delta_floor` 1e-10 (`z` = 2.205, 1.340, 0.832, 0.511) | sampled clipped sum over the public denominator `q n_active`, noise `z C / (q n_active)`; empty rounds release noise only (B1's strategy) | `accounting.dominating_mu` |

13 configurations per federation, 26 in all, 832 runs. The `mu = 0.5` levels exist so that the `mu = 1` target is
inside every curve rather than at its end.

**Private randomness, no sharing between worlds.** Each replicate is one world. Replicate `j` draws its own
`(client_seed, noise_seed, sampling_seed)` from `SeedSequence([P5_DESIGN_SEED = 2026103001, j])`. **No two runs of
one configuration share any private randomness** (server noise, client batch order, the inclusion of candidates, the
Poisson sample). Replicate `j` reuses its seeds and inclusion across configurations, which standard v2 Section 2
item 1 allows and which makes the matched-accuracy differences paired. Shared public randomness only: layout, public
subset, initial head, shadow samples, design matrix. The design seed is fresh (B4 used 2026101801), so P5's
global-DP numbers are a new measurement, not B4's runs again.

**Declared extension (once).** If a mechanism's mean accuracies over its four levels do not bracket a target, one
level is added beyond its outermost level on the missing side (global DP and metric privacy: `z` doubled or halved;
subsampling: dominating `mu` 0.25 or 8), decided on accuracy alone before any leakage statistic is computed.

## Attacker and metrics

- **Primary attacker:** the calibrated (LiRA-style, leave-one-replicate-out) loss attacker of
  `experiments/cia/audit_attacks.py` on each candidate's shadow-loss trajectory after every round `1..T` (round 0 is
  the public initial head, identical in all runs). Shadow sample: stratified 10% of the candidate's training records,
  at least 2 (M0). Every-round scoring (standard v3 item 4) is automatic at `T = 10`.
- **Sampling-aware attacker (standard v3 item 3)** for subsampling: the realized per-round schedule is logged, and
  `reanalyze.py`'s centred window-drop statistic is appended to the loss features (`schedule_statistic`, then the
  calibrated attacker), exactly as `sampling_aware`. Any claim about subsampling names the attacker it holds for.
- Secondary: the uncalibrated threshold attacker.
- **Metrics:** AUC and TPR at 5% and 1% FPR over the 1,024 trials, all candidates and the top decile (5 largest)
  separately; mean final server-test accuracy; lowest per-client local-test accuracy, interpolated at the targets
  (standard v3 item 6, not gated); formal `mu` and `epsilon` at `delta = 1e-5`; the formal AUC bound
  `Phi(mu / sqrt 2)`; implied `mu` from the AUC; for metric privacy the realized `d_t` and noise std per round.
- **Intervals:** pair jackknife at 97.5% (gate interval), two-way bootstrap at 95% alongside.
- **Matched-accuracy statistic (A1's):** per mechanism, AUC linear in mean final accuracy between levels (vanilla
  included as the noise-free end), `AUC_X(a*) - AUC_Y(a*)`, every resample drawing the same pairs in all
  configurations. Comparisons: metric privacy minus global DP; subsampling minus global DP; subsampling minus metric
  privacy; the last two also with subsampling under the sampling-aware attacker.
- **Targets, fixed now from B4 Part D's committed global-DP accuracies** (not from P5's runs): `n = 48`: 0.843
  (`mu = 2`) and 0.714 (`mu = 1`); `n = 480`: 0.921 and 0.910.
- **Power (standard v3 item 2):** the paired matched-accuracy statistic applies. A1's detectable difference (about
  0.03-0.04 for all candidates) is from the 50-round EuroSAT CNN and is not assumed to transfer; FINDINGS reports the
  measured jackknife half-widths. Single-configuration AUCs have half-widths of about 0.05 (B4 Part D).
- **Reproduction first (standard v3 item 7):** before any new number is reported, (a) the modified simulator re-runs
  one committed B4 Part D run (`n48_T10_mu2_pub20_r00`) and must reproduce its shadow losses and final accuracy, and
  (b) P5's analysis code must reproduce B4 Part D's published `n = 48`, `T = 10`, `mu = 2` AUC 0.638 [0.587, 0.689] to
  3 decimals from the committed JSONs. The local feature cache must match B4's fingerprints (train
  `30ee5d8828e2339c`, test `cdd8f364245a012b`).
- **Validity check (B4's):** a jackknife lower end above the formal AUC bound means a pipeline error and triggers
  `ESCALATION.md`.

## Gate (copied verbatim from ROADMAP.md Section 3, P5)

> **G-P5 (report).**
> - Global DP at `mu = 2` with an AUC upper bound of at most 0.60: "protective regime confirmed".
> - The matched-accuracy differences between the three mechanisms are reported with the pair jackknife.

**How it is read (fixed here; the threshold is unchanged):**

- "AUC upper bound" is the upper end of the 97.5% pair-jackknife interval of the calibrated attacker's
  all-candidate AUC for global DP at `mu_total = 2` (the formal bound at `mu = 2` is 0.921, so it cannot be meant).
  Read per federation; the headline verdict is `n = 48` (real clients), and `n = 480` is reported as the
  pseudo-client federation. Upper end at most 0.60: "protective regime confirmed"; otherwise "not confirmed", with
  the measured interval.
- This item uses a measured configuration, so it is always testable.
- **Matched-accuracy differences** are labelled with decision 9's rule ("beats": at least 0.05 lower and the 97.5%
  interval below 0) and A1's labels (better / equivalent within ±0.03 / no detectable difference / worse or
  inconclusive).
- **Untestable targets (standard v3 item 5):** a target outside either curve's accuracy range (point estimates) is
  "not testable" and does not count as a pass. If it is inside both curves but some leave-one-pair-out curve no longer
  reaches it, the gate interval is "not computable": the comparison is "not testable by the gate interval" and the
  two-way bootstrap is reported. A comparison with no testable target is "not testable".

## Files I will touch

- `experiments/frozen_head/fedavg.py`: backward-compatible options for metric-privacy calibration and Poisson
  client sampling (defaults leave every committed B4 run bit-identical; checked by reproduction check (a)).
- `experiments/frozen_head/formal_cia.py` (new): P5's runner and analysis, reusing `audit.py`, `runner.py`, the A1
  matched-accuracy machinery and `reanalyze.py`'s schedule statistic by import.
- `experiments/frozen_head/tests/test_formal_cia.py` (new): synthetic tests (no dataset).
- `results/cia_v3/formal_cia/`: `STATE.md`, `PROTOCOL.md`, `FINDINGS.md`, `analysis.json`, `figures/`, `tables/`,
  run JSONs in `results/`.
- `STATUS.md`: Agent F's rows only, on this branch.

## Log

- 2026-10-30: branch and sparse worktree opened from master `103d26b8`. STATE.md written before any code change.
- 2026-10-30: PROTOCOL.md committed @ `45f3af54` before any run.
- 2026-10-30: `fedavg.run` options (`calibration="metric"`, `sampling_rate`, `sampling_seed`), `formal_cia.py`, 12 new
  synthetic tests; `pytest experiments/frozen_head/tests` 36 passed. Default path checked bit-identical to the
  pre-P5 loop.
- 2026-10-30: feature cache extracted on the owner PC (CPU): fingerprints `8b56243fcb32340f` / `4ae40e575a25aef2`,
  not B4's `30ee5d8828e2339c` / `cdd8f364245a012b`. Reproduction (b) passes (0.638 [0.587, 0.689]); reproduction (a)
  gives identical accuracy and shadow losses within 1.8e-5 on five B4 runs, failing the 1e-12 criterion. Stopped per
  PROTOCOL.md Section 8; ESCALATION.md.
