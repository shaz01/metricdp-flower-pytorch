package: A1
status: running (stage 1, then stage 2; both GPUs, two workers each)
gate: G-A1 = not evaluated yet (report, no kill)
protocol: results/cia_v2/frontier/a1/PROTOCOL.md (fixed 2026-10-19, before any A1 leakage run)
machine: compute server
cost so far: about 2.3 GPU-h (accuracy calibration, 20 runs) of ROADMAP.md's 30 GPU-h cap (stop at 37.5); main runs about 25 GPU-h
next: stage 1 (112 runs), replacement rule on accuracy, stage 2; then analysis
needs from others: none (M2's runs serve as the global-DP curve; M2 found no useful regime, so the targets are M2's two levels around 0.70 accuracy: 2x, 0.740, and 4x, 0.660)

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, A1 (Agent A, branch `feature/cia-v2-frontier`). The design
follows the measurement standard as amended by `results/cia_v2/measurement/ESCALATION.md` (resolution, 2026-10-17).

## Hypothesis

At matched accuracy, the paper's metric privacy leaks no less than global DP; it is either equivalent or worse. Its
advantage in v1 rested on paired, shared-noise measurements. WP3's private-median metric privacy, which carries a
formal guarantee, sits between the two.

## Design

- **Federation and audit regime.** As M2: randomized inclusion with 32 candidates, 16 included per replicate,
  EuroSAT alpha 0.3, seed-42 layout, 50 rounds, equal weights, WP2 hyperparameters. `R = 32` per configuration,
  `design_seed = 2`. Replicate `j` therefore has the same inclusion and the same `(client_seed, noise_seed)` as M2's
  replicate `j`, which pairs A1 with M2 across configurations; within a configuration no two runs share private
  randomness.
- **Baseline.** M2's global-DP runs are the comparison curve. They run under the same audit regime, as rule 6 of
  ROADMAP.md Section 6 requires.
- **Mechanisms.**
  - The paper's metric privacy (`metric-privacy`; noise calibrated by the maximum pairwise client distance).
  - WP3's private-median metric privacy (`PrivateMedianMetricPrivacyServerSideFixedClipping`, WP3's settings).
    The runner builds it inside the worker, the way WP3's smoke run did, so `strategy_factory.py` stays unchanged.
- **Noise levels.** Three per mechanism, chosen to bracket the mean accuracies of M2's operating points (`r_0.6` and
  the next-lower noise, or the two noise levels around 0.70 accuracy if M2 finds no useful regime).
  - **Initial levels:** from an accuracy calibration for both mechanisms, since v1 has the paper's mechanism at one
    noise level only (100 rounds, 48 clients, not this regime). Both add noise `r C / d` with `d` a client distance;
    in v1 on EuroSAT that was 1.8-2.6 times global DP's noise at the same `r`, so the accuracy at a given `r` is
    not known in advance.
  - **Calibration (accuracy only).** Both mechanisms at `r` = 0.25, 0.5, 1 and 2 times 0.001546, 2 replicates each
    (one complementary pair, `design_seed = 3`, distinct from the main design), 16 runs, about 2 GPU-h. Output in
    `a1/calibration/`. Only final accuracy is read; no leakage statistic is computed and these runs are never
    part of the analysis. The three levels per mechanism are then chosen so that both targets lie inside their
    accuracy range, and fixed in the protocol.
  - **Replacement rule** (accuracy only): if after stage 1 the three levels do not bracket both target accuracies,
    the outermost level on the wrong side is replaced once, and the replacement runs both stages.
- **Two stages**, as M2: replicates 0-15, then 16-31, always run. No leakage statistic before both stages are complete.
- **Cost.** 7 configurations x 32 runs = 224 runs (private median has four levels, PROTOCOL.md Section 2), about
  25 GPU-h at M2's measured 0.11 GPU-h per run, within ROADMAP.md's 30 GPU-h cap. The earlier proposal of a 48 GPU-h
  cap is withdrawn.

## Attacker and metrics

- **Calibrated loss attacker** on the 11-checkpoint shadow-loss trajectory (`calibrated_scores`), as M2; all
  candidates and the top decile.
- **Matched-accuracy difference.** For each mechanism, AUC against mean final accuracy, interpolated linearly between
  its levels. The statistic is `AUC_mechanism(a*) - AUC_GDP(a*)` at each target accuracy `a*`, where GDP's AUC is
  interpolated the same way along M2's curve. Both curves are recomputed inside every resample.
- **Intervals.**
  - Gate interval: the pair jackknife at 97.5%. All configurations share the design, so a jackknife replicate leaves
    out the same pair in every configuration.
  - The two-way bootstrap over runs (ROADMAP.md's interval for A1) is reported alongside, and the outcome is stated
    under both.
- **Also reported:** accuracy with a bootstrap over replicates, TPR at 5% and 1% FPR, and the noise-level channel.
- **Formal guarantee.**
  - Global DP: `mu_total = sqrt(50) / (32 r)`.
  - Private median: its accountant (`PrivateMedianMetricPrivacyServerSideFixedClipping.account`).
  - The paper's mechanism has none.
- **Power** (`a1/power.py`, from M2's runs). Two disjoint global-DP curves with A1's structure (vanilla, 2x, 8x
  against 1x, 4x, 16x) give the matched-accuracy difference's jackknife SE. Detectable difference at 80% power for
  the 97.5% gate interval:

  | Subset | `a* = 0.740` | `a* = 0.660` |
  |---|---|---|
  | All candidates | 0.027 (SE 0.008) | 0.037 (SE 0.011) |
  | Top decile | 0.091 (SE 0.027) | 0.096 (SE 0.029) |

  Pairing across configurations (equal replicate seeds) makes the difference much more precise than one
  configuration (SE about 0.023). Metric privacy uses the same noise generator, so its runs should pair as well.
  Private median also draws its bit noise from that generator, which shifts the model-noise stream after round 1,
  so its pairing with global DP is weaker and its detectable difference larger.

## Gate (copied verbatim from ROADMAP.md Section 3, A1)

**G-A1** (report, no kill), one of three outcomes:

- Metric privacy is better than global DP at matched accuracy (interval excludes 0).
- Equivalent: the interval lies within ±0.03 AUC.
- Worse, or inconclusive.

**Reachability.** The power estimate above revises the earlier expectation (ESCALATION.md, open item) that ±0.03
was close to unreachable: for all candidates the paired difference's gate half-width is about 0.02-0.03, so
"equivalent" is reachable when the true difference is near 0. For the top decile it is not. An interval inside
±0.10 that includes 0 is described as "no detectable difference at this power", not as equivalence.

## Files

| File | Change |
|---|---|
| `results/cia_v2/randomized_inclusion/runner.py` | A `private-median` privacy mode, built in the worker (no change to `strategy_factory.py`). Made on `feature/cia-v2-measurement`, the runner's home, and merged into this branch only between M2 stages, so running M2 workers never see a changed runner. |
| `results/cia_v2/randomized_inclusion/tests/` | Tests of the new mode |
| `experiments/reproduce/runner.py`, `server.py` | `--privacy private-median` with `--num-slots` and `--median-*` flags, built in `server.private_median_strategy` like WP6's residual clipping (Agent A's server and runner wiring; `strategy_factory.py` unchanged) |
| `tests/test_private_median_run.py` | Runner and strategy tests of the mode |
| `results/cia_v2/frontier/a1/analyze.py` | Matched-accuracy interpolation, intervals, G-A1 |
| `results/cia_v2/frontier/a1/power.py` | Detectable difference from M2's runs |
| `results/cia_v2/frontier/a1/tests/` | Synthetic tests of the analysis |
| `results/cia_v2/frontier/a1/{PROTOCOL,STATE,FINDINGS}.md`, `analysis.json`, `results/` | This package |

## Log

| When | What |
|---|---|
| 2026-10-17 | STATE written before any A1 code, while M2's stage 1 runs. |
| 2026-10-17 | Private-median mode on `feature/cia-v2-measurement` (4119845e): runner flags, `server.private_median_strategy`, the randomized-inclusion runner's settings `q_0 = 0.3` in `[0.03, 1.0]` with the 32 active clients as public slots (v1's EuroSAT pairwise client distances: median 0.39-0.53). 10 new tests; full suite 734 passed. A 2-round CPU Flower smoke run worked end to end: the quantile tracked the true median distance (0.25, then 0.23). M2 manifests are byte-identical under the new code. |
| 2026-10-19 20:35 | M2 done: no useful regime. A1 targets: GDP 2x (accuracy 0.740) and 4x (0.660). v1 has metric privacy at one noise level only, so the initial levels come from an accuracy calibration for both mechanisms (16 runs, design seed 3), written here before it runs. |
| 2026-10-19 20:40 | Calibration running (4 workers, both GPUs idle and unclaimed at launch). `a1/analyze.py` written with 12 synthetic tests (matched-accuracy difference, jackknife and two-way bootstrap, G-A1 outcomes, no extrapolation). `a1/power.py`: detectable difference 0.027 / 0.037 (all candidates) and 0.091 / 0.096 (top decile) at the two targets. |
| 2026-10-19 21:19 | Calibration (16 runs, accuracy only): metric privacy 0.802 / 0.797 / 0.769 / 0.711 and private median 0.803 / 0.768 / 0.708 / 0.680 at 0.25x / 0.5x / 1x / 2x. Both bracket 0.740; neither reaches 0.660, so 4x is added to the calibration (4 runs, same design seed 3), accuracy only, before the protocol. |
| 2026-10-19 21:40 | 4x calibration: metric privacy 0.636, private median 0.506. Levels fixed in PROTOCOL.md: metric privacy 1x, 2x, 4x; private median 0.5x, 1x, 2x, 4x (four, because its calibration runs differ by up to 0.07 at one level). A1's replicates have M2's seeds and inclusion (checked for replicates 0 and 17). Calibration runs committed (accuracy only, never analyzed). Main runs started on both GPUs (idle and unclaimed). |
