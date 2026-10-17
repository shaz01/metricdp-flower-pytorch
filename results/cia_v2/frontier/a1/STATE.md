package: A1
status: planning (blocked by M2: noise levels need M2's operating points)
gate: G-A1 = not evaluated yet (report, no kill)
protocol: to be fixed in results/cia_v2/frontier/a1/PROTOCOL.md after M2's stage 1, before any A1 run
machine: compute server
cost so far: 0 GPU-h; proposed cap 48 GPU-h (192 runs at the measured 0.20 GPU-h, plus one replacement level), stop at 60
next: private-median hook in the randomized-inclusion runner (Agent A's file; no change to strategy_factory.py), with tests
needs from others: none (M2's runs serve as the global-DP curve)

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
  - **Initial levels:** from v1's accuracy curves for the paper's mechanism (accuracy does not depend on the noise
    coupling) and from a two-replicate accuracy check for private median. They are fixed in the protocol.
  - **Replacement rule** (accuracy only): if after stage 1 the three levels do not bracket both target accuracies,
    the outermost level on the wrong side is replaced once, and the replacement runs both stages.
- **Two stages**, as M2: replicates 0-15, then 16-31, always run. No leakage statistic before both stages are complete.
- **Cost.** 6 configurations x 32 runs = 192 runs, about 38 GPU-h at the M0 cost run's 0.20 GPU-h per run;
  re-measured after the first runs (private median computes per-client distances each round).

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
- **Power.** The M0 power table gives an MDE of 0.087 for a difference of two configurations at `R = 32`.
  Interpolation adds variance, so the matched-accuracy MDE is at least that. It will be simulated for the protocol.

## Gate (copied verbatim from ROADMAP.md Section 3, A1)

**G-A1** (report, no kill), one of three outcomes:

- Metric privacy is better than global DP at matched accuracy (interval excludes 0).
- Equivalent: the interval lies within ±0.03 AUC.
- Worse, or inconclusive.

**Reachability.** At `R = 32` a single configuration's gate interval has a half-width of about 0.05, and a
matched-accuracy difference's is larger. "Equivalent" within ±0.03 is therefore close to unreachable
(ESCALATION.md, open item). The protocol keeps the band and reports the interval. An interval inside ±0.10 that
includes 0 is described as "no detectable difference at this power", not as equivalence.

## Files

| File | Change |
|---|---|
| `results/cia_v2/randomized_inclusion/runner.py` | A `private-median` privacy mode, built in the worker (no change to `strategy_factory.py`). Made on `feature/cia-v2-measurement`, the runner's home, and merged into this branch only between M2 stages, so running M2 workers never see a changed runner. |
| `results/cia_v2/randomized_inclusion/tests/` | Tests of the new mode |
| `results/cia_v2/frontier/a1/analyze.py` | Matched-accuracy interpolation, intervals, G-A1 |
| `results/cia_v2/frontier/a1/tests/` | Synthetic tests of the analysis |
| `results/cia_v2/frontier/a1/{PROTOCOL,STATE,FINDINGS}.md`, `analysis.json`, `results/` | This package |

## Log

| When | What |
|---|---|
| 2026-10-17 | STATE written before any A1 code, while M2's stage 1 runs. |
