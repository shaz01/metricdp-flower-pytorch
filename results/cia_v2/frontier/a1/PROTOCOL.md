# A1 protocol: metric privacy against global DP at matched accuracy

Package A1 of [research/cia_roadmap_v2/ROADMAP.md](../../../../research/cia_roadmap_v2/ROADMAP.md) (Agent A, branch
`feature/cia-v2-frontier`). Fixed 2026-10-19, after the accuracy calibration and before any A1 leakage run.
Hypothesis, design rationale, statistic and gate are in [STATE.md](STATE.md). M2 found no useful regime
([../FINDINGS.md](../FINDINGS.md)), so the targets are M2's two levels around 0.70 accuracy.

## 1. Runs

| Item | Value |
|---|---|
| Runner | `results/cia_v2/randomized_inclusion/runner.py`, as M2 (32 candidates, 16 included per replicate, 32 active of 48 slots), without surrogates |
| Data | EuroSAT, label-Dirichlet alpha 0.3, layout seed 42; FedAvg with equal weights; WP2 hyperparameters; 50 rounds |
| Design | `R = 32`, `design_seed = 2`: M2's design. Replicate `j` has M2's inclusion and `(client_seed, noise_seed)` in every configuration (checked for replicates 0 and 17) |
| Metric privacy | `a1-mp-r1x` (`r = 0.001546`), `a1-mp-r2x` (0.003092), `a1-mp-r4x` (0.006184); `--privacy metric-privacy` |
| Private median | `a1-pm-r0.5x` (0.000773), `a1-pm-r1x` (0.001546), `a1-pm-r2x` (0.003092), `a1-pm-r4x` (0.006184); `--privacy private-median`, `q_0 = 0.3` in `[0.03, 1.0]`, 32 public slots |
| Global DP | M2's runs: `gdp-r1x`, `gdp-r2x`, `gdp-r4x`, `gdp-r8x`, `gdp-r16x` in `results/cia_v2/frontier/results/` |
| Output | `results/cia_v2/frontier/a1/results/`. Committed: run JSON, manifest, measurements, provenance, complete. Kept local: `*.evaluation.json`, `*.predictions.npz` |

Commands, one per configuration and stage. Two workers per GPU run the same command; trajectories are locked and
resumable.

```bash
python -m results.cia_v2.randomized_inclusion.runner --config a1-mp-r1x --privacy metric-privacy \
    --noise-ratio 0.001546 --replicates 32 --design-seed 2 --replicate-subset 0 1 ... 15 --execute \
    --max-parallel-clients 8 --output results/cia_v2/frontier/a1/results
```

## 2. How the levels were chosen

Accuracy calibration, `design_seed = 3` (never analyzed), 2 runs per level, mean final accuracy:

| Multiple of 0.001546 | 0.25x | 0.5x | 1x | 2x | 4x |
|---|---|---|---|---|---|
| Metric privacy | 0.802 | 0.797 | 0.769 | 0.711 | 0.636 |
| Private median | 0.803 | 0.768 | 0.708 | 0.680 | 0.506 |
| Global DP (M2, `R = 32`) | | | 0.786 | 0.740 | 0.660 |

- **Metric privacy:** 1x and 2x bracket 0.740, 2x and 4x bracket 0.660.
- **Private median:** four levels, because its two calibration runs differ by up to 0.07 at one level (0.744 and
  0.673 at 1x). With 0.5x, 1x, 2x and 4x both targets are bracketed wherever 1x lands.

## 3. Order and replacement

1. **Stage 1:** replicates 0-15 of every configuration.
2. **Replacement rule** (accuracy only): if a mechanism's stage-1 mean accuracies do not bracket both targets
   (0.740 and 0.660), its outermost level on the wrong side is replaced once by the next multiple outward in the
   series 0.25x, 0.5x, 1x, 2x, 4x, 8x. The replacement runs both stages.
3. **Stage 2:** replicates 16-31 of every configuration. It always runs.
4. **Analysis**, after stage 2 of every configuration and no earlier:

   ```bash
   python -m results.cia_v2.frontier.a1.analyze \
       --gdp gdp-r1x=0.001546 gdp-r2x=0.003092 gdp-r4x=0.006184 gdp-r8x=0.012368 gdp-r16x=0.024736 \
       --metric-privacy a1-mp-r1x=0.001546 a1-mp-r2x=0.003092 a1-mp-r4x=0.006184 \
       --private-median a1-pm-r0.5x=0.000773 a1-pm-r1x=0.001546 a1-pm-r2x=0.003092 a1-pm-r4x=0.006184 \
       --out results/cia_v2/frontier/a1/analysis.json
   ```

## 4. Analysis and gate

As in STATE.md (Attacker and metrics; Gate), implemented in `a1/analyze.py`:

- Targets `a* = 0.740` and `0.660`; per mechanism and target, `AUC_mechanism(a*) - AUC_GDP(a*)`, each curve linear
  in mean accuracy between its levels, no extrapolation.
- Gate interval: pair jackknife at 97.5%. Two-way bootstrap (2,000 resamples) alongside; the outcome is stated
  under both.
- G-A1 per mechanism and target, on all candidates: better (interval below 0), equivalent (inside ±0.03),
  otherwise worse or inconclusive. An interval inside ±0.10 that contains 0 is "no detectable difference at this
  power". The top decile is reported with the same labels but not gated.
- Per level: AUC, TPR at 5% and 1% FPR, final-round and threshold attackers, accuracy, lowest client accuracy, and
  formal `mu` (global DP: `sqrt(50) / (32 r)`; private median: its accountant; metric privacy: none).

Power (`a1/power.py`): detectable difference at 80% power 0.027 and 0.037 (all candidates) and 0.091 and 0.096
(top decile) at the two targets, for curves paired like global DP's. Private median pairs less tightly.

## 5. Cost

M2 measured about 0.11 GPU-h per run with two workers per GPU, including surrogate scoring, which A1 skips.
224 runs come to about 25 GPU-h, plus about 2.3 for the calibration's 20 runs; a replacement level adds about 3.5.
The cap is ROADMAP.md's 30 GPU-h. If measured cost reaches 1.25 x 30 = 37.5 GPU-h, the package stops and
escalates.

## 6. Machine

Compute server. Runs use a GPU only when it is idle and unclaimed, or with the owner's approval to share.
