# AZ protocol, part 1: M2 on Alzheimer MRI

Owner decision 8 (`research/cia_roadmap_v2/ROADMAP.md` Section 10), with the owner's choices of 2026-10-21 (setup:
alpha 1.0, 100 rounds; ESCALATION.md) and 2026-10-22 (M2 on Alzheimer first, within the 40 GPU-h cap; A1 on Alzheimer
decided afterwards). Branch `feature/cia-v2-alzheimer`. Fixed 2026-10-22, after the accuracy calibration and before
any leakage run. Design and rationale in [STATE.md](STATE.md).

## 1. Runs

| Item | Value |
|---|---|
| Runner | `results/cia_v2/randomized_inclusion/runner.py --dataset alzheimer`: 32 candidates (top decile 4, 29, 31, 22, 15), 16 IN per replicate, 32 active of 48 slots |
| Data and model | Alzheimer MRI, label-Dirichlet alpha 1.0, layout seed 42, at least 10 records per client; `paper_cnn`, v1's hyperparameters; 100 rounds; FedAvg with equal weights |
| Scoring | Shadow samples (25% of each candidate's training records, no augmentation) at rounds 0, 10, ..., 100; no surrogates |
| Design | `R = 32`, `design_seed = 2`, the same for every configuration |
| Configurations | `az-vanilla`; global DP `az-gdp-r0.0025`, `az-gdp-r0.003`, `az-gdp-r0.0035` (noise std `r C`, noise multiplier `32 r`) |
| Output | `results/cia_v2/alzheimer/results/`. Committed: run JSON, manifest, measurements, provenance, complete. Kept local: `*.evaluation.json`, `*.predictions.npz` |

```bash
python -m results.cia_v2.randomized_inclusion.runner --dataset alzheimer --config az-gdp-r0.003 \
    --privacy global-dp --noise-ratio 0.003 --replicates 32 --design-seed 2 --replicate-subset 0 1 ... 15 \
    --execute --max-parallel-clients 8 --output results/cia_v2/alzheimer/results
```

## 2. Why these levels

Calibration (`calibration_a1/`, accuracy only, `design_seed = 3`, 2 runs per level, never analyzed), global DP mean
final accuracy: 0.807 (r = 0.000625), 0.825 (0.00125), 0.782 (0.0025), 0.731 (0.003), 0.688 (0.0035), 0.627 (0.004),
0.483 (0.005); vanilla 0.805. Accuracy is flat up to 0.0025 and collapses by 0.005, so the three levels span the band
from near vanilla to below 0.70, where G-M2 is decided. Levels with no accuracy effect (0.000625, 0.00125) add nothing
to G-M2 and are left out for cost. No fill-in rule: the calibration already resolves the curve.

## 3. Order

1. Stage 1: replicates 0-15 of the four configurations.
2. Stage 2: replicates 16-31, always.
3. Analysis after stage 2 of every configuration, and no earlier:

   ```bash
   python -m results.cia_v2.frontier.analyze results/cia_v2/alzheimer/results \
       --configs az-vanilla=0 az-gdp-r0.0025=0.0025 az-gdp-r0.003=0.003 az-gdp-r0.0035=0.0035 \
       --out results/cia_v2/alzheimer/analysis.json
   ```

   `noise_multiple` in the output is relative to EuroSAT's frontier ratio and carries no meaning here. Formal
   `mu_total = sqrt(100) / (32 r)`.

## 4. Analysis and gate

As M2 (`results/cia_v2/frontier/PROTOCOL.md` Section 4): calibrated attacker, all candidates and top decile, pair
jackknife at 97.5% with the two-way bootstrap alongside, accuracy with a bootstrap over replicates. **G-M2 as applied:**
a level is a useful regime when the all-candidate AUC's upper gate end is at most 0.6, the top-decile point estimate
is at most 0.6, and the lower end of mean accuracy is at least 0.70.

## 5. Cost

The calibration measured about 0.2 GPU-h per run (median 29 minutes of training per worker, two or three workers on a
saturated GPU). 128 runs come to about 24 GPU-h; with about 10 GPU-h spent on the diagnosis and calibration, the
total stays within the 40 GPU-h cap. At 1.25 x 40 = 50 GPU-h the package stops and escalates.

## 6. Machine

Compute server. Runs use a GPU only when it is idle and unclaimed, or with the owner's approval to share.

# AZ protocol, part 2: A1 on Alzheimer MRI

Fixed 2026-10-23, after part 1's analysis (G-M2: useful regime at `r_0.6 = 0.0025`) and before any part-2 run. The
owner chose to run it after seeing that result (vanilla leakage near chance, AUC 0.564).

## 1. Targets and global-DP curve

- A1's rule for a useful regime is `r_0.6` and the next-lower noise. The grid has no global-DP level below 0.0025, so
  the targets are the accuracies of `r_0.6` and of the next level, 0.003, which also meets G-M2: **a* = 0.784 and
  0.732** (part 1's mean final accuracies, 0.7843 and 0.7325).
- **Global-DP curve:** part 1's vanilla (the zero-noise end, accuracy 0.816) and global DP at 0.0025, 0.003 and 0.0035.
  Including vanilla keeps 0.784 inside the curve's range in every resample.

## 2. Runs

As part 1 (Section 1), `design_seed = 2`, so replicate `j` has part 1's inclusion and seeds. Output
`results/cia_v2/alzheimer/a1/results/`.

| Mechanism | Configurations (`r`) | Calibration accuracy (2 runs) |
|---|---|---|
| Metric privacy (`--privacy metric-privacy`) | `az-mp-r0.00125`, `az-mp-r0.0025`, `az-mp-r0.003` | 0.834, 0.774, 0.716 |
| Private median (`--privacy private-median`, `q_0 = 0.4` in `[0.04, 1.3]`) | `az-pm-r0.000625`, `az-pm-r0.00125`, `az-pm-r0.0016` | 0.826, 0.719, not calibrated (expected below 0.70) |

Private median's third level is uncalibrated: its calibration jumped from 0.719 (0.00125) to 0.427 (0.0025), and 0.0016
is chosen to land safely below 0.732 without collapsing. A collapsed round keeps the previous model (guard added
2026-10-23); `--require-all-clients` stops a run if a client fails.

**Replacement rule** (accuracy only, after stage 1): if a mechanism's stage-1 accuracies do not bracket both targets,
its outermost level on the wrong side is replaced once by the next level outward in that mechanism's calibration
series (metric privacy: 0.000625, 0.00125, 0.0025, 0.003, 0.0035; private median: 0.000625, 0.00125, 0.0016, 0.0025).

## 3. Order and analysis

Stage 1 (replicates 0-15) of the six configurations, the replacement check on stage-1 accuracy, stage 2 (16-31), then:

```bash
python -m results.cia_v2.frontier.a1.analyze --dataset alzheimer \
    --gdp-root results/cia_v2/alzheimer/results --root results/cia_v2/alzheimer/a1/results \
    --gdp az-vanilla=0 az-gdp-r0.0025=0.0025 az-gdp-r0.003=0.003 az-gdp-r0.0035=0.0035 \
    --metric-privacy az-mp-r0.00125=0.00125 az-mp-r0.0025=0.0025 az-mp-r0.003=0.003 \
    --private-median az-pm-r0.000625=0.000625 az-pm-r0.00125=0.00125 az-pm-r0.0016=0.0016 \
    --targets 0.784 0.732 --out results/cia_v2/alzheimer/a1/analysis.json
```

G-A1 as on EuroSAT (better / equivalent within ±0.03 / worse or inconclusive, and "no detectable difference at this
power" inside ±0.10). Formal `mu` uses 100 rounds and private median's maximum 1.3.

**Expected:** with leakage near chance for every configuration, A1 here can show at most that the mechanisms do not
differ; it cannot show that one protects better.

## 4. Cost

192 runs. Part 1's relaunch ran about 3.9 runs per hour with two workers (0.26 GPU-h per run): about 50 GPU-h and
about 35 hours with three workers on one GPU. The owner decided to run it on 2026-10-23 with the overrun known
(part 1 alone reached about 44 GPU-h of the 40 GPU-h cap).
