# C1 protocol: the composite (subsampling + logit adjustment) against metric privacy at matched accuracy

Package C1 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9, 13, 14, 15), branch
`feature/cia-v2-composite`. Fixed 2026-10-29, after the accuracy pilot and before any C1 leakage run. Design and
owner decisions in [STATE.md](STATE.md).

## 1. Runs

| Item | Value |
|---|---|
| Runner | `results/cia_v2/randomized_inclusion/runner.py --privacy subsampled-global-dp --sampling-rate 0.25 --logit-adjustment-tau 1.0 --surrogates`: M2's design (32 candidates, 16 IN per replicate, 32 active of 48), EuroSAT alpha 0.3, layout seed 42, 50 rounds, equal weights |
| Composite | B1's Poisson-sampled global DP at `q = 0.25` (denominator `q N`, noise `r C` on the average, `z = r q N`) with B5's logit-adjusted local loss (`tau = 1`) |
| Design | `R = 32`, `design_seed = 2`: replicate `j` has M2's inclusion and seeds, as A1's and B1's runs |
| Configurations | `c1-r0.000773`, `c1-r0.003092`, `c1-r0.006184`, `c1-r0.012368` |
| Scoring | shadow samples (gated attacker); label-matched public surrogates (adaptive attacker a); the logit-adjusted shadow loss with the candidate's label prior (adaptive attacker b) |
| Safety | `--require-all-clients` stops a run at the first failed or missing sampled client; the run check refuses short rounds; 4 parallel clients per worker (GPU 1 is shared) |
| Output | `results/cia_v2/composite/results/`. Committed: run JSON, manifest, measurements, provenance, complete. Kept local: `*.evaluation.json`, `*.predictions.npz` |

## 2. How the levels were chosen

Accuracy pilot (`pilot/`, `design_seed = 3`, 2 runs per level, never analyzed), mean final accuracy:

| `r` | 0.0001 | 0.000773 | 0.003092 | 0.006184 | 0.012368 | 0.024736 |
|---|---|---|---|---|---|---|
| Composite | 0.776 | 0.780 | 0.713 | 0.701 | 0.594 | 0.228 |
| B1, `q = 0.25` alone (main runs) | 0.746 | 0.736 | 0.698 | 0.602 | | |

The four levels bracket 0.740 (0.780 and 0.713) and 0.660 (0.701 and 0.594). **Replacement rule** (accuracy only,
applied automatically after stage 1): if the stage-1 accuracies do not bracket both targets, the chain stops before
stage 2, and the outermost level on the wrong side is replaced once by the next pilot level outward.

## 3. Order and analysis

Stage 1 (replicates 0-15), the replacement check, stage 2 (replicates 16-31), then:

```bash
python -m results.cia_v2.composite.analyze \
    --composite c1-r0.000773=0.000773 c1-r0.003092=0.003092 c1-r0.006184=0.006184 c1-r0.012368=0.012368 \
    --metric-privacy a1-mp-r1x=0.001546 a1-mp-r2x=0.003092 a1-mp-r4x=0.006184 \
    --private-median a1-pm-r0.5x=0.000773 a1-pm-r1x=0.001546 a1-pm-r2x=0.003092 a1-pm-r4x=0.006184 \
    --gdp gdp-r1x=0.001546 gdp-r2x=0.003092 gdp-r4x=0.006184 gdp-r8x=0.012368 gdp-r16x=0.024736 \
    --subsampled b1-q0.25-r0.0001=0.0001 b1-q0.25-r0.000773=0.000773 b1-q0.25-r0.003092=0.003092 b1-q0.25-r0.006184=0.006184 \
    --out results/cia_v2/composite/analysis.json
```

(`--a1-root`, `--gdp-root` and `--b1-root` point at the committed A1, M2 and B1 runs on master.)

## 4. Gate

**G-C1** (decision 5 as amended by decisions 9, 13 and 15), EuroSAT, per target `a*` in {0.740, 0.660}: the composite's
all-candidate AUC is at least 0.05 below the paper's metric privacy at matched accuracy, with the 97.5% pair-jackknife
interval of the difference below 0. The two-way bootstrap is reported alongside and the outcome stated under both. The
amplified formal `mu` is reported where computable, not gated. Reported, not gated: the composite against private
median, global DP and subsampling alone; the top decile; both adaptive attackers.

## 5. Cost and machine

128 runs on GPU 1 (shared with another user's job by owner approval), two workers with 4 parallel clients each, about
18 hours. **Cap 15 GPU-h (owner, 2026-10-29); stop and escalate at 18.75.** GPU 0 is left to other users.
