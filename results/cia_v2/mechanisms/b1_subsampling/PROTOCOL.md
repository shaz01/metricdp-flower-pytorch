# B1 protocol: client subsampling against full-participation global DP at matched accuracy

Package B1 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9, 10 and 13), branch
`feature/cia-v2-mechanisms`. Fixed 2026-10-27, after the accuracy pilot and before any B1 leakage run. Design in
[STATE.md](STATE.md).

## 1. Runs

| Item | Value |
|---|---|
| Runner | `results/cia_v2/randomized_inclusion/runner.py --privacy subsampled-global-dp --sampling-rate q`: M2's design (32 candidates, 16 IN per replicate, 32 active of 48), EuroSAT alpha 0.3, layout seed 42, 50 rounds, equal weights, no surrogates |
| Mechanism | each round every active client joins with probability `q` (generator seeded by the noise seed); release `w + sum_{j in S} clip(u_j, C) / (q N) + N(0, (r C)^2)` with `N = 32`; noise multiplier `z = r q N` |
| Design | `R = 32`, `design_seed = 2`: replicate `j` has M2's inclusion and `(client_seed, noise_seed)` |
| `q = 0.5` | `b1-q0.5-r0.001546`, `b1-q0.5-r0.003092`, `b1-q0.5-r0.006184`, `b1-q0.5-r0.012368` |
| `q = 0.25` | `b1-q0.25-r0.0001`, `b1-q0.25-r0.000773`, `b1-q0.25-r0.003092`, `b1-q0.25-r0.006184` |
| Safety | `--require-all-clients` stops a run at the first round with a failed or missing sampled client; the run check refuses any run with a short round (collapsed rounds accepted) |
| Output | `results/cia_v2/mechanisms/b1_subsampling/results/`. Committed: run JSON, manifest, measurements, provenance, complete. Kept local: `*.evaluation.json`, `*.predictions.npz` |

## 2. How the levels were chosen

Accuracy pilot (`pilot/`, `design_seed = 3`, 2 runs per level, never analyzed), mean final accuracy:

| `r` | 0.0001 | 0.000387 | 0.000773 | 0.001546 | 0.003092 | 0.006184 |
|---|---|---|---|---|---|---|
| `q = 0.5` | | | 0.779 | 0.779 | 0.744 | 0.693 |
| `q = 0.25` | 0.769 | 0.740 | 0.727 | 0.702 | 0.660 | |

Targets 0.740 and 0.660 (M2's accuracy at 2x and 4x). Four levels per `q`, because the pilot's levels nearest the targets
sit within 0.005 of them: `q = 0.5` adds 0.012368 (not piloted) to get below 0.660; `q = 0.25` adds 0.006184 (not
piloted) for the same reason and keeps 0.0001 above 0.740.

**Replacement rule** (accuracy only, applied automatically after stage 1): if a `q`'s stage-1 accuracies do not bracket
both targets, the chain stops before stage 2, and its outermost level on the wrong side is replaced once by the next
level outward in the series 0.0001, 0.000387, 0.000773, 0.001546, 0.003092, 0.006184, 0.012368, 0.024736.

## 3. Order and analysis

Stage 1 (replicates 0-15) of the eight configurations, the replacement check, stage 2 (replicates 16-31), then:

```bash
python -m results.cia_v2.mechanisms.b1_subsampling.analyze \
    --gdp gdp-r1x=0.001546 gdp-r2x=0.003092 gdp-r4x=0.006184 gdp-r8x=0.012368 gdp-r16x=0.024736 \
    --rate 0.5 b1-q0.5-r0.001546=0.001546 b1-q0.5-r0.003092=0.003092 b1-q0.5-r0.006184=0.006184 b1-q0.5-r0.012368=0.012368 \
    --rate 0.25 b1-q0.25-r0.0001=0.0001 b1-q0.25-r0.000773=0.000773 b1-q0.25-r0.003092=0.003092 b1-q0.25-r0.006184=0.006184 \
    --out results/cia_v2/mechanisms/b1_subsampling/analysis.json
```

## 4. Gate

**G-B1** per `q` and target, all candidates: `AUC_B1(a*) - AUC_GDP(a*)` at least 0.05 below 0, with the 97.5% pair-jackknife
interval below 0 (decision 9). The two-way bootstrap is reported alongside, and the outcome is stated under both. The
amplified `mu` (accountant, PLD discretization 0.1) is reported where it computes, not gated (decision 13). The top
decile is reported, not gated.

## 5. Cost

About 0.06 GPU-h per run at `q = 0.5` and 0.03 at `q = 0.25`: 256 runs, about 11.5 GPU-h, within the 15 GPU-h cap (stop
at 18.75).

## 6. Machine

Compute server, GPU 1 (idle and unclaimed at launch); GPU 0 is left to other users.
