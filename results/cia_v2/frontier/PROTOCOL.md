# M2 protocol: corrected leakage-accuracy curves (global DP, vanilla)

Package M2 of [research/cia_roadmap_v2/ROADMAP.md](../../../research/cia_roadmap_v2/ROADMAP.md) (Agent A, branch
`feature/cia-v2-frontier`). Fixed before any M2 run. It runs only after G-M1 returns "confirmed" or "partial"
(ROADMAP.md Section 4). The measurement design is the standard as amended by
`results/cia_v2/measurement/ESCALATION.md` (resolution of 2026-10-17). Hypothesis, rationale and gate are in
[STATE.md](STATE.md).

## 1. Runs

| Item | Value |
|---|---|
| Runner | `results/cia_v2/randomized_inclusion/runner.py` (32 candidates, 16 included per replicate, 32 active of 48 slots) |
| Data | EuroSAT, label-Dirichlet alpha 0.3, layout seed 42; FedAvg with equal weights; WP2 hyperparameters; 50 rounds |
| Design | `R = 32`, `design_seed = 2`, the same for every configuration |
| Configurations | `vanilla`; `gdp-r1x` (`r = 0.001546`), `gdp-r4x` (0.006184), `gdp-r16x` (0.024736); fill-in `gdp-r2x` (0.003092) and `gdp-r8x` (0.012368) by the rule below |
| Scoring | Shadow samples (no augmentation) and label-matched surrogates of all 32 candidates at rounds 0, 5, ..., 50 |
| Output | `results/cia_v2/frontier/results/<config>/`. Committed: run JSON, manifest, measurements, provenance, complete. Kept local: `*.evaluation.json` (3.5 MB per run), `*.predictions.npz` |

Commands, one per configuration and stage. Two workers per GPU run the same command; trajectories are locked and
resumable.

```bash
python -m results.cia_v2.randomized_inclusion.runner --config gdp-r1x --privacy global-dp --noise-ratio 0.001546 \
    --replicates 32 --design-seed 2 --replicate-subset 0 1 ... 15 --surrogates --execute \
    --max-parallel-clients 8 --output results/cia_v2/frontier/results
python -m results.cia_v2.randomized_inclusion.runner --config vanilla --privacy vanilla \
    --replicates 32 --design-seed 2 --replicate-subset 0 1 ... 15 --surrogates --execute ...
```

## 2. Order and fill-in

1. **Stage 1:** replicates 0-15 of vanilla, 1x, 4x and 16x.
2. **Fill-in rule** (accuracy only, fixed now):
   - Add 2x if the stage-1 mean final accuracy at 1x exceeds that at 4x by more than 0.05.
   - Add 8x on the same rule for 4x vs 16x.
   - A fill-in level runs both stages.
3. **Stage 2:** replicates 16-31 of every configuration. It always runs.
4. **Analysis:** `python -m results.cia_v2.frontier.analyze --configs vanilla=0 gdp-r1x=0.001546 ...`. No leakage
   statistic is computed before stage 2 of every configuration is complete.

## 3. Cost

The M0 cost run measured 0.20 GPU-h per run (two workers per GPU). The base grid is 128 runs, about 26 GPU-h; each
fill-in level adds 32 runs, about 6.4 GPU-h. The cap is 64 GPU-h. If measured cost reaches 1.25 x 64 = 80 GPU-h,
the package stops and escalates.

## 4. Analysis and gate

As in STATE.md (Attacker and metrics; Gate):

- Primary calibrated attacker, gate interval `pair_jackknife` at 97.5%, two-way bootstrap alongside.
- All candidates and top decile separately; accuracy with a bootstrap over replicates; formal `mu`; `r_0.6`.
- G-M2 as applied by the ESCALATION.md resolution, item 4.

Power (M0 power table, `R = 32`, 32 candidates): detectable AUC difference 0.087 (all candidates) and 0.196 (top
decile) at 80% power. A single configuration's gate interval has a half-width of about 0.05.

**Consequences, ROADMAP.md Section 4:**
- **Useful regime:** A1, B1, B3 and B5 use `r_0.6` and the next-lower noise; B2 opens.
- **No useful regime:** escalate with the recommendation that B4 lead; A1 still runs.

## 5. Not part of M2

The surrogate scores are recorded for B5's diagnostic, which is Agent B's analysis. M2 reports only the
surrogate-attacker AUC next to the true one, without a gate.
