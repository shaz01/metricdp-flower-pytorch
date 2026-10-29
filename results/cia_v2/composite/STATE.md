package: C1, composite candidate against metric privacy (opened by the owner 2026-10-29, decision 14)
status: planning (STATE before code)
gate: G-C1 (as decided 2026-10-29, decision 15) = not evaluated yet
protocol: results/cia_v2/composite/PROTOCOL.md, fixed after the accuracy pilot and before any leakage run
machine: compute server
cost so far: 0 GPU-h; estimate about 6 GPU-h (cap set by the owner when the protocol is fixed)
next: adaptive scoring (logit-adjusted shadow losses) and the C1 analysis with tests; accuracy pilot
needs from others: none

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, C1, as amended by Section 10 (decisions 9, 13, 14, 15). Branch
`feature/cia-v2-composite` from master d857bf2a. The owner assigned Agent B's packages to the agent that ran Agent A's.

## Owner decisions for C1 (2026-10-29, decision 15)

1. **EuroSAT only.** On Alzheimer (alpha 1.0) leakage is near chance even without noise (AUC 0.564; AZ), so no
   mechanism can be measurably better there. The roadmap's "on EuroSAT and Alzheimer" is narrowed to EuroSAT; FINDINGS
   records why.
2. **The formal clause is reported, not gated,** as decision 13 for B1: the composite uses client subsampling, whose
   guarantee at the same noise on the average is weaker and mostly not computable (B1).
3. **Two adaptive scorings** are added, reported next to the gated attacker.

## Hypothesis

The two mechanisms that passed a B gate act on the same channel, the client's label mix (B5's diagnostic): logit
adjustment removes the client's label prior from local training (B5: -0.100 AUC at 2x), and subsampling at `q = 0.25`
dilutes it across rounds (B1: -0.10 at 0.740 accuracy). Combined, they lower the attacker's AUC at least 0.05 below
the paper's metric privacy at matched accuracy, and the reduction survives an attacker who knows the defenses.

## Design

**Regime: M2's.** Randomized inclusion (32 candidates, 16 IN per replicate, 32 active of 48), EuroSAT alpha 0.3, layout
seed 42, 50 rounds, equal weights, WP2 hyperparameters, `R = 32`, `design_seed = 2`, so every C1 run pairs with the
committed M2, A1 and B1 runs of the same replicate.

**Composite** (no new mechanism code): `--privacy subsampled-global-dp --sampling-rate 0.25` (B1's strategy: Poisson
sampling, denominator `q N`, noise `r C` on the average) with `--logit-adjustment-tau 1.0` (B5's local loss).

**Levels from an accuracy pilot** (accuracy only, `design_seed = 3`, 2 runs per level, never analyzed): `r` = 0.0001,
0.000773, 0.003092, 0.006184 (B1's `q = 0.25` levels). B1 at `q = 0.25` reached at most 0.746, so the pilot checks that
the composite still reaches 0.740; logit adjustment added 1.1 pp in B5. Three or four levels bracketing both targets are
fixed in PROTOCOL.md, with A1's replacement rule applied automatically after stage 1.

**Comparison curves, all committed and on the same design:** the paper's metric privacy and private median (A1), global
DP (M2), subsampling alone at `q = 0.25` (B1). Targets `a*` = 0.740 and 0.660 (A1's).

## Attacker and metrics

- **Gated attacker:** the calibrated loss attacker on the 11-checkpoint shadow-loss trajectory (as everywhere since M0).
- **Adaptive scoring (a), the label-channel attacker:** label-matched public surrogates scored like the shadow samples
  (`--surrogates`, M0's surrogate sets). This attacker holds no candidate records, only its label mix. Comparable with
  M2's runs, which scored surrogates; A1's and B1's did not.
- **Adaptive scoring (b), the defense-aware attacker:** each candidate's shadow sample also scored with the logit-adjusted
  loss `CE(softmax(log p + tau log pi_c))`, `pi_c` the candidate's label prior estimated from its own shadow sample
  (add-one smoothed), `tau = 1`: the attacker evaluates the loss the defended clients trained on. Computed for the
  composite only (the other configurations did not train on it).
- **Matched-accuracy statistic:** A1's (`AUC_composite(a*) - AUC_other(a*)`, curves linear in mean accuracy, pair
  jackknife at 97.5% with every resample drawing the same pairs in all configurations, two-way bootstrap alongside).
- Also reported: TPR at 5% and 1% FPR, top decile, accuracy, lowest client accuracy, amplified `mu` where computable.

## Gate

**G-C1 (decision 5 as amended by decisions 9, 13 and 15):** on EuroSAT, at matched accuracy (targets 0.740 and 0.660),
the composite's all-candidate AUC is at least 0.05 below the paper's metric privacy, with the 97.5% pair-jackknife
interval of the difference below 0. The two-way bootstrap is reported alongside and the outcome stated under both. The
formal guarantee is reported, not gated. On fail, the per-target results are reported. Also reported, not gated: the
composite against private median, global DP and subsampling alone, and both adaptive attackers.

## Cost

Pilot 8 runs and main runs about 4 levels x 32 = 128 runs at about 0.03-0.04 GPU-h (`q = 0.25`, plus surrogate scoring):
about 6 GPU-h. The roadmap's total (about 175 GPU-h) is used up (about 178 spent); the owner sets C1's cap.

## Files

| File | Change |
|---|---|
| `results/cia_v2/randomized_inclusion/runner.py` | score the logit-adjusted shadow loss (`shadow_adjusted_loss`) when a run uses logit adjustment; other runs unchanged |
| `results/cia_v2/randomized_inclusion/analysis.py` | optionally load the adjusted losses |
| `results/cia_v2/composite/` | STATE, PROTOCOL, FINDINGS, `analyze.py` with tests, pilot, results, analysis |

## Log

| When | What |
|---|---|
| 2026-10-29 | B1 closed and `feature/cia-v2-mechanisms` merged (decision 14); C1 opened; branch and sparse worktree created. Owner decisions for C1 (decision 15): EuroSAT only, formal clause reported, two adaptive scorings. STATE written before any code. |
