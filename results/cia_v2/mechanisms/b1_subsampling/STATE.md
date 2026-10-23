package: B1, client subsampling with amplification (owner decision 10: B5 first, then B1)
status: code done (strategy, wiring, analysis, tests); accuracy pilot waits for a GPU
gate: G-B1 (amended by owner decision 9) = not evaluated yet
protocol: results/cia_v2/mechanisms/b1_subsampling/PROTOCOL.md, fixed after the accuracy pilot and before any leakage run
machine: compute server
cost so far: 0 GPU-h of the 15 GPU-h cap (stop and escalate at 18.75)
next: accuracy pilot (12 runs) when a GPU is available; then levels and PROTOCOL.md (formal-mu condition to settle with the owner first)
needs from others: none (the accountant `metricdp_pytorch/accounting.py` is Agent C's, delivered with B4)

Roadmap: `research/cia_roadmap_v2/ROADMAP.md` Section 3, B1, as amended by Section 10 (decisions 9 and 10,
2026-10-20). Branch `feature/cia-v2-mechanisms`.

## Hypothesis

Sampling each client into a round with probability `q` amplifies the formal guarantee, and it also weakens the
membership signal: a candidate influences only about `q` of the rounds. At matched accuracy, client subsampling lowers
the attacker's AUC below full-participation global DP, with an amplified formal `mu` no larger.

## Design

**Regime: M2's.** Randomized inclusion (32 candidates, 16 IN per replicate, 32 active of 48), EuroSAT alpha 0.3,
layout seed 42, 50 rounds, equal weights, WP2 hyperparameters, `R = 32`, `design_seed = 2`, so replicate `j` has M2's
inclusion and `(client_seed, noise_seed)`.

**Mechanism** (`PoissonSampledGlobalDP`, a new server strategy; client code unchanged):

- Each round, every active client joins independently with probability `q`, drawn from a generator seeded by the
  run's noise seed (sampling is private randomness, like the noise).
- Public denominator `q N` with `N = 32`, the public active count: the release is
  `w + (sum_{j in S} clip(u_j, C)) / (q N) + N(0, (r C)^2 I)`, so the noise on the released average is `r C` as in
  M2's global DP. A round with no sampled client releases noise only.
- Accounting: Poisson-subsampled Gaussian with noise multiplier `z = r q N`, `T = 50`;
  `accounting.dominating_mu` (Agent C's accountant) gives the amplified `mu`, reported with its delta floor.
- `q` = 0.5 and 0.25 (the roadmap's values).

**Threat model.** Amplification holds against observers of the released models while each round's sample stays secret
(`research/cia_roadmap_v2/theory/accounting.md`). The attacker here sees only released models through the shadow losses,
so the audit matches that model.

**Levels, from an accuracy pilot** (accuracy only, `design_seed = 3`, 2 runs per level, never analyzed): both `q` at
`r` = 0.000773, 0.001546 and 0.003092 (0.5x, 1x and 2x M2's frontier ratio; fewer clients per round should cost accuracy,
so the levels sit below M2's). Three levels per `q` are then fixed in PROTOCOL.md so that both targets are bracketed.

**Targets and comparison: A1's matched-accuracy design.** Targets `a*` = 0.740 and 0.660 (M2's accuracy at 2x and 4x,
A1's targets). Per `q`, AUC against mean final accuracy is interpolated between its levels, and global DP's curve is
M2's; `AUC_B1(a*) - AUC_GDP(a*)` with the pair jackknife at 97.5% and the two-way bootstrap alongside
(`results/cia_v2/frontier/a1/analyze.py`). The roadmap's "noise matched to global DP's accuracy within 1 pp" is replaced
by interpolation to the exact target, as in A1.

## Gate

**G-B1 (decision 9):** at matched accuracy, the all-candidate AUC is at least 0.05 below full-participation global DP
with the gate interval of the difference below 0, **and** the amplified formal `mu` is no larger than global DP's at the
same accuracy (`mu` of each mechanism interpolated in accuracy between its levels, as the AUC). The top decile is
reported with its interval but not gated.

## Cost

Per-round cost falls about with `q`. At M2's 0.11 GPU-h per full run: about 0.06 (`q = 0.5`) and 0.03 (`q = 0.25`) per
run, so 6 configurations x 32 runs come to about 9 GPU-h, plus about 0.6 for the pilot.

## Files

| File | Change |
|---|---|
| `metricdp_pytorch/subsampled_dp_strategy.py` | `PoissonSampledGlobalDP` (sampling, public denominator, noise, diagnostics) |
| `experiments/reproduce/server.py`, `runner.py` | `--privacy subsampled-global-dp --sampling-rate q` |
| `results/cia_v2/randomized_inclusion/runner.py` | the privacy mode and its sampling rate |
| `results/cia_v2/mechanisms/b1_subsampling/` | STATE, PROTOCOL, FINDINGS, pilot, results, analysis |

## Log

| When | What |
|---|---|
| 2026-10-23 | STATE written before any code. B5 analyzed (FINDINGS.md); the owner decides whether B5 is finished. |
| 2026-10-23 | `feature/cia-v2-alzheimer` merged into this branch (fail-fast on failed clients, the run check, dataset profiles, the private-median collapse guard); all 565 committed manifests (M2, A1, B5) regenerate identically, 230 related tests pass. |
| 2026-10-23 | Code: `metricdp_pytorch/subsampled_dp_strategy.py` (`PoissonSampledGlobalDP`: Poisson sample from a generator seeded by the noise seed, denominator `q N`, noise `z C / (q N)`, noise-only empty rounds), server and runner wiring (`--privacy subsampled-global-dp --sampling-rate q`; the randomized-inclusion runner sets `z = r q N`, records `sampling_rate` in the manifest; the fail-fast check and the run check count the sampled clients). 8 tests. A 3-round CPU smoke run (q = 0.25, r = 0.001546) trained exactly the sampled clients (6, 9, 15 of 32), noise std 0.00773 = r C. `b1_subsampling/analyze.py` (A1's matched-accuracy statistic per q; amplified mu interpolated in log mu; G-B1) with 5 tests. |
| 2026-10-23 | **Accountant finding.** At B1's noise multipliers (z = r q N, about 0.006-0.05) `dominating_mu` with the default PLD discretization did not finish in 10 minutes; with discretization 0.1 it returns 42.9 for q = 0.25, r = 0.003092 (global DP at that r: 71.5) and overflows (inf) for q = 0.5, r = 0.003092. Both mechanisms' mu are far above any meaningful level here (mu above about 5 allows AUC near 1), so G-B1's formal condition compares vacuous numbers and is not always computable. analyze.py reports mu where it computes and marks it not computable otherwise; the owner decides how the condition reads before PROTOCOL.md. |
