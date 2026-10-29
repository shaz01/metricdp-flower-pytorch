package: B1, client subsampling with amplification (owner decision 10: B5 first, then B1)
status: closed by the owner (2026-10-29); report reports/cia_v2_b1_subsampling.md
gate: G-B1: q = 0.25 passes at 0.740 (-0.102 [-0.164, -0.040]) and 0.660 (-0.059 [-0.096, -0.022]); q = 0.5 fails (-0.036, -0.023)
protocol: results/cia_v2/mechanisms/b1_subsampling/PROTOCOL.md (fixed 2026-10-27, before any leakage run)
machine: compute server
cost so far: about 10.9 GPU-h (pilot about 1, main runs about 9.9) of the 15 GPU-h cap
next: none until the owner decides
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

**Amended 2026-10-23 (owner, decision 13):** the formal condition is reported, not gated. At these noise levels both
mechanisms' `mu` are vacuous (about 40-140; any `mu` above about 5 allows an AUC near 1) and the accountant overflows
for some settings. G-B1 is decided on the AUC condition; the amplified `mu` is reported where it computes, flagged as
not computable otherwise.

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
| 2026-10-23 | Owner decision 13: G-B1's formal condition is reported, not gated. `analyze.py` and its test updated. |
| 2026-10-26 10:35 | Run-check fix for collapsed rounds merged from `feature/cia-v2-alzheimer` (combined with B1's sampled-client rule; 46 tests pass). GPU 1 idle and unclaimed: accuracy pilot started (12 runs: q = 0.5 and 0.25 at r = 0.000773, 0.001546, 0.003092; design seed 3, 2 runs each; tmux b1-p0, b1-p1). |
| 2026-10-26 21:00 | Pilot complete (12 runs). Mean final accuracy: q = 0.5: 0.779 / 0.779 / 0.744 and q = 0.25: 0.727 / 0.702 / 0.660 at r = 0.000773 / 0.001546 / 0.003092. Neither q brackets both targets (0.740, 0.660): q = 0.5 needs a noisier level, and q = 0.25 stays below 0.740. Pilot extension (accuracy only, same design, 2 runs each): q = 0.5 at r = 0.006184; q = 0.25 at r = 0.000387 and 0.0001 (its accuracy ceiling). |
| 2026-10-26 21:14 | Pilot extension complete: q = 0.5 at r = 0.006184: 0.693; q = 0.25 at r = 0.000387: 0.740, at r = 0.0001: 0.769. Both q can bracket both targets. |
| 2026-10-27 10:55 | PROTOCOL.md fixed: q = 0.5 at r = 0.001546, 0.003092, 0.006184, 0.012368; q = 0.25 at r = 0.0001, 0.000773, 0.003092, 0.006184 (four per q; one new level each, not piloted). 256 runs, about 11.5 GPU-h. Main runs started: three workers on GPU 1 (idle and unclaimed), each chain running stage 1, the automatic accuracy-only replacement check (`~/cia-server/b1-check.py`), then stage 2. |
| 2026-10-27 20:42 | Main runs complete (256); the automatic replacement check passed after stage 1 (q = 0.5: 0.762 / 0.724 / 0.633 / 0.515; q = 0.25: 0.741 / 0.730 / 0.685 / 0.588). Every run passed the run check. |
| 2026-10-29 14:50 | First analysis attempt timed out after 50 minutes: the accountant's PLD does not finish at noise multipliers far below its supported minimum (0.05). `amplified_mu` now reports such levels as not computable (test added); a reporting fix makes the formal condition read "not computable" instead of false. Analysis: **G-B1 passes for q = 0.25** at 0.740 (-0.102 [-0.164, -0.040], bootstrap [-0.166, -0.040]) and 0.660 (-0.059 [-0.096, -0.022], bootstrap [-0.105, -0.011]); **fails for q = 0.5** (-0.036 and -0.023). FINDINGS.md written. |
