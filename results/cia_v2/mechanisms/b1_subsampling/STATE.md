package: B1, client subsampling with amplification (owner decision 10: B5 first, then B1)
status: planning (STATE before code)
gate: G-B1 (amended by owner decision 9) = not evaluated yet
protocol: results/cia_v2/mechanisms/b1_subsampling/PROTOCOL.md, fixed after the accuracy pilot and before any leakage run
machine: compute server
cost so far: 0 GPU-h of the 15 GPU-h cap (stop and escalate at 18.75)
next: Poisson-sampling global-DP strategy with tests; accuracy pilot when a GPU is free
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
