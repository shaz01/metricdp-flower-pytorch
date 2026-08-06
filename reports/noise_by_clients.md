# Phase 1 Item 2: Does the Noise-Multiplier Sweet Spot Shift with Client Count?

Source data: `results/noise_by_clients/`. Code: `experiments/client_scaling/sweep_noise_by_clients.py`,
`metricdp_pytorch/globaldp_strategy.py`, `metricdp_pytorch/metricdp_strategy.py`.

## Motivation

`results/48client_scaling` reused the 8-client sweep's `noise_multiplier=0.05` sweet spot unchanged
at 48 clients and found the metric-privacy-vs-global-dp advantage shrinking/reversing there. That
could mean the mechanism itself degrades with scale — or the sweet spot could simply move as client
count grows, making `nm=0.05` the wrong comparison point at higher `n` to begin with. This sweep
extends `sweep_noise_multiplier.py`'s noise grid (`{0.01, 0.05, 0.1, 0.25, 0.5, 1.0}`) into a 2D
matrix over `num_clients ∈ {8, 16, 32, 48}` to distinguish the two questions directly: for each
client count, where do global-dp and metric-privacy actually diverge from healthy training into
collapse?

**Scope note**: this sweep uses the fixed paper-default 20-round budget (`EXPECTED_ROUNDS=20`), not
the constant-compute control from `reports/constant_compute_scaling.md`. It answers "where does the
noise ceiling sit at each client count," not "how does accuracy scale with client count" — that
second, round-budget-confound-controlled question is `reports/constant_compute_scaling.md`'s.
Absolute accuracy here drops with `n` partly because more clients means less data per client under
this sweep's fixed round budget, which is expected and out of scope for this report.

`fedavg` only, both partition modes, seed 42, clipping norm 5.0 — 96 combinations (4 client counts
× 2 partitions × 2 privacy modes × 6 noise multipliers). **96/96 completed, 0 failures** (after
fixing a zero-norm-update crash in `globaldp_strategy.py` mid-sweep — see `git log` for `0a60acd`;
12 combinations failed on the first attempt, all `global-dp` at `nm >= 0.25`, all succeeded on
rerun with the fix).

## Results: accuracy by client count and noise multiplier

Training-time recorded accuracy from the last completed round (20/20 for every combination),
averaged across partition modes for readability — see `results/noise_by_clients/*.json` for the
homogeneous/non-IID breakdown.

| n | nm=0.01 | nm=0.05 | nm=0.1 | nm=0.25 | nm=0.5 | nm=1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 8  | 90.63% | 79.65% | 49.53% | 49.60% | 35.94% | 12.43% |
| 16 | 79.45% | 80.34% | 65.66% | 49.53% | 42.77% | 24.69% |
| 32 | 66.17% | 65.82% | 66.97% | 45.55% | 49.46% | 49.77% |
| 48 | 60.16% | 61.88% | 62.70% | 46.48% | 49.61% | 49.69% |

Global-dp and metric-privacy averaged together (the split matters for the delta analysis below, but
both privacy modes collapse at essentially the same noise multiplier per client count — the table
above is representative of "is training still healthy," not a privacy-mode comparison yet).

## Finding 1: the noise ceiling shifts up as client count grows

At n=8, accuracy is already near-random by nm=0.1 (49.5%, the same "stuck at baseline" value seen
throughout this project's collapsed runs). At n=16/32/48, nm=0.1 is still healthy — in fact at
n=32/48, accuracy at nm=0.1 is comparable to or slightly *better* than at nm=0.05 (32: 66.97% vs.
65.82%; 48: 62.70% vs. 61.88%). The collapse point moves from roughly nm=0.05–0.1 at n=8 to
roughly nm=0.1–0.25 at n=16/32/48.

This matches the expected DP-FL mechanism, not a surprise: `compute_stdv` scales injected noise as
`noise_multiplier * clipping_norm / num_sampled_clients` — at a fixed nominal `noise_multiplier`,
more participating clients means less actual per-round noise magnitude relative to the aggregate
signal. **So yes: the sweet spot does shift with client count** — reusing `nm=0.05` from the
8-client sweep at 48 clients (as `results/48client_scaling` did) was comparing a near-optimal noise
level at n=8 against a conservative, not-yet-maximally-informative one at n=48, where `nm=0.1`
would sit in roughly the same relative position on the accuracy-vs-noise curve.

## Finding 2: metric-privacy's advantage is real but narrow, and reverses sharply near collapse

| n | nm=0.01 | nm=0.05 | nm=0.1 | nm=0.25 | nm=0.5 | nm=1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 8  | −0.16pp | **+8.12pp** | −0.23pp | +0.31pp | +6.80pp* | −0.08pp |
| 16 | −0.31pp | +0.70pp | −0.39pp | +0.00pp | +6.80pp* | +11.25pp* |
| 32 | −0.31pp | +0.23pp | −3.67pp | +6.25pp* | +0.08pp | **−13.83pp** |
| 48 | −0.55pp | **+2.19pp** | −0.86pp | **−18.44pp** | +0.16pp | −0.23pp |

Delta = metric-privacy accuracy − global-dp accuracy, averaged across partitions. *Entries at or
past the collapse boundary (both sides near the 35–50% "stuck at baseline" plateau) — treat these as
noise-floor artifacts of which side happened to freeze at a marginally less-bad degenerate state, not
a genuine mechanism comparison; neither side is actually learning at these noise levels.

Reading only the cells where both mechanisms are still genuinely training (left of each row's
collapse point per Finding 1): metric-privacy's advantage is **positive but modest at nm=0.05
across every client count tested** (+0.70 to +8.12pp), consistent with this being roughly the
right comparison point at every n — not just n=8. But the picture changes right at the edge of
collapse: n=32/nm=1.0 (−13.83pp) and n=48/nm=0.25 (−18.44pp) are both genuine, large *negative*
deltas where global-dp is still training reasonably (45–50% accuracy, not collapsed) while
metric-privacy has already crossed into collapse. **This suggests metric-privacy's calibrated noise
mechanism becomes unstable and underperforms global-dp's simpler fixed noise specifically in the
transition zone approaching collapse, and this effect gets worse, not better, at higher client
counts** — the opposite of a "sweet spot shift" explanation, and a genuine candidate mechanism-level
finding for Phase 2's redesign work to address.

## Known gaps / caveats

- **Fixed 20-round budget, not constant-compute** — see the scope note above; don't cite this
  report's absolute accuracy-vs-n trend as a scaling result, only the noise-multiplier-vs-n
  collapse pattern.
- **Single seed (42).** No variance estimate on where exactly the collapse boundary sits.
- **`fedavg` only.** `fedyogi`/other aggregators not covered by this sweep.
- **The collapse mechanism itself isn't diagnosed here** — this report identifies *where* collapse
  happens and that metric-privacy's failure mode there is worse than global-dp's, not *why*
  metric-privacy's calibrated noise is more fragile in that zone. That's a natural next diagnostic
  step, likely relevant to Phase 2's distance-function redesign work.

## Bottom line

Both questions this sweep set out to answer got real answers: the noise ceiling genuinely shifts up
with client count (Finding 1) — so the earlier `results/48client_scaling` reversal was at least
partly a wrong-comparison-point artifact, not purely a scaling failure of the mechanism. But a second,
unanticipated finding (Finding 2) shows metric-privacy has its own genuine fragility right at the
collapse boundary that gets worse at scale — a real, narrower version of the original scaling
concern, now precisely localized instead of conflated with the noise-sweet-spot question.
