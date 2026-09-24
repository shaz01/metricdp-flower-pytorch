# Influence-directed matched-energy noise pilot (EuroSAT) — EMPIRICAL, NO DP CLAIM

Branch `feature/influence-noise-pilot`, from `runs/new-auc-frontier-eurosat` at
`1b846ef` (descendant of the pinned-runtime commit `af3e6e3`). Separate from the
main AUC-frontier experiment; no main code paths, results, or controller state
are modified or reinterpreted. Results go to `results/influence_noise_pilot/`.

## Mechanism (`metricdp_pytorch/influence_noise.py`)

Per round, after Flower-identical flat L2 clipping of each client update at
`C = 5` and with FedAvg sample weights `w_i = n_i / Σ n_j` (all `< 1`, checked):

- `g = Σ w_i u_i`; leave-one-out influence with renormalised weights
  `d_i = g − g_{−i} = (w_i / (1 − w_i)) (u_i − g)`.
- Declared fixed cap `‖d_i‖ ≤ B = 5` (= `C`). Since `‖u_i‖, ‖g‖ ≤ C`,
  `‖d_i‖ ≤ 2C w_i/(1 − w_i)`; the cap only bounds pathological weight
  concentration. Chosen before any outcome, not tuned.
- `Σ = τ² [(1 − f) I + f D M / trace(M)]`, `M = Σ d_i d_iᵀ`, `D = 289,194`
  float parameters (EurosatCNN has GroupNorm only — no buffers; integer arrays
  are rejected). `trace(Σ) = τ² D`: the **same expected squared noise norm as
  the isotropic comparator every round** — noise is redistributed, not added.
- Sampling: `z = τ√(1−f) ξ + τ√(f D / trace M) Σ η_i d_i`, `ξ ~ N(0, I_D)`,
  `η ~ N(0, I_n)`; no D×D matrix. `f = 0` is exactly isotropic; `trace(M) = 0`
  falls back to `τ² I`.
- `τ = compute_stdv(noise_multiplier, C, active clients)`, the global-DP
  convention, with `noise_multiplier = .001546 × active` (48 IN / 47 OUT), so
  `τ = .001546 × 5` for both.
- RNG: dedicated `SeedSequence([tag, seed, round])`; `ξ` then `η` are always
  drawn, so the `f = 0` and `f = .5` arms share common random numbers per round.
  Replies are sorted by client ID before clipping/aggregation.

Consequence known before running (pure algebra, not an outcome): with
`f = .5`, `n ≤ 48` directions carry half of `τ² D`, so the per-direction noise
std along an influence direction is roughly `τ √(.5 D / rank) ≈ 55 τ` versus
`τ` isotropically, while every other direction gets `τ √.5`.

**Not differential privacy.** The covariance depends on this round's private
updates; the isotropic full-rank part does not establish a guarantee. `d_i`
is round-local influence of the clipped aggregate, not trajectory-level
removal sensitivity. The pilot tests feasibility and a matched-energy
comparison only.

### Why a new `f = 0` control instead of the main global-dp runs

The main seed-42 global-dp trajectories use the same `τ` but Flower's
`np.random` global stream and Flower's clipping path (which raises on zero-norm
updates). The `f = 0` control uses this implementation's clipping, ordering and
RNG, so the two arms differ only in `f`. Its noiseless aggregate is tested equal
to Flower's `DifferentialPrivacyServerSideFixedClipping`; the noise realisations
differ, so the main runs are not reused as the control.

## Pilot design (seed 42, alpha .3, targets 0–9, 100 rounds, original hparams)

Per `f ∈ {0, .5}`: one shared IN (evaluates all 10 panel targets every round)
plus OUT-t for t = 0..9 → 22 trainings. Per-round, per-client diagnostics land in
the training JSON's `train_metrics`: client IDs (OUT-federation IDs; canonical
mapping in `influence_protocol.json`), weights, clipped norms, weighted clipped
norms, raw/capped influence norms, cap-bound flags, directional noise std,
trace, fallback flag, realised noise energy. Clean/noisy per-target losses at
every round come from the reused frontier `measurements.json`; final per-class
metrics from the reused `*.evaluation.json`. These dataset-derived artifacts are
research-internal; nothing here is a release guarantee. Don't claim monotonic
behaviour in any parameter.

```bash
# plan (no training)
uv run python -m experiments.influence_noise_pilot.runner --fraction .5 --adjacency in
# one shard (on a GPU worker)
uv run python -m experiments.influence_noise_pilot.runner --fraction .5 --adjacency in \
  --output results/influence_noise_pilot/f0.5-seed-42-in --execute
uv run python -m experiments.influence_noise_pilot.runner --fraction 0 --adjacency out \
  --out-targets 0 --output results/influence_noise_pilot/f0.0-seed-42-out-0 --execute
```

Resume/sharding/locking are inherited from `experiments.auc_frontier.runner`
(per-trajectory manifest, atomic writes, completion marker, full retrain of a
partial trajectory). `influence_protocol.json` pins `f`, cap, `τ`, RNG scheme and
client mapping; a mismatch refuses to write into an existing trajectory.

Tests (synthetic only): `uv run pytest experiments/influence_noise_pilot`.

## Predeclared optional third arm (recorded 2026-09-23 ~23:20 CEST, before any outcome)

After all 22 f ∈ {0, .5} trajectories validate, an f = .05 arm (seed 42, IN + OUT
0–9, same protocol) may run on the same two lab4 slots — only as a whole arm and
only if its projected makespan ((90 + 10 × 55) / 2 min + 30 min collection margin;
OUT estimate raised from 50 to 55 min after observed ~52 min launch-to-collect)
ends before 2026-09-24 15:00 CEST; otherwise it is not started at all. Motivation
is the known pre-run anisotropy (influence-direction noise std ~16–100× τ at
f = .5), not any pilot result. It is an exploratory fixed-energy contrast, not a
confirmatory analysis; f = 0 and f = .5 stay fixed and are not altered mid-arm.

Execution (lab4 only, ≤2 A100): the two IN runs launch first; each arm's OUT runs
start only after that arm's IN is collected and validated (exit 0, torch 2.10.0 on
A100, 1000 finite measurement rows, 100 rounds of finite influence diagnostics,
evaluation JSON). One retry per trajectory; raw failed artifacts move to gitignored local
`.colab/quarantine/`, and only a small metadata JSON is committed under
`results/influence_noise_pilot/failures/`.

## Exploratory utility screen and early stop (owner-authorized 2026-09-24 02:05 CEST)

Supersedes the whole-arm 22-trajectory gate. Declared before any f = .05 outcome.
This is a **utility screen, not a privacy claim**.

Criterion: an arm fails if its seed-42 IN model, on the round-100 server
final-test split (`*.evaluation.json`, `server_final_test`, n = 1350), has
(a) accuracy more than 5 pp below the f = 0 IN, or (b) recall more than 20 pp below
the f = 0 IN for any of the 10 classes. A failing arm launches no new OUT runs;
already-running ones finish and are collected. No other f is searched after a failure.

- **f = .5 (post-hoc, decided after its IN/OUT-0 were seen): fails.** IN accuracy
  70.81% vs 86.00% (−15.2 pp); recall drops > 20 pp for annual crop land
  (91.8→15.8%), brushland/shrubland (73.1→39.7%), river (74.2→46.1%). OUT-0 (for
  context only): 70.67% vs 87.70%. New f = .5 OUT launches were paused at 02:06 CEST
  with OUT 0 collected and OUT 1 still running. The f = .5 data are partial
  (targets 0–1), are not comparable to full 10-target curves, and must not be
  pooled with them.
- **f = 0**: continues through OUT 0–9 (shared baseline). The f = 0 control uses
  its own RNG and clipping path, so it is not interchangeable with the main
  global-dp runs.
- **f = .05 (predeclared arm)**: IN runs first. Its OUT 0–9 launch only if the
  IN passes the criterion above.
