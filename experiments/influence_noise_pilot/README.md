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
