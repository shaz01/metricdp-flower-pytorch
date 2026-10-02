# CIA frontier (EuroSAT, Dirichlet α = 0.3) — Sep 22–24, 2026

Owner's September work, merged from `runs/new-auc-frontier-eurosat` and
`feature/influence-noise-pilot`. Not declared finished.

## Question

With **per-client** scoring, does global DP (GDP) or metric privacy (MP) reduce
client-inference (CIA) leakage compared with vanilla FedAvg on EuroSAT,
label-Dirichlet non-IID (α = 0.3), 48 canonical clients, 100 rounds?

Full pre-run protocols: [`PROTOCOL.md`](PROTOCOL.md) (frontier) and
[`influence_noise/PROTOCOL.md`](influence_noise/PROTOCOL.md) (pilot).

## Layout

| Path | Contents |
|---|---|
| `runner.py`, `data.py`, `analyze.py`, `training_seed_variance.py` | frontier runner, Dirichlet partitioner, analysis |
| `influence_noise/` | influence-direction noise pilot (mechanism: `metricdp_pytorch/influence_noise.py`) |
| `results/eurosat_frontier/` | `alpha_pilot/` (12 vanilla IN runs, α ∈ {.1,.3,1,10} × 3 seeds), `frontier/`, `frontier_torch210/`, `frontier_torch210_s44_mixedenv/`, `failures/` |
| `results/influence_noise/` | 25 pilot trajectories + `failures/` |
| `reports/` | `build_eurosat_frontier_progress.py` → `eurosat_frontier_progress.html` (provisional) |
| `tests/` | synthetic tests (no data/GPU) |

## What exists

**88 attack trajectories** at noise ratio 0.001546, targets 0–9, every round 1–100:

- `frontier/` (54): GDP and MP, seeds 42/43 (IN + OUT 0–9) and seed 44 (IN + OUT 0–3);
  old Colab runtime, torch 2.11.0+cu128.
- `frontier_torch210_s44_mixedenv/` (12): seed-44 OUT 4–9 for GDP and MP under pinned
  torch 2.10.0 (paired with the old-runtime seed-44 IN, hence "mixed env").
- `frontier_torch210/` (22): pinned torch 2.10.0; vanilla seed 42 (IN + OUT 0–9) plus an
  overlap re-run of seed 42 (GDP IN + OUT 0–3, MP IN + OUT 0–4) against the old runtime.

**Influence pilot: 25 runs**, seed 42, GDP-calibrated noise with fraction *f* steered
along the influence direction: f = 0 (isotropic control) and f = 0.05 (IN + OUT 0–9 each),
f = 0.5 (IN + OUT 0–1, early-stopped by the predeclared utility screen).

## Scoring

Per target client *t*: fraction of rounds 1–100 where the IN model's clean-shadow loss on
*t* is lower than the matching-round OUT-without-*t* model's (lower = IN, direction fixed
in advance). Averaged over targets. No `max(AUC, 1−AUC)` folding, no pooling of loss
scales across clients. This replaces the single-target scoring used in
`results/contest_at_scale`.

## Key result (exploratory, seed-level, no CIs)

| Arm | Seed / runtime | Mean per-client IN-lower fraction |
|---|---|---|
| Vanilla | 42, torch 2.10 | 0.685 |
| GDP | 42 / 43 / 44, old (44 mixed) | 0.675 / 0.611 / 0.723 |
| MP | 42 / 43 / 44, old (44 mixed) | 0.713 / 0.604 / 0.678 |
| GDP overlap (4 targets) | 42, torch 2.10 | 0.723 |
| MP overlap (5 targets) | 42, torch 2.10 | 0.834 |

GDP spans 0.61–0.72 and MP 0.60–0.71 versus vanilla 0.69: at ratio 0.001546 neither
mechanism shows a leakage reduction distinguishable from seed-to-seed spread. The overlap
cohort shows runtime changes alone move per-target scores (some deltas flip sign).

Influence pilot (seed 42, round-100 server test accuracy; same scoring):
f = 0: 86.0 %, 0.549 · f = 0.05: 85.2 % (−0.8 pt), 0.583 · f = 0.5: 70.8 %, 0.49 (2 targets
only). f = 0.05 costs accuracy without reducing attack success.

Numbers recomputed from the committed `measurements.json` and run/evaluation JSONs;
the progress HTML shows the matching final-round paired views.

## Open items

- Training-seed variance runs (`training_seed_variance.py`, fixed partition seed 42,
  training seeds 43/44) never ran — A100 quota ran out. No `results/eurosat_frontier/training_seed_variance/` exists.
- Only one noise ratio was attacked; the upward noise sweep (larger ratios toward a
  leakage reduction) was not done.
- Vanilla has a single seed; old vs pinned runtimes are not interchangeable.

## Commands

```bash
uv run pytest results/cia_frontier
uv run python -m results.cia_frontier.analyze results/cia_frontier/results/eurosat_frontier/frontier --seeds 42 43  # final-round only; seed 44 is split across roots
uv run python results/cia_frontier/reports/build_eurosat_frontier_progress.py
```

Results/manifests are historical and unmodified: stored module strings such as
`experiments.auc_frontier.data:create_data_module` predate the move; run names are unchanged.
