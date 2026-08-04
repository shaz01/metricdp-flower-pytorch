# Client Inference Attack (CIA) — Status Report

Two related but distinct CIA experiments exist in this codebase: the original 3-client
reproduction of the paper's first-round single-shot attack, and a newer 48-client
client-count-scaling variant. Code: `experiments/cia/` (`runner.py`, `client_scaling.py`,
`attack_runner.py`, `result.py`, `datasets/paper.py`, `datasets/shadow.py`).

## 1. Paper reproduction (3-client, Tables 9–12)

**Still no result data committed anywhere on `master`** — `results/cia/` does not exist, and the
runner's own default output, `experiments/cia/results/first_round_cia.json`, is not present
either. This section documents scope and design only, unchanged in substance from the previous
version of this report; only file paths below have been updated to match a refactor that
restructured the module (`dataset.py` → `datasets/paper.py` + `datasets/shadow.py`; `attack.py` →
`result.py`; the run loop is now shared with the client-scaling variant via `attack_runner.py`).

### Objective

Reproduces Section 5 and 7.4.1 of Sáinz-Pardo Díaz et al. (2026): a semi-honest client (the
attacker) infers whether another client (the target) participated in training by comparing the
aggregated model's loss on a public test set against its loss on a shadow sample drawn from the
target's own training data. Scope is **first-round, single-shot only** (paper Tables 9–12); the
multi-round AUC-based variant (Section 7.4.2, Table 13) is not implemented.

### Design

1. **Partitioning** (`datasets/paper.py`): splits the Alzheimer MRI train set into 3 clients per
   the paper's Table 9 (client 1 = attacker, client 2 = bystander, client 3 = target). Table 9 as
   published has a typo in client 2's row (1591/180); the code uses the corrected values
   (1491/80), justified by an arithmetic reconciliation against Table 1's row/column/grand totals
   documented in the module's docstring.
2. **Shadow derivation** (`datasets/shadow.py`): `ShadowDataModule` is a dataset-independent
   decorator — it delegates normal client/server loading to any federated data module and derives
   the shadow set from the target client's actual training loader, so the same shadow-splitting
   logic is shared between this experiment and the client-scaling variant in §2.
3. **Training** (`runner.py`): for each of the 18 `(privacy_mode, aggregation)` combinations, runs
   one real 1-round, 3-client Flower simulation with `local_epochs=20` (the paper's CIA-specific
   value), via the shared `attack_runner.run_attack`.
4. **Evaluation**: the resulting model's loss is measured on the global held-out test set and on a
   stratified 10% shadow sample of the target's train indices — which overlaps with, rather than
   is excluded from, what the target trains on, matching the paper's stated "strong adversarial
   assumption."
5. **Scoring** (`result.py`): `relative_difference(aggregated_loss, target_loss)` computes
   `(target_loss - aggregated_loss) / target_loss * 100`, matching Tables 10–12's structure. The
   formula's direction was checked against the paper's own worked example (aggregated=1.032,
   target=1.182 → 12.69%, matching the published 12.719% within table-rounding noise). `result.py`
   also now computes a proper ROC-AUC membership-inference score (`attack_auc`, using per-example
   `-loss` as the membership score over shadow-vs-test examples) — an addition beyond the paper's
   own relative-loss-difference metric. No committed run exists yet to report a value for it.

### What's needed for a results report

Run `uv run python -m experiments.cia.runner --output-dir experiments/cia/results` (18 real
1-round, 3-client training runs) and commit the resulting artifacts; then this section can be
replaced with the per-combination loss table and paper comparison.

## 2. Client-count-scaling CIA (48 clients)

Real result data now exists at `results/cia_client_scaling/` — this is new since the previous
version of this report. It does not yet form one clean, directly comparable table, for reasons
worth stating precisely rather than glossing over.

### What's actually committed

18 trained models exist, falling into two groups by their own recorded training metadata (not by
assumption):

- **Group A — 15 models, `rounds=1, local_epochs=20`**: 12 at `noise_multiplier=0.01` (the full
  2 partitions × 3 privacy modes × 2 aggregations matrix) plus 3 more at `noise_multiplier=0.12`
  (homogeneous / FedYogi only). These hyperparameters match the *original* 3-client CIA design
  (`local_epochs=20`) scaled to 48 clients — i.e. a fresh model trained for exactly 1 round per
  combination, named `first-round` in the filenames.
- **Group B — 3 models, `rounds=20, local_epochs=5`**: homogeneous / FedYogi /
  `noise_multiplier=0.12` only, named `post-convergence`. These hyperparameters match the
  *current* `client_scaling.py`'s "concluded client-scaling settings."

Only **6 of these 18** have a committed attack score, in
`results/cia_client_scaling/cia_client_scaling.json`: all `homogeneous / FedYogi /
noise_multiplier=0.12` — the 3 Group-A `first-round` models paired against the 3 Group-B
`post-convergence` models. The other 12 Group-A models (`noise_multiplier=0.01`, the full
first-round matrix) have **no committed attack score** — only their raw training-run JSON exists.

### These two groups are not checkpoints of one trajectory

`experiments/cia/client_scaling.py`'s own docstring and `experiments/cia/README.md` describe a
unified design: train one 20-round trajectory per combination, keep checkpoints at rounds 1 and
20, and attack both checkpoints from the *same* trajectory. The committed data does not match that
description — Group A and Group B have different `local_epochs` (20 vs. 5), which would not
happen if they were two checkpoints pulled from one run with one fixed `local_epochs` value.

The explanation is version history, not a data bug: `git log` shows `cia_client_scaling.json` was
last written by commits `dbaf7d3`/`7c0d7fc`, both **before** the "unify scaling attacks with round
checkpoints" refactor (`9980232`) landed. The committed data predates the unified-trajectory design
the code and README currently describe — it was produced by an earlier version of the pipeline
that trained `first-round` and `post-convergence` as two independent models. **The table below
should be read as two independently-trained models per privacy mode, not as one trajectory's
before/after.** A same-trajectory comparison would require rerunning under the current
`client_scaling.py`.

The report file's schema is correspondingly stale: it's a `{"results": [...], "failed": [...]}`
dict with a string `"timing"` field, produced by an older version of `attack_runner.py`. The
current `attack_runner.py` writes a plain list of `CiaResult` records keyed by an integer
`server_round`, and includes the `attack_auc` field described in §1 — absent from every entry in
this file.

### The 6 committed attack scores (homogeneous, FedYogi, noise_multiplier=0.12)

| Privacy | Group A "first-round" (rounds=1, local_epochs=20) | Group B "post-convergence" (rounds=20, local_epochs=5) |
|---|---:|---:|
| vanilla | −266.83% (agg=0.8510, target=0.2320) | −1011.52% (agg=0.4097, target=0.0369) |
| global-dp | −69.64% (agg=0.9085, target=0.5356) | −1260.81% (agg=0.5366, target=0.0394) |
| metric-privacy | −80.99% (agg=0.9025, target=0.4987) | −139.27% (agg=0.6572, target=0.2747) |

`diff% = (target_shadow_loss − aggregated_test_loss) / target_shadow_loss × 100`. Every value is
strongly negative, meaning `target_shadow_loss < aggregated_test_loss` in all 6 cases: the model
fits the target's own shadow sample noticeably better than the general test set — the direction
consistent with genuine membership leakage under this metric.

**Caveat on magnitude**: `shadow_size=8` for all 6 rows (10% of the target's local partition at 48
clients, consistent with this project's recurring per-client-data-scarcity theme at high client
counts). An 8-example loss estimate is noisy, and the percentage formula divides by
`target_shadow_loss`, which is small in several of these rows (as low as 0.037) — so a modest
absolute loss gap gets amplified into a triple-or-quadruple-digit percentage. The huge magnitudes
here (up to −1260%) reflect that sensitivity as much as attack strength; they should not be
compared at face value against the paper's own percentages (Tables 10–12, computed on much larger
shadow samples at 3 clients) or over-interpreted as "the attack got 4x stronger by convergence" —
that reading would also require the same-trajectory comparison this data doesn't actually provide
(see above).

### What's needed for a complete, trustworthy results report

- Re-run `experiments/cia/client_scaling.py`'s current 12-combination matrix
  (`noise_multiplier=0.05`, 2 partitions × 3 privacy modes × 2 aggregations) to get one coherent,
  same-trajectory, same-schema dataset with `attack_auc` populated — the 12 Group-A models already
  committed at `noise_multiplier=0.01` are not a substitute; they use different hyperparameters and
  have no attack score computed at all.
- Given `feature/scaling-diagnosis`'s finding that this machine's MPS backend is non-deterministic
  under concurrent multi-actor training (see `reports/constant_compute_scaling.md`), any new CIA
  client-scaling run intended for a committed report should consider `METRICDP_FORCE_CPU=1` rather
  than reusing MPS's default, or should be treated as unverified until spot-checked the same way.

## Test coverage

`experiments/cia/tests/` (`test_dataset.py`, `test_attack.py` → now covering `result.py`,
`test_runner.py`, `test_client_scaling.py`, `test_iter_combos.py`) unit-tests the partitioning
arithmetic, the attack-score formula, and both runners' orchestration logic. All pass as part of
the full suite (`uv run pytest`: 71 passed, 5 reproducibility-marker tests deselected by default).
These exercise the code paths, not the paper comparison itself — no §1 simulation has been run to
completion and captured on `master`, and §2's committed data predates the current unified design.
