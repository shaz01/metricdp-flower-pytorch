# Handoff: next research phase against client inference attacks

Written 2026-10-10 at the end of a literature-review and direction-design session. Status update (2026-10-10, owner):
**approved for autonomous execution by Claude Code agents under
[research/cia_review_2026-10/agent_roadmap.md](research/cia_review_2026-10/agent_roadmap.md)**, whose Section 1 owner
decisions are binding and supersede this file where they differ (participant count secret, equal weights, Alzheimer
MRI as second dataset, small jobs on the owner PC and large jobs on the compute server, agents never merge into
`master`). Nothing was running when this was written.

Read in this order (about 30 minutes):

1. This file.
2. [research/cia_review_2026-10/agent_roadmap.md](research/cia_review_2026-10/agent_roadmap.md): work packages, gates,
   file ownership, escalation rules and the agent prompts.
3. [research/cia_review_2026-10/project_state.md](research/cia_review_2026-10/project_state.md): what the repository has
   established, why the client-specific-noise direction failed, new log analyses.
4. [research/cia_review_2026-10/research_directions.md](research/cia_review_2026-10/research_directions.md): ranked
   directions D1-D6 with hypotheses, formal arguments, reviewer objections, kill criteria and staged experiments.
5. [research/cia_review_2026-10/literature_review.md](research/cia_review_2026-10/literature_review.md) (97 verified
   references, table in `paper_table.csv`, BibTeX in `references.bib`) when you need the prior work behind a claim.

Session-start rules from `AGENTS.md` still apply: read `STATUS.md`, run `git log --oneline -10`, `uv sync`.

## 1. Goal

Develop a federated-learning privacy technique that protects better than the metric-privacy-inspired noise calibration
of Sáinz-Pardo Díaz et al. (Knowledge-Based Systems 343:115993, 2026; `papers/Initial-Paper.pdf`) against **client
inference attacks (CIA)**: a curious participant holding a shadow sample of a target client's data decides whether the
target's whole dataset took part in training. The baseline sets the server noise to `sigma_t = z C / (n_t d_t)` with
`d_t` the maximum pairwise distance between client models (`metricdp_pytorch/metricdp_strategy.py`); global DP (GDP)
uses `z C / n`.

## 2. Where things stand

| Evidence | Result | Source |
|---|---|---|
| Paper, 3-client Alzheimer MRI CIA (Table 13) | AUC vanilla 0.890, GDP 0.397, metric privacy 0.493 (DP intervals overlap 0.5); IN accuracy 0.948 / 0.637 / 0.794 | `papers/Initial-Paper.pdf` |
| Repository scaling | Metric privacy +6.9 pp (homogeneous) and +12.2 pp (non-IID) over GDP at 8 clients, `z = 0.05`; parity at 48 clients (-2.5 to +0.6 pp) | `STATUS.md`, reports at tag `freeze/2026-10-02` |
| AUC-targeted noise sweep | 9 of 10 landed means above 0.55; no mechanism dominates | `results/contest_at_scale/auc_frontier/reports/auc_targeted_noise_sweep.md` |
| EuroSAT frontier (48 clients, Dirichlet 0.3, 100 rounds) | Neither defense leaks less than vanilla at the landed noise (vanilla 0.69, GDP 0.61-0.72, metric privacy 0.60-0.71); per-target scores 0.2-0.99 | `results/cia_frontier/eurosat_frontier/` |
| Client-specific noise (closed 2026-10-09) | No gain from per-client shapes, non-Gaussian densities, influence-directed noise; the surviving stacked-head construction loses to the server mechanisms at matched risk; metric privacy behaves like GDP once leakage is matched; the public-validation step gate nearly doubles utility at risk 0.80 | `research/README.md`, `experiments/stacked_head/protocols/` |
| This session's log analyses | (E1) the metric-privacy noise-level side channel is weak at 48 clients (sigma shift below 1.1% per target); (E2) in 3-vs-2-client runs the noise reveals the participant count for both mechanisms, plus a 3-9% distance factor for metric privacy; (E3) a median dispersion statistic moves at most 2.0% per round when one client is removed, against 12.3% for the max; (E4) per-target exposure rises with client weight (Spearman 0.58-0.59), and weight, update norm and model distance are collinear; (E6) formal GDP mu at the frontier noise is about 135 over 100 rounds (vacuous), so the defenses work only because the realistic attacker is weak | `research/cia_review_2026-10/research_directions.md` Section 2 |

What failed and why, in one line each (details: `project_state.md` Section 4): local noise placement is equivalent to
central noise for a model observer; isotropic Gaussian is optimal for the worst-case envelope; non-Gaussian densities
did not help at matched attack strength; data-dependent noise parameters leak; noise on learning directions destroys
accuracy; little private-signal headroom over 32+ public images; at matched risk the server mechanisms win.

## 3. Recommended next direction

**D2-lite, then D1, with D4 as the formal follow-up** (`research_directions.md` Sections 0 and 5):

- **D1, signature-suppressing local training under a fixed, public client-level DP noise law.** Keep a GDP-style noise
  law that depends only on public quantities (so the guarantee holds for any local algorithm) and reduce the target's
  idiosyncratic contribution in local training: fewer local epochs, FedProx, fixed local steps with equal weights,
  consensus distillation `KL(p_global || p_local)` on local data, drift correction with a public control variate.
  First-order analysis: the participation gap is about `w_i eta K_i <g_i, g_i - g_bar_{-i}>`, so consensus-aligned
  learning carries no CIA signal while noise has to cover all directions.
- **D2, defense-aware CIA audit.** Fixed-denominator small federations to test the noise-level side channel, and a
  per-round loss-increment likelihood-ratio attacker with known noise law; low-FPR reporting. Needed before any frontier
  claim, because E6 shows the current attacker is far from the worst case.
- **D4, robust private dispersion and residual clipping.** A leak-free metric-privacy baseline (private median distance)
  and clipping of residuals around the previous released aggregate, which lowers sensitivity from `C` to `rho`.

Kill criteria are written per direction; respect them rather than tuning past them.

## 4. First experiment, step by step

Stage A of D1 plus the D2 logging it needs. Do not launch training until the owner (or an execution agent the owner
assigns) authorizes it; this matches the convention in `results/cia_frontier/eurosat_frontier/PROTOCOL.md`.

1. Branch from `master`: `git checkout -b feature/signature-suppression`.
2. Write the protocol first: `results/cia_frontier/signature_suppression/PROTOCOL.md`, modelled on
   `results/cia_frontier/influence_noise/PROTOCOL.md`. Declare before any run: EuroSAT, Dirichlet 0.3, 48 clients, seed
   42, 50 rounds, removal adjacency; five targets (the two largest-weight targets plus three spread over the weight range,
   from `research/cia_review_2026-10/data/target_idiosyncrasy_vs_cia_score.csv`); nine configurations (vanilla; weight
   decay 5e-4; local epochs 1; FedProx mu 0.01 / 0.1 / 1; fixed steps + equal weights; consensus KL beta 0.3 / 1); no DP;
   metrics (pooled clean- and noisy-view scores, AUC, TPR at 5% FPR, accuracy, per-target scores); kill criterion D1-1.
3. Runner `results/cia_frontier/signature_suppression/runner.py` that reuses
   `results.cia_frontier.eurosat_frontier.runner` (`build_combos`, `FrontierCombo`, `ADJACENCIES`, `atomic_json`) the way
   `results/cia_frontier/influence_noise/runner.py` does, and plans unless `--execute` is given.
4. Training-loop options with defaults that leave every existing run unchanged:
   expose FedProx `proximal_mu` (fixed at 0.5 in `metricdp_pytorch/strategy_factory.py`); add the consensus-KL term next to
   `proximal_term` in `experiments/reproduce/paper_training.py` (`train_with_adam`) and pass it like `proximal-mu` in
   `experiments/reproduce/client.py`; add a fixed local-step budget. Equal weighting already exists
   (`--aggregation-weighting equal`).
5. Diagnostics: per round and client, the norm of the update minus the leave-one-out mean update (server side, before
   clipping), and on the attack side the shadow-sample loss increment and gradient norm (scalars only).
6. Synthetic unit tests (no datasets, no GPU), then `uv run pytest`.
7. After authorization: run Stage A (about 54 trajectories of 50 rounds, about 20 trajectory-hours; one 100-round,
   48-client trajectory took a median 0.68 h training plus 0.06 h evaluation in the committed frontier runs).
8. Apply kill criterion D1-1 as pre-stated. Write findings next to the protocol; update `STATUS.md` (Active work and the
   Currently running table, machine-role labels only); commit and push the branch without `Co-Authored-By` trailers.

In parallel, D2 Stage A: 3-, 5- and 8-client EuroSAT and Alzheimer federations with a fixed noise standard deviation
across removal (the frontier runner's approach: noise multiplier scaled by the active client count) to test the
distance part of the side channel (kill criterion in `research_directions.md`, D2).

## 5. Owner decisions on the former open questions (2026-10-10)

1. Participant count: **secret**, so every mechanism keeps the noise standard deviation independent of who participates.
2. Aggregation: **equal public weights** for primary runs (sensitivity `C/n`); size weighting only as a secondary
   reference (its removal sensitivity is `2 a_target C`, see `research/threat_specification_review.md` Section 4).
3. Worst-case targets: **top decile by local dataset size**, reported separately everywhere.
4. Second dataset: **Alzheimer MRI** (the paper's 3-client federation and a 48-client homogeneous setting), not CIFAR-10.

The step-by-step first experiment in Section 4 is superseded in detail by work package WP2 of the roadmap (equal
weights, ten configurations); its structure still applies.

## 6. Where everything is

| Path | Contents |
|---|---|
| `research/cia_review_2026-10/agent_roadmap.md` | Binding owner decisions, work packages WP0-WP6, gates, file ownership, escalation, agent prompts |
| `research/cia_review_2026-10/README.md` | Index of this review |
| `research/cia_review_2026-10/project_state.md` | Evidence summary, codebase map, failure analysis |
| `research/cia_review_2026-10/literature_review.md`, `paper_table.csv`, `references.bib` | Literature review and corpus |
| `research/cia_review_2026-10/research_directions.md` | Ranked directions and experiment designs |
| `research/cia_review_2026-10/notes/` | Local-paper notes (both PDFs), broad survey notes, search logs |
| `research/cia_review_2026-10/data/`, `figures/`, `scripts/` | Derived tables, figures, and the scripts that rebuild them from committed logs |
| `research/README.md` | Closed client-specific-noise record (historical) |
| `AGENTS.md` | Conventions: `uv run`, one branch per experiment, protocol before runs, reports only on the owner's call, never `Co-Authored-By` |

Rebuild every derived table and figure of this review from committed result files (no training):

```bash
uv run --with pandas --with scipy --with matplotlib python research/cia_review_2026-10/scripts/analyze_project_history.py
uv run --with pandas --with scipy --with matplotlib python research/cia_review_2026-10/scripts/analyze_direction_evidence.py
uv run --with pandas --with scipy --with matplotlib python research/cia_review_2026-10/scripts/gmip_regime_calculator.py
uv run --with pandas --with matplotlib python research/cia_review_2026-10/scripts/build_taxonomy_figure.py
uv run --with pandas --with matplotlib python research/cia_review_2026-10/scripts/build_direction_scores.py
```

Note: `AGENTS.md` describes `research/` as the closed client-specific-noise record. That remains true for everything
outside `research/cia_review_2026-10/`; the owner may want to update that line once this proposal is accepted.
