# WP2 (D1 Stage A): signature-suppression screen

Branch `feature/signature-suppression`. Written 2026-10-14, before any protocol run; the commit that adds this file is
recorded in `STATE.md`. Specification: `research/cia_review_2026-10/agent_roadmap.md` WP2 (binding), with
`research/cia_review_2026-10/research_directions.md` D1 and `HANDOFF.md` Section 4 for the structure. Changes after
the first protocol run go into the dated Amendments section at the end, committed before the affected runs; no
amendment may change the gate.

## 1. Question

Does a local-training change that suppresses the target's idiosyncratic update component lower client inference
(CIA) leakage at a small accuracy cost, and more than plain regularization (weight decay, fewer local epochs) at the
same accuracy? No DP noise in this screen (D1 Stage A); the noise law comes in WP4.

## 2. Threat model and attack

As in the repository's CIA experiments: a curious participant sees the released global model after every round and
holds the target's shadow sample (the 10% clean shadow split; a second view adds 20%-std pixel noise). Removal
adjacency: the target's whole dataset is either in the federation (IN) or not (OUT); every other partition is unchanged.
Loss statistic: lower shadow loss means IN.

## 3. Design (fixed)

| Item | Value |
|---|---|
| Data and model | EuroSAT, EurosatCNN (289,194 parameters), label-Dirichlet alpha 0.3, 48 canonical clients, layout seed 42 (frontier data module and rules) |
| Training | 50 rounds, FedAvg, 5 local epochs, batch 32, Adam lr 0.001 (the committed frontier hyperparameters); no DP (`privacy = vanilla`), no clipping |
| Seed | 42 (one seed controls layout and training, frontier convention); seed 43 only as the extension in Section 6 |
| Weights | Equal public weights (owner decision) for every configuration except the size-weighted vanilla reference |
| Adjacency | One IN trajectory and one OUT trajectory per target, per configuration |
| Logging (WP0) | Every round 0..50 checkpointed and scored; per target and round: clean and noisy shadow loss, the loss before the round's update, the increment and the shadow-gradient norm at the round's start model; server side the leave-one-out residual norm of every client update (the `norm(e_i)` proxy of D1, P2) |

**Targets**, fixed now from the seed-42 partition (records per client: min 50, median 404, max 1,150):

| Target | Records | Size rank (1 = largest) | Role |
|---|---|---|---|
| 23 | 1,150 | 1 | two largest (top decile) |
| 7 | 1,089 | 2 | two largest (top decile); frontier clean score 0.98 (global DP, seed 42) |
| 3 | 588 | 13 | spread |
| 0 | 304 | 29 | spread |
| 1 | 149 | 46 | spread |

The roadmap names `research/cia_review_2026-10/data/target_idiosyncrasy_vs_cia_score.csv` as the source; it lists only
targets 0-9, of which only client 7 is in the federation's top decile (`ceil(0.1 x 48) = 5` largest: 23, 7, 40, 18, 33).
The owner's worst-case rule (top decile by dataset size) therefore decides the two largest: the federation's two largest
clients. The three spread targets come from the CSV, at evenly spaced positions (ascending positions 0, 3 and 6) of its
eight remaining clients sorted by size: 1, 0, 3 (weights 0.007, 0.014, 0.027, about doubling). Top-decile subgroup: 23
and 7, reported separately everywhere.

**Configurations** (ten, as in the roadmap):

| Name | Change from vanilla (equal weights) |
|---|---|
| `vanilla` | none (reference for the gate) |
| `vanilla-size-weighted` | FedAvg weighted by local dataset size (secondary reference; not a candidate) |
| `weight-decay-5e-4` | Adam `weight_decay = 5e-4` (coupled L2, PyTorch Adam), control |
| `local-epochs-1` | 1 local epoch, control |
| `fedprox-mu-0.01`, `-0.1`, `-1` | FedProx with `(mu/2) ||w - w_t||^2` (see below) |
| `fixed-steps` | every client takes exactly `K = 58` Adam steps per round, cycling through its data |
| `consensus-kl-0.3`, `-1` | local loss `CE + beta KL(p_global(x) || p_local(x))` on the client's own batches; `p_global` from the model received that round, evaluated in eval mode |

Candidates (D1 variants): the three FedProx, `fixed-steps` and the two consensus-KL configurations.

*FedProx convention.* The roadmap fixes `mu in {0.01, 0.1, 1}`. These are the FedProx paper's values for the sum
convention `(mu/2) ||w - w_t||^2`, which this screen uses (`--proximal-normalization sum`). The repository's default
term is the mean squared difference (introduced for the 6.5 M-parameter PaperCNN, where the sum collapsed training at
the paper's `mu = 0.5`); with EurosatCNN's 289,194 parameters it would scale every `mu` down by that factor (penalty
about `mu x 3e-5` for a typical local update norm of 3-4), making all three FedProx configurations indistinguishable
from vanilla. With the sum convention the same update gives a penalty of about `mu x 8`.

*Fixed step budget.* `K = 58` is the mean of the vanilla per-client step counts at seed 42 (5 epochs x
`ceil(train records / 32)`, 2,800 steps over 48 clients, range 10-145), so the federation takes about the same number of
optimizer steps per round as vanilla while every client's `K_i` is equal (D1, P2: equalizes `w_i K_i` with equal
weights).

Trajectories: 10 configurations x 6 = 60.

## 4. Scores and metrics

For target `t` and round `r = 1..50`, `L_IN(t, r)` and `L_OUT(t, r)` are the clean shadow losses of the released model
in the IN trajectory and in the OUT-`t` trajectory of the same configuration, and `g(t, r) = L_OUT - L_IN`.

- **Clean-view score (primary).** Per target, the fraction of rounds with `g > 0` (the review's per-target score,
  `research/cia_review_2026-10/scripts/analyze_project_history.py`); pooled = mean over targets (equivalently over all
  target-round cells). Noisy-view score likewise from the noisy shadow losses (secondary).
- **AUC (primary, pooled over rounds and targets).** Attack score `-(L - m(t, r))` with the reference
  `m = (L_IN + L_OUT) / 2` of the same target and round (an oracle reference, identical for every configuration); IN
  trials are the IN trajectory's cells, OUT trials the OUT trajectories' cells. ROC AUC over all IN-OUT pairs (ties
  count one half).
- **TPR at 5% and 1% FPR** of the same score (secondary; rule `score >= threshold`, the largest TPR whose FPR does not
  exceed the level).
- **Utility.** Server test accuracy at round 50; per configuration the mean over its six trajectories (IN and the five
  OUT, all 47-48-client federations), with the IN value and the range reported. Per-client held-out accuracy at round
  50 of the IN trajectory (fairness check; reported, not gated).
- **Mechanism check (descriptive).** Per target, the median over rounds of its leave-one-out residual norm in the IN
  trajectory, against its score, per configuration (D1, P2).
- **Intervals.** 2,000-resample cluster bootstrap (seed 1729), cluster = target-seed pair (all its rounds, IN and
  OUT); every configuration uses the same resample, so differences are paired. Accuracies enter as point estimates.
- **Subgroups.** Top-decile targets (23, 7) reported separately in every table.

## 5. Gate G2 (copied from agent_roadmap.md Section 3)

| Gate | Pass | On pass | On fail |
|---|---|---|---|
| G2 | Some variant lowers the pooled clean-view score or AUC by at least 0.05 vs vanilla (equal weights) at no more than 1 pp accuracy loss, **and** beats weight decay and local-epochs-1 at matched accuracy | Agent B starts WP4 | D1 stops; findings written; Agent B takes WP6 |

How the statistics are formed (fixed now). `S` is the pooled clean-view score, `A` the pooled AUC, `acc` the mean
round-50 accuracy over a configuration's trajectories; `v` ranges over the six candidates.

1. **Reduction.** `D_S(v) = S(vanilla) - S(v)`, `D_A(v) = A(vanilla) - A(v)`, with bootstrap intervals. Accuracy
   condition: `acc(v) >= acc(vanilla) - 0.01`. Part 1 *holds* when the accuracy condition holds and the lower interval
   end of `D_S(v)` or of `D_A(v)` exceeds 0.05. Part 1 *fails* when the accuracy condition fails, or when the upper
   interval ends of both are at most 0.05.
2. **Beats the controls at matched accuracy**, for each control `c` in {`weight-decay-5e-4`, `local-epochs-1`}: the
   control's attack level at the candidate's accuracy, `X_c(acc(v))`, is read off the segment from
   `(acc(vanilla), X(vanilla))` to `(acc(c), X(c))` by linear interpolation in accuracy, clamped to the nearer
   endpoint outside it. `X` is the statistic through which part 1 holds (`S`; `A` if part 1 holds only through `A`).
   `B_c(v) = X_c(acc(v)) - X(v)`, bootstrapped with accuracies fixed. Part 2 *holds* when the lower interval end of
   `B_c(v)` exceeds 0 for both controls, and *fails* when the upper end is at most 0 for either control.
3. **G2 passes** when some candidate holds both parts. **G2 fails** when every candidate fails part 1 or part 2.
   Otherwise G2 is **inconclusive**: Section 6.

The two best candidates for WP4 are the two with the largest `D_S` among those holding both parts (by `D_A` if they
hold through `A` only).

## 6. Inconclusive results and the extension

If G2 is inconclusive, run the extension once: seed 43 with the layout fixed at seed 42 (`partition_seed = 42`, so the
targets are the same clients and only training randomness changes) for the two best configurations (the two
not-failed candidates with the largest point `D_S`), together with `vanilla` and, when part 2 is what remains open,
the two controls, since every comparison is paired within a seed. At most 5 configurations x 6 = 30 trajectories,
about 11 t-h. Then recompute every statistic with clusters = target-seed pairs over both seeds. If G2 still has no
pass and is not a fail, mark it `inconclusive`, stop and escalate. No further runs to push a result over a threshold.

## 7. Compute

Planned (`runner.py` prints it): 60 trajectories x 50 rounds x 0.74 h / 100 = 22.2 t-h of the 30 t-h cap. On the
compute server WP1's 50-round EuroSAT trajectories took a median 0.37-0.48 h (3-8 clients); 48-client federations add
per-client overhead, `local-epochs-1` costs about a fifth of the training time, and consensus KL adds one forward pass of
the received model per batch. The first trajectories' wall times are recorded in `STATE.md` with a re-estimate; if the
projection exceeds 37.5 t-h (cap + 25%), stop and escalate before continuing.

## 8. Running on the compute server

From the worktree root of `feature/signature-suppression`, with the main clone's environment (worktree skill:
`PYTHONPATH` = worktree, `UV_PROJECT_ENVIRONMENT` = main venv, `uv run --no-sync`). The runner plans unless
`--execute` is given:

```bash
uv run --no-sync python -m results.cia_frontier.signature_suppression.runner           # plan: 60 trajectories
CUDA_VISIBLE_DEVICES=<free gpu> nohup uv run --no-sync python -m results.cia_frontier.signature_suppression.runner \
  --execute --max-parallel-clients 8 > results/cia_frontier/signature_suppression/worker-0.log 2>&1 &
```

Several workers (same command, different GPUs or the same GPU) share the plan: each trajectory is locked while it
trains, locked ones are skipped, completed ones (`complete.json`) are skipped on restart. A trajectory whose folder
shows no file activity for `--stall-minutes` (default 30) is killed with all its descendants and retrained from
scratch by the next run of the same command (the WP1 stall guard). Results go to
`results/cia_frontier/signature_suppression/results/stage_a/<configuration>/<run>/`; each run folder pins its
configuration in `suppression_config.json`. Worker logs are not committed. GPU claims: the shared claims file outside
the repository; every worker is pinned with `CUDA_VISIBLE_DEVICES`.

## 9. Smoke tests before the protocol runs (compute server, scratch output, not results)

The roadmap asks for smoke tests on the owner PC; this agent works on the compute server only, so they run there with
`--smoke-rounds` (at most 5) and an output directory outside `results/`: every configuration's IN trajectory for 2
rounds (all options reach the clients and change training as intended; the per-run JSON records them), and one
configuration end to end (IN and all five OUT, 2 rounds) through `analyze.py`. Results are listed in `STATE.md`.

## 10. Known limitations (stated before the runs)

- One seed and five targets: the bootstrap has five clusters, so intervals are wide and coarse; the extension adds a
  second seed for the decisive configurations only.
- The AUC's reference is the midpoint of the two worlds (an oracle); it ranks configurations on a common scale but
  overstates what a realistic attacker achieves. The clean-view score uses no reference at all.
- No DP: the screen measures how much signal local training leaves in the update, not leakage under the noise law
  (WP4).
- Weight decay is PyTorch Adam's coupled L2 term, not decoupled AdamW.
- The FedProx convention (Section 3) differs from the repository's default for PaperCNN runs; it is recorded in each
  run's JSON (`proximal_normalization`).

## Amendments

### Amendment 1 (2026-10-14, after the 60 seed-42 runs, before any extension run; gate and thresholds unchanged)

The seed-42 analysis (`analysis.json`) left G2 inconclusive: no candidate holds both parts, five fail, and `fixed-steps`
is open on both parts (its `D_S` interval straddles 0.05). Section 6 therefore applies, with these mechanics made
explicit (they only say how Section 6 is computed):

- Extension runs: seed 43 with the seed-42 layout for `fixed-steps` (the only not-failed candidate, so "the two best"
  is one), `vanilla`, and both controls (part 2 is open): 4 configurations x 6 = 24 trajectories, about 6.2 h at the
  measured 0.26 h per trajectory, bringing WP2 to about 21.7 h of the 30 t-h cap. Seed-43 run names carry `-p42`.
- Extension analysis (`analyze.py --extension analysis.json`, written to `analysis_seeds42-43.json`): every statistic
  is recomputed over the configurations that have both seeds, with 10 clusters (target-seed pairs) and the same paired
  bootstrap. Candidates that failed at seed 42 and were not extended keep their seed-42 records and stay failed. G2 then
  passes if `fixed-steps` holds both parts, fails if it fails either, and is otherwise `inconclusive` (stop and
  escalate). `analysis.json` stays the seed-42 record.

### Amendment 2 (2026-10-14, owner decision after the escalation, before any further run; gate and thresholds unchanged)

The owner chose option 3 of `ESCALATION.md`: more seeds for `fixed-steps`. Fixed now:

- Runs: seeds 44 and 45, seed-42 layout, for `fixed-steps`, `vanilla`, `weight-decay-5e-4` and `local-epochs-1` (the
  same four configurations as the extension, so every comparison stays paired): 48 trajectories, about 11.1 h at the
  measured 0.232 h per trajectory. WP2 total about 32.2 h, 7% above the 30 t-h cap (below the 25% escalation line).
- Analysis: `analyze.py --extension analysis.json` over seeds 42-45 (20 clusters), written to
  `analysis_seeds42-45.json`; the other five candidates keep their seed-42 failures. Same gate rule as Section 5.
- This is the last addition. If G2 is still neither pass nor fail, it stays `inconclusive` and goes back to the owner;
  no further seeds without a new owner decision. Expectation stated before the runs: with 20 clusters the interval
  half-width should shrink from about 0.076 to about 0.054, so with a reduction near 0.044 the interval may well still
  straddle 0.05.
