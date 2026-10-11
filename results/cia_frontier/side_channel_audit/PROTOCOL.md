# WP1 (D2 Stage A): noise-level side channel and increment attacker in small federations

Branch `feature/cia-audit` (Agent A). Written 2026-10-11, before any protocol run; the commit that adds this file is
recorded in `STATE.md`. Specification: `research/cia_review_2026-10/agent_roadmap.md` WP1 and
`research/cia_review_2026-10/research_directions.md` D2. Changes after the first protocol run go into the dated
Amendments section at the end, committed before the affected runs; no amendment may change the gate.

## 1. Questions

1. **Side channel.** With a noise standard deviation that does not depend on how many clients participate, does the
   metric-privacy noise level still reveal whether a target took part? Its noise is `sigma_t = r C / d_t`, and the
   maximum pairwise model distance `d_t` depends on who participates (evidence E2: the target is in the maximum pair in
   97-98% of rounds of the 3-client runs, and `d` is 3-9% larger with it).
2. **Increment attacker.** Does a per-round shadow-loss increment likelihood-ratio attacker that knows the noise law
   beat the repository's loss statistic at low false-positive rates?

## 2. Threat model

The attacker is a curious participant. It sees every released global model `theta_0 .. theta_R`, the target's shadow
sample (the 10% clean shadow split; a second view adds 20%-std pixel noise), and the public protocol: mechanism,
clipping norm `C`, noise ratio `r`, number of rounds. It does not see other clients' updates, the participant count or
`d_t`. For reference it can simulate the federation with and without the target (oracle references: the real
trajectories of the same cell trained with the other seeds). The server is trusted.

## 3. Design (fixed)

| Item | Value |
|---|---|
| Datasets | EuroSAT (EurosatCNN, 289,194 parameters) and Alzheimer MRI (PaperCNN, 6.5 M parameters) |
| Partition | Label-Dirichlet alpha 0.3, per-class Dirichlet proportions, multinomial counts, deterministic redraw until every client holds at least the minimum: 10 records for EuroSAT (frontier rule; no client of these federations is near it, the smallest has 1,503), **125 for Alzheimer** (with 10, one client received an empty shadow sample and others 3-6 images; 125 records give at least 10 shadow images). One seed controls partition and training (frontier convention). Owner rule 2026-10-11: non-homogeneous partitions are Dirichlet, also for the 3-client Alzheimer federation |
| Federations | 3, 5 and 8 canonical clients; every client is a target in turn |
| Adjacency | Removal: one IN trajectory per seed, one OUT trajectory per target and seed (all other partitions unchanged) |
| Mechanisms | Global DP and metric privacy (server-side fixed clipping, `C = 5`); vanilla is not run (no noise level, and the loss attackers are measured on the DP runs) |
| Noise | Noise multiplier = `r x active clients`, so `sigma = r C` (global DP) and `r C / d_t` (metric privacy) whatever the participant count. `r = 0.001546` for EuroSAT (the committed frontier noise), `r = 0.01/3` for Alzheimer (`z = 0.01` at 3 clients, the plan-suite runs of evidence E2) |
| Aggregation | Equal weights (owner decision), FedAvg |
| Training | 50 rounds; per dataset the committed frontier hyperparameters: 5 local epochs, batch 32, learning rate 0.001, 20 initialization epochs |
| Seeds | 42, 43, 44 (45 only as the extension below) |
| Logging (WP0) | Every round 0..50 checkpointed and scored; per target and round: clean and noisy shadow loss, the loss before the round's update, the increment, the shadow-gradient norm at the round's start model; per round the released-update norm and parameter count (`releases.json`); server side the applied noise std, pairwise distances, leave-one-out residual norms |

Cells: 2 datasets x 3 client counts x 2 mechanisms = 12; trajectories: 3 seeds x (1 + n) per cell, 228 in total.

## 4. Scores (all oriented: higher means IN)

For target `t` and seed `s` the observed trajectory is the IN trajectory of seed `s` (positive trial) or the OUT-`t`
trajectory of seed `s` (negative trial). References are the same cell's trajectories of the other seeds:
`IN_ref` = their IN trajectories, `OUT_ref` = their OUT-`t` trajectories. Every score is computed per round
`r = 1..50`. Code: `experiments/cia/increment_attack.py`, applied by `analyze.py`.

- **Noise level (blind, primary for question 1).** `sigma_hat_r = ||theta_r - theta_{r-1}|| / sqrt(D)` over all
  floating model entries. Score `-(log sigma_hat_r - mean_{OUT_ref} log sigma_hat_r)`.
- **Oracle noise level (diagnostic, not an attacker).** The same score with the true applied std from the server log;
  an upper bound for what the blind estimate can reach.
- **Increment (primary for question 2).** `delta_r` = clean shadow loss after minus before round `r`, `g_r` = norm of
  the clean shadow-loss gradient at `theta_{r-1}`. Per round `LLR_r = ((delta_r - mu_OUT,r)^2 - (delta_r - mu_IN,r)^2)
  / (2 v_r)` with `mu` the reference means and `v_r = sigma_hat_r^2 g_r^2 + tau^2`, `tau^2` the pooled residual
  variance of the reference increments around their per-round world means (one degree of freedom per world and round
  with two references). Score at round `r` = `LLR_1 + ... + LLR_r`.
- **Loss (the current statistic).** `-(L_r - mean L_r over IN_ref and OUT_ref)`, `L_r` = clean shadow loss at
  `theta_r`: the repository's loss-based score, put on a common scale with the same references.

## 5. Metrics

ROC AUC and TPR at 5% and 1% FPR (rule `score >= threshold`, the largest TPR whose FPR does not exceed the level),
pooled over rounds 1-50 and over targets and seeds. A cluster is one target-seed pair (its IN and OUT trials, all
rounds). Intervals: 2,000-resample cluster bootstrap (seed 1729), resampling clusters within each cell. Top-decile
targets: the `ceil(0.1 n)` largest clients by local dataset size in each federation and seed (here the single largest
client), reported separately everywhere. Primary view: clean shadow sample; the noisy view is secondary. Utility:
server test accuracy at round 50 for every trajectory.

## 6. Gate G1 (copied from agent_roadmap.md Section 3.2)

| Gate | Pass | On pass | On fail |
|---|---|---|---|
| G1 | (a) Noise-level attack AUC > 0.55 for top-decile targets in 3-8 client federations: keep the side-channel claim. (b) Increment attacker beats the current statistic at TPR at 5% FPR: it becomes the primary attacker for WP4 | Continue as stated | (a) drop the side-channel claim, record the negative result; (b) WP4 uses the current statistic as primary and the increment attacker as secondary |

How the statistics are formed (fixed now):

- **G1(a).** Metric privacy, blind noise-level score, top-decile targets, pooled over the six metric-privacy cells
  (both datasets, 3, 5 and 8 clients). Pass when the lower end of the 95% interval of the AUC exceeds 0.55, fail when
  the upper end is at most 0.55. Global DP is reported as a control (its noise is identical IN and OUT, so any
  separation comes from the signal part of the update).
- **G1(b).** Clean view, all targets: the difference "increment minus loss" in TPR at 5% FPR, averaged over the 12
  cells, with the same bootstrap resample for both attackers. Pass when the lower end of the 95% interval exceeds 0,
  fail when the upper end is at most 0.

**Inconclusive and extension.** If an interval straddles its threshold, run the extension once: seed 45 for every
trajectory of the cells that feed the inconclusive part (G1(a): the six metric-privacy cells, about 8.4 t-h; G1(b):
all twelve cells, about 16.9 t-h), then recompute with seeds 42-45. If it still straddles, the part is
`inconclusive`: stop and escalate. No further runs to push a result over a threshold.

## 7. Compute

Measured on the owner PC (Apple MPS, 3-round smoke runs, 2026-10-11): EuroSAT 67 s per round with 3 or 8 clients
(each round covers the whole dataset for 5 epochs), Alzheimer 12-15 s per round. In roadmap units (one 48-client,
100-round EuroSAT trajectory on the reference machine = 1 t-h = 0.74 h) the design costs 50.6 t-h: EuroSAT 42.2,
Alzheimer 8.4 (`runner.py` prints the plan). The roadmap estimate of about 10-15 t-h assumed small federations are
cheap; they are not, because every round still trains on the whole dataset. **Owner decision 2026-10-11: run WP1 on
the compute server** (the roadmap placed it on the owner PC). The cap of 15 t-h is exceeded by more than 25%; the
owner has been asked to confirm the 50.6 t-h budget, and no protocol run starts before that confirmation is recorded
in `STATE.md`.

## 8. Running on the compute server

From the repository root on `feature/cia-audit` (`uv sync` first; EuroSAT and Alzheimer MRI download from the
Hugging Face hub on first use). The runner plans unless `--execute` is given:

```bash
uv run python -m results.cia_frontier.side_channel_audit.runner            # plan: 228 trajectories, 50.6 t-h
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv          # check memory before launching
CUDA_VISIBLE_DEVICES=0 nohup uv run python -m results.cia_frontier.side_channel_audit.runner --execute \
  --max-parallel-clients 8 > results/cia_frontier/side_channel_audit/worker-0.log 2>&1 &
```

Several workers (one per GPU, same command, different `CUDA_VISIBLE_DEVICES`) share the plan: each trajectory is
locked while it trains, locked ones are skipped, and completed ones (`complete.json`) are skipped on restart, so a
crashed worker is restarted with the same command. A trajectory whose folder shows no file activity for
`--stall-minutes` (default 30) is killed with its whole process group and left without `complete.json`; the worker
continues, and the next run of the same command retrains it from scratch (never a partial mix). Reason: on the owner PC
Ray occasionally deadlocks while forking its dashboard process (a forked child blocks in a Python at-fork hook and
never execs; the server then waits forever for client actors). Repeat the command until the plan (run without
`--execute`) reports `complete` equal to `trajectories`. Shard by `--datasets`, `--clients`, `--privacy` or `--seeds`
when useful. Logs (`worker-*.log`) are not committed. Record each launch in `STATE.md` and in this agent's rows of the
`STATUS.md` Currently running table (machine-role label only).

After all 228 trajectories are complete:

```bash
uv run python -m results.cia_frontier.side_channel_audit.analyze results/cia_frontier/side_channel_audit/results \
  --out results/cia_frontier/side_channel_audit/analysis.json
```

## 9. Smoke tests before the protocol runs (owner PC, at most 5 rounds, scratch output, not results)

- EuroSAT 3 and 8 clients, metric privacy, 3 rounds; Alzheimer 3 and 8 clients, metric privacy, 3 rounds (timing).
- End-to-end runner smoke: Alzheimer, 3 clients, both mechanisms, seeds 42-43, 3 rounds (`--smoke-rounds 3`), then
  `analyze.py` on its output. Results are listed in `STATE.md`.

## 10. Known limitations (stated before the runs)

- Two references per world (leave-one-seed-out) make the reference means noisy; the increment attacker is therefore a
  weak version of a LiRA-style attacker with many simulated federations.
- `v_r` counts the release noise twice (once through `sigma_hat^2 g^2`, once inside `tau^2`): conservative.
- The blind noise estimate includes the update signal; with `D` in the hundreds of thousands to millions its
  chi-square fluctuation is small, but the signal bias differs between IN and OUT (that difference is what the global-DP
  control measures).
- Metric privacy at 3-8 clients may become unstable (small `d_t`, large noise); accuracies are reported for every
  trajectory and nothing is excluded.

## Pre-run revision (2026-10-11, before any protocol run; gate and thresholds unchanged)

The end-to-end smoke (Section 9) showed that the blind noise-level score also separates IN from OUT under global DP,
whose noise is identical in both worlds (its oracle score, built from the true noise std, sits at exactly 0.5). The
released update is `noise + mean of the client updates`, and the mean changes when a client is removed; with `D`
between 0.3 and 6.5 million the noise norm is so concentrated that this signal difference is visible. The blind
estimate therefore measures two channels: the **update-norm channel** (both mechanisms) and the **noise-level channel**
(metric privacy only, through `d_t`). G1(a) stays as defined in Section 6 (it is the roadmap's attack). Fixed now for
the write-up:

- The side-channel claim is attributed to metric privacy's noise level only if the oracle noise-level score for
  metric privacy (top-decile targets, pooled as in G1(a)) also has an AUC interval above 0.55, and the blind
  metric-privacy AUC exceeds the blind global-DP AUC on the same targets. Otherwise a G1(a) pass is reported as an
  update-norm channel common to both mechanisms.
- The blind global-DP noise-level AUC (all targets and top decile) is reported as the size of the update-norm channel.
- These numbers are smoke diagnostics (3 rounds, 2 seeds), not results; they are not used anywhere else.

## Amendments

None.
