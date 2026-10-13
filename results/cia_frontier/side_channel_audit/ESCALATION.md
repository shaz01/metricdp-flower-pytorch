# Escalation: WP1 cost is about 3.4 times its cap

Raised 2026-10-11 by Agent A before any protocol run. Rule: agent_roadmap.md Section 4.4 item 2 (projected cost
exceeds the package cap by more than 25%).

## Evidence

Timing smoke runs on the owner PC (Apple MPS), 3 rounds each, frontier hyperparameters (5 local epochs, 20
initialization epochs), equal weights, metric privacy:

| Federation | Seconds per round |
|---|---|
| EuroSAT, 3 clients | 67 |
| EuroSAT, 8 clients | 66 |
| Alzheimer MRI, 3 clients | 12-14 |
| Alzheimer MRI, 8 clients | 12-15 |

Every round trains on the whole dataset for 5 local epochs whatever the number of clients, so small federations are
not cheaper per round than the 48-client frontier. The specified WP1 design (2 datasets x 3, 5, 8 clients x 2
mechanisms x 3 seeds, every client a target, 50 rounds) is 228 trajectories: **50.6 t-h** (EuroSAT 42.2, Alzheimer 8.4)
against a cap of 15 t-h, about 137 h of wall time on the owner PC. The research document's estimate (about 10 t-h)
assumed small federations were cheap.

## Options put to the owner

1. 20 rounds (the paper's CIA horizon) and 3 + 8 clients: about 13.7 t-h, within the cap.
2. 20 rounds, 3 + 5 + 8 clients: about 20 t-h.
3. The design as specified (50 rounds) with a raised cap, or with the EuroSAT part on the compute server.
4. 50 rounds, EuroSAT at 3 clients only, Alzheimer at 3, 5, 8: about 17 t-h.

## Decision so far (owner, 2026-10-11)

"Maybe we can move to the server, let's get ready everything for server movement": WP1 moves to the compute server
with the specified design, and Agent A prepares everything for it (protocol, runner, analysis, tests, smoke runs,
launch instructions). **Open:** explicit confirmation of the 50.6 t-h budget (3.4 times the cap). No protocol run
starts before it is recorded in `STATE.md`.

## Decision (owner, 2026-10-11)

The owner confirms the **50.6 t-h budget** for WP1 as written in `PROTOCOL.md` (design unchanged: 228 trajectories,
50 rounds, seeds 42-44; the declared seed-45 extension stays conditional on an inconclusive gate part). WP1 runs on the
compute server on the GPU the owner assigned (one GPU, shared with another user's job; see `STATE.md`). Escalation
resolved; WP1 unblocked.

---

# Escalation 2: measured cost projects to 65-71 h, above the 63.3 h threshold

Raised 2026-10-11 by Agent A after the first protocol trajectories on the compute server. Rule: the owner's WP1 launch
instruction (re-estimate after the first trajectory; stop and escalate if the projection exceeds 50.6 t-h by more than
25%, i.e. 63.3) and agent_roadmap.md Section 4.4 item 2. Workers are stopped; the two completed trajectories are valid
protocol runs and are kept (the runner skips them on restart).

## Evidence

Compute server, GPU 1, shared with another user's training job (12.4 GB, 60-80% utilization), two WP1 workers from
17:46. From `complete.json`:

| Trajectory (EuroSAT, 3 clients, global DP, seed 42) | Training | Evaluation | Targets scored |
|---|---|---|---|
| IN | 1,475.0 s | 696.2 s (232 s per target; the other worker was training) | 3 |
| OUT-0 | 1,461.6 s | 143.7 s (evaluation ran with the GPU otherwise free of WP1 work) | 1 |

The plan prices a 50-round EuroSAT trajectory at 0.37 h = 1,332 s in total. Training alone takes about 1,468 s (1.10 x),
and evaluation comes on top: every scored target is evaluated at all 51 checkpoints with clean and noisy views and
shadow-gradient norms. An IN trajectory scores all n targets, an OUT trajectory only its own, so a dataset has
2 mechanisms x 3 seeds x 2n target evaluations = 192 for n = 3, 5, 8.

| Part | Basis | Projection |
|---|---|---|
| EuroSAT training | 114 trajectories x 1,468 s | 46.5 h |
| EuroSAT evaluation | 192 target evaluations x 144-232 s | 7.7-12.4 h |
| Alzheimer | not measured yet; the plan's relative cost 0.2 applied to both parts | 10.8-11.8 h |
| **Total** | | **65.0-70.6 h = 1.28-1.40 x 50.6 h** (threshold 63.3 h) |

Uncertain in both directions: Alzheimer is unmeasured on this machine, the per-target evaluation cost depends on GPU
sharing, and 5- and 8-client training was not timed (PROTOCOL.md Section 7: per-round cost on the owner PC was the same
at 3 and 8 clients).

Observed inefficiency (no code changed): `experiments/cia/cia.py::eval_model_with_gradients` recomputes
`aggregate_loss` over the whole server test set once per target per checkpoint, although it does not depend on the
target. Computing it once per checkpoint would give identical outputs and trim evaluation (at most the 7.7-12.4 h
EuroSAT evaluation term shrinks), not enough on its own to be sure of staying under 63.3 h.

## Options

1. **(Recommended)** Keep the design and raise the budget to 71 h (the upper projection), with the usual 25% margin
   (escalate above 89 h) and a re-check after the first Alzheimer trajectory. No code change. Calendar time with two
   workers on the shared GPU 1: about 1.5-2 days.
2. As option 1, plus the identical-output evaluation fix (Agent A owns `experiments/cia/` and the frontier runner;
   a unit test asserts equal rows with and without it). Saves a few hours; adds a code change mid-protocol.
3. Keep about 63 h by amending the design (owner decision, dated amendment before further runs), e.g. drop one seed
   from the 5-client cells. Changes what G1 pools over; the gate rule itself is unchanged.

## Decision (owner, 2026-10-11)

The owner replied "lets start" to this escalation, which recommended option 1; recorded as **option 1**: design
unchanged, no code change, budget raised to **71 h** (projection upper end), escalation above **89 h** (71 x 1.25),
cost re-checked after the first Alzheimer trajectory. Resumed on GPU 1 of the compute server with two workers
(8-client cells at most 4 parallel clients per worker; execution setting only, see `STATE.md`).

# Escalation 3: G1(b) is inconclusive; its declared extension would take WP1 past the 89 h line

Raised 2026-10-13 by Agent A after all 228 protocol trajectories completed and `analyze.py` ran
(`analysis.json`, seeds 42-44). Rule: the owner's WP1 instruction (run the declared extension once if a part is
inconclusive) together with the cost rule (escalate above 89 h, `ESCALATION.md` Escalation 2 decision) and
agent_roadmap.md Section 4.4. Nothing is running.

## Gate G1 at seeds 42-44 (`analysis.json`)

| Part | Statistic | Value [95% interval] | Decision |
|---|---|---|---|
| G1(a) | blind noise-level AUC, metric privacy, top-decile targets, pooled over 6 cells | 0.838 [0.779, 0.902] | pass |
| attribution | oracle noise-level AUC, metric privacy, same targets | 0.839 [0.781, 0.903] | above 0.55 |
| attribution | blind noise-level AUC, global DP, same targets | 0.763 [0.706, 0.824] | below the blind metric-privacy AUC |
| G1(b) | increment minus loss, TPR at 5% FPR, clean view, all targets, mean over 12 cells | +0.074 [-0.060, +0.116] | inconclusive |

G1(a) passes and, by the pre-run attribution rule, is credited to metric privacy's noise level. G1(b) straddles 0.
PROTOCOL.md Section 6 then prescribes the extension once: seed 45 for every trajectory of all twelve cells, then
recompute with seeds 42-45.

## Cost

| Item | Measured or projected |
|---|---|
| Protocol run, 228 trajectories (sum of training + evaluation seconds in `complete.json`) | 75.5 h |
| G1(b) extension, 76 trajectories (seed 45, 12 cells = exactly one third of the design) | about 25.2 h (75.5 / 3) |
| Total with the extension | about 100.7 h, above the 89 h escalation line (budget 71 h) |

Wall time with both compute-server GPUs free (as now): about 7-9 h with three or four workers.

## Options

1. **(Recommended)** Run the extension as declared, once: seed 45, all twelve cells, 76 trajectories; budget raised
   to about 101 h (escalate above 126 h). No code or protocol change. If G1(b) still straddles 0 at seeds 42-45, it is
   `inconclusive` and Agent A stops and escalates (no further runs). G1(a) stays decided at seeds 42-44.
2. Do not run the extension: record G1(b) as `inconclusive` at seeds 42-44. Under the roadmap's G1 rule WP4 then uses
   the current statistic as primary and the increment attacker as secondary (the on-fail branch, since G1(b) did not
   pass).
