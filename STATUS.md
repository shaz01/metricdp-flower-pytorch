# Project Status

**Branch:** `master`
**Last updated:** 2026-11-02 10:45, v3 Agent P: P1 complete, FINDINGS written (G-P1: the published protection claim fails in its own setting; Table 13 a single draw), awaiting the owner; P2 passed G-P2a (vanilla CIFAR-10 AUC 0.998), pilot done, levels chosen, stage b approved in full (ROADMAP decision 6) and queued. 2026-10-31 12:55, v3 Agent P: G-P0 passed; owner raised P2's and P3's caps (ROADMAP decision 5); P1 (`feature/cia-v3-paper-setting`) and P2 stage a (`feature/cia-v3-cifar10`) running on shared GPUs. 2026-10-30, v3 P6 memo stage (design memo, ROADMAP decisions 2-3) closed by the owner and merged into `master`; screens not open. 2026-10-30, v3 P5 (client inference in the formal regime) closed by the owner and merged into `master` (`reports/cia_v3_p5_formal_cia.md`). 2026-10-30, roadmap v3 approved by the owner (Section 1 defaults) and merged into `master` (`research/cia_roadmap_v3/`); v2 results handoff corrected (`research/cia_roadmap_v3/HANDOFF_AUDIT.md`). 2026-10-30, C1 closed and merged (decision 17; `reports/cia_v2_composite.md`); roadmap v2 complete, results handoff `research/cia_roadmap_v2/RESULTS_HANDOFF.md`. 2026-10-29, B5 and B1 merged (`feature/cia-v2-mechanisms`); C1 opened (decision 14). 2026-10-26, AZ (Alzheimer replication) merged. 2026-10-23, B4 confirmatory run merged; B5 closed (report on `feature/cia-v2-mechanisms`); C1 waits (decision 12). 2026-10-20, owner decisions 7-11 (ROADMAP.md Section 10). CIA v2 Agent A packages M0, M1, M2, A1 and A2 merged into `master` (closed by the owner; reports `reports/cia_v2_m0_measurement.md`, `cia_v2_m1_shared_noise.md`, `cia_v2_m2_frontier.md`, `cia_v2_a1_matched_accuracy.md`, `cia_v2_a2_side_channel.md`). 2026-10-17, CIA v2 B4 (formal regime) and the client-level accountant merged into `master` (Agent C, closed by the owner; report `reports/cia_v2_formal_regime.md`); Agent A's M0 (G-M0 passed) came along as a dependency of B4's audit. CIA roadmap v2 approved (`research/cia_roadmap_v2/`). 2026-10-16: CIA programme closed: WP3, WP6, the consensus-KL follow-up and the programme synthesis
(`research/cia_review_2026-10/SYNTHESIS.md`) merged into `master`; the owner is choosing new directions from the
synthesis. 2026-10-15: WP0-WP2 merged, D1 stopped. 2026-10-10: CIA literature review and agent roadmap added
(`HANDOFF.md`, `research/cia_review_2026-10/`)
(see `git log` for anything more recent)

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog. Exception: the Active work
section (including the Currently running table) updates more often, at "worth a commit"
granularity — see `AGENTS.md`'s "Working across machines" section.

## Active work

**CIA roadmap v3: approved by the owner 2026-10-30** (`research/cia_roadmap_v3/ROADMAP.md` Section 1 defaults; decision 1 in Section 10). Agent P (P0-P4, R1) on the compute server, Agent F (P5, P6 memo) on the owner PC; prompts in ROADMAP.md Section 7.
- **Results:**
  - v2 handoff audited and corrected: 8 corrections (C1-C8).
  - Literature and novelty check of Q1-Q9.
  - Zero-GPU re-analysis: the composite's win over metric privacy survives a sampling-aware attacker (-0.071 and
    -0.082 AUC); early-stopped global DP leaks more than subsampling wherever both reach the same accuracy.
  - Paper options: the hybrid paper is recommended.
- **Proposed v3:** packages P0-P6 and R1, about 120 GPU-h expected for the paper packages (148 cap).
- **Running:** nothing yet (agents not started).
- **Agent F (owner PC), 2026-10-30:**
  - **P5: closed by the owner and merged into `master` 2026-10-30** (branch deleted; report
    `reports/cia_v3_p5_formal_cia.md`, data `results/cia_v3/formal_cia/`). 832 head runs, 0.32 CPU-h.
    - G-P5: global DP at `mu = 2` is not protective at 48 clients (AUC 0.639 [0.598, 0.680]); it is at 480
      pseudo-clients (0.480 [0.438, 0.523]).
    - At matched accuracy, metric privacy is equivalent to global DP at 48 clients. Subsampling shows no detectable
      difference, and leaks more under the sampling-aware attacker (+0.090).
  - **P6 memo stage: closed by the owner and merged into `master` 2026-10-30** (branch deleted). The memo is
    `research/cia_roadmap_v3/label_channel/DESIGN.md`; LDPM and RR-Label were read in full. ROADMAP.md decisions 2-3
    set the design:
    - no fill-in, a fixed per-class subset;
    - a head screen (CPU) first, the CNN only if it passes;
    - RR-Label plus logit adjustment (reported) as baselines.

    The screens wait for the paper packages to close and get a new branch.
- **Agent P (compute server), 2026-10-30:**
  - **P0 (`feature/cia-v3-infra`; `results/cia_v3/infra/STATE.md`): G-P0 passes (2026-10-31).** New runner options (defaults change no committed manifest): `--checkpoint-every`, `--in-count 8|24`,
    `--dataset cifar10` (label-Dirichlet 0.3, 48 slots, candidates fixed from the committed seed-42 partition
    summary), and the 3-client Alzheimer paper-protocol runner (`results/cia_v3/infra/paper_runner.py`, separate seeds
    per world).
    - G-P0 so far: suite green (944 passed; skips need MPS), `reanalyze.py` reproduces C1's gate numbers, the CIFAR-10
      partition is deterministic, the IN-count design passes its checks; the four cost runs are recorded (1.3 GPU-h, on GPU 1 shared with another
      user's job under ROADMAP v3 decision 4; peak VRAM 3.6 GB).
    - **Escalation (`results/cia_v3/infra/ESCALATION.md`):** CIFAR-10 runs take 0.36-0.54 run-hours, so P2 projects to
      about 116 GPU-h (cap 85) and P3, with every-round scoring, to about 32 GPU-h (cap 20). Both are above 1.25x;
      options are listed. P1 projects to about 5 GPU-h (cap 40).
    - Owner decision 5 (2026-10-31): caps raised to P2 120 and P3 35 GPU-h.
  - **P1 (`feature/cia-v3-paper-setting`; `results/cia_v3/paper_setting/FINDINGS.md`): complete, awaiting the owner.**
    The paper's 3-client Alzheimer setting, 20 IN/OUT pairs per configuration with separate seeds per world (220 runs,
    about 15 GPU-h). **G-P1: "the published protection claim fails in its own setting"**: the calibrated attacker scores
    1.000 for global DP and metric privacy at the paper's `z = 0.01`, and 0.973 / 0.985 at `z = 0.024`, where global DP
    matches the paper's accuracy (0.672 against 0.637). At `z = 0.024` the paper's own per-pair statistic for global DP
    ranges 0.366-0.646 (median 0.497), so **"Table 13 is a single draw from a wide distribution"** also holds.
  - **P2 (`feature/cia-v3-cifar10`; `results/cia_v3/cifar10/STATE.md`):** G-P2a passed (vanilla all-candidate AUC 0.998
    at `R = 8`; A1's -0.000 [-0.019, 0.018] reproduced first). Pilot done: targets `t2` 0.554, `t4` 0.326; stage-b levels
    metric privacy 2x/4x/8x, composite r 0.000773/0.003092/0.006184. Stage b (312 runs, about 254 run-hours) approved
    in full by the owner (ROADMAP decision 6, cap about 290 GPU-h); first half queued, report to the owner after it.

### Currently running

Owner decisions 7-17 (2026-10-20 to 2026-10-30, all by the agent that ran Agent A's packages): AZ, B4's confirmatory run, B5, B1 and C1 are closed and merged (see "What's established"). **C1 closed and merged** (decision 17, report `reports/cia_v2_composite.md`): the composite (client subsampling at q = 0.25 with logit adjustment) passes G-C1 against metric privacy at both targets and holds against both adaptive attackers; the gain comes from subsampling. **Roadmap v2 is complete;** the owner runs the literature and novelty check separately from `research/cia_roadmap_v2/RESULTS_HANDOFF.md` and has not yet chosen between an audit paper and a new-defense paper. Nothing is running. B2 and B3 stay deprioritized or on hold. GPU 0 is left to other users. Roadmap v2 used about 190 GPU-h (about 178 before C1, C1 about 11.5 of its 15 GPU-h cap).

Update this table whenever a machine picks up new work: add a row, edit the Status column
in place (e.g. `running` -> `done`), and leave a finished row for one update cycle before removing
it, so machine-to-results provenance isn't lost; see `AGENTS.md`'s "Working across machines"
section.

| Command | What | Status |
| --- | --- | --- |
| 2 x `v3-direct.sh` -> `p2-stage-b.sh` (compute server, tmux `p2-b1`, `p2-b2`; branch `feature/cia-v3-cifar10`) | P2 stage b, first half: 152 runs (replicates 0-15; vanilla 8-15), per-run locks | queued 2026-11-02 10:37 for GPU 1 (shared, decision 4); waiting for 25 GB of free host RAM |
| `p2-pilot.sh` (compute server, tmux `p2-pilot`, `p2-pilot-w2`; branch `feature/cia-v3-cifar10`) | P2 accuracy pilot, 26 runs | done 2026-11-02 08:51 (GPU 1, shared) |
| `p2-stage-a.sh` (compute server, tmux `p2`; branch `feature/cia-v3-cifar10`) | P2 stage a: CIFAR-10 vanilla, replicates 0-7 | done 2026-11-01 03:01 (GPU 1, shared); G-P2a passed |
| `p1-run.sh`, `p1-extension.sh` (compute server, tmux `p1`, `p1-ext`; branch `feature/cia-v3-paper-setting`) | P1: pilots (20 runs), main runs (120), extension at `z = 0.024` (80) | done 2026-11-02 03:32 (GPUs 0 and 1, shared; about 15 GPU-h) |
| `p0-cost-wait.sh` -> `p0-cost-run.sh` (compute server, tmux `p0-wait` / `p0-cost`; branch `feature/cia-v3-infra`) | P0 cost runs: CIFAR-10 vanilla and composite, one paper-protocol pair, one 51-checkpoint EuroSAT composite (one worker, 4 parallel clients) | done 2026-10-31 00:41 (1.3 GPU-h on GPU 1, shared with another user's job, ROADMAP v3 decision 4; peak VRAM 3.6 GB); G-P0 passes |
| `formal_cia run` (owner PC, CPU; branch `feature/cia-v3-formal-cia`, merged) | P5: 832 head runs | done 2026-10-30 (0.32 CPU-h); closed and merged |
| 2 x `c1-chain.sh` (compute server, GPU 1 shared by owner approval, 4 parallel clients; branch `feature/cia-v2-composite`) | C1 main runs: composite at 4 levels, 128 runs, surrogates and the defense-aware score | done 2026-10-30 06:24 (about 11.5 GPU-h incl. pilot); G-C1 passes; merged |
| 3 x `b1-chain.sh` (compute server, GPU 1; branch `feature/cia-v2-mechanisms`, merged) | B1 main runs, 256 runs | done 2026-10-27 20:42 (about 9.9 GPU-h); G-B1 passes for q = 0.25 |
| 3 x `az-a1-chain.sh` (compute server, GPU 1; branch `feature/cia-v2-alzheimer`) | A1 on Alzheimer, 192 runs | done 2026-10-25 12:50 (about 40 GPU-h); no robust difference |
| 2 x `b5-worker.sh` (compute server, GPU 1 shared by owner approval; branch `feature/cia-v2-mechanisms`) | B5 stage B: logit adjustment at 4x, 32 runs | done 2026-10-21 20:29 (about 3.7 GPU-h); fails G-B5 |
| 2 x `az1-cal-worker.sh` (compute server, GPU 1 shared by owner approval; branch `feature/cia-v2-alzheimer`) | AZ calibration, alpha 1.0, 100 rounds, 18 runs (accuracy only) | done 2026-10-21 21:23 |
| 3 x `az1-cal-worker.sh` (compute server, GPU 1 idle and unclaimed at launch; branch `feature/cia-v2-alzheimer`) | AZ calibration batch 2, 20 runs (accuracy only) | done 2026-10-22 02:53 (one private-median run collapsed) |
| 3 then 2 x `az-m2-chain.sh` (compute server, GPU 1, shared by owner approval; branch `feature/cia-v2-alzheimer`) | M2 on Alzheimer, 128 runs | done 2026-10-23 20:55 (first launch's runs with failed clients quarantined); G-M2 useful regime at 0.0025 |
| `frozen_head.runner confirm_public --execute --workers 8` (compute server, CPU only; branch `feature/cia-v2-formal-confirm`) | B4 confirmatory run, 480 head-only runs (layout 44, 20 and 50 public images) | done 2026-10-20 10:25 (0.68 CPU-h); confirmed |
| `az-wait.sh` then `az-cal-worker.sh` (compute server, GPU 1 idle and unclaimed at launch; branch `feature/cia-v2-alzheimer`) | AZ Alzheimer accuracy calibration batch 1, 18 runs (design seed 3, accuracy only) | stopped 2026-10-20 11:35 after 8 runs (vanilla 0.55 accuracy; escalated) |
| `b5-wait.sh` then 2 x `b5-worker.sh` (compute server, first idle unclaimed GPU; branch `feature/cia-v2-mechanisms`) | B5 stage A at 2x: cap 326, cap 182, logit adjustment, 96 runs | done 2026-10-20 19:44 (about 8.8 GPU-h); logit adjustment passes G-B5 |
| `randomized_inclusion.runner --config cost-gdp-r1x ... --replicate-subset 0` (compute server, GPU 1, shared by owner approval) | CIA v2 M0 cost run, 1 randomized-inclusion run | done 2026-10-17 18:51 (0.20 GPU-h) |
| `cia_v2.measurement.m1.runner --execute` (compute server, GPU 1, two workers) | CIA v2 M1, 12 trajectories | done 2026-10-17 20:51 (about 2.2 GPU-h; G-M1 confirmed) |
| `cia_v2.side_channel.runner --stage pilot --multiple 1 ...` (compute server, GPU 1, shared by owner approval; branch `feature/cia-v2-side-channel`) | CIA v2 A2 pilot at 1x, 32 trajectories | done 2026-10-18 08:44; cost stop rule triggered, owner chose 4 workers per GPU (`results/cia_v2/side_channel/ESCALATION.md`) |
| `a2-wait.sh` then 4 x `a2-worker.sh` (compute server, first idle unclaimed GPU; branch `feature/cia-v2-side-channel`) | CIA v2 A2 pilot at 4x, 32 trajectories | done 2026-10-18 19:29; Alzheimer 3 and 5 qualify at 4x, cost 0.12 GPU-h per trajectory, projection 34-36 GPU-h (under the stop threshold) |
| 4 x `a2-worker.sh` (compute server, GPU 1, idle and unclaimed at launch; branch `feature/cia-v2-side-channel`) | CIA v2 A2 16x pilot (Alzheimer 8, EuroSAT 3; 16 trajectories), then main runs at 4x (Alzheimer 3 and 5; 64 trajectories) | done 2026-10-19 04:42 (about 7.5 GPU-h); collapsed rounds escalated |
| `randomized_inclusion.runner --config {vanilla,gdp-r1x,gdp-r4x,gdp-r16x} --design-seed 2` replicates 0-15 (compute server, GPU 1, two workers, shared by owner approval; branch `feature/cia-v2-frontier`) | CIA v2 M2 stage 1, 64 runs | done 2026-10-18 03:52 (fill-in rule adds 2x and 8x) |
| `m2-chain.sh` (compute server, GPU 0, two workers; branch `feature/cia-v2-frontier`) | CIA v2 M2 stage 2 plus 2x and 8x, 128 runs | done 2026-10-19 00:51 (192 runs, about 19 GPU-h); G-M2 no useful regime |
| `a1-cal-worker.sh` (compute server, both GPUs, idle and unclaimed at launch; branch `feature/cia-v2-frontier`) | CIA v2 A1 accuracy calibration, 20 runs (design seed 3, accuracy only) | done 2026-10-19 21:32 (about 2.3 GPU-h); levels fixed in `a1/PROTOCOL.md` |
| `a1-chain.sh` (compute server, GPUs 0 and 1, two workers each, idle and unclaimed at launch; branch `feature/cia-v2-frontier`) | CIA v2 A1 main runs, 7 configurations x 32 = 224 runs | done 2026-10-20 08:26 (about 21.7 GPU-h); G-A1 recorded |

## What's established on `master`

- **CIA v2 C1: the composite beats metric privacy at matched accuracy (2026-10-30)** (`reports/cia_v2_composite.md`; data `results/cia_v2/composite/`). Client subsampling at q = 0.25 with logit adjustment (tau 1) has an all-candidate AUC 0.095 [-0.137, -0.053] below the paper's metric privacy at 0.740 accuracy and 0.108 [-0.163, -0.053] below at 0.660 (two-way bootstrap also below 0); also about 0.08-0.10 below private median and global DP; a defense-aware and a label-only attacker recover no leakage. At matched accuracy it is indistinguishable from subsampling alone: logit adjustment adds 3-10 pp accuracy at the same noise, not privacy. Formal guarantee vacuous or not computable. Roadmap v2 complete; results handoff `research/cia_roadmap_v2/RESULTS_HANDOFF.md`.
- **CIA v2 B5 and B1: two defenses against the label channel (2026-10-29)** (`reports/cia_v2_b5_large_clients.md`, `reports/cia_v2_b1_subsampling.md`; data `results/cia_v2/mechanisms/`). EuroSAT's client-inference leakage is mostly label-distribution inference (a label-matched public surrogate attacker is within 0.009-0.025 AUC of the true one). Logit adjustment in local training lowers the AUC by 0.100 [-0.152, -0.049] at 2x with +1.1 pp accuracy (fails at 4x); client subsampling at q = 0.25 lowers it by 0.10 at 0.740 and 0.06 at 0.660 accuracy against global DP at matched accuracy (q = 0.5 fails). Neither strengthens the formal guarantee. Code: `metricdp_pytorch/subsampled_dp_strategy.py`, `--contribution-cap`, `--logit-adjustment-tau`, `--privacy subsampled-global-dp`.

- **CIA v2 AZ, Alzheimer replication of M2 and A1 (2026-10-26)** (`reports/cia_v2_alzheimer_replication.md`; data `results/cia_v2/alzheimer/`): at alpha 1.0 and 100 rounds (alpha 0.3 did not train), leakage is near chance for every configuration (AUC 0.49-0.56, vanilla 0.564). G-M2 finds a useful regime at r = 0.0025 for that reason; G-A1 finds no robust difference (one nominal "better" for metric privacy fails the two-way bootstrap). Also merged: `--require-all-clients` (a run stops at the first failed or missing client), the run check, dataset profiles in the randomized-inclusion runner, the private-median collapse guard.

- **CIA v2 B4 confirmatory run (2026-10-23)** (`reports/cia_v2_formal_confirm.md`; data `results/cia_v2/formal_regime/confirm/`): on a fresh layout, public subsets and seeds, all 13 scarce-public settings that beat the strongest legitimate baseline in B4's sweep do so again (20 public images: 14-26 pp at `mu_total <= 2`; 50: 10-14 pp). B4's scarce-public positive is confirmed for this setup.

- **CIA v2 Agent A: measurement rebuilt, global DP's real frontier, and the audit of metric privacy (2026-10-20)**
  (reports `reports/cia_v2_m0_measurement.md`, `cia_v2_m1_shared_noise.md`, `cia_v2_m2_frontier.md`,
  `cia_v2_a1_matched_accuracy.md`, `cia_v2_a2_side_channel.md`; data `results/cia_v2/{measurement,frontier,side_channel}/`).
  - **M0 (G-M0 pass):** separate layout, init, client and noise seeds; randomized inclusion with 32 candidates and
    `R = 32`; a calibrated LiRA-style loss attacker; the 97.5% pair-jackknife gate interval.
  - **M1 (G-M1 confirmed):** with independent seeds per world, v1's 4x paired score falls from 0.792 to 0.516; v1's flat
    noise response was a shared-noise artifact.
  - **M2 (G-M2: no useful regime, escalated):** global DP keeps the attacker at AUC 0.70-0.72 wherever accuracy is
    0.70 or above (vanilla 0.723, 2x 0.702 at accuracy 0.740); 0.593 only at 16x, accuracy 0.199.
  - **A1 (G-A1):** at matched accuracy neither the paper's metric privacy nor private-median metric privacy leaks
    detectably less than global DP; the paper's mechanism is equivalent at 0.740 (difference -0.000 [-0.019, 0.018]).
  - **A2 (G-A2: channel size recorded):** metric privacy's noise level is visible to a blind observer in small Alzheimer
    federations (AUC 1.000 and 0.875), but at the same noise it also leaks fully through the loss; the strong claim
    fails (one cell, through an attacker artifact).

- **CIA v2 B4: formal client-level regime with frozen features and a small head, plus the accountant (2026-10-17)**
  (`reports/cia_v2_formal_regime.md`; data `results/cia_v2/formal_regime/`). G-B4 fails (owner accepted): a linear head
  on frozen ResNet-18 ImageNet features of EuroSAT gains at most 3.49 [2.37, 4.64] pp over the 1,350-image public-only
  head at `mu_total <= 8` (extended grid 3.81). At 480 (pseudo-)clients, `mu = 1` still gains 2.1-2.4 pp. Recorded as
  exploration: with 20-100 public images, `mu <= 2` gains 5-26 pp, robust to a stronger baseline (fails at 500). The
  randomized-inclusion audit never exceeds the GDP bound (`mu = 1`: AUC 0.52-0.61 vs 0.760; no noise 0.84-0.98). The
  accountant `metricdp_pytorch/accounting.py` (Poisson client sampling, PLD/RDP via `dp-accounting==0.6.0`, now a
  project dependency) reproduces TF Privacy and Abadi et al. epsilons; notes in
  `research/cia_roadmap_v2/theory/accounting.md`.

- **CIA programme WP1 and WP2 (2026-10-15)** (`results/cia_frontier/side_channel_audit/FINDINGS.md`,
  `results/cia_frontier/signature_suppression/FINDINGS.md`): with a fixed noise std in 3-8 client federations, metric
  privacy's noise level reveals the largest client's participation (AUC 0.838 [0.779, 0.902]); a per-round increment
  attacker does not beat the loss statistic at 5% FPR. Signature-suppressing local training (FedProx, fixed local
  steps, consensus KL) does not cut CIA leakage by 0.05 within 1 pp accuracy on EuroSAT without DP (G2 inconclusive,
  D1 stopped by owner decision). No `reports/` write-up yet (owner's call).

- **AUC-targeted noise sweep** (`results/contest_at_scale/auc_frontier/reports/auc_targeted_noise_sweep.md`, `results/contest_at_scale/auc_frontier/reports/auc_frontier.html`):
  4 datasets (EuroSAT n=48, Alzheimer n=48, Fashion-MNIST n=48, CIFAR-10 n=100) x 2 partition modes
  x 2 privacy modes = 16 curves, each an autonomous search for the noise multiplier that pushes CIA
  round-matched attack AUC to ~0.5 (10 landed, 4 collapsed-before-target, 2 anchor-not-found — see
  the report for the full per-curve table). Noise-to-neutralize-attack cost is dataset/partition/
  mechanism-dependent, not a single number: EuroSAT reaches the target cheaply in all 4
  combinations; CIFAR-10 reaches it in all 4 but the accuracy cost swings from negligible
  (non-iid/global-dp) to severe (homogeneous/global-dp, pushed to near-random accuracy); Alzheimer
  and Fashion-MNIST/homogeneous show the sweep's clearest negative result — several combinations
  break the model before the attack is ever actually neutralized, and Fashion-MNIST/homogeneous
  (both privacy modes) never found a usable low-noise anchor at all in the range searched.
  Metric-privacy's cost advantage over global-DP shows up clearly on CIFAR-10/homogeneous but
  reverses on CIFAR-10/non-iid — no blanket "metric-privacy is cheaper" claim survives this data.
- The metric-privacy mechanism reproduces the source paper at 4 clients — the effect is barely
  visible at the paper's `noise_multiplier=0.01` (`reports/paper_reproduction.md`).
- A genuine, previously unpublished effect exists at 8 clients: metric-privacy beats global-DP by
  +6.9pp (homogeneous) / +12.2pp (non-IID) at `noise_multiplier=0.05`
  (`reports/client_count_scaling.md`).
- `MetricPrivacyServerSideFixedClipping.aggregate_train` (`metricdp_pytorch/metricdp_strategy.py`)
  no longer aborts a whole run on a non-finite/non-positive client-model distance or a
  `ZeroDivisionError` from Flower's own clipping code on zero-norm updates — both fall back
  gracefully (last-valid distance, or skip-and-keep-previous-round respectively) so a run's full
  round-by-round history survives instead of being discarded on one bad round. Richer per-round
  diagnostics also landed: full pairwise client-model distance distribution, per-pair client IDs,
  min/median/mean/count, not just the single max used for calibration.
  `LoggedGlobalDPServerSideFixedClipping` (`metricdp_pytorch/globaldp_strategy.py`) has the same
  zero-norm-update guard as of the scale-controlled redo — it never had one before, and 12/96 runs
  in `results/noise_by_clients/` hit exactly this crash before the fix landed.
- **A genuine client-reply-ordering non-determinism was found and fixed**: Flower's own weighted
  aggregation summed replies in network-arrival order, not deterministically — floating-point
  addition isn't associative, so this compounded into real numeric drift over rounds. Fixed via
  `DeterministicReplyOrderMixin` (`metricdp_pytorch/strategy_factory.py`, covers every aggregation
  method, not just metric-privacy) and a matching fix in `metricdp_pytorch/metrics.py`.
- **Phase 1 items 1 and 2 of `docs/RESEARCH_ROADMAP.md` are done**, redone on CUDA hardware (moved
  off this project's original Mac MPS backend, whose own non-determinism made every earlier result
  on these questions untrustworthy — see `reports/archive/constant_compute_scaling_mps_v1v2.md`).
  - **Constant-compute client-count scaling** (`reports/constant_compute_scaling.md`/`.tex`):
    `fedavg` at `n=4/8/48` + `fedyogi` at `n=4/8` — 40/40 combinations, 0 failures, 0
    invalid-distance/collapsed-aggregation rounds anywhere. The metric-privacy-vs-global-dp
    advantage shrinks from `n=4` to `n=48` but converges toward parity (`fedavg`: -2.5pp to
    +0.6pp at `n=48`), not the large reversal the MPS-era attempt found (which never survived its
    own noise-floor check). `fedyogi` at `n=48` not yet run.
  - **Noise-multiplier x client-count sweep** (`reports/noise_by_clients.md`): `n ∈ {8,16,32,48}` x
    6 noise multipliers, `fedavg` only — 96/96, 0 failures. The noise ceiling genuinely shifts up
    with client count (a fixed `noise_multiplier` is relatively less noisy at higher `n`, since
    `compute_stdv` divides by `num_sampled_clients`) — `nm=0.1` collapses training at `n=8` but
    stays healthy through `n=48`. Separately, metric-privacy's own calibrated noise goes unstable
    right at each client count's collapse boundary and underperforms global-dp there by as much as
    -18pp, worse at higher `n` — a real, more precisely localized version of the original
    `results/48client_scaling` scaling concern, and a lead for Phase 2's mechanism redesign.
- `reports/first_round_cia.md` is stale — says "no result data yet," but `results/cia_client_scaling/`
  has real trained models and partial attack scores. Needs a rewrite, not done yet. The Flower-1.32
  port-equivalence check (`reports/port_equivalence.md`) still has no committed result data.
- **CIFAR-100 accuracy sweep + CIA** (`results/contest_at_scale/reports/cifar-100_and_eurosat_results.tex`): `n=100`
  clients, 250 rounds, `fedavg`, `noise_multiplier=0.0182`. Accuracy sweep: 6/6 combos, 0 failed,
  21.4–29.1% accuracy (100-class task, ~1% random-baseline) — metric-privacy roughly ties
  global-dp (within ~0.6pp either way), both DP modes ~7–8pp below vanilla, partition mode barely
  moves the numbers. CIA: seed-42-only (a multi-seed 43/44 rerun was attempted for proper 3-seed
  pooling but dropped after repeated GPU VRAM contention on this large 4.6M-parameter model
  prevented it from ever completing — see `git log` on the deleted `feature/cifar100-scaling` for
  the retry history), 26 round-matched pairs per combo, all 6/6 combos scored. Leakage is highest
  for `homogeneous/vanilla` (0.846) and lowest for `non-iid/vanilla` (0.500, the no-leakage line);
  `global-dp` shows more leakage than `metric-privacy` in every partition/shadow combination on
  this seed. Every 95% CI includes 0.5 (single-seed, underpowered), so read this as a directional
  pattern, not a statistically confirmed ranking.
- **EuroSAT accuracy sweep + CIA** (`results/contest_at_scale/eurosat/reports/eurosat_accuracy_sweep.md`, `results/contest_at_scale/eurosat/reports/eurosat_cia.md`):
  a comparison point on satellite land-use imagery (10-class, genuinely different domain from
  CIFAR-10/CIFAR-100/Fashion-MNIST/Alzheimer), `n=48`. Accuracy sweep: 6/6 combos, 0 failed,
  87.5–90.7% accuracy across all combos; `non-iid` partitioning slightly *outperformed*
  `homogeneous` in every privacy mode, and DP mechanisms cost only 0.4–2.4pp versus vanilla. CIA:
  36/36 trajectories (18 IN + 18 OUT), 0 failed, 3 seeds from the start (CIFAR-100's CIA needed a
  post-hoc multi-seed redo after an underpowered single-seed pilot — this one skipped that
  mistake). Round-matched AUC shows the clearest leak at `homogeneous/vanilla` (0.727), both DP
  mechanisms suppress it there (to 0.606), and `non-iid` leaks much less across every privacy mode
  (one combo's noisy-shadow AUC drops to 0.212, below chance) — but confidence intervals are wide
  (~0.30–0.35 AUC units, all overlapping 0.5), so read this as a directional pattern, not a
  statistically confirmed ranking. Along the way: found and fixed a real bug in `server.py`
  (`_require_trained_arrays`) where a run whose every single round failed to aggregate crashed
  with a confusing "Missing key(s) in state_dict" error instead of a clear one — Flower's
  `Strategy.start()` only assigns `result.arrays` on a successful round, with no fallback to the
  initial model.

- **Client-specific noise / stacked head step** (`research/README.md`, `experiments/stacked_head/protocols/*findings.md`): see the closed-direction
  entry under Active work. Utility versus public-only models only in the limited-public regime (reliable at 32 public images, fades by 512);
  not better than server-side global DP or metric privacy at matched calibrated risk; the public-validation gate is the transferable improvement.

## Where to look

- `research/README.md` — closure note and map of the (closed) client-specific-noise research record;
  `research/literature_review/README.md` — its systematic/integrative review and technique tutorials.
- `experiments/stacked_head/README.md` — the Flower implementation, runner, protocols and findings of that direction.
- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data; `results/archive/` — superseded data kept for comparison.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
