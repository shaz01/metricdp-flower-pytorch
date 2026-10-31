Package: P0, infrastructure and cost runs (CIA roadmap v3, Agent P)
status: G-P0 passes (2026-10-31); P2 and P3 escalated on cost (ESCALATION.md)
gate: G-P0 = pass (all five items)
protocol: this file (P0 has no leakage statistic; its cost runs are never analyzed for leakage)
machine: compute server
cost: 1.3 GPU-h (five runs, one worker, GPU 1 shared with another user's job); cap 3 GPU-h
next: P1 (within its cap) may start; P2 and P3 wait for the owner's decision on ESCALATION.md
needs from others: owner decision on P2's and P3's caps

Roadmap: `research/cia_roadmap_v3/ROADMAP.md` Sections 2, 3 (P0) and 6, decision 1 in Section 10. Branch
`feature/cia-v3-infra` from master 3654f6fa; sparse worktree `../metricdp-pytorch-feature-cia-v3-infra`.
P0 blocks P1-P4: nothing beyond the four cost runs is launched before G-P0 passes.

## Hypothesis

P0 is infrastructure, so its "hypothesis" is a pair of checkable claims:

1. The four additions (every-round scoring, IN-count design, CIFAR-10 data module, paper-protocol runner) can be made
   without changing any committed run: with default options every plan and manifest of M2, A1, B1, B5, C1 and AZ is
   byte-identical, and `reanalyze.py` still reproduces C1's published gate numbers to 3 decimals.
2. The per-run costs fit the downstream caps. Break-even costs per run from the roadmap's caps and run counts:
   - P1: 120 runs (20 pairs x 2 worlds x 3 mechanisms) in 40 GPU-h, so at most about 0.33 GPU-h per run;
   - P2: about 350 runs (stage a 8, pilot about 18, stage b 10 configurations x 32) in 85 GPU-h, so at most about
     0.24 GPU-h per CIFAR-10 run (EuroSAT's is about 0.1, and CIFAR-10 has about 2.3x the training records: 50,000 against 21,600);
   - P3: 192 runs in 20 GPU-h, so at most about 0.10 GPU-h per run on average.
   If a cost run projects a package above 1.25x its cap, I record it here and write `ESCALATION.md` before that
   package's protocol, rather than trimming the design myself.

## Design

**Code items** (ROADMAP P0 items 1-4):

1. **Every-round shadow scoring:** `--checkpoint-every N` on the randomized-inclusion runner (default 5, so
   committed manifests keep `checkpoint_rounds` 0, 5, ..., 50; `1` gives 51 checkpoints). Stored in the manifest only
   when it differs from the profile's default.
2. **IN-count option:** `--in-count k` with k in {8, 24} (default 16 = the committed fixed-count design, unchanged
   and still drawn by `experiments.cia.inclusion.balanced_design`). For k != 16 a balanced block design: each block of
   `32 / k` replicates (k = 8) partitions a fresh random permutation of the 32 candidates into groups of k; k = 24 is
   the complement of the k = 8 design. Every replicate includes exactly k candidates, every candidate is IN in exactly
   `R k / 32` replicates (8 of 32 for k = 8; 24 of 32 for k = 24), and the active count `16 + k` (24 or 40) is fixed
   and public per configuration, so the noise std stays `r C` on the released average. The runner's design checks
   (row and column sums, no duplicate replicate, fixed active count) are unit-tested.
3. **CIFAR-10 label-Dirichlet module:** a `cifar10` dataset profile on the same runner: label-Dirichlet alpha 0.3, 48
   slots, layout seed 42, at least 10 records per client (the EuroSAT rule), the ten-class CIFAR-10 of
   `experiments/reproduce/dataset/cifar10.py`, model `experiments.reproduce.cifar10_cnn`, WP2/M2 hyperparameters (clip
   5, 5 local epochs, batch 32, lr 0.001, 20 initialization epochs, 50 rounds), shadow fraction 10%. Candidates are
   chosen by the runner's existing `select_candidates` rule from the committed seed-42 partition summary
   (`results/cia_v3/infra/tables/cifar10_partition_summary.json`) and fixed in code before any run. A test checks the
   partition is identical for the same seed and differs for another.
4. **Paper-protocol runner** (`results/cia_v3/infra/paper_runner.py`): 3-client Alzheimer with the paper's Table 9
   client counts (`experiments/cia/datasets/paper.py`, target client 2), the paper's aggregation, rounds and noise for
   global DP and metric privacy, plus vanilla. One pair = an IN world (3 clients) and an OUT world (the target
   removed). Each world draws its own `client_seed` and `noise_seed` from `SeedSequence([design_seed, pair, world])`;
   public randomness (data split, initial model, shadow sample, evaluation set) is shared. It stores the target's
   shadow loss at every round, so both the paper's pooled-checkpoint ROC (per pair) and a calibrated per-trajectory
   attacker can be computed in P1. The exact rounds and noise are taken from the paper in P1's protocol; the runner
   takes them as arguments.

**Cost runs** (item 5; accuracy and time only, never analyzed for leakage, outputs under
`results/cia_v3/infra/cost/`, `design_seed = 9`, a seed no analysis uses):
- CIFAR-10 RI vanilla, one replicate;
- CIFAR-10 RI composite (`subsampled-global-dp`, `q = 0.25`, `tau = 1`, `r` = 0.003092, EuroSAT's middle level), one
  replicate;
- one 3-client Alzheimer paper-protocol pair (global DP), i.e. two worlds;
- one EuroSAT composite run with `--checkpoint-every 1` (51 checkpoints), `r` = 0.003092.

Each records training and evaluation seconds, the device, worker parallelism and the concurrent load on the GPU, so
GPU-h of occupancy can be compared with v2's.

**No private randomness is shared between worlds.** In the randomized-inclusion runs every replicate draws its own
`client_seed` and `noise_seed` from `(design_seed, replicate)`, and inclusion is randomized; in the paper protocol the
IN and OUT worlds of a pair draw separate `client_seed` and `noise_seed`, and pairs draw separate seeds.

## Attacker and metrics

None gated in P0. The code paths store what the downstream attackers need: per-checkpoint shadow losses (and the
logit-adjusted loss under logit adjustment) for the calibrated 11-checkpoint attacker, the 51-checkpoint attacker and
the sampling-aware attacker (per-round `client-ids` are already logged); per-round target shadow losses for the
paper's pooled-checkpoint ROC. The cost runs report final accuracy, lowest-client accuracy and GPU-h only.

## Gate

**G-P0** (ROADMAP v3 Section 3, verbatim):
- `uv run pytest` is green.
- `reanalyze.py` still reproduces C1's published gate numbers.
- The CIFAR-10 partition is deterministic in its seed.
- The IN-count design passes the runner's design checks.
- All four cost runs are recorded.

Fail: nothing else runs (Section 4). **Untestable targets:** P0 has no matched-accuracy targets, so standard v3 item 5
does not apply here; a cost run that fails to finish counts as "not recorded", G-P0 then fails, and I report why
rather than substituting a different run.

## Files

| File | Change |
|---|---|
| `results/cia_v2/randomized_inclusion/runner.py` | `--checkpoint-every`, `--in-count`, the `cifar10` profile and its candidates; defaults leave every committed plan and manifest unchanged |
| `results/cia_v2/randomized_inclusion/data.py` | `cifar10` branch in `base_module` |
| `results/cia_v2/randomized_inclusion/tests/` | tests for the new options and for unchanged default manifests |
| `results/cia_v3/infra/cifar10_data.py` | label-Dirichlet CIFAR-10 module and `partition_summary` (new) |
| `results/cia_v3/infra/paper_runner.py` | 3-client Alzheimer paper-protocol runner (new) |
| `results/cia_v3/infra/tests/` | synthetic tests (no datasets, no GPU) |
| `results/cia_v3/infra/tables/cifar10_partition_summary.json` | committed partition summary |
| `results/cia_v3/infra/cost/` | the four cost runs' manifests, measurements and completion records |
| `results/cia_v3/infra/STATE.md` | this file |
| `STATUS.md` | Agent P rows only |

## Progress

**2026-10-30.** Code items 1-4 done (commit 4cc8af25).

G-P0 so far:

| Item | Result |
|---|---|
| `uv run pytest` green | **Pass.** 944 passed, 3 skipped, 5 deselected (the reproducibility marker). Two skips need an Apple MPS device. The third (`test_power.py`, WP2 trajectories not in the sparse worktree) passes, 3 of 3, once those trajectories are checked out. |
| `reanalyze.py` reproduces C1's gate numbers | **Pass.** `"reproduced": true`: -0.095 [-0.137, -0.053] at 0.740 and -0.108 [-0.163, -0.053] at 0.660. The regenerated `analysis.json` equals the committed one in every key and string; its 586 floats differ by at most 2.2e-16 (summation order), so the committed file was left unchanged. |
| CIFAR-10 partition deterministic in its seed | **Pass.** `partition_summary(0.3, 48, 42)` is identical on two calls and differs at seed 43; the committed summary equals a fresh one (test). Seed-42 layout: 263 to 2,454 records per client, 50,000 in all. Candidates by the runner's rule: top decile 26, 27, 36, 12, 45 (2,454 to 1,704 records), then 27 by size quantile (269 to 1,699). |
| IN-count design passes the design checks | **Pass.** For k = 8 and 24 at `R = 32`: every replicate includes exactly k candidates, every candidate is IN in `R k / 32` replicates, no replicate repeats; k = 24 is the complement of k = 8. Active count 24 or 40, noise std `r C` on the average. `load_audit` checks a k = 8 configuration against its own design. |
| Four cost runs recorded | **Pass** (2026-10-31 00:41; table in ESCALATION.md). |

**Backward compatibility.** Every committed randomized-inclusion manifest under `results/cia_v2/` was rebuilt from its
own settings with the new runner. 1,337 of 1,351 match exactly, the same count as with master's code. The other 14
are the same under both: Alzheimer calibration runs from before the Alzheimer profile was fixed (old panel and alpha).
The new options appear in a manifest only when they are used (`in_count` when not 16; `checkpoint_rounds` lists every
round with `--checkpoint-every 1`), so run names and default manifests are unchanged.

**Paper protocol.** The paper's Section 7.4.2 settings are built in: Table 9 split (`experiments/cia/datasets/paper.py`,
target client 3), FedAvg weighted by examples (Flower's default, as in the paper), 20 rounds, 5 local epochs, batch 32,
clipping 5, noise multiplier `z` with std `z C / n_t` (divided by the round's maximum client distance for metric
privacy), shadow sample 10% of the target's training records with Gaussian noise of 20% of the maximum pixel value.
Scored at rounds 0-20 (the paper's statistic uses 1-20); the clean shadow sample is scored too. P1's protocol fixes `z`
and the pair count. A CPU smoke run (1 pair, 2 rounds) trained both worlds with their own seeds and stored every
round.

**GPUs at 2026-10-30 23:04.** Both GPUs ran other users' active training jobs (99-100% utilization, 25-26 GB of 32 GB
used), so under the server rules neither may be used. The cost runs start on the first GPU with no process and no
claim (`~/cia-server/p0-cost-wait.sh`, polling every 5 minutes; runs in tmux `p0-cost`, logs
`~/cia-server/logs/p0-cost-*.log`, GPU samples every 60 s in `p0-cost-gpu.log`).

**2026-10-30 23:23. Cost runs started under ROADMAP decision 4** (owner: share a GPU with another user's job when the
work needs about 3-4 GB of VRAM). The launcher took GPU 1 (6.8 GB free; the other user's job at 97-100% utilization).
Our processes use about 2.4-2.5 GB. **Timings from these runs overstate an unshared GPU's cost**: the GPU is
compute-saturated by the other job, so the cost projection is an upper bound. `p0-cost-gpu.log` records the GPU's
utilization and our VRAM every 60 s, so this can be stated per run.

**2026-10-31 00:41. Cost runs complete, G-P0 passes.** All four finished at the first attempt (23:23-00:41; no
out-of-memory, no retry), every run passed the runner's every-client check, and the claim line was removed. Our peak
VRAM was 3.6 GB; the other user's job kept GPU 1 at 97-100% utilization throughout.

| Run | Run-hours | Final accuracy |
|---|---|---|
| CIFAR-10 RI vanilla | 0.542 | 0.630 |
| CIFAR-10 RI composite (`r` 0.003092) | 0.360 | 0.546 |
| Paper pair (global DP, `z` 0.01), IN + OUT | 0.067 + 0.060 | 0.920 / 0.831 |
| EuroSAT RI composite, 51 checkpoints | 0.247 | 0.671 |

The EuroSAT run trained in 460 s against a median of 440 s for C1's 128 committed runs of the same configuration, so
the shared-GPU timings match v2's conditions. Projected against the caps (ESCALATION.md): P1 about 5 GPU-h (cap 40);
**P2 about 116 GPU-h (cap 85, 1.25x = 106) and P3 about 32 GPU-h (cap 20, 1.25x = 25), both above 1.25x**, so both are
escalated before their protocols, with options. `*.evaluation.json` files are kept locally and not committed, as in v2.
