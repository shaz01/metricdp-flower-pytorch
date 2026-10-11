# Escalation: asynchronous host-to-MPS copies corrupt evaluation batches (and likely training)

Raised 2026-10-11 by Agent A during WP0. Escalation rule (agent_roadmap.md Section 4.4): item 3 (the fix needs a file
another agent owns) and item 5 (a dependency is broken for owner-PC runs). WP0 itself is complete (G0 passed); what is
blocked is any training or evaluation on the owner PC, which includes WP1 and the 3-client part of WP5.

## What happened

Scoring the round-0 and round-1 checkpoints of a 1-round smoke run (3 Alzheimer clients) on the owner PC (Apple MPS,
torch 2.10.0) gave different losses on two identical calls of the existing, unmodified `experiments.cia.cia.eval_model`.
The values were also far from the CPU values for the same checkpoint.

| Round-0 checkpoint, clean shadow sample (136 images) | Mean loss |
|---|---|
| CPU, twice | 1.440792, 1.440792 |
| MPS, blocking copies (`non_blocking=False`), three times | 1.440792, 1.440792, 1.440792 |
| MPS, `non_blocking=True` (the repository's code), three times | 0.830654, 0.918062, 0.699525 |

Cause: `tensor.to(device, non_blocking=True)` for DataLoader batches. On CUDA this is safe; on MPS the model reads the
batch before the copy has finished. A synthetic regression test reproduces it (fails with the old copy, passes with the
fix): `experiments/cia/tests/test_round_scalars.py::test_mps_shadow_scalars_match_cpu` (skipped where MPS is absent).

## Where the pattern is

| File | Role | Owner | State |
|---|---|---|---|
| `experiments/cia/cia.py` | CIA loss evaluation | Agent A | fixed on `feature/cia-infra` (`_to_device`: asynchronous copies only on CUDA) |
| `experiments/reproduce/paper_training.py` (lines 108-109) | client local training | Agent B | not fixed (same pattern; its effect on training was not measured separately) |
| `experiments/reproduce/paper_loss.py` (lines 55-56) | server evaluation during training (accuracy, loss per round) | nobody | not fixed |
| `experiments/reproduce/detailed_evaluation.py` (line 43) | post-training evaluation | nobody | not fixed |

## Impact on committed results

Every committed result JSON that records a device ran on CUDA (887 `"device": "cuda:0"` entries, no MPS entry, in a
scan of all tracked `results/` and `experiments/` JSON files). Many result files record no device, so an older MPS run
cannot be excluded from that scan alone. The compute server (CUDA) is unaffected; WP2 and WP4 can proceed there.

## Options

1. **(Recommended)** Apply the same one-line `_to_device` rule to the three remaining files (asynchronous copies only on
   CUDA). It changes nothing on CUDA or CPU, so committed CUDA results stay reproducible. Agent B (or the owner) edits
   `paper_training.py`; Agent A can edit the two unowned files on `feature/cia-infra` if the owner agrees.
2. Run everything on the CPU of the owner PC (`resolve_device` unchanged, but force CPU for owner-PC runs). Slower;
   leaves the bug in place for anyone else.
3. Move WP1 to the compute server. Contradicts the owner decision that small federations run on the owner PC.

Until the owner decides, Agent A does not launch WP1 training on the owner PC.
