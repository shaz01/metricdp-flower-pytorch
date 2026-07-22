# Paper Reproduction Progress

## Scope

CIA experiments are excluded. Planned utility matrix:

- Homogeneous and non-IID 4-client paper partitions.
- Vanilla FL, global DP, and metric privacy.
- FedAvg, FedAvgM, FedMedian, FedProx, FedOpt, and FedYogi.
- Main paper settings: 20 rounds, 5 local epochs, batch size 32, Adam learning rate 0.001, clipping norm 5, noise multiplier 0.01.
- Stateful strategy initialization: 20 validation epochs for FedAvgM, FedOpt, and FedYogi.
- Five runs for the homogeneous matrix and fixed-seed runs for non-IID, following the paper's stated protocol.
- Appendix B: homogeneous global-DP/metric-privacy matrix at noise 0.003 and FedAvg checks at 0.05 and 0.1.

## Environment validation — 2026-07-22 UTC

- Remote repository: `/teamspace/studios/this_studio`
- Git revision: `79d4e78`
- Unit tests: **26 passed, 5 deselected** in 5.94 s.
- `uv run` environment: PyTorch 2.10.0+cu128.
- CUDA check: **unavailable** (`torch.cuda.is_available() == False`, device count 0).
- `/dev/nvidia*`: absent; `nvidia-smi`: unavailable.
- Lightning CLI reports Studio `metricdp-fl-flower` is **Running / CPU**, not L4.

## Status / blocker

No reproduction runs were launched in this session because the requested L4 is not attached. Running the full matrix on the currently attached CPU would be impractically slow and would not satisfy the requested GPU execution. Existing JSON files in `results/reproduce/` predate this session and were not counted as new results.

## Repository note

The remote working tree already contains a necessary one-line runner import correction (`metricdp_pytorch.runtime` to `metricdp_pytorch.utils.runtime`). It was present before this session and was left unchanged.
