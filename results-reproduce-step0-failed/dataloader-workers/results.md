# Dataloader Workers Paper Reproduction

- Branch: dataloader-workers (555a29e)
- GPU: NVIDIA L4
- Started: 2026-07-22T15:21:49Z
## Smoke test
- Smoke exit: 127
- Smoke retry exit: 0 (20s)

## Full matrix

- Started: 2026-07-22T15:24:02Z
- Main homogeneous: five seeds (42–46), 3 privacy modes × 6 aggregators.
- Main non-IID: fixed seed 42, 3 privacy modes × 6 aggregators.
- Appendix B: homogeneous GDP/metric privacy × 6 aggregators at noise 0.003; GDP FedAvg at 0.05 and 0.1.
- Common settings: 4 clients, 20 rounds, 5 local epochs, batch 32, Adam LR 0.001, clipping norm 5, stateful initialization 20 epochs.
- Runtime: two concurrent clients, 0.5 L4 GPU per client.
- CIA excluded. The implementation records accuracy/loss, not F1/precision predictions.

| Run | Status | Seconds | Server accuracy | Server loss | Client accuracy last 5 (mean ± std) |
|---|---:|---:|---:|---:|---:|
| main__homogeneous__vanilla__fedavg__seed-42__noise-0.01 | ok | 230 | 0.960938 | 0.250811 | 0.964648 ± 0.004297 |
| main__homogeneous__vanilla__fedavgm__seed-42__noise-0.01 | ok | 238 | 0.903125 | 0.308406 | 0.895508 ± 0.017238 |
| main__homogeneous__vanilla__fedmedian__seed-42__noise-0.01 | ok | 239 | 0.962500 | 0.257689 | 0.964063 ± 0.005572 |
