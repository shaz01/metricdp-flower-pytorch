# Non-CIA Paper Reproduction Results

Started: 2026-07-22 UTC

## Scope and protocol

- CIA experiments are excluded.
- Main homogeneous matrix: seeds 42–46 × {vanilla, global-DP, metric-privacy} × {FedAvg, FedAvgM, FedMedian, FedProx, FedOpt, FedYogi} = 90 runs.
- Main non-IID matrix: seed 42 × the same 3 × 6 matrix = 18 runs.
- Main settings: 4 clients, 20 rounds, 5 local epochs, batch 32, Adam LR 0.001, clipping norm 5, noise multiplier 0.01. Stateful FedAvgM/FedOpt/FedYogi initialization uses 20 epochs.
- Appendix B: homogeneous seed 42, global-DP and metric-privacy × 6 aggregators at noise 0.003 (12 runs), plus homogeneous FedAvg global-DP at 0.05 and 0.1 (2 runs).
- The paper does not publish five seed numbers; seeds 42–46 are used as an explicit reproducibility convention.
- Every run has a unique JSON path and log. Existing results predating this attempt are not counted.
- Current implementation persists centralized loss/accuracy and client accuracy only; F1/precision are not persisted and will not be invented.

## Environment

- Lightning Studio reports **L4**.
- GPU: NVIDIA L4, 23034 MiB, driver 580.159.03.
- PyTorch: 2.10.0+cu128; CUDA available; one CUDA device.
- Repository revision: 79d4e78, with a pre-existing remote runner import correction.

## Status

- Preparing unit tests and CUDA smoke experiment.

## Run results

| Run | Status | Final centralized accuracy | Final centralized loss | Notes |
|---|---:|---:|---:|---|

- `2026-07-22 10:45:54 UTC` — Unit tests PASS: 26 passed, 5 deselected. CUDA smoke PASS: final centralized accuracy 0.5, loss 1.65556001663208. Matrix launch uses four concurrent Ray clients at 0.25 L4 GPU each.

- `2026-07-22 10:45:54 UTC` — STARTED main__homogeneous__vanilla__fedavg__noise-0.01__seed-42

- `2026-07-22 10:56:42 UTC` — Unit tests PASS: 26 passed, 5 deselected. CUDA smoke PASS: final centralized accuracy 0.5, loss 1.65556001663208. Matrix launch uses four concurrent Ray clients at 0.25 L4 GPU each.
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-42` | PASS | 0.9609375 | 0.15919359670951963 | current-attempts; exit 0 |

- `2026-07-22 10:56:42 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedavg__noise-0.01__seed-42 — PASS (not rerun)

- `2026-07-22 10:56:42 UTC` — STARTED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-42
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-42` | PASS | 0.925 | 0.30436030998826025 | 219s; exit 0 |

- `2026-07-22 11:00:21 UTC` — COMPLETED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-42 — PASS (exit 0, 219s)

- `2026-07-22 11:00:21 UTC` — STARTED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-42
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-42` | PASS | 0.9578125 | 0.24632993377745152 | 216s; exit 0 |

- `2026-07-22 11:03:57 UTC` — COMPLETED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-42 — PASS (exit 0, 216s)

- `2026-07-22 11:03:57 UTC` — STARTED main__homogeneous__vanilla__fedprox__noise-0.01__seed-42
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-42` | PASS | 0.540625 | 0.9606019735336304 | 228s; exit 0 |

- `2026-07-22 11:07:45 UTC` — COMPLETED main__homogeneous__vanilla__fedprox__noise-0.01__seed-42 — PASS (exit 0, 228s)

- `2026-07-22 11:07:45 UTC` — STARTED main__homogeneous__vanilla__fedopt__noise-0.01__seed-42
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-42` | PASS | 0.359375 | 1.1029120802879333 | 225s; exit 0 |

- `2026-07-22 11:11:30 UTC` — COMPLETED main__homogeneous__vanilla__fedopt__noise-0.01__seed-42 — PASS (exit 0, 225s)

- `2026-07-22 11:11:30 UTC` — STARTED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-42
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-42` | PASS | 0.953125 | 1.1325476241716885 | 221s; exit 0 |

- `2026-07-22 11:15:11 UTC` — COMPLETED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-42 — PASS (exit 0, 221s)

- `2026-07-22 11:15:11 UTC` — STARTED main__homogeneous__global-dp__fedavg__noise-0.01__seed-42
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-42` | PASS | 0.940625 | 0.23916386477649212 | 215s; exit 0 |

- `2026-07-22 11:18:46 UTC` — COMPLETED main__homogeneous__global-dp__fedavg__noise-0.01__seed-42 — PASS (exit 0, 215s)

- `2026-07-22 11:18:46 UTC` — STARTED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-42
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-42` | PASS | 0.75 | 0.6208976864814758 | 227s; exit 0 |

- `2026-07-22 11:22:33 UTC` — COMPLETED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-42 — PASS (exit 0, 227s)

- `2026-07-22 11:22:33 UTC` — STARTED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-42
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-42` | PASS | 0.940625 | 0.2835912951501086 | 225s; exit 0 |

- `2026-07-22 11:26:18 UTC` — COMPLETED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-42 — PASS (exit 0, 225s)

- `2026-07-22 11:26:18 UTC` — STARTED main__homogeneous__global-dp__fedprox__noise-0.01__seed-42
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-42` | PASS | 0.5578125 | 0.9510888695716858 | 236s; exit 0 |

- `2026-07-22 11:30:14 UTC` — COMPLETED main__homogeneous__global-dp__fedprox__noise-0.01__seed-42 — PASS (exit 0, 236s)

- `2026-07-22 11:30:14 UTC` — STARTED main__homogeneous__global-dp__fedopt__noise-0.01__seed-42
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-42` | PASS | 0.359375 | 1.074967336654663 | 244s; exit 0 |

- `2026-07-22 11:34:19 UTC` — COMPLETED main__homogeneous__global-dp__fedopt__noise-0.01__seed-42 — PASS (exit 0, 244s)

- `2026-07-22 11:34:19 UTC` — STARTED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-42
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-42` | PASS | 0.9546875 | 0.4574632991664657 | 229s; exit 0 |

- `2026-07-22 11:38:08 UTC` — COMPLETED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-42 — PASS (exit 0, 229s)

- `2026-07-22 11:38:08 UTC` — STARTED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-42
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-42` | PASS | 0.9578125 | 0.2213929073885083 | 217s; exit 0 |

- `2026-07-22 11:41:45 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-42 — PASS (exit 0, 217s)

- `2026-07-22 11:41:45 UTC` — STARTED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-42
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-42` | PASS | 0.76875 | 0.5736923143267632 | 228s; exit 0 |

- `2026-07-22 11:45:33 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-42 — PASS (exit 0, 228s)

- `2026-07-22 11:45:33 UTC` — STARTED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-42
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-42` | PASS | 0.9453125 | 0.27011424000374973 | 227s; exit 0 |

- `2026-07-22 11:49:21 UTC` — COMPLETED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-42 — PASS (exit 0, 227s)

- `2026-07-22 11:49:21 UTC` — STARTED main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-42
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-42` | PASS | 0.4984375 | 1.1741622775793075 | 236s; exit 0 |

- `2026-07-22 11:53:17 UTC` — COMPLETED main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-42 — PASS (exit 0, 236s)

- `2026-07-22 11:53:17 UTC` — STARTED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42` | FAIL |  |  | 83s; exit 1 |

- `2026-07-22 11:54:40 UTC` — COMPLETED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42 — FAIL (exit 1, 83s)

- `2026-07-22 11:54:40 UTC` — STARTED main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-42
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-42` | PASS | 0.9546875 | 0.66376755728852 | 235s; exit 0 |

- `2026-07-22 11:58:35 UTC` — COMPLETED main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-42 — PASS (exit 0, 235s)

- `2026-07-22 11:58:35 UTC` — STARTED main__homogeneous__vanilla__fedavg__noise-0.01__seed-43
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-43` | PASS | 0.94375 | 0.3629816941800527 | 208s; exit 0 |

- `2026-07-22 12:02:03 UTC` — COMPLETED main__homogeneous__vanilla__fedavg__noise-0.01__seed-43 — PASS (exit 0, 208s)

- `2026-07-22 12:02:03 UTC` — STARTED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-43
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-43` | PASS | 0.8984375 | 0.308837766200304 | 219s; exit 0 |

- `2026-07-22 12:05:42 UTC` — COMPLETED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-43 — PASS (exit 0, 219s)

- `2026-07-22 12:05:42 UTC` — STARTED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-43
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-43` | PASS | 0.9375 | 0.3733043486252427 | 217s; exit 0 |

- `2026-07-22 12:09:19 UTC` — COMPLETED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-43 — PASS (exit 0, 217s)

- `2026-07-22 12:09:19 UTC` — STARTED main__homogeneous__vanilla__fedprox__noise-0.01__seed-43
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-43` | PASS | 0.5328125 | 0.9261418014764786 | 229s; exit 0 |

- `2026-07-22 12:13:08 UTC` — COMPLETED main__homogeneous__vanilla__fedprox__noise-0.01__seed-43 — PASS (exit 0, 229s)

- `2026-07-22 12:13:08 UTC` — STARTED main__homogeneous__vanilla__fedopt__noise-0.01__seed-43
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-43` | PASS | 0.4953125 | 1.071137297153473 | 225s; exit 0 |

- `2026-07-22 12:16:53 UTC` — COMPLETED main__homogeneous__vanilla__fedopt__noise-0.01__seed-43 — PASS (exit 0, 225s)

- `2026-07-22 12:16:53 UTC` — STARTED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-43
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-43` | PASS | 0.9390625 | 0.4779103420863976 | 220s; exit 0 |

- `2026-07-22 12:20:33 UTC` — COMPLETED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-43 — PASS (exit 0, 220s)

- `2026-07-22 12:20:33 UTC` — STARTED main__homogeneous__global-dp__fedavg__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-43` | PASS | 0.934375 | 0.30373870697803795 | 217s; exit 0 |

- `2026-07-22 12:24:10 UTC` — COMPLETED main__homogeneous__global-dp__fedavg__noise-0.01__seed-43 — PASS (exit 0, 217s)

- `2026-07-22 12:24:10 UTC` — STARTED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-43` | PASS | 0.7515625 | 0.5576686561107635 | 226s; exit 0 |

- `2026-07-22 12:27:56 UTC` — COMPLETED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-43 — PASS (exit 0, 226s)

- `2026-07-22 12:27:56 UTC` — STARTED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-43` | PASS | 0.9265625 | 0.35366105642169715 | 224s; exit 0 |

- `2026-07-22 12:31:40 UTC` — COMPLETED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-43 — PASS (exit 0, 224s)

- `2026-07-22 12:31:40 UTC` — STARTED main__homogeneous__global-dp__fedprox__noise-0.01__seed-43

- `2026-07-22 12:32:59 UTC` — Unit tests PASS: 26 passed, 5 deselected. CUDA smoke PASS: final centralized accuracy 0.5, loss 1.65556001663208. Matrix launch uses four concurrent Ray clients at 0.25 L4 GPU each.
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-42` | PASS | 0.9609375 | 0.15919359670951963 | current-attempts; exit 0 |

- `2026-07-22 12:32:59 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedavg__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-42` | PASS | 0.925 | 0.30436030998826025 | current-attempts; exit 0 |

- `2026-07-22 12:32:59 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedavgm__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-42` | PASS | 0.9578125 | 0.24632993377745152 | current-attempts; exit 0 |

- `2026-07-22 12:32:59 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedmedian__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-42` | PASS | 0.540625 | 0.9606019735336304 | current-attempts; exit 0 |

- `2026-07-22 12:32:59 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedprox__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-42` | PASS | 0.359375 | 1.1029120802879333 | current-attempts; exit 0 |

- `2026-07-22 12:32:59 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedopt__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-42` | PASS | 0.953125 | 1.1325476241716885 | current-attempts; exit 0 |

- `2026-07-22 12:32:59 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedyogi__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-42` | PASS | 0.940625 | 0.23916386477649212 | current-attempts; exit 0 |

- `2026-07-22 12:32:59 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedavg__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-42` | PASS | 0.75 | 0.6208976864814758 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedavgm__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-42` | PASS | 0.940625 | 0.2835912951501086 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedmedian__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-42` | PASS | 0.5578125 | 0.9510888695716858 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedprox__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-42` | PASS | 0.359375 | 1.074967336654663 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedopt__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-42` | PASS | 0.9546875 | 0.4574632991664657 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedyogi__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-42` | PASS | 0.9578125 | 0.2213929073885083 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-42` | PASS | 0.76875 | 0.5736923143267632 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-42` | PASS | 0.9453125 | 0.27011424000374973 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-42` | PASS | 0.4984375 | 1.1741622775793075 | current-attempts; exit 0 |

- `2026-07-22 12:33:00 UTC` — VALIDATED current-attempt result main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-42 — PASS (not rerun)

- `2026-07-22 12:33:00 UTC` — STARTED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42` | FAIL |  |  | 90s; exit 1 |

- `2026-07-22 12:34:30 UTC` — COMPLETED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42 — FAIL (exit 1, 90s)
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-42` | PASS | 0.9546875 | 0.66376755728852 | current-attempts; exit 0 |

- `2026-07-22 12:34:30 UTC` — VALIDATED current-attempt result main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-42 — PASS (not rerun)
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-43` | PASS | 0.94375 | 0.3629816941800527 | current-attempts; exit 0 |

- `2026-07-22 12:34:30 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedavg__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-43` | PASS | 0.8984375 | 0.308837766200304 | current-attempts; exit 0 |

- `2026-07-22 12:34:30 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedavgm__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-43` | PASS | 0.9375 | 0.3733043486252427 | current-attempts; exit 0 |

- `2026-07-22 12:34:30 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedmedian__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-43` | PASS | 0.5328125 | 0.9261418014764786 | current-attempts; exit 0 |

- `2026-07-22 12:34:30 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedprox__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-43` | PASS | 0.4953125 | 1.071137297153473 | current-attempts; exit 0 |

- `2026-07-22 12:34:31 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedopt__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-43` | PASS | 0.9390625 | 0.4779103420863976 | current-attempts; exit 0 |

- `2026-07-22 12:34:31 UTC` — VALIDATED current-attempt result main__homogeneous__vanilla__fedyogi__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-43` | PASS | 0.934375 | 0.30373870697803795 | current-attempts; exit 0 |

- `2026-07-22 12:34:31 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedavg__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-43` | PASS | 0.7515625 | 0.5576686561107635 | current-attempts; exit 0 |

- `2026-07-22 12:34:31 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedavgm__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-43` | PASS | 0.9265625 | 0.35366105642169715 | current-attempts; exit 0 |

- `2026-07-22 12:34:31 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedmedian__noise-0.01__seed-43 — PASS (not rerun)

- `2026-07-22 12:34:31 UTC` — STARTED main__homogeneous__global-dp__fedprox__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-43` | PASS | 0.540625 | 0.9330323934555054 | 303s; exit 0 |

- `2026-07-22 12:36:43 UTC` — COMPLETED main__homogeneous__global-dp__fedprox__noise-0.01__seed-43 — PASS (exit 0, 303s)

- `2026-07-22 12:36:43 UTC` — STARTED main__homogeneous__global-dp__fedopt__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-43` | PASS | 0.528125 | 0.9586720436811447 | 341s; exit 0 |

- `2026-07-22 12:40:12 UTC` — COMPLETED main__homogeneous__global-dp__fedprox__noise-0.01__seed-43 — PASS (exit 0, 341s)

- `2026-07-22 12:40:12 UTC` — STARTED main__homogeneous__global-dp__fedopt__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-43` | PASS | 0.359375 | 35.809368419647214 | 345s; exit 0 |

- `2026-07-22 12:42:29 UTC` — COMPLETED main__homogeneous__global-dp__fedopt__noise-0.01__seed-43 — PASS (exit 0, 345s)

- `2026-07-22 12:42:29 UTC` — STARTED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-43` | PASS | 0.946875 | 0.3768413831014186 | 294s; exit 0 |

- `2026-07-22 12:47:23 UTC` — COMPLETED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-43 — PASS (exit 0, 294s)

- `2026-07-22 12:47:23 UTC` — STARTED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-43
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-43` | PASS | 0.9359375 | 0.306721996422857 | 218s; exit 0 |

- `2026-07-22 12:51:01 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-43 — PASS (exit 0, 218s)

- `2026-07-22 12:51:01 UTC` — STARTED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-43
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-43` | FAIL |  |  | 875s; exit 137 |

- `2026-07-22 12:54:47 UTC` — COMPLETED main__homogeneous__global-dp__fedopt__noise-0.01__seed-43 — FAIL (exit 137, 875s)
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-43` | PASS | 0.946875 | 0.3768413831014186 | current-attempts; exit 0 |

- `2026-07-22 12:54:48 UTC` — VALIDATED current-attempt result main__homogeneous__global-dp__fedyogi__noise-0.01__seed-43 — PASS (not rerun)
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-43` | PASS | 0.9359375 | 0.306721996422857 | current-attempts; exit 0 |

- `2026-07-22 12:54:48 UTC` — VALIDATED current-attempt result main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-43 — PASS (not rerun)

- `2026-07-22 12:54:48 UTC` — STARTED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-43
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-43` | PASS | 0.76875 | 0.5410368517041206 | 229s; exit 0 |

- `2026-07-22 12:54:50 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-43 — PASS (exit 0, 229s)

- `2026-07-22 12:54:50 UTC` — STARTED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-43
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-43` | PASS | 0.7859375 | 0.5208857387304306 | 345s; exit 0 |

- `2026-07-22 13:00:33 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-43 — PASS (exit 0, 345s)

- `2026-07-22 13:00:33 UTC` — STARTED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-43
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-43` | PASS | 0.91875 | 0.35135576240718364 | 228s; exit 0 |

- `2026-07-22 13:04:21 UTC` — COMPLETED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-43 — PASS (exit 0, 228s)

- `2026-07-22 13:04:21 UTC` — STARTED main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-43
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-43` | PASS | 0.359375 | 1.1851451575756073 | 238s; exit 0 |

- `2026-07-22 13:08:19 UTC` — COMPLETED main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-43 — PASS (exit 0, 238s)

- `2026-07-22 13:08:19 UTC` — STARTED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43` | FAIL |  |  | 104s; exit 1 |

- `2026-07-22 13:10:03 UTC` — COMPLETED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43 — FAIL (exit 1, 104s)

- `2026-07-22 13:10:03 UTC` — STARTED main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-43
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-43` | PASS | 0.9390625 | 0.43286938095698135 | 230s; exit 0 |

- `2026-07-22 13:13:53 UTC` — COMPLETED main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-43 — PASS (exit 0, 230s)

- `2026-07-22 13:13:53 UTC` — STARTED main__homogeneous__vanilla__fedavg__noise-0.01__seed-44
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-44` | PASS | 0.9546875 | 0.3653107556514442 | 210s; exit 0 |

- `2026-07-22 13:17:23 UTC` — COMPLETED main__homogeneous__vanilla__fedavg__noise-0.01__seed-44 — PASS (exit 0, 210s)

- `2026-07-22 13:17:23 UTC` — STARTED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-44
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-44` | PASS | 0.9125 | 0.3156677935272455 | 220s; exit 0 |

- `2026-07-22 13:21:03 UTC` — COMPLETED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-44 — PASS (exit 0, 220s)

- `2026-07-22 13:21:03 UTC` — STARTED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-44
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-44` | PASS | 0.934375 | 0.28801789730787275 | 217s; exit 0 |

- `2026-07-22 13:24:40 UTC` — COMPLETED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-44 — PASS (exit 0, 217s)

- `2026-07-22 13:24:40 UTC` — STARTED main__homogeneous__vanilla__fedprox__noise-0.01__seed-44
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-44` | PASS | 0.5078125 | 0.9785351395606995 | 229s; exit 0 |

- `2026-07-22 13:28:30 UTC` — COMPLETED main__homogeneous__vanilla__fedprox__noise-0.01__seed-44 — PASS (exit 0, 229s)

- `2026-07-22 13:28:30 UTC` — STARTED main__homogeneous__vanilla__fedopt__noise-0.01__seed-44
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-44` | PASS | 0.359375 | 1.0733623921871185 | 223s; exit 0 |

- `2026-07-22 13:32:13 UTC` — COMPLETED main__homogeneous__vanilla__fedopt__noise-0.01__seed-44 — PASS (exit 0, 223s)

- `2026-07-22 13:32:13 UTC` — STARTED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-44
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-44` | PASS | 0.9546875 | 0.6021389024099335 | 220s; exit 0 |

- `2026-07-22 13:35:53 UTC` — COMPLETED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-44 — PASS (exit 0, 220s)

- `2026-07-22 13:35:53 UTC` — STARTED main__homogeneous__global-dp__fedavg__noise-0.01__seed-44
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-44` | PASS | 0.9375 | 0.2306265017017722 | 218s; exit 0 |

- `2026-07-22 13:39:31 UTC` — COMPLETED main__homogeneous__global-dp__fedavg__noise-0.01__seed-44 — PASS (exit 0, 218s)

- `2026-07-22 13:39:31 UTC` — STARTED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-44
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-44` | PASS | 0.7765625 | 0.5750907167792321 | 227s; exit 0 |

- `2026-07-22 13:43:18 UTC` — COMPLETED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-44 — PASS (exit 0, 227s)

- `2026-07-22 13:43:18 UTC` — STARTED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-44
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-44` | PASS | 0.94375 | 0.2523373161442578 | 225s; exit 0 |

- `2026-07-22 13:47:03 UTC` — COMPLETED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-44 — PASS (exit 0, 225s)

- `2026-07-22 13:47:03 UTC` — STARTED main__homogeneous__global-dp__fedprox__noise-0.01__seed-44
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-44` | PASS | 0.5328125 | 0.940448722243309 | 238s; exit 0 |

- `2026-07-22 13:51:01 UTC` — COMPLETED main__homogeneous__global-dp__fedprox__noise-0.01__seed-44 — PASS (exit 0, 238s)

- `2026-07-22 13:51:01 UTC` — STARTED main__homogeneous__global-dp__fedopt__noise-0.01__seed-44
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-44` | PASS | 0.4953125 | 44.07766399383545 | 243s; exit 0 |

- `2026-07-22 13:55:05 UTC` — COMPLETED main__homogeneous__global-dp__fedopt__noise-0.01__seed-44 — PASS (exit 0, 243s)

- `2026-07-22 13:55:05 UTC` — STARTED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-44
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-44` | PASS | 0.94375 | 0.5317413987941109 | 228s; exit 0 |

- `2026-07-22 13:58:53 UTC` — COMPLETED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-44 — PASS (exit 0, 228s)

- `2026-07-22 13:58:53 UTC` — STARTED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-44
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-44` | PASS | 0.9484375 | 0.2542394444346428 | 219s; exit 0 |

- `2026-07-22 14:02:32 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-44 — PASS (exit 0, 219s)

- `2026-07-22 14:02:32 UTC` — STARTED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-44
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-44` | PASS | 0.775 | 0.5660552889108658 | 228s; exit 0 |

- `2026-07-22 14:06:20 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-44 — PASS (exit 0, 228s)

- `2026-07-22 14:06:20 UTC` — STARTED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-44
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-44` | PASS | 0.934375 | 0.31456311298534273 | 227s; exit 0 |

- `2026-07-22 14:10:07 UTC` — COMPLETED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-44 — PASS (exit 0, 227s)

- `2026-07-22 14:10:07 UTC` — STARTED main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-44
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-44` | PASS | 0.4953125 | 1.0595332950353622 | 238s; exit 0 |

- `2026-07-22 14:14:05 UTC` — COMPLETED main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-44 — PASS (exit 0, 238s)

- `2026-07-22 14:14:05 UTC` — STARTED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44` | FAIL |  |  | 247s; exit 1 |

- `2026-07-22 14:18:13 UTC` — COMPLETED main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44 — FAIL (exit 1, 247s)

- `2026-07-22 14:18:13 UTC` — STARTED main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-44
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-44` | PASS | 0.9546875 | 0.5916550347930751 | 231s; exit 0 |

- `2026-07-22 14:22:04 UTC` — COMPLETED main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-44 — PASS (exit 0, 231s)

- `2026-07-22 14:22:04 UTC` — STARTED main__homogeneous__vanilla__fedavg__noise-0.01__seed-45
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-45` | PASS | 0.959375 | 0.18579281428828837 | 206s; exit 0 |

- `2026-07-22 14:25:30 UTC` — COMPLETED main__homogeneous__vanilla__fedavg__noise-0.01__seed-45 — PASS (exit 0, 206s)

- `2026-07-22 14:25:30 UTC` — STARTED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-45
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-45` | PASS | 0.934375 | 0.23323795944452286 | 218s; exit 0 |

- `2026-07-22 14:29:08 UTC` — COMPLETED main__homogeneous__vanilla__fedavgm__noise-0.01__seed-45 — PASS (exit 0, 218s)

- `2026-07-22 14:29:08 UTC` — STARTED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-45
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-45` | PASS | 0.95625 | 0.2713696072343737 | 216s; exit 0 |

- `2026-07-22 14:32:44 UTC` — COMPLETED main__homogeneous__vanilla__fedmedian__noise-0.01__seed-45 — PASS (exit 0, 216s)

- `2026-07-22 14:32:44 UTC` — STARTED main__homogeneous__vanilla__fedprox__noise-0.01__seed-45
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-45` | PASS | 0.5046875 | 0.9701113432645798 | 230s; exit 0 |

- `2026-07-22 14:36:34 UTC` — COMPLETED main__homogeneous__vanilla__fedprox__noise-0.01__seed-45 — PASS (exit 0, 230s)

- `2026-07-22 14:36:34 UTC` — STARTED main__homogeneous__vanilla__fedopt__noise-0.01__seed-45
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-45` | PASS | 0.359375 | 1.1337204158306122 | 228s; exit 0 |

- `2026-07-22 14:40:22 UTC` — COMPLETED main__homogeneous__vanilla__fedopt__noise-0.01__seed-45 — PASS (exit 0, 228s)

- `2026-07-22 14:40:22 UTC` — STARTED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-45
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-45` | PASS | 0.9625 | 0.558505106859775 | 219s; exit 0 |

- `2026-07-22 14:44:01 UTC` — COMPLETED main__homogeneous__vanilla__fedyogi__noise-0.01__seed-45 — PASS (exit 0, 219s)

- `2026-07-22 14:44:01 UTC` — STARTED main__homogeneous__global-dp__fedavg__noise-0.01__seed-45
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-45` | PASS | 0.94375 | 0.22943491116166115 | 215s; exit 0 |

- `2026-07-22 14:47:36 UTC` — COMPLETED main__homogeneous__global-dp__fedavg__noise-0.01__seed-45 — PASS (exit 0, 215s)

- `2026-07-22 14:47:36 UTC` — STARTED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-45
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-45` | PASS | 0.775 | 0.5948600724339486 | 233s; exit 0 |

- `2026-07-22 14:51:29 UTC` — COMPLETED main__homogeneous__global-dp__fedavgm__noise-0.01__seed-45 — PASS (exit 0, 233s)

- `2026-07-22 14:51:29 UTC` — STARTED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-45
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-45` | PASS | 0.940625 | 0.250091852247715 | 226s; exit 0 |

- `2026-07-22 14:55:15 UTC` — COMPLETED main__homogeneous__global-dp__fedmedian__noise-0.01__seed-45 — PASS (exit 0, 226s)

- `2026-07-22 14:55:15 UTC` — STARTED main__homogeneous__global-dp__fedprox__noise-0.01__seed-45
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-45` | PASS | 0.5609375 | 0.9120569586753845 | 237s; exit 0 |

- `2026-07-22 14:59:13 UTC` — COMPLETED main__homogeneous__global-dp__fedprox__noise-0.01__seed-45 — PASS (exit 0, 237s)

- `2026-07-22 14:59:13 UTC` — STARTED main__homogeneous__global-dp__fedopt__noise-0.01__seed-45
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-45` | PASS | 0.4953125 | 1.068834227323532 | 245s; exit 0 |

- `2026-07-22 15:03:18 UTC` — COMPLETED main__homogeneous__global-dp__fedopt__noise-0.01__seed-45 — PASS (exit 0, 245s)

- `2026-07-22 15:03:18 UTC` — STARTED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-45
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-45` | PASS | 0.95625 | 0.4711193893803284 | 231s; exit 0 |

- `2026-07-22 15:07:09 UTC` — COMPLETED main__homogeneous__global-dp__fedyogi__noise-0.01__seed-45 — PASS (exit 0, 231s)

- `2026-07-22 15:07:09 UTC` — STARTED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-45
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-45` | PASS | 0.940625 | 0.24046224681660533 | 219s; exit 0 |

- `2026-07-22 15:10:48 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-45 — PASS (exit 0, 219s)

- `2026-07-22 15:10:48 UTC` — STARTED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-45
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-45` | PASS | 0.7453125 | 0.6641025125980378 | 227s; exit 0 |

- `2026-07-22 15:14:36 UTC` — COMPLETED main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-45 — PASS (exit 0, 227s)

- `2026-07-22 15:14:36 UTC` — STARTED main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-45

-  — Scheduling stopped for machine switch. Preserved 65 distinct completed matrix JSONs with final centralized accuracy/loss in . One in-flight run was interrupted at round 4 and is not counted. No further experiments are scheduled.

- 2026-07-22 15:16 UTC — CORRECTION: Scheduling stopped for machine switch. Preserved 65 distinct completed matrix JSONs with final centralized accuracy/loss in results/paper_reproduction_preswitch_inventory.tsv. One in-flight run was interrupted at round 4 and is not counted. No further experiments are scheduled.
