# CIFAR-10 noise-ratio sweep and 48-client validation

## Purpose and scope

This report documents the two accuracy-only runs added in the `PLAN.md` update
that unblocks the replacement-adjacency client-scaling experiment:

1. a 3-active-client noise-ratio sweep, and
2. a 48-active-client validation at the same ratios.

The purpose is utility calibration, not a CIA measurement.  No checkpoints,
shadow-loss evaluations, or IN/OUT AUCs were generated for these chunks.  The
results therefore say which ratios retain classification utility; they do not
establish the privacy or attack behavior of the subsequent full runs.

All values below are read from the recorded run and evaluation JSON artifacts
under `results/planned_runs/cifar/`.  Each configuration has one calibration
seed (42), so differences should be treated as directional rather than as
multi-seed estimates.

## Protocol

| Item | 3-client sweep | 48-client validation |
|---|---|---|
| Dataset | filtered CIFAR-10: airplane, automobile, bird, cat | same |
| Adjacency | IN-replace | IN-replace |
| Active / canonical clients | 3 / 4 | 48 / 49 |
| Partition and aggregation | non-IID, FedAvg | non-IID, FedAvg |
| Training | 20 rounds, 5 local epochs, batch size 32, learning rate 0.001 | same |
| Seed | 42 | 42 |
| Privacy modes | Vanilla once; Global-DP and metric-privacy per ratio | same |
| Clipping norm | 5.0 | 5.0 |
| Held-out server test set | 2,000 examples (500 per class) | same |

For the non-vanilla runs, the planned noise ratio is
`noise_multiplier / active_clients`.  Flower's applied aggregate-noise
standard deviation is `sigma = noise_multiplier * clipping_norm / active_clients`;
with clipping norm 5, this is five times the reported ratio.  This makes the
three ratios directly comparable between 3 and 48 active clients, although
the trained models still differ because the client partition changes.

| Noise ratio | 3-client multiplier | 48-client multiplier | Applied sigma |
|---:|---:|---:|---:|
| 0.002500 | 0.007500 | 0.120000 | 0.012500 |
| 0.003333 | 0.009999 | 0.159984 | 0.016665 |
| 0.006250 | 0.018750 | 0.300000 | 0.031250 |

The vanilla controls use the repository's baseline multiplier field of 0.01,
but no DP perturbation is applied.

## Results: 3 active clients

Final metrics are server-test metrics after round 20.  The trajectory columns
show accuracy at selected communication rounds and make a late-round collapse
visible without implying an average over seeds.

| Privacy | Ratio | Final accuracy | Macro F1 | Log loss | Macro OVR AUC | Accuracy at rounds 1 / 5 / 10 / 15 / 20 |
|---|---:|---:|---:|---:|---:|---|
| Vanilla | -- | 0.8135 | 0.8129 | 1.0808 | 0.9544 | 0.6690 / 0.8155 / 0.8160 / 0.8160 / 0.8135 |
| Global-DP | 0.002500 | 0.8140 | 0.8146 | 0.8537 | 0.9535 | 0.4795 / 0.8100 / 0.8070 / 0.8230 / 0.8140 |
| Metric-privacy | 0.002500 | 0.8185 | 0.8189 | 0.9047 | 0.9539 | 0.4445 / 0.8130 / 0.8320 / 0.8250 / 0.8185 |
| Global-DP | 0.003333 | 0.8120 | 0.8132 | 0.8052 | 0.9519 | 0.4340 / 0.7900 / 0.7980 / 0.8260 / 0.8120 |
| Metric-privacy | 0.003333 | 0.8225 | 0.8224 | 0.8547 | 0.9524 | 0.4630 / 0.8200 / 0.8115 / 0.8230 / 0.8225 |
| Global-DP | 0.006250 | 0.6885 | 0.6943 | 1.2711 | 0.9095 | 0.2560 / 0.6890 / 0.6800 / 0.7880 / 0.6885 |
| Metric-privacy | 0.006250 | 0.8225 | 0.8230 | 0.7941 | 0.9553 | 0.4765 / 0.8025 / 0.8115 / 0.8260 / 0.8225 |

At 3 clients, ratios 0.002500 and 0.003333 retain vanilla-level accuracy for
both mechanisms.  At 0.006250, Global-DP falls 12.5 percentage points below
vanilla (0.6885 vs. 0.8135), while metric-privacy remains at 0.8225.  This is
mechanism-specific evidence, not a general clearance for the highest ratio.

## Results: 48 active clients

| Privacy | Ratio | Final accuracy | Macro F1 | Log loss | Macro OVR AUC | Accuracy at rounds 1 / 5 / 10 / 15 / 20 |
|---|---:|---:|---:|---:|---:|---|
| Vanilla | -- | 0.8025 | 0.8016 | 0.5363 | 0.9503 | 0.4305 / 0.6435 / 0.7640 / 0.7940 / 0.8025 |
| Global-DP | 0.002500 | 0.7855 | 0.7870 | 0.5967 | 0.9434 | 0.4670 / 0.6715 / 0.7235 / 0.7685 / 0.7855 |
| Metric-privacy | 0.002500 | 0.7820 | 0.7826 | 0.6357 | 0.9391 | 0.4660 / 0.6575 / 0.7050 / 0.7580 / 0.7820 |
| Global-DP | 0.003333 | 0.7775 | 0.7790 | 0.6516 | 0.9378 | 0.4195 / 0.6580 / 0.6810 / 0.7575 / 0.7775 |
| Metric-privacy | 0.003333 | 0.7550 | 0.7565 | 0.7367 | 0.9288 | 0.4205 / 0.6445 / 0.6525 / 0.7470 / 0.7550 |
| Global-DP | 0.006250 | 0.6635 | 0.6619 | 1.0295 | 0.8898 | 0.3360 / 0.6230 / 0.4955 / 0.6335 / 0.6635 |
| Metric-privacy | 0.006250 | 0.6505 | 0.6457 | 1.0472 | 0.8778 | 0.3390 / 0.5900 / 0.5000 / 0.6570 / 0.6505 |

The 48-client validation reverses the reassuring three-client result at the
highest ratio.  Relative to vanilla, accuracy losses are 1.7 and 2.1 points at
0.002500, 2.5 and 4.8 points at 0.003333, and 13.9 and 15.2 points at 0.006250
for Global-DP and metric-privacy respectively.  Thus 0.006250 is destructive
for both mechanisms at the intended upper client scale.

## Cross-scale interpretation

| Ratio | 3-client Global-DP / metric accuracy | 48-client Global-DP / metric accuracy | Interpretation |
|---:|---|---|---|
| 0.002500 | 0.8140 / 0.8185 | 0.7855 / 0.7820 | Best utility-preserving common setting. |
| 0.003333 | 0.8120 / 0.8225 | 0.7775 / 0.7550 | Usable, but metric-privacy loses more utility at 48 clients. |
| 0.006250 | 0.6885 / 0.8225 | 0.6635 / 0.6505 | Not suitable as a general scaling ratio. |

The final-round diagnostic recorded in the run JSONs is consistent with the
utility results: at the same applied sigma, the 48-client runs have much
larger noise-to-signal ratios than the 3-client runs.  For example, it is
4.93 (Global-DP) and 2.36 (metric-privacy) at ratio 0.006250 with three
clients, versus 16.12 and 17.96 with 48 clients.  This is an empirical
training diagnostic, not a privacy accounting result.

## Decision supported by this validation

- Use 0.002500 as the conservative common ratio for the held-off 3/8/16/48
  client replacement experiments.
- Keep 0.003333 as a second, moderately stronger-noise operating point; its
  48-client metric-privacy accuracy is 0.7550, so it carries a visible but
  non-collapse utility cost.
- Do not use 0.006250 as a default in the full scaling grid.  It is destructive
  at 48 clients for both mechanisms, even though it appeared safe for
  three-client metric-privacy.

`PLAN.md` asks for an additional ratio of 0.005 if 0.006250 is destructive.
No `ratio-0p005` artifact exists in either the sweep or validation directory,
so this follow-up has not been completed and this report does not interpolate
its outcome.  If a third intermediate operating point is desired, run 0.005
at both 3 and 48 clients before selecting it for the full grid.

## Artifact inventory and limitations

- The sweep contains the one vanilla control plus six DP trajectories: three
  ratios times two mechanisms.  All seven manifests/runs are complete.
- The validation likewise contains the one vanilla control plus six DP
  trajectories.  All seven manifests/runs are complete.
- Both groups use only seed 42 and report accuracy-only evaluation.  They are
  calibration evidence, not uncertainty estimates and not multi-round CIA
  evidence.
- The 48-client task has 48 active participants and 49 canonical partitions
  because IN-replace holds out one replacement partition; the 3-client task
  analogously uses 3 active participants and 4 canonical partitions.

### Source artifacts

- `results/planned_runs/cifar/noise_sweep/`
- `results/planned_runs/cifar/validation_48/`
- `experiments/cia/scripts/cifar_chunks.py`
- `PLAN.md` (the "Update: unblocking above blockers" section)
