# FedOpt/FedYogi Initial Probe

Commit under test: `6b0bb92` (merged into master `5ffafba`).

Settings: homogeneous four-client split, seed 42, 20 rounds, 5 local epochs, batch 32, LR 0.001, noise 0.01, clipping norm 5, 20 initialization epochs.

| Privacy | Aggregator | Outcome | Initial server accuracy | Final server accuracy | Last-5 server accuracy | Distance finding |
|---|---|---|---:|---:|---:|---|
| vanilla | FedOpt | Completed but numerically collapsed | 0.606250 | 0.495312 | 0.495312 | n/a |
| global-DP | FedOpt | Completed but oscillated/collapsed | 0.615625 | 0.495312 | 0.440938 | n/a |
| metric-privacy | FedOpt | Failed at round 11 | — | — | — | Round-10 distance 0.00976335; round-11 distance exactly 0, so calibration would divide by zero |
| metric-privacy | FedYogi | Completed and converged | 0.621875 | 0.964063 | 0.956250 | Minimum 1.0805874; final 1.1010148; all positive |

## Assessment

The change from FedOpt `tau=1e-9` to `tau=1e-3` delays the metric-privacy failure (previously round 4, now round 11), but does not fix FedOpt. Even without metric privacy, FedOpt collapses to majority-class accuracies and alternates between approximately 0.3594 and 0.4953. The metric-private run still reaches an exact zero client-model distance and is rejected rather than applying an undocumented divide-by-zero workaround.

FedYogi passes this probe and shows smooth convergence. A larger FedYogi matrix is reasonable; a larger FedOpt matrix is not recommended until its update behavior is corrected.

Artifacts are under `runs/` and `logs/`. The failed FedOpt metric-private run has no final JSON; its complete trace is retained in its log.
