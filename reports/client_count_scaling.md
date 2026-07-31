# Client-Count Scaling: 8 vs 48 Clients

Source data: `results/8client_scaling/`, `results/48client_scaling/`, `results/noise_sweep/`,
`results/sigma_calibration/`, `results/fedyogi_rerun_4clients/`, `results/probe_fedopt_fedyogi/`.
Code: `experiments/client_scaling/`.

## Overview

This experiment asks whether the paper's reported vanilla/global-DP/metric-privacy comparison
holds up as the number of clients grows well past the paper's 4-client setup. It runs the same
Alzheimer MRI classification task, `metricdp_pytorch` strategy factory, and
`experiments.reproduce.runner` machinery at 8 and 48 clients, then separately sweeps the DP noise
multiplier to characterize each mechanism's collapse point. All runs use the paper's architecture,
20 rounds, 5 local epochs, batch 32, learning rate 0.001, clipping norm 5.0, and seed 42 unless
noted otherwise.

Six result sets feed this report:

| Result set | Clients | Aggregators | Privacy modes | Noise multiplier(s) | Runs |
|---|---:|---|---|---|---:|
| `8client_scaling` | 8 | fedavg, fedavgm, fedyogi | vanilla, global-dp, metric-privacy | 0.01 | 18 |
| `48client_scaling` | 48 | fedavg, fedavgm, fedyogi | vanilla, global-dp, metric-privacy | 0.05 | 18 |
| `noise_sweep` | 8 | fedavg | global-dp, metric-privacy | 0.01, 0.05, 0.1, 0.25, 0.5, 1.0 attempted | 11/24 (13 failed, see below) |
| `sigma_calibration` | 48 | fedyogi | vanilla, global-dp, metric-privacy | 0.12, 0.3, 0.6 (sigma-matched, see below) | 13/14 |
| `fedyogi_rerun_4clients` | 4 | fedyogi | vanilla, global-dp, metric-privacy | 0.01 | 20/20 |
| `probe_fedopt_fedyogi` | 4 | fedopt, fedyogi | vanilla, global-dp, metric-privacy | 0.01 | 4 (design probe, not a matrix) |

FedProx and FedMedian, which are part of the paper-reproduction matrix (see
`reports/paper_reproduction.md`), are not part of these scaling sweeps.
FedOpt is deliberately excluded from the scaling matrices — see the FedOpt/FedYogi probe below.

## Why sigma, not noise_multiplier, is the cross-scale unit

Flower's DP-FedAvg noise stdev is `sigma = noise_multiplier * clipping_norm / num_clients`
(`flwr.supercore.differential_privacy.compute_stdv`). The `1/N` factor is not a library quirk —
it is the standard Gaussian-mechanism calibration for an *average* of `N` updates each clipped to
`C`: any single client's contribution to that average has sensitivity `C/N` (McMahan et al. 2018,
arXiv:1710.06963). `noise_multiplier` is the actual privacy parameter, so a fixed value carries the
same guarantee at any cohort size — what changes with `N` is the noise magnitude actually applied
to the model, and with it the accuracy cost.

Consequently, `8client_scaling` (nm=0.01, sigma=0.01·5/8=6.25e-3) and `48client_scaling`
(nm=0.05, sigma=0.05·5/48=5.21e-3) are **not** sigma-matched — they happen to sit at similar noise
multipliers, but not the same effective perturbation. `noise_sweep` and `sigma_calibration` fix
this: `noise_sweep`'s nm=0.05/0.1 at 8 clients land at sigma=3.125e-2/6.25e-2, and
`sigma_calibration`'s nm=0.3/0.6 at 48 clients were chosen specifically to reproduce those same two
sigma values (`nm = sigma * N / C`), so the two sweeps together give a genuine same-sigma,
different-`N` comparison (see Finding 3).

## Results: 8-client scaling (nm=0.01)

| Partition | Privacy | Aggregator | Accuracy | Loss | Macro F1 | Macro precision | Macro AUC | Distance min/mean/max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| homogeneous | global-dp | fedavg | 0.9016 | 0.3763 | 0.9017 | 0.9021 | 0.9806 | n/a |
| homogeneous | global-dp | fedavgm | 0.7719 | 0.5806 | 0.7666 | 0.7648 | 0.9409 | n/a |
| homogeneous | global-dp | fedyogi | 0.9563 | 0.3920 | 0.9562 | 0.9563 | 0.9916 | n/a |
| homogeneous | metric-privacy | fedavg | 0.8969 | 0.4017 | 0.8970 | 0.8977 | 0.9796 | 0.7686/0.9335/1.2477 |
| homogeneous | metric-privacy | fedavgm | 0.7688 | 0.5769 | 0.7637 | 0.7616 | 0.9415 | 1.1286/1.1723/1.1983 |
| homogeneous | metric-privacy | fedyogi | 0.9563 | 0.4936 | 0.9562 | 0.9574 | 0.9905 | 0.7959/0.9454/1.2049 |
| homogeneous | vanilla | fedavg | 0.9359 | 0.3043 | 0.9357 | 0.9359 | 0.9873 | n/a |
| homogeneous | vanilla | fedavgm | 0.8469 | 0.4180 | 0.8421 | 0.8386 | 0.9698 | n/a |
| homogeneous | vanilla | fedyogi | 0.9453 | 0.3721 | 0.9454 | 0.9468 | 0.9907 | n/a |
| non-iid | global-dp | fedavg | 0.9000 | 0.3865 | 0.9002 | 0.9048 | 0.9790 | n/a |
| non-iid | global-dp | fedavgm | 0.7797 | 0.5487 | 0.7766 | 0.7754 | 0.9472 | n/a |
| non-iid | global-dp | fedyogi | 0.9625 | 0.3603 | 0.9624 | 0.9636 | 0.9924 | n/a |
| non-iid | metric-privacy | fedavg | 0.9094 | 0.3389 | 0.9094 | 0.9105 | 0.9830 | 0.8419/0.9982/1.4650 |
| non-iid | metric-privacy | fedavgm | 0.7844 | 0.5474 | 0.7809 | 0.7787 | 0.9475 | 1.2913/1.3595/1.4132 |
| non-iid | metric-privacy | fedyogi | 0.9641 | 0.3698 | 0.9639 | 0.9645 | 0.9919 | 0.9085/1.0845/1.3381 |
| non-iid | vanilla | fedavg | 0.9141 | 0.3676 | 0.9145 | 0.9197 | 0.9845 | n/a |
| non-iid | vanilla | fedavgm | 0.8578 | 0.3769 | 0.8531 | 0.8484 | 0.9760 | n/a |
| non-iid | vanilla | fedyogi | 0.9578 | 0.3328 | 0.9579 | 0.9588 | 0.9927 | n/a |

## Results: 48-client scaling (nm=0.05)

| Partition | Privacy | Aggregator | Accuracy | Loss | Macro F1 | Macro precision | Macro AUC | Distance min/mean/max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| homogeneous | global-dp | fedavg | 0.6203 | 0.8288 | 0.6039 | 0.6031 | 0.8726 | n/a |
| homogeneous | global-dp | fedavgm | 0.6875 | 0.7321 | 0.6829 | 0.6786 | 0.9080 | n/a |
| homogeneous | global-dp | fedyogi | 0.9094 | 0.4284 | 0.9092 | 0.9113 | 0.9810 | n/a |
| homogeneous | metric-privacy | fedavg | 0.6266 | 0.8360 | 0.6187 | 0.6131 | 0.8747 | 0.5276/0.5778/0.8549 |
| homogeneous | metric-privacy | fedavgm | 0.6984 | 0.7338 | 0.6946 | 0.6909 | 0.9073 | 0.5545/0.5896/0.6025 |
| homogeneous | metric-privacy | fedyogi | 0.8922 | 0.4044 | 0.8922 | 0.8963 | 0.9787 | 0.4571/0.5495/0.6293 |
| homogeneous | vanilla | fedavg | 0.6094 | 0.8557 | 0.5890 | 0.6013 | 0.8622 | n/a |
| homogeneous | vanilla | fedavgm | 0.6859 | 0.7333 | 0.6796 | 0.6751 | 0.9081 | n/a |
| homogeneous | vanilla | fedyogi | 0.9109 | 0.3974 | 0.9111 | 0.9121 | 0.9822 | n/a |
| non-iid | global-dp | fedavg | 0.6156 | 0.8240 | 0.6061 | 0.6313 | 0.8745 | n/a |
| non-iid | global-dp | fedavgm | 0.7125 | 0.7031 | 0.7091 | 0.7067 | 0.9123 | n/a |
| non-iid | global-dp | fedyogi | 0.8953 | 0.4060 | 0.8904 | 0.8856 | 0.9835 | n/a |
| non-iid | metric-privacy | fedavg | 0.6375 | 0.8179 | 0.6317 | 0.6469 | 0.8780 | 0.5966/0.6756/1.0097 |
| non-iid | metric-privacy | fedavgm | 0.7156 | 0.7060 | 0.7127 | 0.7115 | 0.9116 | 0.6469/0.6847/0.7137 |
| non-iid | metric-privacy | fedyogi | 0.8938 | 0.4062 | 0.8889 | 0.8845 | 0.9813 | 0.5139/0.6455/0.7425 |
| non-iid | vanilla | fedavg | 0.6172 | 0.8239 | 0.6139 | 0.6164 | 0.8756 | n/a |
| non-iid | vanilla | fedavgm | 0.7063 | 0.7006 | 0.7012 | 0.6970 | 0.9130 | n/a |
| non-iid | vanilla | fedyogi | 0.8922 | 0.3831 | 0.8875 | 0.8835 | 0.9838 | n/a |

## Results: noise-multiplier sweep, 8 clients, FedAvg only

| Partition | Privacy | Noise multiplier | Sigma | Accuracy | Loss | Macro F1 | Macro precision | Macro AUC | Distance min/mean/max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | global-dp | 0.01 | 6.25e-3 | 0.9094 | 0.3453 | 0.9093 | 0.9097 | 0.9824 | n/a |
| homogeneous | global-dp | 0.05 | 3.13e-2 | 0.7516 | 0.7435 | 0.7420 | 0.7389 | 0.9303 | n/a |
| homogeneous | global-dp | 0.1 | 6.25e-2 | 0.4953 | 1.0517 | 0.3281 | 0.2453 | 0.7797 | n/a |
| homogeneous | metric-privacy | 0.01 | 6.25e-3 | 0.8922 | 0.3924 | 0.8926 | 0.8947 | 0.9791 | 0.7602/0.9165/1.2477 |
| homogeneous | metric-privacy | 0.05 | 3.13e-2 | 0.8203 | 0.5815 | 0.8144 | 0.8100 | 0.9619 | 1.1890/1.3309/1.4080 |
| non-iid | global-dp | 0.01 | 6.25e-3 | 0.9156 | 0.3557 | 0.9158 | 0.9178 | 0.9828 | n/a |
| non-iid | global-dp | 0.05 | 3.13e-2 | 0.7281 | 0.8098 | 0.7129 | 0.7313 | 0.9300 | n/a |
| non-iid | global-dp | 0.1 | 6.25e-2 | 0.4953 | 1.0526 | 0.3281 | 0.2453 | 0.7797 | n/a |
| non-iid | metric-privacy | 0.01 | 6.25e-3 | 0.9219 | 0.3166 | 0.9219 | 0.9221 | 0.9841 | 0.8659/1.0008/1.4650 |
| non-iid | metric-privacy | 0.05 | 3.13e-2 | 0.8500 | 0.4795 | 0.8442 | 0.8408 | 0.9703 | 1.3040/1.3910/1.5410 |
| non-iid | metric-privacy | 0.1 | 6.25e-2 | 0.4969 | 43.9412 | 0.3316 | 0.6051 | 0.6646 | 0.0254/0.8630/1.4650 |

The table above shows only the completed runs. The sweep actually attempted 24 combinations
(`noise_multiplier` ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 1.0} × 2 partitions × 2 privacy modes), and
`results/noise_sweep/sweep_progress.log` records 13 failures (exit code 1):

- `homogeneous / metric-privacy / nm=0.1` — the one gap in the table above. It ran for 364.7s before
  failing, consistent with a mid-training crash rather than an immediate configuration error.
- Every `nm ∈ {0.25, 0.5, 1.0}` combination failed, for **both** global-dp and metric-privacy, in
  both partitions (12 further failures). Durations shrink as `nm` grows (up to 250.6s at nm=0.25,
  down to 55.8–90.7s at nm=1.0), suggesting an earlier and earlier crash as noise increases.

The committed log only records exit status and wall-clock duration, not a traceback, so the exact
exception per failure is not available from this repository. Because global-dp fails alongside
metric-privacy at `nm ≥ 0.25`, these failures are not simply instances of the metric-privacy
non-finite-distance guard (which is specific to `metric-privacy` and does not apply to global-dp) —
something else about very high noise multipliers breaks training generally at 8 clients. This is
unresolved; the sweep was not re-run with `--force` or extended logging to isolate it.

## Results: sigma-calibrated sweep, 48 clients, FedYogi only

Chosen to reproduce the three regimes seen in the noise sweep above (edge-of-threshold,
mid-collapse, full collapse) at a client count where `noise_multiplier` alone would be
incomparable. FedYogi is used here instead of FedAvg because, at 48 clients without any DP noise,
FedAvg does not converge within 20 rounds (vanilla FedAvg was not part of this sweep; FedYogi's own
vanilla reaches 0.8688–0.8812, see table). The swept multipliers are a strictly stronger privacy
guarantee than the paper's 0.01 operating point — chosen for measurability at 48 clients, not to
reproduce the paper's setting.

| Partition | Privacy | Noise multiplier (nm) | Sigma | Accuracy | Loss | Macro F1 | Macro precision | Macro AUC | Distance min/mean/max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | vanilla | 0.01 | n/a | 0.8688 | 0.5017 | 0.8688 | 0.8716 | 0.9715 | n/a |
| homogeneous | global-dp | 0.12 | 1.25e-2 | 0.8453 | 0.5316 | 0.8451 | 0.8489 | 0.9705 | n/a |
| homogeneous | global-dp | 0.3 | 3.13e-2 | 0.6266 | 1.1765 | 0.5793 | 0.6283 | 0.8732 | n/a |
| homogeneous | global-dp | 0.6 | 6.25e-2 | 0.4953 | 1.1311 | 0.3281 | 0.2453 | 0.7797 | n/a |
| homogeneous | metric-privacy | 0.12 | 1.25e-2 | 0.8422 | 0.5319 | 0.8412 | 0.8510 | 0.9660 | 0.5883/0.6613/0.7230 |
| homogeneous | metric-privacy | 0.3 | 3.13e-2 | **FAILED** | — | — | — | — | non-finite at round of failure |
| homogeneous | metric-privacy | 0.6 | 6.25e-2 | 0.4953 | 2.8147 | 0.3281 | 0.2453 | 0.7049 | 0.1368/0.4111/0.5725 |
| non-iid | vanilla | 0.01 | n/a | 0.8812 | 0.4541 | 0.8802 | 0.8818 | 0.9789 | n/a |
| non-iid | global-dp | 0.12 | 1.25e-2 | 0.8297 | 0.4982 | 0.8245 | 0.8213 | 0.9675 | n/a |
| non-iid | global-dp | 0.3 | 3.13e-2 | 0.6562 | 1.0451 | 0.6468 | 0.6611 | 0.9034 | n/a |
| non-iid | global-dp | 0.6 | 6.25e-2 | 0.4953 | 1.1263 | 0.3281 | 0.2453 | 0.7797 | n/a |
| non-iid | metric-privacy | 0.12 | 1.25e-2 | 0.8359 | 0.5104 | 0.8310 | 0.8266 | 0.9706 | 0.6179/0.7320/0.8060 |
| non-iid | metric-privacy | 0.3 | 3.13e-2 | 0.4984 | 43.8047 | 0.3350 | 0.6055 | 0.6656 | 0.0162/0.4938/0.7821 |
| non-iid | metric-privacy | 0.6 | 6.25e-2 | 0.4953 | 44.0777 | 0.3281 | 0.2453 | 0.6635 | 0.0173/0.4397/0.7217 |

`homogeneous / metric-privacy / nm=0.3` is a confirmed failure, not a missing run: per the commit
that added this sweep, the maximum pairwise client-model distance went non-finite at
sigma=3.125e-2, so the strategy's `noise_multiplier / distance` calibration is undefined and the
run aborts by design (same fail-fast policy documented for FedOpt in
`reports/paper_reproduction.md`). 13/14 configurations completed.

## Baselines feeding this comparison

- **`fedyogi_rerun_4clients`** (20/20 complete): the paper's native 4-client setting, FedYogi only,
  5 homogeneous seeds. Accuracy: vanilla 0.9472±0.0084, global-dp 0.9531±0.0083,
  metric-privacy 0.9538±0.0091 — i.e. at 4 clients and nm=0.01, metric-privacy and global-dp are
  statistically indistinguishable and both track vanilla closely. Full detail in
  `results/fedyogi_rerun_4clients/README.md`.
- **`probe_fedopt_fedyogi`** (4-run design probe, homogeneous, seed 42, nm=0.01): the reason FedOpt
  is excluded from every scaling sweep above. FedOpt collapses to majority-class accuracy (final
  0.4953 for vanilla, oscillating down to 0.4409 for global-dp) regardless of privacy mode, and the
  metric-privacy/FedOpt combination hits the same
  non-finite-distance failure as above (there at round 11 with `tau=1e-3`, vs. round 4 with the
  default `tau=1e-9` reported in `reports/paper_reproduction.md`). FedYogi
  converges cleanly under the same probe (0.9641 final, 0.9563 last-5 mean), which is why it was
  chosen as the aggregator for `sigma_calibration`. Full detail in
  `results/probe_fedopt_fedyogi/README.md`.

## Findings

1. **FedAvg and FedAvgM do not scale from 8 to 48 clients at fixed `noise_multiplier`.** Homogeneous
   vanilla FedAvg drops from 0.9359 (8 clients) to 0.6094 (48 clients); FedAvgM drops from 0.8469 to
   0.6859. FedYogi is far more scale-robust: vanilla FedYogi only drops from 0.9453 to 0.9109. This
   holds under every privacy mode, not just vanilla — see the two scaling tables above.

2. **At matched `noise_multiplier` (not matched sigma), metric-privacy tracks global-dp closely and
   is not systematically worse.** Across the 8-client (nm=0.01) and 48-client (nm=0.05) scaling
   tables, the metric-privacy vs. global-dp accuracy delta is small and sign-mixed: it ranges from
   −0.0172 (48-client FedYogi, homogeneous) to +0.0219 (48-client FedAvg, non-iid), and in one case
   (8-client homogeneous FedYogi) the two are accuracy-identical to four decimal places
   (0.95625 vs 0.95625).

3. **At matched sigma, the two mechanisms share a collapse threshold that is independent of client
   count and (largely) of aggregator.** `noise_sweep`'s 8-client FedAvg run at sigma=6.25e-2 and
   `sigma_calibration`'s 48-client FedYogi run at the same sigma both land at accuracy 0.4953125 —
   the dataset's majority-class rate — under global-dp, in both homogeneous and non-iid partitions.
   The same sigma also collapses metric-privacy to the same accuracy where it completes at all.

4. **Metric-privacy's failure mode under high noise is qualitatively worse than global-dp's, not
   just occasionally a hard failure.** Where global-dp collapses gracefully to near-uniform
   predictions (loss ≈ 1.05–1.18 at the collapse threshold), metric-privacy's collapse produces
   wildly overconfident wrong predictions instead: loss explodes to 43.8–44.1 in three of the four
   sigma=3.13e-2/6.25e-2 metric-privacy runs that did complete, alongside a calibration distance
   that has collapsed to near zero (min distance as low as 0.0162–0.0254). The fourth
   (`sigma_calibration` homogeneous, sigma=3.13e-2) does not "complete degenerate" at all — it hits
   the non-finite-distance guard and aborts. `MetricPrivacyServerSideFixedClipping`'s
   `noise_multiplier / distance` calibration is undefined once client updates converge to
   near-identical vectors; the strategy raises rather than silently dividing by (near) zero, which
   is a correctness choice, not a bug, but it means metric-privacy's practical safety margin before
   collapse is narrower than global-dp's at high client counts / high noise.

5. **The 8-client and 48-client scaling sweeps are not directly sigma-comparable.** They were run at
   matched `noise_multiplier` (0.01 vs 0.05), not matched sigma (6.25e-3 vs 5.21e-3) — see "Why
   sigma, not noise_multiplier" above. Only `noise_sweep` and `sigma_calibration` are genuinely
   sigma-matched to each other, and even that comparison mixes aggregators (FedAvg at 8 clients vs.
   FedYogi at 48 clients) because FedAvg does not converge at 48 clients within the round budget.

## Known gaps / caveats

- All scaling and noise-sweep runs use a single seed (42); only `fedyogi_rerun_4clients` has
  multi-seed variance estimates, and only for the 4-client setting. The deltas in Finding 2 are not
  backed by seed-level confidence intervals.
- `noise_sweep` and `48client_scaling`/`8client_scaling` only cover FedAvg/FedAvgM/FedYogi; FedProx
  and FedMedian (part of the paper-reproduction matrix) are untested at 8 or 48 clients.
- `noise_sweep`'s 13 failures (`nm=0.1` metric-privacy homogeneous, plus every `nm ≥ 0.25`
  combination) have no traceback in the repository, only exit status and duration. The cause of the
  global-dp failures at high `nm` in particular is unexplained and unresolved.
- The 48-client sigma-calibration sweep moved client training onto GPU (`--client-gpus 0.08`) to
  avoid multi-minute HuggingFace rate-limit backoff at 12 concurrent actors; results are therefore
  not bit-comparable with the CPU-only `48client_scaling`/`noise_sweep` runs, only comparable at the
  level of aggregate accuracy/loss.
