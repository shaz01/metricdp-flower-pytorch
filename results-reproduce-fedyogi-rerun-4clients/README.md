# Four-client FedYogi rerun

Twenty FedYogi cells from the paper 120-run matrix: 15 homogeneous main, 3 non-IID main, and 2 Appendix-B noise=0.003. FedOpt is excluded.

Complete: **20/20**. Detailed metrics and raw predictions are stored in `runs/`.

## Homogeneous main five-seed summary

| Privacy | Accuracy mean ± SD | Macro F1 mean ± SD | Macro precision mean ± SD | Macro AUC mean ± SD | n |
|---|---:|---:|---:|---:|---:|
| vanilla | 0.947187 ± 0.008371 | 0.928752 ± 0.041183 | 0.919462 ± 0.063808 | 0.985959 ± 0.004745 | 5 |
| global-dp | 0.953125 ± 0.008268 | 0.941244 ± 0.018249 | 0.956609 ± 0.023360 | 0.987804 ± 0.001247 | 5 |
| metric-privacy | 0.953750 ± 0.009084 | 0.937388 ± 0.026608 | 0.952194 ± 0.033205 | 0.989045 ± 0.001640 | 5 |

## Fixed-seed non-IID and Appendix-B results

| Partition | Noise | Privacy | Accuracy | Macro F1 | Macro precision | Macro AUC |
|---|---:|---|---:|---:|---:|---:|
| homogeneous | 0.003 | global-dp | 0.960938 | 0.946802 | 0.940484 | 0.990007 |
| homogeneous | 0.003 | metric-privacy | 0.954688 | 0.941679 | 0.970350 | 0.991590 |
| non-iid | 0.01 | global-dp | 0.942187 | 0.697657 | 0.696294 | 0.986109 |
| non-iid | 0.01 | metric-privacy | 0.946875 | 0.931310 | 0.925876 | 0.986887 |
| non-iid | 0.01 | vanilla | 0.959375 | 0.963706 | 0.968883 | 0.991704 |
