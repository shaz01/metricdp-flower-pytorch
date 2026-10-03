# Important ideas

## 1. Tests ran without Dirichlet are misleading 

## 2. Gap is related to how hard the dataset is

On a hard dataset, local training takes big steps and clients disagree more, 
so their updates land far apart and metric-privacy adds less noise than
global-DP. On an easier dataset the updates stay close, and it adds more.

At 48 clients, CIFAR-10's updates are 2× larger than EuroSAT's (6.6 vs 3.3 before clipping) and 3×
further apart (1.28 vs 0.41). Metric-privacy adds 0.6× global-DP's noise on CIFAR-10, but 1.9× on
Alzheimer and 1.8–2.6× on EuroSAT, where it loses accuracy first. 

Could we say metric-privacy leads to higher accuracy on harder datasets and lower accuracy on easier datasets?

## 3. The distances settle within the first 5–10 rounds and then stay flat

Numbers are medians of the per-round `dp-update-norms-before-clipping`,
`metric-dp-pairwise-distances` and `dp-expected-noise-l2-norm` in the run JSONs of
`../contest_at_scale/cifar10/results/3_fixed_ratio/n48-*`,
`../contest_at_scale/auc_frontier/results/alzheimer` and `eurosat_frontier/results/frontier`.
No committed script computes them.



