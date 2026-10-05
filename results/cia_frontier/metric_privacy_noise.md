# Important ideas

## 1. Tests ran without Dirichlet are misleading 

## 2. Gap is NOT about how hard the dataset is (corrected by Stage A)

Metric-privacy adds less noise than global-DP when client updates land far apart, and more
when they stay close. In the old runs that looked like a dataset-difficulty effect: at 48
clients, CIFAR-10's updates were 2× larger than EuroSAT's (6.6 vs 3.3) and 3× further apart
(1.28 vs 0.41), so metric-privacy added 0.6× global-DP's noise on CIFAR-10 but 1.8–2.6× on
EuroSAT and 1.9× on Alzheimer.

`dataset_vs_model` Stage A tested it with the confounds removed (same 450 images per client,
2 datasets × 2 models, homogeneous, 10 rounds, seed 42):

- **The old 0.6× came from data per client.** CIFAR-10 clients had ~1,040 images vs EuroSAT's
  450, so 2.3× more local steps per round. With equal data, metric-privacy adds **more** noise
  than global-DP in every cell, CIFAR-10 included (1.3×; 2.3–2.9× in the other cells).
- **Distance depends on the dataset × model pair.** CIFAR-10 clients drift apart only with
  `cifar10_cnn` (0.76); with `eurosat_cnn` they stay as close as EuroSAT's (0.41 vs 0.34–0.42),
  even though they train worse. Difficulty alone doesn't push clients apart.

So "harder dataset → less metric-privacy noise" is wrong. What matters is how far apart the
clients' updates land, which depends on data per client and on the dataset–model pair.
Compare the mechanisms at equal accuracy, never at equal noise ratio.
See `dataset_vs_model/README.md` (Stage A) for the numbers.

## 3. The distances settle within the first 5–10 rounds and then stay flat

Numbers are medians of the per-round `dp-update-norms-before-clipping`,
`metric-dp-pairwise-distances` and `dp-expected-noise-l2-norm` in the run JSONs of
`../contest_at_scale/cifar10/results/3_fixed_ratio/n48-*`,
`../contest_at_scale/auc_frontier/results/alzheimer` and `eurosat_frontier/results/frontier`.
No committed script computes them.



