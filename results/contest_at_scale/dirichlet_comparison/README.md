# Dirichlet comparison (Aug 20, Olcay)

## Question

Our old "non-IID" split wasn't label-skewed at all. It gave clients different amounts of data
(up to 4×) but the same class mix: distance from IID 0.01–0.06, the same as α ≈ 100. So every
"homogeneous vs non-IID" comparison in `contest_at_scale` compared two near-identical splits.

So: with real label skew, do global-DP and metric-privacy still trade accuracy against the client
inference attack the same way?

## Setup

CIFAR-10, 8 clients, Dirichlet α = 0.1, 1.5 and 3. FedAvg, 20 rounds, seed 42. Client 0 is the
target (IN/OUT removal). Noise ratios 0.0025, 0.004 and 0.00625. Script: `cifar10_dirichlet.py`.

## Result

Mild skew gave the same results as the old splitting method. 
Strong skew crashed accuracy and made defenses useless. 
However this doesn't mean our old non-IID results are innocent, see Caveats below.

**The old "non-IID" results should not be considered non-IID anymore.**

## Caveats

**The alphas were badly spread.** 1.5 and 3 are both close to IID, so they gave almost the same
results (vanilla accuracy 69.4% vs 70.9%). 0.1 is the only skewed one, and it's extreme: vanilla
only reaches 46.4%. The useful middle, around 0.3–0.5, was never run.

<details>
<summary>Why: alpha works on a log scale</summary>

Almost all of the change happens below 1. Above 1, every client already holds every class.

| α | A client's biggest class | Classes per client | Distance from IID |
|---:|---:|---:|---:|
| 0.1 | 54% | 4 | 0.68 |
| 0.3 | 37% | 6 | 0.51 |
| 0.5 | 33% | 7 | 0.43 |
| 1 | 26% | 8 | 0.33 |
| 1.5 | 23% | 9 | 0.27 |
| 3 | 19% | 10 | 0.19 |
| 100 | 12% | 10 | 0.04 |

Distance from IID: 0 = same class mix as the whole dataset, 0.9 = a single class. Computed with
`dirichlet_label_partitions`, 8 clients, median over seeds 0–19.

</details>
