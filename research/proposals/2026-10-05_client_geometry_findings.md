# Real-data client-geometry audit and construction decision

2026-10-05. Owner resumed with “lets move to the next step”. This is a bounded research diagnostic, not a declaration that the overall experiment is finished.

Persistent client-specific gradient covariance exists in the explicit label-skew stress regime. However, full-information personalization improves held-out cross-entropy by at most **0.000363**, below the **0.001** feasibility threshold set in the calculator before its first run. This supports neither a private estimator implementation yet nor a universal impossibility claim: the utility intervention covers only two fixed bias coordinates.

## Question and material

The preceding [rotated/residual comparison](2026-10-04_rotated_residual_findings.md) showed little adaptive benefit. We now ask two separate questions: do real-data clients have distinguishable, persistent geometry, and does knowing their raw updates create enough global utility headroom to justify the privacy cost of constructing personalized laws?

The local inventory found cached Fashion-MNIST and Alzheimer data, but no suitable saved multi-round model trajectory. The available repository checkpoint is a one-round failed-reproduction Alzheimer smoke run, unsuitable as evidence of mature-model geometry. This is a local inventory, not a claim about other machines. We therefore generated a small fresh trajectory on Fashion-MNIST. The [official dataset repository](https://github.com/zalandoresearch/fashion-mnist) describes the 28×28 grayscale dataset; this audit uses only repository-supported labels 0–3.

Code: [client_geometry_audit.py](../calculations/client_geometry_audit.py). Data: [JSON measurements](../../results/client_specific_noise/client_geometry_audit.json) and [matrix/index archive](../../results/client_specific_noise/client_geometry_audit_matrices.npz). Independent review: [geometry audit math review](2026-10-05_geometry_audit_math_review.md).

## Reproducible protocol

- Seeds 42, 43 and 44; eight clients. Each seed draws 1,024 training-pool examples per class and a disjoint development set of 256 per class. Each client pool is split by label into disjoint fit/check halves; only fit halves train the head. Evaluation uses the same fixed 256 official test examples per class across seeds.
- Balanced and quantity-skewed partitions use existing repository helpers. Quantity weights are 1 through 8. The separately disclosed **label stress** partition gives each client 410 examples of its primary class and 34 of each other class, before splitting; global class counts remain balanced. This is a controlled stress test, not measured natural prevalence.
- Fixed public features average each image into a 4×4 grid, center its 16 values and append a bias. A four-class softmax has 68 parameters. Starting at zero, 20 sample-count-weighted full-batch FedAvg gradient steps use learning rate 0.5 and one local step. Snapshots are rounds 0, 5 and 20: 27 records in total. This is a reduced model, not the repository CNN or a Flower/CIA run.
- Estimate per-client mean gradient, uncentered second moment, centered covariance and exact local softmax Hessian at each common checkpoint. Training/check geometry is compared without using check examples for training or utility selection.

Run from the root with `uv run python research/calculations/client_geometry_audit.py --self-check`, then `uv run python research/calculations/client_geometry_audit.py`. The latter reads locally cached Arrow files and never downloads them. Use `--cache-dir PATH` on another machine. JSON includes source-file SHA-256 hashes; the archive preserves pool/development/test indices, concatenated client splits, model heads, matrices, means, weights and projected updates. Split counts in JSON delimit concatenated fit indices. No raw images are committed.

## Geometry findings

For top-two orthonormal subspaces, overlap is ||UᵀV||²_F / 2: one means identical subspaces, zero means orthogonal. Table entries average the three seeds at round 20; persistence compares round 5 with round 20. These are descriptive means, not independent-population confidence intervals.

| Partition | Local fit/check covariance overlap | Between-client covariance overlap | Pooled within-client covariance → check overlap | Same-client persistence |
|---|---:|---:|---:|---:|
| Balanced | 0.995848 | 0.996433 | 0.997773 | 0.999127 |
| Quantity skew | 0.994996 | 0.976873 | 0.992596 | 0.998703 |
| Label stress | 0.976449 | 0.633021 | 0.765299 | 0.984232 |

Balanced clients largely share the same dominant geometry. Quantity-skew clients show modest differences; differing finite sample counts also affect estimation accuracy. Label-stress clients have substantially different covariance directions that repeat on disjoint examples and persist across learning. Their local geometry predicts their held-out geometry better than the pooled estimate. “Pooled covariance” here is the weighted mean **within-client** covariance, not the covariance of the full mixture including between-client mean variation.

At round 20 the mean-gradient energy fraction ||μ||² / tr(M) is 0.006956 for balanced, 0.009669 for quantity skew and **0.476139** for label stress. Almost half the stress-regime uncentered gradient energy comes from the mean; calling the uncentered second moment “gradient variance” would be misleading. The centered result above survives that correction.

Hessian shape is less differentiated. Normalized Frobenius cosine measures whole-matrix shape without selecting unstable eigenvectors:

| Partition | Local fit/check Hessian cosine | Between-client Hessian cosine | Pooled Hessian → check cosine |
|---|---:|---:|---:|
| Balanced | 0.998793 | 0.999141 | 0.999256 |
| Quantity skew | 0.998941 | 0.998009 | 0.999342 |
| Label stress | 0.998887 | 0.959202 | 0.981690 |

High cosine does not establish equal eigenvalues, identical utility effects or equal sensitivity. The recorded eigengap diagnostic applies to uncentered moments; centered-covariance eigengap diagnostics remain absent. Round-20 accuracy is approximately 0.667, 0.663 and 0.667 respectively. This modest head establishes a learning trajectory, not CNN representativeness.

## Full-information utility test

Freeze the checkpoint except the public class-0/class-1 bias coordinates. Each client's candidate step is its two-coordinate gradient multiplied by −0.5. Compare a common profile with personalized assignments from the same four-profile bank: a diamond with equal axes and elongated diamonds with axis ratio 1:0.25 at 0°, 90° and 45°. Common radius grid: 0.001, 0.005, 0.02, 0.1, 0.3. Counterfactual noise labels: 4, 8, 16.

Both diagnostics see raw fit updates and the common global development gradient/Hessian. Shared choice optimizes one profile for every client; the oracle enumerates all 4⁸ assignments at each common radius. It includes every shared assignment. Development selection minimizes global second-order loss change, including the mean of clipped updates and weighted aggregate noise covariance. Neither selector, the raw checkpoint, nor the metadata is privately accounted. **These epsilon labels are not privacy budgets for the audit transcript.** This oracle is an optimistic finite-bank diagnostic, not a deployable decentralized client constructor or an upper bound over all mechanisms.

Selected choices are evaluated using direct nonlinear test cross-entropy over 2,048 fresh paired noise draws, plus quadratic and noiseless controls. There are 81 comparisons. None reaches the 0.001 absolute CE improvement gate. The largest gain is in label stress, seed 44, round 20, noise label 16:

| Control | Held-out CE |
|---|---:|
| No intervention | 1.230857280 |
| Noiseless raw projected step | 1.230608364 |
| Shared profile, noiseless clipped step | 1.230324211 |
| Personalized oracle, noiseless clipped step | 1.229947483 |
| Shared profile, noisy mean | 1.230343499 |
| Personalized oracle, noisy mean | 1.229980366 |

Oracle-minus-shared is −0.000363133; the paired noise-only standard error is 0.000002688 and its exploratory normal 95% interval is [−0.000368402, −0.000357864]. This is about 0.0295% of the shared loss. The interval conditions on these examples, checkpoints and selected settings; it excludes population, partition and estimation uncertainty and has no multiple-comparison adjustment.

Crucially, the oracle has a **larger** quadratic noise penalty: 0.000037551 versus 0.000024628. Its tiny improvement comes from changing the clipped aggregate learning signal, not reducing the perturbation penalty. The 0.001 gate is a pragmatic bounded feasibility criterion specified before execution, not a preregistered universal standard or a privacy threshold.

## Meaning for the proposed mechanism

A client having different covariance does not automatically need a different noise law. With a common global utility Hessian, equal privacy constraints and the same admissible profile family, each client's noise-only minimization has the same minimizer; aggregation weights merely multiply its cost. A useful personalized construction must exploit a justified additional difference, such as how joint clipping changes each update's contribution, while accounting for how the construction itself depends on private data.

We have established a concrete estimation target in controlled label skew, but not material utility headroom for constructing a private distribution in this bank. Do not implement a paid private covariance estimator or claim a new law improves CIA based on these results. Natural covariance, worst-case sensitivity, global curvature and attack information remain distinct quantities.

The next proposed step is a design review for a **broader fixed parameter-block oracle**, with equally strong shared controls and aggregate clipping bias measured explicitly. Choose the block/family and its feasibility criterion on development data before any new held-out tuning; fresh evaluation is needed after this repeated exploratory test use. Only if that richer full-information comparison gives material gain should we design a protected estimator/sampler and account for the complete client-contribution observer law. Aggregate-only versus individual-upload observers may change feasibility and must remain explicit. No additional run is launched by this note.

This audit neither proves nor tests CIA mitigation, worst-case client privacy, a novel density, full-model utility or generalization to natural client populations. The original frontier remains the eventual utility/leakage comparator, with its existing evaluation qualifications preserved.
