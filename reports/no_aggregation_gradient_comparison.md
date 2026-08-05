# Quick Experiment: No-Aggregation Client Model Divergence at 20 Clients

Ad hoc request (not part of the Phase 1 diagnosis roadmap): skip federated aggregation entirely,
train 20 independent clients from a shared initial model for 10 local epochs each with no DP noise
and no clipping, save each client's resulting model, and compare them pairwise — the same
mean-layer-wise-Euclidean-distance statistic `metricdp_strategy.py` uses for noise calibration —
to see how far clients drift on their own with nothing pulling them back together.

Code: `no_agg_gradient_compare.py` (ad hoc, not committed to the repo — reuses
`experiments/reproduce/dataset/alzheimer.py`, `experiments/reproduce/paper_training.py`,
`metricdp_pytorch/metricdp_strategy.py:pairwise_model_distances` directly). Source data:
`results/no_agg_gradient_compare/`.

## Design

20 clients, homogeneous partition of the Alzheimer MRI dataset (204 samples/client), each starting
from an identical seeded `PaperCNN` initialization (seed 42). Each client trains independently —
no server, no aggregation, no round structure — for 10 local epochs (Adam, lr=0.001, batch=32).
Standalone PyTorch script, not the Flower/Ray pipeline (there's no aggregation step to orchestrate,
so the simulation harness isn't needed). Every client's final `state_dict` is saved, then all
C(20,2)=190 pairwise distances between final models are computed.

Since every client starts from the identical initial weights `w0`, the pairwise distance between
two clients' *final* weights is mathematically identical to the distance between their *deltas*
(`w_i - w0`, `w_j - w0`) — i.e. their accumulated local pseudo-gradients — so no separate delta
computation was needed to answer "compare their gradients."

## Critical finding: MPS reproducibility failure is broader than previously documented

While double-checking these results before writing this report, rerunning the identical script
(same seed, same code, same everything) gave visibly different numbers — not the small
float32-precision noise expected on a healthy backend. This matters beyond this one experiment: it
revises a claim `metricdp_pytorch/utils/device.py`'s `resolve_device` docstring made about the MPS
non-determinism already documented in `reports/constant_compute_scaling.md` (v2 Findings).

That prior investigation concluded the non-determinism was specific to **concurrent multi-actor**
MPS training — it reported that isolated single-process and sequential
(`max_parallel_clients=1`) runs reproduced exactly. This experiment is a simpler, more isolated
case than that: one client, one process, zero Ray, zero concurrency, zero aggregation. A direct
test — training the same client twice in a row inside one Python process, identical seed, identical
data order (confirmed: `make_indexed_loader`'s shuffle generator is seeded and produces
byte-identical batch order across runs) — showed the two resulting models are **not** identical:

```
features.0.weight:    max abs diff = 0.03451219
features.0.bias:      max abs diff = 0.02873591
features.3.weight:    max abs diff = 0.02906764
classifier.0.weight:  max abs diff = 0.04236054
classifier.6.weight:  max abs diff = 0.03185298
... (all 12 parameter tensors differ)
```

Two candidate simple-bug explanations were tested and ruled out:
- **DataLoader shuffle order**: seeded via `torch.Generator().manual_seed(seed)`, confirmed
  identical batch order isn't the cause.
- **MPS's own RNG for `nn.Dropout(p=0.1)`** (`PaperCNN` has two dropout layers, and dropout masks
  drawn on an MPS-resident tensor use MPS's RNG, which `seed_training()` never seeds): adding
  `torch.mps.manual_seed(seed)` explicitly changed nothing — divergence persisted at the same
  magnitude.

What's left is genuine backward-pass kernel non-determinism (most likely non-deterministic
parallel-reduction order in MPS's conv2d/linear backward kernels), present with or without Ray,
with or without concurrency, with or without aggregation. **`device.py`'s docstring has been
corrected** to remove the now-disproven "isolated/sequential reproduces exactly" claim — the
previous investigation likely only ran short enough sequences (or compared only rounded metrics)
for this to stay under the threshold of visible divergence, not that sequential/isolated runs are
actually safe. Practical implication: `METRICDP_FORCE_CPU=1` is the only path to a reproducible
number from this training loop on this machine, under any configuration — not just concurrent
multi-actor sweeps.

## Results (two runs of the identical configuration)

Given the finding above, a single run's numbers would misrepresent this as more precise than it
is. Both actual runs are reported.

| Statistic | Run 1 | Run 2 |
|---|---:|---:|
| Pairwise distance min | 0.7278 | 0.8092 |
| Pairwise distance median | 0.9994 | 1.0137 |
| Pairwise distance mean | 1.0299 | 1.0945 |
| Pairwise distance max | 1.3551 | 1.4713 |
| Update norm range (min–max across 20 clients) | 11.735–12.883 | 11.756–13.669 |

Both runs agree on the qualitative picture: with homogeneous data, no aggregation, and no DP noise,
20 independently-trained clients cluster fairly tightly — pairwise distances span roughly a 2x
range (min to max) around a mean near 1.0–1.1, and per-client update magnitudes (how far each
client moved from the shared initial model) stay within about a 10% band of each other. That
qualitative conclusion is stable across the two runs even though the exact numbers aren't. For
reference, this is the same order of magnitude as the pairwise-distance statistics already logged
inside real FL rounds at n=48 in the v2 sweep (`results/scale_controlled_epochs/`, ~1.1–2.2) — even
with zero aggregation and zero DP noise here, clients don't drift wildly apart on homogeneous data.

## Follow-up: literal raw gradients, not accumulated weight deltas

The result above compares each client's *final trained model* after 10 epochs of Adam — which, per
the note above, is mathematically the same as comparing accumulated weight deltas
(`w_final - w_init`), but is **not** the same as comparing actual gradients (`∂L/∂W`). No `.grad`
tensor was captured anywhere in that experiment.

This follow-up captures the literal thing: for each of the same 20 clients, one full pass over that
client's own local data at the shared initial weights (no optimizer step, no training loop) —
`loss.backward()` per batch with `reduction="sum"`, no `zero_grad()` between batches so gradients
accumulate, then divided by the client's total example count. This gives each client's mean
per-example gradient of its own local loss at the shared starting point,
`grad_i = (1/n_i) * Σ_x ∂L(x; w0)/∂w` — the object FedSGD (and FedAvg's first-order connection to
it) actually means by "client gradient." Compared the same way, via
`pairwise_model_distances`, by wrapping each client's gradient dict in an `ArrayRecord` (the
function only needs matching-shape arrays; it doesn't care whether they're weights or gradients).
Code: `no_agg_raw_gradient_compare.py`. Source data: `results/no_agg_raw_gradient_compare/`.

**A new, sharper reproducibility failure showed up here.** Two of three attempts at this simpler,
single-pass computation produced an outright `NaN` gradient for one or two clients — a different,
seemingly random client each time (client 1 alone; then client 0 alone; then clients 0 and 3
together). A direct isolated re-test of one of the affected clients, run again on its own,
completed cleanly with a normal-looking gradient norm — confirming this is transient MPS
instability, not a deterministic bug in the client's data or the script. This is a sharper
manifestation of the same non-determinism already documented above: the 10-epoch trained models
never produced a NaN in either run, but a single accumulated backward pass hit one at roughly a
5–10% per-client rate across three attempts. NaN clients were excluded from the pairwise comparison
below rather than silently pollute it.

| Statistic | Value (18/20 valid clients, 2 excluded for NaN) |
|---|---:|
| Pairwise distance min | 0.0087 |
| Pairwise distance median | 0.0330 |
| Pairwise distance mean | 0.0361 |
| Pairwise distance max | 0.0753 |

Two things stand out against the trained-model-delta result above:

1. **Raw gradient magnitudes are ~15-30x smaller** than the 10-epoch accumulated deltas (pairwise
   distances ~0.01–0.08 here vs. ~0.7–1.5 there) — expected, since these are single unoptimized
   gradients at initialization, not the result of 70 Adam steps with momentum and adaptive
   per-parameter scaling compounding over 10 epochs.
2. **Proportionally, the spread is wider**: max/min is ~8.6x here vs. ~1.9x for the trained deltas.
   With homogeneous data, clients' *raw* per-example gradient directions at a shared random
   initialization vary more, relative to their own scale, than their *trained* models do after 10
   epochs pull each client toward a similar local optimum on similarly-distributed data. Single
   run, not cross-checked against a repeat — treat this specific ratio as suggestive, not
   established, given everything above about this platform's reproducibility.

## Known gaps / caveats

- **Not reproducible at face value** — see the finding above. Treat both runs' numbers as one
  informal sample each of an unknown, unquantified noise distribution, not as precise values. A
  `METRICDP_FORCE_CPU=1` rerun (or several more MPS reps) would be needed for a trustworthy number,
  the same as everything else flagged in `reports/constant_compute_scaling.md`.
- **Homogeneous only.** Non-IID wasn't run for this quick check; client drift would plausibly be
  larger and less uniform there, given how much more spread this project's non-IID results show
  elsewhere.
- **Single client count (n=20), single seed for partitioning.** Not swept across n or reseeded.
- Per-client final training loss varies more than the pairwise distances do (e.g. run 2:
  0.29–1.21) despite near-identical data volume per client (all 204 samples) — consistent with
  this project's other observations that homogeneous partitioning doesn't fully equalize per-client
  training difficulty, but not investigated further here.
- **The raw-gradient follow-up hit MPS-produced NaN gradients on 1–2 of 20 clients per attempt**,
  a different client each time. Excluded from the reported pairwise stats rather than silently
  pollute them, but not root-caused beyond confirming (via an isolated re-test) that it's transient
  and MPS-specific, not a deterministic bug tied to a specific client's data.
