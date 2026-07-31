# Controlled Comparison: Original Strategy vs. Flower 1.32 Port

Source: `experiments/port_equivalence/` (`README.md`, `compare.py`, `test_equivalence.py`,
`original_metricdp_fixed_clipping.py`). No `results/port_equivalence/` directory exists — this
experiment's output is a pass/fail numerical comparison, reported directly in its own README
rather than as JSON artifacts, and reproduced below.

## Objective

Test whether the original metric-aware DP strategy and its message-based port
(`metricdp_pytorch/metricdp_strategy.py`) produce numerically equivalent results for the same
valid client updates and pseudorandom seeds.

## Implementations compared

- **Original:** unmodified snapshot at `experiments/port_equivalence/original_metricdp_fixed_clipping.py`,
  copied from the published fork's commit `90cda0361d94e14f74933d7d5bf2ce2cda98ee31` (Flower 1.16.0
  base). The paper cites Flower 1.13.0, but the published branch containing the tested source is
  based on 1.16.0.
- **Port:** `metricdp_pytorch/metricdp_strategy.py`, using Flower 1.32.1.

Environments: the original runs in an isolated legacy environment (Python 3.12, Flower 1.16.0,
NumPy 1.26.4, SciPy 1.14.1) that the runner creates automatically; the port runs in the repo's
`uv.lock`-pinned environment (Python 3.13.5, Flower 1.32.1, NumPy 2.5.1).

## Controlled design

One aggregation round with three equally weighted synthetic client models (two arrays each). Both
implementations receive identical `float64` values, a zero-valued global model, clipping norm
`C = 2.5`, base noise multiplier `0.7`, and three sampled clients, with clipping active. For each
seed in `{0, 1, 42, 123456, 20260721}`, NumPy's global RNG is reset immediately before aggregation,
and the run compares: maximum pairwise mean layer distance, calibrated Gaussian standard
deviation, and every element of the noised aggregate. Equivalence is accepted when the maximum
absolute difference is at most `1e-15`.

## Observed results

| Seed | Distance diff | Stdev diff | Max output diff | Within 1e-15 |
|---:|---:|---:|---:|:---:|
| 0 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| 1 | 0.0e+00 | 0.0e+00 | 5.6e-17 | True |
| 42 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| 123456 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |
| 20260721 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True |

All five seeds pass the `1e-15` criterion; the largest discrepancy observed was `5.6e-17` (seed 1).
Distance and noise standard deviation matched exactly across every seed tested.

## Conclusion and scope

For this controlled, valid one-round aggregation, the port is numerically equivalent to the
original implementation within floating-point precision. This does **not** establish equivalence
for complete multi-round training, unequal client weights, malformed replies, client failures,
fewer than two clients, or zero/non-finite model distance — those cases are out of scope for what
was tested.

## Reproducing

```bash
uv run pytest -m reproducibility experiments/port_equivalence/test_equivalence.py
```

runs five parameterized pytest cases, one per seed (excluded from the default `uv run pytest`
because it spins up an isolated legacy environment). The same numerical report can be printed
directly with `uv run python experiments/port_equivalence/compare.py`. `compare.py` executes the
vendored, byte-identical snapshot of the original source in that isolated environment — it does
not reimplement the original algorithm.
