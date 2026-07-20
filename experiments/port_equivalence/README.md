# Controlled comparison: original strategy vs. Flower 1.32 port

TLDR: For this controlled, valid one-round aggregation, the port is numerically equivalent to the original implementation within floating-point precision.

## Objective

Test whether the original metric-aware DP strategy and its message-based port produce numerically equivalent results for the same valid client updates and pseudorandom seeds.

## Implementations

- **Original:** the unmodified snapshot at `experiments/port_equivalence/original_metricdp_fixed_clipping.py`, copied from repository commit `90cda0361d94e14f74933d7d5bf2ce2cda98ee31` (Flower 1.16.0 base).
- **Port:** `metricdp_pytorch/metricdp_strategy.py`, using Flower 1.32.1.

The paper mentions Flower 1.13.0, but the published branch containing the tested source is based on Flower 1.16.0.

## Controlled design

One aggregation round uses three equally weighted synthetic client models with two arrays each. Both implementations receive identical `float64` values, a zero-valued global model, clipping norm `C = 2.5`, base noise multiplier `0.7`, and three sampled clients. Clipping is active for these updates.

For each seed in `{0, 1, 42, 123456, 20260721}`, NumPy's global RNG is reset immediately before aggregation. The experiment compares:

1. Maximum pairwise mean layer distance
2. Calibrated Gaussian standard deviation
3. Every element of the noised aggregate

Numerical equivalence is accepted when the maximum absolute difference is at most `1e-15`.

## Environments

The runner creates an isolated legacy environment:

- Python 3.12
- Flower 1.16.0
- NumPy 1.26.4
- SciPy 1.14.1

The port environment, locked by `uv.lock`, used:

- Python 3.13.5
- Flower 1.32.1
- NumPy 2.5.1

## Reproduction

The comparison depends only on files in this package; the separate Flower fork is not required.

```bash
cd metricdp-pytorch
uv sync
uv run pytest -m reproducibility experiments/port_equivalence/test_equivalence.py
```

This runs five parameterized pytest cases, one per seed. The same numerical report can be printed with:

```bash
uv run python experiments/port_equivalence/compare.py
```

The `reproducibility` marker is excluded from ordinary `uv run pytest` runs because the experiment creates an isolated legacy environment. `compare.py` executes the vendored, byte-identical snapshot of the original source directly in the isolated legacy environment; it does not use a reimplementation of the original algorithm. Its SHA-256 fingerprint below matches the source in the published fork at the cited commit.

## Observed results

```text
seed=0        distance_diff=0.0e+00 stdv_diff=0.0e+00 output_max_diff=0.0e+00 within_1e-15=True
seed=1        distance_diff=0.0e+00 stdv_diff=0.0e+00 output_max_diff=5.6e-17 within_1e-15=True
seed=42       distance_diff=0.0e+00 stdv_diff=0.0e+00 output_max_diff=0.0e+00 within_1e-15=True
seed=123456   distance_diff=0.0e+00 stdv_diff=0.0e+00 output_max_diff=0.0e+00 within_1e-15=True
seed=20260721 distance_diff=0.0e+00 stdv_diff=0.0e+00 output_max_diff=0.0e+00 within_1e-15=True
```

The largest discrepancy was `5.6e-17`; all outcomes satisfied the `1e-15` criterion. The distance and noise standard deviation were identical for every seed.

## Conclusion and scope

For this controlled, valid one-round aggregation, the port is numerically equivalent to the original implementation within floating-point precision. This experiment does not establish equivalence for complete model training, unequal client weights, malformed replies, client failures, fewer than two clients, or zero/non-finite model distance. 

