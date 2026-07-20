# Controlled comparison: original strategy vs. Flower 1.32 port

## Objective

Test whether the original metric-aware DP strategy and its message-based port produce numerically equivalent results for the same valid client updates and pseudorandom seeds.

## Implementations

- **Original:** the unmodified snapshot at `experiments/original_metricdp_fixed_clipping.py`, copied from repository commit `90cda0361d94e14f74933d7d5bf2ce2cda98ee31` (Flower 1.16.0 base).
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
uv run python experiments/compare.py
```

`compare.py` executes the vendored, byte-identical snapshot of the original source directly in the isolated legacy environment; it does not use a reimplementation of the original algorithm. Its SHA-256 fingerprint below matches the source in the published fork at the cited commit.

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

For this controlled, valid one-round aggregation, the port is numerically equivalent to the original implementation within floating-point precision. This experiment does not establish equivalence for complete model training, unequal client weights, malformed replies, client failures, fewer than two clients, or zero/non-finite model distance. It also does not establish a formal differential-privacy guarantee.

## Source fingerprints

SHA-256 at the time of the experiment:

```text
a40b1357c0fae0ff0764d73a82272ecca2b3302d6a5e0fa8afcbab03d7b4bd69  experiments/original_metricdp_fixed_clipping.py
8b33e0c44ed874edf9f69ba94c803f0b0fbb9eb0eeff9528ff937a05cd7f4e4e  port metricdp_strategy.py
f26daaf72b03ff8c6447bb11eff9a949a328a51c94a9bae9965793ef64c57a60  compare.py
6fa5bdda21ce81aa78fdee34f3daac2b2f94e0b05e86021fef0efe60c7e5fb04  legacy_runner.py
9bd1cb0871f698866edff7a504077cc54ece0a91f4b2bd9e32d82f4843ebf825  modern_runner.py
```
