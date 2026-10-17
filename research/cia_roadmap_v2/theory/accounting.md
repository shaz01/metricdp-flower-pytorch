# Client-level accountant for DP-FedAvg with client sampling

Agent C, 2026-10-17. Code: [`metricdp_pytorch/accounting.py`](../../../metricdp_pytorch/accounting.py), tests
[`tests/test_accounting.py`](../../../tests/test_accounting.py), tables from
[`accounting_checks.py`](accounting_checks.py) in [`tables/`](tables/). Agent B uses the accountant for B1
(client subsampling) and C1. B4 uses it at full participation.

## 1. Mechanism and adjacency

One round of DP-FedAvg works as follows:

- Each of the `N` slots holds a client, and each client joins the round independently with probability `q` (Poisson
  sampling; `q = 1` means full participation).
- Every participant's update is clipped to norm `C`.
- The server releases `(sum of clipped updates + N(0, (z C)^2 I)) / D`. The denominator `D` is public and fixed: `N`
  at full participation, `q N` under sampling.
- Equivalently, the noise std on the average is `z C / D`. It depends on public quantities only, as Section 1
  decision 4 requires.

Two clients' datasets are neighbours when one client is added or removed (add/remove-one-client adjacency, client
level). Under this adjacency one round is the Poisson-subsampled Gaussian mechanism with noise multiplier `z` and
sensitivity 1 after scaling by `C`, and `T` rounds compose it adaptively. The accountant measures privacy against
someone who observes the released models.

**Threat-model condition for amplification (B1).** The amplification by sampling holds only if the observer does not
learn which clients were sampled in each round. In this project the attacker is a curious participant who sees the
releases and knows only its own selection, so the condition can be met. Two parties fall outside it:

- The server that performs the sampling, or anyone who sees who uploaded.
- Any channel that reveals the round's participant count. Here the denominator is fixed at `q N`, so the count
  does not show in the noise or in the scaling. It still shows in the norm of the released update, the same way as
  the randomized-inclusion norm channel (ROADMAP Section 8).

Against those parties, use the full-participation value `sqrt(T) / z`.

## 2. What the module computes

| Function | Quantity | Exactness |
|---|---|---|
| `gdp_mu(z, T)` | `sqrt(T) / z` | Exact GDP at `q = 1` (Dong, Roth and Su 2022, Corollary 3.3, composition) |
| `gaussian_delta(eps, mu)`, `gdp_epsilon(mu, delta)` | Privacy profile of `mu`-GDP and its inverse | Exact: Dong, Roth and Su 2022, Corollary 2.13; equal to the analytic Gaussian mechanism of Balle and Wang 2018, Theorem 8 |
| `epsilon(m, delta, method=...)`, `delta(m, eps)` | `(eps, delta)` of `T` rounds | `"pld"`: privacy loss distributions of `dp-accounting` 0.6.0, pessimistic discretization `1e-4`, maximum of the add and remove directions, so it never understates; at `q = 1` the closed form. `"rdp"`: the library's RDP accountant. `"rdp-classical"`: RDP with Mironov 2017, Proposition 3 |
| `dominating_mu(m, delta_floor=1e-10)` | Smallest `mu` with `delta(eps) <= max(delta_mu(eps), delta_floor)` for all `eps >= 0` | Rigorous upper bound `mu` and a pointwise lower bound `mu_lower`, refined until they agree to 0.1% |
| `mu_clt(z, q, T)` | `q sqrt(T (exp(1/z^2) - 1))` | Central-limit approximation (Bu, Dong, Long and Su 2020, Eq. (5) and Theorem 5); **not a bound** |
| `noise_multiplier_for_mu`, `noise_multiplier_for_epsilon` | Smallest `z` meeting a target | Bisection to relative tolerance `rtol` |
| `report(m)` | All of the above, for result JSONs | |

**The amplified formal `mu` (G-B1).** The subsampled Gaussian is not exactly GDP. Its privacy loss has a heavier tail
than any Gaussian whose `mu` is close to its bulk, so the `mu` needed for every `eps` is driven by values of `delta`
far below any meaningful level. `dominating_mu` therefore states a floor: the mechanism is `mu`-GDP up to an additive
`delta_floor`. Equivalently, it is `(eps, max(delta_mu(eps), delta_floor))`-DP for every `eps`.

Method:

- The PLD's profile is checked on a uniform `eps` grid that ends where the profile first falls below the floor.
- The condition is `delta(e_k) <= delta_mu(e_{k+1})`. Both profiles decrease in `eps`, so this implies the claim
  between grid points, not only at them.
- The grid is refined until the bound is within 0.1% of the pointwise value.

Default floor: `1e-10`, far below `1/N` for `N` = 48 or 480 slots. **Report `mu` with its floor.** For G-B1, compare
the subsampled mechanism's `mu_dominating` at the default floor with full participation's exact `sqrt(T) / z`. Also
report `epsilon` at `delta = 1e-5` for both.

## 3. Checks

**Published values** ([tables/accounting_published.csv](tables/accounting_published.csv); `delta = 1e-5`). Each value
was checked against its source on 2026-10-17:

- Abadi et al. 2016, Section 5.1, moments accountant.
- TensorFlow Privacy `tutorials/README.md`, MNIST table: RDP accountant, `N = 60,000`, batch 256,
  `T = ceil(epochs N / batch)`.

| Source | `q` | `z` | `T` | Published `eps` | RDP, classical conversion | RDP, improved conversion | PLD |
|---|---|---|---|---|---|---|---|
| Abadi et al. 2016, E = 100 | 0.01 | 4.0 | 10,000 | 1.26 | 1.2586 | 1.0355 | 0.9470 |
| Abadi et al. 2016, E = 400 | 0.01 | 4.0 | 40,000 | 2.55 | 2.5737 | 2.2097 | 2.0334 |
| TF Privacy, noise 1.1, 60 epochs | 0.004267 | 1.1 | 14,063 | 3.01 | 3.0084 | 2.5967 | 2.3818 |
| TF Privacy, noise 1.3, 15 epochs | 0.004267 | 1.3 | 3,516 | 1.19 | 1.1923 | 0.9546 | 0.8646 |
| TF Privacy, noise 0.7, 45 epochs | 0.004267 | 0.7 | 10,547 | 7.10 | 7.1016 | 6.3197 | 5.6397 |

- The classical conversion reproduces four of the five published values to their two decimals.
- Abadi's E = 400 value comes out 0.024 higher. The moments accountant evaluated its bound at integer orders up to
  32 with its own numerics, and both numbers bound the same mechanism.
- On every row, PLD < improved RDP < classical RDP. That is the expected order of tightness, and it is the order the
  module uses.

**Closed forms** (in the tests):

- `gaussian_delta` equals the hockey-stick divergence between `N(mu, 1)` and `N(0, 1)` by quadrature (relative
  error 1e-6) for `mu` in {0.3, 1, 2.5, 8} and `eps` in {0, 0.5, 2, 6}.
- At `q = 1` the library's PLD `epsilon` lies in `[exact, exact + 0.01]`.
- `dominating_mu` is exact there.

**Dominance, monotonicity and the CLT** ([tables/accounting_floor.csv](tables/accounting_floor.csv)):

| `z` | `q` | `T` | `sqrt(T)/z` | `mu_CLT` | `mu_dom`, floor 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 | PLD `eps` at 1e-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.01 | 10,000 | 100.0 | 1.311 | 1.337 | 1.352 | 1.365 | 1.376 | 1.388 | 6.19 |
| 2.0 | 0.25 | 50 | 3.536 | 0.942 | 0.998 | 1.027 | 1.050 | 1.071 | 1.091 | 4.44 |
| 1.0 | 0.25 | 50 | 7.071 | 2.317 | 2.357 | 2.473 | 2.567 | 2.649 | 2.722 | 12.65 |
| 1.5 | 0.25 | 100 | 6.667 | 1.870 | 1.924 | 1.974 | 2.016 | 2.054 | 2.087 | 9.69 |
| 0.8 | 0.5 | 50 | 8.839 | 6.865 | 5.328 | 5.481 | 5.608 | 5.712 | 5.804 | 36.94 |

How to read the table:

- Sampling at `q = 0.25` lowers `mu` by a factor of 2.7-3.3 against full participation at the same `z` and `T`,
  at the default floor.
- Lowering the floor from `1e-4` to `1e-12` raises `mu` by 4-15%.
- The CLT is close in its regime (small `q`, large `T`: 1.311 against 1.376 at the default floor) but is not a bound
  in either direction. At `q = 0.5, z = 0.8` it overstates `mu` (6.87 against 5.71).

The tests also check:

- The returned `mu` dominates the PLD profile at 2,000 random `eps` values.
- 0.99 times the pointwise lower bound fails somewhere.
- `mu` rises with `q` and `T`, and falls with `z`.
- The calibration functions round-trip.

**Calibration for B1 and B4** ([tables/accounting_calibration.csv](tables/accounting_calibration.csv)): the noise
multiplier needed for `mu` = 1, 2, 4 and 8 at `q` in {1, 0.5, 0.25} and `T` in {10, 20, 50, 100}. See Section 4.

## 4. Calibration table

Noise multiplier `z` (noise std `z C` on the sum of clipped updates) whose `mu_dominating` (floor 1e-10) meets the
target, from bisection to 1% (`rtol = 0.01`). The last column is the noise std on the averaged update, `z / (q N)`,
relative to full participation at the same `mu` and `T`, as the range over `mu` = 1 to 8 (it grows with `mu`).

| `q` | `T` | `mu` = 1 | 2 | 4 | 8 | Noise on the average vs full participation (`mu` = 1 to 8) |
|---|---|---|---|---|---|---|
| 1 | 10 | 3.1623 | 1.5811 | 0.7906 | 0.3953 | 1.00-1.00x |
| 1 | 20 | 4.4721 | 2.2361 | 1.1180 | 0.5590 | 1.00-1.00x |
| 1 | 50 | 7.0711 | 3.5355 | 1.7678 | 0.8839 | 1.00-1.00x |
| 1 | 100 | 10.0000 | 5.0000 | 2.5000 | 1.2500 | 1.00-1.00x |
| 0.5 | 10 | 1.9678 | 1.1084 | 0.6243 | 0.3460 | 1.24-1.75x |
| 0.5 | 20 | 2.5519 | 1.3990 | 0.7880 | 0.4414 | 1.14-1.58x |
| 0.5 | 50 | 3.7687 | 1.9892 | 1.0846 | 0.6043 | 1.07-1.37x |
| 0.5 | 100 | 5.1874 | 2.6649 | 1.4142 | 0.7795 | 1.04-1.25x |
| 0.25 | 10 | 1.3397 | 0.8318 | 0.5109 | 0.3022 | 1.69-3.06x |
| 0.25 | 20 | 1.5931 | 0.9733 | 0.5978 | 0.3633 | 1.42-2.60x |
| 0.25 | 50 | 2.1228 | 1.2352 | 0.7465 | 0.4585 | 1.20-2.07x |
| 0.25 | 100 | 2.7678 | 1.5339 | 0.9022 | 0.5512 | 1.11-1.76x |

Reading for B1: at a matched formal `mu`, Poisson sampling with denominator `q N` puts 1.04-3.06 times *more*
noise on each released average than full participation (most at few rounds, small `q` and weak guarantees).
Amplification does not buy less noise per release here; any CIA gain from B1 has to come from the target being
absent from most rounds and from the different per-round signal, which is what B1 measures.

## 5. Use

```bash
# Main checkout
uv run --with dp-accounting==0.6.0 python -c "from metricdp_pytorch import accounting as a; print(a.report(a.SubsampledGaussian(1.0, 0.25, 50)))"
# Worktree sharing the main venv (AGENTS.md worktree rules)
PYTHONPATH="$PWD" UV_PROJECT_ENVIRONMENT=<main>/.venv uv run --no-sync --with dp-accounting==0.6.0 pytest tests/test_accounting.py
```

- `dp-accounting` is not a project dependency. The module imports it lazily, and the closed forms work without it.
  The library-backed tests skip themselves when it is missing, so a plain `uv run pytest` stays green. Adding it to
  `pyproject.toml` is the owner's call.
- The RDP accountant logs absl warnings when its series for a fractional order does not converge. It then excludes
  that order, and the result is still a valid bound.
- Runtime: `dominating_mu` takes 1-10 s at moderate noise and longer at small `z` (the PLD support grows like `1/z^2`);
  `noise_multiplier_for_mu` with sampling takes 0.5-3 min at `rtol = 1e-2`. The search brackets from `z = 1` and
  refuses targets that would need `z < 0.05`.
