# Stacked-head experiment (Flower)

A Flower implementation of the **stacked, validation-gated head step** found in the client-specific-noise research
(`research/proposals/2026-10-08_*_findings.md`, results in `results/client_specific_noise/`). It answers one question on
the repo's real ServerApp/ClientApp message path: when the server has very little public labelled data, can eight
clients' clipped, noised head gradients improve on the best public-only model, with the attack strength calibrated and
checked?

## What happens in one release

1. **Public base (server, once):** a small CNN is trained on the public examples (32 or 128 images); six candidates
   (epochs x learning rate) are scored on a public validation set and the best becomes the base. Its body is frozen.
   The server also stores the public class gradients and the top-51 public-Hessian directions of the 4-class head.
2. **Round (clients):** each client embeds its local images with the broadcast base, forms one class-balanced head
   gradient (`target_center`), projects it onto the public directions, clips it to `cap`, adds its own Gaussian noise
   share, and replies with that single 51-number vector.
3. **Aggregate (server):** sum the eight vectors, decode, and try step multipliers (0, 1/30, ..., 30) x the frozen step
   on the public validation set; keep the best. This only reads the released sum and public data (post-processing).
4. **Evaluate (server):** score the released model on held-out images.

The frozen construction (`mode`, projection dimension, `cap`, step `eta`) is read from
`results/client_specific_noise/stacked_constructor_freeze.json`; noise is calibrated so the optimal known-alternative
attack on a client's contribution has AUC equal to `--risk`.

`--rounds N` (replicate mode) re-sends the ORIGINAL base every round, so each round is an independent release from the
same base. It estimates the release distribution inside one simulation; it is **not** a multi-round protocol and privacy
is not composed across rounds. `--single-release` is one real one-shot round.

## Layout

| File | Role |
| --- | --- |
| `data.py` | deterministic per-class roles (public sets, validation, two private cohorts, held-out evaluation), label-stress client partitions, image loading |
| `cnn.py` | CNN (`body.*`, `head.*`), full-batch CPU training, feature embedding |
| `bundle.py` | trains/selects the public base and stores every public artifact the server needs (the server never reads the dataset) |
| `client.py` | `ClientApp.train` handler (noise sources `independent` / `probe`, dummy clients, optional diagnostics) |
| `server.py` | `ServerApp`: bundle, strategy (`metricdp_pytorch/stacked_head_strategy.py`), held-out evaluation, result JSON |
| `local.py` | in-process `Grid` running the same server/client code without Ray |
| `runner.py` | `prepare`, `run`, `matrix`, `attack` commands |
| `attack.py` | known-alternative IN/OUT attack on the real message log |
| `analysis.py` | summarises finished cells and applies the research phase's gate |
| `validate_equivalence.py` | compares a probe-noise run with the stored research-probe cell |

## Running it

```sh
uv sync
# One cell: 512 independent releases through Flower's Ray simulation (about 1.5 min on a laptop CPU)
uv run python -m experiments.stacked_head.runner run --task kmnist_classes0to3 --budget 32 --public-set 0 --cohort A --risk 0.65
# Same code path without Ray
uv run python -m experiments.stacked_head.runner run --backend inprocess --rounds 64
# A resumable grid: smoke (1 cell), gate (24 cells, KMNIST x 3 public sets x 2 cohorts x q.65/.80; roughly 1 hour
# on a laptop CPU), budgets (48 cells, budgets 32 and 128; roughly 2 hours)
uv run python -m experiments.stacked_head.runner matrix --preset gate
uv run python -m experiments.stacked_head.analysis results/stacked_head
# Attack check on the real message path (noise-free contribution run + IN + OUT worlds)
uv run python -m experiments.stacked_head.runner attack --target 3 --risk 0.65 --rounds 2048
# Replay the research probe's noise and compare with results/client_specific_noise/cnn_head.*
uv run python -m experiments.stacked_head.runner run --noise-source probe --rounds 512
uv run python -m experiments.stacked_head.validate_equivalence results/stacked_head/kmnist_classes0to3_b32_s0_A_q65_probe.json
```

Results go to `results/stacked_head/<task>_b<budget>_s<set>_<cohort>_q<risk>_<tag>.json` (config, construction,
bundle metadata, control, every release, summary, provenance); `--log-messages` adds `<name>.messages.npz`. Public
bundles are cached in `.stacked_head_cache/bundles/` (gitignored). `matrix` skips cells whose result file exists.

### On a server

* **No GPU needed.** The CNN is tiny, training and features are float32 on CPU by design (`torch` threads pinned to 1 in
  clients) so base models are reproducible across hosts; the construction is float64 NumPy.
* **Data:** tasks download from Hugging Face on first use (`tanganke/kmnist`, `ylecun/mnist`,
  `zalando-datasets/fashion_mnist`). On a host without internet, run `prepare` (or any `run`) once where the data is
  available and copy `~/.cache/huggingface/datasets` (clients also read the dataset to find their images).
* **Parallelism:** each simulation uses 8 single-CPU clients (`--client-cpus`, `--max-parallel-clients`). Run several
  `matrix` invocations in parallel on different presets/outputs if cores allow; there is no cross-process locking.
* **Reproducibility:** `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED` are set like the other experiments; results are
  repeatable run to run on one host, and probe-replay runs match the research probe to about 1e-8.

## Privacy contract and caveats

* The client reply is one vector, `clip(projected gradient) + own noise share`; shares of the 7 peers sum to the
  calibrated std, so the server sees the 8/7-inflated total (documented convention). The server in this simulation is
  trusted to only sum; it does not add noise itself. A single client's message carries only a 1/sqrt(7) share of the
  noise, so a server that can read individual messages learns more than the calibrated aggregate view; a real
  deployment needs secure aggregation (not implemented here). A round with a missing client reply is skipped, not
  re-scaled.
* `--diagnostics` makes clients also return their noise-free contribution and `--log-messages` stores individual
  messages. Both exist for audits and **leak private-dependent values**; never enable them in a real deployment.
* The validation set is extra public labelled data (512 images; 128 also worked in the research phase). The public
  base selection already used it. "Limited public" means limited public TRAINING data with a modest public validation set.
* Evidence so far: Gaussian noise only; a 4-class linear head on a small CNN; KMNIST. The attack guarantee is the
  conditional known-alternative contract (peer view), not a general CIA defense. Offline tuning, public-set choice and
  result release are unaccounted. See the research findings for the limits.
