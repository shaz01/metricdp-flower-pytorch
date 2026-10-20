# B5 protocol: contribution caps and a prior-neutral local loss against global DP

Package B5 of `research/cia_roadmap_v2/ROADMAP.md` (Section 3, as amended by Section 10, decisions 9 and 10), branch
`feature/cia-v2-mechanisms`. Fixed 2026-10-20, before any B5 run. Hypothesis, item 1's diagnostic and the rationale are
in [STATE.md](STATE.md).

## 1. Runs

| Item | Value |
|---|---|
| Runner | `results/cia_v2/randomized_inclusion/runner.py`, M2's design: 32 candidates, 16 IN per replicate, 32 active of 48; EuroSAT alpha 0.3, layout seed 42, 50 rounds, equal weights, WP2 hyperparameters; no surrogates |
| Design | `R = 32`, `design_seed = 2`: replicate `j` has M2's inclusion and `(client_seed, noise_seed)` |
| Baseline | M2's global-DP runs at the same noise (`results/cia_v2/frontier/results/`, `gdp-r2x`, `gdp-r4x`) |
| Stage A (2x, `r = 0.003092`) | `b5-cap326-r2x` (`--contribution-cap 326`), `b5-cap182-r2x` (`--contribution-cap 182`), `b5-la-r2x` (`--logit-adjustment-tau 1.0`) |
| Stage B (4x, `r = 0.006184`) | the same intervention for each stage-A configuration with a point all-candidate AUC reduction of at least 0.03 and an accuracy loss of at most 1 pp at 2x |
| Output | `results/cia_v2/mechanisms/b5_large_clients/results/`. Committed: run JSON, manifest, measurements, provenance, complete. Kept local: `*.evaluation.json`, `*.predictions.npz` |

```bash
python -m results.cia_v2.randomized_inclusion.runner --config b5-cap326-r2x --privacy global-dp \
    --noise-ratio 0.003092 --contribution-cap 326 --replicates 32 --design-seed 2 --execute \
    --max-parallel-clients 8 --output results/cia_v2/mechanisms/b5_large_clients/results
```

**Interventions** (client-side; the server, the noise and the formal guarantee are those of global DP at the same
noise):

- Contribution cap `m`: each round a client trains on at most `m` of its training records, a fresh uniform draw
  without replacement from its round seed. `m` = 326 (median training-set size) and 182 (25th percentile).
- Logit adjustment: local loss `CE(softmax(log p + tau log pi_k))` with the client's add-one-smoothed label prior
  `pi_k`, `tau = 1`; prediction uses `p`.

## 2. Order

1. Stage A: all 32 replicates of the three configurations.
2. Stage B is decided on stage A's complete results (point AUC reduction at least 0.03 and accuracy loss at most
   1 pp). It is a further noise level, not a continuation of a stage-A configuration.
3. Analysis per configuration once its 32 runs are complete:

   ```bash
   python -m results.cia_v2.mechanisms.b5_large_clients.analyze \
       --pairs b5-cap326-r2x=gdp-r2x:0.003092 b5-cap182-r2x=gdp-r2x:0.003092 b5-la-r2x=gdp-r2x:0.003092 \
       --out results/cia_v2/mechanisms/b5_large_clients/analysis.json
   ```

## 3. Analysis and gate

- Calibrated loss attacker (M2), all candidates and top decile.
- Paired difference `AUC_B5 - AUC_GDP`: pair jackknife at 97.5% (gate), two-way difference bootstrap alongside.
- Accuracy difference with a paired bootstrap over replicates.
- **G-B5** per configuration (owner decision 9): all-candidate AUC at least 0.05 below global DP at the same noise,
  with the gate interval of the difference below 0, and an accuracy loss of at most 1 pp (point estimate). The top
  decile is reported with its interval but not gated.
- Power: about 0.03 detectable difference for paired all-candidate comparisons (A1's estimate from M2's runs).

## 4. Cost

Stage A: 96 runs, about 10.6 GPU-h at M2's 0.11 GPU-h per run (caps shorten local training, so likely less). Stage B:
at most 96 more. The cap is 15 GPU-h; at 1.25 x 15 = 18.75 GPU-h the package stops and escalates, so a stage B with
more than one configuration is escalated before it runs unless stage A's measured cost leaves room.

## 5. Machine

Compute server. Runs use a GPU only when it is idle and unclaimed, or with the owner's approval to share.
