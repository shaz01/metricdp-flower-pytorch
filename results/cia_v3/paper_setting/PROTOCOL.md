# P1 protocol: the paper's 3-client setting under measurement standard v3

Fixed 2026-10-31, before any P1 run. Package description, hypothesis and gate: `STATE.md`. Roadmap:
`research/cia_roadmap_v3/ROADMAP.md` Section 3 (P1).

## 1. Runs

Runner: `results/cia_v3/infra/paper_runner.py` (P0). Output: `results/cia_v3/paper_setting/results/`.

| Config | `--privacy` | `--noise-multiplier` | Pairs | Design seed | Runs |
|---|---|---|---|---|---|
| `vanilla` | vanilla | 0 | 20 | 1 | 40 |
| `gdp-z0.01` | global-dp | 0.01 | 20 | 1 | 40 |
| `mp-z0.01` | metric-privacy | 0.01 | 20 | 1 | 40 |
| extension `gdp-z<z_acc>` | global-dp | `z_acc` | 20 | 1 | 40 |
| extension `mp-z<z_acc>` | metric-privacy | `z_acc` | 20 | 1 | 40 |

All other settings are the runner's defaults (the paper's: 20 rounds, FedAvg weighted by examples, clipping 5, shadow
10% with 20% pixel noise). Each run is checked by the runner: every round aggregated every active client
(`--require-all-clients`). A run that fails is rerun (same seeds) up to three times.

## 2. Extension accuracy pilot (fixed now; accuracy only, never analyzed for leakage)

- Global DP, `design_seed = 3`, pairs 0 and 1 (both worlds), at `z` in {0.03, 0.06, 0.12}: 12 runs, output
  `results/cia_v3/paper_setting/pilot/`.
- Accuracy = the aggregated client-test accuracy (`eval_acc`) averaged over rounds 16-20, in the IN world (Table 13's
  definition), averaged over the two IN runs.
- `z_acc` = linear interpolation in `log z` between the two grid points that bracket 0.637, rounded to two significant
  digits. If 0.637 is not bracketed, the grid is extended by halving below 0.03 or doubling above 0.12, at most two
  more points; if still not bracketed, the extension is dropped and FINDINGS says so.
- The value is recorded in `STATE.md` before any extension run.

## 3. Analysis (`analyze.py`)

Per configuration, the 40 worlds are ordered pair 0 IN, pair 0 OUT, pair 1 IN, ... and form arrays with one candidate
(the target): losses `(1, 40, 20)` at rounds 1-20, membership `(1, 40)` (IN = True).

1. **Paper's statistic per pair:** `roc_auc` of the 20 IN and 20 OUT scores `-noisy_shadow_loss` at rounds 1-20.
   Reported per pair, with the median and the 5th and 95th percentiles (`numpy.percentile`, linear) over pairs.
2. **Calibrated attacker:** `calibrated_scores(losses, membership)` (projected, as in v2), then `roc_auc` over the 40
   trajectories, with `pair_jackknife(..., level=0.95)` (the units are the IN/OUT pairs). The roadmap asks for a 95%
   interval here; v2's 97.5% gate level was calibrated for 32 candidates at `R = 32` and does not apply to one target.
3. Also reported: the calibrated attacker on clean-shadow losses; the final-round threshold attacker
   (`threshold_scores`, noisy shadow) with its pair jackknife; TPR at 5% FPR; mean and range over pairs of the
   last-5-round client-test accuracy (IN, OUT) and of the lowest client's accuracy at round 20.
4. **Decision (G-P1)**, for the main configurations:
   - **fails in its own setting** if the calibrated attacker's 95% lower bound is above 0.5 for both `gdp-z0.01` and
     `mp-z0.01`;
   - **single draw from a wide distribution** if, for at least one of them, the paper's statistic has its 5th
     percentile below 0.5 and its 95th percentile above 0.5;
   - both statements are made if both hold; **consistent with the paper** if neither holds.
   The extension is reported under the same rule as a separate line.
5. **Reproduction check before any new number:** `analyze.py --reproduce` loads M2's committed vanilla runs
   (`results/cia_v2/frontier/results`, config `vanilla`), applies the same `calibrated_scores` and `roc_auc`, and must
   print 0.723 (M2's published all-candidate AUC) to 3 decimals. The P1 analysis refuses to run until it does.

## 4. Cost and machine

About 0.064 run-hours per world (P0 cost pair, GPU shared with another user's job): main 7.7, pilot 0.8, extension 5.1
run-hours, about 14 in all (cap 40 GPU-h). One worker on the compute server, 3 parallel clients, on a GPU chosen under
ROADMAP decision 4.
