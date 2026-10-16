# Consensus-KL follow-up: findings (closed 2026-10-16)

Branch `feature/consensus-kl`. Owner request 2026-10-16, after WP2 stopped D1: look deeper at WP2's one lead,
consensus KL with beta 0.3 (local loss `CE + 0.3 KL(p_global || p_local)` on the client's own batches). Protocol
`PROTOCOL.md` (step 1 and step 3 fixed before their runs), log `STATE.md`. 54 trajectories, 16.51 h on the compute
server. **Owner decision 2026-10-16: step 2 (smaller beta) is not run; the consensus-KL line is closed.**

## Conclusion

The seed-42 leakage reduction that made consensus KL the lead in WP2 (0.148 [0.048, 0.224]) does not replicate at
seeds 43-45. Over four seeds consensus KL 0.3 leaks no less than vanilla training and costs 2.6 pp accuracy. Under
global DP at 1-4x the frontier noise it costs accuracy at every noise level and never lowers leakage significantly.
WP2's verdict (G2 inconclusive, D1 stops) stands.

## Step 1: seeds 42-45, no DP (`analysis_step1.json`)

WP2's federation and statistics unchanged (EuroSAT, alpha 0.3, 48 clients, 50 rounds, equal weights, targets 23, 7,
1, 0, 3; 20 target-seed clusters, 2,000-resample paired cluster bootstrap, seed 1729). New runs: consensus KL 0.3 at
seeds 43-45 (18 trajectories, 5.60 h); everything else reused from WP2.

| Configuration | Clean-view score [95%] | AUC | Top-decile score | Accuracy | Lowest client (IN, mean over seeds) |
|---|---|---|---|---|---|
| `vanilla` | 0.762 [0.659, 0.854] | 0.822 | 0.865 | 0.8276 | 0.628 |
| `consensus-kl-0.3` | 0.795 [0.712, 0.869] | 0.871 | 0.863 | 0.8018 | 0.595 |
| `weight-decay-5e-4` | 0.763 [0.666, 0.851] | 0.824 | 0.833 | 0.8159 | 0.612 |
| `local-epochs-1` | 0.848 [0.780, 0.911] | 0.917 | 0.928 | 0.6846 | 0.328 |
| `fixed-steps` | 0.763 [0.704, 0.822] | 0.830 | 0.795 | 0.8247 | 0.643 |

- Reduction vs vanilla: -0.033 [-0.112, +0.048] (score), -0.049 [-0.142, +0.038] (AUC); accuracy -2.58 pp, which fails
  WP2's 1 pp condition. Against weight decay -0.032 [-0.111, +0.048]; against one local epoch -0.017 [-0.088, +0.057].
- Per seed (42 / 43 / 44 / 45), consensus KL 0.668 / 0.848 / 0.844 / 0.820 vs vanilla 0.816 / 0.808 / 0.632 / 0.792:
  the seed-42 gap reverses at the other three seeds.
- Exploratory, leakage at matched accuracy (`explore.py`, rounds binned by IN accuracy, `explore_matched_accuracy.csv`,
  `figures/matched_accuracy.png`): mean OUT-IN loss gap in the bins 0.75-0.78 / 0.78-0.82 is 0.020 / 0.023 for
  consensus KL, 0.032 / 0.030 vanilla, 0.027 / 0.028 weight decay, 0.026 / 0.022 fixed steps. Differences of about 0.01,
  not the seed-42 picture (+0.003 / -0.012 for consensus KL vs +0.040 / +0.035 vanilla). Not a test.
- Per-client accuracy (`explore_fairness.csv`): consensus KL's cost falls on the weakest clients (lowest client per seed
  0.545 / 0.576 / 0.619 / 0.639 vs vanilla 0.611 / 0.624 / 0.639 / 0.636).

## Step 3: under global DP, seed 42 (`analysis_step3.json`)

Global DP, clipping norm 5, noise std `r C` on the average for `r` = 1, 2, 4 x 0.001546 (the committed frontier
noise), with (`gdp-consensus-kl-0.3`) and without (`gdp`) consensus KL. 36 trajectories, 10.91 h.

| Noise | `gdp` score / AUC / accuracy | `+ consensus KL` score / AUC / accuracy | Score reduction [95%] | Accuracy difference |
|---|---|---|---|---|
| 1x | 0.788 / 0.859 / 0.7909 | 0.876 / 0.932 / 0.7663 | -0.088 [-0.164, -0.012] | -2.46 pp |
| 2x | 0.808 / 0.843 / 0.7588 | 0.776 / 0.835 / 0.7477 | +0.032 [-0.028, +0.092] | -1.11 pp |
| 4x | 0.792 / 0.858 / 0.6565 | 0.764 / 0.852 / 0.6528 | +0.028 [-0.072, +0.088] | -0.37 pp |

- **Pre-declared primary (accuracy at matched leakage).** The rule returns **"supports consensus KL"**: at the matched
  score 0.792 (gdp 4x), Delta = +0.094 [+0.008, +0.098]. This reading is recorded as returned, but it is not
  meaningful: the rule assumes the score falls with noise, and gdp's score is flat over 1-4x (0.788, 0.808, 0.792). The
  match therefore pairs consensus KL near 2x noise (accuracy about 0.75) with gdp at 4x (0.657). At the matched score
  0.788 (gdp 1x) the same rule gives -0.041 [-0.138, -0.029]. No extension was run (the rule calls for one only when
  inconclusive).
- **Matched noise (secondary).** Consensus KL costs accuracy at every noise level and is significantly worse at 1x
  (more leakage); the reductions at 2x and 4x include zero.
- Side observation for the programme: at these noise levels (all formally vacuous) global DP does not lower the
  clean-view score; doubling or quadrupling the noise costs 3-13 pp accuracy without measurable leakage reduction.

## Not run

Step 2 (beta 0.1, 0.05): owner decision 2026-10-16, since step 1 removed the effect it was meant to make cheaper.

## Files

`PROTOCOL.md`, `STATE.md`, `analyze.py` / `analysis_step1.json` (step 1), `explore.py` and its CSVs and figure,
`runner_dp.py` / `analyze_dp.py` / `analysis_step3.json` (step 3), `results/seeds43-45/`, `results/step3/`
(prediction `.npz` files are gitignored). Worker logs stay outside the repository.
