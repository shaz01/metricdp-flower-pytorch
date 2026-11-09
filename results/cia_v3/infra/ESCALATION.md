# Escalation: P2 and P3 project above 1.25x their caps (P0 cost runs, 2026-10-31)

Raised by Agent P under the rule in `STATE.md` (Hypothesis, claim 2) and ROADMAP v3 Section 1, decision 3 (escalation at 1.25x).
Nothing in P2 or P3 is launched until the owner decides. P1 is far below its cap and is not affected.

## Measured cost (P0 cost runs, `results/cia_v3/infra/cost/`)

One worker, 4 parallel clients (3 for the paper pair), on GPU 1 shared with another user's job that kept it at 97-100%
utilization throughout (ROADMAP decision 4). Our peak VRAM: 3.6 GB.

| Run | Checkpoints | Training (s) | Evaluation (s) | Run-hours | Final accuracy | Lowest client |
|---|---|---|---|---|---|---|
| CIFAR-10 RI vanilla | 11 | 1,869 | 81 | 0.542 | 0.630 | 0.504 |
| CIFAR-10 RI composite (`q` 0.25, `tau` 1, `r` 0.003092) | 11 | 1,047 | 248 | 0.360 | 0.546 | 0.392 |
| Paper pair, IN world (global DP, `z` 0.01) | 21 | 228 | 15 | 0.067 | 0.920 | 0.926 |
| Paper pair, OUT world | 21 | 201 | 15 | 0.060 | 0.831 | 0.877 |
| EuroSAT RI composite, every round (`r` 0.003092) | 51 | 460 | 429 | 0.247 | 0.671 | 0.400 |

**Calibration against v2.** The EuroSAT composite trained in 460 s; C1's 128 committed runs of the same configuration
(11 checkpoints, two workers on the same shared GPU) have a median of 440 s. So these timings match v2's conditions.
C1 recorded 11.5 GPU-h of occupancy for 16.9 run-hours (ratio 0.68 with two workers). Below, run-hours are the upper
bound (one worker) and GPU-h at C1's ratio the expected value.

CIFAR-10 trains 2.4-3.0x longer than EuroSAT per run (2.3x the training records: 50,000 against 21,600; a different CNN).
Evaluating 51 checkpoints costs about 3.6x evaluating 11.

## Projection per package

Per-run assumptions: CIFAR-10 global DP and metric privacy as vanilla plus 4% (EuroSAT M2: 648 s against 621 s);
EuroSAT full participation with every-round scoring = M2's 648 s training plus the cost run's 429 s evaluation; the
IN-count runs scale M2's training with the active count (24 or 40 of 32).

| Package | Runs | Run-hours | GPU-h (x0.68) | Cap | 1.25x cap | Verdict |
|---|---|---|---|---|---|---|
| P1 | 120 (20 pairs x 3 mechanisms x 2 worlds) | 7.6 | 5.2 | 40 | 50 | within |
| P2 | stage a 8; pilot 18; stage b 320 (vanilla, GDP x3, MP x3, composite x3, `R = 32`) | 171 | 116 | 85 | 106 | **above** |
| P3 | 192 (composite x2 and MP x2 every round; GDP 2x at 8 and 24 IN) | 47 | 32 | 20 | 25 | **above** |
| P0 (spent) | 5 | 1.3 | 1.3 | 3 | 3.75 | within |

P2 in detail (run-hours): stage a 4.3; pilot 8.8; stage b vanilla 17.3, global DP 52.8, metric privacy 52.8, composite
34.6. P3: composite 15.8, metric privacy 19.2, IN-count 12.3.

The P0-P3 and P5 total would be about 160 GPU-h expected (227 run-hours upper bound), against the 148 cap.

## Options for the owner

1. **Raise the caps** to P2 120 GPU-h and P3 35 GPU-h (total for P0-P3 and P5 about 170). The design stays as
   written.
2. **Trim P2's stage b** (keeps the gates):
   - drop vanilla at `R = 32` (stage a's `R = 8` gives the vanilla reference; the matched-accuracy gates do not use
     vanilla): -17 run-hours;
   - metric privacy at 2 levels chosen from the pilot to bracket both targets, instead of 3: -18 run-hours.
   Together about 136 run-hours, 92 GPU-h: within 1.25x (106) but above 85. The risk is that 2 levels fail to bracket a
   target; the untestable-target rule then applies.
3. **Cheaper every-round evaluation** (code change, P3 only): score the test set only at the 11 standard checkpoints
   when `--checkpoint-every 1` is used. The shadow losses are still scored at every round. This saves an estimated
   5-8 run-hours.
4. **Wait for idle GPUs and run more workers per GPU.** This lowers GPU-h of occupancy per run but not wall time, and
   both GPUs have been busy with other users' jobs since 2026-10-30.

**Recommendation:** option 1 for P3 (35), and options 1 or 2 for P2. Meanwhile, allow P2's stage a (8 runs, about 3
GPU-h, under any of these caps), because G-P2a can stop P2 before the large spend.

## Observations for the downstream protocols (not gated)

- CIFAR-10 vanilla reaches 0.630 at 50 rounds (EuroSAT 0.803), so P2's targets (global DP's mean accuracy at its 2x
  and 4x levels) will be lower than EuroSAT's 0.740 and 0.660.
- The paper-protocol global-DP pair at `z = 0.01` reaches 0.920 (IN) and 0.831 (OUT). The paper reports 0.637 (IN) and
  0.680 (OUT) for global DP, and the 2026-08 plan suite also found about 0.94. P1's protocol has to state which noise
  reproduces the paper's setting.
