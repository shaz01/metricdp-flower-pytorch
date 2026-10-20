# A2 findings: metric privacy's noise-level side channel in small federations

Package A2 of [research/cia_roadmap_v2/ROADMAP.md](../../../research/cia_roadmap_v2/ROADMAP.md). Design and gate are
in [STATE.md](STATE.md), the run order and the 2026-10-19 amendment in [PROTOCOL.md](PROTOCOL.md) (Section 6), and
the two escalations in [ESCALATION.md](ESCALATION.md). Numbers come from [analysis.json](analysis.json), which this
command regenerates from the committed trajectories:

```
python -m results.cia_v2.side_channel.analyze --out results/cia_v2/side_channel/analysis.json
```

Pilot numbers come from `python -m results.cia_v2.side_channel.analyze results/cia_v2/side_channel/results/pilot --pilot`.

## Verdict

**G-A2: channel size recorded.** One cell, Alzheimer 3, meets both conditions as the gate is written. The strong
claim needs two. Alzheimer 5 meets neither.

Alzheimer 3's pass is not a finding about global DP. Its global-DP loss AUC of 0.047 comes from one diverged
trajectory that reverses the calibrated attacker's direction (see Reading). A plain loss threshold on the same runs
separates the worlds with AUC 0.797.

## Pilot

Global DP, 4 replicates per world. Statistic: the calibrated loss attacker's point AUC; a level qualifies at 0.6 or
below, and since the amendment only if no trajectory collapses.

| Cell | 1x | 4x | 16x | Level |
|---|---|---|---|---|
| Alzheimer 3 | 1.000 (acc 0.870) | 0.375 (acc 0.496) | not run | 4x |
| Alzheimer 5 | 1.000 (acc 0.775) | 0.563 (acc 0.460) | not run | 4x |
| Alzheimer 8 | 0.813 (acc 0.764) | 0.625 (acc 0.495) | 0.375 (acc 0.247), all 8 trajectories collapse | none |
| EuroSAT 3 | 1.000 (acc 0.936) | 1.000 (acc 0.892) | 1.000 (acc 0.438) | none in the run grid (64x not run) |

Accuracies are mean IN accuracies.

## Main runs at 4x

8 replicates per world, 16 trajectories per mechanism and cell. Intervals: pair jackknife at 97.5%; it can leave
[0, 1] at this size.

| Cell | Mechanism | Loss AUC | Final-round loss AUC | Noise-level AUC (mean `sigma_hat`) | 50-round noise-level AUC | Oracle noise AUC | Accuracy IN / OUT | Collapsed trajectories (rounds) |
|---|---|---|---|---|---|---|---|---|
| Alzheimer 3 | global DP | 0.047 (-0.105-0.199) | 0.062 | 0.281 (-0.066-0.628) | 0.375 | 0.500 | 0.476 / 0.427 | 0 |
| Alzheimer 3 | metric privacy | 0.984 (0.918-1.051) | 0.875 | **1.000 (1.000-1.000)** | 1.000 | 1.000 | 0.755 / 0.461 | 5 (16) |
| Alzheimer 5 | global DP | 0.672 (0.242-1.102) | 0.906 | 0.516 (0.007-1.024) | 0.234 | 0.500 | 0.427 / 0.359 | 0 |
| Alzheimer 5 | metric privacy | 1.000 (1.000-1.000) | 1.000 | 0.875 (0.520-1.230) | 1.000 | 0.906 | 0.665 / 0.331 | 8 (141) |

## Reading

- **Metric privacy's noise level is observable.** In both cells the blind attacker, which sees only the released
  updates' norms, separates IN from OUT almost perfectly: 1.000 and 0.875 on the round mean, 1.000 on the 50-round
  vector. Global DP's noise is fixed, so the same attacker is at chance there (oracle 0.500).
- **But at the same noise multiple metric privacy also leaks through the loss.** Its loss AUC is 0.984 and 1.000.
  It keeps far more accuracy (IN 0.755 and 0.665 against 0.476 and 0.427 for global DP), so it adds much less
  noise at the same multiple. The side channel is not the only leak, nor the largest.
- **Global DP's loss AUC is unreliable at this noise.** In Alzheimer 3 one IN trajectory diverged (final loss 63.9
  against 1.1-1.5 for the others). Every leave-pair-out reference set but its own contains it, so the calibrated
  attacker learns "IN has higher loss" and scores 0.047. The final-round loss threshold, which has no fitted
  direction, gives 0.797 in Alzheimer 3 and 0.969 in Alzheimer 5. Global DP at 4x therefore still leaks in both cells,
  and the pilot decisions made with the calibrated attacker at 4x (0.375, 0.563, 0.625) share this weakness.
- **The noise levels that silence global DP also break training.** Wherever global DP's attacker drops at all,
  Alzheimer accuracy is 0.43-0.50 (majority-class rate about 0.5). Alzheimer 8 at 16x collapses to chance (0.247).
  EuroSAT 3 still leaks fully at 16x with accuracy 0.438.
- **Collapse is part of metric privacy's channel.** 13 of its 32 main trajectories collapse in late rounds; the
  amendment counts those rounds as a released norm of 0, which an observer of the released models sees.

## Limits

- Two cells, 8 replicates per world, one target per cell, one layout seed.
- The calibrated attacker is not robust to diverged trajectories, which matters wherever global DP's noise
  approaches collapse.
- The mechanisms are compared at equal noise multiples, not at matched accuracy.
- The 2026-10-19 amendment (collapsed rounds; no further runs) was made after the runs but before any main-run
  statistic.

## Cost

About 25 GPU-h of the 35 GPU-h cap.

| Step | Trajectories | GPU-h |
|---|---|---|
| 1x pilot (one worker) | 32 | 13.7 |
| 4x pilot (4 workers per GPU) | 32 | 3.8 |
| 16x pilot | 16 | 1.9 |
| Main runs at 4x | 64 | 5.6 |
