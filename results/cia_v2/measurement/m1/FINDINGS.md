# M1 findings: shared-noise confirmation

Package M1 of [research/cia_roadmap_v2/ROADMAP.md](../../../../research/cia_roadmap_v2/ROADMAP.md). Design, statistic
and gate are in [STATE.md](STATE.md). All numbers come from [analysis.json](analysis.json), which
`python -m results.cia_v2.measurement.m1.analyze --out results/cia_v2/measurement/m1/analysis.json` regenerates from
the committed trajectories in `results/`.

## Verdict

**G-M1: confirmed.** With independent noise and client seeds per world, the 4x paired clean-view score is **0.516**
(cluster-bootstrap 95% interval 0.468-0.548). The threshold is 0.70 or below, and REVIEW.md predicted 0.62 or lower.
v1's coupled design gave 0.792 at the same noise.

Consequences (ROADMAP.md Section 4):
- M2 proceeds.
- From now on, every new document labels v1's DP leakage numbers "paired, shared noise".

## Results

The paired score is an effect size, not attack success (measurement standard, Section 2 item 6). It is based on one
seed and five targets, so the bootstrap has five clusters.

| Noise (r) | Paired score, decoupled | 95% interval | Coupled (v1) | Oracle-midpoint AUC | Top decile (23, 7) | Mean final accuracy (range) |
|---|---|---|---|---|---|---|
| 1x (0.001546) | 0.788 | 0.632-0.928 | 0.788 | 0.822 | 0.90 | 0.797 (0.784-0.818) |
| 4x (0.006184) | **0.516** | 0.468-0.548 | 0.792 | 0.506 | 0.55 | 0.668 (0.645-0.688) |

Per target (fraction of rounds 1-50 at which OUT's clean shadow loss exceeds IN's):

| Target | 0 | 1 | 3 | 7 | 23 |
|---|---|---|---|---|---|
| 1x | 0.62 | 0.54 | 0.98 | 0.92 | 0.88 |
| 4x | 0.52 | 0.42 | 0.54 | 0.54 | 0.56 |

## Decoupling check

The realized-noise-energy correlation between the IN trajectory and each OUT trajectory, over 50 rounds:

- At 1x it ranges from -0.241 to 0.078.
- At 4x it ranges from -0.180 to 0.038.
- The largest magnitudes come from target 3, about 1.7 standard errors of a zero correlation over 50 rounds
  (1/sqrt(50) ≈ 0.14).

On v1's coupled step-3 trajectories the same statistic is 0.925-0.997. This is the regression test in
`tests/test_m1_analyze.py`. The runs therefore drew independent noise, as designed.

## Reading

- **v1's flat noise response was an artifact of shared noise.** At 4x, independent noise removes the effect: both
  the paired score and the oracle-midpoint AUC are at chance. v1's 0.792 at 4x came from IN and OUT drawing the same
  noise. Their difference then contained only the target's signal, so it did not shrink as the noise grew.
- **At 1x the effect survives decoupling.** The paired score stays at 0.788, with a wide interval (lower end 0.632),
  and the top decile is at 0.90. The leakage at the frontier noise is real. Between 1x and 4x it falls to chance,
  while mean accuracy drops by 0.13.
- **Limits.**
  - The design has one seed and one IN trajectory per noise level, so every target's comparison shares that IN
    trajectory.
  - The interval treats the five targets as clusters and is not calibrated.
  - The paired score is v1's statistic, kept on purpose so that only the coupling changes. Calibrated attacker AUCs
    with randomized inclusion come from M2.

## Cost

12 trajectories, 4.08 worker-hours in total (training plus frontier evaluation). Workers on GPU 1 (one, then two from the end of the M0 cost run) ran from
18:39 to 20:51, about 2.2 GPU-h of occupancy, within the 6 GPU-h cap.
