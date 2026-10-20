# A2 escalation: the stop rule triggers after the 1x pilot

Raised 2026-10-18 by Agent A under PROTOCOL.md Section 3 ("If the projected total exceeds the 35 GPU-h cap by more
than 25%, the package stops and escalates before the main runs"). Nothing further runs until the owner decides.

## What the 1x pilot found

Global DP at WP1's base ratio, 4 replicates per world (`python -m results.cia_v2.side_channel.analyze
results/cia_v2/side_channel/results/pilot --pilot`). The pilot statistic is the calibrated loss attacker's point AUC.

| Cell | Loss AUC (point) | Noise-level AUC (point) | Mean IN accuracy | Pilot decision |
|---|---|---|---|---|
| Alzheimer, 3 clients | 1.000 | 0.750 | 0.870 | next multiple (4x) |
| Alzheimer, 5 clients | 1.000 | 1.000 | 0.775 | next multiple (4x) |
| Alzheimer, 8 clients | 0.813 | 0.875 | 0.764 | next multiple (4x) |
| EuroSAT, 3 clients | 1.000 | 1.000 | 0.936 | next multiple (4x) |

No cell qualifies at 1x: in every one, global DP's loss attacker still separates the worlds (AUC 0.81-1.00).
With 8 trajectories per cell these are pilot numbers only, not results.

## Cost

| Item | Measured |
|---|---|
| Time per trajectory, one worker, training plus evaluation | 0.40-0.45 h in every cell (EuroSAT 0.42) |
| Planning figure (PROTOCOL.md) | 0.07 h Alzheimer, 0.35 h EuroSAT |
| 1x pilot | 32 trajectories, about 13.7 GPU-h |

During the pilot the machine was CPU-overloaded by other users (load average 50-80 on 48 cores). Small federations
leave the GPU almost idle (1-2% utilization), so they are bound by CPU, not GPU.

**Projection at the measured cost.** Every cell needs at least the 4x pilot level.
- The least the package can still need: 4x pilot (32 trajectories) plus main runs (128) = 160 trajectories, about
  69 GPU-h, for a total of about 83 GPU-h.
- The worst case, with 16x and 64x pilots, adds 64 trajectories, about 27 GPU-h.
- Either exceeds the stop threshold of 1.25 x 35 = 43.75 GPU-h.

## Options for the owner

1. **Run several trajectories per GPU (recommended).** The GPU is nearly idle during small-federation runs, so 4
   workers on one GPU should cost about a quarter of the GPU-h per trajectory if the CPUs are free (load is now about
   6). Re-measure on the 4x pilot (32 trajectories) and continue only if the projection then falls under the stop
   threshold. Protocol, cells and gate are unchanged.
2. **Raise the cap** to about 90 GPU-h at the measured single-worker cost.
3. **Reduce the scope.** G-A2 needs the claim in at least two cells. For example, keep two Alzheimer cells (5 and 8
   clients) and jump the pilot from 1x straight to 16x. That changes the fixed protocol and lowers the gate's
   chance of a strong claim.
4. **Stop A2** and record the channel at 1x only. With global DP's loss attacker still at AUC 0.81-1.00, no strong
   claim is possible there.

## Resolution (2026-10-18, owner)

Option 1: several trajectories per GPU. The 4x pilot (32 trajectories, global DP, unchanged protocol) runs with
4 workers on one GPU. It starts on the first GPU with no process and no claim (`~/cia-server/a2-wait.sh`, tmux
`a2-wait`). The cost per trajectory is then re-measured as GPU occupancy divided by the trajectories completed. A2
continues only if the projected total falls under the 43.75 GPU-h stop threshold; otherwise it escalates again.

## Re-measurement on the 4x pilot (2026-10-18)

The 4x pilot ran with 4 workers on GPU 1 from 15:39 to 19:29: 3.84 GPU-h of occupancy for 32 trajectories, about
0.12 GPU-h per trajectory (about 0.10 for Alzheimer, 0.16 for EuroSAT), against 0.40-0.45 with one worker.

| Item | GPU-h |
|---|---|
| Spent (1x pilot 13.7, 4x pilot 3.8) | 17.5 |
| 16x pilot, Alzheimer 8 and EuroSAT 3 | 2.1 |
| Main runs, all four cells (128 trajectories) | 14.7 |
| Projected total if both remaining cells qualify at 16x | 34.3 |
| Worst case, with a 64x pilot | 36.4 |

The projection is under the 43.75 GPU-h stop threshold, so A2 continues under the fixed protocol. The worst case is
slightly above the 35 GPU-h cap but within the 25% margin.

# Second escalation: collapsed rounds (2026-10-19)

Raised 2026-10-19 by Agent A. The 16x pilot (Alzheimer 8, EuroSAT 3) and the main runs at 4x (Alzheimer 3 and 5,
global DP and metric privacy, 64 trajectories) finished at 04:42. No further A2 run starts until the owner decides.

## What happened

Some runs contain **collapsed rounds**: every client's update has zero norm (total divergence), the strategy skips
aggregation, keeps the previous model and logs `*-aggregation-collapsed = 1` without a released-update norm
(`metricdp_pytorch/globaldp_strategy.py`, `metricdp_strategy.py`). `analyze.py` reads that norm in every round, so
it stops with a `KeyError`. PROTOCOL.md has no rule for such rounds.

| Cell and level | Trajectories with collapsed rounds | Collapsed rounds | Mean accuracy IN / OUT |
|---|---|---|---|
| Alzheimer 8, global DP 16x (pilot) | 8 of 8, from round 3-5 on | 370 of 400 | 0.247 / 0.250 |
| Alzheimer 3, metric privacy 4x (main) | 5 of 16, late rounds | 16 of 800 | 0.755 / 0.461 |
| Alzheimer 5, metric privacy 4x (main) | 8 of 16, from round 24 on | 141 of 800 | 0.665 / 0.331 |
| All other A2 levels | none | 0 | |

A diagnostic outside the committed code (collapsed rounds counted as a released norm of 0) gives these pilot
decisions:

| Cell | Loss AUC (point) | Mean IN accuracy | Literal pilot decision |
|---|---|---|---|
| Alzheimer 8, 16x | 0.375 | 0.247 (chance, 4 classes) | qualifies at 16x |
| EuroSAT 3, 16x | 1.000 | 0.438 | next level, 64x |

## Why it needs the owner

1. **Alzheimer 8 "qualifies" only because its federation is dead.** From round 3-5 on nothing is released, and the
   loss attacker sees no signal because no model is trained. Main runs there would compare two dead federations.
2. **EuroSAT 3 still leaks fully at 16x** with accuracy 0.438. 64x would almost surely end like Alzheimer 8 at 16x.
3. **The analysis needs a rule for collapsed rounds** in any case, because the metric-privacy main runs contain them.
   The rule changes the noise-level attacker's feature, which PROTOCOL.md fixed.
4. The two cells with main runs compare the mechanisms at the same noise multiple, not at matched accuracy: global DP
   at 4x has accuracy 0.43-0.48 (IN), metric privacy at 4x 0.67-0.76 (IN). That is the design, but FINDINGS will
   need to say so.

## Proposed amendment

- **Collapsed rounds:** the released update norm is 0, since an observer of the released models sees the model
  unchanged. The noise-level attacker then treats collapse as part of the observable channel, which it is for metric
  privacy (its collapse follows its distance-calibrated noise).
- **Degenerate levels do not qualify:** a pilot level qualifies only if no global-DP trajectory collapses. Then
  Alzheimer 8 has no qualifying level in the grid, EuroSAT 3's 64x pilot is not run, and A2 is analyzed on the two
  cells with main runs (Alzheimer 3 and 5). G-A2 needs two cells, so the strong claim stays reachable.

## Cost

About 25 GPU-h spent of the 35 GPU-h cap: 1x pilot 13.7, 4x pilot 3.8, 16x pilot about 1.9 (21:15-23:07), main runs
at 4x about 5.6 (23:07-04:42). Continuing by the literal protocol (Alzheimer 8 main at 16x, EuroSAT 64x pilot and
possibly its main runs) would add about 4.5-10 GPU-h.
