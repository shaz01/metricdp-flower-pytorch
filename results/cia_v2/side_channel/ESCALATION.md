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
