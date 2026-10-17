# M0 protocol: measurement infrastructure (CIA roadmap v2)

Package M0 of [research/cia_roadmap_v2/ROADMAP.md](../../../research/cia_roadmap_v2/ROADMAP.md) (Agent A, branch
`feature/cia-v2-measurement`). Written before the only M0 training run, the cost run. Design decisions and their
reasons are in [STATE.md](STATE.md); this file fixes what runs and what is measured.

## 1. Deliverables

| Item | Where | Check |
|---|---|---|
| Seed plumbing (`layout`, `init`, `client`, `noise` seeds, each defaulting to `seed`) | `experiments/reproduce/seeds.py`, `server.py`, `runner.py`, `client.py` | `tests/test_run_seeds.py` |
| Server noise from a dedicated generator; per-round released-update norm and 64-block noise energy | `metricdp_pytorch/seeded_noise.py` and the four strategy files | `tests/test_noise_seed.py` |
| Attack library: calibrated loss attacker (balanced leave-one-replicate-out references, pooled direction), threshold baseline, AUC, TPR at FPR, two-way bootstrap, paired difference bootstrap | `experiments/cia/audit_attacks.py`, `experiments/cia/inclusion.py` | `experiments/cia/tests/test_audit_attacks.py` |
| Randomized-inclusion runner | `results/cia_v2/randomized_inclusion/` | `results/cia_v2/randomized_inclusion/tests/` |
| Pipeline check 1: Gaussian calibration | `test_calibration_recovers_the_gaussian_tradeoff` | G-M0 |
| Pipeline check 2: noise independence (real 2-round Flower runs, CPU) | `results/cia_v2/measurement/tests/test_noise_independence.py` | G-M0 |
| Cost measurement | `results/cia_v2/measurement/cost_run/` | recorded in STATE.md |
| Power table | `results/cia_v2/measurement/power.py`, `tables/power_table.csv`, `tables/power_grid.csv` | copied by later protocols |

The noisy shadow view of v1 is not scored: no v2 attacker uses it. Shadow samples are scored without the training
augmentation (v1 scored them through the augmented training transform, which added random crops and flips to every
loss evaluation).

## 2. Cost run

One randomized-inclusion run, as M2 will run them. The design follows the ESCALATION.md resolution of 2026-10-17
(32 candidates); this section was amended before the cost run started:

- EuroSAT, alpha 0.3, seed-42 layout, 48 slots, 32 active clients (the 16 non-candidates plus the 16 candidates of
  replicate 0), FedAvg with equal weights, WP2 hyperparameters (5 local epochs, batch 32, learning rate 0.001, 20
  initialization epochs, clipping norm 5), 50 rounds.
- Global DP at the frontier noise: std `r C` with `r = 0.001546`, noise multiplier `r x 32`.
- Design: `R = 32`, `design_seed = 0`, replicate 0. Seeds: layout and init 42; client and noise seeds from
  `SeedSequence([0, 0])`.
- Scoring: the 32 candidates' shadow samples and label-matched surrogates at rounds 0, 5, ..., 50; test loss and
  accuracy at every checkpoint.
- Command (compute server, one idle GPU from the claims file):

  ```bash
  python -m results.cia_v2.randomized_inclusion.runner --config cost-gdp-r1x --privacy global-dp \
      --noise-ratio 0.001546 --replicates 32 --design-seed 0 --replicate-subset 0 --surrogates --execute \
      --max-parallel-clients 8 --output results/cia_v2/measurement/cost_run
  ```

Recorded: training and evaluation wall time (`complete.json`), GPU model, other jobs sharing the GPU, and the GPU-h
estimate for one run of M2 and A1. The run is a cost measurement. Its leakage numbers are not analysed (one replicate
gives no references).

## 3. Power table

`python -m results.cia_v2.measurement.power`: the model and the simulation are described in the module docstring.
Inputs: WP2's fixed-layout vanilla runs at seeds 42-45 (no DP). Output: the minimum detectable AUC difference at 80%
power between two configurations of one design, at a vanilla-sized baseline effect and at half of it, for all
candidates and for the top decile. Later protocols copy the matching row.

The specified design (16 candidates, `R` = 8, 12, 16, two-way bootstrap; tables in `tables/spec16/`) could not resolve
0.05. That was escalated (ESCALATION.md) and resolved: 32 candidates, `R = 32`, and the pair-jackknife gate interval
at the 97.5% level, calibrated by `interval_calibration.py` (`tables/power_table.csv`, `power_grid.csv`,
`interval_calibration.csv`). Later protocols copy the `R = 32` row: a detectable difference of 0.087-0.088 for all
candidates and 0.196 for the top decile at a vanilla-sized baseline.

## 4. Gate

**G-M0** passes when the tests are green, the calibration test recovers the Gaussian trade-off and the
noise-independence test separates equal from different seeds. On fail, fix before any other package runs.

The cost run and the power table are M0 deliverables but not part of the gate. On pass, the branch is pushed and
STATE.md is set to `ready` even if the cost run is still waiting for a GPU; it is then completed and recorded on the
same branch.
