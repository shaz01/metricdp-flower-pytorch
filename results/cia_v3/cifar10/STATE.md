Package: P2, CIFAR-10 replication of A-3 and D-4 (CIA roadmap v3, Agent P; direction F2)
status: stage a running (GPU 1, shared; ROADMAP decision 4)
gate: G-P2a, then G-P2b; not yet evaluated
protocol: results/cia_v3/cifar10/PROTOCOL.md (fixed before any run)
machine: compute server
cost: 0 GPU-h; cap 120 GPU-h (ROADMAP decision 5), escalate at 150
next: stage a (vanilla, replicates 0-7), then G-P2a
needs from others: none

Roadmap: `research/cia_roadmap_v3/ROADMAP.md` Section 3 (P2), Sections 1 (decisions 2, 8), 2 and 4; decision 5 in
Section 10. Branch `feature/cia-v3-cifar10` from `feature/cia-v3-infra` a8e52039 (P0's CIFAR-10 profile; G-P0 passed
2026-10-31). Sparse worktree `../metricdp-pytorch-feature-cia-v3-cifar10`.

## Hypothesis

The two central EuroSAT results replicate on a second label-skewed dataset with real leakage:
- **A-3:** at matched accuracy, metric privacy leaks like global DP (G-A1: equivalent or no detectable difference);
- **D-4:** the composite (client subsampling `q = 0.25` with logit adjustment `tau = 1`) leaks at least 0.05 AUC less
  than metric privacy at matched accuracy, under the standard and the sampling-aware attackers.
First CIFAR-10 must leak at all under the measurement standard (G-P2a), which Alzheimer did not (A-6).

## Design

- **Regime: M2's on CIFAR-10** (P0's `cifar10` profile): ten-class CIFAR-10, label-Dirichlet alpha 0.3, 48 slots, layout
  seed 42, 32 candidates (top decile 26, 27, 36, 12, 45 plus 27 by size quantile), 16 IN per replicate, 16 always in,
  32 active; 50 rounds, 5 local epochs, clipping 5, equal weights; `R = 32`, `design_seed = 2`; 1,024 trials per
  configuration.
- **Stage a:** vanilla, replicates 0-7 of the `R = 32` design (four complementary pairs, each candidate IN in 4).
  These runs are reused as vanilla's first 8 replicates in stage b.
- **Accuracy pilot** (only after G-P2a passes; `design_seed = 3`, replicates 0-1, accuracy only): global DP and
  metric privacy at 1x, 2x, 4x, 8x of `r = 0.001546`; the composite at `r` = 0.0001, 0.000773, 0.003092, 0.006184,
  0.012368 (B1's and C1's levels). 26 runs.
- **Stage b:** vanilla (all 32 replicates), global DP at 1x, 2x and 4x, metric privacy at 3 levels and the composite at
  3 levels chosen from the pilot by the rule in PROTOCOL.md, `R = 32` each, in two halves with A1's replacement rule
  between them. The composite is scored at every round (standard v3 item 4).
- **Targets:** global DP's stage-b mean final accuracy at its 2x and 4x levels (as on EuroSAT, where they were 0.740
  and 0.660).
- **No private randomness is shared between worlds:** every replicate draws its own `client_seed` and `noise_seed`
  from `(design_seed, replicate)`, the inclusion of the 32 candidates is randomized by the balanced design, and the
  composite's client sampling uses the run's own seeds. The same replicate of two configurations shares its seeds and
  inclusion, which is allowed across configurations and makes the matched-accuracy differences paired. Public
  randomness (layout, initial model, shadow samples, test split) is shared.

## Attacker and metrics

- **Gated attacker:** v2's calibrated likelihood-ratio attacker on the 11-checkpoint shadow-loss trajectory (rounds 0,
  5, ..., 50), all candidates; for the composite also the **sampling-aware attacker** of
  `research/cia_roadmap_v3/reanalysis/reanalyze.py` (`sampling_aware`, centred window-drop statistic).
- **Matched-accuracy statistic:** A1's (curves linear in mean final accuracy; pair jackknife at 97.5%, the same pairs
  in every configuration; two-way bootstrap alongside).
- Reported, not gated: top decile; TPR at 5% and 1% FPR; the composite's 51-checkpoint attacker; accuracy and the
  lowest-client accuracy interpolated at the targets (standard v3 item 6).
- **Power** (standard v3 item 2): the gates are paired matched-accuracy differences, detectable at about 0.03-0.04
  (all candidates) and 0.09-0.10 (top decile) at `R = 32`.
- **Reproduction first:** the P2 analysis reproduces A1's published metric privacy against global DP at 0.740,
  -0.000 [-0.019, 0.018], to 3 decimals before reporting any CIFAR-10 number.

## Gate

**G-P2a** (ROADMAP v3, verbatim): "Vanilla all-candidate AUC at least 0.65 at `R = 8`. Otherwise stop, write
`ESCALATION.md` and propose FEMNIST (decision 2)." Applied to the calibrated attacker's point estimate on stage a's
8 replicates; the pair-jackknife interval (4 pairs) is reported.

**G-P2b** (ROADMAP v3, verbatim):
- "G-A1 for metric privacy against global DP, reported."
- "G-C1 for the composite against metric privacy, under the standard and the sampling-aware attacker."

G-A1: better (interval excludes 0, negative), equivalent (interval within ±0.03), or worse or inconclusive; at each
target. G-C1 (decision 9): the composite's all-candidate AUC at least 0.05 below metric privacy's, with the 97.5%
pair-jackknife interval of the difference below 0, at each target; stated per attacker.

**Untestable targets** (standard v3 item 5): a target that a mechanism's curve does not reach after the replacement
rule is "not testable" for every comparison involving that mechanism and does not count as a pass. Outcomes:
- D-4 **replicates**: G-C1 passes at both targets (stated per attacker);
- **partial**: passes at one target, the other not testable;
- **does not replicate**: fails at a testable target, or no target is testable ("not testable").
A-3 replicates if G-A1 is "equivalent" or "no detectable difference" at every testable target.

## Cost

P0 cost runs (one worker; GPU shared with another user's job): CIFAR-10 vanilla 0.542 run-hours, composite 0.360 at
11 checkpoints. With every-round scoring the composite's evaluation grows from 248 s to about 1,150 s (EuroSAT ratio),
about 0.61 run-hours. Estimate: stage a 4.3, pilot 12.4, stage b vanilla 13.0, global DP 53.8, metric privacy 53.8,
composite 58.6; **about 196 run-hours**.

Note: ESCALATION.md (P0) converted run-hours to GPU-h with C1's two-worker ratio (0.68). Under ROADMAP decision 4 only
one of our workers fits on a shared GPU (peak 3.6 GB), so occupancy equals run-hours. At that rate P2 projects above
its 150 GPU-h escalation point; this is raised with the owner before stage b (stage a and the pilot cost about 17).

## Files

| File | Change |
|---|---|
| `results/cia_v3/cifar10/` | STATE, PROTOCOL, `analyze.py` with `tests/`, `pilot/`, `results/`, `analysis.json`, FINDINGS, figures, tables |
| `STATUS.md` | Agent P rows only |

The runner and data module are P0's, unchanged.

## Log

| When | What |
|---|---|
| 2026-10-31 12:43 | Stage a started on GPU 1, shared with another user's job (4.3 GB free), 4 parallel clients. |
| 2026-10-31 12:44 | **Out of memory in round 1:** the other user's job on GPU 1 grew from 6.5 to 8.1 GB. One client failed, and `--require-all-clients` stopped the run before anything was saved. Nothing partial is kept (no `complete.json`), and the replicate retrains from scratch with the same seeds. Stopped (own session only). |
| 2026-10-31 12:46 | Restarted through a retrying queue (`~/cia-server/v3-queue.sh`): 2 parallel clients (about 1.5 GB; parallelism changes scheduling, not the trajectories), and a shared GPU qualifies only with at least 3.5 GB free (our 1.5 GB plus the other job's observed 1.6 GB swing). Measured 1.4-1.5 GB. |
