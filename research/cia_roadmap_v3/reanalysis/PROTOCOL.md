# Re-analysis protocol: three checks on committed v2 runs (no training)

Fixed 2026-10-30, before any of the statistics below was computed. They are new statistics on runs whose standard
results are already reported (M2, A1, B1, C1), so this protocol fixes the attackers, comparisons and decision rules,
not the runs. CPU only; no run is launched.

**Inputs** (committed on master `c885c13f`):
- `results/cia_v2/frontier/results/`: M2, global DP; configs `gdp-r1x` ... `gdp-r16x`, and `vanilla`.
- `results/cia_v2/frontier/a1/results/`: A1, metric privacy `a1-mp-*` and private median `a1-pm-*`.
- `results/cia_v2/mechanisms/b1_subsampling/results/`: B1, `b1-q0.25-*` and `b1-q0.5-*`.
- `results/cia_v2/composite/results/`: C1, `c1-*`.
- The seed-42 layout's per-client records and class counts: the `partitions.json` committed with
  `results/cia_frontier/consensus_kl/results/step3/` (alpha 0.3, 48 clients, seed 42, the v2 layout).

**Shared machinery.** The attacker is `experiments.cia.audit_attacks.calibrated_scores`, fitted leave-one-replicate-out
as everywhere since M0. The matched-accuracy statistic is A1's:
- curves linear in mean final accuracy, no extrapolation;
- pair jackknife at 97.5% for the gate, two-way bootstrap at 95% (2,000 resamples, M2's seed) alongside.

The implementation must first reproduce C1's published gate numbers against metric privacy (-0.095 [-0.137, -0.053]
at 0.740; -0.108 [-0.163, -0.053] at 0.660) to 3 decimals before any new number is read.

## (a) Sampling-aware attacker

**Threat model.**
- The attacker knows, for each candidate, the rounds in which that candidate's slot would be sampled. This is the case,
  for example, when the schedule comes from a public seed or is announced to participants.
- It does not learn whether the candidate's data took part.
- This is the strongest attacker against amplification by sampling, whose guarantee assumes the sample is secret.

**Schedules.**
- IN trials: the real per-round schedule (`train_metrics[t]["client-ids"]` of the run).
- OUT trials: an i.i.d. Bernoulli(`q`) schedule per round, from `numpy.random.default_rng([20261030, candidate,
  replicate])`. Under OUT the trajectory does not depend on any schedule, so a simulated one is exactly what this
  attacker would hold.
- Check before scoring: the realized IN sampling fraction per configuration is reported and must lie within 0.02 of `q`.

**Features.**
- `n_w` is the number of sampled rounds in window `w` (rounds `5w-4` to `5w`, `w = 1..10`).
- `d_w = x_{w-1} - x_w` is the drop in shadow loss over the window.
- `z = sum_w (n_w - 5q) (d_w - dbar_{c,w})`, where `dbar_{c,w}` is candidate `c`'s mean drop in window `w` over all
  its replicates, IN and OUT. That mean uses no membership labels, so the attacker can compute it from the reference
  trajectories it already holds. Under OUT, `z` has mean 0 whatever the candidate's difficulty. Under IN it is
  positive if training on the candidate lowers its loss.
- **Amendment, 2026-10-30, before any real-data statistic was computed.** The first definition used uncentred drops,
  `z = sum_w (n_w - 5q) d_w`. Its synthetic unit test (`tests/test_reanalyze.py`) showed that the common loss decay,
  multiplied by the schedule deviations, swamps the per-round effect: per-trial signal-to-noise was about 0.1.
  Centring removes the common decay and leaves the definition otherwise unchanged.

**Attackers.**
- **Primary (SA):** `calibrated_scores` on the 12 features, the 11 standard shadow losses plus `z`.
- **Diagnostic:** `z` alone, through `calibrated_scores`.
- Full-participation mechanisms are scored with the standard attacker: with every round sampled, `n_w = 5` and `z`
  carries no information.

**Comparisons** (all candidates; top decile reported, not decided on):
1. C1 under SA against metric privacy under the standard attacker, at `a*` = 0.740 and 0.660. **This is the
   deciding comparison.**
2. B1 `q = 0.25` under SA against global DP and against metric privacy, at the same targets.
3. Per configuration (B1 `q = 0.25`, `q = 0.5`, C1): SA AUC minus standard AUC on the same trials, with the pair
   jackknife.

**Decision rule** (comparison 1):

| Outcome | Condition |
|---|---|
| **Survives** | Difference at most -0.05 and gate upper end below 0, at both targets. |
| **Survives, weakened** | Survives, but the point estimate at either target is at least 0.03 higher than under the standard attacker (-0.095 and -0.108). |
| **Lost** | Otherwise. "Beats metric privacy" then holds only for secret sampling. |

## (b) Early-stopping control

**Question.** At `q = 0.25` a client trains in about 12.5 of 50 rounds. Is the gain a specific effect of sampling,
or what any mechanism with fewer contributions per client gives?

**Comparator.** Full-participation global DP (M2's `gdp-r1x` ... `gdp-r16x` and `vanilla`) stopped after `T` rounds,
for `T` in {10, 15, 20}:
- The attacker sees only the checkpoints up to `T` (3, 4 and 5 checkpoints), and is fitted as above.
- Accuracy is the server test accuracy recorded at round `T`.
- One curve per `T` across noise levels. Metric privacy (A1) is treated the same way and reported.

**Comparisons.** B1 `q = 0.25` and C1 (standard attacker, 50 rounds) against early-stopped global DP at each `T`, at
every target in {0.740, 0.700, 0.660} that both curves bracket.

**Bias, stated in advance.** With fewer checkpoints the early-stopped attacker is weaker. This favours the comparator,
so a subsampling win here is conservative.

**Decision rule:**

| Outcome | Condition |
|---|---|
| **Specific to sampling** | B1 `q = 0.25` is at least 0.05 below early-stopped global DP with gate upper end below 0, at every bracketed target, for both `T = 10` and `T = 15`. |
| **Reducible to fewer contributions** | For some `T` in {10, 15}, the gate interval at a bracketed target contains 0 and lies within ±0.05. |
| **Inconclusive** | Otherwise. |
| **Not testable** | No target is bracketed at `T = 10` or 15. |

## (c) Per-candidate breakdown (descriptive, no decision)

**Statistic.** Per candidate:
- AUC over its 32 trials, per configuration;
- interpolated in configuration mean accuracy to `a*` = 0.740 and 0.660;
- for C1 and for metric privacy.

`Delta_c` = C1 minus metric privacy.

**Reported:**
- the distribution of `Delta_c`;
- the fraction of candidates with `Delta_c < 0`;
- Spearman correlations of `Delta_c` with the candidate's records and with the entropy of its label distribution
  (natural log).

Each candidate's AUC rests on 16 IN and 16 OUT trials, so these are noisy and only the pattern is read.

## Outputs

- `reanalysis/reanalyze.py`: one script, reading committed runs only. Tests in `reanalysis/tests/`.
- `reanalysis/analysis.json` and `reanalysis/FINDINGS.md`: at most two plain-language claims, each with one plot.
- Figures: `figures/sampling_aware_attacker.png` and `figures/early_stopping_control.png`.

Run from the repo root:
`PYTHONPATH="$PWD" uv run --no-sync python research/cia_roadmap_v3/reanalysis/reanalyze.py`
