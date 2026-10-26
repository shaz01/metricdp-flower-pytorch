# AZ findings, part 1: M2 on Alzheimer MRI

Owner decision 8, as amended by the owner's choices recorded in [STATE.md](STATE.md) and [ESCALATION.md](ESCALATION.md)
(alpha 1.0, 100 rounds; M2 first within the cap). Protocol: [PROTOCOL.md](PROTOCOL.md) part 1, fixed before any
leakage run. Numbers come from [analysis.json](analysis.json):

```
python -m results.cia_v2.frontier.analyze results/cia_v2/alzheimer/results \
    --configs az-vanilla=0 az-gdp-r0.0025=0.0025 az-gdp-r0.003=0.003 az-gdp-r0.0035=0.0035 \
    --out results/cia_v2/alzheimer/analysis.json
```

## Verdict

**G-M2: useful empirical regime at `r = 0.0025`.** All three conditions hold there: the all-candidate AUC's upper gate
end is 0.568 (at most 0.6), the top-decile point estimate 0.586 (at most 0.6), and the lower end of mean accuracy 0.777
(at least 0.70).

This does **not** replicate EuroSAT's "no useful regime", and the reason is the baseline: on Alzheimer in this setup the
attacker is weak even without noise (vanilla AUC 0.564, upper end 0.620). Global DP's regime is "useful" mainly because
there is little leakage to remove.

## Results

Randomized inclusion (32 candidates, 16 IN per replicate, 32 active of 48), Alzheimer MRI, label-Dirichlet alpha 1.0,
100 rounds, `paper_cnn`, shadow samples at 25%, 11 checkpoints. `R = 32`, `design_seed = 2`. Every round of every run
had all 32 clients (`check_every_client_trained`).

| Config | `mu_total` | AUC, all candidates (97.5%) | Two-way bootstrap | Top-decile AUC (97.5%) | TPR at 5% FPR | Final-round AUC | Threshold AUC | Mean final accuracy (95%) | Lowest client accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | none | 0.564 (0.507-0.620) | 0.492-0.634 | 0.576 (0.429-0.722) | 0.078 | 0.576 | 0.513 | 0.815 (0.809-0.822) | 0.533 |
| Global DP 0.0025 | 125.0 | 0.540 (0.512-0.568) | 0.476-0.599 | 0.586 (0.471-0.702) | 0.041 | 0.489 | 0.479 | 0.784 (0.777-0.791) | 0.433 |
| Global DP 0.003 | 104.2 | 0.523 (0.490-0.556) | 0.463-0.583 | 0.502 (0.406-0.598) | 0.061 | 0.547 | 0.498 | 0.732 (0.724-0.741) | 0.341 |
| Global DP 0.0035 | 89.3 | 0.546 (0.510-0.582) | 0.471-0.616 | 0.472 (0.335-0.609) | 0.086 | 0.509 | 0.507 | 0.670 (0.658-0.681) | 0.279 |

`mu_total = sqrt(100) / (32 r)`.

| Level | All-candidate upper end (at most 0.6) | Top-decile point (at most 0.6) | Accuracy lower end (at least 0.70) |
|---|---|---|---|
| 0.0025 | 0.568, meets | 0.586, meets | 0.777, meets |
| 0.003 | 0.556, meets | 0.502, meets | 0.724, meets |
| 0.0035 | 0.582, meets | 0.472, meets | 0.658, fails |

## Reading

- **Leakage is near chance across the board.** Vanilla's AUC is 0.564 and the noisy levels 0.523-0.546; no difference
  between levels is distinguishable at this power. TPR at 5% FPR is 0.04-0.09, against 0.17-0.23 on EuroSAT.
- **Why weaker than EuroSAT:** B5's diagnostic found EuroSAT's leakage to be mostly label-distribution inference, and
  this setup has milder label skew (alpha 1.0 against 0.3; chosen because alpha 0.3 did not train). That is the
  likeliest reason, but this package did not test it (no surrogate scoring here).
- **The guarantee is still vacuous** in the useful regime: `mu_total` 104-125.
- **For the paper:** "no useful regime" is setup-dependent. With strong label skew (EuroSAT, alpha 0.3) global DP
  cannot silence the attacker at usable accuracy; with milder skew (Alzheimer, alpha 1.0) the attacker is weak to begin
  with, and DP's added protection is not measurable.

## Limits

- One layout, one skew level; the calibrated loss attacker only.
- Different model, rounds, skew and shadow fraction from EuroSAT, so absolute AUCs do not compare across datasets.
- The operating points are `r_0.6` = 0.0025 only: the grid has no global-DP level below it, so "the next-lower noise"
  does not exist (vanilla).

## Cost

| Step | GPU-h (occupancy) |
|---|---|
| Diagnosis and calibration (stopped alpha-0.3 batch, diagnosis, batches 1 and 2) | about 10 |
| First launch, 2026-10-22 11:05-13:44 (three workers on each GPU, then on GPU 1; runs with failed clients quarantined) | about 2.7 |
| Relaunch, 2026-10-22 13:47 to 2026-10-23 20:55 (two workers on GPU 1, shared by owner approval) | about 31.1 |
| Total | about 44 of the 40 GPU-h cap (stop at 50) |

# AZ findings, part 2: A1 on Alzheimer MRI

Protocol: [PROTOCOL.md](PROTOCOL.md) part 2, fixed before its runs. Numbers come from
[a1/analysis.json](a1/analysis.json) (command in PROTOCOL.md part 2, Section 3).

## Verdict

**G-A1 on Alzheimer: no robust difference between the mechanisms; one nominal "better" that does not survive the
reference interval.**

| Mechanism | `a* = 0.784` | `a* = 0.732` |
|---|---|---|
| Paper's metric privacy | **"better" by the gate interval:** -0.051 [-0.100, -0.002]; two-way bootstrap [-0.125, 0.016] | no detectable difference: -0.026 [-0.076, 0.025] |
| Private-median metric privacy | no detectable difference: -0.019 [-0.064, 0.027] | no detectable difference: -0.023 [-0.078, 0.033] |

All candidates; difference `AUC_mechanism(a*) - AUC_GDP(a*)`; pair jackknife at 97.5% (gate), two-way bootstrap (95%)
alongside. Global-DP curve: vanilla and global DP at 0.0025, 0.003 and 0.0035 (part 1).

**The one "better" outcome is fragile:**

- Its gate interval excludes 0 by 0.002, and the two-way bootstrap, which also resamples the candidate panel,
  includes 0. On EuroSAT every outcome agreed under both intervals.
- The mechanism's AUC there is 0.489, below 0.5. With leave-pair-out references the calibrated attacker's null AUC sits
  slightly below 0.5 (M0's simulations: 0.49-0.50; A2's: 0.45-0.47 at smaller sizes), so a difference between two
  near-chance AUCs partly reflects that bias.
- Four outcomes are gated (two mechanisms, two targets); at 97.5% two-sided, the chance of at least one false
  "better" among four independent null comparisons is about 5%.

So the protocol's expectation holds: with leakage near chance for every configuration, A1 on Alzheimer cannot show
that a mechanism protects better.

## Results

| Config | `mu_total` | AUC, all candidates (97.5%) | Top-decile AUC | TPR at 5% FPR | Final-round AUC | Mean final accuracy (95%) | Lowest client accuracy |
|---|---|---|---|---|---|---|---|
| Vanilla (part 1) | none | 0.564 | 0.576 | 0.078 | 0.576 | 0.815 | 0.533 |
| Global DP 0.0025 (part 1) | 125.0 | 0.540 | 0.586 | 0.041 | 0.489 | 0.784 | 0.433 |
| Global DP 0.003 (part 1) | 104.2 | 0.523 | 0.502 | 0.061 | 0.547 | 0.732 | 0.341 |
| Global DP 0.0035 (part 1) | 89.3 | 0.546 | 0.472 | 0.086 | 0.509 | 0.670 | 0.279 |
| Metric privacy 0.00125 | none | 0.506 (0.472-0.540) | 0.594 | 0.047 | 0.547 | 0.829 (0.821-0.836) | 0.543 |
| Metric privacy 0.0025 | none | 0.485 (0.436-0.533) | 0.532 | 0.057 | 0.556 | 0.773 (0.762-0.783) | 0.418 |
| Metric privacy 0.003 | none | 0.526 (0.472-0.580) | 0.545 | 0.066 | 0.479 | 0.639 (0.597-0.680) | 0.258 |
| Private median 0.000625 | 685.2 | 0.540 (0.494-0.585) | 0.590 | 0.045 | 0.534 | 0.825 (0.819-0.832) | 0.530 |
| Private median 0.00125 | 342.6 | 0.499 (0.451-0.548) | 0.458 | 0.061 | 0.500 | 0.736 (0.721-0.752) | 0.364 |
| Private median 0.0016 | 267.6 | 0.545 (0.508-0.581) | 0.624 | 0.070 | 0.530 | 0.518 (0.484-0.552) | 0.052 |

Matched-accuracy differences, top decile (reported, not gated): metric privacy -0.042 [-0.163, 0.080] at 0.784 and
+0.034 [-0.126, 0.194] at 0.732; private median -0.057 [-0.155, 0.042] and -0.041 [-0.148, 0.067].

**Replacement check** (automatic, stage-1 accuracy only): metric privacy 0.829 / 0.771 / 0.639 and private median
0.829 / 0.741 / 0.525 bracket both targets, so no level was replaced. Every run passed the run check; 4 rounds in total
collapsed (zero-norm updates, model kept).

## Reading

- **On Alzheimer the audit question has no measurable leakage to act on.** Every configuration, vanilla included, sits
  at an AUC of 0.49-0.56, and differences between them are mostly within each interval.
- **Formal guarantees:** private median's `mu_total` is 268-685 against global DP's 104-125 at similar accuracy; the
  paper's mechanism has none. As on EuroSAT, the guarantees differ while the measured leakage does not.
- **For the paper:** the EuroSAT audit (A1) stands, with matched-accuracy equivalence at 0.740. Alzheimer adds that in a
  setting with near-chance leakage no mechanism is measurably better; its single nominal "better" should be reported
  with the caveats above, not as a positive result.

## Cost (part 2)

192 runs, three workers on GPU 1 (idle and unclaimed at launch; other users' processes joined later), 2026-10-23 21:16
to 2026-10-25 12:50, plus one rerun: about 40 GPU-h of occupancy. AZ in total about 84 GPU-h (part 1 about 44).
