# Practical model-based client inference attack and the utility-leakage frontier

Owner (2026-10-09): next steps one by one, autonomous, report at the end. Rules fixed before any frontier cell is run.

## Why

So far leakage was measured only by the calibrated known-alternative likelihood-ratio attack, which sees the individual messages and the target's exact noise-free contribution. That bounds the risk but says nothing about what a realistic attacker, who only sees the released MODEL, can do, nor how utility trades against leakage as the noise level changes.

## Attack

Target: client t of 8, whose data are label-stressed (205 examples of its dominant class t mod 4 and 17 of each other class). Worlds: IN (all eight clients contribute) and OUT (client t is replaced by a dummy that sends only its noise share, the contract used throughout). The attacker sees ONLY the released head after the validation gate and the public base, and holds a shadow set: the public validation images, reweighted to the target's class mix (205:17:17:17). Secondary statistic per release: the shadow-weighted cross-entropy of the public base minus that of the released model. The attacker needs no knowledge of the noise scale. One IN run serves every target; each target has its own OUT run.

AUC per (task, public set, risk, target) over 256 independent releases per world (replicate rounds; every release is an independent draw from the same base, as in earlier protocols).

## Amendment (before any frontier cell was run)

A 128-release sanity run of the class-mix statistic (KMNIST 0-3, set 0, risk .95, target 0) gave AUC 0.42, below chance: the construction class-balances every client's query (uniform class weights, target_center), so a client's label skew is deliberately not transmitted and a class-mix shadow attacker has no signal. This was observed on that single test only, not on any frontier cell. The PRIMARY attack is therefore the stronger, standard client-level membership attacker who holds the target's own records: **statistic = mean cross-entropy of the public base minus that of the released model, measured on the target client's actual 256 training records** (larger decrease suggests the client contributed). The class-mix statistic is kept as a secondary attack. Everything else in this protocol is unchanged and the claims below refer to the primary (own-records) attack.

## Second amendment (also before any frontier cell was run)

The raw own-records statistic also gave AUC 0.43 on the same 128-release sanity pair, because IN and OUT releases differ in overall quality (eight versus seven contributing clients; mean gated CE gain +0.022 versus +0.012), which moves the loss on every record set. The standard remedy in membership attacks is to calibrate against non-member records. **The PRIMARY statistic is therefore calibrated: (base-minus-released CE on the target's own records) minus (base-minus-released CE on the public shadow records re-weighted to the target's class mix).** On the same sanity pair (target 0, risk .95, 128 releases per world) it gave AUC 0.575. These two sanity pairs are the only attack data examined before the grid was fixed; no frontier cell has been looked at. The raw own-records and class-mix statistics are reported as secondary.

## Grid

Tasks KMNIST 0-3 and 4-7; budget 32; public sets 0-4; cohort A; targets 0-3 (one per dominant class); risk targets .55, .65, .80, .90, .95 plus a noise-free release (sigma = 0, one deterministic release per world). The construction is the frozen q.65 configuration (target_center, d=51, cap .003, eta 100) at EVERY risk, so only the noise scale changes along the frontier; gate V0 as before. Cells: IN + 4 OUT worlds per (task, set, risk) = 300 noisy runs + 50 noise-free runs.

## Quantities and pre-stated claims

Per risk: mean practical AUC across the 40 (set, target) pairs per task with a bootstrap 95% interval (resampling sets), mean utility gain over the control (IN runs, V0 gate), mean accuracy delta.

1. **Conservative calibration:** the calibrated target upper-bounds the practical attack iff, at every risk and both tasks, the bootstrap upper limit of the practical AUC is below the calibrated target. (The model-only attacker sees a function of the aggregate and of public data, so its AUC cannot exceed that of the likelihood-ratio attacker with message access; a violation would indicate a bug or miscalibration.)
2. **Detectability:** a risk level is "detectable by the practical attack" iff the bootstrap interval of the mean practical AUC excludes 0.5.
3. **Frontier:** report (mean practical AUC, mean utility gain) per risk and for the noise-free release. The noise-free release is deterministic, so its AUC is reported as 1.0 when the IN statistic exceeds the OUT statistic (a perfect threshold exists) and as 0.0 or 0.5 otherwise, and is labelled as such.

No tuning follows from these results; no claim beyond the stated attack, the frozen construction, the two KMNIST tasks and five public sets.

## Limits

The practical attack is one natural model-only statistic, not the strongest possible; a stronger attacker (shadow models, per-class calibration, many releases from the same federation) could do better; releases here are independent draws from one base, not repeated observation of a single trained federation. Single cohort; small CNN head; Gaussian noise; validation set doubles as the attacker's shadow data (public by construction).
