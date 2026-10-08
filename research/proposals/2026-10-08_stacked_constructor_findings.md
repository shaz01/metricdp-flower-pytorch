# Stacked public-plus-private constructor: development freeze and fresh-task transfer findings

Protocol: [stacked-constructor protocol](2026-10-08_stacked_constructor_protocol.md) (gates stated before fresh-task data was opened). Code `research/calculations/stacked_constructor_probe.py`; artifacts `results/client_specific_noise/stacked_constructor_{freeze,transfer}.{json,npz}`; independent reconstruction `audit_stacked_constructor.py` (378 checks: role disjointness/class counts, the whole public candidate family and strong-control selection refit from scratch, queries at the control, clipping, noise, 512 CE draws per cell/risk, gains, expected AUCs; max noisy-CE draw error 2.8e-8, amplified by normalizing small-norm clipped queries; control CE matches to 1e-9).

## Construction and freeze

Base = strongest selection-frozen public model T0; clipped, projected, noised client gradients at T0 are added as theta = T0 - eta*decode(total). Pooled freeze on the six Fashion-MNIST 0-3 development cells (selection CE only): all risks use `target_center`, d=51; caps/eta .01/10 (q.55), .003/100 (q.65, q.80); none at the eta grid edge. Pooled selection CE .6024/.5951/.5933 against .6052 for the strong public control alone.

## Fresh-task transfer (no re-tuning)

Two tasks no frozen choice ever saw: Fashion-MNIST train classes 4-7 (same domain, new classes) and MNIST train digits 0-3 (new dataset). Per task: 3 public 32-sets x 2 private cohorts = 6 cells, 2,048 held-out images, 512 noise draws, 2,048-draw attack per target slot. All roles disjoint.

Gain in held-out CE over the strongest selection-frozen public-only model (= what a deployer without private data would use), cells where gain > .001 / mean / range:

| Task | q.55 | q.65 | q.80 |
|---|---|---|---|
| Fashion-MNIST 4-7 | 6/6, .0081, .0015-.0173 | 6/6, .0224, .0102-.0446 | 6/6, .0247, .0131-.0467 |
| MNIST 0-3 | 6/6, .0081, .0053-.0104 | 6/6, .0219, .0127-.0270 | 6/6, .0243, .0143-.0300 |

Accuracy (mean over cells, q.80): Fashion-MNIST 4-7 70.5% vs 68.8% control; MNIST 0-3 81.3% vs 80.5%. Winning public base families varied (public_step, public_refine, public_scratch), so the gain is not tied to one base.

Attack check (peer-conditioned known-alternative likelihood ratio, 8 target slots x 6 cells): mean AUC .549/.650/.800 (FMNIST) and .548/.649/.800 (MNIST) against calibrated .55/.65/.80. Largest single standardized deviation |z| = 3.83 (MNIST q.55) among 288 comparisons, 2.2-2.9 elsewhere; mean deviations are within .002, so I treat this as sampling plus a slightly optimistic standard error rather than a miscalibration, but have not run a replication to prove it.

## Pre-stated gate: NOT MET

The protocol required beating BOTH the strongest public control and the previous unstacked frozen construction in at least 5 of 6 cells per task. Versus the unstacked construction the stacked one is a wash: mean difference −.0007 (FMNIST q.65/.80) and +.0010 (MNIST q.65/.80), per-cell range −.0031 to +.0049; it clears .001 in 0/6 and 2/6 cells at q.65 (FMNIST/MNIST). So stacking does not give a reliable improvement over the unstacked construction. What it does provide is robustness by design (it cannot fall below its public base): it never lost to the public control, whereas the unstacked version lost in 1 of 3 fresh test-split subsets earlier.

## Interpretation

The reproducible result is that a single pre-frozen client-query construction (clip each client's balanced gradient to a small public cap, sum, add calibrated peer-contract Gaussian noise, step from the best public model) beats the strongest public-only alternative in 24/24 fresh-task cells at calibrated conditional risk .65 and .80 (and .55, with small gains), on new classes and a new dataset, while the calibrated known-alternative attack stays at its target. This is the first positive utility result under matched, verified attack strength. Open: gains at q.55 are small; the stacked/unstacked difference is unresolved; and nothing here shows a non-Gaussian or client-specific noise law helps.

## Limits

Gaussian only; the "client-specific" element is only the per-client clip/normalization and the balanced-gradient construction, not a distinct per-client noise distribution; tiny linear model on pooled 4x4 features, not a CNN/Flower federation; both tasks reuse the same 4-class label-stress partition design; cells share cohorts/evaluation images; offline hyperparameter tuning, public-reference choice and artifact release unaccounted; conditional peer contract with known-alternative attack only; the earlier 32-vs-512 anchor finding still holds (no private gain once public data are plentiful, not re-tested here at larger budgets on the new tasks).

## Next (proposed)

(1) Larger-public check on the fresh tasks (128/512) to map where the private gain disappears, with the same frozen configuration. (2) Matched-strength comparison of non-Gaussian and per-client-shaped noise on this construction, the only place the original client-specific-distribution question can still add value. (3) Port the construction to the actual CNN/Flower pipeline.
