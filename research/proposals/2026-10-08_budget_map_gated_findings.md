# Public-budget map and validation-gated step: findings

Protocols: [budget map](2026-10-08_budget_map_protocol.md), [gated step](2026-10-08_gated_step_protocol.md) (gates fixed before data were opened). Code `budget_map_probe.py`, `gated_step_probe.py`; artifacts `results/client_specific_noise/{budget_map,gated_step}.{json,npz}`. Independent reconstruction `audit_gated_step.py` covers the fresh KMNIST tasks at budgets 32 and 128 (320 checks: roles, base-candidate refit and selection, queries, 512 noise draws x 5 multipliers x 3 risks per cell, gating; max eval-CE error 6e-14). Budget 512 and the seen-pool (Fashion-MNIST 4-7, MNIST 0-3) rows are NOT independently audited.

## Correction to the stacked-transfer summary

The earlier "24/24 cells beat the public-only control" used one draw of three public 32-sets per task. A budget map with a new role seed on the same pools shows it does not generalize across public sets for the FIXED step: at 32 examples, q.65, Fashion-MNIST 4-7 mean gain falls to −.0036 (2/6 cells > .001; two cells −.016/−.022 on the public set whose strongest public base was already best, CE .570), MNIST 0-3 stays positive (+.0138, 6/6). Pooling both public-set draws (12 cells per task, q.65): Fashion-MNIST 4-7 mean +.0085 with 8/12 cells > .001 and two clearly harmful; MNIST 0-3 12/12 positive. The first result was real but over-optimistic. Cause: the frozen step length (eta*cap = .3) is not adaptive, so it overshoots when the public base is strong.

## Budget map (fixed step, seen pools, new roles)

Gain over the strongest public control, mean across 6 cells: at 128 examples +.0011 (Fashion-MNIST 4-7) / +.0055 (MNIST 0-3) at q.65 and +.0030 / +.0072 at q.80; at 512 examples zero or negative in every setting (−.0019 to −.0002 at q.65/.80). The earlier 512 anchor result is confirmed on new tasks: no private gain once public training data are plentiful.

## Validation-gated step (grid 0, .25, .5, 1, 2 x frozen eta, chosen per release on public validation CE; post-processing of the released aggregate)

Primary gate (fresh dataset KMNIST, budget 32, validation 512 images, q.65 and q.80): **PASSED on both tasks.** 6/6 cells > .001 each, no cell below +.0071 (gate floor −.003), gated mean >= fixed mean in all four task/risk pairs:

| Task | q.65 gated / fixed | q.80 gated / fixed |
|---|---|---|
| KMNIST 0-3 | +.0149 / +.0137 | +.0174 / +.0160 |
| KMNIST 4-7 | +.0117 / +.0110 | +.0144 / +.0136 |

With only a 128-image validation set the gain persists on KMNIST at 32 (6/6 cells > .001; mean +.0063 to +.0157). q.55 gated gains are +.0046 to +.0063 (6/6), small but positive.

Seen-pool replicate (labelled as such): the gate removes the fixed step's losses. At budget 32 all 24 task x risk x cell values are positive (min +.0012; Fashion-MNIST 4-7 q.65 gated +.0049, 6/6, versus fixed +.0022 with a −.0029 cell). MNIST 0-3 q.65 gated +.0086 (6/6).

Larger public budgets: at 128 gated gains are +.0024-.0066 (q.65/.80), 4-6/6 cells; at 512 gated gains are zero within noise (−.0011 to +.0009) and the gate caps the damage (fixed step reached −.0078 in one cell, gated −.0031 at worst).

## Interpretation

A single pre-frozen client-query construction, with a post-processing step-size choice on a modest public validation set, gives a reliable utility gain over the strongest public-only model at a calibrated, verified attack target (.65/.80) when public TRAINING data are scarce (32 examples, about +.012 to +.017 CE, 0.3 to 0.9 accuracy points on KMNIST), shrinking by 128 and vanishing by 512. This is a limited-public result: the signal is useful because the public base is poor, and the gate protects against hurting a good base.

## Limits

The validation set is public labelled data beyond the training examples (512 images; 128 also tested); "limited-public" means limited public training data with a modest validation set. Gaussian only; the client-specific element is per-client normalization, not a distinct noise law. Linear model on pooled 4x4 features. Single frozen configuration tuned on Fashion-MNIST 0-3 only. Attack calibration unchanged and not re-run here. Cells share cohorts and evaluation images; offline tuning, base selection and artifact release unaccounted. The gated selection uses 512 draws of the released aggregate in the evaluation but a deployment would release one aggregate and choose once: the per-draw mean is the right expectation but its variance across single releases is larger (not reported).
