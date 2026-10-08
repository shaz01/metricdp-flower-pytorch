# Stacked public-plus-private constructor: development freeze and fresh-task transfer protocol

Owner: "lets move with the research autonomously until you find a solution". Follows the frozen-constructor findings, where the unstacked construction (private step applied to the scratch reference b) passed 2 of 3 fresh public subsets and lost where the strong public control's tuned gradient step was strong, because the construction replaced rather than added to the public step. The Fashion-MNIST test split is now consumed for that constructor, so this phase uses data that has never been touched.

## Construction (stacked)

Base model T0 = the strongest selection-frozen public model (same 24 scratch + 4 refine + 24 public-step candidate family as before, selected on a SELECTION set that is disjoint from every evaluation role). Client queries are the gradients at T0 (`target_balanced`, or `target_center` = balanced minus the public class-gradient mean at T0), projected on the top-d public Hessian directions at T0, clipped per client to a public cap, summed, plus peer-contract Gaussian noise at risk target q (same `public_scale` and 8/7 convention), decoded, and applied as theta = T0 - eta*decode(total). There is no extra public offset: the public information is already in T0. The tuned grid is modes (2) x d (1,3,12,51) x cap (.003,.01,.03,.1,.3) x eta (.1,.3,1,3,10,30,100,300,1000) x risk (.55,.65,.80).

## Stage 1: pooled freeze (development)

Fashion-MNIST classes 0-3, the six budget-32 development cells (subset seeds 42-44 x cohorts A/B) with their saved strongest public control models, queries recomputed at T0, selection-half CE only. Freeze ONE configuration per risk by pooled mean selection CE. No assessment, test or fresh-task data is used. If a frozen eta is the grid maximum, say so; no further extension without an addendum.

## Stage 2: fresh-task transfer (no re-tuning)

Two tasks never used by any frozen choice:

1. Fashion-MNIST TRAIN classes 4-7 (relabelled 0-3): same domain, different classes, large untouched pool.
2. MNIST TRAIN digits 0-3: different dataset.

Per task and per class, disjoint roles (asserted): 3 public sets of 8 (so 32 per set), a 128 selection set (candidate/base selection only), two private cohorts A/B of 512 (label-stress, 8 clients of 205 dominant + 17 off-class each, as in development), 512 held-out evaluation. Each of 3 public sets x 2 cohorts is a cell (6 per task; public sets share cohorts and evaluation images). 512 fresh noise draws; attack check with 2,048 draws per world per target slot.

Arms per cell: strongest selection-frozen public control; the previous unstacked frozen construction (frozen_constructor_freeze.json, at the cell's scratch reference); stacked noiseless clipped reference; stacked noisy at q.55/.65/.80.

## Gates (descriptive; stated before fresh-task data is opened)

Per task and risk q in (.65,.80): the stacked noisy gain over BOTH the strongest public control and the unstacked frozen construction exceeds .001 CE in at least 5 of 6 cells and is positive on average, with no cell worse than −.005. Both tasks must pass for the claim "transfers to new classes and a new dataset". Passing one task is reported as partial. q.55 is reported but not gated (near chance).

## Limits

Gaussian only; offline tuning, base selection and artifact release unaccounted; tiny linear model on pooled 4x4 features, not a CNN/Flower federation; cells share cohorts/evaluation images; hyperparameters were tuned on Fashion-MNIST 0-3 only, which the transfer test deliberately does not retune. Passing supports a useful bounded-risk client-query construction at the limited-public setting under the calibrated conditional-risk contract; it is not a new density, metric-privacy superiority, deployed privacy guarantee or independent-federation result. Failure is reported as failure.
