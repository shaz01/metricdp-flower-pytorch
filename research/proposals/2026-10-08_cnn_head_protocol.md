# CNN-head port of the stacked, validation-gated constructor

Owner: "go straight with this plan" (step 3). Scope decision made here: a minimal faithful port in plain PyTorch/NumPy, not yet a multi-round Flower strategy (that would change the experimental object: rounds, server aggregation state, the repo's CIA pipeline). Question: does the finding survive replacing the fixed pooled-pixel linear model with a small CNN whose penultimate features are learned on the scarce public data?

## Construction

Base: a small CNN (Conv(1,8,3)-ReLU-pool-Conv(8,16,3)-ReLU-pool-Linear(400,32)-ReLU features, Linear(32,4) head) trained on the public set (full batch, Adam, weight decay 1e-3, candidates epochs (30,100,300) x lr (1e-3,3e-3), choose by public validation CE). Features phi (32) plus a bias column are frozen after base selection; the private step acts on the 4-class head in the same identifiable contrast parameterisation as before (3 x 33 = 99 parameters). Client queries (`target_center`), projection onto the top d=51 public-Hessian directions, per-client clip to cap, summed, calibrated peer-contract Gaussian noise: mode, d, cap and risk targets are the FROZEN stacked configuration (`stacked_constructor_freeze.json`), unchanged. Because the feature scale of a CNN differs from pooled pixels, the step multiplier grid is widened a priori to (0, 1/30, 1/10, 1/3, 1, 3, 10, 30) x the frozen eta and chosen per release on the public validation set (post-processing, no private-data cost). The widening is the only change and is fixed here; no tuning on CNN results.

## Data and gates

Fresh dataset KMNIST classes 0-3 and 4-7, public budgets 32 and 128 (3 public sets each), 2 private cohorts of 512/class (8 label-stress clients), 128/class validation, 512/class held-out evaluation, new role seed, 512 noise draws. Primary gate (budget 32, q.65 and q.80, validation 512), per task: gain in held-out CE over the strongest public-only CNN > .001 in at least 5/6 cells, mean > 0, no cell below −.003. Budget 128 and q.55 are reported descriptively. Also report accuracy deltas, the base CNN's control CE, and the chosen-multiplier distribution (a rough proxy for how well the frozen scale transfers).

## Limits

Head-only private step on a feature extractor trained from the small public set only (no private data touches the feature extractor); not a full Flower federation; one-shot release; Gaussian only; 4-class tasks; calibration of the attack depends only on cap and risk target and is not repeated; offline tuning, base selection and artifact release unaccounted; validation set is extra public labelled data. Failure or partial success is reported as is.
