# Validation-gated step size for the stacked constructor

Owner: "go straight with this plan". The budget map exposed a failure mode of the frozen stacked constructor: with a fixed step length (eta*cap = .3 for q.65/.80), the private step overshoots when the public base is already strong (Fashion-MNIST 4-7, 32 examples, public set whose base CE was .570: gains −.016/−.022 at q.65; pooled over six public sets per task the gain is +.0085 for Fashion-MNIST 4-7 with 4/12 cells at or below zero, +.0166 for MNIST 0-3 with 12/12 positive).

## Variant

Same frozen stacked configuration (mode, d, cap from `stacked_constructor_freeze.json`), noisy aggregate released once per draw. A step-size multiplier m in the a-priori grid (0, .25, .5, 1, 2) times the frozen eta is chosen per release by the CE of theta_m = T0 - m*eta*decode(total) on a PUBLIC validation set; m=0 is the public base. The choice uses only the released noisy aggregate and public data, i.e. post-processing: no extra private-data privacy cost under the stated contract. Nothing is tuned: the grid is fixed here before data are opened. Validation sizes: 512 images (128/class) and a 128-image subset (32/class).

IMPORTANT limit: the validation set is public labelled data beyond the 32/128/512 training examples. The base-model selection in all earlier phases already used a 512-image public selection set; this phase makes that explicit and tests a 128-image validation set as a more modest case. "Limited-public" therefore means limited public TRAINING data with a modest public validation set.

## Data

Fresh dataset never used by any choice: KMNIST (HF `tanganke/kmnist`) train, classes 0-3 and 4-7 as two tasks. Per class and task, disjoint roles: three public sets per budget in (32,128,512), a 128 validation/selection set, private cohorts A/B of 512 (label-stress, 8 clients), 512 held-out evaluation. 6 cells per task and budget (3 public sets x 2 cohorts), 512 noise draws per cell and risk. Secondary replicate (seen pools, labelled as such): Fashion-MNIST classes 4-7 and MNIST digits 0-3 with a new role seed.

## Gates (primary: KMNIST, budget 32, q.65 and q.80, validation 512)

For each task: gated mean gain over the strongest selection-frozen public control > 0 and > .001 in at least 5/6 cells, no cell worse than −.003, and gated mean gain >= fixed-step (m=1) mean gain. Also report 128-image validation, budgets 128/512, q.55, accuracy delta, and the frequency of each chosen m. Passing both tasks is reported as "gated variant transfers to a fresh dataset at 32"; any other outcome is reported as is. Gaussian only; linear model on pooled 4x4 features; attack calibration depends only on cap and risk target and is unchanged (not repeated); offline tuning, base selection and artifact release unaccounted.
