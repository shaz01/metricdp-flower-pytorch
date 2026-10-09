# Public-set sweep and cross-dataset transfer (stacked-head, real Flower path)

Owner (2026-10-09): "lets move with your suggested steps work autonomously without stopping". Decision rules below are fixed BEFORE any new cell is run. Implementation: `experiments/stacked_head/` at the commit that adds this file; frozen construction: `results/client_specific_noise/stacked_constructor_freeze.json` (tuned on Fashion-MNIST classes 0-3 pooled-pixel development cells; never retuned).

## Why

All gains so far depend on which 32 public images the base is trained on. The first transfer result looked like 24/24 positive and turned out fragile when new public sets were drawn; the gate matrix has only three public sets per task and one failing cell is a single public set. We need the distribution over public sets, not three draws.

## Step 1: public-set sweep (KMNIST)

Tasks `kmnist_classes0to3`, `kmnist_classes4to7`; 32 public images; public sets 0..29 (sets 0-2 are the earlier ones, re-run for uniform artifacts; sets 3..29 are new and drawn from the same per-class shuffles AFTER the existing roles, so all earlier role indices are unchanged); cohort A only; risks .65 and .80; 256 independent releases per cell; in-process backend (validated equal to the Ray simulation to ~1e-8). 120 cells. Each release also records, for all 8 step multipliers, the validation CE on the 512-image public validation set and on a 128-image subset (first 32 per class), and the held-out CE/accuracy, so alternative gates can be evaluated offline on identical releases.

Per task and risk, over the 30 public sets, with per-set gain = control CE minus mean held-out CE of the gated release:
- fraction of sets with gain > .001 (p_win) and fraction with gain < -.003 (p_loss), median gain, 10th percentile, mean accuracy delta;
- Wilson 95% intervals for p_win and p_loss;
- Spearman correlation of gain with the control CE (base weakness);
- the same quantities for (a) the 128-image-validation gate and (b) no gating (fixed step multiplier 1);
- failure anatomy for sets with gain < -.003: whether some multiplier would have beaten the control on held-out data (gate error) or none would (no usable step).

**Decision rule (stated now): "reliable at 32 public images" for a task and risk iff p_win >= 0.80 AND p_loss <= 0.10 AND median gain > 0.** The 128-image gate is "adequate" if it also satisfies the rule wherever the 512-image gate does. No retuning of cap, eta, dimension, multiplier grid or validation rule follows from these results; any change needs a new protocol and new data.

## Step 2: cross-dataset transfer

Tasks `mnist_classes0to3`, `mnist_classes4to7`, `fmnist_classes4to7` (Fashion-MNIST classes 0-3 are excluded: the frozen construction was tuned on that class set's development pools). 32 public images, public sets 0..9, cohort A, risks .65 and .80, 256 releases, same recorded quantities and the same decision rule applied per task and risk (10 sets, so p_win >= 8/10, p_loss <= 1/10). 60 cells. MNIST/Fashion-MNIST pools were used in earlier linear-model diagnostics but never to tune this construction.

A dataset "transfers" iff the rule holds for at least 3 of its (task, risk) pairs (MNIST has 4 pairs, Fashion-MNIST 2: both must hold). Failure of the rule is reported as such.

## Limits

Single cohort per public set (cohort-to-cohort variation was small relative to public-set variation in the gate matrix); 4-class tasks; small CNN; Gaussian only; the validation set (512 public labelled images, 128 in the alternative gate) is extra public data beyond the 32 training images; replicate rounds are independent releases from one base, not a multi-round protocol; no secure aggregation; offline tuning and base selection unaccounted; held-out evaluation uses 2,048 images per task.
