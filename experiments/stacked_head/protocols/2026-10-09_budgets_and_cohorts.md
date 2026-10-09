# Larger public budgets on the new datasets, and cohort-to-cohort variation

Owner (2026-10-09): next steps one by one, autonomous, report at the end. Rules fixed before the cells are run. Frozen construction and V0 gate as in the earlier protocols; train-split evaluation images; 256 independent releases per cell; cohort A unless stated.

## Step B: public-budget map

Tasks: KMNIST 0-3 and 4-7, MNIST 0-3 and 4-7, Fashion-MNIST 4-7 (5 tasks) x budgets 32, 128, 512 x public sets 0-9 x risks .65 and .80 (budget-32 cells are the existing sweep/transfer cells; 128- and 512-image cells are new: 200 cells). Budget-512 sets come from an appended role region, so no earlier role moves.

Per task, risk and budget: p_win (gain > .001), p_loss (gain < -.003), median gain. Per budget, pooled over the ten task/risk groups: a budget is **"reliable"** iff the earlier reliability rule (p_win >= .80, p_loss <= .10, median > 0) holds in at least 8 of the 10 groups. Reported as a map; the smallest budget at which pooled p_win < .50 is the **fade budget** (reported if it exists in {128, 512}). No retuning at any budget.

## Step C: cohorts

Tasks: KMNIST 0-3 and 4-7, budget 32, public sets 0-9, risks .65 and .80, cohorts A (existing sweep cells), B, C, D (new, 120 cells). For each task and risk, balanced two-way ANOVA (10 sets x 4 cohorts, one observation per cell, interaction absorbed in the residual): sigma_set^2 = (MS_set - MS_resid)/4, sigma_cohort^2 = (MS_cohort - MS_resid)/10, sigma_resid^2 = MS_resid, negatives truncated to 0. **Claim "cohort variation is small relative to public-set variation" iff sigma_cohort^2 <= 0.10 x sigma_set^2 in all four task/risk groups.** Also report whether the reliability rule holds over all 40 (set x cohort) cells per group.

## Limits

Ten sets per group in step B; four cohorts per set in step C (cohorts share the label-stress design, differ in sampled private records); same dataset/model/Gaussian limits as before.
