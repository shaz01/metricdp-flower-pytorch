# Limited-public protected-query decomposition protocol

Owner: "lets start the research continue from where we left", following the headroom findings' proposed next step. Development-only diagnostic. No reserve features or remaining reserve indices are opened. No new systematic search. This is not overall experiment completion, a final defense, or a new-density claim.

## Question

At the limited-public setting (32 public examples) where the headroom phase found unprotected private signal, how much survives (1) low-dimensional public projection, (2) per-client public clipping, and (3) peer-contract Gaussian noise at a fixed conditional-risk target? Is the surviving gain attributable to the private signal, or to the public offset that every arm shares? The 512 setting stays as the negative anchor and 128 as the intermediate setting. Smaller budgets are distinct resource settings, never a rescue of the 512 benchmark.

## Inputs (frozen, read-only)

`headroom_development.{json,npz}`: nested public budgets 32/128/512 x subset seeds 42/43/44, record-disjoint private cohorts A/B (8 label-stress clients each), per-cohort unclipped noiseless client queries (`class_center`, `target_balanced`), public reference b, class gradients gamma, public Hessian, selection/assessment development halves (512 each, both reused/exploratory), and the SELECTION-frozen strongest public control per budget/subset.

## Stages (all one-step: theta = b - eta*(offset + decode(total)))

Per budget, subset seed, cohort and query mode (class_center, target_balanced), for public Euclidean projection dimension d in (1,3,12,51), eta in (.1,.3,1,3,10,30):

1. `public_zero`: offset only, no private signal (matched public control for this decoder form).
2. `unclipped`: projected query, no clip, no noise.
3. `clipped`: per-client public clip cap in (.003,.01,.03,.1,.3), no noise.
4. `noisy`: clipped plus peer-contract Gaussian noise (sigma = `public_scale(cap, risk)`, shares summed over 8 slots divided by sqrt(7), the documented 8/7 convention) at conditional-risk targets .55/.65/.80.

Caps are public constants; noise scale is derived only from cap and risk target, never from private norms.

Selection: each stage/risk picks its configuration by mean SELECTION CE (32 common-random-number draws for noisy stages). The selected configuration is then rescored on assessment, with 128 FRESH independent draws for noisy stages. Assessment is never used for selection. Save every candidate's selection score before assessment values are summarized.

## Gates (descriptive, not confirmatory)

For each budget/subset/risk, in BOTH cohorts: noisy selected assessment CE improves by more than .001 over (a) the headroom strongest selection-frozen public control and (b) the selected `public_zero` arm. Report the loss decomposition (unclipped -> clipped -> noisy gain) separately, to attribute losses to projection, clipping bias and noise. A noisy gain over (a) but not (b) is public adaptation, not private signal (cf. the class-conditional attribution). 512 duplicates (3 subset seeds, 2 unique cells) must not inflate counts.

## Limits

Raw offline development evidence: same source population, reused selection/assessment halves, historical 512 anchor tuned on both halves, offline tuning unaccounted. No CIA evaluation, no privacy certificate for tuning, no independent-population, novelty, density or deployability claim. A positive result only licenses an independent-cohort confirmation under the peer-conditioned contribution/dummy CIA contract; a negative result is reported as such.

Verification: unit tests for clipping bound, noise scale/peer convention and stage monotonic identities; independent reconstruction of selected rows; default suite; STATUS update and commit/push.
