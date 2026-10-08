# Limited-public protected-query decomposition: findings

Protocol: [limited-public protocol](2026-10-08_limited_public_protocol.md). Development-only, reused selection/assessment halves (512 images each, shared with the headroom phase). No reserve access (136 unused). Artifact: `results/client_specific_noise/limited_public_development.json` (all 18 cells, every candidate's selection CE, selected configurations, assessment scores). Code: `research/calculations/limited_public_probe.py`; independent recomputation `audit_limited_public.py` (1,278 checks, max error 2.2e-16: selection minimality over all saved candidates, selected-row selection/assessment CE via a separate CE/clip/noise implementation, 25 random candidates per cell, gain arithmetic).

## Result

Mean assessment-CE gain over the matched zero-private public offset (`public_zero`), by public budget (6 cells each; 512 has 2 unique cells):

| Budget | unclipped | clipped | noisy q.80 | noisy q.65 | noisy q.55 |
|---|---:|---:|---:|---:|---:|
| 32 | .0161 | .0194 | .0159 | .0089 | .0008 |
| 128 | .0046 | .0049 | .0037 | .0021 | .0004 |
| 512 | .0001 | .0001 | -.0001 | -.0001 | -.0001 |

Descriptive gate (gain > .001 over BOTH the headroom strongest selection-frozen public control and `public_zero`, in both cohorts):

- **32**: passes at q.65 and q.80 for all three public subsets (min gain over `public_zero` at q.80 is .0101; over the strongest control the per-cell range at q.65 is .0070-.0135). Fails at q.55 (gains .0007-.0042 vs the control; one cell negative).
- **128**: passes at q.65/q.80 for subset seeds 42 and 44; fails for seed 43, where the strongest public control (.5534) beats `public_zero` (.5621) by more than the private gain, i.e. the private step helps over the zero offset but not over the stronger public model.
- **512**: no pass; selected noisy configurations are .0017-.0020 CE WORSE than the strongest public control in every cell (the control is ahead before any private signal is used). Anchor stays negative.

## Attribution

- **Projection**: no loss at 32/128 from the public Euclidean basis; selections mostly use all 51 dimensions or 12.
- **Clipping**: not a loss; mean clipped gain exceeds unclipped (.0194 vs .0161 at 32). Clipping/eta act together as shrinkage, so this is a selection-tuned effect, not a mechanism finding.
- **Noise**: the dominant cost. At 32 the q.80 noisy gain retains about 98% of the unclipped gain, q.65 about 55%, q.55 about 5%. Near chance (q.55) nothing useful survives at this cap/eta grid, consistent with the matched-CIA phase where utility vanished near chance.

## Limits and cautions

- **Grid-edge selections**: eta=30, the largest grid value, is selected in 13/18 noisy q.80, 12/18 q.65 and 9/18 clipped cases. The grid truncates the optimum, so q.65/q.80 gains are conservative as tuned values but the selected step is a large multiple of the raw public step; extending the grid would be a protocol change and was not done. Treat the stage ordering as reliable and magnitudes as grid-dependent.
- Risk targets are conditional peer-contract targets from the matched-CIA calibration, not a CIA evaluation here. CIA at these exact selected configurations has not been run, and the earlier descriptor audit showed utility gains need not be matched-leakage wins.
- 32 is a different resource setting from the 512 anchor, not a rescue of it. Private cohorts share one source population; selection/assessment halves are reused; offline tuning, public-reference choice and artifact release are unaccounted. No new noise density, end-to-end private tuning, independent-population or deployable-privacy claim. Gaussian is the only noise law tested here.

## Next (proposed, not run)

Run the frozen 32 constructor (selected mode/dimension/cap/eta per risk, from this JSON) under the peer-conditioned contribution/dummy CIA contract with matched-AUC Gaussian vs radial controls on genuinely new cohort evidence, and decide separately whether to extend the eta grid. Any non-Gaussian density comparison should wait for that matched-leakage result.
