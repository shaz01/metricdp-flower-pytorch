# Frozen limited-public constructor: pooled freeze and fresh-data confirmation findings

Protocol: [frozen-constructor protocol](2026-10-08_frozen_constructor_protocol.md). Code `research/calculations/frozen_constructor_probe.py`; artifacts `results/client_specific_noise/frozen_constructor_{freeze,confirmation}.{json,npz}`; independent reconstruction `audit_frozen_constructor.py` (113 checks, max noisy-CE error 1.7e-14: role disjointness/class counts, queries, clipping, noise, CE draws, gains, expected AUCs).

## Stage 1: pooled freeze (extended eta grid)

One configuration per risk, chosen by pooled mean SELECTION CE over the six 32-example development cells: all use `target_balanced`, d=51, cap .003 (the smallest cap on the grid), eta 30 (q.55) / 100 (q.65, q.80). None is at the new eta maximum (1000), so the earlier grid-edge caveat is resolved. Pooled selection CE: public-zero .6057 (class_center, d=1, eta 3), q.55 .6042, q.65 .5957, q.80 .5938. Because noise and signal both scale with the cap, eta*cap is what matters; the smallest cap means nearly every client update is clipped, so the construction is a noisy average of normalized client directions. Descriptive reused-assessment check: frozen q.65/q.80 beat frozen public-zero in all six cells (e.g. .6963/.6941 vs .7100).

## Stage 2: fresh-data confirmation (Fashion-MNIST test split, no re-tuning)

Disjoint roles (asserted): 3 public 32-sets, one 2,048-image label-stress cohort of 8 clients, 1,856 held-out evaluation images. 512 fresh noise draws per risk.

| Subset | Strong public control (family) CE | Frozen public-zero CE | q.55 gain vs control / zero | q.65 gain vs control / zero | q.80 gain vs control / zero |
|---|---:|---:|---:|---:|---:|
| 0 | .6136 (public_step) | .6174 | +.0016 / +.0054 | +.0165 / +.0203 | +.0184 / +.0223 |
| 1 | .7005 (public_step) | .7024 | −.0043 / −.0024 | +.0137 / +.0156 | +.0159 / +.0178 |
| 2 | .5557 (public_step) | .5903 | −.0280 / +.0065 | −.0082 / +.0264 | −.0064 / +.0282 |

- **Gate result (q.65 and q.80, both controls, all three fresh subsets): PARTIAL. 2 of 3 pass; subset 2 fails against the strong public control.** q.55 fails (no consistent gain).
- Subset 2: the construction still beats the matched zero-private offset by .026-.028 CE, but the per-subset strong public control (a tuned larger public gradient step) is .035 better than the frozen public-zero arm there. The frozen d=51 target_balanced construction carries almost no public gradient offset (the offset is the part of the public gradient outside the projection), so the private step REPLACES the public step instead of adding to it; where the public step alone is strong, the construction loses. This is a design limitation exposed by confirmation, not noise.
- The "unclipped" noiseless arm (frozen eta applied without clipping) diverges (CE 1.4-13.8): the tuned step is only valid in the clipped regime. It is not a meaningful decomposition arm here and is excluded from claims.
- Attack check (known-alternative likelihood-ratio statistic, 8 target slots, 2,048 draws/world, peer-conditioned): mean AUC .552/.653/.802, .555/.650/.799, .549/.651/.802 against calibrated .55/.65/.80; largest single-target deviation +.0224 (z 2.68 of 72 comparisons, consistent with sampling variation). This verifies the calibrated construction, not a general CIA defense.
- Accuracy (q.80): .804/.762/.816 on the three subsets.

## Limits

One fresh cohort (same source population), three public subsets sharing it and the evaluation images; test split reused from earlier fixed-feature phases (not for choosing this constructor); offline tuning and public-reference choice unaccounted; Gaussian only; no CNN/Flower or independent-population transfer. Not a new density or metric-privacy superiority claim, and the pre-stated "solution" criterion is NOT met (2/3).

## Next (development-only until fresh data exists)

Because the test split is now consumed for this constructor, any redesign is adaptive. Proposed: a stacked construction that keeps the strongest public model as base and adds the clipped noisy private step (so public and private information add), developed on the six dev cells with a pooled freeze, then checked on a different dataset/population as transfer evidence.
