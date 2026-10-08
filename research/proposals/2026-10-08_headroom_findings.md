# Private-signal headroom: development diagnostic and research direction

2026-10-08, `feature/client-specific-noise`. Owner: “move to that”. This is a bounded diagnostic within active research, not overall completion or a final root report.

**Concrete finding:** private-data usefulness depends strongly on public-reference resources. With 32 public examples, selection-frozen unprotected retraining and one-step corrections improve reused assessment CE in both disjoint private cohorts for all 3 public subsets. At 512, the selection-frozen oracle and one-step gates fail. Some private-refinement families help cohortA but barely helpB, so this does not prove private information is absent. It establishes where to investigate useful correction and why smaller residual norms alone were misleading.

## Protocol and evidence

[Frozen protocol](2026-10-08_headroom_protocol.md); calculator `research/calculations/headroom_probe.py`; raw `results/client_specific_noise/headroom_development.json`/`.npz`; independent `headroom_artifact_audit.json` from `audit_headroom.py`. No additional literature query or change to 31 frozen families.

Public budgets 32/128/512 are balanced NESTED subsets of the same public 512, public ordering seeds 42/43/44. All subsets are disjoint from private/development records. Smaller budgets are separately declared resource settings;512 remains anchor.512 seed variants use identical record sets and are ordering duplicates, not three replications.

Private cohortsA/B each 2048 records (8×256, dominantclass 205/otherclasses 17), original seed 42 fit versus check halves, disjoint and together cover the 4096 private bank. They share the source population, not independent population draws. Original IDs/client allocations saved. Previously used development 1024 is split into 512 selection/512 assessment (128/class each); both halves are reused, not fresh confirmation. No reserve features or remaining reserve indices opened;136 unused images preserved.

2775 saved models:9 public rows with 52/53 candidates,18 private subset/cohort cells with 128 candidates. Scratch steps 80/320/1280/5120 with six logit multipliers; matching public/pooled/private refinement controls; one-step raw, class_center, center_only and exact target-balanced directions, eta/dimension grids. Families and overall oracle chosen ONLY on selection CE. Assessment does not replace a losing selection winner. All gradient queries here are UNCLIPPED/NOISELESS and all fitted models unprotected.

## Measured assessment headroom

Positive gain means lower CE than strongest selection-frozen public control. Oracle is selection-frozen best retraining family; one-step is selection-frozen best one-step family.512 orderingduplicates omitted below; all 18 cells retained in artifacts.

| Public budget | Subset seed | Cohort | Public CE | Oracle CE | Oracle gain | One-step gain |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 42 | A | 0.709514472 | 0.498936868 | +0.210577604 | +0.017625435 |
| 32 | 42 | B | 0.709514472 | 0.503594538 | +0.205919934 | +0.015530585 |
| 32 | 43 | A | 0.626637261 | 0.498057877 | +0.128579383 | +0.017062007 |
| 32 | 43 | B | 0.626637261 | 0.503871031 | +0.122766229 | +0.021054894 |
| 32 | 44 | A | 0.633964675 | 0.499618372 | +0.134346303 | +0.036549522 |
| 32 | 44 | B | 0.633964675 | 0.505186350 | +0.128778325 | +0.017119153 |
| 128 | 42 | A | 0.548600232 | 0.500610610 | +0.047989622 | +0.006191569 |
| 128 | 42 | B | 0.548600232 | 0.504833237 | +0.043766996 | +0.006234637 |
| 128 | 43 | A | 0.553366305 | 0.499628960 | +0.053737345 | -0.003975991 |
| 128 | 43 | B | 0.553366305 | 0.504938316 | +0.048427989 | +0.005187680 |
| 128 | 44 | A | 0.549167543 | 0.500251493 | +0.048916050 | +0.007919603 |
| 128 | 44 | B | 0.549167543 | 0.504810939 | +0.044356604 | +0.002946400 |
| 512 | 42 | A | 0.489472544 | 0.497741696 | -0.008269153 | -0.001480072 |
| 512 | 42 | B | 0.489472544 | 0.502507452 | -0.013034909 | -0.001899829 |

32 examples: oracle gains.122766–.210578 CE; one-step gains.015531–.036550 CE, all 6 cells exceed the descriptive.001 gate.128 examples: oracle gains.043767–.053737 all 6, but one-step fails subset 43/cohortA(−.003976).512 examples: selectedoracle gains−.008269/−.013035 and one-step−.001480/−.001900. The stronger 512 public control is public fine-tuning 1280 steps after its selected 5120-step scratch reference; its assessmentCE.489472544. This current assessment slice differs from previous fresh 320 images, so compare controls WITHINthis diagnostic, not cross-phase CE levels.

All selected scratch oracles use 5120 steps×1.25 logit multiplier. They are bounded grids, not mathematical/global utility upper bounds. Selection-frozen family results explain why the 512 failure is not a no-private-signal theorem:

| Cohort at 512 | pooled_scratch gain | pooled_refine gain | private_refine gain |
|---|---:|---:|---:|
| A | -0.008269153 | +0.004603694 | +0.004872994 |
| B | -0.013034909 | +0.000554025 | +0.000103053 |

Private refinement is useful inA(+.004873) butB(+.000103) does not meet.001; pooled refinement behaves similarly. Assessment cannot be used to switch to that family and declare a successful gate. This suggests selection/generalization and cohort variation alongside signal size, rather than only noise degradation.

## What the residual statistics mean

At publicB, each example residual z=g_example−gamma_label. Its client mean is r_i. The saved stratified sampling-fluctuation estimate is

    V_i = sum_k (n_ik/n_i)^2 trace(sample covariance(z | k)) / n_ik.

Conditional on fixed publicB/gamma and fixed class counts, subtracting gamma_k leaves within-class covariance EXACTLY unchanged. Thus lower residual norm is not lower within-class estimation fluctuation. The IID-mixture variance can differ when labels fluctuate; it is not the only relevant model in this fixed-count experiment. Public-reference uncertainty is excluded, and finite-population dependence is not accounted in these IID-within-class proxies. They are not certified uncertainty, true signal-to-noise or privacy quantities.

Aggregate mean variance proxy=sum_i V_i/64. The following ratio uses OBSERVED squared residual norm, not latent useful signal. Large magnitude does not prove correct direction or generalization.

| Budget/subset/cohort | Aggregate residual squared norm | Mean variance proxy | Observed ratio | Fraction in top 3/top 12 public Hessian directions |
|---|---:|---:|---:|---|
| 32/42/A | 0.010305192 | 0.000194143 | 53.080 | 69.79%/94.94% |
| 32/42/B | 0.015645337 | 0.000209575 | 74.653 | 80.98%/96.42% |
| 32/43/A | 0.004504720 | 0.000244570 | 18.419 | 46.89%/91.74% |
| 32/43/B | 0.002954513 | 0.000239276 | 12.348 | 33.58%/87.36% |
| 32/44/A | 0.006503996 | 0.000255636 | 25.442 | 58.22%/97.19% |
| 32/44/B | 0.008591312 | 0.000252874 | 33.975 | 70.02%/97.87% |
| 128/42/A | 0.002058382 | 0.000193775 | 10.623 | 74.98%/97.66% |
| 128/42/B | 0.003659814 | 0.000209494 | 17.470 | 79.06%/97.69% |
| 128/43/A | 0.001996718 | 0.000233040 | 8.568 | 65.61%/93.29% |
| 128/43/B | 0.000570281 | 0.000241417 | 2.362 | 26.04%/77.03% |
| 128/44/A | 0.001416386 | 0.000216651 | 6.538 | 71.37%/95.65% |
| 128/44/B | 0.001880796 | 0.000230640 | 8.155 | 78.82%/94.46% |
| 512/42/A | 0.000372361 | 0.000197361 | 1.887 | 85.91%/95.43% |
| 512/42/B | 0.000102257 | 0.000203698 | 0.502 | 26.62%/78.92% |

At 512 the observed norm/variance ratios 1.887(A)/.502(B) are much smaller than 32's 12.348–74.653. The 32 reference leaves a larger private correction; public top 12 directions retain 87.36–97.87%of its squared norm, versus top 3 retaining 33.58–80.98%. These are query geometry diagnostics, not guarantees of utility after projection/noise. The all-IN unbounded raw/class_center-with-restoration models are equal because mean class proportions areuniform; centering has not created additional aggregate learning information.

## Verification and limits

Three meaningful tests verify nestedbalancedsubsets/disjointsplits, gradient descent/mean variance and fixed-class covariance invariance. Full default suite 302 passed/5 deselected. Independent auditor reconstructs every fitted model, query, selection/assessment score and residual statistic with split/hash checks and no reserve extraction; final independent receipt passes 224,895 numeric checks/max 3.20e-14, all 2775 models independently retrained,9 public/18 privatecells/144 clientvariance records and 18 projected decompositions. Hashes, original IDs, cohort separation and model selections match. The historical 512 anchor was tuned on both olddevelopmenthalves in a prior phase; it is a disclosed reused-information control, not clean split validation. The first diagnostic omitted an explicit projected-mean-fraction field; added it and regenerated the SAME models/choices. A metadata-patch indentation error failed beforeexecution and was fixed; final artifacts supply evidence, not a new replication.

This is raw-information, reused-development evidence. No privacy calibration, CIA, noisy/protected utility, client-specific distribution novelty, historical metric-privacy superiority or deployable tuning claim follows. None of the 18 cells is fresh population confirmation.512 duplicates and sharedassessmentimages must not inflate sample count.

## Research direction and next bounded question

Keep the 512 negative anchor. The 32 public-resource regime is the clearest **developmental opportunity** for learning a useful private correction, with consistent one-step as well as retraining gains in both disjoint cohorts. It must be declared as a different setting, not a weaker comparator substituted to rescue 512.128 is intermediate and less stable for one-step transfer.

Next proposed mechanism comparison: freeze a limited-public 32 setting alongside 512 anchor, retain strong public-only/raw/balanced/central controls, and separate projection loss, clipping bias and noise cost for the useful residual. Client-local within-class uncertainty may inform a deterministic residual shrinkage rule BEFORE joint public clipping; such a rule would need bounded-query/empty-client verification and closest-work review. Do not derive public noise variance from unprotected private variance or norms: that reintroduces distribution/scale leakage. A change to the noise density should follow a demonstrated protected query benefit.

Before any further reserve confirmation, require a fixed constructor and stable development gains under the same peer-conditioned contribution/dummy CIA contract and genuinely new cohort evidence. For 512, inspect refinement/selection stability and richer task representations rather than assuming no private information exists. The next constructor/diagnostic is proposed, not run, selected as the final defense, or certified.
