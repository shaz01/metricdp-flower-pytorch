# Independent mathematics: development-frozen public slot profiles

2026-10-05. Design review of the follow-up to the three narrow broader-oracle positive cells. This is a raw-checkpoint utility diagnostic, not a privacy-accounted training mechanism, CIA result or novelty claim. Actual implementation review follows once the calculator exists.

## Frozen rule and four comparison arms

Before looking at the new official-test slice, freeze the bias-contrast profile roster `[0,2,0,1,0,2,0,1]`, common radius $C=0.3$ and postprocessing $t=1$. The model checkpoint, gradients, weights and three-coordinate Helmert projection remain those of the saved audit. Four arms have different information access:

- **Shared tuned:** choose one profile for all slots, radius and shrinkage using the saved global development score.
- **Oracle tuned:** choose all eight profiles, common radius and shrinkage using the same raw development information and exact finite assignment enumeration.
- **Frozen configuration:** use the already fixed roster, radius and shrinkage without any fresh calibration. The raw update is still clipped/noised; freezing means those mechanism parameters are not reselected.
- **Frozen assignment tuned:** keep the roster fixed, but select radius and shrinkage on development data. This is not a fully frozen configuration and remains raw-information calibration.

All arms must share the declared radius/body family and derivative conventions where tuning is allowed. The frozen arm does not use test loss to choose whether its configuration should be applied. No-intervention and raw/noiseless-step logs remain useful context when a relative gain is small.

## Mathematics and narrow fixed-profile guarantee

For public profile radii $r_{ij}$ and clipped contrast update $v_i$, the aggregate mean and covariance are

$$
m=\sum_i\alpha_i v_i,\qquad
V=\frac8{\epsilon^2}\sum_i\alpha_i^2\operatorname{diag}(r_i^2).
$$

An intervention with fixed postprocessing is $t(m+Z)$, so the evaluated mean is $tm$ and covariance is $t^2V$. Development tuning of $t$ must use

$$
\ell t+Dt^2,\qquad
\ell=g^\top m,\quad D=\tfrac12m^\top Hm+\tfrac12\operatorname{tr}(HV).
$$

For positive $D$, $t^*=\operatorname{clip}[-\ell/(2D),0,1]$, with explicit zero-cost endpoints if needed. Apply shrinkage to sampled noise as well as the mean. Direct nonlinear CE is evaluated by reconstructing the contrast increment through the fixed orthonormal basis, not by replacing it with its development quadratic score.

Because each fixed weighted-L1 body clips every raw input into that body, its matching independent Laplace law has a whole-input replacement certificate for the isolated upload under its public conditional configuration. A publicly fixed heterogeneous roster does not incur an extra private profile-selection charge: profiles may differ across slots without depending on those slots' current private input. This narrow fact does not certify the present entire experiment. Checkpoints, source gradients, sample weights and prior calibration came from unprotected data, and the study's epsilon labels are counterfactual fixed-profile parameters rather than end-to-end budgets.

Freezing a configuration after development does not retroactively protect the development computations. The relevant scientific question is whether a fixed rule reproduces the tested benefit without new per-client estimation at assessment time. A deployable public roster would additionally need a defensible public-information source and prospective applicability beyond the same previously studied clients/partition.

## Slot-shift semantics

For the stress-only diagnostic, permute client updates **and their associated weights** by the slot order `[1,2,3,4,5,6,7,0]`. Keep the profile IDs attached to destination public slot positions. The underlying samples, common model checkpoint and global development/evaluation objective remain unchanged. Then

$$
\sum_i\alpha'_i u'_i=\sum_i\alpha_i u_i,
\qquad\sum_i(\alpha'_i)^2=\sum_i\alpha_i^2.
$$

Thus the raw unnoised aggregate is invariant. Every shared-profile mean/covariance is invariant, and an exact oracle assignment family has the same optimal development score after permutation: its assignments can be permuted to follow the data. A fixed heterogeneous roster need not be invariant because it deliberately remains attached to slots rather than clients.

Holding the trained checkpoint fixed makes this an **allocation-order diagnostic**, not a new partition/training trajectory. If the frozen gain changes under this shift, it identifies dependence on the original slot-to-data arrangement. It does not prove that geometry itself vanishes or that all public per-client allocations fail. If data are permuted but weights are not, quantity effects confound this interpretation; always preserve their association, including when the stress example happens to have equal weights.

Common standardized noise across arms is valid for paired contrasts. Under permutation, distributional invariance of the shared/oracle optimum is different from bitwise equality of sampled realizations: reordered weight/profile scales can attach the same random innovations to different local laws. Preserve this distinction instead of treating a Monte Carlo fluctuation as an optimization error. Within a row, if frozen and oracle configurations coincide exactly, their paired draws and losses should be bitwise identical.

## Source separation, endpoints and counts

Use per-class official-test positions 512–767, verify their original indices exclude both earlier slices, and keep them out of all profile/radius/shrink selection. Dataset hashes and saved checkpoint identity should remain checked. The expected family has 18 ordinary rows from three seeds, three regimes and two checkpoints, plus six stress shifted-slot rows: 24 rows and 72 epsilon-label comparisons. Four-arm paired noise sampling uses 2048 draws per comparison.

The registered primary setting is original-order label stress, checkpoint 20, epsilon label 8. State whether the $0.001$ gain gate refers to a point estimate or a noise-only interval, and distinguish replication across seeds from independent client populations. For the frozen-versus-oracle $0.0001$ gap, define the endpoint explicitly: a signed noninferiority gap is $J_{\mathrm{frozen}}-J_{\mathrm{oracle}}\le0.0001$; an absolute matching claim instead requires $|J_{\mathrm{frozen}}-J_{\mathrm{oracle}}|\le0.0001$. Nonlinear held-out ordering need not follow oracle development-score dominance.

Conditional Monte Carlo intervals describe noise uncertainty on the fixed test examples, not test/client sampling uncertainty. All primary configurations and gates must remain frozen when interpreting the fresh slice. Additional regimes, checkpoint/budget cells and slot shifts are diagnostic context; do not select a new winning configuration on those outcomes and call it the already registered rule.

A successful frozen-rule replication would narrow the finding to publicly prescribed profile allocation in this measured setup and weaken the claim that raw per-client estimation is needed. A failed replication or harmful slot shift would narrow its applicability further. Either result is an informative construction-design outcome, not yet an accounted distribution estimator or CIA defense. No code/results/jobs were modified by this design review.

## Actual code and frozen protocol check

Independently reviewed `research/calculations/public_slot_profile_probe.py` and `2026-10-05_public_slot_protocol.md` after they appeared. No material implementation error was found. The protocol explicitly defines the oracle gap as **absolute matching** within $0.0001$, and the calculator implements that endpoint. The protocol is a locally frozen design, not external preregistration.

The frozen configuration retains the fixed roster, radius $0.3$ and shrink one. Its raw development score is computed for descriptive logging but does not alter these parameters. The fixed-assignment tuned arm instead chooses radius/shrink on development information, matching its separate label. Both tuned shared and tuned oracle reuse the same exact finite family and shrink optimizer. Their inclusion assertions are mathematically consistent.

Mean/noise shrinkage, squared-weight covariance, contrast reconstruction and direct CE evaluation are correct. Evaluated test derivatives enter only the logged noise-curvature penalty, not configuration selection. The fresh per-class slice is asserted to have 1024 total examples and exclude both earlier stored index sets. Dataset hashes and checkpoint source identity are retained.

The cyclic shift carries updates and weights together, leaves the head/global objective unchanged, and keeps frozen profile IDs attached to slots. Both numerical self-checks and actual-row assertions test invariant shared/oracle development optima. Original and shifted rows use different innovation seeds, so sampled losses across those rows are not expected to be bitwise equal; the relevant paired arm contrasts are within each row. A configuration identity within a row does use identical innovations and should yield exact trial-loss identity.

The source checkpoints/development information are reused. Consequently, if the three primary frozen/oracle configurations coincide, their exact matching is a known consequence of fixing the previously selected rule, not proof that a new client estimator discovers the oracle. The genuinely fresh component is assessment on the disjoint test slice and new noise draws. The altered-slot diagnostic separately assesses dependence on the prior slot-data alignment.

The implementing agent reports a successful self-check and a running full calculation. Results were not yet assessed at this code review. No calculator, result files or jobs were changed by this independent review.

## Independent completed artifact verification

The saved artifacts contain the expected 24 rows and 72 epsilon-label comparisons, with four arms. Recomputed all 360 saved paired means/standard errors from the NPZ; they match the JSON. No full experiment or new sampling was run by this review.

In the three registered original-roster primary cells, frozen and oracle trial-loss arrays are bitwise identical, with exactly zero matching gap. Fresh-slice frozen CE gains over tuned shared are $0.0009594168$, $0.0009983264$ and $0.0009381095$. **All three fail the predefined $0.001$ point-estimate gain gate.** This is a small beneficial signed effect in the declared setting, not no effect at all. It does not justify redefining the threshold after observing the fresh test data. Seed 43's noise-only interval straddles the gain threshold, while the other two are wholly short of it; those intervals do not change the point-gate result.

The roster-shift primary contrasts reverse sign: frozen-minus-shared losses are $+0.0017478776$, $+0.0017685343$ and $+0.0017826251$. Retuning radius/shrink while retaining the shifted hardcoded roster reduces the harm but does not remove it: corresponding tuned-fixed minus shared losses are $+0.0006131938$, $+0.0006363988$ and $+0.0005589229$.

Across all 18 comparisons in each subgroup, frozen minus shared is positive for balanced, positive for quantity skew, negative for original-order label stress, and positive for shifted label stress. There are no frozen gain-gate passes among all 72 comparisons. This coherent sign pattern supports dependence on the constructed slot-label arrangement; it is not a general fixed-profile advantage for real client populations.

## Decision and bounded next design question

The fixed roster reproduces the oracle's configuration in the reused primary clients/checkpoints and yields a small fresh-slice benefit, but fails the material threshold and becomes harmful when client data move between slots. There is no present basis for implementing a private constructor or claiming a deployable contribution defense. Nor does this rule out all useful client geometry: it narrows the current slot-tied assignment strategy under the tested family.

A next **design/cost review** can ask whether a local data statistic or already protected history supplies a permutation-equivariant rule. For a true local rule with the same public function at every slot, permuting client data, associated weights and local histories should permute the chosen profiles. The resulting aggregate law is then invariant to slot renaming in distribution, unlike the hardcoded roster. This property does not itself certify privacy, robustness or utility. The global assignment oracle optimizes coupled mean/curvature terms across all clients, so its assignment need not be recoverable by any independent local statistic.

Specify the candidate statistic's information source, prediction target and complete accounting before implementation. If using raw information, analyze protected selection; if using prior protected uploads, include the prior release budget and utility cost. Show the remaining expected utility headroom against tuned shared and relevant fixed public allocations after selection errors, added perturbation cost and stale-profile effects. Merely reproducing dominant-label/slot mapping on the stress corpus would not establish broad applicability.

As a cost diagnostic, even holding a favorable profile fixed, allocating selector cost $\xi$ from an update epsilon label $E$ increases a fixed-shape quadratic noise penalty by the factor $[E/(E-\xi)]^2$. This ignores selection error and changing means, so it is not an estimator certificate or an attainable benefit bound. It illustrates why roughly $0.001$ raw-information CE headroom can be fragile after construction costs. No numerical selector experiment is implied by this observation.

Both the fixed and tuned arms remain interventions at raw unprotected source checkpoints, in three class-bias contrasts. Their epsilon labels still do not account for the source trajectory or complete observer. The result establishes neither full-model privacy nor CIA mitigation, novelty or natural-client generalization. This independent review made only note edits and launched no new jobs.
