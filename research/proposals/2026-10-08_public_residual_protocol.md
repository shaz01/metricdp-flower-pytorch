# Public-reference residual and compression feasibility

2026-10-08, `feature/client-specific-noise`. Owner: “continue with this step dont stop until you find something”. This authorizes the bounded representation/construction probe recommended by the matched-CIA findings. Overall research remains active; no completion/merge decision. The research question and controls are frozen here before utility results.

## Question and information contract

Can a small task-relevant client residual improve learning over an equally informed public-only reference at near-chance whole-client CIA? Public-reference construction and subspace projection are prior ideas, not novelty claims. Primary risk target q=0.55; secondary q=0.65. Gaussian is a reference channel; the preceding comparison found no consistent radial-law benefit.

Keep eight registered slots, weights1/8, absent client's contribution zero with retained traffic/noise. A peer knows its own upload and coins; the conditional strong attacker additionally knows target/background datasets. Protect contribution-versus-empty, not arbitrary-dataset replacement, physical connection or individually visible uploads. Private bank, class-stress partitions and seeds42/43/44 are reused. Fresh image/noise confirmation is not independent client populations.

This extension explicitly grants all methods the same512LABELLED public images. It is a different auxiliary-information contract from the initial unlabelled-prototype constructor. A public-only learner is the primary control; private improvements must exceed that control, rather than crediting the public labels to privacy.

## Public reference and public geometry

Train the existing identifiable51-parameter four-class fixed-feature learner on public data for80/320/1280steps. Select a scalar logit multiplier (inverse conventional temperature) from0.5/0.75/1/1.25/1.5/2 by old development CE, jointly with step count. Freeze the best public-only reference B BEFORE private constructor selection. This strengthens the public control against simply supplying additional optimization. Development labels/reference selection remain offline research choices, unaccounted in an end-to-end deployment claim.

Compute the CE Hessian at B using only public features/predictions. Its ordered eigenvectors define nested parameter subspaces of dimensions1/3/12/51. Two public metrics: Euclidean orthonormal projection, and Fisher coordinates scaled by sqrt(eigenvalue+0.01). The inverse map restores parameter coordinates. Geometry depends on public reference information, not private teacher covariance. It remains fixed in every target/world. This tests public task geometry; it does not implement a different privately estimated noise distribution for each client.

## Constructors and causal controls

- **Absolute:** project saved zero-initialized local teachers (raw/prior-corrected/balanced,20/80steps). Decoder offset B−a*projection(B), gain a. Empty query0. Thus preserve B outside the subspace; do not drop useful public information.
- **Centered:** project those teachers minus B; empty residual0. Decoder B+a*decoded aggregate. Before clipping, all-IN centered/absolute learners coincide. In OUT centered imputes B for the absent teacher, while absolute imputes0 within the subspace. Report that distinction.
- **Fine-tuned residual:** initialize local teachers at B;20/80ordinary or balanced-loss steps, with updates constrained to the same public subspace. Encode theta_i−B. Empty local training returns B and residual0. Decoder B+a*decoded aggregate. Compare separately with centered zero-initialized teachers to isolate initialization/training.
- **Legacy smooth/vote controls:** start from the Gaussian per-family development choices in the preceding matched-CIA comparison at the corresponding risk. Reoptimize public clipping caps and a for model_direct/model_distribution/logits/probabilities/votes. Decode using the previous learner, then interpolate with B as B+a*(theta_old−B). They receive the identical public reference. These controls retain earlier selected transforms/steps/prototypes/ridges; they are not an exhaustive new search of every possible learner.
- **Public-only B:** no private contribution; CE/accuracy on exactly the same data. Include noiseless diagnostics solely to locate a signal/noise bottleneck.

Use public caps C=0.025/0.05/0.1/0.2/0.4/0.8/1.6 and gains a=0.1/0.3/1, common to all private constructor/legacy arms. The public-only arm covers a=0. Clip the whole encoded local vector to C BEFORE multiplying by1/8. Fixed public cap, not maximum norm of the eight saved clients, calibrates noise.

## Channel and accounting boundary

Let x_i=clip_C(encoded_i)/8. Each independent Gaussian share has covariance sigma²/7 I, where sigma=C/(8*sqrt(2)*Phi^-1(q)). All seven unknown shares have covariance sigma²I. For any saved or future fixed query with contribution norm<=C/8, conditional exact descriptor AUC is Phi(norm(x_target)/(sqrt(2)*sigma))<=q. The full aggregate coordinate variance is8/7sigma². Fixed public cap avoids the preceding private-M calibration gap for a FIXED configuration; adaptive research selection/publication, finite precision and actual distributed deployment remain outside certification. Replacement adjacency can have2C/8shift; it is not covered by the same q bound. No common epsilon comparison with radial/metric privacy is claimed.

## Development and frozen confirmation

No reserve features during development. Evaluate32paired fresh Gaussian draws per each primary target0–3 and world, equally averaging the eight fixed strata on the OLD1,024development examples. Apply the same streams across gain/cap controls. Keep distinct stream namespaces from all previous experiments. Save every candidate CE array, chosen configurations, public B/basis, bounded query/clip diagnostics, source hashes and protocol hash BEFORE confirmation. Select each family minimum; primary candidate is best fine-tuned residual, and strongest alternative is best absolute/centered/legacy control or public-only B.

After selections are saved, consume the first128previously unused reserve images perclass (512total) from the preceding remaining968;456remain. Verify disjointness from all role and earlier evaluation indices. On frozen family selections evaluate256fresh utility draws/world and4,096new attack draws/world for all8targets, with0–3primary. Peer7excepttarget7usespeer6. Save ordinary oriented ROC-AUC, theoretical AUC, CE/accuracy, every score/draw and indices. Add central same-query Gaussian with identical unknown residual sigma but total marginal variance sigma² rather than8/7sigma²; never attribute a placement advantage to an unmatched channel. Reuse common draws for paired controls, not across attacker-known and unknown coins.

Primary positive gate: fine-tuned residual improves CE over BOTH public-only B and strongest alternative in all three seeds, lower fixed-stratum paired-noise95%interval>0.001, at q0.55. Accuracy must be reported; CE gains need not be accuracy gains. All8theoretical AUCs must satisfy the public-cap ceiling; empirical deviation is measured, not treated as exact calibration failure. Intervals condition on fixed targets/utility images and are not selection-adjusted or population intervals. Report partial/negative gates honestly; no further sweep or defense claim is inferred.

## Implementation and verification plan

Write meaningful tests for empty/reference semantics, constrained local training, Hessian finite differences, whole-vector cap/shift bound and exact Gaussian risk mapping. Observe failing tests before implementation. Implement separate development/evaluation stages using existing saved teachers/features. Obtain independent contract/query review and separate saved-artifact arithmetic/split/hash/selection/ROC verification. Run default regression suite. Record bounded findings under research/proposals, update handoff/STATUS and commit/push on the active branch; no root completed-experiment report or merge.
