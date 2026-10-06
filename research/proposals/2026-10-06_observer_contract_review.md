# Observer-specific protection contract and client-side construction

2026-10-06. Owner: “lets move to that”, approving the observer/aggregate-output design review proposed after the [protected-history comparison](2026-10-05_protected_real_history_findings.md). This is a research recommendation and proof sketch, not an implemented protocol, final defense selection or completed experiment. The owner already chose **dataset contribution among registered slots** as the secret. No physical-connection secrecy is assumed.

## Main decision

Use the **participating-peer, released-model transcript** as the proposed primary protection contract, retaining the earlier [threat specification](../threat_specification_review.md). Include the peer's own data, messages, state and privacy coins. Treat privacy against a server that reads every upload as a separate, stronger extension. Our recent per-upload calculations cannot decide the achievable utility of this narrower peer contract.

Client-side noise placement does not require local DP. It can instead implement an aggregate law that protects the peer's view, provided the server follows the protocol and does not disclose individual uploads or private diagnostics. This explicitly retains the earlier server access-boundary trust; it does not add secure aggregation. A curious or colluding server breaks this boundary unless a separately specified protocol protects uploads.

The worthwhile question is now: **can client-specific construction improve the complete peer-conditioned aggregate law, at equal protection and learning protocol, beyond a strong server-noise reference?** Merely moving the same noise distribution onto clients cannot establish that improvement. The existing metric-inspired calibration remains an empirical CIA comparator, not a certified epsilon-matched DP baseline.

## What the current benchmark actually observes

Independent read-only code audit at parent commit `a2a17b2`:

| Item | Verified location | Consequence |
|---|---|---|
| Removal really changes the active federation | `experiments/cia/scripts/fashion_mnist_remove.py:116` and `:144` | Historical OUT is deletion/renormalization, not fixed-slot dummy contribution. |
| Canonical data preserved, active IDs remapped | `experiments/cia/datasets/partitions.py:144` and `:175` | Data coupling does not preserve surviving client RNG streams. |
| Local RNG uses active ID | `experiments/reproduce/client.py:25` and `:83` | Removing slot 0 shifts background local seeds. |
| Clear uploads and per-client counts/losses reach server | `experiments/reproduce/client.py:122`; `metricdp_pytorch/metrics.py:14` | Current server observes more than the shadow-loss scorer. |
| Checkpoints load full aggregated model | `experiments/cia/cia.py:36` and `:53`; `attack_runner.py:99` | Model access permits stronger attacks than the scalar currently scored. |
| Shadow uses actual target training examples | `experiments/cia/shadow_dataset.py:27`; `datasets/shadow.py:134` | Current 10% overlap is a strong auxiliary-data setting. |
| Score uses clean-shadow loss with folded paired concordance | `experiments/cia/scripts/score_stage.py:34` | The frontier is not independent ROC AUC or a full temporal attacker. |
| Incoming models reach clients; histories include operator metrics | `experiments/reproduce/client.py:92`; `server.py:137` | Specify what a peer receives; do not accidentally publish diagnostic releases. |

The metric strategy sees client models to compute pairwise distances before server clipping/noise (`metricdp_pytorch/metricdp_strategy.py:112`, `:136`, `:257`). Its source explicitly says this empirical distance calibration does not itself establish formal metric DP (`:71`). FedAvg uses the default `num-examples` weighting (`metricdp_pytorch/strategy_factory.py:175`), normalized over successful replies. Fixed public weights/dummy traffic therefore require a new benchmark protocol. Client-noised uploads also change the input law of the metric distance computation; distances computed from them are a new comparator. Keeping raw-distance calibration would require raw server access or a separately protected computation. These observations do not rewrite or invalidate historical outputs; they bound their interpretation.

## Proposed primary game

Fix registered candidate slots, target T, curious peer A≠T, background datasets, auxiliary-data policy, public bounded weights and release schedule. IN supplies the target dataset; OUT supplies a zero learning query in the same slot, retaining prescribed fresh noise and traffic. Do not reassign background data or normalize weights using the hidden target's realized record count.

The peer receives one realized model transcript and its own causal execution, not the paired counterfactual or auditor labels. Its initial data and private random coins are target-independent; subsequent state and messages are causal functions of those inputs and the protected aggregate history, with no extra target-information channel. Permit target-domain auxiliary data under explicit overlapping and disjoint regimes. A default passive single-peer case is a starting contract, not an assumed guarantee for arbitrary coalitions. Honest-but-curious parties follow training and randomness rules. Active manipulation, a server choosing different models for different clients, and adversarial dropout require separate analysis.

Attacker-facing metadata may contain the fixed roster, public weights, public bank/budgets and predetermined rounds. It must not contain private sample counts, individual target uploads, raw geometry/profile statistics, seeds reconstructing other clients' noise, target-dependent failure flags or operator loss/distance diagnostics. A release needed by deployment cannot simply be hidden from evaluation: privatize it, include it in the law, or revise the claim.

For conditional proofs, each round uses the same preceding transcript value in adjacent worlds. Later background queries may change because the realized model histories differ; adaptive composition handles this through uniform conditional bounds. Sum the round budgets or use a justified accountant. Private local memory/estimates require a bound over all adjacent datasets and states; raw-history replay cannot supply that proof.

## Three legitimate construction routes

| Route | Protected observer | Required boundary | Research consequence |
|---|---|---|---|
| Full per-upload mechanism | Server and peers, within its stated view | Each target upload/metadata has whole-client accounting | Stronger protection; paid local construction can be expensive. Recent spikes address restricted versions of this route. |
| Client-generated aggregate noise | Peer viewing models and own coins | Server follows protocol and withholds other uploads; guaranteed unknown noise shares | Fits the proposed primary peer contract. Under-noised individual messages need not be locally DP. |
| Aggregate noise with secure aggregation | Server/peer coalitions within a stated threshold | Audited cryptographic protocol, honest-share bound, dropout handling | Distinct trust extension; not installed or implicitly authorized by this review. |

Distributed discrete Gaussian and Skellam are established aggregate-noise precedents, not novelty opportunities just because they run on clients: [ICML 2021 distributed discrete Gaussian](https://proceedings.mlr.press/v139/kairouz21a.html), [NeurIPS 2021 Skellam](https://papers.neurips.cc/paper_files/paper/2021/hash/285baacbdf8fda1de94b19282acd23e2-Abstract.html). Both study secure aggregation and finite-precision concerns. The existing [distributed technique cards](../literature_review/distributed_sources.md) explain their inspected methods and limitations.

## Worked non-Gaussian reference: divisible Laplace shares

Read the [independent mathematical review](2026-10-06_observer_contract_math_review.md) for checked assumptions and variance factors. The following is our conditional reference derivation, built on established infinite divisibility. It is not a new distribution or privacy theorem for the current code.

Write the aggregate increment as Y=sum_i a_i(u_i+Z_i), with fixed public positive weights a_i. For each coordinate j, let each client sample independent Gamma variables G_ij^+,G_ij^- with **public shape k_i>0 and shared aggregate scale b_j>0**, and upload noise Z_ij=(G_ij^+−G_ij^-)/a_i. Gamma uses shape/scale, not rate. Zero-weight slots can be omitted from this arithmetic under an explicitly public policy.

For a peer/coalition S, discard its known noise from the privacy calculation. For the remaining honest participating set H, define K_H=sum_{i∈H}k_i. The difference G_ij^+−G_ij^- is already a weighted share; do not multiply it by a_i again. Weighted residual noise has characteristic function (1+b_j²t²)^(-K_H). When K_H≥1 it is exactly Laplace(b_j) convolved with an independent Gamma-difference residual of shape K_H−1; at equality the residual is zero. Independent, data-independent extra convolution preserves the Laplace shift bound.

Thus if every allowed target query satisfies sum_j |a_T u_Tj|/b_j≤epsilon, the one-round contribution-versus-dummy aggregate channel has conditional epsilon-DP against that coalition. Require T∉S, same surviving noise-law parameters in both worlds, independent hidden coins and K_H≥1 for every permitted coalition/dropout pattern. Arbitrary target replacement doubles the displayed bound. This is a sufficient bound, not the exact optimal privacy loss for K_H>1.

With at least h_min unknown honest contributors, equal shape k=1/h_min suffices. Dropout cannot be inferred as hidden dataset absence; either retain the dummy noise share or use a public, dataset-independent permitted-dropout model and robust calibration. Abort/retry behavior belongs in the view. Noise generated by someone colluding with the attacker, or recoverable from a shared seed, does not count.

**Illustrative arithmetic, not experiment data.** Eight equally weighted slots, one curious peer, no dropout: a=1/8, h_min=7, k=1/7, L1 clipping radius C and b=C/(8 epsilon). Unknown seven shares sum to exactly Laplace(b), while the model includes all eight shares. Per-coordinate model noise variance is 2C²/(56 epsilon²). Full local Laplace(C/epsilon) at each client gives aggregate variance 2C²/(8 epsilon²): seven times larger under the stronger individual-upload contract. A trusted-server Laplace(b) reference has variance 2C²/(64 epsilon²), so the distributed model is **8/7 noisier than this reference**, because the curious peer's own noise still harms learning while providing it no privacy. The apparent gain is relative to a stronger requirement, not proof that client placement beats server placement.

For k=1/7, an individual Gamma-difference density is unbounded at zero. A nonzero shift compares a shrinking neighborhood of that singularity to one around a finite density, precluding finite pure-DP upload protection. This example claims singularity only for k≤1/2 (logarithmic at equality), not for every k<1. Aggregate protection and local protection are demonstrably different here.

## Where client-specific distribution construction could add research value

Public unequal k_i can allocate noise effort across clients, but the coalition-robust constraints require sum_{i∈H}k_i≥1. For a fixed total shape, common scales and clipping, repartitioning shapes leaves the same aggregate noise law. It is an allocation baseline, not model-utility personalization by itself.

The open construction problem needs more: geometry, clipping or an aggregate law adapted to a useful statistic while protecting its estimation and the complete observer view. A raw-data choice of shape, scale, basis or clipping body changes the neighboring output law; checking only the realized mean-shift sensitivity does not account for that change. Initially use public calibration or previously protected *global* history with a uniform conditional certificate. A per-client protected upload can still cost as much as the earlier probes. Moving private calibration to the server does not make its influence on released noise free.

Use covariance as a utility signal only after verifying it differs meaningfully from gradient means, curvature and sensitivity. Our earlier real-data findings showed small clipping-signal headroom, not compelling covariance-noise benefit. Explore a different law only against strong matched central and aggregate controls, at the same adjacency, weights, release schedule and total protection.

## Decision and next bounded step

The proposed primary contract is the peer/model transcript above; stronger server privacy remains explicitly separate. No automatic trust upgrade, secure aggregation implementation, sampler or training sweep follows from this review.

Next proposed work is a **methods-level comparator and proof-feasibility review of infinitely divisible non-Gaussian aggregate laws**, prioritizing Harrison–Manurangsi's 2025 generalized/multi-scale Laplace mechanisms over treating Gamma-share Laplace as new. The [focused literature handoff](../literature_review/infinitely_divisible_followup.md) records new sources and read limits. Determine vector calibration, composition, coalition residual law, quantization and private-construction costs before a bounded utility calculation. This can reject the family without another learning sweep.

Any eventual pilot needs a genuine fixed-slot dummy runner, canonical random-stream identity, an attacker-view export, independent trajectory trials and a defense-aware temporal/model attack. Preserve the historical frontier as a separate empirical reference. No model accuracy, CIA AUC or novelty result was produced in this step.
