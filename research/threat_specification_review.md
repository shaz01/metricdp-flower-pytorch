# Client-participation threat specification: review and decision boundaries

Date: 2026-10-01. Repository inspected on `feature/client-specific-noise` at `1081ff7`. Audience: owner and agents preparing the next research protocol. **Status: proposed threat/evaluation specification for review; no mechanism has been selected or implemented.** This document separates verified code observations, mathematical deductions under stated assumptions, and research decisions still requiring agreement.

Background: [evidence audit](project_evidence_audit.md), [privacy/noise primer](literature_review/privacy_and_noise_primer.md), [evaluation requirements](literature_review/evaluation_requirements.md), and [distributed-source cards](literature_review/distributed_sources.md). The frontier remains reproducible exploratory evidence; this document does not declare experiments complete or replace historical outputs.

## 1. Candidate primary game

**Proposed primary research target:** a passive, honest-but-curious participating peer tries to infer whether a specified *other client's complete dataset* contributed to the global-model trajectory. The server is trusted to follow the protocol and does not reveal individual honest-client uploads to that peer. The initial-paper observer is the starting point; adding server protection is a separate extension.

Let target T and curious peer A be distinct. Fix A's dataset, other background-client datasets, the auxiliary-information policy, public algorithm parameters, candidate identities, and the release schedule. The challenger samples a bit B independently of training randomness:

- **IN, B=1:** the target contributes its whole dataset and all other clients keep their datasets.
- **OUT, B=0:** the target's whole learning contribution is removed; background datasets are not repartitioned or reassigned.

Background clients train from the evolving global model in both worlds. Consequently their later updates can differ as a legitimate consequence of target inclusion; “same background data” does not mean freezing their numerical updates across an entire trajectory.

The attacker receives **one realized transcript**, not its paired counterfactual. It observes the released global models at the prescribed rounds and its own local execution: dataset, local optimizer state, unperturbed/clipped updates, uploaded messages, weights if known, sampled noise, private seeds it owns, and allowed protocol metadata. It may use target-distribution auxiliary records under a declared access model. The current overlapping 10% target-training shadow is a strong auxiliary-data setting; independently acquired, disjoint target-domain records should be evaluated separately.

This is a *candidate* game because two meanings of “remove a client” must still be resolved:

| Event to hide | OUT execution | Consequence |
|---|---|---|
| Physical participation | The target sends no messages and is absent from the active federation | A visible roster, message existence, or count can reveal the event. Distributed target noise also disappears. |
| Data/learning contribution | A registered candidate slot sends a prescribed dummy contribution and retains required traffic/randomness | Network enrollment is public; the hidden event is whether the target's dataset influenced training. This requires a genuine dummy-share protocol assumption. |

The existing removal benchmark follows the first execution at the simulator level, while its loss-only attack sees a restricted model/shadow view. A fixed candidate-party experiment is a separate protocol, not a cosmetic relabeling of existing OUT results.

## 2. Exact attacker view and metadata contract

**Proposed contract.** Write the allowed view before choosing noise. At minimum define whether the peer sees total active-client count, active identities, normalization denominator, target-specific weights, per-client clipping/noise/covariance settings, timing, sampling/dropout flags, failed rounds, and the existence of individual messages. Algorithm parameters that are public must be identically defined in both worlds or included as observable releases with their privacy implications.

If T's identity appears in an active roster visible to A, the participation game is trivial regardless of model noise. Even without identities, a publicly known background size plus `n` versus `n-1` reveals B. Hiding only client IDs while publishing an otherwise diagnostic count is insufficient. A server may know enrollment while the peer does not; that is compatible with the proposed trusted-server game but must be said explicitly.

**Verified code observation.** `runtime_config()` merges Flower run configuration with `METRICDP_RUN_CONFIG`. The runner stores `num-clients`, seed, privacy mode, and data-module configuration; `experiments/reproduce/client.py` consumes this runtime configuration. The current IN/OUT runner changes `num-clients`. Thus the simulated training app has world-dependent configuration that the evaluated shadow-loss attack deliberately does not consume. Existing attack results do **not** demonstrate secrecy against every item available to that app. This observation concerns the simulation; it does not establish which configuration a future deployment would expose.

**Proposed correction.** Separate a research-oracle manifest from an explicit attacker-facing view. The former may contain labels, counterpart links, true participant counts, run names, reproducibility seeds, target IDs, and private diagnostics. The latter may contain only the declared observer's information. A rule denying oracle labels to the attack must not silently deny information that a real peer would possess. Define a deployable masking/dummy-traffic assumption if physical participation must be hidden; otherwise explicitly restrict the claim to global-model information or data contribution conditional on public enrollment.

Reproducibility seeds for hidden privacy noise belong to the audit manifest, not public attack metadata. Client-owned seeds remain available to that client. A globally shared seed capable of reconstructing all perturbations would defeat the intended unknown-noise argument against a participating peer.

## 3. Peer conditioning: what randomness actually protects T?

**Mathematical deduction under independent centered noises and fixed known coefficients.** For one round, let the clipped client update be u_i, aggregation coefficient a_i, and sampled noise Z_i with covariance Sigma_i. The released increment is

\[
Y=\sum_i a_i u_i+\sum_i a_iZ_i.
\]

A knows its own update and Z_A. Where the weights and linear aggregation rule are known, it can subtract their weighted contribution. The randomness remaining against A has covariance

\[
V_{\neg A}=\sum_{i\ne A}a_i^2\Sigma_i.
\]

Do not count A's known noise as protection against A. A coalition removes all contributions/noise it can reconstruct. For correlated noise, this sum-of-variances formula does not apply; use the **conditional joint covariance/law given the coalition's known seeds and messages**, including cross-client terms. Marginal upload variance is not enough.

Proofs should bound the distribution of the complete causal attacker view with its own dataset/randomness fixed or modeled as specified. Per-round composition conditions on the preceding transcript and establishes bounds uniformly over allowable histories. Arbitrarily conditioning after the fact on a selected outcome is not a substitute for such a proof. Own updates that depend on earlier global models belong in the causal view, rather than being assumed identical constants across all rounds.

**Decision boundary.** Local upload privacy, distributed aggregate privacy, and trusted-server model-release privacy are three different claims. Client-side placement alone does not select one. To promise protection from a server seeing individual uploads, analyze those uploads and their metadata directly. To promise distributed protection, specify secure aggregation, collusion bounds, unknown honest noise, dropout handling, and target-absent execution. None is supplied automatically by the inspected current server-side training flow.

## 4. Current aggregation is sample-weighted; client count is not the denominator

**Verified code observation.** Client train replies report `num-examples = len(trainloader.dataset)` in `experiments/reproduce/client.py`. `make_base_strategy()` constructs FedAvg without overriding its weighting key. The installed Flower FedAvg defaults to `weighted_by_key="num-examples"`, and `aggregate_arrayrecords()` uses each reported count divided by the sum of counts. Flower's DP wrapper clips each model update relative to the previous global model before passing it to this weighted aggregation.

For a successful FedAvg round the clipped update average therefore has the form

\[
q(D)=\frac{\sum_{i\in P}m_i u_i}{\sum_{i\in P}m_i},\qquad \|u_i\|_2\le C,
\]

where m_i is the reported training sample count. The server-noise wrapper instead calls `compute_stdv(z,C,n)=zC/n`, using configured client count n. The frontier runner sets `z = noise_ratio * active_client_count`, preserving this *numerical noise standard deviation* across removal. Metric calibration additionally divides by the maximum pairwise model distance computed before clipping. Neither statement proves that the noise is calibrated to the appropriate whole-client sensitivity under unequal weights or a privately changing denominator.

**Mathematical deduction at one round conditional on the same prior global model.** Let W be the background sample-count sum, m_T the target's count, and q_OUT the background clipped average. Then

\[
q_{IN}-q_{OUT}=\alpha_T(u_T-q_{OUT}),\qquad \alpha_T=\frac{m_T}{W+m_T},\qquad
\|q_{IN}-q_{OUT}\|_2\le2C\alpha_T.
\]

This bound changes with target mass. Even equal-client active averages have the renormalization term; they are not the fixed-denominator zero-contribution mechanism whose add/remove sensitivity is C/n. If sample counts themselves can change privately, bound the resulting complete query and all observable count releases. Do not plug the realized m_T or variance into an accountant without a uniform adjacent-input analysis.

**Proposed alternatives to specify, not choices already made:** preserve sample-weighted active averaging and derive its correct sensitivity/noise law; use bounded public weights with a fixed public denominator and explicit zero target contribution; or define a replacement experiment with its own adjacency. Switching denominator changes optimization and privacy semantics, so compare it as a new protocol rather than silently “correcting” old artifacts.

With fixed public coefficients a_i and a dummy target contribution u_T=0 in OUT, the one-round add/remove mean shift is bounded by `|a_T| C`. Replacing one arbitrary norm-C update with another instead gives `2|a_T| C`. These examples illustrate why adjacency and denominator must be locked together.

## 5. Current data coupling is not complete random-stream coupling

**Verified code observation.** `PartitionViewDataModule` preserves canonical data partitions while mapping surviving partitions onto contiguous active IDs. The removal scripts use target partition 0; canonical client j>0 maps to active ID j in IN and j-1 in OUT. `_client_round_seed()` in `experiments/reproduce/client.py` uses the active `partition-id`: `base_seed + partition_id * 100000 + server_round`. Removing target 0 therefore changes surviving clients' local random streams even when their data partitions stay fixed.

This does not invalidate comparisons between the mechanisms' marginal IN/OUT training distributions. It does mean same-seed counterfactual pairs do not isolate only the target's immediate influence under identical background random innovations. The current folded paired concordance is conditional on this coupling.

**Proposed correction for influence diagnostics.** Use stable canonical identity for local training/randomness coupling, keep paired siblings together during splits, and label the coupling. Independent-trial CIA testing need not reuse random streams between IN and OUT; it should sample them from the same prespecified distribution and infer uncertainty over independent realizations. Both designs can coexist if their estimands remain separate.

## 6. Fixed candidate parties and dummy noise shares: what would be assumed?

**Candidate protocol assumption, not current code.** Public candidate slots and scheduling are unchanged in both worlds. An enrolled target whose dataset is absent sends the prescribed zero learning contribution plus its required noise/dummy traffic. The aggregation denominator/coefficients remain as specified. Noise distribution parameters in the empty slot cannot require unavailable private data; a public/reference fallback needs analysis. Fixed slots do not imply all parties are honest, and unknown-noise guarantees still need collusion/dropout bounds.

This hides use of a client's dataset, rather than whether the machine enrolled or sent messages. If OUT truly removes the machine, its share cannot magically remain. A replacement or a server-generated missing share is another protocol with changed trust or adjacency. Failure to send dummy traffic can expose the event; do not assume the privacy definition hides a physical network fact that its transcript reveals.

Even with fixed traffic, a privately constructed noise distribution can reveal target inclusion through covariance, higher moments, estimator metadata, or temporal structure. Ensuring a common public covariance floor may help a proof but is not itself a proof for all private adaptations.

## 7. Concrete matched local/server placement controls

These are **research controls with public fixed covariance**, used to identify what placement changes before evaluating a learned distribution. They do not select the eventual protection mechanism or claim DP without sensitivity/accounting.

### Model-only observer control

Fix aggregation coefficients, clipping operation, update computation, observation schedule, and per-round conditional covariance Sigma_i. In the local arm sample independent `Z_i ~ N(0,Sigma_i)` after clipping and aggregate `u_i+Z_i`. In the central arm aggregate the same clipped u_i, then sample `E ~ N(0,V)` with

\[
V=\sum_i a_i^2\Sigma_i.
\]

These give the same conditional released-increment law for a model-only observer. With matching transition rules in every round, their model-trajectory laws match as well. Under equal weights 1/n and identical local covariance Sigma, `V=Sigma/n`; matching central coordinate standard deviation sigma requires local standard deviation `sqrt(n)*sigma`. Post-noise reclipping, private covariance adaptation, different quantization, dropout, or changed weights invalidates this simple equality.

### Peer observer control

Matching only total V is insufficient because A knows Z_A in the local arm. Construct a diagnostic **hybrid central arm**: let A add its own sampled noise in both arms and retain identical knowledge of it; other clients' clipped updates are clear to the trusted server, which replaces their individual noise draws by one independent draw

\[
E_H\sim\mathcal N(0,V_H),\qquad V_H=\sum_{i\ne A}a_i^2\Sigma_i.
\]

Both arms release `sum_i a_i u_i + a_A Z_A + unknown honest noise`. The peer's joint view and utility can therefore have the same conditional transition law, while only the placement of **honest-client unknown noise** differs. The server's view differs intentionally; this is a trusted-server peer control, not a server-privacy equivalence result. For a coalition, preserve all coalition-owned coins/messages in both arms and replace only the unknown independent contributions.

The existing `global-dp` implementation has no client-noise input and is not this hybrid control. Keep it as a historical comparator. Report total-noise matching and peer-residual-noise matching separately if a simple central arm is also used; they answer different questions. A proposed client-specific distribution must then show a gain over these appropriately specified fixed-noise controls, rather than attributing an unmatched variance or denominator change to placement.

## 8. Formal guarantee versus measured loss-attack success

**Proposed formal question.** For the chosen adjacency and full attacker view V_A, establish for every allowable neighboring pair and output event S,

\[
\Pr[V_A(D)\in S]\le e^\varepsilon\Pr[V_A(D')\in S]+\delta.
\]

Analyze privately estimated distribution parameters, all released diagnostics, mean/covariance changes, hidden residual randomness, and adaptive composition. Record-level accounting cannot be relabeled as whole-client protection. Current metric calibration is an empirical calibration comparator, not an already established certificate under this game.

**Proposed empirical question.** Given a frozen defense-aware attack and a defined population of independent target/federation realizations, estimate ordinary trial-level ROC-AUC, TPR at prespecified FPRs where sample size supports them, and utility with grouped uncertainty. Attack development selects score sign, thresholds, checkpoints/features, and hyperparameters; test labels do not. The current scorer compares same-round IN/OUT losses then folds the sign on those observations. Retain that statistic as a separately named diagnostic. Neither its near-chance value nor a confidence interval merely containing 0.5 certifies the formal inequality or practical equivalence to chance.

## 9. Decisions to resolve before implementation

1. Hidden event: physical absence, or data contribution among registered candidate parties?
2. Primary observer: model-only, curious peer with own state/coins, or server; which extensions are separate?
3. Public metadata: counts, identities, weights/denominator, estimator outputs, scheduling/failures, seed visibility?
4. Aggregation/adjacency: current active sample-weighted average, fixed public coefficients/denominator, or replacement?
5. Distribution target: valid whole-client DP, modeled-population protection, empirical CIA suppression, or separately evaluated combinations?
6. Distribution construction: public statistics, already privatized history, or private local data with explicit analysis? No choice is made here.
7. Evaluation: independent causal trials, attack-training/calibration split, paired influence diagnostics, matched placement controls, and precision/power plan?

The next agent should resolve these boundaries in a concise protocol contract and proof sketch before selecting a sampler or launching training. No implementation changes or experiment launches are authorized by this document itself.

## Verified code locators

- [client training/round seed and sample-count reply](../experiments/reproduce/client.py): `_client_round_seed`, `_client_data`, `train`.
- [runtime configuration](../metricdp_pytorch/utils/runtime.py): `runtime_config`; [runner](../experiments/reproduce/runner.py): config construction and `run_simulation` setup.
- [strategy factory](../metricdp_pytorch/strategy_factory.py): `make_base_strategy`, `make_strategy`; [metric calibration](../metricdp_pytorch/metricdp_strategy.py): `aggregate_train`, `_add_noise_to_aggregated_arrays`; [global wrapper](../metricdp_pytorch/globaldp_strategy.py).
- [CIFAR removal runner](../experiments/cia/scripts/cifar10_remove.py): `_active_clients`, `build_combos`; [partition views](../experiments/cia/datasets/partitions.py): `PartitionViewDataModule.canonical_partition_id`, `client_loaders`.
- Installed Flower source inspected via `uv run python`/`inspect`: `flwr.serverapp.strategy.FedAvg.__init__` and `aggregate_train`; `strategy_utils.aggregate_arrayrecords`; `DifferentialPrivacyServerSideFixedClipping.aggregate_train`; `flwr.supercore.differential_privacy.compute_stdv`. These are dependency-source observations tied to the installed environment, not new repository code.
- [current CIA scorer](../experiments/cia/scripts/score_stage.py), [checkpoint loss evaluator](../experiments/cia/cia.py), [attack runner](../experiments/cia/attack_runner.py).
