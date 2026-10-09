# Individual privacy filters: a client-construction predecessor

2026-10-06. Focused primary-method review and project transfer. [Feldman–Zrnic, NeurIPS2021 official record](https://papers.neurips.cc/paper_files/paper/2021/hash/ec7f346604f518906d35ef0492709f78-Abstract.html); [inspected arXiv v4, 8January2022](https://arxiv.org/abs/2008.11193v4), [PDF](https://arxiv.org/pdf/2008.11193v4). Read Definitions2.5/2.6, Example2.7/2.8, Theorem3.1 proof and Remark3.2, Theorems4.3/4.5 and Corollary4.7, Algorithm7/Proposition5.2. Official PDF fetch failed; methods were read in the author version. No empirical claim reproduced. This is a separately tracked follow-up, not a version-1 screening amendment.

## What the technique does

**Source explanation.** Ordinary accounting pays a worst-case privacy cost for every individual at every step. The paper instead measures each individual's influence on each adaptive query and enforces a cumulative privacy filter. For Gaussian linear queries, influence cost depends on squared query norm divided by noise variance. Filtering removes contributions that would exceed the cap. Algorithm7 applies this to gradient descent: clipping uses both the ordinary per-step radius and the square root of remaining cumulative norm budget. The resulting privacy proof uses fully adaptive Rényi composition, not just a favorable observed sequence of gradients. Releasing the active-set size needs additional protection. This construction and its basic accounting are existing research, not our new method. Source scope: §3–5, especially Corollary4.7 and Algorithm7/Proposition5.2.

## Why it changes our question

The local statistic need not always be released or used to choose a visible covariance. It can guide a client's contribution *inside a uniform enforced influence cap*. A fixed public noise law then protects the resulting channel. This avoids the inference “all privately computed construction statistics require a separately released DP selector.” What matters is whether the complete released mechanism has a uniform neighboring-input certificate. The earlier RR selector paid for its own output channel; this alternative does not publish a raw statistic or privately change noise variance.

This is not free use of a measured local norm in an accountant. Every dataset and every possible observed history must satisfy the enforced cap. An empirical trajectory with low norm is evidence about possible utility, not a privacy proof.

## Whole-client transfer: own conditional derivation

Treat one client's entire dataset as one atomic input. Fix registered positive public weights a_i, public release rounds and a peer/coalition outside the target. A client query is deterministic given its own dataset, fixed training randomness and shared protected model history; its private filter state may depend only on these inputs. No additional cross-client private-state channel is permitted.

Each round retains fresh independent Gaussian noise shares in every registered slot, including a zero-dataset or exhausted slot. After conditioning on the peer's known coins, the unknown aggregate has isotropic variance S_H,t≥V_t>0. V_t is a public guaranteed lower bound for every allowed coalition/dropout pattern. Distribution parameters cannot change with private gradient norms. The server protects the individual-upload access boundary; this is not server privacy.

Let v_it=a_i u_it be the weighted learning contribution after ordinary clipping. Track spent_i=sum_{s<t} ||v_is||²/(2V_s). Before release, further shrink v_it to have norm at most sqrt(2V_t*max(rho0-spent_i,0)), then add its unchanged noise share. Charge the ledger using the actual post-filter learning contribution, excluding the added noise; do not charge the pre-filter proposal. Exhaustion produces zero learning contribution but does not stop traffic or noise. Do not publish spent_i or an exhaustion indicator.

For target dataset versus zero dummy, at fixed history other queries agree and the conditional mean shift is v_Tt. At Rényi order alpha, its symmetric Gaussian divergence is alpha*||v_Tt||²/(2S_H,t), at most alpha*||v_Tt||²/(2V_t). The pathwise filter bounds the sum by alpha*rho0 for all datasets/histories. Fully adaptive pairwise composition therefore gives rho0-zCDP for the complete peer transcript. Standard conversion gives epsilon≤rho0+2sqrt(rho0*log(1/delta)). Arbitrary replacement instead has the conservative4rho0 bound; do not reuse the dummy bound unchanged. [Independent proof review](../proposals/2026-10-06_client_energy_filter_math_review.md) checks the transfer and restrictions.

The bound does not divide by records in the client's dataset. Learning a local gradient-norm distribution does not change the privacy unit. Local optimization cannot reintroduce private cross-client messages or data-dependent server calibration that changes the observable noise kernel.

## Potential benefit and essential controls

An ordinary T-round radius-C envelope pays for T*C². The filter imposes a total energy budget B, allowing clients with useful low-norm updates to spread it differently over time. If B is materially below T*C² while useful signal survives, a smaller public noise variance can attain the same transcript budget. Persistent heterogeneous gradients may instead exhaust the cap early and severely bias learning; no advantage is assumed.

At fixed noise variance, standard public per-round radii C_t with sum C_t²≤B provide an important control. Optimize their time allocation and clipping before crediting a private learned schedule. Also compare the source Algorithm7-style remaining-energy rule itself, matched server/peer aggregate noise laws, and one-time sanitized surrogate construction. A learned policy must beat these existing mechanisms; adapting the privacy filter alone is not novelty.

Client-specific distribution construction could learn where useful updates lie and allocate the admissible influence budget to preserve them. Initially keep the noise covariance public and fixed: privately changing its variance or shape invalidates the displayed proof. A non-Gaussian law requires a fresh per-step Rényi/shift-cost certificate and compatible filter; it cannot inherit the Gaussian squared-energy cost by analogy. Gaussian is a proof/control reference, not a restriction on the final research mechanism.

## Handoff and next feasibility question

Ask whether a distribution-guided policy can preserve materially more held-out learning signal than optimized public schedules under the same *enforced whole-client cumulative influence budget*. This is a promising construction question with a tractable proof boundary, not evidence that it already wins. The method remains client-side, uses individual client histories, and addresses repeated contribution cost rather than merely stretching a one-step covariance.

Before a pilot, specify a deterministic/shared-history client execution, fixed-slot dummy and metadata law, worst coalition noise floor, public calibration and a baseline-inclusive clipping grid. Reject a learned policy if its raw-information oracle has no headroom before building a private/noisy estimator. Require genuinely coupled noisy trajectories and independent loss/CIA evaluation for later claims. Do not promote the prior saved-update replay to such evidence.

Search provenance: `Feldman Zrnic Individual Privacy Accounting Renyi Filter 2021 paper`; `client level differential privacy filters cumulative gradient energy norm budget`. Broader discovery queries about trajectory clipping found adaptive-clipping/DPFL methods, but unread publisher snippets are not assigned guarantees. Full novelty work must assess these nearest methods before naming a new algorithm.
