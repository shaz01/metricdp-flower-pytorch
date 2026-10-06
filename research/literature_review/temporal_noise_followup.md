# Temporal noise: full-participation methods review and transfer boundary

2026-10-06. Follow-up methods review outside the frozen 31-source review count. No experiments or mechanism implementation. Primary proceedings PDFs inspected locally with `pdftotext -layout`; mathematical transfer statements below are our deductions, not reported empirical results.

**Finding:** matrix mechanisms are formally applicable to all-client-every-round contribution secrecy. The strongest direct framework explicitly includes every-step participation. Their published utility gains, however, concern different participation schedules and workloads. A bounded, public temporal filter with persistent client-local noise state is a legitimate research comparator against a peer observing broadcasts and its own randomness; it does not silently protect a server observing every upload. Neither temporal correlation nor noise cancellation alone establishes a gain.

## Primary methods cards

### T01 — Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning

Choquette-Choo, McMahan, Rush and Thakurta, ICML 2023. [Proceedings](https://proceedings.mlr.press/v202/choquette-choo23a.html), [full PDF](https://proceedings.mlr.press/v202/choquette-choo23a/choquette-choo23a.pdf).

**Methods read:** Sections 2–3, Eq. (1)–(6), Corollary 2.1, Theorems 2.1–2.2; Appendix D stamping discussion and H.2 counterexample scope. Proof pointer: Theorem 2.2, printed p. 5, invokes the adaptive-stream theorem of Denisov et al.; Corollary 2.1 proof referred to Appendix H.1. We inspected the stated adaptive guarantee and its sensitivity conditions, not an independent reconstruction of the cited Denisov proof.

**Source explanation (bounded):** Treat one client's bounded updates across rounds as a stream. A participation schema specifies every round in which the client may contribute. Neighboring streams zero or replace one client's contributions while preserving the schedule. Section 2, printed p. 3, explicitly includes every-step participation, `(k=n,b=1)`. Factor a public workload `A=BC`, release `B(CX+Z)`, and calibrate Gaussian noise to `sup ||C(X-X')||_F`, rather than a single column norm. Corollary 2.1 makes scalar and vector sensitivity equal when `CᵀC` is elementwise nonnegative; arbitrary matrices do not enjoy this reduction. Equation (3) computes sensitivity from participating column Gram sums. Theorem 2.2 extends the guarantee to adaptively chosen inputs under its bounded stream-difference condition. Section 3 optimizes sensitivity times decoder error for a chosen workload and participation schema. This is a central matrix-mechanism framework, not a protocol concealing individual uploads from a server.

**Transfer deduction:** Our schedule is allowed, but its sensitivity must use all T contributions. Single-participation column norms or a six-participation calibration cannot be reused. Public optimization is an attractive alternative to a separately paid private selector; whole-client private adaptation of the filter remains another mechanism requiring proof.

### T02 — A Hassle-free Algorithm for Strong Differential Privacy in Federated Learning Systems (BLT)

McMahan, Xu and Zhang, EMNLP Industry 2024. [Proceedings](https://aclanthology.org/2024.emnlp-industry.64/), [full PDF](https://aclanthology.org/2024.emnlp-industry.64.pdf). The arXiv version uses a different title, “Don't Use Tree Aggregation, Use BLTs”; this card concerns the proceedings file.

**Methods read:** Sections 2–3, Eq. (2)–(7), Theorem 3.1; Appendix A Algorithm 1, Algorithms 2–3; Appendix C.2 Eq. (8)–(9), C.3 sensitivity warning. Exact locators: recurrence Eq. (3), printed p. 845; Theorem 3.1, p. 845; Algorithm 3, p. 852; adjacency and sensitivity, pp. 853–854.

**Source explanation (bounded):** BLT compresses a lower-triangular Toeplitz encoder into decay coefficients and a small number of vector buffers. To produce correlated update noise, sample fresh Gaussian `Z_t`, output `z_t=Z_t−ωᵀS_{t−1}`, then update `S_t=diag(θ)S_{t−1}+1 z_t`. This computes `C⁻¹Z` causally using O(buffer-count × model-dimension) state. Theorem 3.1 gives sensitivity for nonnegative, non-increasing Toeplitz coefficients: the worst schedule puts bounded participations as early as possible at the minimum permitted separation. Appendix C.2 uses zeroed user contributions with unchanged participation. Section 3 optimizes maximum or RMS prefix-sum error multiplied by sensitivity. Published FedAvg Algorithm 1 adds correlated noise to the aggregate on the server. Its main language-model simulation uses six participations, separation 342, over 2052 rounds. Appendix C.3 explicitly warns that a sensitivity lower bound used for tree baselines can yield optimistic privacy accounting.

**Transfer deduction:** For our full schedule, Theorem 3.1 reduces to `||C1_T||₂` times the bounded contribution scale. The recurrence is implementable inside each persistent client without cryptography. That relocation is our proposed transfer, not the paper's evaluated server architecture. A client restarting its buffers or skipping a round changes the joint law and needs a new analysis.

### T03 — DMM: Distributed Matrix Mechanism for Differentially-Private Federated Learning Based on Constant-Overhead Linear Secret Resharing

Bienstock, Kumar and Polychroniadou, ICML 2025. [Proceedings](https://proceedings.mlr.press/v267/bienstock25a.html), [full PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/bienstock25a/bienstock25a.pdf).

**Methods read:** Section 2.1–2.3, protocol construction in Sections 3–4 and Figure 3, Theorem 4.1 statement, Appendix C.2 security model and Theorem C.2 proof structure. Locators: security assumptions printed p. 4; protocol Figure 3, p. 7; sensitivity/privacy Theorem 4.1, p. 8; Appendix C.2–C.3, pp. 21–22. We did not audit every finite-field rounding constant in Appendix B.

**Source explanation (bounded):** DMM aims to release correlated noisy aggregate gradients while concealing the individual gradients and noise from a potentially malicious server and corrupted clients. Different round committees inherit secret-shared state; a linear resharing protocol moves that state securely despite dropouts. The participation schema preserves a user's rounds across neighboring datasets. Section 2.2 requires corrupted plus dropped-out clients per committee to satisfy `t_c+t_d<(1/2−μ)n`, with `0<μ<1/2`. Figure 3 specifies sharing, resharing and recovery. Privacy combines matrix sensitivity with distributed discrete Gaussian noise, discretization and finite-field effects; Theorem 4.1 states the resulting concentrated-DP bound. Appendix C.2 defines the ideal released functionality, and Theorem C.2 proves secure computation under the stated threshold. The evaluated factorizations use minimum-separation sensitivity. This result relies on the cryptographic protocol; its aggregate-only adversarial view is not supplied by ordinary plaintext Flower messages.

**Transfer deduction:** Keep as an architecture-changing comparator if secure aggregation/resharing is later authorized. Fixed clients eliminate committee churn, but do not eliminate the need to hide individual uploads if claiming its server-observer guarantee. The mathematical factorization can be reused separately; the DMM security theorem cannot.

## A client-local construction that fits the peer contract

**Our deduction and proposal; no source claims of novelty or utility superiority.** Fix registered slots, stable public weights `a_i`, public horizon T, public clipping bounds L_i, a public invertible lower-triangular temporal encoder C, and a public noise allocation τ_i. Every slot emits a message in both IN and null worlds. Null replaces its clipped contribution by zero while preserving its noise sampler. Noise draws and persistent state are private to that client.

1. Each client clips its round update and generates a fresh innovation `G_{i,t}~N(0,τ_i² I_d)` independently across clients and rounds.
2. It uses the causal recurrence represented by `E_i=C⁻¹G_i`, maintaining the required private buffers, and uploads clipped update plus `E_{i,t}`.
3. Ordinary weighted aggregation receives these individual noisy messages. The server therefore observes them individually; the peer contract gives the adversary broadcasts and its own local view, not server inbox access.
4. For a coalition A whose randomness is known, remove those shares from the noise law. Before observing the output transcript, the remaining joint noise covariance is

   `K_H ⊗ I_d`, where `K_H=(Σ_{i∉A} a_i² τ_i²) C⁻¹C⁻ᵀ`.

   This describes the full trajectory distribution conditional on known coins and fixed public choices. It is **not** a claim that posterior noise given past released models stays independent or has this unconditional covariance.
5. For target j outside the coalition, same-history clipped contribution differences have row norm ≤ `a_j L_j` for zero-versus-contribution, or ≤ `2a_j L_j` for arbitrary replacement. Set

   `s_j(C)=sup_{||v_t||₂≤1, t=1…T} ||CV||_F`.

   The corresponding Gaussian transcript bound is

   `ρ_j ≤ (a_j L_j s_j(C))² / [2 Σ_{i∉A} a_i² τ_i²]`

   for the zeroing game; multiply by four for the unrestricted replacement bound. The adaptive-stream theorem supplies the appropriate proof route once same-history contribution bounds, honest-client behavior and causal release are established. This displayed expression is an obligation/calibration candidate, not an audited implementation certificate.

If `CᵀC` is elementwise nonnegative, T01 Corollary 2.1 and Eq. (3) give `s_j(C)=||C1_T||₂`. Otherwise scalar signs alone may understate vector sensitivity. Calibrate for the largest allowed coalition, not the total number of registered clients. With only one unknown honest client, aggregate-noise averaging provides no additional unknown shares.

Public client-specific encoders C_i are also mathematically possible: independent Gaussian shares produce `K_H=Σ_{i∉A} a_i²τ_i² C_i⁻¹C_i⁻ᵀ`. An SPD K_H has a causal Cholesky factor, permitting an effective encoder from its inverse. But sensitivity and utility then concern that **whole residual covariance**, not each client's marginal variance. Data-dependent C_i, weights, clipping or innovation laws are outside this public construction unless their effect on the complete transcript is accounted for.

## A useful limit before spending implementation effort

**Our mathematical deduction.** Even full-participation matrix noise cannot obtain single-participation scaling by cancellation. Consider a fixed, nonadaptive worst-case transcript in one spatial direction. Per-round target differences may all equal `c=a_jL_j`; hence the admissible difference is `δ=c1_T`. For any SPD update-noise covariance K satisfying Gaussian zCDP budget ρ,

`c²1_TᵀK⁻¹1_T ≤ 2ρ`.

Cauchy–Schwarz gives

`(1_TᵀK1_T)(1_TᵀK⁻¹1_T) ≥ T²`,

so the final prefix-sum noise variance obeys

`Var(Σ_t E_t) ≥ c²T²/(2ρ)`.

Independent noise with per-round variance `c²T/(2ρ)` reaches this final-prefix lower bound. Thus a matrix mechanism cannot improve that final-prefix variance over this Gaussian control under these unconstrained every-round contribution bounds. It may redistribute errors across intermediate prefixes, exploit a different public optimizer workload, or benefit from a separately proved restriction on whole-client differences. This is a Gaussian workload limit, not a claim about final trained-model utility, non-Gaussian optimality, or all objectives. A final model can depend nonlinearly on the noise through training.

A two-round check makes the cancellation cost concrete. Let `C=[[1,0],[r,1]]`, `r≥0`, and prefix workload `A=[[1,0],[1,1]]`. Then `s(C)²=r²+2r+2` and `||AC⁻¹||_F²=r²−2r+3`. Their product is `r⁴+r²+2r+6`, minimized at r=0 within this family. Its visible negative update-noise correlation does not improve total prefix MSE once full-participation sensitivity is paid. This calculation compares that family with equal per-round independent noise only; it does not establish global optimality over all encoders or optimized independent schedules. In fact, the diagonal encoder `C=diag(x,y)` has objective `(x²+y²)(2/x²+1/y²)`, minimized at `(√2+1)²≈5.828` when `x²/y²=√2`. This public time-allocation control already improves the equal-allocation value6; correlation must beat it before earning a gain claim.

### Stronger robust Gaussian workload limit (our derivation; independently reviewed)

Suppose spatial dimension d≥T and the privacy certificate protects **every** sequence of target differences with each row norm ≤c. This is the unconstrained row-ball envelope used above, not a claim that every sequence is reachable from a particular client's dataset. Let update noise have covariance `K⊗I_d`, K SPD, and use `b=2ρ/c²`. For normalized difference rows V, the Gram matrix `Q=VVᵀ` ranges over every PSD matrix with diagonal ≤1 because d≥T. The Gaussian privacy constraint is therefore exactly

`max_{Q≽0, diag(Q)≤1} tr(K⁻¹Q) ≤ b`.

The SDP dual is `min Σ_t λ_t` subject to `diag(λ)≽K⁻¹`, `λ≥0`. Strictly feasible primal points ensure strong duality. Thus any feasible K admits λ with `Σλ≤b`; positive definiteness makes each λ_t positive. Inverting the PSD order yields

`K≽diag(1/λ_t)`.

For any public linear workload A and `W=AᵀA≽0`,

`tr(WK) ≥ Σ_t W_tt/λ_t ≥ (Σ_t √W_tt)²/b`.

The second inequality is Cauchy–Schwarz. A diagonal K with `λ_t=b√W_tt/(Σ_s√W_ss)` attains the bound when all W_tt>0; zero workload entries can be handled by a limit. Thus **optimized independent temporal noise is optimal for this robust, spatially isotropic Gaussian linear-workload-MSE problem**. Temporal covariance cannot lower this objective. For two-round prefix sums, W has diagonal(2,1), recovering5.828/b.

The trace expressions above are per parameter coordinate. Total d-dimensional Frobenius MSE multiplies them by d. The [independent review](../proposals/2026-10-06_temporal_geometry_math_review.md) verifies duality, inverse order, normalization and scope.

The restriction matters: this is not a theorem that all temporal mechanisms are useless. It does not cover non-Gaussian laws, data-dependent private estimators, a tighter proved set of whole-client trajectory differences, smaller spatial dimension, arbitrary spatiotemporal covariance, or nonlinear training utility. It does establish a concrete reason to avoid implementing generic correlated Gaussian noise merely on published sparse-participation intuition. A potentially meaningful temporal research question is whether our local training dynamics permit a **provable tighter whole-client trajectory sensitivity set**, and whether obtaining that restriction has a privacy or utility cost. Empirical trajectory similarity alone cannot replace the worst-case set.

## Research decision and handoff

Retain temporal noise as a **public, causal, observer-conditioned comparator**, with BLT providing an inexpensive persistent-state implementation recipe. Deprioritize a claim based only on cancellation or published cross-device gains. The promising knowledge is the correct adaptive whole-trajectory construction and sensitivity objective, rather than a demonstrated better protection technique.

Before any implementation, specify the utility workload: final prefix, all prefixes, momentum/learning-rate workload, or actual model-training loss. Compare against independently optimized public per-round noise allocations at the same coalition-conditioned transcript budget. If pursuing client-specific filters, justify why the public slot assignment or a paid/otherwise proved private local choice beats a single optimized public filter. Distinguish client-local persistent noise from server-local BLT and cryptographic DMM in every result. No sampling amplification applies when every client contributes in every round.

## Retrieval and screening record

Date: 2026-10-06. This is a bounded three-family follow-up, not an update to the frozen systematic count.

- Parent-supplied primary seeds: T01 PMLR v202/choquette-choo23a; T03 PMLR v267/bienstock25a.
- Exact discovery query: `BLT 2024 differential privacy banded lower triangular matrix factorization participation`.
- T01: included, full methods scope above. Proceedings PDF downloaded successfully.
- T02: included, full methods scope above. ACL proceedings PDF downloaded successfully. arXiv2408.08868 identified as alternate-title version, not a separate source.
- T03: included as architecture-changing comparator. Browser raw-GitHub PDF open returned internal error; guessed PMLR nested PDF URL returned404. Exact raw-GitHub PDF from proceedings metadata downloaded successfully using urllib; no access restriction bypass.
- NeurIPS2023 Amplified Banded Matrix Factorization; NeurIPS2024 Banded Square Root; arXiv2405.15913 scaling; BLT inversion2025: discovered related primary candidates, deferred because the three inspected sources already answer this bounded transfer question. Their methods are not claimed assessed here.
- Opaque documentation, author directories, ResearchGate copies and search snippets: excluded as mechanism-evidence sources for this follow-up; no technical claims rely on them.
