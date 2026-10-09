# Independent whole-client energy-filter transfer review

2026-10-06. Verdict: **valid conditional whole-client star-adjacency transfer**, provided the energy cap is mechanically enforced before every upload for every local dataset and every shared history, with the same fresh noise and traffic after exhaustion. It is a meaningful way to impose a tighter trajectory set than unrestricted row bounds. It is not a new privacy-filter theorem or evidence of utility superiority. No experiment or implementation was run.

## Primary method actually read

Feldman and Zrnic, [Individual Privacy Accounting via a Rényi Filter, arXiv2008.11193v4](https://arxiv.org/pdf/2008.11193v4). Read definitions2.5–2.6/Example2.7 (pp5–6); Theorem3.1 proof and Remark3.2 (pp7–8); Theorem4.3/Remark4.4 (pp9–10); Theorem4.5 proof/Remark4.6 (pp11–12); Corollary4.7 (p13); Algorithm7 and Proposition5.2 (pp17–18).

Theorem3.1 handles adaptively chosen two-direction conditional RDP costs through a supermartingale when their pathwise sum stays below a fixed bound. Remark3.2 explicitly allows one fixed dataset pair. Theorem4.5's individual filter relies on a point's inclusion depending on itself and previous shared reports, so other points' inclusion agrees at the same history in deletion worlds. Remark4.6 discusses local implementation. Corollary4.7 uses squared query norms for Gaussian accounting. Algorithm7 clips each gradient to both its per-step cap and the square root of remaining norm budget; its budget is an enforced restriction, not an empirical assertion that unfiltered gradients stay small.

These are per-point/linear-sum predecessor methods. Treating a complete client dataset as the atomic point, with the registered dummy replacement and coalition view below, is our transfer deduction. Existing prior art precludes describing the basic private energy-filter principle as a novel estimator or accounting invention.

## Precise conditional client construction

Fix a registered roster, positive public weights a_i, public round schedule and a public fresh-noise law. Let u_it(D_i,h_<t) be a deterministic local query given its own entire dataset and full observed shared history, possibly using fixed public seeds and a deterministic private state reconstructible from those arguments. A filtered query must also obey this dependence. Random local hidden training states require an additional conditional-kernel argument; they are not automatically covered by describing the output as a fixed Gaussian translation.

For every allowed coalition H-complement, let S_H,t be the variance per coordinate of the remaining independent weighted Gaussian noise after the observer subtracts its own contributions and known coins. Require a public V_t>0 with S_H,t≥V_t in every allowed observer/history/dropout condition. The target is outside the coalition. Registered dummy input sets the target learning query to zero but still emits its normal fresh-noise message. No exhaustion, remaining-energy, activity/count, timing or empty-state metadata is released.

Write the actual weighted filtered query as x_it=a_i*filtered_u_it. At history h, keep private used energy

    E_i,t-1 = sum_s<t ||x_is(D_i,h_<s)||²/(2V_s).

Before sending the next query, use remaining R_i,t=max(0,rho0−E_i,t-1) and radially clip the proposed weighted query to norm≤sqrt(2V_t R_i,t), optionally also imposing an independent public per-round cap. Debit the energy of the **actually transmitted filtered query**. This inductively guarantees sum_t ||x_it||²/(2V_t)≤rho0 for every dataset and history. Once R=0, learning remains zero while fresh noise continues. A private accumulator must not be reset or renewed without additional privacy accounting.

The background clients' filtered queries agree between target IN and OUT when the full history is held fixed: they depend only on their own unchanged inputs and that history. This dependence restriction is essential. If another client's cap or gradient reads a raw target statistic, unprotected global model or dataset-dependent roster, zeroing the target does not isolate the aggregate shift x_Tt.

## Two-direction transcript proof

For fixed target D and dummy, each same-history coalition residual is Gaussian with a common variance S_H,t and mean displacement x_Tt(D,h_<t). Its two conditional Rényi divergences of order alpha>1 are both

    alpha*||x_Tt||²/(2S_H,t)
      ≤ alpha*c_t,   c_t=||x_Tt||²/(2V_t).

Here c_t is private but predictable: for a fixed counterfactual pair it is a function of the previous reports, not the fresh round noise. It does not need to be publicly computable. The full pathwise cap holds under both transcript measures because it holds for every history, including the virtual D query evaluated along a dummy-world history.

For explicit verification, let L_t be the cumulative log likelihood ratio of D to dummy. Under the dummy law, the conditional Gaussian moment gives

    E[exp(alpha*(L_t−L_t-1)) | h_<t]
      ≤ exp(alpha*(alpha−1)*c_t).

Thus M_t=exp(alpha*L_t−alpha*(alpha−1)*sum_s≤t c_s) is a nonnegative supermartingale with initial value1. The uniform cap implies E_dummy[exp(alpha*L_T)]≤exp(alpha*(alpha−1)*rho0), and hence D_alpha(P_D||P_dummy)≤alpha*rho0. Reversing the comparison gives the same bound. This is the paper's filter/singleton-pair argument specialized to the stated Gaussian kernel; it avoids taking a per-round supremum over every possible client value and then summing incompatible worst cases.

The mechanism is consequently rho0-zCDP on each client-versus-dummy edge for every finite prefix. Standard conversion gives epsilon=rho0+2*sqrt(rho0*log(1/delta)) for 0<delta<1. A fixed finite horizon is the clean default. An infinite released stream or a private stopping-time output requires a precisely stated extension/filter stopping protocol, rather than silently equating it with finite-prefix privacy.

For two arbitrary nonempty datasets, let x_t and x'_t be their respective filtered queries along the same history. Then

    sum_t ||x_t−x'_t||²/(2V_t)
      ≤ 2 sum_t ||x_t||²/(2V_t)
        +2 sum_t ||x'_t||²/(2V_t)
      ≤4rho0.

The same martingale reasoning gives the conservative 4rho0-zCDP replacement guarantee. Do not describe rho0 dummy privacy as rho0 arbitrary replacement privacy, and do not obtain the factor by applying an invalid additive triangle inequality for zCDP.

## Observer and implementation obligations

Own coalition coins can be included in the observer view because their initial law is target-independent and the conditional proof is uniform in them. Its causal state may follow protected history and its own fixed data. Fresh unknown Gaussian innovations retain their residual variance despite that state; a server's individual inbox is excluded. Every allowed dropout or share loss must preserve the V_t lower bound and the same observation contract. Public/private noise-scale changes are distinct from private cap changes: hiding the cap does not permit changing sigma or a client's noise law privately.

The cap affects only the local query mean. Because the entire released conditional kernel is covered by the proof, its private branch does not require a separate RR/noise release. If the branch, energy trace or active-client count is separately exposed, that additional output needs accounting. A data-dependent global stopping decision or normalization by the number of active clients similarly lies outside this fixed-weight contract. Noise must continue when the learning contribution is zero.

Actual end-to-end privacy still requires public or protected initial/global history, weights, radius/noise schedules and calibration. Raw saved trajectories cannot be made private by retrospective accounting. The query used for proof must match the implemented upload, including local optimizer effects and weighted contribution; counting some internal gradient norm while releasing another unconstrained model delta is insufficient.

## Research value and fair utility comparison

The construction can spend a common public privacy cap according to a client's realized **filtered** contribution energy. It may retain low-energy clients longer or use a smaller fixed noise law coupled with a smaller enforced trajectory budget. Those are possible utility benefits, not proven ones: tight budgets can suppress important updates, and fresh noise keeps accumulating after learning has exhausted its cap.

Compare against optimally chosen public per-round clipping/noise schedules under the same coalition-conditioned budget, plus any stronger same-query release controls. Keep horizon, noise, weights, clipping semantics and approximate-DP conversion matched. Show whether private energy allocation beats public allocation after distortion and all persistent noise are included. Measuring small norms on unfiltered raw histories is useful feasibility evidence, but cannot certify a smaller budget without enforcing the restriction on every neighboring input/history. Privately estimating energy to choose a new public budget/noise schedule would introduce another mechanism.

This transfer supplies a concrete proof-feasible route to a narrower whole-client trajectory sensitivity domain. It addresses the earlier robust temporal-Gaussian obstruction by changing the admissible queries through certified filtering, rather than claiming generic correlated noise defeats that obstruction. It neither establishes a client-specific noise-distribution constructor nor demonstrates better CIA protection than metric privacy. A bounded protocol-design and fair distortion/noise-cost review is justified before implementing a sampler or training experiment.
