# Independent review: primer, client-noise cards, and two snowball candidates

Review date: 2026-10-01. Reviewer scope: mathematical checks of both documents; source comparison for CN1 and CN2; bounded methods screening of priority-based adaptation; primary publisher preview of FedFR-ADP. This is an AI-agent review, not human dual screening or proof certification. No reviewed files were modified.

## Findings for the editor

### 1. Primer equations are substantively consistent within their stated assumptions

No correction required to squared-weight aggregate covariance, sqrt(n) local-standard-deviation matching, equal-covariance Gaussian ROC-AUC `Phi(d/sqrt(2))`, Gaussian likelihood ratios for unequal covariances, or the second-order `0.5 trace(H Sigma)` utility approximation. The primer correctly distinguishes conditional independence, fixed versus renormalized weights, upload presence, covariance adaptation, peer side information, and record versus client adjacency.

One optional clarification: covariance matching gives a Gaussian equality **conditional on fixed updates/history/calibration parameters**. The unconditional distribution can be a mixture if covariance is data/history-dependent; the per-history equality requirement is already stated and should be preserved.

### 2. Primer has Markdown/LaTeX delimiter defects (presentation, not mathematical errors)

Many inline formulas use literal parentheses rather than math delimiters, sometimes with an unmatched backslash: e.g. Section 3 `(u_{i,t}\\in\\mathbb R^p)`, Section 4 `(C_i\\)`, Section 5 `(H\\)`, Section 7 `(m\\)` and `(m-1\\)`, Section 8 `(K\\)`. Replace these with valid `\(...\)` or `$...$` consistently. Similar notation defects occur in CN3 around `2C/|D_i|` and `2C`. This materially affects the user's ability to understand the tutorial without the papers.

### 3. CN1's probability rule and scope match the inspected source

The exponential rule with `exp(-alpha d(v,y)/2)` is explicitly printed in the source. The finite numerical universe, clipping/discretization, metric dependence, and anonymous-update assumptions are supported. Retain the warning that local value/metric privacy does not itself prove whole-client removal privacy.

Source checked: [LDP-Fed primary PDF](https://arxiv.org/pdf/2006.03637v1), Sections 2.3.1-3; physical PDF pp. 2-3.

### 4. CN2's covariance cancellation is correct and important

Algorithm 1 sets `c_n^t=c sigma^t/sigma_n^t`; Algorithm 2 ClientUpdate line 10 adds covariance `(c_n^t)^2(sigma_n^t)^2 I/N`. Therefore absolute upload covariance is indeed common within a round, although clipping and sampling schedules are personalized. Missing clients receive server Gaussian shares and the aggregate is divided by predetermined `q^t N`, rather than realized participant count. The card's concise description is faithful. Retain its trusted-server/HbC-peer distinction and do not silently claim the accountant is independently verified.

Source checked: [primary time-adaptive paper](https://arxiv.org/pdf/2502.18706v1), Section 3, Algorithms 1-2, physical pp. 4-6.

### 5. Other cards' main privacy qualifications are appropriate; source verification remains bounded

CN3's general objection to dividing final-model sensitivity by dataset size is mathematically valid: two vectors in a C-ball can differ by 2C without additional algorithm stability. CN4 correctly says support mismatch is compatible with approximate DP only when the mismatch probability fits delta, and ordinary Gaussian calibration does not automatically transfer. CN5 correctly distinguishes privately estimating a quantile from privately releasing individual uploads. CN6 correctly distinguishes per-record clipping/accounting from whole-client privacy. CN7's public-subspace/context approach must remain labeled central record-level DP. Their detailed paper extraction was not independently repeated in this review.

No urgent substantive correction identified. For card titles containing "verified," retain the scope sentence distinguishing inspected methods from certified guarantees; a cleaner label would be "methods-read source cards."

## Snowball candidate S1: priority-based adaptive FL

**Mahtab Talaei and Iman Izadi. Adaptive Differential Privacy in Federated Learning: A Priority-Based Approach.** [arXiv 2401.02453v1](https://arxiv.org/abs/2401.02453), 4 January 2024. Full primary PDF downloaded and methods read: Section III.A-D, Eqs. (4)-(6), Section IV.

This is a directly relevant **construction precedent**, not a verified CIA defense. Clients prioritize first-layer features before uploading noisy weights. The sensitivity-based priority score measures the accuracy change after perturbing one feature's connected weights. The variance-based score combines two-point variation between incoming global and locally trained weights with absolute weight magnitudes: `FI_i=sum_j Var(w_ij)*|w_ij|`. Additional noise is allocated to less important weights; the paper also discusses smaller coordinate epsilon for less important features. The base mechanism follows NbAFL and adds Gaussian uplink noise, with optional downlink noise.

Its protected unit is a changed sample, not whole-client participation. Eq. (4)'s calibration inherits the unsupported general shortcut `Delta=2C/m` for an arbitrary final local model. Importance scores also depend on private local training/accuracy, and the inspected methods do not supply a complete privacy analysis of the adaptive noise-selection law. Accuracy robustness to perturbation is not the same as a DP sensitivity bound or membership leakage direction. Experiments vary additional noise/proportion and report MNIST utility, rather than adapted CIA matched-protection comparisons.

**Decision:** include as empirical adaptive coordinate-noise predecessor with mathematical caveats. Do not treat it as a reliable accountant or exact scalar recipe. Exact Gaussian sampler/released priority statistics need code verification before reproducing it.

## Snowball candidate S2: FedFR-ADP

**Debao Wang and Shaopeng Guan. FedFR-ADP: Adaptive differential privacy with feedback regulation for robust model performance in federated learning.** Information Fusion 116, 102796, April 2025. DOI [10.1016/j.inffus.2024.102796](https://doi.org/10.1016/j.inffus.2024.102796). Primary [publisher preview](https://www.sciencedirect.com/science/article/pii/S1566253524005748) available; full methods PDF not acquired.

The primary abstract/introduction describe measuring each client's heterogeneity by Earth Mover's Distance against a global distribution, adapting Gaussian noise accordingly, and changing the privacy budget using global model error feedback. This is a close precedent for individualized distribution-driven noise calibration. However, the preview does not reveal precise estimator, released statistics, noise law, budget/composition proof, adjacency, or sensitivity. No mathematical/noise recipe should be inferred from secondary summaries. The server is discussed as honest but curious, which differs from our trusted-server/global-model-peer setting.

**Decision:** reserve / awaiting full text. Required follow-up: inspect Section 4 to see how local/global distributions are acquired and protected, whether noise shape or only scalar variance changes, how private feedback informs budget selection, and what the privacy theorem actually guarantees. Keep the online-2024/issue-2025 distinction explicit where relevant.

## Mergeable screening log for this follow-up

```json
{
  "search_date":"2026-10-01",
  "queries":[
    {"id":"COV-SQ01","query":"site.arxiv.org/abs/2401.02453","source_family":"adaptive client noise snowball"},
    {"id":"COV-SQ02","query":"\"FedFR-ADP\" Wang Guan","source_family":"adaptive client noise snowball"},
    {"id":"COV-SQ03","query":"\"Privacy-Aligned Personalized\" federated 2609.15950","source_family":"client-card identity verification"},
    {"id":"COV-SQ04","query":"\"FedFR-ADP\" \"sciencedirect\"","source_family":"adaptive client noise snowball"},
    {"id":"COV-SQ05","query":"\"FedFR-ADP\" pdf","source_family":"adaptive client noise snowball"}
  ],
  "records":[
    {"id":"COV-S01","title":"Adaptive Differential Privacy in Federated Learning: A Priority-Based Approach","url":"https://arxiv.org/abs/2401.02453","year":2024,"publication_status":"arXiv v1 preprint","decision":"include","reason":"Explicit per-client feature-priority-based noise construction, retained with adaptive accounting and inherited sensitivity caveats","read_scope":"Primary PDF methods read; no exhaustive proof or code audit","locators":["Section III.A-D","Equations 4-6","Section IV"],"query_ids":["COV-SQ01"]},
    {"id":"COV-S02","title":"FedFR-ADP: Adaptive differential privacy with feedback regulation for robust model performance in federated learning","url":"https://doi.org/10.1016/j.inffus.2024.102796","year":2025,"publication_status":"Information Fusion 116, 102796; April 2025 issue","decision":"awaiting_full_text","reason":"Close distribution-driven client Gaussian calibration predecessor; full mathematical methods not acquired","read_scope":"Primary publisher abstract, introduction and section preview only","locators":["Publisher abstract","Introduction","Section 4 preview"],"query_ids":["COV-SQ02","COV-SQ04","COV-SQ05"]}
  ]
}
```

The CN7 identity search confirmed the primary arXiv title/authors/date; it did not independently repeat the card's methods extraction and should not be counted as an extra mechanism record. Third-party discovery results were not used as evidence for either candidate's technique.
