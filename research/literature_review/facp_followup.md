# FACP follow-up: primary methods recovered partially, full artifact pending

Search/read date: 2026-10-01. Source: Lei Shi, Cheng Gu, Yuqi Fan, Hailong Tang, and Yingfei Zhu, *Fisher-driven privacy preservation against category inference attacks in federated learning*, DOI [10.1016/j.hcc.2026.100399](https://doi.org/10.1016/j.hcc.2026.100399), High-Confidence Computing, available online 13 May 2026.

**Read-status upgrade:** the source is no longer abstract-only. Normal web searches surfaced substantial publisher HTML methods text, including Section 3.3.3 and its proof, Section 3.3.4, most of Section 3.3.2, and Section 4.6. **A complete PDF/HTML artifact has not been acquired.** Search-index omissions and math flattening remain material. Algorithm 1 appears as an image and was not read. Exact local Fisher Eq. (6), zero-range handling, every boundary case, complete experimental setup, and training-run accountant remain pending. This is a qualified construction precedent, not a certified privacy mechanism.

## Source-derived summary (under 200 words)

FACP targets category inference by a curious server observing uploads and auxiliary statistics (Section 3.2), rather than client participation. Clients estimate importance from squared gradients; the global profile averages participating estimates (Section 3.3.1; Eq. 7; Section 4.6). Historical norm percentiles set client bounds; global importance exponentially scales update coordinates (Section 3.3.2; Eqs. 8-13). A reverse-linear importance map sets bounded factors b; noise multipliers become lambda times sigma times b (Section 3.3.3; Eqs. 14-16). Coordinate Gaussian RDP costs are summed; lambda matches that sum to the isotropic reference (Eqs. 17-22; Proposition 1).

Section 3.3.4 explicitly excludes transmitted normalized Fisher statistics from formal DP accounting, restricting its guarantee to noisy model updates. Section 4.6 describes one importance value per parameter vector, creating a scalar-versus-vector interpretation to verify against the derivation. The preliminary local noise formula uses variance C-squared times sigma-squared divided by participating count (Section 3.1.2; Eq. 5). A full sampler/accountant cannot be reconstructed safely from these partially recovered passages.

## Our independent mathematical checks and implications

These are deductions under stated assumptions, not claims proved by the paper.

### A. The complete observed transcript needs a guarantee

If an observer receives `(noisy_update, importance_statistic)`, a DP guarantee for only the first component does not protect the pair. To see why, consider the extreme counterexample where the second component encodes a sensitive attribute exactly: arbitrarily strong noise on the first component cannot hide the second. Averaging, normalization, compression, and weak performance of one empirical classifier do not replace an analysis of that pair.

Therefore the explicitly excluded auxiliary channel must remain visible in our review. This does not prove every FACP implementation is unsafe; it establishes that we cannot cite the stated model-update guarantee as an end-to-end certificate for the described observer.

### B. Realized history-dependent clipping is not automatically sensitivity

Suppose a bound is selected from private history. A standard fixed-C argument assumes both neighboring outputs are clipped to the same C. If bounds become C(D) and C(D'), knowing only one realized value does not bound the neighboring mean difference. A future distribution constructor should use a public bound, privatized estimation with composition, or a direct analysis of the complete neighboring laws. Merely checking a returned update's norm after clipping is insufficient.

### C. Fixed-covariance Gaussian accounting and adaptive covariance differ

For equal positive-definite covariance V and neighboring mean shift h, Gaussian order-alpha RDP is

\[
D_\alpha=\frac{\alpha}{2}h^\top V^{-1}h.
\]

With diagonal V this is a sum of coordinate contributions. Summing worst-case coordinate bounds can be a conservative bound; calling it exact requires that the combined neighboring shift/geometry is actually characterized. If V changes with private data, the equal-covariance formula alone does not apply. Establishing a normalization identity for a fixed realized importance profile is not the same as proving the adaptive mechanism.

### D. Noise shares protecting an aggregate do not necessarily protect an upload

Consider k equally weighted clients, with independent local variance `C^2 sigma^2/k`. Their **average** has variance `C^2 sigma^2/k^2`. For fixed-slot add/remove adjacency and clipped norm C, the average's sensitivity is C/k, yielding RDP `alpha/(2 sigma^2)` under fixed covariance. An observer of a single upload instead sees variance `C^2 sigma^2/k`; the same fixed-slot sensitivity C yields `alpha k/(2 sigma^2)`. Replacement can introduce another factor of four in squared sensitivity when the neighboring contribution difference is bounded only by 2C.

This deduction explains why the release being accounted for matters. It does not certify the paper's normalization, which needs the complete sampler, adjacency, weights, and transcript. In particular, an aggregate interpretation cannot automatically certify an observer who sees individual uploads.

### E. Vector granularity requires vector sensitivity

If one noise/importance factor applies to an entire parameter tensor, its DP contribution depends on that tensor's Euclidean neighboring shift, not simply on treating its name as one scalar coordinate. A properly bounded block sensitivity can support blockwise Gaussian accounting; a scalar bound must not silently be applied to many unconstrained entries. Verify whether the derivation and implementation mean scalar parameters or parameter tensors before copying the formula.

### F. A fixed-round identity does not supply a training-run certificate

Even a correct one-round Gaussian bound needs composition or an adaptive transcript argument over repeated releases. Client sampling, count changes, changing clipping, auxiliary channels, and knowledge of noise shares affect that proof. The complete selected-user list can also expose participation directly to a server; our model-only peer observer remains a different game.

## Access routes and outcomes

Exact calls/queries are recorded in [followup_facp_log.json](followup_facp_log.json).

| Normal route tried | Result |
|---|---|
| DOI resolution through web open | Redirect target inaccessible; no full article acquired |
| ScienceDirect article URL through web open | Earlier HTTP 403; abstract was initially accessible through search |
| ScienceDirect standard PDF download URL | HTTP 403; downloaded response is HTML, not a PDF |
| Crossref works API | HTTP 200; author/license metadata and standard publisher TDM links |
| Publisher TDM XML link from Crossref | HTTP 200 **metadata only**; no original text or section tree |
| Same documented endpoint with FULL view | HTTP 401, API key required; stopped at that access restriction |
| Publisher TDM text/plain link | HTTP 400, invalid view; no full text |
| Exact-title/DOI/section normal web searches | Partial indexed primary methods text, including critical accounting limitation |
| Exact-title arXiv/GitHub searches | No identified author manuscript or implementation |
| ResearchGate publication listing | Says no public full text; no request/contact sent |
| Coauthor university profile surfaced by search | No verified linked manuscript from the surfaced result |

No credentials, paywall bypass, challenge solving, author contact, purchases, or unofficial shadow-library routes were used.

## What should change in the review package

1. Replace “abstract only” with **partial publisher methods read; full artifact pending** for this source.
2. Treat FACP as close prior work for client-history bounds and importance-shaped Gaussian perturbation.
3. Explicitly state the auxiliary-statistics exclusion; avoid citing it as complete-transcript DP protection.
4. Keep local-versus-aggregate accounting and scalar-versus-block interpretation on the verification queue.
5. Preserve the distinction between category inference and whole-client participation throughout CIA comparisons.

## Remaining acquisition/verification needs

Obtain a complete authorized article PDF/HTML and Algorithm 1 image; verify exact Fisher estimator and normalization, small/zero-range cases, first-round/history initialization, covariance at every observer, add/remove versus replacement adjacency, noisy/statistic release composition, and total training budget. The indexed methods identify decisive research caveats, but do not justify declaring the full paper audited or its sampler ready to implement.
