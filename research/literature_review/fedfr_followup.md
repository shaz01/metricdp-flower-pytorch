# FedFR-ADP: open-access follow-up and unresolved construction details

Date: 2026-10-01. Branch context: `feature/client-specific-noise`, after `1081ff7`. This follow-up sought a legally open methods manuscript or author code. No complete methods text was obtained. No author contact, purchase, login, or access-control workaround was attempted.

## Identity and access outcome

Debao Wang and Shaopeng Guan. **FedFR-ADP: Adaptive differential privacy with feedback regulation for robust model performance in federated learning.** *Information Fusion*, 116, 102796 (April 2025). DOI: [10.1016/j.inffus.2024.102796](https://doi.org/10.1016/j.inffus.2024.102796). Publisher PII: `S1566253524005748`.

The [publisher preview](https://www.sciencedirect.com/science/article/pii/S1566253524005748) and [author's university bibliography](https://xd.sdtbu.edu.cn/info/1113/3704.htm) confirm the identity. Crossref records the April 2025 issue; OpenAlex indexes a 2024-11-19 publication date. That index date was not independently confirmed as the first-online date. Both are before this review's cutoff.

**Read scope:** publisher abstract, highlights, introduction/contribution description and publicly exposed section snippets; official bibliographic/API metadata. **Not read:** complete Section 4, algorithms, mathematical noise rule, guarantee/accounting proofs, full experiments, or code. Do not promote this source to a full-methods card.

OpenAlex marks the article closed and reports no repository full text or best OA location. Semantic Scholar has no open PDF URL. The author faculty page contains a citation but no linked manuscript. ResearchGate explicitly reports no full text. Crossref lists Elsevier text-mining endpoints; an ordinary unauthenticated XML request succeeded but returned only bibliographic metadata with `openaccess=0`, not article sections. The plaintext endpoint returned HTTP 400. These checks establish the outcome of this search, not impossibility of obtaining an authorized copy elsewhere. Exact queries and outcomes are in [followup_fedfr_log.json](followup_fedfr_log.json).

## Source-based explanation — preview only

The paper describes a client-side adaptive Gaussian defense built from two signals. First, it measures each client's data heterogeneity using Earth Mover's Distance (EMD), comparing its distribution with a global distribution. That heterogeneity controls the Gaussian perturbation of local model updates. Second, feedback from the global model's error changes the privacy budget during training. The stated aim is to balance model performance and perturbation strength under heterogeneous data. The introduction discusses honest-but-curious servers and local privacy, while the public evaluation description reports image-classification training accuracy and MSE. These observations establish a direct precedent for heterogeneity-driven client noise and error-feedback budget adaptation. They do not reveal the EMD estimator, construction/accessibility of the global reference distribution, exact Gaussian covariance, feedback equation, privacy adjacency, or total-budget accounting. No claim of formal whole-client CIA protection or a verified implementation can be made from this preview. [Publisher abstract, highlights, introduction/contributions and exposed evaluation snippet.](https://www.sciencedirect.com/science/article/pii/S1566253524005748)

## Extraction questions that remain open

| Component | Verified from preview | Still needed from full methods |
|---|---|---|
| Noise location | Local model updates are perturbed | Final upload versus local optimization steps; clipping order |
| Noise family | Gaussian | Scalar variance versus diagonal/full covariance; sampling correlation |
| Client specificity | Heterogeneity affects client perturbation | Exact mapping, direction of adjustment, clipping/privacy coupling |
| EMD | Local distribution is compared with a global reference | Histogram/features used, ground cost, sample requirements, normalization |
| Global reference | Named as a distribution | Who estimates it, whether raw/private summaries leave clients, estimation/privacy cost |
| Feedback | Global model error influences the budget | Error's data source, private validation access, release, control law and stability |
| Privacy target | DP/LDP language and server concern appear | Record versus whole-client adjacency; upload existence/identity assumptions |
| Accounting | No complete rule visible | Multi-round composition, adaptive budget limits, private estimator protection |
| Evaluation | Classification accuracy/MSE appear | Held-out utility, membership/participation attack, confidence estimates and matched protection |
| Implementation | No author code repository verified | Author code, pseudo-code, exact hyperparameters and dataset partition protocol |

Absence from the preview means **not assessed**, rather than absent from the full paper.

## Mathematical interpretation, not the paper's implementation

For a simple illustration, consider two labels with unit transport cost between labels. A client label distribution `(0.9, 0.1)` and reference `(0.5, 0.5)` have EMD 0.4: moving 0.4 probability mass from the first label to the second makes them equal. A client with `(0.5, 0.5)` has EMD zero. A function of this distance could assign distinct client Gaussian scales. This illustrates what heterogeneity-driven calibration might mean; the paper's actual feature space, transport cost, estimator, and scale function are unknown.

EMD measures difference from a reference, not a bound on neighboring-output sensitivity. A distinctive client might create a strong participation signal, but greater heterogeneity need not imply a particular sensitivity or optimal noise variance. A scalar distance also does not identify parameter directions to protect. Thus heterogeneity-based scalar scaling and learning a noise covariance are different research objects.

If the scale is computed from private client data, even an unpublished scale can influence the observed noise distribution. If feedback depends on private validation error, that statistic and adaptive decisions must also enter the privacy analysis. Neither issue can be settled by the preview's use of DP terminology.

## Consequence for the research plan

This is a **high-priority unresolved comparator**. The broad idea of assigning Gaussian client noise according to client data heterogeneity has prior publication evidence. A novelty claim at that level is therefore unsafe. The narrower question of estimating an appropriate parameter-space distribution for whole-client participation protection still requires full-method comparison and other literature.

Proceed with the threat-game and covariance-estimation derivations independently, while keeping FedFR-ADP reserved for a precise comparator assessment. Do not invent an implementation labeled FedFR-ADP from the abstract. An authorized full paper would let the next agent resolve the table above and determine whether it contributes a baseline, a proof template, or a methodological caution. This follow-up selects no mechanism and launches no experiment.
