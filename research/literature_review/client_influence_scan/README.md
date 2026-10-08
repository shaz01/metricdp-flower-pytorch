# Client influence scan (Oct 3, 2026)

## Question

The professor's suggestion: influence can be defined in dozens of ways, so check which definitions
actually correlate with client inference (CIA) results. What does the literature already say about
which clients leak, and how to define a client's influence?

## Findings

**No work correlates client influence with client inference in FL.** The closest is Kandpal 2023,
which does it for users of fine-tuned LLMs. This was a quick scan with one round of citation
tracking, not a full review.

| Paper | Finding | For us |
|---|---|---|
| Sáinz-Pardo Díaz et al. 2025, *Metric Privacy in FL for Medical Imaging* (arXiv:2502.01352) | Metric-privacy beats DP on accuracy with similar CIA protection. | Our base paper. `../../../results/cia_frontier/metric_privacy_noise.md` explains why: it adds less noise when client models are far apart. |
| Hu et al. 2021, *Source Inference Attacks in FL* (arXiv:2109.05659); extended 2023 (arXiv:2310.00222) | The attack gets stronger as data gets more non-IID. A client leaks when its local model behaves differently from the others. | Supports "label skew raises the attack". Gives the loss-gap definition. |
| Kandpal et al. 2023, *User Inference Attacks on LLMs* (arXiv:2310.09266) | Users leak most when their data is an outlier, their examples share features, or they hold a larger share of the data. Attack success tracks the user-level fit gap (Spearman ≥ 0.994). 95% of it arrives in the first 5–18% of training. Clipping doesn't help; capping data per user does. | The closest match to our question. |
| Yu et al. 2022, *Individual Privacy Accounting for DP-SGD* (arXiv:2206.02617) | Per-example privacy depends on gradient norms over training and correlates with training loss (Pearson > 0.9). Low-accuracy groups get worse privacy. | A template for the correlation study. |
| Xue et al. 2021, *Influence of Individual Clients in FL* (doi:10.1609/aaai.v35i12.17263) | Fed-Influence: a client's effect on the final model, estimated without retraining. | A principled definition. Our pilot's "distance from average" is a one-round shortcut of it. |
| Carlini et al. 2022, *The Privacy Onion Effect* (arXiv:2206.10469) | Removing the most exposed points exposes the next layer. | Influence-based noise must track the worst client, not just the average. |
| Humphries et al. 2023, *MIA under Data Dependencies* (doi:10.1109/csf57540.2023.00013) | Correlated samples make attacks much stronger and DP much weaker. | A client's data is correlated by nature; may explain why DP didn't lower our attack. |
| Boenisch et al. 2023, *Individualized Privacy Assignment for DP-SGD* (arXiv:2303.17046) | Different budgets per example improve the privacy–utility trade-off. | The formal-DP version of per-client noise. |
| Athanasiou et al. 2026, *Protection against Source Inference Attacks in FL* (arXiv:2603.02017) | Standard DP doesn't stop source inference without big accuracy loss. | Same group as our base paper; matches what we saw. |
| Zhu et al. 2025, *FedMIA* (doi:10.1109/cvpr52734.2025.01922) | An attack using all clients' updates; robust to defenses and non-IID. | A stronger attack to test later. |
| Bai et al. 2024, *MIA and Defenses in FL: A Survey* (doi:10.1145/3704633) | Survey. | Read its client-level section. |
| Pejó & Biczók 2023, *Quality Inference in FL with Secure Aggregation* (doi:10.1109/tbdata.2023.3280406) | Per-participant quality and contribution can be inferred from aggregates. | Another way to measure contribution. |

## Influence definitions

| # | Definition | Source | New logging needed? |
|---|---|---|---|
| 1 | Client size (share of the data) | us; Kandpal | No, in the partition |
| 2 | Label skew (distance from the global class mix) | us; Kandpal | No, in the partition |
| 3 | How similar a client's examples are to each other | Kandpal | No, from the data |
| 4 | Update size before clipping | us | Partly: logged per round, not per client ID |
| 5 | How often the client gets clipped | us | Same as #4 |
| 6 | Cumulative update size over training | Yu | Same as #4 |
| 7 | Distance from the average update | us (influence pilot) | Yes |
| 8 | Agreement with the average update (cosine) | us | Yes |
| 9 | Fed-Influence | Xue | Yes, plus their estimator |
| 10 | Loss gap: the client's own model vs the others' models, on its data | Hu | Yes, and costly |
| 11 | The client's training loss | Yu | Yes |
| 12 | Fit gap: global model's loss on the client's unseen data vs unseen clients' data | Kandpal | No, already measured |

#12 is almost the attack score itself, so it can't be a predictor. Use it only to check the attack.

## Files

`papers/` has each PDF and a `.md` with its raw extracted text (not cleaned up).
