# P6 design memo: a label-channel defense against client inference

Roadmap v3, package P6, item 1 (Agent F), 2026-10-30. **Design only.** No code and no runs. ROADMAP.md decision 6
keeps the screen (item 2) closed until the owner opens it in Section 10. Every number quoted below comes from a
committed report.

## 0. Summary

- **Why this channel.** On label-skewed EuroSAT, client inference is mostly inference of the client's label mix:
  - An attacker holding only label-matched public surrogates gets within 0.009-0.025 AUC of the attacker that holds
    real records (B5, D-1).
  - Logit adjustment removes the label prior from the local loss and cuts the AUC by 0.100 at 2x noise (B5, D-2).
    It fails at 4x, and it has no guarantee of its own.
  - Subsampling works (D-3, D-4) by diluting participation, at an accuracy ceiling of 0.746-0.772.
- **Proposal: the prior-neutral update (PN).**
  - Each client trains every round on a class-balanced batch with a fixed number `s` of records per class.
  - Its own records are used for class `c` only if a once-released, Gaussian-noised label histogram says the client
    has enough of them. Otherwise public images of class `c` fill in.
  - Full participation, server-side client-level DP-FedAvg unchanged.
- **Guarantee.** Under a Pufferfish data model (records iid within class), the client's whole transcript depends on its
  label histogram only through the noised histogram. That gives a `||n - n'||_2 / sigma_h`-GDP statement for
  label-mix secrets, plus a failure term `delta_F`, with no composition over rounds (Section 3).
  - It sits on top of DP-FedAvg's client-level guarantee and does not replace it.
- **Baselines for the screen:** RR-Label (Zheng and Li 2023) as the implementable published competitor, and the
  committed metric-privacy (A1) and composite (C1) curves.
- **Biggest open items:**
  - LDPM's full text (Section 7). The owner's institutional access is needed.
  - The public fill-in pool, which has to be disjoint from evaluation and from the attacker's surrogates.
  - A data-sharing confound: public fill-in alone is known to help non-IID accuracy.

## 1. The guarantee targeted

The roadmap names two options. This memo chooses the second, in a sharper form.

| Option | Secret protected | Status |
|---|---|---|
| LDPM's `(m, gamma, xi)`-label distributional privacy (Jiang et al., S&P 2024) | the client's label distribution | stated for one-vs-all models, in cross-silo FL; also DP. Known here from its abstract only (Section 7) |
| Client-level DP on a released label prior | the client's label histogram | chosen, as a Pufferfish statement on label-mix secrets (Section 3) |

**Why not just "DP on the prior used for logit adjustment".**
- In B5 the prior never leaves the client, so privatizing it protects nothing that is released.
- The leakage travels through the update, because the gradient reflects the label mix.
- The memo therefore targets the update's dependence on the label histogram, not the prior.

**What the guarantee is about.**
- Two secrets are in play:
  - membership (the CIA's in/out question), which DP-FedAvg's client-level `mu_total` covers;
  - the label mix, which the label-channel statement covers.
- D-1 says the label mix carries almost all of the measured CIA signal on EuroSAT. If the update no longer carries
  the label mix (beyond the noised histogram), the remaining in/out signal is within-class feature information. D-1
  bounds that at about 0.01-0.03 AUC above the surrogate attacker in v2's runs.
- **This is the hypothesis the screen tests. It is not something the guarantee implies.**

## 2. The mechanism: prior-neutral update (PN)

**Parameters:**
- `s`: records per class per local epoch;
- `sigma_h`: histogram noise;
- `tau`: safety margin;
- public pools `P_c`, one per class.

**Once, before round 1 (client `k`, local):**
1. `h~ = n(D_k) + N(0, sigma_h^2 I_C)`, where `n(D_k)` is the label histogram.
2. `A_k = {c : h~_c >= s + tau}`, the classes served from private records.

`h~` is never sent. Only its effect on the updates is released, which is why it has to be private.

**Every round `t`, for each local epoch:**
- For `c` in `A_k`: draw `s` class-`c` records of `D_k` without replacement. If the true `n_c < s` (the failure
  event `F`), take all of them and fill up from `P_c`.
- For `c` not in `A_k`: draw `s` images from `P_c`.
- Minibatch SGD on the union, `C * s` records with equal class weight, then clip.
- The server runs global DP's fixed-clipping aggregation with public denominator and noise `z C / n`, unchanged. The
  client-level `mu_total` is therefore exactly global DP's.

**Properties:**
- Full participation, so no subsampling ceiling.
- Prediction is unchanged: a plain softmax head.
- Class-balanced sampling is a resampling form of the balanced-softmax and logit-adjustment family (Menon et al.
  2020). It should keep B5's accuracy benefit (+1.1 pp at 2x).
- Unlike logit adjustment, it equalizes how much each class contributes, so absent classes stop being visible
  through missing gradient mass.

**Parameter guidance (fixed in a protocol, not here):**
- `s` near the 25th percentile of nonzero class counts in the seed-42 layout, computed from `partition_summary`.
- `sigma_h` chosen so that changing `m` labelled records costs `mu_h = 1`, for `m` in {10, 40}. One label change
  moves `n` by `sqrt 2` in L2, so `m` changes move it by at most `m sqrt 2`; hence `sigma_h = m sqrt 2`, i.e. about
  14 and 57.
- `tau = 2 sigma_h`, so `Pr[F]` per class is at most `Phi(-2)`, about 0.023. The bound is tight only when
  `n_c = s - 1`.

## 3. Formal statement to be proved

**Data model (Pufferfish; Kifer and Machanavajjhala 2014).**
- Client `k`'s class-`c` records are iid from a fixed class-conditional `P_{k,c}`.
- A secret pair is two label histograms `n, n'` of the same client, with the class-conditionals held fixed.
- The public pools are independent of `D_k`.

**Proposition (to prove).** Let `T_k` be the sequence of client `k`'s clipped updates over all rounds, given the
broadcast models. For any `n, n'`, the distributions of `T_k` under `n` and `n'` are `(G_mu, delta)`-close:
- `mu = ||n - n'||_2 / sigma_h`;
- `delta = Pr_n[F] + Pr_{n'}[F] <= 2 sum_c Phi(-tau / sigma_h)`.

The same holds for the released global models (post-processing). There is no composition over rounds.

**Proof sketch.**
1. Condition on `h~` and on not-`F`.
2. For `c` in `A_k`, `s` records drawn without replacement from `n_c >= s` iid draws of `P_{k,c}` are `s` iid draws
   of `P_{k,c}`, whatever `n_c` is. For `c` not in `A_k` the draws come from `P_c`.
3. So, given `h~`, the per-round inputs (and the client's other private randomness) have the same law under `n` and
   `n'`. `T_k` is a randomized post-processing of `h~`.
4. `h~` is the Gaussian mechanism with L2 distance `||n - n'||_2` between the two inputs, hence `G_mu` (Dong, Roth
   and Su 2022).
5. `F` adds at most its probability in total variation.

**What remains to check:**
- Whether "without replacement from iid draws" interacts with re-drawing across rounds. Each round draws afresh from
  the same `n_c` records, so the joint law over rounds depends on `n_c` (overlaps between rounds).
  - This is the main gap. It is likely closable by drawing one fixed size-`s` subset per class once, at round 0, and
    reusing it. That costs accuracy: fewer private records used.
  - The alternative is a bound on the dependence through overlap counts. The protocol must pick one. **Recommended:
    the fixed subset, because the proof is then exact.**
- Large clients: with the fixed subset, a client with `n_c >> s` uses only `s` of its class-`c` records. That is the
  price of the exact statement, and it is also what removes the size signal (B5: large clients gain least from logit
  adjustment).
- The guarantee is only as good as the data model: iid within class.

**Relation to metric privacy.** `mu` scales with a distance between histograms. Formally this is a `d`-privacy
statement over label histograms (metric privacy in the sense of Chatzikokolakis et al. 2013), applied to the label
channel only and with a client-local, data-independent noise scale. The audited paper's metric privacy instead
calibrates server noise by a data-dependent model distance (the side channel of A-4). The paper should state the
difference explicitly.

## 4. How it differs from LDPM and RR-Label

| | LDPM (Jiang et al. 2024) | RR-Label (Zheng and Li 2023) | PN (this memo) |
|---|---|---|---|
| Setting | cross-silo FL, one-vs-all classifiers (abstract) | client-targeted MI in FL under label skew | cross-device-scale (48 slots, 32 active), multi-class softmax, attacker sees global models only |
| What is perturbed | label-distribution release (details need the full text) | each training label, by randomized response | which records enter local training (class-balanced, histogram-gated, public fill-in); labels are never corrupted |
| Guarantee | `(m, gamma, xi)`-label distributional privacy, also DP | label-level local DP per record (by the usual reading; to be confirmed in the paper) | Pufferfish `G_mu` over label histograms, `mu = ||n - n'||_2 / sigma_h`, released once; on top of DP-FedAvg's client-level `mu_total` |
| Expected cost | unknown | label noise lowers accuracy as `epsilon_label` falls | fewer private records per class; public fill-in |
| Composition over rounds | unknown | none (labels perturbed once) or per round, depending on the implementation | none (histogram released once; fixed subset) |

**Novelty risk.** If LDPM's mechanism is itself a gated or balanced resampling with a released histogram, PN is not
new. That cannot be decided from the abstract. It is the first thing to check in the full text.

## 5. The screen (only after a new owner decision; cap 15 GPU-h)

As ROADMAP.md P6 item 2, made concrete for a protocol:
- **Regime:** M2's. EuroSAT alpha 0.3, randomized inclusion (32 candidates, 16 IN, 32 of 48 active), 50 rounds, equal
  weights, `R = 32`, `design_seed = 2`, so it pairs with the committed M2, A1 and C1 runs.
- **Configurations:**
  - RR-Label at one `epsilon_label` (fixed from the paper's recommended setting once its text is read), at 2
    server-noise levels;
  - PN at one `(s, sigma_h)` (`m = 10`), at the same 2 levels;
  - levels 1x and 2x (`r` 0.001546 and 0.003092), from an accuracy pilot (2 runs per level, `design_seed = 3`, never
    analyzed) with A1's replacement rule;
  - 4 configurations x 32, about 13 GPU-h at v2's 0.1 GPU-h per run, plus about 1 for the pilot. **The cap is
    tight**, so a third configuration does not fit.
- **Attackers:**
  - the gated calibrated attacker (11 checkpoints);
  - the label-only surrogate attacker (M0's surrogate sets), which is the attacker this defense is built against;
  - a defense-aware variant: shadow losses on class-balanced batches.
- **Statistic:** A1's matched-accuracy difference at 0.740 against A1's metric-privacy curve (gate), and against C1's
  composite (point estimate, 0.03 rule).
- **G-P6 (verbatim):** "At 0.740, at least 0.05 below metric privacy (decision 9's rule) and no worse than the
  composite by more than 0.03 (point estimate). Fail: the second paper does not start."
  - Not reaching 0.740 means "not testable", which is a fail for that mechanism.
- **Also reported:** lowest-client accuracy at 0.740, since class-balanced training should help rare classes;
  `mu_total`; the label-channel `mu_h` and `delta_F`.

## 6. Risks and confounds

| Risk | Effect | Handling |
|---|---|---|
| Public fill-in is data sharing (Zhao et al. 2018: a small shared global set lifts non-IID FedAvg) | an accuracy gain or leakage drop that is not the label channel | if the screen passes, a follow-up control: global DP plus the same public fill-in without balancing or gating. It does not fit in the 15 GPU-h screen |
| Public pool overlaps evaluation or the attacker's surrogates | biased accuracy or attack | the pool is a fixed public subset disjoint from the server test images and from M0's surrogate sets; the protocol checks disjointness |
| Fixed subset of `s` per class loses accuracy | PN cannot reach 0.740 | `s` from the layout's class counts; accuracy pilot; "not testable" is a fail, as written |
| Within-class signal is larger than D-1 suggests | leakage stays near metric privacy's | that is what G-P6 tests |
| The Pufferfish iid assumption | the guarantee is model-dependent | stated as such; the client-level DP-FedAvg guarantee still holds without it |
| LDPM already does this | no novelty | read the full text before any screen |

## 7. Needs from the owner

1. **LDPM's full text** (Jiang et al., S&P 2024, [10.1109/sp54263.2024.00113](https://doi.org/10.1109/sp54263.2024.00113)),
   through institutional access. Needed for Section 4's comparison and the novelty check. RR-Label's paper
   ([10.1109/tdsc.2023.3346692](https://doi.org/10.1109/tdsc.2023.3346692)) is needed too, to fix its `epsilon_label`
   and its exact perturbation.
2. **A decision on the screen** (ROADMAP.md decision 6: only after the paper packages close, by a new decision).
3. **The public fill-in pool:** which images may be treated as server-public for the mechanism, given that the
   EuroSAT test split already serves as server test set and as the attacker's surrogate source.

## 8. References

- Jiang et al. 2024, Protecting label distribution in cross-silo FL (LDPM), S&P. 10.1109/sp54263.2024.00113.
- Zheng and Li 2023, CMI: client-targeted membership inference in FL (RR-Label), TDSC. 10.1109/tdsc.2023.3346692.
- Menon et al. 2020, Long-tail learning via logit adjustment. 10.48550/arxiv.2007.07314.
- Kifer and Machanavajjhala 2014, Pufferfish: a framework for mathematical privacy definitions, TODS.
- Dong, Roth and Su 2022, Gaussian differential privacy, JRSS-B.
- Chatzikokolakis et al. 2013, Broadening the scope of differential privacy using metrics, PETS.
- Zhao et al. 2018, Federated learning with non-IID data. 10.48550/arxiv.1806.00582.
- Committed results: `reports/cia_v2_b5_large_clients.md` (D-1, D-2), `reports/cia_v2_b1_subsampling.md` (D-3),
  `reports/cia_v2_composite.md` (D-4), `reports/cia_v2_a1_matched_accuracy.md` (A-3).
