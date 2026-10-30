# Results handoff: CIA roadmap v2 (closed 2026-10-30)

Written 2026-10-30, when the last package (C1) closed. Roadmap v2 has no remaining package: B2 is deprioritized and B3
is on hold.

**Corrected 2026-10-30** after an audit against the committed reports and run manifests
(`research/cia_roadmap_v3/HANDOFF_AUDIT.md`, corrections C1-C8). The changes: the randomized-inclusion design (32
candidates, 1,024 trials), the power figures, the attribution of the shared-noise finding (it concerns this project's
earlier measurement, not the paper), logit adjustment's accuracy gain, and the fairness framing. Every other number
was confirmed against its source.

**Purpose.** This file is a self-contained input for:
- an external literature and novelty check, which the owner runs separately;
- the owner's choice between two papers:
  - an **audit paper**: metric privacy's reported advantage does not hold;
  - a **new-defense paper**: a mechanism that beats metric privacy against client inference.

It takes neither side. Every number is quoted from a committed report or analysis file, listed in Section 8. Nothing
here is new.

## 0. One-paragraph summary

A published federated-learning defense (Sáinz-Pardo Díaz et al., *Knowledge-Based Systems* 343:115993, 2026) scales
server-side Gaussian noise by the maximum pairwise distance between client models ("metric privacy"). It reports better
accuracy than global DP and comparable protection against client inference attacks (CIA).
- **The protection result rests on a weak estimand.** The paper's Table 13 pools 20 IN and 20 OUT checkpoint scores
  from single trajectories of a 3-client federation and reads intervals that overlap 0.5 as protection. Under a powered,
  calibrated measurement at 48 clients, global DP at any useful accuracy still leaks (AUC about 0.70, A-2). The
  paper's own setting has not yet been re-measured under standard v2.
- **At matched accuracy, metric privacy leaks like global DP** (equivalent within ±0.03 at 0.740 accuracy; no
  detectable difference at 0.660), under a measurement that is rebuilt and statistically powered.
- **A methodological finding on the way:** this project's earlier measurement shared private randomness between the
  IN and OUT worlds, which made leakage under DP look insensitive to noise (A-1).
- **Metric privacy adds a noise-level side channel** in small federations.
- **Its formal repair is never better than global DP** at a matched guarantee.

On the defense side:
- On non-IID EuroSAT, leakage is mostly **label-distribution inference**.
- **Client subsampling at `q = 0.25`** lowers the attacker's AUC at matched accuracy against global DP, by 0.10 at
  0.740 accuracy and 0.06 at 0.660.
- Combined with **logit adjustment**, it beats metric privacy by about 0.10 AUC at matched accuracy. That holds for a
  defense-aware attacker against metric privacy, and for a label-only attacker against global DP.
- **Caveats:**
  - The gain comes from subsampling.
  - The formal guarantee at these noise levels is vacuous or not computable.
- **The one non-vacuous formal regime:** a private linear head on frozen public features, useful only when public data
  is scarce (20-50 images).

## 1. Setting

### 1.1 The audited mechanism

- **Metric privacy.** Server noise per round is `sigma_t = z C / (n_t d_t)`, with:
  - `d_t` the maximum pairwise distance between client models;
  - `C` the clipping norm;
  - `n_t` the number of participants.
- **Global DP.** Client-level DP-FedAvg (McMahan et al. 2018), with noise `z C / n`.
- **Private median.** A repaired variant built in this project's earlier programme. It tracks a privately released
  median distance instead of the maximum, and has a proven GDP guarantee.
- **The paper's evidence:**
  - Table 13, 3-client Alzheimer MRI CIA: AUC vanilla 0.890, global DP 0.397, metric privacy 0.493 (DP intervals
    overlap 0.5).
  - This repository's earlier reproduction: metric privacy +6.9 pp (homogeneous) and +12.2 pp (non-IID) accuracy over
    global DP at 8 clients and `z = 0.05`; parity at 48 clients.

### 1.2 Threat model: the client inference attack

A curious participant holds a shadow sample of a target client's data, 10% of the target's records. It sees the
released global models and decides whether the target's whole dataset took part in training (client-level membership).

### 1.3 Measurement standard v2 (binding for every result below)

1. **Independent private randomness between the IN and OUT worlds:** server noise, client training seeds, and the
   inclusion of other clients. Public randomness (initial model, partition, evaluation sets) may be shared.
2. **Randomized-inclusion design**, after Steinke et al. 2023 and Andrew et al. 2023 (one-run auditing):
   - 48 client slots; 32 fixed candidate clients, of which each replicate includes 16, each candidate IN in exactly
     half the replicates; the other 16 clients always take part, so every run has 32 active clients.
   - `R = 32` replicates per configuration, giving 1,024 membership trials (32 candidates x 32 replicates).
   - Candidates: the five largest-exposure ("top-decile") clients (23, 7, 40, 18, 33) plus 27 chosen by size quantile.
3. **Primary attacker:** a per-candidate Gaussian likelihood ratio on the candidate's shadow-loss trajectory at 11
   checkpoints (rounds 0, 5, ..., 50). This is LiRA-style (Carlini et al. 2022) with difficulty calibration (Watson et
   al. 2021), fitted leave-one-replicate-out. Secondary attackers: an uncalibrated final-round loss threshold and a
   blind noise-level attacker.
4. **Metrics:** AUC and TPR at 5% and 1% FPR over (candidate, replicate) trials, final accuracy, and the lowest
   per-client accuracy (fairness).
5. **Matched-accuracy comparison:**
   - Each mechanism's AUC is a curve in mean final accuracy, linear between noise levels.
   - The comparison is `AUC_X(a*) - AUC_Y(a*)` at targets `a*` = 0.740 and 0.660.
   - The gate interval is a pair jackknife at 97.5%; a 95% interval was shown by simulation to be liberal at this `R`.
     Resamples draw the same pairs in every configuration.
   - A two-way bootstrap over replicates and candidates is reported alongside.
   - **"Beats"** means the AUC is at least 0.05 lower and the gate interval excludes 0. At `R = 32` the paired
     matched-accuracy difference is detectable at 80% power from about 0.03-0.04 (all candidates) and 0.09-0.10 (top
     decile), per A1's power analysis; unpaired comparisons need about 0.09.
6. **Pipeline checks before any run:** the analysis recovers a known Gaussian-DP trade-off on synthetic data, and a
   noise-independence regression test passes.

### 1.4 Data and models

- **EuroSAT** (main):
  - 10-class satellite images, 48 slots, label-Dirichlet alpha 0.3, layout seed 42.
  - A small CNN (289,194 parameters); 50 rounds, 5 local epochs, clip 5, equal weights.
- **Alzheimer MRI** (replication): 48 slots, alpha 1.0, 100 rounds. At alpha 0.3 the model did not train (vanilla
  accuracy 0.554).

### 1.5 Programme structure

- Track M (measurement): M0-M2.
- Track A (audit): A1-A2.
- Track B (mechanisms): B1, B4, B5, C1. B2 was deprioritized and B3 put on hold by the owner.
- An Alzheimer replication (AZ).
- Owner decisions are in `ROADMAP.md` Section 10.

## 2. Audit results (Track M and A)

### A-1. The published protection numbers came from a shared-noise artifact (M1)

- In the earlier programme's evaluation, the IN and OUT trajectories drew the same server noise and the same client
  seeds. Their difference therefore contained only the target's signal and could not shrink with noise.
- Global DP at 1x, 2x and 4x the frontier noise gave paired scores of 0.788, 0.808 and 0.792: flat.
- Repeating the experiment with only the coupling changed, at 4x: **0.516 (0.468-0.548) decoupled against 0.792
  coupled**.
- The standard v2 (Section 1.3) was built because of this. It also found that the original design (`R = 16`) could not
  resolve a 0.05 AUC difference (M0).

### A-2. Under the corrected measurement, global DP has no useful regime on EuroSAT (M2)

| Noise | `mu_total` | AUC, all (97.5%) | Top-decile AUC | Accuracy | Lowest client acc. |
|---|---|---|---|---|---|
| Vanilla | none | 0.723 (0.674-0.773) | 0.738 | 0.803 | 0.589 |
| 1x (`r` 0.001546) | 142.9 | 0.720 (0.663-0.778) | 0.741 | 0.786 | 0.573 |
| 2x | 71.5 | 0.702 (0.643-0.761) | 0.746 | 0.740 | 0.512 |
| 4x | 35.7 | 0.666 (0.602-0.731) | 0.741 | 0.660 | 0.366 |
| 8x | 17.9 | 0.671 (0.622-0.721) | 0.724 | 0.543 | 0.177 |
| 16x | 8.9 | 0.593 (0.545-0.642) | 0.603 | 0.199 | 0.003 |

- The attacker stays at about 0.70 wherever accuracy is 0.70 or above, and approaches 0.6 only near chance accuracy.
- Every level with any utility has a vacuous formal guarantee (`mu` 36-143).

### A-3. At matched accuracy, metric privacy leaks like global DP (A1; the central audit result)

| Mechanism | Subset | `a*` | Mechanism AUC | Global-DP AUC | Difference [97.5%] | Outcome |
|---|---|---|---|---|---|---|
| Metric privacy | all | 0.740 | 0.702 | 0.702 | -0.000 [-0.019, 0.018] | **equivalent** (within ±0.03) |
| Metric privacy | all | 0.660 | 0.681 | 0.666 | +0.015 [-0.016, 0.045] | no detectable difference |
| Private median | all | 0.740 | 0.696 | 0.702 | -0.006 [-0.042, 0.031] | no detectable difference |
| Private median | all | 0.660 | 0.656 | 0.666 | -0.011 [-0.051, 0.029] | no detectable difference |

- The top-decile outcomes are also "no detectable difference" or inconclusive.
- **Reading:** metric privacy's apparent accuracy advantage is a rescaling of the noise. At equal accuracy it gives the
  same empirical protection as global DP.
- **Formal guarantees:** the paper's version has none. Private median's is 2-4 times weaker than global DP's at the
  same noise.

### A-4. Metric privacy's noise level is an unaccounted release (A2 and the earlier programme's WP1)

- **Earlier programme, 3-8 client federations:** a blind attacker that estimates each round's noise level from the
  released models separates the worlds with and without the largest client at AUC 0.838 [0.779, 0.902]. The oracle
  attacker gets 0.839.
  - Part of this is generic: global DP's change in update size gives 0.763.
  - It fades with federation size: metric privacy 0.891 and 0.795 at 8 clients; per-target noise shift below 1.1% at
    48 clients.
- **v2, A2, two small Alzheimer federations:** noise-level AUC 1.000 and 0.875 at noise strong enough to weaken the loss
  attacker.
- **The strong claim failed.** It needed two qualifying cells and got one, and that cell's global-DP condition held
  only through an attacker artifact.
- **Status:** real but secondary. In small federations the participant count is visible under either mechanism unless
  the denominator and noise are fixed and public.

### A-5. Repairing metric privacy removes its advantage (earlier programme, WP3; proof)

- Replacing the maximum distance by a privately tracked median gives a proven
  `sqrt(T (q_max^2/z^2 + 1/sigma_b^2))`-GDP guarantee, with the median release at 10% of the budget.
- This closes the side channel by construction.
- Because the guarantee rests on a public noise floor, **at a matched formal guarantee the repaired mechanism never
  adds less noise than global DP**.

### A-6. Replication on Alzheimer MRI (AZ)

- Leakage is near chance even without noise: vanilla AUC 0.564.
- M2's gate finds a "useful regime" at `r = 0.0025`, but only because nothing leaks.
- A1: no robust difference. Metric privacy shows a nominal -0.051 [-0.100, -0.002] at 0.784 accuracy, but the two-way
  bootstrap includes 0.
- **Reading:** in a low-leakage setting no mechanism is measurably better. Whether a useful regime exists depends on
  how strongly client data is skewed.

## 3. Defense results (Track B and C1)

### D-1. EuroSAT's leakage is mostly label-distribution inference (B5 diagnostic)

An attacker with label-matched public surrogates, holding no candidate records and only the candidate's label mix, gets
within 0.009-0.025 AUC of the attacker with the true shadow sample.

### D-2. Logit adjustment (B5)

- Local loss `CE(softmax(f(x) + tau log pi_client))` with `tau = 1` (Menon et al. 2020; Zhang et al. 2022, FedLC). It
  removes the client's label prior from local training.
- **At 2x noise:** -0.100 [-0.152, -0.049] AUC at the same noise, with +1.1 pp accuracy. **Passes.**
- **At 4x:** fails.
- **Contribution caps for large clients:** fail.

### D-3. Client subsampling with amplification (B1)

- Poisson sampling with rate `q` per round, release denominator `q N`, noise `r C` on the average.
- **Matched accuracy against global DP:**

| `q` | `a*` = 0.740 | `a*` = 0.660 | Verdict |
|---|---|---|---|
| **0.25** | **-0.102 [-0.164, -0.040]** | **-0.059 [-0.096, -0.022]** | **passes** |
| 0.5 | -0.036 [-0.064, -0.008] | -0.023 [-0.054, 0.009] | fails (below 0.05) |

- **The reduction comes from sampling, not noise:** at `q = 0.25` the AUC stays at 0.585-0.617 over a 60-fold range of
  `r`.
- **It caps accuracy:** at most 0.746, against about 0.80 at full participation.
- **The formal guarantee is not better:**
  - The amplified `mu` is mostly not computable: the noise multiplier is below the accountant's minimum of 0.05. At such
    multipliers any accountant gives a vacuous bound, so "not computable" means vacuous, not unknown.
  - Where it computes, it is larger than global DP's at the same `r` (29.7 against 17.9).

### D-4. The composite: subsampling with logit adjustment (C1)

- **Configuration:** `q = 0.25` with `tau = 1`, at 4 noise levels, 128 runs.

**Against the paper's metric privacy at matched accuracy (gate):**

| `a*` | Composite AUC | Metric privacy AUC | Difference [97.5%] | Two-way [95%] | Verdict |
|---|---|---|---|---|---|
| 0.740 | 0.606 | 0.702 | **-0.095 [-0.137, -0.053]** | [-0.154, -0.036] | **passes** |
| 0.660 | 0.573 | 0.681 | **-0.108 [-0.163, -0.053]** | [-0.183, -0.035] | **passes** |

**Reported comparisons:**

| Composite against | `a*` = 0.740 | `a*` = 0.660 |
|---|---|---|
| Private median | -0.090 [-0.139, -0.041] | -0.083 [-0.131, -0.034] |
| Global DP | -0.096 [-0.148, -0.043] | -0.093 [-0.157, -0.030] |
| **Subsampling alone, `q = 0.25`** | **+0.006 [-0.048, 0.061]** | **-0.034 [-0.084, 0.016]** |
| Metric privacy, top decile | -0.050 [-0.138, 0.037] | -0.055 [-0.141, 0.030] |

**Adaptive attackers:**
- **Defense-aware** (it scores the logit-adjusted loss the clients trained on): -0.096 [-0.136, -0.057] and -0.094
  [-0.143, -0.044] against metric privacy. No leakage recovered.
- **Label-only surrogates**, against global DP under the same attacker: -0.093 [-0.134, -0.053] and -0.066
  [-0.122, -0.010].

**Readings:**
- **Logit adjustment** adds 2.5-6.0 pp accuracy at the same noise in the main runs (3-10 pp in the 2-run pilot) and
  lifts the best reachable accuracy to 0.772. At
  matched accuracy it adds no measurable privacy over subsampling alone.
- **Cost in fairness:** the lowest client accuracy is 0.33 near the 0.660 target. At matched accuracy that is 4-5 pp
  below metric privacy and global DP (0.36-0.37 at 0.660; point estimates,
  `research/cia_roadmap_v3/tables/fairness_matched_accuracy.csv`).
- **Formal guarantee:** not computable at 3 of 4 levels; 37.1 at the noisiest.

### D-5. Negative results that bound the defense space

- **Earlier programme, WP2:** none of ten local-training variants cut leakage by 0.05 within 1 pp accuracy (FedProx,
  consensus KL, fixed local steps, weight decay, one local epoch). The one lead (consensus KL) did not replicate.
- **Earlier programme, WP6:** residual clipping lowered leakage only through 1.41-1.70x more noise, not through lower
  sensitivity.
- **Earlier programme, closed 2026-10-09:** client-specific noise shapes, non-Gaussian densities and influence-directed
  noise gave nothing.
- **Earlier programme, WP3:** a partial-knowledge ("crowd") bound is 0.55-0.62x the GDP bound, but only where both are
  vacuous.
- **v2:** contribution caps (B5) and subsampling at `q = 0.5` (B1) fail.

## 4. Formal regime (B4 and its confirmation)

- The full network allows no non-vacuous client-level guarantee. At 48 clients and 50 rounds, `mu_total = 1` needs a
  noise norm of about 396 against a signal norm of at most 5.
- **B4 shares only a linear head** (5,130 parameters) on frozen public ResNet-18 features.
  - **Pre-registered gate failed:** with a 1,350-image public set, the DP head gains at most 3.49 pp over the
    public-only head at `mu_total <= 8`.
  - **Confirmed positive** (new protocol, fresh layout 44, 13 of 13 settings): with **20 public images the DP head at
    `mu <= 2` beats the strongest public-only baseline by 14-26 pp, and with 50 images by 10-14 pp**.
- **CIA was not measured in this regime** against metric privacy; no package compared mechanisms there. B4's report
  records a realistic attacker at an implied `mu` of 0.09-0.41.

## 5. Two possible papers: what each has and lacks

### 5.1 Audit paper

**Thesis.** Metric privacy's reported advantage against client inference does not hold under a powered measurement:
- At matched accuracy it leaks like global DP (A-3).
- Its published protection rests on an estimand that pools checkpoints of single trajectories in a 3-client
  federation (Section 0); the paper's own setting still needs re-measuring under standard v2.
- Measuring paired worlds with shared private randomness makes DP leakage look insensitive to noise (A-1, a pitfall
  found in this project's earlier measurement).
- It adds a noise-level side channel in small federations (A-4).
- Its formal repair is never better than global DP (A-5).

The defense results supply a positive ending: a standard tool, client subsampling, does beat it (D-3, D-4).

**It has:**
- A powered, pre-registered measurement standard.
- An equivalence result (A-3).
- A proof (A-5).
- A replication that bounds the claim (A-6).

**It lacks or needs checking:**
- Whether the shared-noise pitfall, or this audit of this paper, has already been published (Section 6, Q1-Q3).
- A second non-IID dataset with real leakage: Alzheimer has none.
- The paper's own 3-client setting, measured under standard v2. A1 runs at 48 slots.

### 5.2 New-defense paper

**Candidate thesis.** On label-skewed federations, client inference is mostly label-distribution inference. Client
subsampling, optionally with logit adjustment for accuracy, lowers it by about 0.10 AUC at matched accuracy against
metric privacy, private median and global DP, and holds against defense-aware and label-only attackers.

**It has:**
- Gated, matched-accuracy wins against three baselines.
- Two adaptive attackers.
- The label-channel diagnostic.

**Weaknesses a reviewer will raise:**
1. **Subsampling and amplification are standard** (Balle et al. 2018; McMahan et al. 2018). The new part would be the
   empirical matched-accuracy finding against client inference and the label-channel explanation, not the mechanism.
2. **The privacy gain does not come from the new ingredient:** logit adjustment adds accuracy, not privacy (D-4).
3. **The formal guarantee is vacuous or not computable.** Amplification assumes the sample stays secret.
4. **One dataset, one layout,** and an accuracy cap (0.772 at best against 0.803 vanilla).
5. **Fairness:** the lowest client accuracy drops to 0.33, which is 4-9 pp below the full-participation mechanisms
   at matched accuracy.

A genuinely new defense would need a mechanism that targets the label channel directly and with a guarantee, for
example a privatized label-prior release. No such mechanism was built.

### 5.3 Hybrid

An audit paper whose last section shows that subsampling beats the audited mechanism at matched accuracy. This uses
every result without claiming a new mechanism.

## 6. Questions for the literature and novelty check

| # | Question | Why it matters | Search terms |
|---|---|---|---|
| Q1 | Has metric privacy, or distance-calibrated noise in FL, been audited or critiqued, especially the Sáinz-Pardo Díaz et al. line? | Decides whether the audit is new | metric privacy federated learning; d-privacy FL; distance-calibrated noise; Sáinz-Pardo Díaz |
| Q2 | Is the shared-noise / coupled-randomness evaluation pitfall in membership or client inference documented? | A-1's novelty; it may be a known instance of Aerni et al. 2024 | membership inference evaluation pitfalls; paired worlds shared randomness; DP auditing coupling |
| Q3 | Are matched-accuracy comparisons of DP variants against MIA or CIA standard, and has anyone shown a "better utility" DP variant to be noise rescaling? | Framing of A-3 | privacy-utility frontier matched accuracy; DP variants empirical comparison; relaxed DP critique |
| Q4 | Is user- or client-level inference on label-skewed FL known to be mostly label-distribution inference? | D-1; Wainakh et al. 2022 is close | user-level label leakage; property inference label distribution FL; client inference non-IID |
| Q5 | Has client subsampling been evaluated empirically as a defense against client inference at matched accuracy, beyond its formal amplification? | Novelty of D-3 and D-4 | client sampling membership inference FL; participation rate privacy empirical; amplification by sampling attack |
| Q6 | Has logit adjustment or label-prior calibration been studied as a privacy defense, not only for label skew? | D-2 | logit adjustment privacy; FedLC membership inference; label-prior debiasing privacy |
| Q7 | Data-dependent noise levels as side channels in DP-FL (noise-scale leakage)? | A-4 | data-dependent noise side channel; adaptive clipping privacy leakage; noise multiplier leakage |
| Q8 | DP heads on frozen public features with scarce public data (20-50 images): known? | B4's positive; Tramèr and Boneh 2021 is the obvious prior | DP linear probe few public samples; DP transfer learning scarce public data |
| Q9 | One-run / randomized-inclusion auditing applied to client inference in FL | Method positioning (Steinke et al. 2023; Andrew et al. 2023) | one-run auditing federated; client-level DP auditing |

Each question should come back as one of three answers: already published (cite it), partially (state the difference),
or not found.

## 7. Threats to validity

- **Attacker strength.** The primary attacker is a calibrated loss attacker, with two adaptive variants in C1 only.
  Gradient- or update-based attackers that observe individual client updates are out of scope: the threat model sees
  only global models.
- **Scale.** 48 slots, 32 or 48 active clients, 50 rounds, one small CNN.
- **Data.** EuroSAT carries the leakage results. Alzheimer leaks too little to discriminate. There is one layout (42)
  for the CIA results; B4's confirmation used layout 44.
- **Formal privacy.** Every empirically useful setting has a vacuous client-level guarantee (`mu` 36-143, or not
  computable). Claims are empirical protection against the stated attacker, not DP guarantees.
- **Power.** Paired matched-accuracy differences below about 0.03-0.04 (all candidates) or 0.09-0.10 (top decile)
  cannot be detected at `R = 32`; unpaired comparisons need about 0.09.
- **Earlier numbers.** Numbers from the earlier programme (2026-10) that involve DP leakage are labelled "paired, shared
  noise". The audit should cite only v2 measurements for leakage under DP.

## 8. Artifact index (repository paths)

| Result | Report | Data and analysis |
|---|---|---|
| M0 measurement standard | `reports/cia_v2_m0_measurement.md` | `results/cia_v2/measurement/`, `results/cia_v2/randomized_inclusion/` |
| A-1 shared noise | `reports/cia_v2_m1_shared_noise.md` | `results/cia_v2/measurement/m1/` |
| A-2 global-DP frontier | `reports/cia_v2_m2_frontier.md` | `results/cia_v2/frontier/` |
| A-3 matched accuracy | `reports/cia_v2_a1_matched_accuracy.md` | `results/cia_v2/frontier/a1/` |
| A-4 side channel | `reports/cia_v2_a2_side_channel.md`; earlier `results/cia_frontier/side_channel_audit/FINDINGS.md` | `results/cia_v2/side_channel/` |
| A-5 formal repair | `research/cia_review_2026-10/theory/FINDINGS.md` | same folder |
| A-6 Alzheimer | `reports/cia_v2_alzheimer_replication.md` | `results/cia_v2/alzheimer/` |
| D-1, D-2 logit adjustment, caps | `reports/cia_v2_b5_large_clients.md` | `results/cia_v2/mechanisms/b5_large_clients/` |
| D-3 subsampling | `reports/cia_v2_b1_subsampling.md` | `results/cia_v2/mechanisms/b1_subsampling/` |
| D-4 composite | `reports/cia_v2_composite.md` | `results/cia_v2/composite/` |
| Formal regime | `reports/cia_v2_formal_regime.md`, `reports/cia_v2_formal_confirm.md` | `results/cia_v2/formal_regime/` (confirmation in `confirm/`) |
| Earlier programme | `research/cia_review_2026-10/SYNTHESIS.md` | `results/cia_frontier/` |
| Owner decisions | `research/cia_roadmap_v2/ROADMAP.md` Section 10 | |

Code:
- Mechanisms: `metricdp_pytorch/` (`subsampled_dp_strategy.py`, the metric-privacy and private-median strategies).
- Runner and attackers: `results/cia_v2/randomized_inclusion/runner.py` and `analysis.py`.
- Matched-accuracy statistic: `results/cia_v2/frontier/a1/analyze.py`.

**Compute:** about 190 GPU-h for roadmap v2, including C1's 11.5. The earlier programme took about 156
trajectory-hours.

## 9. References already verified (OpenAlex, 2026-10-16)

- Aerni, Zhang and Tramèr 2024, evaluation pitfalls of privacy defenses (10.48550/arxiv.2404.17399).
- Watson et al. 2021, difficulty calibration in membership inference (10.48550/arxiv.2111.08440).
- Zanella-Béguelin et al. 2022, Bayesian estimation of DP from attacks (10.48550/arxiv.2206.05199).
- Ponomareva et al. 2023, practical DP regimes (10.48550/arxiv.2303.00654).
- Wainakh et al. 2022, user-level label leakage in FL (10.48550/arxiv.2105.09369).
- Menon et al. 2020, logit adjustment (10.48550/arxiv.2007.07314).
- Zhang et al. 2022, FL with label-distribution skew via logits calibration (10.48550/arxiv.2209.00189).
- From the earlier corpus: Steinke et al. 2023; Andrew et al. 2023; Carlini et al. 2022 (LiRA); Balle et al. 2018; McMahan
  et al. 2018. There is a 97-reference review in `research/cia_review_2026-10/literature_review.md`, with BibTeX in
  `references.bib`.
