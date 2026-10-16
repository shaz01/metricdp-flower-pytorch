# CIA research programme: synthesis (2026-10-16)

The end-of-programme step of [agent_roadmap.md](agent_roadmap.md) Section 6: read every package's `FINDINGS.md`, say
what the programme established, and list what is left for the owner. Written on `feature/cia-synthesis` at the owner's
request ("move to the next step") after WP0-WP3, WP6 and the consensus-KL follow-up were merged into `master`. It
adds no runs and no new numbers: every figure below is quoted from a committed `FINDINGS.md`, `STATE.md` or analysis
file, cited per line. Not a `reports/` write-up (agents do not write there; the owner decides on one).

## 1. Where every direction ended

| Direction (role) | Package, gate | Outcome | Source |
|---|---|---|---|
| D2 defense-aware audit (primary evaluation) | WP1, G1(a) **pass**, G1(b) **fail** | Metric privacy's noise level reveals the largest client in 3-8 client federations; the increment attacker does not beat the loss statistic | `results/cia_frontier/side_channel_audit/FINDINGS.md` |
| D1 signature-suppressing local training (primary mechanism) | WP2, G2 **inconclusive**, owner: D1 stops | No local-training change cut leakage by 0.05 within 1 pp accuracy; the one lead (consensus KL 0.3) did not replicate | `results/cia_frontier/signature_suppression/FINDINGS.md`, `results/cia_frontier/consensus_kl/FINDINGS.md` |
| D3 partial-knowledge theory | WP3, G3 **pass** | Crowd bound 0.55-0.62x the GDP bound, but only where both are vacuous | `research/cia_review_2026-10/theory/FINDINGS.md` |
| D4 item 1, formal metric privacy | WP3, G3 **pass** | Proven guarantee, side channel closed; at a matched guarantee it never uses less noise than global DP | same |
| D4 item 2, residual clipping | WP6, G6 **pass as written** | Leakage fell, but through 1.41-1.70x more noise, not lower sensitivity | `results/cia_frontier/residual_clipping/FINDINGS.md` |
| D5 temporal noise allocation | on hold | Unlock condition (WP4 finished) cannot be met; owner may assign | roadmap Section 2 |
| D6 low-dimensional sharing | on hold | Needs a useful D3 bound in the `d ~ n` regime; WP3 computed the full-model regime only | roadmap Section 2 |
| WP4 (D1 Stage B), WP5 (Alzheimer replication) | not started | Their gates (G2, then G4) were never reached | roadmap Section 3 |

Infrastructure (WP0, G0 pass) and two bug fixes it found (MPS asynchronous batch copies corrupting training; the
EuroSAT frontier runner rejecting `dirichlet_alpha`) are in `results/cia_frontier/cia_infra/STATE.md`.

Compute: WP1 98.2 h (304 trajectories), WP2 34.5 h (132), WP6 7.08 h (24), consensus-KL follow-up 16.51 h (54); WP0
and WP3 ran no training. About 156 trajectory-hours in total.

## 2. What the evidence supports

The programme's target contribution (research_directions.md Section 0) was: *metric-privacy calibration is equivalent
to global DP once leakage is matched, its data-dependent noise level is an unaccounted release, and a client-level DP
mechanism with a fixed public noise law plus signature-suppressing local training reaches lower defense-aware CIA risk at
equal accuracy.* The first two parts are supported; the third is not.

**2.1 Metric privacy's noise level is an unaccounted release, and it leaks.** In 3-8 client federations a blind
attacker who estimates each round's noise level from the released models separates the worlds with and without the
largest client at AUC 0.838 [0.779, 0.902]; the oracle with the true noise level gives 0.839, so the attack reads the
noise level itself (WP1 Claim 1). Part of the separation is generic: under global DP the change in the average update's
size still gives 0.763 [0.706, 0.824], strong at 3 clients (0.982 EuroSAT, 0.983 Alzheimer) and fading at 8 (0.690,
0.531), while metric privacy stays at 0.891 and 0.795 at 8 clients. At 48 clients the channel is weak (per-target noise
shift below 1.1%, `project_state.md`). Practical reading: in small federations the participant count and the largest
client's presence are visible to any participant, under either mechanism, unless the denominator and noise are fixed
and public.

**2.2 Repairing metric privacy removes both its leak and its advantage.** Replacing the maximum pairwise distance by a
privately tracked median distance gives a proven `sqrt(T (q_max^2/z^2 + 1/sigma_b^2))`-GDP guarantee with the median
release at 10% of the budget, and closes the noise-level channel by construction (WP3 Claim 2). The guarantee comes from
a public noise floor, so **at a matched formal guarantee the repaired mechanism never adds less noise than global DP**.
Combined with the earlier finding that metric calibration behaves like global DP once leakage is matched
(`project_state.md`), metric calibration offers no formal or empirical privacy-utility advantage that the programme
could find; its reported advantage is a rescaling of the noise.

**2.3 An attacker who does not know the other clients is weaker only at vacuous noise.** For the loss attacker the
crowd bound is 0.546x (8 EuroSAT clients) and 0.619x (48 clients) the GDP bound at the frontier noise, but both bounds
are vacuous there (GDP `mu` 572 and 95 over 50 rounds); the ratio is 0.817 at 2x and 0.924 at 4x that noise and
extrapolates to 1.000 at the noise a `mu = 1` guarantee needs (WP3 Claim 1). This explains why realistic attackers sit
far below the GDP bound at practical noise; it is not a usable guarantee.

**2.4 Local training does not suppress client signatures in this setting.** On EuroSAT with 48 clients and no DP, none
of ten local-training variants (FedProx, consensus KL, fixed local steps, weight decay, one local epoch) cut the
clean-view score by 0.05 within 1 pp accuracy (WP2). The seed-42 lead, consensus KL 0.3 (reduction 0.148 [0.048,
0.224]), reversed at seeds 43-45: over four seeds it changes the score by -0.033 [-0.112, +0.048] at -2.6 pp accuracy,
and under global DP it costs accuracy at every noise level without a significant leakage reduction (consensus-KL
FINDINGS). Fixed local steps, the other open candidate, gives -0.001 [-0.079, +0.076] over four seeds. The first-order
mechanism behind D1 did not show up either: FedProx shrinks the leave-one-out residual norm most without a matching drop
in leakage (WP2 Further observations).

**2.5 Around the frontier noise, measured leakage barely responds to noise.** In the consensus-KL step 3, global DP at
1, 2 and 4 times the frontier noise gives clean-view scores 0.788, 0.808 and 0.792 (five targets of mixed size) while
accuracy falls from 0.791 to 0.759 and 0.657. This matches the earlier frontier observation that at the landed noise
neither defense leaked less than vanilla (`project_state.md`). WP6 saw top-decile leakage fall with 1.41-1.70x noise at
no accuracy cost (0.936 to 0.892 and 0.844); the target sets differ (WP6: five largest clients; step 3: two largest plus
three smaller), and step 3's top-decile score also drops at 4x (0.92, 0.94, 0.84), so the two are compatible: the
effect of noise in this range is small, target-dependent and visible mainly for the largest clients.

**2.6 Methodological lessons.** (i) Five targets at one seed resolve only large effects: a single seed moved vanilla's
score from 0.816 (seed 42) to 0.762 (seeds 42-45), and the consensus-KL reversal shows that a seed-42 interval excluding
zero can still be a seed artefact. Clusters are target-seed pairs, so more seeds narrow intervals only when targets
agree. (ii) A matched-leakage comparison needs leakage to move with the knob being matched; step 3's pre-declared rule
returned "supports" on a flat score curve, which is why its FINDINGS reports the rule's output and its failure
separately. (iii) The noisy-view shadow attack is not a stable ranking of local-training variants (WP2: 0.080 to 0.796
across configurations at seed 42).

## 3. What did not hold up, stated plainly

- D1 as a mechanism: not supported on EuroSAT at 48 clients (WP2 plus follow-up). WP4 and WP5 therefore never ran, and
  nothing here speaks to Alzheimer at 48 clients.
- Residual clipping as a sensitivity reduction: the centre carries the previous round's noise, so residuals (median
  6.58-7.72) and the private `rho` (6.69, 8.09) settle above C = 5; the noise feedback is absorbed only when
  `1/z > sqrt(d)/n` (11.2 here), i.e. only at vacuous guarantees (WP6 Claim 2 and `THEORY.md`).
- The increment attacker as the primary attacker: unstable at low FPR with two or three reference federations per world
  (WP1 Claim 2).

## 4. Open items for the owner

1. **WP6 control (recommended if WP6's result is to be cited).** Plain clipping at the same `z` as WP6's plain arm
   (0.0742, so the same guarantee) with C = 7.05 and 8.53 (1.41 and 1.70 times C = 5) reproduces the residual variants'
   noise std. If it shows the same leakage drop,
   residual clipping adds nothing beyond a larger clipping norm. 2 configurations x 6 trajectories, about 3.5 t-h; needs a
   protocol amendment on a new branch since WP6 is merged.
2. **D5 and D6.** Both are on hold by the roadmap. Section 2.5 weighs against D5 (moving noise between rounds is unlikely
   to matter when 1-4x more noise barely moves the score). D6 would need a new D3 calculation for a small head; the
   research-direction numbers (GMIP/GDP noise ratio 3.3-5.6 for a 51-132 parameter head at 8 clients) are the only
   evidence so far.
3. **A `reports/` write-up and the paper framing.** What the evidence supports is an audit-and-repair paper: the
   noise-level side channel (2.1), the formal repair and its matched-guarantee equivalence to global DP (2.2), the
   crowd-bound explanation (2.3), and the negative results on local training (2.4) and residual clipping (Section 3).
4. **The roadmap's open questions** (research_directions.md Section 6) now have partial answers: if the participant count
   is public, small-federation CIA is close to trivial under either mechanism (2.1); equal weights with a public slot
   count is the setting every package used; "worst case" targets were the largest clients, which are also where noise
   helps most (2.5).

## 5. Candidate directions for the next round (proposals, not vetted)

Proposed 2026-10-16 from the evidence above, as input for choosing new directions; no protocol, gate or literature check
yet. Costs are rough trajectory-hour estimates at the measured 0.27-0.42 h per 50-round EuroSAT trajectory.

| # | Direction | Why the evidence points here | First step and rough cost |
|---|---|---|---|
| P1 | **Where leakage starts to fall**: the client-level leakage-accuracy curve at much higher noise | 1-4x the frontier noise left the clean-view score at 0.79-0.81 while accuracy fell 13 pp (2.5); every defense claim needs the operating point where leakage reaches chance, compared with the D3 bounds | Global DP at about 8, 16, 32, 64x the frontier noise, WP2 federation and targets; 15-25 t-h |
| P2 | **Client subsampling** (privacy amplification by sampling clients per round) | Every programme run had all 48 clients in every round, and the code has no client-sampling option; sampling is the standard deployment of client-level DP and reduces the rounds in which an attacker sees the target | Add per-round client sampling with accounting, then a Stage A style screen; about 15 t-h plus code in Agent B's files |
| P3 | **Why some clients leak under every mechanism** | The largest clients leak most everywhere, and client 40 stays at 0.98-1.00 under every WP6 mechanism (2.5, Section 3) | Characterise high-leakage clients from committed logs (label mix, size, leave-one-out residual norm), no training; then a targeted fix such as splitting large clients into pseudo-clients or capping records per round |
| P4 | **Noise-level side channels in other adaptive DP mechanisms** | The side channel (2.1) is the programme's strongest positive result; adaptive clipping by quantile tracking and other data-dependent noise rules may release the same kind of signal if part of it is unaccounted | Reuse WP1's attack code on adaptive clipping in 3-8 client federations; 30-60 t-h |
| P5 | **Public-validation step-size gate on the 48-client frontier** | The earlier client-specific-noise work left one transferable piece, a post-processing gate that nearly doubled utility at risk 0.80 (`STATUS.md`, closed direction); it changes no guarantee | Global DP with and without the gate at two noise levels; about 10 t-h |
| P6 | **A stronger reference-based attacker** | The increment attacker failed partly because it had only two or three reference federations per world (WP1 Claim 2) | More reference federations per world in small federations; expensive (each reference is a full trajectory) |

D5 and D6 (roadmap, on hold) remain: Section 2.5 weighs against D5; D6 first needs a D3 calculation for a small head.

## 6. Package index

| Package | Directory | Status |
|---|---|---|
| WP0 | `results/cia_frontier/cia_infra/` | ready (G0 pass) |
| WP1 | `results/cia_frontier/side_channel_audit/` | passed (a), failed (b) |
| WP2 | `results/cia_frontier/signature_suppression/` | inconclusive; D1 stopped |
| WP3 | `research/cia_review_2026-10/theory/` | passed |
| WP6 | `results/cia_frontier/residual_clipping/` | passed as written |
| Consensus-KL follow-up | `results/cia_frontier/consensus_kl/` | closed by owner decision |
