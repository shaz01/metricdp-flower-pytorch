# Project Status

**Branch:** `feature/client-specific-noise`
**Last updated:** 2026-10-08, frozen-constructor fresh-data confirmation partial (2/3); no jobs running
(see `git log` for anything more recent)

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog. Exception: the Active work
section (including the Currently running table) updates more often, at "worth a commit"
granularity — see `AGENTS.md`'s "Working across machines" section.

## Active work

**Working branch:** `feature/client-specific-noise`. The owner requested a separate branch for
this research. The initial audit and literature-review version 1 were already committed on
`master`; further research and implementation stay on this branch until the owner declares
the work finished. No experiments were launched during the branch transition.

**Latest frozen-constructor chunk (2026-10-08, owner: "move autonomously until you find a solution"):** [findings](research/proposals/2026-10-08_frozen_constructor_findings.md), [protocol](research/proposals/2026-10-08_frozen_constructor_protocol.md). Pooled freeze with extended eta grid (frozen: target_balanced, d=51, cap .003, eta 30/100/100 for q.55/.65/.80; none at the new edge). Fresh Fashion-MNIST test-split confirmation (3 new public 32-sets, 1 new 2,048-image cohort, 1,856 held-out images, 512 draws, no re-tuning): q.65/.80 beat both the strongest public control and the zero-private offset on 2 of 3 subsets (+.014 to +.018 CE); subset 2 loses to its strong public control (−.006 to −.008) though it beats the zero-private offset by .026-.028, because the construction replaces rather than adds to the public step. q.55 fails. Attack AUC matches calibrated targets (mean within .003). Pre-stated solution criterion NOT met. Independent audit 113 checks. Test split now consumed for this constructor; next steps are adaptive/development-only (stacked public+private construction, cross-dataset transfer). No jobs running.

**Preceding limited-public chunk (2026-10-08):** [findings](research/proposals/2026-10-08_limited_public_findings.md), [protocol](research/proposals/2026-10-08_limited_public_protocol.md). Development-only decomposition of saved headroom queries into public projection, per-client clipping and peer-contract Gaussian noise (q.55/.65/.80), 18 cells, selection-only tuning. At 32 public examples the noisy gain beats both the strongest public control and the zero-private public offset by >.001 CE in both cohorts at q.65/.80 for all 3 subsets (mean gain over zero-private offset .0089/.0159); fails at q.55 (.0008). 128 passes for subsets 42/44 only; 512 fails (private-signal arms .0017-.0020 CE worse than the control). Clipping is not a loss; noise is the dominant cost. Caveats: eta=30 grid edge selected in 12-13/18 noisy cases, reused development halves, no CIA run at the selected configs, no reserve used (136 unused). Independent audit 1,278 checks/max 2.2e-16; 305 passed/5 deselected. Next proposed: frozen-32 constructor under the peer-conditioned CIA contract with matched-AUC Gaussian vs radial controls on new cohort evidence; decide separately on extending the eta grid. No defense/new-density/overall-completion claim. No jobs running.

**Preceding development-only headroom chunk (2026-10-08):** [findings/roadmap](research/proposals/2026-10-08_headroom_findings.md).2775models, public32/128/512nestedsettings, two record-disjoint2048privatecohorts, OLDdev512selection/512assessment.32unprotectedone-stepgains.015531–.036550CEbothcohorts/3publicsubsets;512selection-frozenoracle/one-stepgatefails, butrefinement helpsA morethanB. Centering leaves fixed-class within-class covariance unchanged. Independent224,895checks/max3.20e-14/allmodelsretrained;302passed/5deselected. Noreserveaccess;136unused. Nextproposed limited-public projection/clipping/noise feasibility with512anchor andstrongcontrols; no successfuldefense/newdensity/privateconfirmation/overallcompletionclaim. Nojobsrunning.

**Latest bounded class-conditional chunk (2026-10-08):** [findings/roadmap](research/proposals/2026-10-08_class_conditional_findings.md), [mechanism explained](research/proposals/class_conditional_mechanism_explained.md).8640development settings,1176fresh attack/utility cells,320newimages/136unused. Public-cap CIA channel verified but q.55utilitygatefails; gains overpublic +.000119/−.000072/−.000010 and strongestnonprimarywinsall3. Refreshed-mask apparentgain is publicadaptation: zero-private matchedpublicoffset is better by.000097–.000141all3. Independent primary6,235,039checks/max8.88e-16;posthoc attribution35,108/max1.11e-16;299passed/5deselected. Next proposed: DEV-only genuinely useful private-signal/headroom diagnosis before anotherdensity/reserve sweep. No newdensity/end-to-endtuning/population/novelty or overallcompletion claim. No jobs running.

**Latest verified research chunk (2026-10-08):** [public-reference findings](research/proposals/2026-10-08_public_residual_findings.md). Public-cap Gaussian comparison:16,758development settings,54arms/432target rows,512fresh images;456reserve remain. At q0.55fine residual CE0.451109/85.75%accuracy loses to public-only decoder adjustment CE0.447582/86.52%; projected absolute matches but is slightly worse than the latter. Fixed selected fine queries lose even without noise, so a zero-mean additive density change alone cannot repair their expected CE deficit. Main audit260,338numeric checks;293passed/5deselected. Follow-up DEVELOPMENT-ONLY gradient diagnostic: client empirical class distributions and public class-gradient references reduce mean client norms84–88%while preserving the balanced all-IN unbounded aggregate. Independent874checks; utility/OUT/shift/noise superiority remains untested, and private-data headroom is inconsistent. Next: signal-first public class-conditional control-variate feasibility with stronger public-gradient controls and explicit prior/conditional-shift stress. No final defense/novelty/end-to-end tuning or overall experiment-completion decision. No jobs running.

**Preceding matched-CIA chunk (2026-10-07):** [matched-CIA findings](research/proposals/2026-10-07_matched_cia_findings.md). Reoptimized Gaussian/radial laws at conditional AUC targets0.55/0.65/0.80:17,496development settings,117arms/936target rows,1,024fresh utility images;968reserve images remain. Near chance, radial votes lose to strongest Gaussian controls in all three seeds (mean CE1.430741 versus1.354335). At0.80votes improve over non-voting controls, but same-query Gaussian matches radial; no consistent noise-law benefit. Offline private norm calibration/tuning is unaccounted, so this is not a deployable privacy guarantee or independent-population result. Independent query/contract review plus33,174saved-artifact checks;287passed/5deselected. Next: development-only task-relevant compression/public-reference residual review before a new frozen constructor; this hypothesis is untested. No overall experiment-completion or merge decision; no jobs running.

**Preceding CIA boundary audit (2026-10-07):** [fixed-slot conditional CIA findings](research/proposals/2026-10-07_descriptor_cia_findings.md). Three saved federations, four dominant-class targets, 156 cells, 1,024 fresh evaluation releases per world. At epsilon8 aggregate balanced votes remain distinguishable: mean descriptor AUC0.879624 Gaussian /0.792654 radial, versus equal-world CE0.955695 /1.003976. Gaussian non-voting control leaks less (meanAUC0.785588), so the earlier utility gain is not a matched-leakage win. Reduced metric adaptation has IN/OUT std ratios2.103–2.371 and exact LR AUC>=0.999998; identity student decoder exposes that same scale channel. This is strong known-alternative auxiliary knowledge under fixed datasets, not new client populations or a historical CNN/Flower replication. Student finite attack bank is weaker; raw individual-object view differs from Q_i-only. Corrected initial target-class coverage and low-FPR arithmetic are documented. Independent contract review plus4,188numeric checks/48Gaussian expectation checks;282passed/5deselected. No remaining reserve images used. Next: frozen matched attack-strength/variance controls before broad sweeps; no final defense or overall completion decision.

**Preceding utility feasibility (2026-10-07):** [one-time descriptor findings](research/proposals/2026-10-07_private_descriptor_findings.md). Three initial noise laws:81cells/1,215arms; raw votes fail label stress. Follow-up corrects local imbalance BEFORE joint clipping/noise, with equally corrected/reweighted model/logit/probability controls. Development17,496configs; frozen choices confirmed on2,048fresh reserve images,18cells/810arms,512new paired draws. At epsilon8 aggregate balanced-loss votes gain0.113975–0.177312CE against selected strong analytic-Gaussian controls in all three seeds (63.14–66.40%accuracy); radial Laplace also passes but is worse on this48-dimensional voting query. Individually reusable client distributions fail against aggregate controls; central can use the same constructor. This supports a useful client-query construction, not a new noise density, CIA/metric-privacy superiority or end-to-end private tuning/publication. Remaining reserve1,992images. Independent saved-artifact audit68,532numeric checks/max error4.44e-16; kernel checks and277passed/5deselected. Active recommendation: freeze full-peer fixed-slot contribution/dummy CIA evaluation plus novelty/shift stress checks. Overall experiment remains active; no merge/completion decision inferred.

**Research direction reset (2026-10-01).** The owner wants to investigate client-side noise
sampled from a separately constructed distribution for each client, aiming to improve CIA
protection relative to server-side metric calibration. Distribution construction remains an
open question; a careful literature review precedes mechanism selection and the new roadmap.
The old local `docs/RESEARCH_ROADMAP.md` was removed with owner authorization as irrelevant.
This deletion does not propagate to other machines because `docs/` is gitignored.

Catch-up findings and source pointers: `research/project_evidence_audit.md`. The frontier HTML
was reproduced byte-for-byte and all 110 recorded stage entries were checked against raw attack
losses/accuracy files without discrepancies. Its "round-matched AUC" is folded paired
concordance, and 9/10 landed curves have three-seed mean scores above 0.55. Treat it as an
exploratory utility/leakage baseline, not a validated demonstration of CIA neutralization.
The owner requested both systematic and integrative review and confirmed the protocol. Version 1
is in `research/literature_review/README.md`: 31 included primary source families, technique
explanations, bounded search/screening records, synthesis and a provisional research plan.
Follow-up: `research/literature_review/next_step_handoff.md`. FACP now has a partial primary-
methods assessment; FedFR-ADP remains preview-only. Complete methods are still pending.
`research/threat_specification_review.md` specifies candidate games and verifies weighting,
metadata and active-ID seed issues; `research/noise_construction_proof_obligations.md` derives
reference noise controls and estimation obligations. The owner has now confirmed dataset-contribution secrecy among registered slots. See
`research/proposals/2026-10-04_mechanism_proposal.md` for public and privately selected joint
clipping/non-Gaussian profiles, an explicit sampler and construction accounting. Independent
math review rules out common-ball ellipsoid shaping as a quadratic-utility improvement. The
first closed-form diagnostic loses at total budgets 4 and 8 and gains slightly at 16 against
only the listed fixed public controls; an optimized public clipping radius beats the private
selector at all three budgets. This is not CIA or trained-model evidence. A reproducible follow-up checked 30 heterogeneous population/budget cases, optimizing profile
geometry and selector allocation and adding diamond-law controls. No positive-cost private
selector beat the best public control on the bounded grids; see
`research/proposals/2026-10-04_heterogeneity_findings.md` and
`results/client_specific_noise/analytical_heterogeneity.json`. Deprioritize separately paid RR
selection for FL implementation. The two-round protected-history calculation is now recorded in
`research/proposals/2026-10-04_protected_history_findings.md` and
`results/client_specific_noise/protected_history_two_round.json`: 15 budget/client-count cases,
selection/noise correlation and aggregate bias accounted. Adaptation improves the tested fixed
two-round schedule in some cases, but public one-release control wins or ties every case.
The changing-update quadratic spike is now recorded in
`research/proposals/2026-10-04_dynamic_quadratic_findings.md`: six cells, 1,024 development
and 8,192 fresh evaluation federations each. Every selected adaptive profile has equal axes
and is exactly the static mechanism; one-release controls win, including public shrinkage.
The rotated-bank/residual-aware comparison is recorded in
`research/proposals/2026-10-04_rotated_residual_findings.md`: residual selection beats stale
selection in four of six cells but beats tuned public geometry in only one cell, by ~0.39%
at E=16,N=8. One-release controls still win throughout. The owner's Oct-4 stop instruction
was respected. On Oct-5 “lets move to the next step” authorized the client-geometry audit.
See `research/proposals/2026-10-05_client_geometry_findings.md`: cached Fashion-MNIST,
a 68-parameter fixed-feature classifier, three seeds, balanced/quantity/explicit label-stress
partitions and 27 snapshots. Centered covariance is distinct and persistent in label stress;
full-information profile assignment gains at most 0.000363 CE in 81 two-bias-coordinate
comparisons, below the 0.001 feasibility gate. The gain changes clipped means rather than
reducing noise penalty. These are raw-information diagnostics, not DP/CIA evidence.
The owner then approved the broader oracle. See
`research/proposals/2026-10-05_broader_oracle_findings.md`: fresh test slice, 36 rows/108
comparisons, shared/oracle clipping and step-shrink optimization. All three bias contrasts
show a marginal label-stress round-20/label-8 gain of 0.001036–0.001126 CE in three seeds;
two of three noise-only intervals clear 0.001. Full 51-dimensional head has no gate pass.
The bias gain changes aggregate means despite higher noise cost. The owner approved the
frozen per-slot control: `research/proposals/2026-10-05_public_slot_findings.md`. On a third
fresh test slice, the fully frozen configuration exactly matches the oracle in three aligned
primary cells; gains 0.000938–0.000998 CE fail the retained 0.001 gate. Cyclically reassigning
the same clients makes it 0.001748–0.001783 worse than tuned shared, exposing slot-label
alignment. 24 rows/72 comparisons/four arms saved. The owner approved the local rule/cost
review: `research/proposals/2026-10-05_local_selector_findings.md`. A local dominant-class
route with purity fallback is slot-permutation invariant; conditional RR+L1-upload law is
specified with full assumptions. Exact development moments include selector variance and
remaining-budget noise. 18 records/54 cells: 12 small label-stress gains at labels 8/16,
42 losses, zero 0.001 gates; largest optimistic gain 0.000850. Public mixtures tie shared.
Raw history/global calibration are still unaccounted; no actual DP training/CIA claim.
Deprioritize paid RR implementation. The owner approved the hidden-mixture analysis:
`research/proposals/2026-10-05_hidden_mixture_findings.md`. Generic tails approach the
additive bound, so hiding the category gives no uniform discount for this law. Separate
dummy-edge accounting is xi+eta/2 rather than replacement xi+eta; recalibrating both routes
and controls fairly leaves 12/54 small development gains, zero 0.001 gates, maximum 0.000655.
Uniform sharpness is not a fixed-trained-head/domain claim; raw history/calibration remain
unaccounted. The owner approved the paid protected-history replay:
`research/proposals/2026-10-05_protected_real_history_findings.md`. Saved early/later bias
updates, full probe-selection correlation and paid probe budget, 18 records/54 cases, fresh
16,384 probe draws per case. Adaptive beats matched static in 18, optimized static in 12
(label stress, budgets 8/16), and full-budget one release in six (label stress, budget 8).
Strong-control gains 0.000109–0.000247 quadratic loss; zero 0.001 gates. Offline raw-anchor
replay, development-data reuse and unaccounted calibration exclude private-training/CIA
claims. The owner approved the observer review:
`research/proposals/2026-10-06_observer_contract_review.md`. Historical scorer sees model
shadow losses, not every upload; recommended primary contract retains the earlier peer/model
view and includes own coins/state. Server privacy is a stronger separate extension. Conditional
Gamma-share Laplace reference illustrates aggregate accounting; factor-7 variance reduction
versus stronger local uploads in an 8-slot example, but 8/7 more variance than matched server
Laplace. No placement superiority or new density claim. Independent math/code audit complete.
Fixed-slot weights/dummies differ from current removal runner; no training code changed.
Focused follow-up adds separately tracked Arete (ALT2022) and Harrison–Manurangsi (FORC2025),
without changing frozen version-1 systematic counts. Owner “move on” and sustained-autonomy
instruction extended the review. See `research/proposals/2026-10-06_research_direction_decision.md`:
full scalar methods/certificate audit, temporal MF/BLT/DMM, one-time surrogates and individual
Rényi filters. Deterministic scalar variance ratios versus Laplace at epsilon2/4/8/16:
4.853/1.904/0.393/0.00695; high-budget scalar gains cannot bypass vector/round costs.
Independently reviewed robust full-row-ball Gaussian workload limit rules out correlation gains
within that class; narrower enforced client influence is a tractable construction question.
Recommended next: distribution-guided cumulative influence allocation under a hard filter,
compared with optimized public clipping schedules. Whole-client peer/dummy proof transfer is
valid under documented assumptions; no utility gain or novelty shown. One-time private
surrogates remain a backup/control. Noise covariance remains public in the initial proof;
private noise-shape changes need a new certificate. No sampler or FL/CIA training launched.
New focused cards stay outside frozen systematic counts; no full experiment completion.
Owner “lets move wtih that dont stop until you had a concrete result on this direction” authorized
coupled influence-filter feasibility. See `research/proposals/2026-10-06_client_energy_filter_findings.md`:
27 real-data cells,100 development configs/cell,16 development and128 fresh evaluation trajectories.
Fixed public/carry-forward schedules versus greedy/remaining/norm-trend/energy-EMA policies.
Neither matched history policy reaches the0.001 CE gate; largest forecast gain0.00001717.
Development-only spending/alignment diagnosis shows strong clipping and little unused energy.
One-release bounded local-model reference wins21/27 cells but often uses80 local steps.
Audit-directed post-hoc20-step/matched-energy reference wins6/27, all label-stress E4/E8;
E8 gains0.01236/0.03936/0.03593 acrossseeds. Feedback/clipping/release schedule still differ,
so no noise-only causal attribution. Promote one-time private descriptors/surrogates as next
bounded construction question; filter remains reference, not final defense. Fixed-law conditional
peer/dummy accounting applies; offline tuning/artifact release are unaccounted. No CIA/metric
superiority, CNN transfer or full experiment-completion claim. Independent numeric/code audits
and `uv run pytest`277passed/5deselected. All raw artifacts under `results/client_specific_noise/`.
No further jobs running, no CNN/Flower/CIA sweep or experiment-completion decision inferred.
The owner-authorized 141 staged result files were committed and pushed as `1081ff7`.
The owner broadened the design to potentially new non-Gaussian mechanisms/distributions; see
`research/non_gaussian_mechanism_research.md`. Gaussian is a control, not a requirement.
Bounded analytical calculations, synthetic quadratic spikes and a reduced real-data geometry audit have been run; no final
mechanism has been selected. The historical next steps
below are context, not the active agenda.

**`feature/auc-targeted-noise-sweep` is complete (2026-09-01) and merged into `master`.** See
`reports/auc_targeted_noise_sweep.md` for the full writeup and "What's established" below for the
summary — not repeated here to avoid drifting out of sync.

---

Nothing else currently running. `reports/accuracy_vs_roc_auc.html` (refreshed 2026-08-15) was sent to
the project supervisor for review; his feedback asked for a step back from the numbers-heavy
format toward two plain-language, plot-supported claims: (1) more clients → lower CIA attack AUC,
especially vanilla, and (2) DP noise lowers attack AUC at a heavy accuracy cost. **New follow-up
report, 2026-08-16: `reports/cia_takeaways.html`** (generator: `reports/build_cia_takeaways.py`,
independent of `build_accuracy_vs_roc_auc.py` — recomputes everything from source rather than
reusing the old script's embedded numbers) builds one plot per claim, one plot per dataset where
data exists, and checked both claims against the actual numbers rather than assuming them true:
- **Claim 1 (client count) holds only partially, and only on CIFAR-10** — the only dataset ever
  run at more than one client count. Round-matched attack AUC does drop net (95%→55% from 8→100
  clients) but isn't monotonic (bumps back to 100% at 48), and Global-DP/Metric-privacy show the
  *same* shape as vanilla, not a weaker one — so the "especially vanilla" framing isn't supported.
  Flagged as an open question in the report, not silently smoothed over.
- **Claim 2 (noise vs. accuracy) holds directionally on every dataset checked** (CIFAR-10 full
  sweep, EuroSAT, CIFAR-100, Alzheimer, Fashion-MNIST — 6 checks, 6/6 show DP attack AUC ≤
  vanilla's) but "heavy cost" is dataset-dependent, not universal: real on CIFAR-100 (~8pp) and
  CIFAR-10 under high noise (~20pp for global-dp), negligible on EuroSAT/Fashion-MNIST (<2pp).
- Old `reports/accuracy_vs_roc_auc.html` was left as-is (not overwritten) per explicit instruction
  to generate a new file instead — both now exist; `cia_takeaways.html` is the one to send back to
  the supervisor.
- Housekeeping: 4 remote branches confirmed fully merged into `origin/master` (0 unique commits
  each) and deleted, both locally-absent and on `origin` — `feature/cia-client-scaling`,
  `feature/cifar10-scaling` (already reflected below), `improve-logging`, `new_experiments`. `git
  branch -a` now shows only `master`.

Separately, both `feature/eurosat-scaling` (the EuroSAT accuracy sweep and its CIA attack) and
`feature/cifar100-scaling` (the CIFAR-100 accuracy sweep and its CIA attack) are complete, merged
into `master`, and deleted (locally and on `origin`) — see "What's established" below and
`reports/eurosat_accuracy_sweep.md`/`reports/eurosat_cia.md`/
`reports/cifar-100_and_eurosat_results.tex` for the full writeups.

**CIFAR-100 CIA multi-seed retry was dropped.** The seed-42 single-seed CIA run (6/6 combos, 0
failures) was complete and reportable. A follow-on rerun to add seeds 43/44 — for a properly
3-seed-pooled analysis matching the Alzheimer/CIFAR-10/Fashion-MNIST/EuroSAT CIA protocols —
launched 2026-08-10 20:08 but never finished: the large 100-client/4.6M-parameter model combos hit
persistent, severe GPU VRAM contention on this shared machine. Four separate retry attempts (each
recovering 1-3 of the failing combos, with diminishing returns, despite reducing
`--max-parallel-clients` 16→6→10 and fixing a real `dp_diagnostics.py` crash-on-client-error bug
along the way) never reached a complete 3-seed run. Decision: drop seeds 43/44 entirely and keep
only the complete seed-42 data. `results/cia_cifar100_scaling/cia_in.json`/`cia_out.json`/
`cia_analysis.json` and the 12 per-trajectory result JSONs now hold seed-42-only data for all 6
combos; the 22 seed-43/44 per-trajectory JSONs were deleted.
`reports/cifar-100_and_eurosat_results.tex`'s CIFAR-100 CIA table reflects this (a single
seed-42-only table, all 6 combos scored, replacing the earlier 3-seed-pooled table that had two
"in progress" `global-dp` rows).

**Model history** (this repo went through several CIFAR-100 architectures before settling):
v1-v3 was a plain 3-block CNN (v2 briefly added a 4th conv block, reverted after its natural
per-round update magnitude exceeded `clipping_norm=5.0` and froze every clipping privacy mode).
v4 replaced it with a DenseNet+SELU architecture (553,220 params, concatenative skip connections,
GroupNorm(8), SELU with LeCun-normal init and AlphaDropout), built and verified for robustness to
that clipping-related freeze — and a second, independent freeze mode was found and fixed in that
generation too (`vanilla`, no DP at all, froze at n=128/homogeneous from many highly-correlated
client updates reinforcing rather than averaging out). **Per project-owner direction, v4 was itself
replaced** with the current model (v5): an adaptation of the project supervisor's own `CNNCIFAR100`
reference architecture (3 blocks of 2x[Conv3x3-GroupNorm-ReLU], channels 128/256/512,
global-average-pooled classifier, 4,631,268 params, 0 buffers — see
`experiments/cifar100_scaling/sweep_cifar100_scaling.py`'s docstring for the full model-history
record and `experiments/reproduce/cifar100_cnn.py`'s docstring for the adaptation details). This
consolidation also removed the separate `feature/cifar100-scaling-supervisor` branch/worktree that
had briefly held this model in isolation (merged into this branch, then deleted) and cleared every
prior CIFAR-100 result (v1-v4 sweep data, the supervisor model's own earlier narrower grid) —
none were kept, since all described a model, grid, or directory layout no longer in use.

GroupNorm has no running-stats buffers — unlike BatchNorm — so this still needs no
`metricdp_pytorch/metricdp_strategy.py` changes, same as v1's "no normalization at all" workaround.
Training supports weight decay (`WEIGHT_DECAY = 5e-4`) and an opt-in cosine LR schedule; the
schedule was tried and dropped — a decaying LR eventually drops client updates below
`clipping_norm`, so clipping stops binding late in each run and the DP noise-to-signal ratio drifts
back up against `noise_multiplier`, fighting the whole point of tuning it — so the sweep uses a
fixed LR instead. `noise_multiplier=0.0182`, calibrated specifically for this model at n=100 (the
only client count this sweep runs — see the `NOISE_MULTIPLIER` comment in
`experiments/cifar100_scaling/sweep_cifar100_scaling.py` for the full derivation), confirmed via a
verification run: noise-to-signal ratio 1.001 at n=100.

Full sweep and CIA results (tables, protocol, discussion) are in "What's established" below and
`reports/cifar-100_and_eurosat_results.tex` — not repeated here to avoid drifting out of sync with
those. Raw data: sweep JSONs at `results/cifar100_scaling/*.json`; CIA raw/analysis JSONs at
`results/cia_cifar100_scaling/`. Both experiments' `.evaluation.json`/`.predictions.npz`
artifacts stay local-only, gitignored (see `.gitignore` comment) — they blow past GitHub's 100MB
push limit.

Separately, on `master`: nothing currently running. `feature/scale-controlled-redo` (Phase 1 items
1 and 2 of `docs/RESEARCH_ROADMAP.md`) merged into `master` 2026-08-06 and was deleted — see
"What's established" below for what it left behind. The natural next steps there, not yet started:
`fedyogi` at `n=48` (the redo's matrix only covers `n=4/8` for `fedyogi`), and Phase 1's remaining
item (NaN/failure-mode logging in `runner.py`, motivated directly by the zero-norm-update crashes
found during the redo). After that, Phase 2 (mechanism redesign) is the next major phase.

### Currently running

Local CPU development-only headroom diagnostic completed and independently verified. No jobs running; reserve untouched.

Update this table whenever a machine picks up new work: add a row, edit the Status column
in place (e.g. `running` -> `done`), and leave a finished row for one update cycle before removing
it, so machine-to-results provenance isn't lost; see `AGENTS.md`'s "Working across machines"
section.

| Command | What | Status |
| --- | --- | --- |
| `uv run python -m research.calculations.frozen_constructor_probe --stage freeze` then `--stage confirm` | Local CPU: pooled freeze + fresh test-split confirmation | Done; independent 113 checks |
| `uv run python -m research.calculations.limited_public_probe` | Local CPU: projection/clipping/noise decomposition at budgets 32/128/512 | Done; independent 1,278 checks |
| `uv run python -m research.calculations.headroom_probe` | Local CPU: private signal vs public budget/optimization | Done; independent224,895checks |
| `uv run python -m research.calculations.class_conditional_attribution` | Local CPU: post-hoc public-offset attribution on SAME320images | Done; independent35,108checks |
| `uv run python -m research.calculations.class_conditional_probe --stage evaluation` | Local CPU: frozen320-image utility, all8targets/shift transfer | Done; independent6,235,039checks |
| `uv run python -m research.calculations.class_conditional_probe --stage development` | Local CPU: public class-gradient residuals, six modes | Done; independent150,722checks |
| `uv run python -m research.calculations.public_residual_probe --stage development` | Local CPU: public-cap Gaussian, public reference/subspace, equal IN/OUT tuning | Done;16,758settings, choices frozen |
| `uv run python -m research.calculations.public_residual_probe --stage evaluation` | Local CPU:432target rows,512fresh utility images | Done; independently checked |
| `uv run python -m research.calculations.public_residual_public_control` (two stages) | Local CPU: supplementary frozen public-only model | Done;16development records,54contrasts |
| `uv run python -m research.calculations.public_residual_signal_diagnostic` | Local CPU: raw-information/development-only gradient diagnostic | Done; independent874checks |
| `uv run python -m research.calculations.matched_cia_probe --stage development` | Local CPU:17,496configs, three risk targets, Gaussian/radial | Done; selections saved before reserve features opened |
| `uv run python -m research.calculations.matched_cia_probe --stage evaluation` | Local CPU:117arms,8target alternatives,1,024fresh utility images | Done;936target rows; independently checked |
| `uv run python -m research.calculations.descriptor_cia_probe` | Local CPU: conditional fixed-federation IN/dummy audit,96certified cells+60adapted metric cells | Done;156cells; independently checked |
| `uv run python -m research.calculations.private_descriptor_probe` (three `--law` routes) | Local CPU: one-time descriptor pilots and laws | Done;81cells/1,215arms |
| `uv run python -m research.calculations.private_prior_constructor_probe --stage development` | Local CPU: three label-stress seeds, two laws, raw/prior/balanced teachers | Done;17,496configs; choices frozen |
| `uv run python -m research.calculations.private_prior_constructor_probe --stage evaluation` | Local CPU:2048fresh reserve examples,512draws/cell | Done;18cells/810arms |
| `uv run python -m research.calculations.audit_private_descriptor_artifacts` | Local CPU: independent artifact arithmetic/split/calibration audit | Passed;68,532numeric checks |

## What's established on `master`

- **AUC-targeted noise sweep** (`reports/auc_targeted_noise_sweep.md`, `reports/auc_frontier.html`):
  4 datasets (EuroSAT n=48, Alzheimer n=48, Fashion-MNIST n=48, CIFAR-10 n=100) x 2 partition modes
  x 2 privacy modes = 16 curves, each an autonomous search for the noise multiplier that pushes CIA
  round-matched attack AUC to ~0.5 (10 landed, 4 collapsed-before-target, 2 anchor-not-found — see
  the report for the full per-curve table). Noise-to-neutralize-attack cost is dataset/partition/
  mechanism-dependent, not a single number: EuroSAT reaches the target cheaply in all 4
  combinations; CIFAR-10 reaches it in all 4 but the accuracy cost swings from negligible
  (non-iid/global-dp) to severe (homogeneous/global-dp, pushed to near-random accuracy); Alzheimer
  and Fashion-MNIST/homogeneous show the sweep's clearest negative result — several combinations
  break the model before the attack is ever actually neutralized, and Fashion-MNIST/homogeneous
  (both privacy modes) never found a usable low-noise anchor at all in the range searched.
  Metric-privacy's cost advantage over global-DP shows up clearly on CIFAR-10/homogeneous but
  reverses on CIFAR-10/non-iid — no blanket "metric-privacy is cheaper" claim survives this data.
- The metric-privacy mechanism reproduces the source paper at 4 clients — the effect is barely
  visible at the paper's `noise_multiplier=0.01` (`reports/paper_reproduction.md`).
- A genuine, previously unpublished effect exists at 8 clients: metric-privacy beats global-DP by
  +6.9pp (homogeneous) / +12.2pp (non-IID) at `noise_multiplier=0.05`
  (`reports/client_count_scaling.md`).
- `MetricPrivacyServerSideFixedClipping.aggregate_train` (`metricdp_pytorch/metricdp_strategy.py`)
  no longer aborts a whole run on a non-finite/non-positive client-model distance or a
  `ZeroDivisionError` from Flower's own clipping code on zero-norm updates — both fall back
  gracefully (last-valid distance, or skip-and-keep-previous-round respectively) so a run's full
  round-by-round history survives instead of being discarded on one bad round. Richer per-round
  diagnostics also landed: full pairwise client-model distance distribution, per-pair client IDs,
  min/median/mean/count, not just the single max used for calibration.
  `LoggedGlobalDPServerSideFixedClipping` (`metricdp_pytorch/globaldp_strategy.py`) has the same
  zero-norm-update guard as of the scale-controlled redo — it never had one before, and 12/96 runs
  in `results/noise_by_clients/` hit exactly this crash before the fix landed.
- **A genuine client-reply-ordering non-determinism was found and fixed**: Flower's own weighted
  aggregation summed replies in network-arrival order, not deterministically — floating-point
  addition isn't associative, so this compounded into real numeric drift over rounds. Fixed via
  `DeterministicReplyOrderMixin` (`metricdp_pytorch/strategy_factory.py`, covers every aggregation
  method, not just metric-privacy) and a matching fix in `metricdp_pytorch/metrics.py`.
- **Phase 1 items 1 and 2 of `docs/RESEARCH_ROADMAP.md` are done**, redone on CUDA hardware (moved
  off this project's original Mac MPS backend, whose own non-determinism made every earlier result
  on these questions untrustworthy — see `reports/archive/constant_compute_scaling_mps_v1v2.md`).
  - **Constant-compute client-count scaling** (`reports/constant_compute_scaling.md`/`.tex`):
    `fedavg` at `n=4/8/48` + `fedyogi` at `n=4/8` — 40/40 combinations, 0 failures, 0
    invalid-distance/collapsed-aggregation rounds anywhere. The metric-privacy-vs-global-dp
    advantage shrinks from `n=4` to `n=48` but converges toward parity (`fedavg`: -2.5pp to
    +0.6pp at `n=48`), not the large reversal the MPS-era attempt found (which never survived its
    own noise-floor check). `fedyogi` at `n=48` not yet run.
  - **Noise-multiplier x client-count sweep** (`reports/noise_by_clients.md`): `n ∈ {8,16,32,48}` x
    6 noise multipliers, `fedavg` only — 96/96, 0 failures. The noise ceiling genuinely shifts up
    with client count (a fixed `noise_multiplier` is relatively less noisy at higher `n`, since
    `compute_stdv` divides by `num_sampled_clients`) — `nm=0.1` collapses training at `n=8` but
    stays healthy through `n=48`. Separately, metric-privacy's own calibrated noise goes unstable
    right at each client count's collapse boundary and underperforms global-dp there by as much as
    -18pp, worse at higher `n` — a real, more precisely localized version of the original
    `results/48client_scaling` scaling concern, and a lead for Phase 2's mechanism redesign.
- `reports/first_round_cia.md` is stale — says "no result data yet," but `results/cia_client_scaling/`
  has real trained models and partial attack scores. Needs a rewrite, not done yet. The Flower-1.32
  port-equivalence check (`reports/port_equivalence.md`) still has no committed result data.
- **CIFAR-100 accuracy sweep + CIA** (`reports/cifar-100_and_eurosat_results.tex`): `n=100`
  clients, 250 rounds, `fedavg`, `noise_multiplier=0.0182`. Accuracy sweep: 6/6 combos, 0 failed,
  21.4–29.1% accuracy (100-class task, ~1% random-baseline) — metric-privacy roughly ties
  global-dp (within ~0.6pp either way), both DP modes ~7–8pp below vanilla, partition mode barely
  moves the numbers. CIA: seed-42-only (a multi-seed 43/44 rerun was attempted for proper 3-seed
  pooling but dropped after repeated GPU VRAM contention on this large 4.6M-parameter model
  prevented it from ever completing — see `git log` on the deleted `feature/cifar100-scaling` for
  the retry history), 26 round-matched pairs per combo, all 6/6 combos scored. Leakage is highest
  for `homogeneous/vanilla` (0.846) and lowest for `non-iid/vanilla` (0.500, the no-leakage line);
  `global-dp` shows more leakage than `metric-privacy` in every partition/shadow combination on
  this seed. Every 95% CI includes 0.5 (single-seed, underpowered), so read this as a directional
  pattern, not a statistically confirmed ranking.
- **EuroSAT accuracy sweep + CIA** (`reports/eurosat_accuracy_sweep.md`, `reports/eurosat_cia.md`):
  a comparison point on satellite land-use imagery (10-class, genuinely different domain from
  CIFAR-10/CIFAR-100/Fashion-MNIST/Alzheimer), `n=48`. Accuracy sweep: 6/6 combos, 0 failed,
  87.5–90.7% accuracy across all combos; `non-iid` partitioning slightly *outperformed*
  `homogeneous` in every privacy mode, and DP mechanisms cost only 0.4–2.4pp versus vanilla. CIA:
  36/36 trajectories (18 IN + 18 OUT), 0 failed, 3 seeds from the start (CIFAR-100's CIA needed a
  post-hoc multi-seed redo after an underpowered single-seed pilot — this one skipped that
  mistake). Round-matched AUC shows the clearest leak at `homogeneous/vanilla` (0.727), both DP
  mechanisms suppress it there (to 0.606), and `non-iid` leaks much less across every privacy mode
  (one combo's noisy-shadow AUC drops to 0.212, below chance) — but confidence intervals are wide
  (~0.30–0.35 AUC units, all overlapping 0.5), so read this as a directional pattern, not a
  statistically confirmed ranking. Along the way: found and fixed a real bug in `server.py`
  (`_require_trained_arrays`) where a run whose every single round failed to aggregate crashed
  with a confusing "Missing key(s) in state_dict" error instead of a clear one — Flower's
  `Strategy.start()` only assigns `result.arrays` on a successful round, with no fallback to the
  initial model.

## Where to look

- `research/literature_review/README.md` — systematic/integrative review, technique tutorials,
  agent handoff and provisional new research plan.
- `research/project_evidence_audit.md` — verified starting evidence for the new client-side
  noise research direction; the obsolete local roadmap was retired on 2026-10-01.
- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data; `results/archive/` — superseded data kept for comparison.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
