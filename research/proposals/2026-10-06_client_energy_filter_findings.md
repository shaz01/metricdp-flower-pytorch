# Coupled influence filtering: concrete utility result and direction update

2026-10-06. Owner authorized continuing until a concrete result. This bounded phase reached one: **the two tested local history policies fail the retained utility gate; a one-time bounded client-model release has appreciable label-stress headroom.** The overall research remains active. This note is a diagnostic handoff, not a completed-experiment report or evidence of superiority over metric privacy.

## What actually ran

Cached Fashion-MNIST classes0–3; deterministic public16 pooled image features+bias,51 identifiable softmax parameters, eight registered slots,20 rounds, three partition regimes and seeds42/43/44. There are27 partition/seed/epsilon cells. Each cell evaluates100 development configurations (20 policies×5 nominal energy budgets),16 complete noisy trajectories each. Nine selected arms then receive128 fresh paired noise trajectories on928 previously unused official test examples. These are genuinely coupled runs: every arm computes gradients on its own current noisy model, rather than replaying raw saved updates.

The [protocol](2026-10-06_client_energy_filter_protocol.md) specifies epsilon4/8/16,delta1e-5, public fixed weights, a single curious peer and no dropout. Public schedule controls include five exponential tilts and three two-block shapes, each with a fixed-cap and carry-forward version. Strong controls include greedy exhaustion and equal remaining-round allocation. Policy parameters/energy are selected on development loss, never held-out loss. This is a finite search, not optimization over all possible schedules.

Code and raw artifacts: [pilot calculator](../calculations/client_energy_filter_probe.py), [pilot JSON](../../results/client_specific_noise/client_energy_filter_probe.json), [paired losses and spending](../../results/client_specific_noise/client_energy_filter_probe.npz), [one-release calculator](../calculations/client_energy_one_release.py), [one-release JSON](../../results/client_specific_noise/client_energy_one_release.json), [compute control](../../results/client_specific_noise/client_energy_one_release_compute_control.json), and [development-only diagnostic](../../results/client_specific_noise/client_energy_filter_diagnostic.json). Reproduction commands are in the [result README](../../results/client_specific_noise/README.md).

## How the tested mechanism works

Each client computes a deterministic gradient update from its own data and the broadcast noisy model. It tracks how much *weighted learning signal* it has contributed. A private policy proposes how much remaining energy to spend now; the final radial filter clips the actual signal and debits its squared norm. All clients add fresh noise every round, including empty or exhausted slots. No spending/exhaustion flags enter the peer transcript.

The first candidate estimates the ratio of successive update norms, smooths it and predicts a geometrically changing future norm sequence. The second maintains an exponential moving average of proposed squared norms. Both allocate remaining energy according to estimated current versus future demand. These are simple estimates of update-magnitude behavior, **not fitted full client distributions or learned noise densities**. Private statistics guide the protected mean; the noise law remains public.

For weighted per-client cap b=B/64, enforce sum_t||x_it||²≤b for every dataset and observed history. Set the guaranteed unknown-peer coordinate variance V=b/(2rho), with rho=(sqrt(log(1/delta)+epsilon)−sqrt(log(1/delta)))². Each of eight already-weighted Gaussian shares has varianceV/7. The curious peer can subtract its own share; seven unknown shares supplyV. Full-model aggregate variance is8V/7. All51 coordinates are treated jointly by the L2 cap, with no record-count sensitivity discount.

Under the independently reviewed deterministic/shared-history assumptions this supplies the fixed-configuration whole-client contribution-versus-dummy rho-zCDP transcript bound. A central implementation of this identical law has the same utility distribution: placement itself gives no improvement. Multiple colluding peers, dropout and server inbox access are excluded. Quantity-mode unweighted caps vary with public weights; B is a nominal equal-weight reference.

**Privacy boundary:** offline raw-data-dependent configuration selection, publication of diagnostics and benchmark artifacts are not accounted. This complete research pipeline is not end-to-end DP. The certificate concerns ideal Gaussian kernels with fresh unknown coins; this NumPy finite-precision simulator is not an audited deployable DP sampler. Energy traces, seeds and pooled diagnostic gradients are outside the modeled peer view. No CIA evaluation was performed.

## Main comparison: no history-policy gate passes

Positive gain means public-control CE minus candidate CE. At identical B, accounting, release schedule and aggregate noise law:

| Candidate | Positive point gains /27 | Gain≥0.001 /27 | Lower95% conditional interval>0.001 /27 |
|---|---:|---:|---:|
| Norm-trend forecast |4|0|0|
| Energy EMA |0|0|0|
| Equal remaining-round allocation |0|0|0|
| Greedy remaining-energy filter |0|0|0|

The largest matched forecast gain is **0.0000171731**, far below0.001. Allowing each family to tune B independently leaves forecast with zero gates. Tuned EMA has one isolated point gain0.00272294, but its lower interval endpoint is0.000579464 and it uses a different noise scale; no interval gate or three-seed label-stress criterion passes. This is not evidence against every possible learned policy.

Three-seed means, conditioning on these datasets and configurations:

| Partition | Epsilon | Public CE | Matched forecast gain | Matched EMA gain |
|---|---:|---:|---:|---:|
| Balanced |4|1.262382|−0.002337|−0.000691|
| Balanced |8|1.243801|−0.000652|−0.000388|
| Balanced |16|1.236275|−0.000001|0.000000|
| Quantity |4|1.283169|−0.008887|−0.001238|
| Quantity |8|1.244333|−0.003643|−0.000657|
| Quantity |16|1.236095|−0.000824|−0.000939|
| Label stress |4|1.384419|0.000014|−0.000011|
| Label stress |8|1.364156|−0.004737|−0.006121|
| Label stress |16|1.288870|−0.002085|−0.000421|

Intervals are paired Monte Carlo intervals with128 noise draws, conditional on data and selected parameters. They exclude population uncertainty, development-selection privacy and multiplicity corrections. Seeds reuse the same underlying corpus. The weak label-stress E4 utility (public mean accuracy27.81%, four-class chance25%) cannot substantiate useful CIA protection.

## Why the spending rules did not help

A separate development-only diagnostic reruns15 selected arms: five policies×three label-stress seeds atE8,128 new noise draws. It computes pooled development descent directions **only for diagnosis**; those directions do not enter a client policy or any upload.

Selected public controls consume96.0%–100% of the available energy. Proposed updates are clipped in94.4%–100% of client-rounds. Their mean signal/proposal norm ratios are0.247–0.498; mean cosines to development descent are0.156–0.193, and32.5%–33.9% of local proposals point against that descent direction. Thus unspent budget is not the main opportunity in these cells. Norm forecasting redistributes heavily constrained signal without resolving directional conflict.

There is nevertheless positive within-round correlation between squared proposal norm and retained first-order descent (mean per-round correlations0.865–0.892 for the public controls). **Do not claim update magnitude contains no utility information.** These descriptive diagnostics show that the tested forecasting/allocation formulas did not turn that information into a loss advantage. They do not establish a causal explanation or rule out direction-aware policies.

## One-time release: a stronger bounded reference

A client can train internally, clip its final weighted51-dimensional local model to sqrt(b), and release it once with the same Gaussian share calibration. Internal deterministic local optimization is unobserved; it does not consume one charge per local step. Subsequent learning using only these sanitized objects is post-processing. Future direct raw-data access would require additional accounting.

The secondary reference specified before inspecting pilot utility selects local steps5/20/80 and B on development data. It beats the iterative public control in21/27 cells; all21 paired lower intervals clear0.001. Under label stress atE8 it selects80 local steps,B5.12 in all three seeds:

| Seed | Iterative public CE | One-release CE | CE gain | Iterative accuracy | One-release accuracy |
|---|---:|---:|---:|---:|---:|
|42|1.365160|1.223896|0.141264|31.27%|45.73%|
|43|1.363703|1.221883|0.141820|32.36%|45.71%|
|44|1.363606|1.219992|0.143613|33.19%|47.19%|

This changes both schedule and local computation. It is not an exact20-round noise comparison, and its large advantage cannot be attributed solely to avoiding repeated noise. The80-step/local-training choices and B hit the tested upper boundaries; no global optimum is established.

An audit-directed **post-hoc** control fixes local training to20 steps, evaluates the existing development-grid choice, and also matches B to the iterative public control. It uses the same held-out examples and paired noise; it is supportive analysis, not independent confirmation:

| Seed, label stress E8 | Matched B | One-shot20 CE | CE gain | Paired95% interval |
|---|---:|---:|---:|---|
|42|0.32|1.352797|0.012363|[0.006562,0.018164]|
|43|1.28|1.324338|0.039365|[0.029586,0.049144]|
|44|1.28|1.327672|0.035933|[0.026006,0.045861]|

The matched20-step one-shot wins in six of27 cells, all label-stress E4/E8; it loses elsewhere, including label-stress E16. Local computation and nominal budget now match, but feedback, clipping timing and release schedule still differ. Therefore **one-shot is a promising regime-specific route, not a universal replacement for iterative training**.

## What changes in the roadmap

1. Keep the cumulative filter as a valid reference and possible enforcement layer. Deprioritize a larger learned **magnitude-only** allocator unless a development-only direction-aware upper bound demonstrates headroom beyond these schedules.
2. Promote one-time private client descriptors/surrogates from backup to the next bounded construction question. The executable bounded local-model release now gives real utility evidence for that route, strongest in label stress. A client could build a sampling/soft-label distribution entirely from its sanitized model and public anchors; later samples and training would be post-processing. Reusing raw data to refit that distribution would invalidate the one-time argument.
3. First distinguish representation quality from density choice: compare sanitized local models, bounded predictions on public prototypes and simple distribution summaries at matched whole-client accounting, with public-only learning and this one-shot model as strong controls. Use fresh data/noise for confirmation and account public-calibration/configuration choice before a deployment claim. No new full distribution is selected here.
4. Gaussian remains the proof/control law, not a design requirement. Candidate non-Gaussian descriptor laws need the complete vector/output certificate; the reviewed scalar multi-scale laws do not remove dimensional cost automatically. Novelty lies, if anywhere, in a useful client-specific constructor and its measured CIA/utility tradeoff, not renaming known filters or one-shot perturbation.
5. Only after useful, independently confirmed feasibility: fixed-slot IN/OUT CIA trajectories with full curious-peer state, clean auxiliary-data splits and meaningful utility. Compare the historical [frontier](../../reports/auc_frontier.html) as an empirical benchmark with its original limitations; no epsilon-equivalence or protection superiority follows from this pilot.

## Verification and handoff

Independent audit recomputed1,512 pilot/one-shot comparisons with zero discrepancy, including selection minima, matched B, CE/accuracy means/SE, paired gains and intervals. Max recorded spent fraction1.0000000000000004 is within the explicit floating-point tolerance. A separate audit verified all compute-control means/SE and selection rules, plus stored spending/alignment diagnostics. Direct checks cover radial/zero clipping, all-policy cumulative caps, empty-client zero queries, contrast orthogonality, epsilon inversion and a projected-gradient finite difference. `uv run pytest`:277 passed,5 deselected.

All jobs have stopped. Code, raw numbers and this diagnostic remain on`feature/client-specific-noise`. No Flower/CNN/CIA experiment, new systematic-review inclusion, final defense selection, paper-novelty claim or overall experiment-completion decision is inferred.
