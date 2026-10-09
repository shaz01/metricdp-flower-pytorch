# Client-specific-noise research (closed 2026-10-09)

**Status: closed by the owner on 2026-10-09. Nothing here is an active plan.** "Next", "proposed" and "active direction"
wording inside the documents below is historical: those steps were either carried out in a later phase or dropped. The
roadmap documents `literature_review/research_plan.md` and `literature_review/next_step_handoff.md` were removed at closure;
they remain in git history (last present at merge parent `0d504bfc`).

## The question

Can noise that each client constructs for itself (an individually built distribution) improve protection against client
inference attacks (CIA) over the paper's server-side metric calibration, at useful accuracy?

## Where it ended

- **Per-client noise shapes and non-Gaussian densities gave no benefit.** Analytical calculations, quadratic spikes and
  matched-attack-strength comparisons (radial Laplace versus Gaussian) found nothing that beats a public Gaussian control.
- **One construction did help against public-only models**, only when public TRAINING data are scarce: each client sends a
  clipped, public-subspace-projected, class-balanced head gradient plus its own calibrated Gaussian noise share; the server
  sums, and a public validation set picks the step size (post-processing). Reliable at 32 public images on four datasets-by-class-split
  (30 public sets per KMNIST task, fresh data), small at 128, gone at 512; the attack AUC was verified on the real Flower
  path. Implementation: `experiments/stacked_head/` (README and `protocols/` with the findings of every phase).
- **That construction is NOT better than the paper's server-side mechanisms** (Flower global DP, metric privacy) at matched
  calibrated risk: pre-stated verdict mixed, baselines clearly better at risk 0.80; metric calibration performs like global DP
  once leakage is matched. The transferable piece is the public-validation gate, which nearly doubles the paper mechanism's
  utility at risk 0.80.

## Map

- `literature_review/` - systematic and integrative review (version 1), technique primers, comparator follow-ups.
- `proposals/` - protocols, findings and reviews of every empirical phase, chronological (`2026-10-04` ... `2026-10-08`);
  the later stacked-head phases (`2026-10-09`) live with their code in `experiments/stacked_head/protocols/`.
- `calculations/` - the one-off probes, audits and tests behind `results/client_specific_noise/`.
- `project_evidence_audit.md`, `threat_specification_review.md`, `noise_construction_*.md`, `non_gaussian_mechanism_research.md` -
  evidence audit, threat game, proof obligations and mechanism notes.
- Data: `results/client_specific_noise/` (probes) and `results/stacked_head/` (Flower experiment, validation, baselines).

Scope limits that apply to everything here: 4-class tasks, small models, Gaussian noise for the working construction, a
conditional peer/known-alternative contract, offline tuning not accounted for. The written report required by
`AGENTS.md` for a finished experiment is not part of this record; it is the owner's call.
