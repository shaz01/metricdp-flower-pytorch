package: P6, label-channel defense (roadmap v3, F1): design memo only
status: design memo complete; owner decision 2 recorded (design choices); screens NOT open
gate: G-P6, applies to the screen only; the screen needs a new owner decision in ROADMAP.md Section 10 (decision 6)
protocol: none yet (the screen's protocol is written only after the owner opens it)
machine: owner PC (memo: no compute)
cost: memo 0; screen cap 15 GPU-h, if opened
next: wait for the paper packages to close and the owner to open the screens; then the stage-1 (head, CPU) protocol
needs from others: owner (opening the screens when the paper packages close)

Roadmap: `research/cia_roadmap_v3/ROADMAP.md` Section 3, P6, item 1. Agent F. Branch `feature/cia-v3-label-channel`
from master `103d26b8`. No code; nothing Agent P owns is touched.

## Hypothesis (for the screen, if it opens)

On label-skewed EuroSAT, a full-participation mechanism that makes each client's update independent of its label
mix (up to a privatized label histogram) reaches the composite's leakage (AUC about 0.60) at matched accuracy 0.740.
The composite cannot reach that accuracy without subsampling's ceiling.

## Gate (copied verbatim from ROADMAP.md Section 3, P6)

> **G-P6.** At 0.740, at least 0.05 below metric privacy (decision 9's rule) and no worse than the composite by more
> than 0.03 (point estimate). Fail: the second paper does not start.

Untestable target (standard v3 item 5): if a screened mechanism's curve does not reach 0.740, that comparison is "not
testable" and counts as a fail of G-P6 for that mechanism.

## Log

- 2026-10-30: worktree opened. P5 is escalated (`feature/cia-v3-formal-cia`, ESCALATION.md), so the memo was started
  while waiting. The memo needs no compute and does not depend on P5.
- 2026-10-30: RR-Label read from its author copy and LDPM from the owner's copy (kept in the gitignored `docs/`).
  DESIGN.md Section 4 corrected: RR-Label has no formal guarantee and targets a malicious server; LDPM hides `m`
  label flips on fixed features, for one-vs-all models only.
- 2026-10-30: owner decision 2 (ROADMAP.md Section 10): no fill-in, one fixed per-class subset, RR-Label kept,
  screens on the head first (CPU) and the CNN only if PN passes there, timing as decision 6. DESIGN.md Section 5
  carries the proposed stage-1 gate.
