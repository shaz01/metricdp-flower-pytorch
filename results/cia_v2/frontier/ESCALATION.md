# M2 escalation: no useful empirical regime

Raised 2026-10-19 by Agent A under ROADMAP.md Section 4: "G-M2 | no useful regime | Escalate with the
recommendation that B4 lead; A1 still runs (the audit claim needs it)". Numbers are in [FINDINGS.md](FINDINGS.md).

## Finding

Global DP with independent randomness keeps the calibrated attacker's AUC above 0.6 at every noise level where mean
accuracy is 0.70 or higher: 0.720 at 1x and 0.702 at 2x, with lower gate-interval ends of 0.663 and 0.643. Leakage
reaches 0.593 only at 16x, where accuracy is 0.199.

## Why the roadmap's recommendation needs the owner

The roadmap's consequence is "B4 leads". B4 closed on 2026-10-17 with a G-B4 fail (no setting with `mu_total <= 8`
gains 5 pp over the public-only model), and the owner recorded its scarce-public positive as exploration only. The
consequence as written therefore has no package to point to. The decision on the paper's main line (ROADMAP.md
Section 1, item 6) is the owner's.

## What continues without a decision

- **A1 runs**, as Section 4 requires. Its noise levels bracket M2's two levels around 0.70 accuracy, 2x (0.740) and
  4x (0.660), per A1's STATE.md.
- **B1, B3 and B5** were to use `r_0.6`, which does not exist. Agent B's packages are not this agent's to re-plan.
  2x and 4x are the nearest candidates if the owner wants operating points for them.
- **B2** opens only if "M2 shows leakage falls with noise". It falls by 0.13 AUC from vanilla to 16x, but each step
  between 1x and 8x is below the detectable difference (0.087). Whether that counts is the owner's call.

## Resolution (2026-10-20, owner)

The owner accepted the four recommendations, recorded as decisions 7-11 in ROADMAP.md Section 10: the audit form is
the paper's main line; M2 and A1 are replicated on Alzheimer; B5 then B1 run at 2x and 4x with decision 5 amended to an
all-candidate gate at `R = 32`; B2 is deprioritized; the B4 scarce-public positive gets a confirmatory run.
