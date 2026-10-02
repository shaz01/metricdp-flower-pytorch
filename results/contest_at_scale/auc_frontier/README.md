# AUC-targeted noise sweep (Aug 18–21, 2026, Ata)

## Question

How much noise does it take to bring the client inference attack down to chance (AUC ≈ 0.5), and
what does that noise cost in accuracy? The question came from the supervisor; see
[`auc_targeted_noise_sweep.md`](reports/auc_targeted_noise_sweep.md).

## Setup

16 curves: EuroSAT, Alzheimer and Fashion-MNIST at 48 clients, CIFAR-10 at 100 clients ×
homogeneous / "non-IID" × global-DP / metric-privacy. For each curve, an automatic search doubles
the noise until the attack score on seed 42 lands in 0.45–0.55, then reruns that noise on seeds 43
and 44 to confirm. A curve stops early if accuracy collapses. The score is computed on client 0 at
11 checkpoints (20 for CIFAR-10) and folded with max(AUC, 1−AUC).

## Result

**The sweep can't answer the question; the "landings" didn't hold.** 10 of 16 curves landed, but
only one of them (CIFAR-10 non-IID metric-privacy) stayed in the band on both confirmation seeds.
The rest had at least one confirmation seed come back at 0.64–1.00 at the same noise, so the
seed-42 landings were mostly luck.

**The accuracy side is still useful.** Accuracy stays near vanilla until a point, then falls off
within one or two doublings. Where that point sits depends on the dataset: EuroSAT loses at
most ~6 points at its landing noise, while CIFAR-10 homogeneous global-DP drops to 17%. Four
curves collapsed to near-chance accuracy before the attack reached the band, and two
(Fashion-MNIST homogeneous) never found a starting point.

There is no general "metric-privacy is cheaper" result here: it is cheaper on some curves and
not on others.

## Caveats

- **One target client, scored at only 11 checkpoints.** The score moves in steps of 1/11, and
  one client's score swings a lot between seeds.
- **The fold pushes the score up.** With no real signal, max(AUC, 1−AUC) averages about 0.62 at 11
  checkpoints, so 0.5 is hard to reach even for a perfect defense.
- **The search stops at the first in-band seed,** which favours lucky draws.
- **The "non-IID" split isn't label-skewed;** it only varies client size. See
  [`dirichlet_comparison`](../dirichlet_comparison/README.md).
- An audit of this sweep: `research/project_evidence_audit.md`. A per-client rescoring on EuroSAT:
  [`cia_frontier/eurosat_frontier`](../../cia_frontier/eurosat_frontier/README.md).
