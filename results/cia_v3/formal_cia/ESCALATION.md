# P5 escalation: the feature cache differs from B4's (2026-10-30)

**Trigger:** PROTOCOL.md Sections 2 and 8: a feature fingerprint different from B4's stops everything; reproduction
check (a) failed its 1e-12 criterion. No P5 run was launched.

## What happened

The owner PC had no frozen-feature cache, so it was extracted here (`python -m experiments.frozen_head.features
--threads 8`, CPU, Apple silicon), with the same code, weights (`IMAGENET1K_V1`) and Hugging Face split.

| Split | Shape and class counts | Fingerprint here | B4's (compute server) |
|---|---|---|---|
| train | (21600, 512) | `8b56243fcb32340f` | `30ee5d8828e2339c` |
| test | (2700, 512) | `4ae40e575a25aef2` | `cdd8f364245a012b` |

## Evidence that only float32 rounding differs

- **Labels and record order are identical.** `audit.panel(48)` and `panel(480)` passed their checks: the candidates
  equal M0's fixed panel, and every candidate's training records equal the layout's. Both depend on the labels in
  split order.
- **The code path is bit-identical.** `test_default_path_bit_identical` checks the modified `fedavg.run` against the
  pre-P5 loop written out (`np.array_equal`).
- **Reproduction check (b) passes.** P5's analysis on B4's committed records gives 0.638 [0.587, 0.689], equal to
  the published value to 3 decimals.
- **Reproduction check (a), five committed B4 Part D runs re-run here:**

| Run | Final accuracy here / committed | Max shadow-loss difference | Clip here / committed |
|---|---|---|---|
| `n48_T10_mu2_pub20_r00` | 0.836296 / 0.836296 (equal) | 4.6e-06 | (not printed) |
| `n48_T10_mu1_pub20_r07` | 0.671111 / 0.671111 (equal) | 1.5e-05 | 0.39427491 / 0.39427492 |
| `n48_T10_nonoise_pub20_r12` | 0.928889 / 0.928889 (equal) | 6.6e-06 | 2.55331873 / 2.55331874 |
| `n480_T10_mu2_pub20_r03` | 0.920000 / 0.920000 (equal) | 1.8e-05 | 0.93281848 / 0.93281694 |
| `n480_T10_mu1_pub20_r20` | 0.914815 / 0.914815 (equal) | 4.2e-06 | 0.33467109 / 0.33467117 |

The differences are what float32 convolution kernels on a different CPU produce (relative 1e-7 to 2e-6 in derived
quantities). They are 3-4 orders of magnitude below anything the analysis resolves: AUC intervals are about ±0.05,
and the targets are quoted to 3 decimals.

## Options

1. **Recommended: tolerance amendment.** P5 runs on this machine's cache, and records its fingerprints. All 832 P5
   runs use the same features, so every comparison inside P5 is unaffected. Reproduction check (a) becomes: final
   accuracy identical, and shadow losses within 1e-4, on the five runs above (observed at most 1.8e-5). The amendment
   is dated in PROTOCOL.md before any P5 run.
2. **Copy B4's cache** (`eurosat_{train,test}_resnet18-imagenet1k-v1.npz`, about 50 MB) from the compute server to
   `~/.cache/metricdp_frozen_head/` on this PC. This needs the owner. It makes the fingerprints match, but the float64
   NumPy training may still differ at the last bit across CPUs, so the 1e-12 check could still fail.
3. **Run P5 on the compute server**, where the cache exists. This departs from decision 4 (Agent F on the owner PC)
   and shares a machine with Agent P.

## State

- Code (`fedavg.py` options, `formal_cia.py`) and tests are committed. `pytest experiments/frozen_head/tests` passes
  (36 tests).
- P5 waits for the owner's choice.

## Resolution (2026-10-30, owner)

The owner chose **option 1, "accept a tolerance"**. It is recorded as PROTOCOL.md Amendment 1, committed before any P5
run. P5 resumes.
