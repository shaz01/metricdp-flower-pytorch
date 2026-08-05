# Archived: MPS-era constant-compute sweep (v1 rounds-fixed, v2 epoch-scaled)

Archived 2026-08-05. These are Phase 1 Part 1 (constant-compute control sweep,
`docs/RESEARCH_ROADMAP.md`) results from before two fixes landed:

1. **Client-ID-ordered aggregation** (`fix(strategy): sort client replies
   before aggregation`, `fix(metrics): aggregate client records in stable
   order`) — prior to this, floating-point aggregation order followed
   network arrival order rather than a deterministic client ID, a genuine
   source of run-to-run non-determinism.
2. These runs executed on `resolve_device()`'s MPS fallback (Apple Silicon),
   which has its own documented non-determinism (see git history on
   `metricdp_pytorch/utils/device.py`) that CUDA does not share.

Kept for historical reference and comparison, but superseded by a full redo
on CUDA hardware (RTX workstations/laptop) with the determinism fix in
place, tracked on `feature/deterministic-aggregation`. See
`reports/archive/constant_compute_scaling_mps_v1v2.md` for the write-up this
data supported.

- `scale_controlled/` — v1, rounds fixed per client count regardless of
  local epochs.
- `scale_controlled_epochs/` — v2, epoch-scaled constant-compute control.

Both were partial/incomplete matrices (some combinations never finished);
see the archived report for exact gaps.
