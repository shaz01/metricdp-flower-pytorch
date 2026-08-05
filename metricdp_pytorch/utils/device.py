"""Shared accelerator selection for training and evaluation."""

from __future__ import annotations

import os

import torch

FORCE_CPU_ENV = "METRICDP_FORCE_CPU"


def resolve_device() -> torch.device:
    """Return CUDA if available, else Apple MPS, else CPU.

    Set ``METRICDP_FORCE_CPU=1`` to skip MPS even when available and fall
    back to CPU instead.

    Measured directly (feature/scaling-diagnosis, 2026-08-02): MPS's backward
    pass is non-deterministic across separate process launches when multiple
    Ray client actors train concurrently on the shared MPS device, even with
    every seed fixed. torch.use_deterministic_algorithms(True) does not fix
    it -- it silently no-ops for MPS in this torch version, zero warnings
    even with warn_only=True, meaning these ops aren't hooked into the
    determinism-checking machinery at all. PYTORCH_ENABLE_MPS_FALLBACK=1
    doesn't help either and made the real pipeline hang.

    **Correction, 2026-08-04**: this was originally scoped to concurrent
    multi-actor training only, on the belief that isolated single-process and
    sequential (max_parallel_clients=1) runs reproduced exactly. That belief
    was wrong -- it was tested through the Flower/Ray pipeline over too few
    steps to show it, and never checked with direct tensor equality over a
    realistic number of steps. A minimal, no-Ray, no-Flower, single-process,
    single-client repro (10 local epochs, one PaperCNN, plain PyTorch) shows
    the same non-determinism with zero concurrency involved: two back-to-back
    training runs of the identical model/data/seed inside one process land
    ~0.02-0.04 max-abs-diff apart per parameter tensor. Ruled out as causes:
    DataLoader shuffle order (seeded generator, confirmed byte-identical
    across runs) and MPS's own RNG for dropout (adding
    torch.mps.manual_seed(seed) changed nothing). What's left is genuine
    backward-pass kernel non-determinism (most likely non-deterministic
    parallel-reduction order in MPS's conv2d/linear backward), present with
    or without Ray, with or without concurrency. **Treat MPS as
    non-reproducible for this project's training loop under any
    configuration**, not just concurrent multi-actor ones.

    **No settings-based fix exists (checked, 2026-08-04).** Every real,
    discoverable lever was tested directly against the same single-process,
    single-client, 10-epoch repro above, individually and combined -- none
    changed the outcome:
    torch.use_deterministic_algorithms(True) (confirmed no-op for MPS, see
    above); torch.mps.manual_seed(seed) (rules out MPS's own RNG, e.g. for
    dropout, as the cause); torch.mps.synchronize() after every optimizer
    step (rules out async/overlapping kernel dispatch as the cause);
    PYTORCH_MPS_FAST_MATH=0 (rules out relaxed-precision fast-math reduction
    as the cause); PYTORCH_MPS_PREFER_METAL=1 (rules out the MPSGraph vs.
    raw-Metal kernel-selection path as the cause). torch.backends.mps also
    has no cudnn.deterministic-equivalent flag at all in this torch version
    (2.10.0) -- inspected directly, it exposes only is_available/is_built/
    get_core_count/get_name. This backend currently has no way to make its
    backward kernels deterministic; ``METRICDP_FORCE_CPU=1`` is not a
    workaround among several, it is the only one that has actually worked.
    CUDA (unavailable on this machine, not tested) has real equivalents --
    torch.backends.cudnn.deterministic=True,
    torch.backends.cudnn.benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8 --
    that would be worth trying first if this project ever runs there,
    rather than assuming CPU fallback is needed on every backend.

    The resulting weight divergence is real (~5e-4 per parameter after a
    single federated round under concurrent multi-actor training, confirmed
    via direct state_dict diffing, not floating-point-summation-order noise)
    and compounds over many rounds/epochs into accuracy swings of the same
    order of magnitude as the deltas this project's research reports treat as
    signal (~15pp on identical settings across three reruns) -- unlike CPU,
    which reproduces exactly (differences at float32's own ~1e-8 precision
    floor, the same harmless noise multi-threaded CPU BLAS always has,
    confirmed via metrics matching to full displayed precision across runs).

    MPS stays the *default* despite this: measured directly, CPU is ~3x
    slower than MPS even at matched Ray parallelism (a real n=4/20-round run
    took >11 minutes on CPU vs. MPS's 214.7-220.0s for the same settings),
    and most work in this repo is exploratory where that cost isn't worth
    paying. Set ``METRICDP_FORCE_CPU=1`` specifically for reproducibility-
    critical validation runs -- e.g. a run whose numbers are going into a
    committed report or a cross-run comparison -- where correctness matters
    more than wall-clock time.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if os.environ.get(FORCE_CPU_ENV) == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_device_cache(device: torch.device) -> None:
    """Return pooled accelerator memory to the OS after a train/eval task.

    CUDA's and MPS's caching allocators reuse freed tensor memory instead of
    releasing it back to the OS. Flower's Ray simulation backend keeps each
    client as a long-lived actor process for the whole run, so on MPS (unified
    memory, shared with system RAM) that pooled-but-idle memory accumulates
    round over round within one actor until it exceeds total system memory on
    long/high-round-count runs -- call this once per task after the model is
    done with the device to keep each actor's footprint bounded.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
