"""Autonomous per-curve noise-multiplier search.

Finds, for one (dataset, partition, privacy-mode) curve, the noise-ratio span
from a low-noise anchor (indistinguishable from vanilla) to a high-noise
landing point (CIA round-matched clean-shadow attack AUC in [0.45, 0.55]).

The search state machine (``find_low_noise_anchor``, ``step_up_to_target``)
is pure -- it takes a ``run_and_score`` callable and never touches training
code directly, so it is fully unit-testable with a fake. ``search_curve`` is
the thin real-training glue that wires the state machine to a per-dataset
``run_stage``/``score_stage`` pair and persists progress to
``search_state.json`` after every stage, so an interrupted curve resumes
without repeating completed stages.

Usage:
    uv run python -m experiments.cia.scripts.auc_target_search \\
      --dataset eurosat --partition homogeneous --privacy global-dp
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.cia.result import CiaResult
from experiments.cia.scripts.score_stage import score_stage
from experiments.reproduce.matrix.combo import format_noise

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "auc_target_sweep"

TARGET_BAND = (0.45, 0.55)
ANCHOR_TOLERANCE = 0.03
MAX_HALVINGS = 4
STEP_MULTIPLIER = 2.0
STAGE_CAP = 12
COLLAPSE_MARGIN = 0.02
VANILLA_SEEDS = (42, 43, 44)
PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES = ("global-dp", "metric-privacy")


@dataclass(frozen=True)
class DatasetSpec:
    module: str
    canonical_num_clients: int
    starting_noise_ratio: float
    random_baseline_accuracy: float


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "alzheimer": DatasetSpec(
        module="experiments.cia.scripts.alzheimer_remove",
        canonical_num_clients=48,
        starting_noise_ratio=7.783466863014625e-05,
        random_baseline_accuracy=0.25,
    ),
    "fashion-mnist": DatasetSpec(
        module="experiments.cia.scripts.fashion_mnist_remove",
        canonical_num_clients=48,
        starting_noise_ratio=0.0006691085733778867,
        random_baseline_accuracy=0.25,
    ),
    "eurosat": DatasetSpec(
        module="experiments.cia.scripts.eurosat_remove",
        canonical_num_clients=48,
        starting_noise_ratio=0.0007730650439020523,
        random_baseline_accuracy=0.10,
    ),
    "cifar10": DatasetSpec(
        module="experiments.cia.scripts.cifar10_remove",
        canonical_num_clients=100,
        starting_noise_ratio=0.0025,
        random_baseline_accuracy=0.10,
    ),
}


@dataclass(frozen=True)
class StageResult:
    noise_ratio: float | None
    seed: int
    accuracy: float
    auc: float


@dataclass
class CurveState:
    dataset: str
    partition: str
    privacy: str
    status: str = "searching"
    anchor_ratio: float | None = None
    anchor_stages: list[StageResult] = field(default_factory=list)
    search_stages: list[StageResult] = field(default_factory=list)
    landing_ratio: float | None = None
    confirmation_stages: list[StageResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "CurveState":
        payload = json.loads(text)
        payload["anchor_stages"] = [
            StageResult(**s) for s in payload["anchor_stages"]
        ]
        payload["search_stages"] = [
            StageResult(**s) for s in payload["search_stages"]
        ]
        payload["confirmation_stages"] = [
            StageResult(**s) for s in payload["confirmation_stages"]
        ]
        return CurveState(**payload)


RunAndScore = Callable[[float], StageResult]


def find_low_noise_anchor(
    *,
    starting_ratio: float,
    vanilla_accuracy: float,
    vanilla_auc: float,
    anchor_tolerance: float,
    max_halvings: int,
    run_and_score: RunAndScore,
) -> tuple[float | None, list[StageResult]]:
    """Halve ``starting_ratio`` up to ``max_halvings`` times looking for a
    stage whose accuracy and AUC both fall within ``anchor_tolerance`` of
    vanilla's. Returns ``(anchor_ratio_or_None, stages_tried)``."""
    ratio = starting_ratio
    stages: list[StageResult] = []
    for _ in range(max_halvings + 1):
        stage = run_and_score(ratio)
        stages.append(stage)
        if (
            abs(stage.accuracy - vanilla_accuracy) <= anchor_tolerance
            and abs(stage.auc - vanilla_auc) <= anchor_tolerance
        ):
            return ratio, stages
        ratio /= 2.0
    return None, stages


def step_up_to_target(
    *,
    anchor_ratio: float,
    target_band: tuple[float, float],
    step_multiplier: float,
    stage_cap: int,
    random_baseline_accuracy: float,
    collapse_margin: float,
    run_and_score: RunAndScore,
) -> tuple[str, float | None, list[StageResult]]:
    """Step ``anchor_ratio`` up geometrically until a stage's AUC lands in
    ``target_band``, a stage collapses, or ``stage_cap`` stages are tried.
    Returns ``(status, landing_ratio_or_None, stages_tried)``."""
    ratio = anchor_ratio
    stages: list[StageResult] = []
    low, high = target_band
    for _ in range(stage_cap):
        stage = run_and_score(ratio)
        stages.append(stage)
        if stage.accuracy <= random_baseline_accuracy + collapse_margin:
            return "collapsed-before-target", None, stages
        if low <= stage.auc <= high:
            return "landed", ratio, stages
        ratio *= step_multiplier
    return "stage-cap-reached", None, stages


# ---------------------------------------------------------------------------
# Real-training glue (not unit-tested directly; exercised by the pilot run).
# ---------------------------------------------------------------------------


def _module_for(dataset: str) -> Any:
    return importlib.import_module(DATASET_REGISTRY[dataset].module)


def _curve_dir(dataset: str, partition: str, privacy: str) -> Path:
    return RESULTS_ROOT / dataset / partition / privacy


def _state_path(dataset: str, partition: str, privacy: str) -> Path:
    return _curve_dir(dataset, partition, privacy) / "search_state.json"


def _load_state(dataset: str, partition: str, privacy: str) -> CurveState | None:
    path = _state_path(dataset, partition, privacy)
    if not path.exists():
        return None
    return CurveState.from_json(path.read_text())


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` atomically so an interruption mid-write can't leave a
    truncated/corrupt file behind (same idiom as ``attack_runner._write_report``)."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def _save_state(state: CurveState) -> None:
    path = _state_path(state.dataset, state.partition, state.privacy)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, state.to_json())


def _stage_dir(base_dir: Path, ratio: float | None, seed: int) -> Path:
    """A stage-isolated output directory.

    ``run_stage()`` writes its report to ``<output_dir>/runs/cia.json`` and
    ``run_attack`` returns *every* row ever accumulated in that report file,
    not just the rows from the current call. Stages sharing one output_dir
    would therefore silently pool every prior stage's rows together, so each
    (ratio, seed) pair gets its own subdirectory here.
    """
    ratio_token = "vanilla" if ratio is None else format_noise(ratio)
    return base_dir / f"nm-{ratio_token}-seed-{seed}"


def _run_and_score_real(
    *,
    dataset: str,
    partition: str,
    privacy: str,
    seed: int,
    output_dir: Path,
    max_parallel_clients: int | None,
) -> Callable[[float], StageResult]:
    spec = DATASET_REGISTRY[dataset]
    module = _module_for(dataset)

    def run_and_score(ratio: float) -> StageResult:
        stage_dir = _stage_dir(output_dir, ratio, seed)
        results: Sequence[CiaResult] = module.run_stage(
            privacy=privacy,
            partition=partition,
            seed=seed,
            noise_ratio=ratio,
            canonical_num_clients=spec.canonical_num_clients,
            output_dir=stage_dir,
            max_parallel_clients=max_parallel_clients,
        )
        score = score_stage(results, stage_dir / "runs")
        return StageResult(
            noise_ratio=ratio, seed=seed, accuracy=score.accuracy, auc=score.auc
        )

    return run_and_score


def ensure_vanilla_reference(
    *,
    dataset: str,
    partition: str,
    max_parallel_clients: int | None = None,
) -> StageResult:
    """Run (or reuse) the 3-seed vanilla reference for one (dataset, partition)."""
    path = RESULTS_ROOT / dataset / partition / "vanilla_reference.json"
    if path.exists():
        payload = json.loads(path.read_text())
        return StageResult(**payload)

    spec = DATASET_REGISTRY[dataset]
    module = _module_for(dataset)
    base_dir = RESULTS_ROOT / dataset / partition / "vanilla"
    per_seed: list[StageResult] = []
    for seed in VANILLA_SEEDS:
        stage_dir = _stage_dir(base_dir, None, seed)
        results = module.run_stage(
            privacy="vanilla",
            partition=partition,
            seed=seed,
            noise_ratio=None,
            canonical_num_clients=spec.canonical_num_clients,
            output_dir=stage_dir,
            max_parallel_clients=max_parallel_clients,
        )
        score = score_stage(results, stage_dir / "runs")
        per_seed.append(
            StageResult(noise_ratio=None, seed=seed, accuracy=score.accuracy, auc=score.auc)
        )

    reference = StageResult(
        noise_ratio=None,
        seed=-1,
        accuracy=statistics.fmean(s.accuracy for s in per_seed),
        auc=statistics.fmean(s.auc for s in per_seed),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(asdict(reference), indent=2))
    return reference


def search_curve(
    *,
    dataset: str,
    partition: str,
    privacy: str,
    max_parallel_clients: int | None = None,
) -> CurveState:
    """Run (or resume) the full autonomous search for one curve."""
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"dataset must be one of {sorted(DATASET_REGISTRY)}.")
    if partition not in PARTITION_MODES:
        raise ValueError(f"partition must come from {PARTITION_MODES}.")
    if privacy not in PRIVACY_MODES:
        raise ValueError(f"privacy must come from {PRIVACY_MODES}.")

    existing = _load_state(dataset, partition, privacy)
    # "searching" and "confirming" are the only non-terminal statuses: the
    # former means the anchor/target search hasn't finished, the latter means
    # a landing ratio was found but its 2 confirmation seeds haven't all run
    # yet. Anything else (landed, anchor-not-found, collapsed-before-target,
    # stage-cap-reached) is a terminal, idempotent no-op on resume.
    if existing is not None and existing.status not in ("searching", "confirming"):
        return existing

    spec = DATASET_REGISTRY[dataset]
    vanilla = ensure_vanilla_reference(
        dataset=dataset, partition=partition, max_parallel_clients=max_parallel_clients
    )
    output_dir = _curve_dir(dataset, partition, privacy)
    run_and_score = _run_and_score_real(
        dataset=dataset,
        partition=partition,
        privacy=privacy,
        seed=42,
        output_dir=output_dir,
        max_parallel_clients=max_parallel_clients,
    )

    state = existing or CurveState(dataset=dataset, partition=partition, privacy=privacy)

    if state.anchor_ratio is None and state.status == "searching":
        anchor, anchor_stages = find_low_noise_anchor(
            starting_ratio=spec.starting_noise_ratio,
            vanilla_accuracy=vanilla.accuracy,
            vanilla_auc=vanilla.auc,
            anchor_tolerance=ANCHOR_TOLERANCE,
            max_halvings=MAX_HALVINGS,
            run_and_score=run_and_score,
        )
        state.anchor_stages = anchor_stages
        _save_state(state)
        if anchor is None:
            state.status = "anchor-not-found"
            _save_state(state)
            return state
        state.anchor_ratio = anchor
        _save_state(state)

    if state.landing_ratio is None:
        assert state.anchor_ratio is not None
        status, landing_ratio, search_stages = step_up_to_target(
            anchor_ratio=state.anchor_ratio,
            target_band=TARGET_BAND,
            step_multiplier=STEP_MULTIPLIER,
            stage_cap=STAGE_CAP,
            random_baseline_accuracy=spec.random_baseline_accuracy,
            collapse_margin=COLLAPSE_MARGIN,
            run_and_score=run_and_score,
        )
        state.search_stages = search_stages
        if status != "landed":
            state.status = status
            _save_state(state)
            return state

        # Landing ratio and "confirming" status are persisted together, only
        # once both are known -- never persist a terminal "landed" status
        # before the landing ratio itself is saved, or a crash mid-confirmation
        # would leave the state file stuck with status="landed" but
        # landing_ratio=None forever (the resume check above would treat it
        # as done and never retry).
        assert landing_ratio is not None
        state.landing_ratio = landing_ratio
        state.status = "confirming"
        _save_state(state)

    # Confirmation phase: resumable per-seed, so a crash here only re-runs
    # whichever confirmation seeds hadn't completed yet.
    assert state.landing_ratio is not None
    already_confirmed = {stage.seed for stage in state.confirmation_stages}
    for seed in (43, 44):
        if seed in already_confirmed:
            continue
        stage = _run_and_score_real(
            dataset=dataset,
            partition=partition,
            privacy=privacy,
            seed=seed,
            output_dir=output_dir,
            max_parallel_clients=max_parallel_clients,
        )(state.landing_ratio)
        state.confirmation_stages.append(stage)
        _save_state(state)

    state.status = "landed"
    _save_state(state)
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY), required=True)
    parser.add_argument("--partition", choices=PARTITION_MODES, required=True)
    parser.add_argument("--privacy", choices=PRIVACY_MODES, required=True)
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    return parser


def main() -> None:
    args = _parser().parse_args()
    state = search_curve(
        dataset=args.dataset,
        partition=args.partition,
        privacy=args.privacy,
        max_parallel_clients=args.max_parallel_clients,
    )
    print(f"{args.dataset}/{args.partition}/{args.privacy}: {state.status}")
    if state.landing_ratio is not None:
        print(f"  landing_ratio={state.landing_ratio}")


if __name__ == "__main__":
    main()
