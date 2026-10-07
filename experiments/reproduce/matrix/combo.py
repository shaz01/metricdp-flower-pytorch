"""Run configuration shared by client-scaling sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.reproduce.matrix.hyperparams import Hyperparams
from metricdp_pytorch.strategy_factory import aggregation_token


@dataclass(frozen=True)
class Combo:
    """One configured invocation of the paper-reproduction runner."""

    name_prefix: str

    num_clients: int
    partition: str
    privacy: str
    aggregation: str
    seed: int
    noise_multiplier: float
    hyperparams: Hyperparams
    data_module: str
    model_module: str
    dirichlet_alpha: float = 0.5
    # Run-name token for the data module; defaults to the module's leaf name.
    # Set it when a script is renamed so historical run names (and resume /
    # analysis lookups against committed results) stay unchanged.
    data_tag: str | None = None
    # Per-client sample cap passed to the runner (0 = unlimited, the default).
    # Not part of run_name(); put it in name_prefix when it matters.
    max_client_samples: int = 0
    # FedAvg client weighting (metricdp_pytorch.strategy_factory.AGGREGATION_WEIGHTINGS).
    # "equal" adds "-eqw" to the aggregation token of run_name(); the default leaves it unchanged.
    aggregation_weighting: str = "num-examples"

    def run_name(self) -> str:
        """Build the complete deterministic name from this combo's parameters."""
        module_path = self.data_module.rsplit(":", 1)[0]
        data_module_name = self.data_tag or module_path.rsplit(".", 1)[-1]
        model_path = self.model_module.rsplit(":", 1)[0]
        model_suffix = f"__{model_path.rsplit('.', 1)[-1]}"
        partition_suffix = (
            f"__alpha-{format_noise(self.dirichlet_alpha)}"
            if self.partition == "dirichlet"
            else ""
        )
        return (
            f"{self.name_prefix}__{self.partition}{partition_suffix}__{self.privacy}__"
            f"{aggregation_token(self.aggregation, self.aggregation_weighting)}__clients-{self.num_clients}__seed-{self.seed}__"
            f"nm{format_noise(self.noise_multiplier)}__"
            f"clip{self.hyperparams.clipping_norm:g}__"
            f"rounds-{self.hyperparams.rounds}__"
            f"epochs-{self.hyperparams.local_epochs}__"
            f"{data_module_name}{model_suffix}"
        )

    def result_path(self, output_dir: Path) -> Path:
        """Return the reproduction runner's JSON result path."""
        return output_dir / f"{self.run_name()}.json"

    def runner_args(
        self,
        *,
        output_dir: Path,
        max_parallel_clients: int,
        client_cpus: float,
        checkpoint_rounds: tuple[int, ...] = (),
    ) -> tuple[str, ...]:
        """Return command-line arguments for the reproduction runner."""
        args = (
            "--data-module",
            self.data_module,
            "--model-module",
            self.model_module,
            "--num-clients",
            str(self.num_clients),
            "--partition",
            self.partition,
            "--dirichlet-alpha",
            str(self.dirichlet_alpha),
            "--privacy",
            self.privacy,
            "--aggregation",
            self.aggregation,
            "--seed",
            str(self.seed),
            "--noise-multiplier",
            str(self.noise_multiplier),
            "--clipping-norm",
            str(self.hyperparams.clipping_norm),
            "--rounds",
            str(self.hyperparams.rounds),
            "--local-epochs",
            str(self.hyperparams.local_epochs),
            "--batch-size",
            str(self.hyperparams.batch_size),
            "--learning-rate",
            str(self.hyperparams.learning_rate),
            "--initialization-epochs",
            str(self.hyperparams.initialization_epochs),
            "--weight-decay",
            str(self.hyperparams.weight_decay),
            "--lr-schedule",
            self.hyperparams.lr_schedule,
            "--max-parallel-clients",
            str(max_parallel_clients),
            "--client-cpus",
            str(client_cpus),
            "--output-dir",
            str(output_dir),
            "--run-name",
            self.run_name(),
        )
        if self.max_client_samples:
            args = (*args, "--max-client-samples", str(self.max_client_samples))
        if self.aggregation_weighting != "num-examples":
            args = (*args, "--aggregation-weighting", self.aggregation_weighting)
        if checkpoint_rounds:
            return (
                *args,
                "--checkpoint-rounds",
                *(str(value) for value in checkpoint_rounds),
            )
        return args


def format_noise(noise_multiplier: float) -> str:
    """Render a noise multiplier as a filename-safe token."""
    return f"{noise_multiplier:g}".replace(".", "p")
