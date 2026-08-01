"""Run configuration shared by client-scaling sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.reproduce.matrix.hyperparams import Hyperparams


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

    def run_name(self) -> str:
        """Build the complete deterministic name from this combo's parameters."""
        module_path = self.data_module.rsplit(":", 1)[0]
        data_module_name = module_path.rsplit(".", 1)[-1]
        return (
            f"{self.name_prefix}__{self.partition}__{self.privacy}__"
            f"{self.aggregation}__clients-{self.num_clients}__seed-{self.seed}__"
            f"nm{format_noise(self.noise_multiplier)}__"
            f"clip{self.hyperparams.clipping_norm:g}__"
            f"rounds-{self.hyperparams.rounds}__"
            f"epochs-{self.hyperparams.local_epochs}__"
            f"{data_module_name}"
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
            "--num-clients",
            str(self.num_clients),
            "--partition",
            self.partition,
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
            "--max-parallel-clients",
            str(max_parallel_clients),
            "--client-cpus",
            str(client_cpus),
            "--output-dir",
            str(output_dir),
            "--run-name",
            self.run_name(),
        )
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
