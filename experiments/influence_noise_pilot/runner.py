"""Influence-directed matched-energy noise pilot (EuroSAT). Plans unless --execute.

EMPIRICAL feasibility pilot, NO DP CLAIM. Reuses the new-AUC-frontier protocol
(label-Dirichlet alpha .3, 48 canonical clients, 100 rounds, original EuroSAT
hyperparameters, fixed target panel, IN/OUT removal adjacency, every-round
clean/noisy shadow losses) and replaces only the aggregate noise with
``metricdp_pytorch.influence_noise``. ``f = 0`` is the matched isotropic control,
run with the same code path and random-number scheme as ``f > 0``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

from experiments.auc_frontier.runner import ADJACENCIES, FrontierCombo, atomic_json, build_combos
from experiments.cia.scripts.eurosat_remove import HYPERPARAMS
from metricdp_pytorch.influence_noise import RNG_DOMAIN_TAG, isotropic_stdv

ALPHA = 0.3
CLIENTS = 48
NOISE_RATIO = 0.001546
# Declared before any outcome: the influence cap equals the update clipping norm C.
# Motivation: ||u_i||, ||g|| <= C, so ||d_i|| <= 2C w_i/(1-w_i); the cap only
# bounds pathological weight concentration and is not tuned on results.
INFLUENCE_CAP = HYPERPARAMS.clipping_norm
PARAMETER_COUNT = 289_194  # EurosatCNN, float parameters only; no buffers
PROTOCOL_SCHEMA = 1


@dataclass(frozen=True)
class InfluenceCombo(FrontierCombo):
    influence_fraction: float
    influence_cap: float

    def runner_args(self, **kwargs):
        return (*super().runner_args(**kwargs),
                "--influence-fraction", repr(self.influence_fraction),
                "--influence-cap", repr(self.influence_cap))


def build_influence_combos(*, fraction, seeds, targets, adjacency="both", out_targets=None,
                           alpha=ALPHA, clients=CLIENTS, ratio=NOISE_RATIO, cap=INFLUENCE_CAP):
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("--fraction must be in [0, 1]")
    base = build_combos(alpha=alpha, seeds=seeds, targets=targets, clients=clients,
                        privacy="global-dp", ratios=[ratio], adjacency=adjacency,
                        out_targets=out_targets)
    out = []
    for combo in base:
        values = {field.name: getattr(combo, field.name) for field in fields(FrontierCombo)}
        side = "in" if combo.out_target is None else f"out-{combo.out_target}"
        values.update(privacy="influence-noise",
                      name_prefix=f"eurosat-dirichlet-a{alpha!r}-{side}-r{ratio!r}-f{fraction!r}-cap{cap!r}")
        out.append(InfluenceCombo(**values, influence_fraction=fraction, influence_cap=cap))
    return out


def protocol(combo: InfluenceCombo) -> dict:
    """Pinned per-trajectory protocol; any change needs a new output directory."""
    active = [c for c in range(combo.canonical_clients) if c != combo.out_target]
    return {
        "schema": PROTOCOL_SCHEMA,
        "mechanism": "influence-directed matched-energy Gaussian aggregate noise",
        "dp_claim": False,
        "notes": ("Data-dependent covariance: NOT differential privacy. d_i is round-local "
                  "leave-one-out influence, not trajectory removal sensitivity."),
        "covariance": "tau^2 [(1-f) I + f D M / trace(M)], M = sum_i capped d_i d_i^T",
        "influence": "d_i = (w_i/(1-w_i)) (u_i - g), g = sum_i w_i u_i, w_i = n_i/sum n",
        "fraction": combo.influence_fraction,
        "influence_cap": combo.influence_cap,
        "tau": isotropic_stdv(combo.noise_multiplier, combo.hyperparams.clipping_norm, combo.num_clients),
        "tau_convention": "Flower compute_stdv(noise_multiplier, clip, active clients)",
        "noise_ratio": combo.noise_ratio,
        "noise_multiplier": combo.noise_multiplier,
        "clipping_norm": combo.hyperparams.clipping_norm,
        "active_clients": combo.num_clients,
        "parameter_count": PARAMETER_COUNT,
        "expected_noise_sq_norm_per_round": isotropic_stdv(
            combo.noise_multiplier, combo.hyperparams.clipping_norm, combo.num_clients) ** 2 * PARAMETER_COUNT,
        "degenerate_rule": "trace(M) == 0 or f == 0 -> isotropic tau^2 I",
        "rng": {"generator": "numpy PCG64 via SeedSequence([tag, seed, round])",
                "tag": RNG_DOMAIN_TAG, "draw_order": "xi (D) then eta (n), always both"},
        "client_order": "ascending OUT-federation client-id; canonical ids below",
        "canonical_client_ids": active,
        "seed": combo.seed,
        "run_name": combo.run_name(),
    }


def execute(combos, targets, output, max_parallel_clients):
    from experiments.auc_frontier.runner import execute as frontier_execute
    for combo in combos:
        folder = output / combo.run_name()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "influence_protocol.json"
        pinned = protocol(combo)
        if path.exists() and json.loads(path.read_text()) != pinned:
            raise ValueError(f"Influence protocol mismatch: {folder}; use a separate output directory")
        atomic_json(path, pinned)
        frontier_execute([combo], targets, output, max_parallel_clients)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, required=True, help="orientation fraction f")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--targets", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--adjacency", choices=ADJACENCIES, default="both")
    parser.add_argument("--out-targets", type=int, nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("results/influence_noise_pilot"))
    args = parser.parse_args(argv)
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    combos = build_influence_combos(fraction=args.fraction, seeds=args.seeds, targets=args.targets,
                                    adjacency=args.adjacency, out_targets=args.out_targets)
    print(json.dumps({"training_runs": len(combos), "execute": args.execute,
                      "fraction": args.fraction, "target_panel": args.targets,
                      "adjacency": args.adjacency, "protocols": [protocol(c) for c in combos]},
                     indent=2))
    if args.execute:
        execute(combos, args.targets, args.output, args.max_parallel_clients)


if __name__ == "__main__":
    main()
