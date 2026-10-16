"""Figures and tables for FINDINGS.md, from analysis.json and the committed trajectories only.

    uv run --no-sync --with matplotlib python -m results.cia_frontier.residual_clipping.findings
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORDER = ("plain", "residual-fixed", "residual-median", "residual-q90")
COLORS = {"plain": "#222222", "residual-fixed": "#7f8c8d", "residual-median": "#2e86c1", "residual-q90": "#c0392b"}


def figure_leakage_vs_noise(a):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in ORDER:
        v = a["configurations"][name]
        x, y, (lo, hi) = v["noise_std_median_rounds_11_on"], v["clean"]["score"], v["clean_ci95"]["score"]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=COLORS[name], capsize=3, ms=7,
                    label=f"{name} (accuracy {v['accuracy_mean']:.3f})")
    ax.set_xlabel("Noise std on the average update (median over rounds 11-50)")
    ax.set_ylabel("Clean-view CIA score, top-decile targets (95% CI)")
    ax.set_title("Same formal guarantee; leakage falls as the noise grows with rho", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = HERE / "figures" / "g6_leakage_vs_noise.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_rho(a):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name in ORDER:
        rho = a["configurations"][name]["rho_median_by_round"]
        ax.plot(range(1, len(rho) + 1), rho, color=COLORS[name], lw=1.6, label=name,
                ls="--" if name in ("plain", "residual-fixed") else "-")
    ax.axhline(5.0, color="#555", lw=0.8)
    ax.text(12, 4.95, "C = 5 (plain clipping norm)", ha="left", va="top", fontsize=8)
    ax.set_xlabel("round")
    ax.set_ylabel("rho_t (median over trajectories)")
    ax.set_title("The private rho settles above C: no sensitivity reduction", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = HERE / "figures" / "g6_rho_by_round.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def tables(a):
    lines = ["| Mechanism | Score [95%] | AUC [95%] | TPR@5% | TPR@1% | Noisy score | Accuracy mean | IN accuracy | Range | "
             "rho (r11+) | Noise std (r11+) | Residuals clipped (r11+) | Median residual norm (r11+) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        v = a["configurations"][name]
        c, k = v["clean"], v["clean_ci95"]
        lines.append(
            f"| `{name}` | {c['score']:.3f} [{k['score'][0]:.3f}, {k['score'][1]:.3f}] | {c['auc']:.3f} "
            f"[{k['auc'][0]:.3f}, {k['auc'][1]:.3f}] | {c['tpr_at_0.05_fpr']:.3f} | {c['tpr_at_0.01_fpr']:.3f} | "
            f"{v['noisy']['score']:.3f} | {v['accuracy_mean']:.4f} | {v['accuracy_in'][0]:.4f} | "
            f"{v['accuracy_range'][0]:.4f}-{v['accuracy_range'][1]:.4f} | {v['rho_median_rounds_11_on']:.3f} | "
            f"{v['noise_std_median_rounds_11_on']:.5f} | {v['fraction_clipped_mean_rounds_11_on']:.3f} | "
            f"{v['residual_norm_median_rounds_11_on']:.3f} |")
    lines += ["", "Per-target clean-view scores (seed-target):", "",
              "| Mechanism | " + " | ".join(a["clusters"]) + " |", "|---|" + "---|" * len(a["clusters"])]
    for name in ORDER:
        scores = a["configurations"][name]["per_target_score"]
        lines.append(f"| `{name}` | " + " | ".join(f"{scores[c]:.2f}" for c in a["clusters"]) + " |")
    lines += ["", "Against plain clipping (gate rows are G6; the `residual-fixed` row is descriptive):", "",
              "| Mechanism | Score reduction [95%] | AUC reduction | Noise / plain | rho / plain | Accuracy difference | G6 verdict |",
              "|---|---|---|---|---|---|---|"]
    for name in ORDER[1:]:
        d = a["decomposition"][name]
        g = a["gate"]["candidates"].get(name)
        auc = f"{g['auc_reduction']['point']:.3f} [{g['auc_reduction']['ci95'][0]:.3f}, {g['auc_reduction']['ci95'][1]:.3f}]" if g else "n/a"
        lines.append(f"| `{name}` | {d['score_reduction']:.3f} [{d['score_reduction_ci95'][0]:.3f}, "
                     f"{d['score_reduction_ci95'][1]:.3f}] | {auc} | {d['noise_ratio_to_plain']:.2f} | "
                     f"{d['rho_ratio_to_plain']:.2f} | {d['accuracy_difference']:+.4f} | "
                     f"{g['verdict'] if g else 'diagnostic'} |")
    lines.append(f"\nG6 = **{a['gate']['G6']}**.")
    rows = ["", "Every trajectory (round-50 server accuracy, training and evaluation seconds):", "",
            "| Mechanism | Trajectory | Accuracy | Training s | Evaluation s |", "|---|---|---|---|---|"]
    total = 0.0
    for folder in sorted((HERE / "results" / "stage_a").glob("*/*")):
        manifest = json.loads((folder / "manifest.json").read_text())
        config = json.loads((folder / "residual_config.json").read_text())["config"]
        result = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        timing = json.loads((folder / "complete.json").read_text())
        side = "IN" if manifest["out_target"] is None else f"OUT-{manifest['out_target']}"
        accuracy = result["server_evaluate_metrics"][str(manifest["rounds"])]["accuracy"]
        rows.append(f"| `{config}` | {side} | {accuracy:.4f} | {timing['training_seconds']:.0f} | {timing['evaluation_seconds']:.0f} |")
        total += (timing["training_seconds"] + timing["evaluation_seconds"]) / 3600
    rows.append(f"\n{len(rows) - 5} trajectories, {total:.2f} h of trajectory wall time.")
    return "\n".join(lines + rows) + "\n"


def main():
    a = json.loads((HERE / "analysis.json").read_text())
    print(figure_leakage_vs_noise(a))
    print(figure_rho(a))
    (HERE / "findings_tables.md").write_text("# WP6 tables (generated by `findings.py`)\n\n" + tables(a))


if __name__ == "__main__":
    main()
