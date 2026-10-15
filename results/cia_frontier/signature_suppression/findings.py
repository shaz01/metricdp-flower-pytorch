"""Figures and tables for FINDINGS.md, from committed files only (analysis JSONs and trajectory folders).

    uv run --no-sync --with matplotlib python -m results.cia_frontier.signature_suppression.findings
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "stage_a"
ANALYSES = {"42": "analysis.json", "42-43": "analysis_seeds42-43.json", "42-45": "analysis_seeds42-45.json"}
ORDER = ("vanilla", "vanilla-size-weighted", "weight-decay-5e-4", "local-epochs-1", "fedprox-mu-0.01", "fedprox-mu-0.1",
         "fedprox-mu-1", "fixed-steps", "consensus-kl-0.3", "consensus-kl-1")
ROLE = {"vanilla": "reference", "vanilla-size-weighted": "secondary reference", "weight-decay-5e-4": "control",
        "local-epochs-1": "control"}


# Label offsets (points) where the default would overlap a neighbour.
LABEL_OFFSETS = {"weight-decay-5e-4": (-6, -12), "fixed-steps": (-6, 6), "vanilla": (6, 6),
                 "vanilla-size-weighted": (6, -16), "fedprox-mu-0.1": (-8, -3)}


def load(name):
    return json.loads((HERE / ANALYSES[name]).read_text())


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def ci(pair, digits=3):
    return f"[{pair[0]:.{digits}f}, {pair[1]:.{digits}f}]"


def figure_seed42(a):
    """Claim 1: leakage (clean-view score) against accuracy, seed 42, every configuration."""
    configs = a["configurations"]
    vanilla = configs["vanilla"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvspan(vanilla["accuracy_mean"] - 0.01, 1, color="#d9ead3", alpha=0.6, lw=0,
               label="within 1 pp of vanilla accuracy")
    ax.axhline(vanilla["clean"]["score"] - 0.05, color="#555", ls="--", lw=1, label="vanilla score - 0.05")
    for name in ORDER:
        v = configs[name]
        x, y, (lo, hi) = v["accuracy_mean"], v["clean"]["score"], v["clean_ci95"]["score"]
        color = "#222" if name in ROLE else ("#c0392b" if name.startswith("consensus") else
                                             "#2e86c1" if name.startswith("fedprox") else "#8e44ad")
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=color, capsize=3, ms=6)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=LABEL_OFFSETS.get(name, (6, 4)), fontsize=8,
                    ha="right" if LABEL_OFFSETS.get(name, (6, 4))[0] < 0 else "left")
    ax.set_xlabel("Server test accuracy at round 50 (mean of IN and 5 OUT trajectories)")
    ax.set_ylabel("Clean-view CIA score (95% bootstrap over targets)")
    ax.set_title("Seed 42: no variant cuts leakage by 0.05 within 1 pp of vanilla accuracy", fontsize=10)
    ax.set_xlim(0.66, 0.90)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path = HERE / "figures" / "g2_score_vs_accuracy_seed42.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_fixed_steps(analyses):
    """Claim 2: the fixed-steps reduction as seeds are added."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = []
    for i, (name, a) in enumerate(analyses.items()):
        record = a["gate"]["candidates"]["fixed-steps"]["reduction_score"]
        lo, hi = record["ci95"]
        ax.errorbar(i, record["point"], yerr=[[record["point"] - lo], [hi - record["point"]]], fmt="o",
                    color="#8e44ad", capsize=4, ms=7)
        labels.append(f"seeds {name}\n({len(a['clusters'])} target-seed pairs)")
    ax.axhline(0.05, color="#c0392b", ls="--", lw=1, label="G2 threshold 0.05")
    ax.axhline(0.0, color="#555", lw=0.8)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("Score reduction vs vanilla (95% paired bootstrap)")
    ax.set_title("Fixed local steps: no leakage reduction once four seeds are pooled", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = HERE / "figures" / "g2_fixed_steps_by_seeds.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def configuration_table(a, configs):
    rows = ["| Configuration | Role | Score [95%] | AUC [95%] | TPR@5% | TPR@1% | Noisy score | Top-decile score | "
            "Top-decile AUC | Accuracy mean | IN accuracy | Range | Client acc. min / mean (IN) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in configs:
        v = a["configurations"][name]
        c, k = v["clean"], v["clean_ci95"]
        clients = "; ".join(f"{fmt(x['min'])} / {fmt(x['mean'])}" for x in v["client_accuracy_in"])
        rows.append(f"| `{name}` | {ROLE.get(name, 'candidate')} | {fmt(c['score'])} {ci(k['score'])} | "
                    f"{fmt(c['auc'])} {ci(k['auc'])} | {fmt(c['tpr_at_0.05_fpr'])} | {fmt(c['tpr_at_0.01_fpr'])} | "
                    f"{fmt(v['noisy']['score'])} | {fmt(v['top_decile']['score'])} | {fmt(v['top_decile']['auc'])} | "
                    f"{fmt(v['accuracy_mean'], 4)} | {', '.join(fmt(x, 4) for x in v['accuracy_in'])} | "
                    f"{fmt(v['accuracy_range'][0], 4)}-{fmt(v['accuracy_range'][1], 4)} | {clients} |")
    return rows


def gate_table(a):
    rows = ["| Candidate | Accuracy | Within 1 pp | Score reduction [95%] | AUC reduction [95%] | Part 1 | "
            "vs weight decay [95%] | vs 1 local epoch [95%] | Part 2 | Source |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, g in a["gate"]["candidates"].items():
        beats = g.get("beats_controls", {})
        cell = lambda c: f"{fmt(beats[c]['point'])} {ci(beats[c]['ci95'])}" if c in beats else "n/a"
        rows.append(f"| `{name}` | {fmt(g['accuracy'], 4)} | {'yes' if g['accuracy_condition'] else 'no'} | "
                    f"{fmt(g['reduction_score']['point'])} {ci(g['reduction_score']['ci95'])} | "
                    f"{fmt(g['reduction_auc']['point'])} {ci(g['reduction_auc']['ci95'])} | {g['part1']} | "
                    f"{cell('weight-decay-5e-4')} | {cell('local-epochs-1')} | {g['part2']} | "
                    f"{'seed 42 (carried)' if g.get('carried_from_seed_42') else 'this analysis'} |")
    rows.append(f"\nG2 = **{a['gate']['G2']}**; passing candidates: {a['gate']['passing'] or 'none'}.")
    return rows


def per_target_table(a, configs):
    clusters = a["clusters"]
    rows = ["| Configuration | " + " | ".join(clusters) + " |", "|---|" + "---|" * len(clusters)]
    for name in configs:
        scores = a["configurations"][name]["per_target_score"]
        rows.append(f"| `{name}` | " + " | ".join(fmt(scores[c]["clean"], 2) for c in clusters) + " |")
    return rows


def loo_table(a, configs):
    targets = ("23", "7", "3", "0", "1")
    rows = ["| Configuration | " + " | ".join(f"target {t}" for t in targets) + " |", "|---|" + "---|" * len(targets)]
    for name in configs:
        medians = a["configurations"][name]["loo_residual_median_in"][0]
        rows.append(f"| `{name}` | " + " | ".join(fmt(medians[t], 2) for t in targets) + " |")
    return rows


def trajectory_table():
    rows = ["| Configuration | Seed | Trajectory | Clients | Accuracy (round 50) | Training s | Evaluation s |",
            "|---|---|---|---|---|---|---|"]
    entries = []
    for manifest_path in sorted(RESULTS.rglob("manifest.json")):
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        config = json.loads((folder / "suppression_config.json").read_text())["config"]
        result = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        timing = json.loads((folder / "complete.json").read_text())
        accuracy = result["server_evaluate_metrics"][str(manifest["rounds"])]["accuracy"]
        side = "IN" if manifest["out_target"] is None else f"OUT-{manifest['out_target']}"
        entries.append((ORDER.index(config), manifest["seed"], manifest["out_target"] is not None,
                        manifest["out_target"] or 0, config, side, result["metadata"]["num_clients"], accuracy,
                        timing["training_seconds"], timing["evaluation_seconds"]))
    total = 0.0
    for *_, config, side, clients, accuracy, train, evaluate in sorted(entries):
        seed = _[1]
        rows.append(f"| `{config}` | {seed} | {side} | {clients} | {accuracy:.4f} | {train:.0f} | {evaluate:.0f} |")
        total += (train + evaluate) / 3600
    rows.append(f"\n{len(entries)} trajectories, {total:.2f} h of trajectory wall time (training + evaluation).")
    return rows


def main():
    analyses = {name: load(name) for name in ANALYSES}
    seed42 = analyses["42"]
    figures = [figure_seed42(seed42), figure_fixed_steps(analyses)]
    extended = ("vanilla", "weight-decay-5e-4", "local-epochs-1", "fixed-steps")
    lines = ["# WP2 findings tables (generated by `findings.py` from committed files)", "",
             "## Seed 42, all configurations (`analysis.json`, 5 clusters)", "",
             *configuration_table(seed42, ORDER), "", "### Gate G2 at seed 42", "", *gate_table(seed42), "",
             "### Per-target clean-view scores, seed 42", "", *per_target_table(seed42, ORDER), "",
             "### Mechanism check (descriptive): median leave-one-out residual norm of each target, IN trajectory, "
             "seed 42", "", *loo_table(seed42, ORDER), ""]
    for name in ("42-43", "42-45"):
        a = analyses[name]
        lines += [f"## Seeds {name} (`{ANALYSES[name]}`, {len(a['clusters'])} clusters; extended configurations)", "",
                  *configuration_table(a, extended), "", f"### Gate G2 over seeds {name}", "", *gate_table(a), "",
                  f"### Per-target clean-view scores, seeds {name}", "", *per_target_table(a, extended), ""]
    lines += ["## Every trajectory", "", *trajectory_table(), ""]
    (HERE / "findings_tables.md").write_text("\n".join(lines))
    print("\n".join(str(f.relative_to(HERE)) for f in figures))


if __name__ == "__main__":
    main()
