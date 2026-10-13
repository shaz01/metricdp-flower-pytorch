"""Tables and figures for FINDINGS.md, read from committed files only (no training, no new statistics).

Inputs: ``analysis.json`` (written by ``analyze.py``) and the committed trajectory folders under ``results/``
(``manifest.json``, ``partitions.json`` and the run JSON's round-50 server accuracy).

Outputs (beside this file):
  figures/g1a_noise_level_auc.png   blind and oracle noise-level AUC per cell, top-decile targets, with G1(a)
  figures/g1b_tpr_at_5pct_fpr.png   increment and loss TPR at 5% FPR per cell, all targets, with G1(b)
  findings_tables.md                the full tables pasted into FINDINGS.md

Run from the repository root:
    uv run --with matplotlib python -m results.cia_frontier.side_channel_audit.findings
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATASETS = ("eurosat", "alzheimer")
CLIENTS = (3, 5, 8)
MECHANISMS = ("global-dp", "metric-privacy")
ATTACKERS = ("noise_level", "oracle_noise_level", "increment", "loss")
METRICS = ("auc", "tpr_at_5pct_fpr", "tpr_at_1pct_fpr")


def fmt(summary):
    low, high = summary["ci95"]
    return f"{summary['value']:.3f} [{low:.3f}, {high:.3f}]"


def trajectories():
    """One row per committed trajectory: dataset, clients, mechanism, seed, OUT target (None = IN), accuracy, records."""
    rows = []
    for manifest_path in sorted(RESULTS.rglob("manifest.json")):
        folder = manifest_path.parent
        if not (folder / "complete.json").exists():
            raise ValueError(f"Incomplete trajectory: {folder}")
        manifest = json.loads(manifest_path.read_text())
        run = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        dataset, _, mechanism = folder.relative_to(RESULTS).parts[:3]
        rows.append(dict(
            dataset=dataset, clients=manifest["clients"], mechanism=mechanism, seed=manifest["seed"],
            out=manifest["out_target"],
            accuracy=float(run["server_evaluate_metrics"][str(manifest["rounds"])]["accuracy"]),
            records=[c["records"] for c in json.loads((folder / "partitions.json").read_text())["clients"]],
        ))
    return rows


def cell_entry(analysis, dataset, clients, mechanism):
    for entry in analysis["cells"]:
        if (entry["dataset"], entry["clients"], entry["privacy"]) == (dataset, clients, mechanism):
            return entry
    raise KeyError((dataset, clients, mechanism))


def figures(analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{d}\n{c} clients" for d in DATASETS for c in CLIENTS]
    x = list(range(len(labels)))

    def series(mechanism, view, attacker, metric):
        values, low, high = [], [], []
        for dataset in DATASETS:
            for clients in CLIENTS:
                s = cell_entry(analysis, dataset, clients, mechanism)["attacks"][view][attacker][metric]
                values.append(s["value"])
                low.append(s["value"] - s["ci95"][0])
                high.append(s["ci95"][1] - s["value"])
        return values, [low, high]

    # Figure 1: G1(a), noise-level AUC for top-decile targets.
    g1a = analysis["gate"]["G1a"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for offset, (mechanism, attacker, label, color, marker) in zip(
            (-0.2, 0.0, 0.2),
            (("metric-privacy", "noise_level", "metric privacy, blind (G1(a) attack)", "#1f6fb4", "o"),
             ("metric-privacy", "oracle_noise_level", "metric privacy, oracle (true noise std)", "#7a9cc6", "s"),
             ("global-dp", "noise_level", "global DP, blind (control)", "#c4512b", "^"))):
        values, errors = series(mechanism, "clean/top_decile", attacker, "auc")
        ax.errorbar([i + offset for i in x], values, yerr=errors, fmt=marker, color=color, capsize=3, label=label)
    ax.axhline(0.55, color="black", linestyle="--", linewidth=1, label="G1(a) threshold 0.55")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("AUC (top-decile target, rounds 1-50)")
    low, high = g1a["ci95"]
    ax.set_title(f"G1(a): pooled blind metric-privacy AUC {g1a['value']:.3f} [{low:.3f}, {high:.3f}]: "
                 f"{g1a['decision']}", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "g1a_noise_level_auc.png", dpi=150)
    plt.close(fig)

    # Figure 2: G1(b), TPR at 5% FPR of the increment and loss attackers, all targets, clean view.
    g1b = analysis["gate"]["G1b"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, mechanism in zip(axes, MECHANISMS):
        for offset, (attacker, label, color, marker) in zip(
                (-0.12, 0.12),
                (("increment", "increment attacker", "#1f6fb4", "o"),
                 ("loss", "loss statistic (current)", "#c4512b", "s"))):
            values, errors = series(mechanism, "clean/all", attacker, "tpr_at_5pct_fpr")
            ax.errorbar([i + offset for i in x], values, yerr=errors, fmt=marker, color=color, capsize=3, label=label)
        ax.axhline(0.05, color="grey", linestyle=":", linewidth=1, label="chance (0.05)")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(mechanism, fontsize=10)
    axes[0].set_ylabel("TPR at 5% FPR (all targets, clean view)")
    axes[0].legend(fontsize=8, loc="best")
    low, high = g1b["ci95"]
    fig.suptitle(f"G1(b): increment minus loss, mean over 12 cells {g1b['value']:+.3f} [{low:+.3f}, {high:+.3f}]: "
                 f"{g1b['decision']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "g1b_tpr_at_5pct_fpr.png", dpi=150)
    plt.close(fig)


def tables(analysis, rows):
    out = []
    g1a, g1b = analysis["gate"]["G1a"], analysis["gate"]["G1b"]
    out += ["### Gate G1", "",
            "| Part | Statistic | Value [95% interval] | Decision |", "|---|---|---|---|",
            f"| G1(a) | blind noise-level AUC, metric privacy, top-decile targets, pooled over 6 cells | {fmt(g1a)} | "
            f"{g1a['decision']} |",
            f"| (attribution) | oracle noise-level AUC, metric privacy, same targets | "
            f"{fmt(g1a['oracle_metric_privacy_auc'])} | |",
            f"| (attribution) | blind noise-level AUC, global DP, same targets (update-norm channel) | "
            f"{fmt(g1a['blind_global_dp_auc'])} | |",
            f"| G1(b) | increment minus loss, TPR at 5% FPR, all targets, clean view, mean over 12 cells | "
            f"{g1b['value']:+.3f} [{g1b['ci95'][0]:+.3f}, {g1b['ci95'][1]:+.3f}] | {g1b['decision']} |"]
    if "attribution" in g1a:
        out += ["", f"Attribution rule (pre-run revision): {g1a['attribution']}."]

    for view in ("clean", "noisy"):
        for subgroup, title in (("top_decile", "top-decile targets"), ("all", "all targets")):
            out += ["", f"### Attacks per cell: {view} view, {title}", "",
                    "AUC; TPR at 5% FPR; TPR at 1% FPR; value [95% cluster-bootstrap interval].", "",
                    "| Dataset | Clients | Mechanism | Attacker | AUC | TPR@5% | TPR@1% |", "|---|---|---|---|---|---|---|"]
            for dataset in DATASETS:
                for clients in CLIENTS:
                    for mechanism in MECHANISMS:
                        attacks = cell_entry(analysis, dataset, clients, mechanism)["attacks"][f"{view}/{subgroup}"]
                        for attacker in ATTACKERS:
                            out.append(f"| {dataset} | {clients} | {mechanism} | {attacker} | "
                                       + " | ".join(fmt(attacks[attacker][m]) for m in METRICS) + " |")

    out += ["", "### Top-decile targets", "",
            "The largest client by local dataset size in each federation and seed (`ceil(0.1 n)` = 1 here). The "
            "partition depends only on dataset, client count and seed, so both mechanisms share it.", "",
            "| Dataset | Clients | Seed | Top-decile target | Its records | Records of every client |", "|---|---|---|---|---|---|"]
    seen = set()
    for row in rows:
        key = (row["dataset"], row["clients"], row["seed"])
        if row["out"] is not None or key in seen:
            continue
        seen.add(key)
        records = row["records"]
        top = max(range(len(records)), key=lambda i: (records[i], i))  # analyze.top_decile tie order
        out.append(f"| {row['dataset']} | {row['clients']} | {row['seed']} | {top} | {records[top]:,} | "
                   + ", ".join(f"{r:,}" for r in records) + " |")

    out += ["", "### Server test accuracy at round 50, every trajectory", "",
            "IN = all clients; OUT-t = the federation without client t. Accuracy as a fraction.", ""]
    for dataset in DATASETS:
        width = max(CLIENTS)
        out += ["", f"**{dataset}**", "",
                "| Clients | Mechanism | Seed | IN | " + " | ".join(f"OUT-{t}" for t in range(width)) + " |",
                "|---|---|---|---|" + "---|" * width]
        for clients in CLIENTS:
            for mechanism in MECHANISMS:
                for seed in (42, 43, 44):
                    mine = {r["out"]: r["accuracy"] for r in rows
                            if (r["dataset"], r["clients"], r["mechanism"], r["seed"]) == (dataset, clients, mechanism, seed)}
                    cells = [f"{mine[t]:.4f}" if t in mine else "" for t in range(width)]
                    out.append(f"| {clients} | {mechanism} | {seed} | {mine[None]:.4f} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    analysis = json.loads((HERE / "analysis.json").read_text())
    rows = trajectories()
    if len(rows) != 228:
        raise ValueError(f"Expected 228 trajectories, found {len(rows)}")
    figures(analysis)
    (HERE / "findings_tables.md").write_text(tables(analysis, rows))
    print(f"{len(rows)} trajectories; wrote figures/ and findings_tables.md")


if __name__ == "__main__":
    main()
