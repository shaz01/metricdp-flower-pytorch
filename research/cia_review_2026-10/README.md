# CIA literature review and next-direction proposal (2026-10)

Status: written 2026-10-10; approved the same day for autonomous execution under [agent_roadmap.md](agent_roadmap.md); nothing running at the time of writing. Entry point for a new session:
[`../../HANDOFF.md`](../../HANDOFF.md). Everything here was derived from the two PDFs in `papers/`, committed result
files, and a verified literature corpus; no training was run.

## Documents

| File | Contents |
|---|---|
| [agent_roadmap.md](agent_roadmap.md) | Execution plan for autonomous agents: binding owner decisions, work packages WP0-WP6, gates, file ownership, escalation, prompts |
| [project_state.md](project_state.md) | Goal and baseline, codebase map, chronology, why client-specific noise failed, new log analyses, constraints |
| [literature_review.md](literature_review.md) | Review of client-level inference threats, metric and data-dependent privacy, client-level DP, distributional privacy (GMIP), signal-reduction defenses, measurement practice; gaps mapped to D1-D6 |
| [research_directions.md](research_directions.md) | Ranked directions D1-D6, new evidence E1-E6, reviewer objections and responses, kill criteria, staged experiments, compute |
| [paper_table.csv](paper_table.csv), [references.bib](references.bib) | 97 references (all DOIs resolved, none retracted at the time of checking): family, protected unit, guarantee, role, relevance |
| [notes/local_papers_notes.md](notes/local_papers_notes.md) | Close reading of `papers/Initial-Paper.pdf` and `papers/Gaussian Membership Inference Privacy.pdf` |
| [notes/broad_survey_notes.md](notes/broad_survey_notes.md) | Broad sweep and the gap table |
| `notes/targeted_corpus.csv`, `notes/search_logs/*.json` | Search queries and raw hits (OpenAlex, arXiv) |

## Data and figures

| File | Built by | Contents |
|---|---|---|
| `data/results_history.csv`, `figures/results_history.png` | `scripts/analyze_project_history.py` | Project results over time |
| `data/eurosat_frontier_scores.csv`, `data/eurosat_frontier_gap_by_round.csv`, `figures/client_noise_vs_baseline.png` | same | Per-target frontier scores and gap by round |
| `data/metric_noise_3client.csv`, `data/metric_noise_48client_counterfactual.csv`, `figures/metric_noise_side_channel.png` | same | Metric-privacy noise level IN vs OUT |
| `data/gmip_regimes.csv`, `figures/gmip_regimes.png` | `scripts/gmip_regime_calculator.py` | GMIP vs GDP noise in the project's regimes |
| `data/defense_attack_matrix.csv`, `data/taxonomy_tags.json`, `figures/defense_attack_coverage.png` | `scripts/build_taxonomy_figure.py` | Defense-by-attack coverage of the corpus |
| `data/noise_level_side_channel_48client.csv`, `data/dispersion_statistic_loo_sensitivity.csv`, `data/target_idiosyncrasy_*.csv`, `data/oracle_projection_snr_proxy.csv`, `figures/direction_evidence.png` | `scripts/analyze_direction_evidence.py` | Evidence E1-E6 in `research_directions.md` |
| `data/direction_reviewer_pass.json`, `data/direction_scores.csv`, `figures/direction_scores.png` | `scripts/build_direction_scores.py` | Adversarial reviewer output and direction scores |

All scripts run from the repository root (`uv run --with pandas --with scipy --with matplotlib python
research/cia_review_2026-10/scripts/<name>.py`) and read only committed files under `results/`. A re-run on
2026-10-10 reproduced all 14 CSV tables byte for byte.

## Limits

Literature coverage is OpenAlex and arXiv plus four web searches. Read depth per entry is in the `read_scope` column of
`paper_table.csv`: full text only for the two local PDFs; abstract for 40 entries; metadata only for 35; the rest rely on
the repository's earlier review (`research/literature_review/`). "Not found in the corpus" statements are search results, not proofs of absence. The direction
scores are judgements (author and one model-based reviewer), not measurements.
