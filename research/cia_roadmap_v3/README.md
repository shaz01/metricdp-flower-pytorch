# CIA roadmap v3 (draft, 2026-10-30)

Follow-up to roadmap v2 (`research/cia_roadmap_v2/`, complete 2026-10-30). Status: **approved by the owner on
2026-10-30** (ROADMAP.md Section 1 defaults; decision 1 in Section 10). Agents follow `ROADMAP.md`.

Read in this order:

| File | What it is |
|---|---|
| `HANDOFF_AUDIT.md` | Check of the v2 results handoff against the committed reports and run manifests: 249 of 260 numbers match their cited source and the other 11 resolve; 8 corrections (C1-C8), including the shared-noise attribution (C3) and the experiment design (C1). The handoff itself (`research/cia_roadmap_v2/RESULTS_HANDOFF.md`) now carries the corrections |
| `NOVELTY_CHECK.md` | Answers to the handoff's nine literature questions. Q1, Q5 and Q6 not found; Q2, Q3, Q4 and Q9 partial; Q7 and Q8 published. Evidence in `tables/novelty_evidence.csv`, BibTeX in `references_v3.bib` |
| `reanalysis/` | Zero-GPU checks on committed v2 runs, with the protocol fixed first: a sampling-aware attacker (the composite's win survives), an early-stopping control (inconclusive by rule; every bracketed comparison favours subsampling) and a per-candidate breakdown. `FINDINGS.md`, `analysis.json`, `reanalyze.py`, `tests/` |
| `PAPER_OPTIONS.md` | Audit, defense or hybrid paper: what each has, lacks and costs. Recommends the hybrid with the audit as its spine |
| `DIRECTIONS.md` | Eight future directions, scored, each with a first decisive test (`tables/direction_scores.csv`) |
| `ROADMAP.md` | The v3 programme: owner decisions (proposed defaults), measurement standard v3, packages P0-P6 and R1 with gates and caps, agent prompts |

Figures are in `figures/`:
- `novelty_matrix.png`;
- `sampling_aware_attacker.png`;
- `early_stopping_control.png`;
- `paper_evidence_map.png`;
- `direction_scores.png`;
- `v3_package_flow.png`.

Reproduce the re-analysis from the repo root (CPU, about 1 minute):

```
PYTHONPATH="$PWD" uv run --no-sync python research/cia_roadmap_v3/reanalysis/reanalyze.py
PYTHONPATH="$PWD" uv run --no-sync python research/cia_roadmap_v3/reanalysis/exploratory.py
PYTHONPATH="$PWD" uv run --no-sync python -m pytest research/cia_roadmap_v3/reanalysis/tests/
```
