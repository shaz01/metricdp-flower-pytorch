# CIA roadmap v2 (2026-10-16)

Successor to the 2026-10 CIA programme (`research/cia_review_2026-10/`, closed with `SYNTHESIS.md`). Status: approved by the owner on
2026-10-17 (`ROADMAP.md` Section 1 defaults accepted); agents follow `ROADMAP.md`. Nothing is running. No training was run to produce these files.

| File | Contents |
|---|---|
| [REVIEW.md](REVIEW.md) | Re-examination of the v1 programme against its committed trajectories. Main finding: the IN and OUT worlds of a seed draw the same DP noise, so v1's paired leakage numbers under DP largely cancel the noise. Also covers shared training randomness, mismatched WP1 references and the unreachable formal regime, and gives each v1 conclusion a status |
| [ROADMAP.md](ROADMAP.md) | New roadmap: owner decisions, the binding measurement standard v2, work packages M0-M2, A1-A2, B1-B5 and C1 with gates and caps, decision flow, agent ownership and prompts |
| [scripts/measurement_audit.py](scripts/measurement_audit.py) | Rebuilds every table and the figure below from committed result files (`uv run --with pandas --with scipy --with matplotlib python research/cia_roadmap_v2/scripts/measurement_audit.py`, from the repository root) |
| `tables/noise_coupling_correlations.csv` | Realized-noise-energy correlation between trajectory pairs (WP1 3-8 clients; consensus-KL step 3) |
| `tables/decoupled_score_bound.csv` | Paired score as measured vs the one-round independent-noise upper bound, global DP 1-4x |
| `tables/cross_seed_scores_wp2.csv` | WP2 paired scores with IN and OUT from the same vs different seeds (fixed layout, no DP) |
| `tables/label_skew_vs_exposure.csv` | Size, label skew and label share against per-target exposure (frontier targets) |
| `figures/measurement_audit.png` | The three measurement findings in one figure |

Derived tables live in `tables/`, not `data/`, because the repository-wide `.gitignore` rule `data/` would hide them.
