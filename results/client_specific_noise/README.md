# Client-specific noise construction calculations

`analytical_heterogeneity.json` contains deterministic two-dimensional reconstruction-risk calculations, not trained-model results or CIA scores. No FL experiment is declared finished by these artifacts.

Regenerate with `uv run python research/calculations/heterogeneous_profile_risk.py` from the repository root. Assumptions, boundaries and interpretation are in [the analytical findings note](../../research/proposals/2026-10-04_heterogeneity_findings.md). The saved grid search does not certify continuous global optimality.

`protected_history_two_round.json` contains two-round stationary-update risk calculations, complete controls and fixed-seed sampler/formula validation. Regenerate with `uv run python research/calculations/protected_history_two_round.py`. Read [the findings and limits](../../research/proposals/2026-10-04_protected_history_findings.md) before interpreting its conditional improvements.
