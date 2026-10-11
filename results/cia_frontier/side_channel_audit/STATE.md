package: WP1
status: planning
gate: G1 = not evaluated yet
protocol: not yet written (results/cia_frontier/side_channel_audit/PROTOCOL.md)
machine: compute server (owner decision 2026-10-11; the roadmap planned the owner PC)
cost so far: 0 t-h; projection 50 t-h for the specified design (cap 15 t-h; see ESCALATION.md once written)
next: protocol, runner, attack code and tests; smoke on the owner PC; launch on the compute server
needs from others: none

## Owner decisions taken in this session (2026-10-11)

- WP1 runs on the compute server, not the owner PC: measured on the owner PC (Apple MPS), one 50-round EuroSAT
  trajectory takes about 56 min of training at 3 or 8 clients (67 s per round), so the specified design (228
  trajectories) projects to about 137 h there, about 50 t-h in roadmap units, against a 15 t-h cap.
- Every non-homogeneous partition is label-Dirichlet; the 3-client Alzheimer federation uses Dirichlet alpha 0.3 like
  the 5- and 8-client ones (not the paper's fixed Table 9 counts).
