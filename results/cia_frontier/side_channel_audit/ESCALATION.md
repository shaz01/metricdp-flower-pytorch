# Escalation: WP1 cost is about 3.4 times its cap

Raised 2026-10-11 by Agent A before any protocol run. Rule: agent_roadmap.md Section 4.4 item 2 (projected cost
exceeds the package cap by more than 25%).

## Evidence

Timing smoke runs on the owner PC (Apple MPS), 3 rounds each, frontier hyperparameters (5 local epochs, 20
initialization epochs), equal weights, metric privacy:

| Federation | Seconds per round |
|---|---|
| EuroSAT, 3 clients | 67 |
| EuroSAT, 8 clients | 66 |
| Alzheimer MRI, 3 clients | 12-14 |
| Alzheimer MRI, 8 clients | 12-15 |

Every round trains on the whole dataset for 5 local epochs whatever the number of clients, so small federations are
not cheaper per round than the 48-client frontier. The specified WP1 design (2 datasets x 3, 5, 8 clients x 2
mechanisms x 3 seeds, every client a target, 50 rounds) is 228 trajectories: **50.6 t-h** (EuroSAT 42.2, Alzheimer 8.4)
against a cap of 15 t-h, about 137 h of wall time on the owner PC. The research document's estimate (about 10 t-h)
assumed small federations were cheap.

## Options put to the owner

1. 20 rounds (the paper's CIA horizon) and 3 + 8 clients: about 13.7 t-h, within the cap.
2. 20 rounds, 3 + 5 + 8 clients: about 20 t-h.
3. The design as specified (50 rounds) with a raised cap, or with the EuroSAT part on the compute server.
4. 50 rounds, EuroSAT at 3 clients only, Alzheimer at 3, 5, 8: about 17 t-h.

## Decision so far (owner, 2026-10-11)

"Maybe we can move to the server, let's get ready everything for server movement": WP1 moves to the compute server
with the specified design, and Agent A prepares everything for it (protocol, runner, analysis, tests, smoke runs,
launch instructions). **Open:** explicit confirmation of the 50.6 t-h budget (3.4 times the cap). No protocol run
starts before it is recorded in `STATE.md`.
