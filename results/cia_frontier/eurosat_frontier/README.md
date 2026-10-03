# EuroSAT per-client frontier (Sep 22–24, 2026, Olcay)

## Question

The AUC-targeted sweep scored one client, and its "landings" didn't hold across seeds. Scored
properly, across many clients and on a really label-skewed split, do global-DP and
metric-privacy leak less than vanilla?

## Setup

EuroSAT, 48 clients, Dirichlet α = 0.3, FedAvg, 100 rounds. Noise ratio 0.001546, the level the
sweep had "landed" EuroSAT at. Targets are clients 0–9: one shared IN model per mode and seed,
plus one OUT model per removed target. Defenses on seeds 42–44, vanilla on seed 42. The score is
the share of rounds 1–100 where a target's loss was lower with it in training, averaged over the
10 targets (0.5 = no signal). Full protocol: [`PROTOCOL.md`](PROTOCOL.md).

<details>
<summary>Why α = 0.3</summary>

A vanilla pilot over α = 0.1, 0.3, 1 and 10 (3 seeds each) gave 88.0–90.6% accuracy. 0.3 keeps
real label skew (a client's biggest class is about 46% of its data) without hurting accuracy.

</details>

## Result

**No. At this noise, neither defense leaks less than vanilla.** Vanilla scores 0.69. Global-DP
scores 0.61–0.72 and metric-privacy 0.60–0.71 across three seeds, the same range.

- **Global-DP behaves like no defense here:** same accuracy as vanilla (~88%) and the same score.
- **Metric-privacy costs 2–4 points of accuracy** and still doesn't lower the score.

**Single-client scores are unreliable.** Within one vanilla model, individual clients score
anywhere from 0.37 to 0.98, which is why the sweep's one-client "landings" didn't hold.

## Caveats

- **Vanilla has one seed,** and the defenses were tested at one noise level only; the upward noise
  sweep was never run.
- **Two software environments.** Seeds 42–43 ran on torch 2.11, vanilla on a pinned torch 2.10,
  and seed 44 mixes both. Re-running seed 42 on torch 2.10 moved per-target scores.
- **The score is computed ad hoc** from `measurements.json`. No committed script outputs it;
  `analyze.py` and the progress report use a final-round score instead.
- **The training-seed variance runs never ran** (no A100 quota left).
