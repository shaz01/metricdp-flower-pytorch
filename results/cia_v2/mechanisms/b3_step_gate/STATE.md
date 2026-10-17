package: B3
status: planning
gate: G-B3 = not evaluated
protocol: results/cia_v2/mechanisms/b3_step_gate/PROTOCOL.md (not written yet; fixed and committed before any run)
machine: compute server
cost so far: 0 GPU-h of the 10 GPU-h cap
next: write the gate strategy (default off) and its synthetic tests; draft PROTOCOL.md; wait for G-M0 to launch anything
needs from others: Agent A, G-M0 (feature/cia-v2-measurement STATE.md = ready): seed plumbing and the randomized-inclusion runner. Agent A, M2: the operating points `r_0.6` and the next-lower noise, plus each M2 replicate's `noise_seed`, `client_seed` and inclusion pattern so that the gated runs reuse them. Agent A: a default-off hook in `experiments/reproduce/server.py` `main` that hands the server's public validation loader to an optional strategy wrapper, and a passthrough for extra run-config keys in the randomized-inclusion runner (both are Agent A's files; the request is written here rather than edited)

Agent B (roadmap v2 Section 6) runs B3, B1, B5, then B2 if M2 opens it, then C1 if at least one B gate passes. This
file covers B3; the later packages get their own directories under `results/cia_v2/mechanisms/`.

## Plan (roadmap v2 Section 3, B3)

- **Hypothesis.** The public-validation step-size gate, a post-processing step that changes no guarantee, raises
  final accuracy under global DP at equal noise without raising top-decile CIA AUC by more than 0.02.
- **Mechanism.** Each round the server takes the released global-DP aggregate `w_t` and the previous model
  `w_{t-1}`. It scores `w_{t-1} + m (w_t - w_{t-1})` for the multipliers `m` in {0, 1/8, 1/4, 1/2, 1, 2, 4, 8} (the
  stacked-head set) on its public validation set. It keeps the candidate with the lowest validation cross-entropy
  (variant V0, the one confirmed on fresh data in `experiments/stacked_head/protocols/2026-10-09_next_steps_findings.md`).
  The gate reads only the released aggregate and public data, so the global-DP guarantee is unchanged. The gated model
  is the release that clients train from and the attacker observes.
- **Design.** Randomized inclusion (measurement standard v2, Section 2 item 3): 48 slots with the public slot count as
  denominator, 16 fixed candidates in a balanced design, the other 32 clients always in, seed-42 EuroSAT layout (alpha
  0.3, top decile 23, 7, 40, 18, 33), 50 rounds, equal weights. Global DP at two noise levels, `r_0.6` and the
  next-lower level from M2, with `R = 8` gated runs each (16 runs, about 8 GPU-h). The ungated runs are M2's.
- **Randomness.** Within a configuration every replicate has its own `noise_seed` and `client_seed`, so worlds share
  no private randomness. Across the gated and ungated configurations, replicate `r` reuses M2's `noise_seed`,
  `client_seed` and inclusion pattern (allowed across configurations, Section 2 item 1) so that gate-vs-no-gate
  differences are paired by replicate. Shared public randomness: initial model, partition, shadow samples,
  evaluation sets.
- **Gate rule (verbatim, roadmap v2):** "**G-B3:** at least 2 pp accuracy gain at equal noise, with top-decile AUC
  rising by no more than 0.02." The protocol fixes how the intervals enter (which end of which interval, and whether
  one or both noise levels must pass) before any run, together with the minimum detectable effect from M0's power
  table.

## Log

| When | What |
|---|---|
| 2026-10-17 | Agent B session start. `feature/cia-v2-measurement` not on origin yet (G-M0 pending), so `feature/cia-v2-mechanisms` branched from `master` 7fda7242; Agent A's branch is merged in (real merge) once its `STATE.md` says ready. Nothing launched. |
