---
name: running-experiments
description: Run long-lived project experiments on the Lightning.ai GPU machine. Use when launching, resuming, monitoring, or collecting remote experiment runs.
---

# Running experiments

Connect to the remote machine with:

```bash
ssh s_01ky2wfah285g4z020c9b1d3hn@ssh.lightning.ai
```

Before running anything, inspect the remote checkout and confirm the intended branch/revision is present without overwriting remote changes.

## Requirements

- Experiments must run on the remote GPU, never as CPU-only jobs.
- Long runs must continue after SSH disconnects and after the agent exits. Launch them in a detached remote session such as `tmux`, with logs stored outside the repository.
- Do not create one-off shell scripts or ad hoc experiment files on the remote machine. Add reusable Python modules under `experiments/<name>/`
- Prefer resumable scripts that skip validated completed outputs, use deterministic run names, and continue past individual failures.
- Keep real experiment artifacts under `results/<name>/`. 
- Avoid duplicate runs: check existing detached sessions and result artifacts before launching.

Use `uv run` for all project commands. Follow `AGENTS.md` and the active experiment branch workflow for code and results.

When reporting a launch, include the remote revision, detached session name, command/module, GPU allocation, log location, and result directory. Validate completion from the result artifacts rather than merely from process termination.
