---
name: running-experiments-lightningai
description: Run long-lived project experiments on the Lightning.ai GPU machine. Use when launching, resuming, monitoring, or collecting remote Lightning.ai experiment runs, checking GPU use, detached sessions, logs, revisions, and result artifacts.
---

# Running experiments on Lightning.ai

Connect to the remote machine with:

```bash
ssh s_01ky2wfah285g4z020c9b1d3hn@ssh.lightning.ai
```

Before running anything, inspect the remote checkout and confirm the intended
branch/revision is present without overwriting remote changes. Confirm the user
requested an actual launch before allocating or consuming GPU resources.

## Requirements

- Run experiments on the remote GPU, never as CPU-only jobs.
- Check `nvidia-smi`, existing detached sessions, logs, and result artifacts
  before launching so work is not duplicated.
- Keep long runs alive after SSH disconnects in a detached session such as
  `tmux`; store operational logs outside the repository.
- Do not create one-off shell scripts or ad hoc experiment files remotely. Add
  reusable Python modules under `experiments/<name>/` locally and synchronize a
  committed revision.
- Prefer resumable runners that validate completed outputs, use deterministic
  run names, and continue past individual failures.
- Keep real artifacts under `results/<name>/`; do not put lock files, status
  files, or launcher scripts there.
- Use `uv run` for every project command and follow `AGENTS.md` plus the active
  experiment branch workflow.

## Monitor and complete

Inspect the detached session, log tail, process state, `nvidia-smi`, and result
files during training. Treat process exit as insufficient evidence of success:
validate the expected result artifacts and their internal completion markers.

When reporting a launch or completion, include the remote revision, detached
session name, command/module, GPU allocation/utilization, log location, result
directory, artifact validation, and Git commit/push status. Do not declare an
experiment finished or write its report until the project owner explicitly
makes that decision.
