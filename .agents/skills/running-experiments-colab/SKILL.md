---
name: running-experiments-colab
description: Run long-lived project experiments on Google Colab GPUs through the repository's Colab CLI controller. Use when launching, resuming, monitoring, collecting, pushing, or stopping Colab experiment runs, including live NVIDIA GPU checks and recovery after a local interruption.
---

# Running experiments on Colab

Use `scripts/colab/run_experiment.py` from the repository root. It provisions a
named Colab GPU session, uploads a credential-free source snapshot, launches a
detached worker, monitors the job, downloads its result directory, commits and
pushes only those results, and releases the VM.

## Before launching

1. Read `AGENTS.md` and the relevant experiment README.
2. Confirm the user requested an actual remote launch before allocating a GPU.
3. Inspect `git status`, the current branch, and the intended module and result
   directory. Prefer committed and pushed experiment code so the recorded source
   revision is reproducible. Never stage unrelated user changes.
4. Check `colab sessions` to avoid duplicate runs or abandoned billable VMs.
5. Confirm the experiment is resumable, uses deterministic run names, and writes
   only real artifacts below `results/<name>/`.

Never upload Git credentials or embed tokens in notebooks, source, commands, or
Colab logs. The controller keeps Git authentication on the local machine and
rejects common credential patterns in the source archive.

## Launch

Use a unique session name and an explicit accelerator. Two A100 sessions in parallel
is the normal default for this repository unless the user requests another supported GPU.

```bash
uv run python scripts/colab/run_experiment.py run \
  --session <session-name> \
  --gpu A100 \
  --module experiments.<name>.<entrypoint> \
  --results results/<name> \
  --commit-message "results(<name>): add Colab run"
```

Pass experiment arguments after the controller options. **The module's output
path must be the same directory supplied to `--results`**: explicitly pass its
`--output-dir <the --results path>` when the module has one. Otherwise the
controller can successfully collect and commit only `colab_run.json` and the
training log while the actual result JSON/NPZ/checkpoints remain in the
module's default directory on the released VM. Keep the foreground controller
attached when possible; successful completion means artifacts were downloaded,
committed, pushed, and the VM was stopped—not merely that training exited.

On macOS, the controller sends a best-effort local notification after results
have been collected and pushed; it never changes the remote job's state.

### Five-minute launch watchdog

Treat the first five minutes after every allocation as an attended startup
window, especially for A100 sessions. The agent must remain in its active task
loop for the entire five minutes: wait in intervals of 2.5 minutes (150
seconds), then perform the check below at the five-minute deadline. Do not
replace this with a detached shell, background terminal session, queued
watcher, reminder, or an instruction for a later agent turn.

At the five-minute check:

1. Inspect the local controller process tree and output. A controller still
   blocked in `remote_setup.py` or `remote_start.py` after five minutes is a
   startup hang, not useful experiment progress.
2. Run `status` and require the remote state to be `running` or `complete`, with
   real training output that has advanced beyond startup. Include an
   `nvidia-smi` utilization and memory sample. Do not interpret a single idle
   sample as failure when the log is advancing between rounds or evaluations.
3. If the job is `not-started`, the log is absent or unchanged, or the local
   launch command is hung, investigate immediately. Do not let an idle
   allocation continue consuming runtime or compute credits.
4. If no useful artifacts have been produced, release the stuck VM and relaunch
   on a fresh session. Do not recover a new long run onto an allocation whose
   startup hang has already consumed a material part of its lifetime. If useful
   artifacts may exist, preserve the VM and use the recovery procedure before
   deciding whether to stop it.

Launches queued behind another run need the same watchdog, measured from when
the queued controller actually allocates its VM rather than when the local
queue process was created. The watchdog is incomplete until the agent has
personally performed the deadline check and, when training has not begun,
actively stopped or recovered the session.

## Monitor

After a run has passed the five-minute launch watchdog with real round
advancement and a non-idle GPU sample, use a 4.5-minute (270-second) monitoring
cadence. Keep the foreground controller attached, but do not poll an otherwise
healthy worker more often than that; reserve tighter checks for the startup
window, completion/collection, or an observed anomaly.

Run this from another terminal whenever a progress or GPU check is needed:

```bash
uv run python scripts/colab/run_experiment.py status --session <session-name>
```

Report the latest training progress and the `nvidia-smi` utilization/memory
sample. A transient status-probe failure does not imply the detached worker
failed; retry and inspect `colab status -s <session-name>`.

## Recover and collect

If local monitoring was interrupted, resume the automatic finalizer:

```bash
uv run python scripts/colab/run_experiment.py wait --session <session-name>
```

Use `collect` when the remote job is already complete. Use `stop` only for an
explicitly abandoned job. If collection or Git push fails, preserve the VM,
repair the local problem, and retry collection before stopping it.

For Colab CLI 0.6.0, an `AttributeError` saying
`jupyter_kernel_client` lacks `KernelClient` means the tool resolved the wrong
upstream package. Repair its environment with the Google fork as documented in
`experiments/cia/README.md`, then retry without reallocating the session.

## Verify completion

- Inspect the experiment's result/report JSON rather than trusting process exit.
- Check `colab_run.json` for `state: complete` and `returncode: 0`.
- Verify the result commit is on the expected branch and present on `origin`.
- Run relevant local tests when code or result interpretation changed.
- Run `colab sessions` and confirm the completed session is absent.
- Report the source revision, session, GPU, module, result directory, result
  verdict, pushed commit, and shutdown status.

Do not declare the experiment finished or write its narrative report unless the
project owner explicitly makes that decision.
