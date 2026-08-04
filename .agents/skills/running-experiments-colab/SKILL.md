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

Use a unique session name and an explicit accelerator. L4 is the normal default
for this repository unless the user requests another supported GPU.

```bash
uv run python scripts/colab/run_experiment.py run \
  --session <session-name> \
  --gpu L4 \
  --module experiments.<name>.<entrypoint> \
  --results results/<name> \
  --commit-message "results(<name>): add Colab run"
```

Pass experiment arguments after the controller options. Keep the foreground
controller attached when possible; successful completion means artifacts were
downloaded, committed, pushed, and the VM was stopped—not merely that training
exited.

## Monitor

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
