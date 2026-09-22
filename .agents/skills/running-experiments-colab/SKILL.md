---
name: running-experiments-colab
description: Run long-lived project experiments on Google Colab GPUs through the repository's Colab CLI controller. Use when launching, resuming, monitoring, collecting, or stopping Colab experiment runs across one or more Colab accounts, including live NVIDIA GPU checks and recovery after a local interruption.
---

# Running experiments on Colab

Use `scripts/colab/run_experiment.py` from the repository root. It stages a
credential-free source snapshot, provisions a named Colab GPU session on a
chosen account, launches a detached remote worker, downloads that run's result
directory, and commits it locally.

Controllers **detach by default**: `run` returns as soon as the launch is
recorded, and a background controller owns the session from there. Several
accounts can be driven at once, so the GPU ceiling is
`accounts × --max-per-account` rather than one account's limit.

`sweep` is the observation command. It probes every recorded session, collects
finished runs whose controller has died, relaunches waiters for orphaned runs,
and flags everything else that needs a human.

## Before launching

1. Read `AGENTS.md` and the relevant experiment README.
2. Confirm the user requested an actual remote launch before allocating a GPU.
3. Inspect `git status`, the current branch, and the intended module and result
   directory. Prefer committed and pushed experiment code so the recorded source
   revision is reproducible. Never stage unrelated user changes.
4. Run `sweep` to see what is already running, and `accounts` to see which
   accounts are logged in and how loaded they are.
5. Confirm the experiment is resumable, uses deterministic run names, and writes
   only real artifacts below `results/<name>/`.

Never upload Git credentials or embed tokens in notebooks, source, commands, or
Colab logs. The controller keeps Git authentication on the local machine and
rejects common credential patterns in the source archive.

## Accounts

Each account gets its own `HOME` under `~/.colab-accounts/<account>/`, because
`colab_cli` hardcodes its token path. The machine's ordinary login is the
`default` account.

```bash
uv run python scripts/colab/run_experiment.py accounts            # who is configured
uv run python scripts/colab/run_experiment.py accounts --remote   # plus live sessions
uv run python scripts/colab/run_experiment.py login --account lab2
```

`login` runs the interactive OAuth paste flow under that account's `HOME`; sign
in with the Google account that slot should own. Nothing else in the controller
touches `HOME`, so accounts cannot bleed into each other.

## Launch

```bash
uv run python scripts/colab/run_experiment.py run \
  --session <session-name> \
  --gpu A100 \
  --module experiments.<name>.<entrypoint> \
  --results results/<name> \
  --commit-message "results(<name>): add Colab run" \
  -- --output-dir results/<name>
```

Pass experiment arguments after `--`. **The module's output path must be the
same directory supplied to `--results`**; only that directory is downloaded from
the VM, and the controller now refuses a mismatched `--output-dir` outright.

`--account auto` (the default) picks a logged-in account below the
`--max-per-account` cap (default 2). Name an account explicitly with
`--account lab2`, and raise or bypass the cap with `--max-per-account` /
`--force` only when the user asked for it. `--attach` keeps the controller in
the foreground; use it only when a single run is being watched deliberately.

A successful `run` prints the session, account, controller pid, and log path.
That means the launch was *recorded*, not that training started — the first
`sweep` is what confirms that.

## Monitor

```bash
uv run python scripts/colab/run_experiment.py sweep
```

Sweep every few minutes while work is in flight. It prints one row per session
(session, account, phase, remote state, round, GPU utilization, age, action) and
then an explicit list of sessions needing attention. Act only on the flagged
rows; everything else is either healthy or already handled.

Flags and their meaning:

- `no training after Nm` — the VM was allocated but the worker never started.
  Inspect `.colab/logs/<session>.log`; a controller still stuck in setup after
  ~10 minutes is a startup hang, not progress.
- `no log progress for Nm` — training output stopped advancing. Check the GPU
  column and the last log line before deciding.
- `controller died during launch` — provisioning was interrupted. The VM may
  still be allocated; check `accounts --remote` and stop it if it is idle.
- `controller gone; waiter relaunched` — sweep already adopted the run. No
  action needed unless it repeats.
- `collect failed: ...` — auto-collection broke. This is the case that loses
  data if ignored: fix the cause and rerun `collect --session <name>` before the
  VM disappears.
- `probe failed` — a transient kernel or network problem. Retry the sweep before
  concluding anything.

`sweep --dry-run` reports without collecting or adopting. `status --session
<name>` prints one session's full probe output including `nvidia-smi`.

## Collect, commit, and push

Collection is automatic: whichever controller owns the session downloads,
extracts, commits, and releases the VM, and `sweep` does it for sessions whose
controller is gone. Collection is deliberately *not* serialized — a Colab VM can
vanish at any moment, so downloads never queue.

Only the Git commit is mutually exclusive, through `.colab/commit.lock`. Each
run commits exactly its own `--results` directory, so concurrent sessions
produce independent, path-disjoint commits.

**Nothing is pushed.** After a batch lands, review the commits and push once:

```bash
git log --oneline -10
git push
```

Use `collect --session <name>` for a finished job that needs a manual retry, and
`stop --session <name>` only for an explicitly abandoned job. If collection
fails, preserve the VM, repair the local problem, and retry collection before
stopping it.

For Colab CLI 0.6.0, an `AttributeError` saying `jupyter_kernel_client` lacks
`KernelClient` means the tool resolved the wrong upstream package. Repair its
environment with the Google fork as documented in `experiments/cia/README.md`,
then retry without reallocating the session.

## Verify completion

- Inspect the experiment's result/report JSON rather than trusting process exit.
- Check `colab_run.json` for `state: complete` and `returncode: 0`.
- Confirm `sweep` no longer lists the session and that its state file reads
  `phase: collected`.
- Verify the result commits are on the expected branch, then push them.
- Run relevant local tests when code or result interpretation changed.
- Report the source revision, session, account, GPU, module, result directory,
  result verdict, commit, and shutdown status.

Do not declare the experiment finished or write its narrative report unless the
project owner explicitly makes that decision.
