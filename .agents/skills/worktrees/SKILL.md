---
name: worktrees
description: Open, slim, and clean up git worktrees for this repo. Use whenever an experiment or task needs its own branch checked out next to the main checkout, or when worktrees are taking too much disk.
---

# Worktrees

The repo tracks ~11 GB of result data. A plain `git worktree add` checks all of it out again,
and a fresh `.venv` adds ~1.2 GB more. Always open worktrees sparse.

## Open

From the main checkout (`metricdp-pytorch/`), with `<exp>` = `results/<group>/<experiment>`:

```bash
git worktree add --no-checkout ../metricdp-pytorch-<name> <branch>
cd ../metricdp-pytorch-<name>
git sparse-checkout set --no-cone '/*' '!/results/*/*/results/' '/<exp>/results/'
git checkout <branch>
```

Run these as one `&&` chain. If the `cd` fails, the later commands would hit the main checkout.

- `--no-checkout` first, then set the patterns, then check out. Otherwise git writes all 11 GB
  before the patterns apply.
- This keeps every file except other experiments' `results/` data folders. Code, READMEs and
  reports stay, so cross-experiment imports still work.
- Your own experiment's `results/` must be inside the sparse set, or `git add` refuses new files
  there.
- Need another experiment's data? Add its path: `git sparse-checkout add '/<other>/results/'`.
- Never run `git sparse-checkout disable` in a worktree.

## Environment

Reuse the main checkout's venv instead of creating a new one:

```bash
UV_PROJECT_ENVIRONMENT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv" uv run --no-sync <cmd>
```

Only `uv sync` a separate venv if the branch changes dependencies (`pyproject.toml`/`uv.lock`).

## Name

`../metricdp-pytorch-<name>`, where `<name>` is the branch name with `/` replaced by `-`
(e.g. `runs/dataset-vs-model` → `metricdp-pytorch-runs-dataset-vs-model`).

## Clean up

Once the branch is merged and nothing runs from it:

```bash
git worktree remove ../metricdp-pytorch-<name>   # refuses if there are uncommitted changes
```

Check first for local-only ignored files worth keeping (`.colab/` logs, `*.predictions.npz`):
`git status --short --ignored`.

To slim an existing full worktree instead:
`git sparse-checkout set --no-cone '/*' '!/results/*/*/results/' && git sparse-checkout reapply`,
and delete its `.venv` if it can use the main one.
