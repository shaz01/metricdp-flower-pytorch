# Design: Smoother Cross-Machine Claude Code Handoffs

**Date:** 2026-08-05
**Status:** Approved, implementing

## Problem

Work on this project now spans three machines: this Mac, an RTX laptop, and a shared A5000
workstation (SSH-only, one of its two GPUs). Each runs its own independent Claude Code
installation — there is no cross-machine session sync in the CLI; conversation history is stored
locally, keyed by machine and absolute project path (confirmed: a session's transcripts live under
`~/.claude/projects/<hashed-path>/`, which differs per machine even for the same repo).

Three concrete pain points, in the user's own words:
1. Remembering to update `STATUS.md` (the existing git-tracked pickup-point doc) before switching
   machines or ending a session — easy to forget, leaving the next session to start from stale
   information.
2. Losing track of which machine is running (or has run) what, with three machines now working in
   parallel on different parts of the same sweep.
3. The resume-prompt itself being tedious to hand-write fresh every time a new session starts on a
   different machine.

## Constraints

- `docs/RESEARCH_ROADMAP.md` is gitignored by repo convention (`AGENTS.md`) and stays that way —
  the user explicitly chose manual copying over changing this when asked directly. Out of scope
  for this design.
- No workstation hostnames, IPs, or usernames may appear in any committed project doc (standing
  rule from this session). Any "which machine ran what" tracking must use generic role labels.
- Solution should not require a new tool/script that itself needs remembering to run — that just
  relocates pain point #1 rather than solving it.

## Approach

`AGENTS.md` (this repo's canonical instructions file, symlinked from `CLAUDE.md`) is automatically
loaded into every Claude Code session's context on every machine, the moment the session starts —
no manual action required, and it distributes itself via ordinary `git pull` like any other
tracked file. This makes it the natural place to encode a standing cross-machine protocol, rather
than something that has to be re-communicated by hand each time (which is pain point #3 exactly).

### Component 1: "Working across machines" section in `AGENTS.md`

New section, placed after "Git workflow". Contents:

- **Session start**: before making changes, read `STATUS.md` and skim recent history
  (`git log --oneline -10`) to pick up state left by other machines/sessions.
- **Session end**: when a meaningful chunk of work wraps up (not every message — use judgment,
  same granularity as "worth a commit"), update `STATUS.md`'s Active Work section — current state,
  what's running where, next steps — then commit and push.
- One-line pointer noting `docs/RESEARCH_ROADMAP.md` is gitignored and needs manual copying if it
  changes, so the gap is documented rather than silently rediscovered.

This directly addresses pain point #1 (the checklist is now something Claude does automatically,
not something the human has to remember) and pain point #3 (no hand-written resume prompt needed —
starting a session and reading `AGENTS.md` + following its instructions to check `STATUS.md` *is*
the resume prompt).

### Component 2: "Currently running" table in `STATUS.md`

A small structural addition to `STATUS.md`'s existing "Active work" section:

```markdown
### Currently running

| Machine role | Task | Status |
|---|---|---|
| GPU workstation | fedavg matrix, sweep_scale_controlled(_epochs) | running |
| GPU laptop | fedyogi n=4/n=8 | running |
```

Machine roles are generic, stable labels the user assigns informally (e.g. "GPU workstation", "GPU
laptop", "this Mac") — never hostnames/IPs. Rows are added/updated/removed as part of the
session-end step in Component 1. Addresses pain point #2.

## Non-goals

- Not building a git hook or helper script (considered and rejected — see approaches B/C discussed
  with the user; both either don't distribute automatically across machines or duplicate what
  Component 1 already gives for free).
- Not changing `docs/RESEARCH_ROADMAP.md`'s gitignore status.
- Not automating `STATUS.md` updates — a human/agent still writes the content; only the *habit* of
  doing so is being made durable.

## Verification

Success is behavioral, not a test suite: the next time a Claude Code session starts on any of the
three machines and the user says something like "continue" with no further briefing, the session
should already reflect current project state (active branch, what's running where, next steps)
purely from having read `AGENTS.md` → `STATUS.md` per the new checklist — no hand-composed resume
prompt required from the user.

## Implementation

Two file edits, no new files/scripts:
1. `AGENTS.md` — add the "Working across machines" section.
2. `STATUS.md` — add the "Currently running" table under Active work, populated with the sweep
   currently in progress (GPU workstation: fedavg matrix; GPU laptop: fedyogi n=4/n=8).

Both are small, low-risk documentation edits; no code changes, no tests to update.
