# Golden Set

47 Hard DABstep tasks solved independently from `manual.md`, used to diagnose exactly how
the agent fails — not just whether it does. See the top-level README's "What made the
difference" section for how these results were actually used.

## Not included in this repo

`golden_set.json` is deliberately excluded (see `.gitignore`) — it is a curated answer key
for real DABstep Hard tasks, and shipping it would hand out benchmark answers. This file
describes what it contains and how it's built, not the data itself.

## What it contains

Per task: our own answer, the agent's submitted answer, a one-line failure label, a
confidence level, and an archetype tag.

Confidence levels: `validated` — matches a benchmark-withheld dev label exactly (7 tasks) ·
`high` — determinate (17) · `medium` — depends on one stated reading (5) ·
`convention-dependent` — `manual.md` doesn't specify a convention, both values given (14) ·
`unsolved` (4).

Convention-dependent tasks (total-fees, delta) carry both readings: **most-specific** (the
rule with the most non-wildcard fields wins) and **sum-all** (every matching rule charges).

## `fee_core.py` in this folder

Not a separate implementation — a thin shim that re-exports the canonical `code/fee_core.py`
by path, so golden-set scoring shares the exact same fee-matching logic the agent's tools
use, rather than risking a parallel copy that drifts.

## Regenerating it

Solve the task from `manual.md`, reproduce the agent's wrong answer by deliberately
re-introducing the suspected mistake, then record the answer/failure/confidence triple.
Don't assign a label without reproducing the failure first.
