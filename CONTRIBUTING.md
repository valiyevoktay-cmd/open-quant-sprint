# Contributing to open-quant-sprint

This page contains the workflow shared by every sprint track. Track issues should
describe the research question, track-specific deliverables and acceptance
criteria instead of repeating these instructions.

## Claiming and tracking a track

1. Comment on the track issue before starting. The maintainer assigns one owner.
2. The maintainer assigns the issue, adds it to the current sprint milestone and
   updates its status label.
3. Put the issue and pull request links at the top of both descriptions.
4. Keep the issue's status block current:

   ```text
   Status: Part 2 under review
   Owner: @username
   Part 1: #123 — merged
   Part 2: #456 — review needed
   Next action: maintainer review
   Last updated: YYYY-MM-DD
   ```

The current cross-track view is in [`SPRINT_STATUS.md`](SPRINT_STATUS.md).

## Two-part workflow

- **Part 1** is the small feature or measurement primitive, with focused tests.
- **Part 2** is the main research result or strategy. Open it separately; it may
  branch from Part 1 while Part 1 is under review.
- Put analysis scripts that are not intended for the library in
  `research/<name>/` in your fork. Link the exact commit from the PR so the
  analysis remains reproducible.
- Write the hypothesis, parameters and planned comparisons in a draft PR before
  the first full-panel run. Clearly label later follow-ups as exploratory.

## Local verification

Use Python 3.11 and the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.lock
pip install -e . --no-deps
python -m pytest -q
```

For a strategy, also run:

```bash
python -m src.audit strategies/<name>.py
```

A strategy audit `REJECT` is a valid research outcome. It is different from a
unit-test, installation or workflow failure. Include the report and explain the
failed research checks in the PR.

## Pull requests

- Work in your own named files under `src/contrib/` and `strategies/` unless the
  issue explicitly authorizes a shared-file change.
- Add tests with a fixed random seed and include a causality/prefix-invariance
  check where applicable.
- Complete `.github/pull_request_template.md` and link the track issue.
- Keep one current PR for each deliverable. Close an earlier attempt as
  superseded and link its replacement.
- Review the participant assigned by the maintainer for look-ahead leaks.

The maintainer merges contributions after the required tests pass and the
write-up is present. A PASS is a hygiene gate, not evidence of a durable edge;
see [`RULES.md`](RULES.md) and [`LIMITATIONS.md`](LIMITATIONS.md).

