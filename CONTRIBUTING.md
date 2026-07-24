# Contributing

This guide covers the day-to-day developer workflow: how to set up, run the
checks, keep the two schema definitions in sync, and structure changes. For
first-time setup see [INSTALL.md](INSTALL.md); for how the system fits together
see [ARCHITECTURE.md](ARCHITECTURE.md).

## Repository layout

A monorepo with three parts — see the [folder structure](README.md#folder-structure)
in the README and the [component breakdown](ARCHITECTURE.md#components) in
ARCHITECTURE.

- `apps/web` — Next.js PWA + admin dashboard (TypeScript).
- `ml` — PyTorch pipeline + FastAPI inference service (Python).
- `supabase/migrations` — Postgres schema.

## Local setup

Follow [INSTALL.md](INSTALL.md). In short:

```bash
# web
cd apps/web && cp .env.example .env.local && npm install

# ml
cd ml && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Running the checks

Run these before opening a PR. CI-equivalent commands:

**Web (`apps/web`):**

```bash
npm run typecheck     # tsc --noEmit — must pass
npm run lint          # next lint
npm run build         # next build — the strongest signal
```

**ML (`ml`, with the venv active):**

```bash
python -m pytest                      # full test suite
python -m src.data.validate_dataset   # sanity-check a dataset before training
```

The `ml/tests/` suite covers schema parsing, the multi-head model & losses,
annotations, evaluation, ONNX export, retrain materialization, and a training
smoke test. Add a test alongside any change to that pipeline.

## Keeping the schema in sync

The measurement schema is defined **twice** and the two must stay identical in
shape and defaults:

- `ml/config.py` — `DEFAULT_SCHEMA` and the schema dataclasses (Python).
- `apps/web/src/lib/schema.ts` — `DEFAULT_SCHEMA` and the schema types
  (TypeScript).

They also mirror the SQL seed in `supabase/migrations/`. When you change one:

1. Change the matching structure/defaults in the other.
2. Preserve the field-for-field parity of the helper functions the files call
   out in their comments (`measurementApplies` / `applies`, `rangeForValue` /
   `range_for`, `applies_when` shapes, etc.).
3. Add/adjust a test in `ml/tests/test_schema_parsing.py` if you changed
   parsing/serialization.

A mismatch here is the most common source of subtle bugs (the admin UI offers a
field the model can't train, or vice-versa), so treat it as a single change
spanning both files.

## Adding a measurement (usually no code)

The whole point of the schema-driven design: adding a capability is normally an
**admin-UI edit at `/member/schema`**, then relabel + retrain — not a code
change. The model, losses, dataset loader, and predictor all grow the new head
generically. Only reach for code when you need a genuinely new *input modality*
(a second photo angle, sensor input) or a new measurement *type*. See the
[roadmap](docs/STEP_BY_STEP.md#long-term-roadmap-future-expansion).

## Code style

- **TypeScript:** follow the existing patterns — the ESLint config
  (`eslint-config-next`) is the source of truth; keep `npm run typecheck` clean
  (no `any` escape hatches without reason).
- **Python:** type hints throughout, `from __future__ import annotations`,
  small pure functions, and docstrings that explain *why* (the existing files
  are the reference — match their density and tone).
- Match the surrounding code's naming and idioms rather than importing a new
  style. Keep comments meaningful — explain intent and non-obvious constraints,
  not the obvious.
- Don't hardcode environment-specific values (URLs, tokens, repo names, account
  ids). They belong in environment variables or `ml/config.py`. See the
  [env var reference](INSTALL.md#environment-variables).

## Branches, commits, and PRs

- **Branch** off the default branch for each change; don't commit directly to
  it.
- **Commit messages:** imperative mood, one logical change per commit, explain
  the *why* in the body when it isn't obvious (e.g. "Cache the active schema to
  speed up label saves").
- **PRs:** describe what changed and why, note any schema/migration changes
  explicitly, and confirm the checks above pass. Keep unrelated changes out.
- **Migrations** are append-only and timestamp-prefixed
  (`YYYYMMDDNNNNNN_name.sql`) — add a new one; never edit an applied migration.

## Secrets & data hygiene

- Never commit `.env*`, `kaggle.json`, checkpoints, datasets, or any credential
  — `.gitignore` already covers these; keep it that way.
- Dataset images live on Kaggle, not git. Model checkpoints are hosted and
  downloaded at runtime, not committed.
- `SUPABASE_SERVICE_ROLE_KEY` and `RELOAD_TOKEN`/`ML_API_ADMIN_TOKEN` are
  privileged — server-side only, never shipped to the browser.

## Questions

Start with [ARCHITECTURE.md](ARCHITECTURE.md), [MODEL.md](MODEL.md), and
[docs/STEP_BY_STEP.md](docs/STEP_BY_STEP.md) — between them they cover almost
every "how does X work" question about this codebase.
