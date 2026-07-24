# Installation & Setup

This guide takes you from a fresh clone to a working local setup: the ML
inference service, the web app, and (optionally) Supabase for the admin
dashboard. For the ML training/evaluation workflow see [MODEL.md](MODEL.md);
for production deployment see the [Deployment](#deployment) section and the
`docs/DEPLOY_*.md` guides.

## Prerequisites

- **Node.js 20+** and npm
- **Python 3.11+**
- **Git**
- A **Supabase** project (for the admin dashboard, auth, and labeled data)
- A **Vercel** account (for deploying the web app) — optional for local dev
- A host for the ML service (Hugging Face Spaces / Render) — optional for local dev

Check your versions:

```bash
node --version      # v20+
python3 --version   # 3.11+
```

## 1. Clone

```bash
git clone <your-repo-url> MantisVision
cd MantisVision
```

> **Handoff note:** nothing in the code hardcodes a repository owner, deploy
> URL, or account name — those all come from environment variables and the
> per-deploy config below. After transferring the repo, you only need to set
> the new values here; there are no source edits required to re-point it.

## 2. ML pipeline & inference service

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # full set (training + eval + serving)
```

To run **only** the inference service (smaller install, no training deps):

```bash
pip install -r requirements-serve.txt
```

The service needs a trained checkpoint at `ml/checkpoints/best_model.pt`. You
get one by training (`python -m src.train`, see [MODEL.md](MODEL.md)) or by
copying a hosted one down. Then:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` should report
`{"status":"ok","model_loaded":true,...}` once a checkpoint is present. With no
checkpoint, the service still starts and reports `model_loaded:false`.

## 3. Web app

```bash
cd apps/web
cp .env.example .env.local
npm install
npm run dev                          # http://localhost:3000
```

Fill in `.env.local` (see the [environment variables](#environment-variables)
table). For a purely local frontend + ML loop you only need
`ML_API_URL=http://localhost:8000`; the Supabase and GitHub values are required
for the admin dashboard and retrain features.

Useful scripts (from `apps/web`):

```bash
npm run dev          # dev server
npm run build        # production build
npm run start        # serve the production build
npm run typecheck    # tsc --noEmit
npm run lint         # next lint
```

## 4. Supabase (admin dashboard, auth, labeled data)

The admin dashboard, dataset labeling, and retrain history all live in
Supabase.

1. Create a project at [supabase.com](https://supabase.com).
2. **Apply the migrations** in `supabase/migrations/` (in filename order).
   Either paste each file into the Supabase SQL editor, or use the Supabase
   CLI (`supabase db push` against a linked project). They create the
   `profiles` (roles), `measurement_schema`, `training_images` /
   `training_labels`, and `model_runs` tables plus their RLS policies.
3. Create a **Storage bucket** for uploaded specimen/training photos (e.g.
   `specimen-photos`).
4. Copy the project **URL**, **anon key**, and **service role key** into
   `apps/web/.env.local` (see the table below).
5. **Bootstrap the first admin** — there is no public signup:
   - Create the user in the Supabase Auth dashboard (or via
     `supabase.auth.admin.createUser`). A trigger auto-creates a `viewer`
     profile row.
   - Promote them in the SQL editor:
     ```sql
     update profiles set role = 'admin' where email = 'someone@example.com';
     ```
   Roles are `admin` (full access), `contributor` (dataset labeling only), and
   `viewer` (default, no dashboard access).

## Environment variables

### Web app (`apps/web/.env.local`)

| Variable | Required for | Description |
|---|---|---|
| `ML_API_URL` | Predictions | URL of the FastAPI inference service. Defaults to `http://localhost:8000` if unset. |
| `ML_API_ADMIN_TOKEN` | Model promotion | Shared secret for hot-swapping the live model. Must equal `RELOAD_TOKEN` on the ML host. |
| `NEXT_PUBLIC_SUPABASE_URL` | Dashboard | Supabase project URL. |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Dashboard | Supabase anon (public) key. |
| `SUPABASE_SERVICE_ROLE_KEY` | Dashboard (server) | Supabase service-role key. **Server-side only — never exposed to the browser.** |
| `NEXT_PUBLIC_SITE_URL` | Team invites | Public base URL of the web app (e.g. `https://your-app.vercel.app`). Used to build invite/reset links; also add it to Supabase → Auth → URL Configuration. Falls back to request origin if unset. |
| `GITHUB_TOKEN` | Retrain button | Fine-grained PAT with `actions:write` + `contents:write` on this repo, used to dispatch the retrain workflow. |
| `GITHUB_REPO` | Retrain button | `owner/repo` of this repository (e.g. `your-org/mantisvision`). |

### ML inference service (host env / secrets)

| Variable | Required for | Description |
|---|---|---|
| `MODEL_URL` | Cloud cold start | Direct-download URL of `best_model.pt` (e.g. a GitHub Release asset). The service fetches it at boot since `ml/checkpoints/` is gitignored. |
| `RELOAD_TOKEN` | Model promotion | Shared secret; must equal the web app's `ML_API_ADMIN_TOKEN`. With it unset, `/admin/reload` returns 503. |
| `WEB_APP_ORIGIN` | CORS lockdown | The deployed PWA origin. Defaults to `*` (all origins) for local dev. |
| `ENABLE_GRADCAM` | Optional | Enable Grad-CAM in serving (memory-heavier). Safe on hosts with enough RAM. |
| `MANTIS_SCHEMA_PATH` | Optional | Override the path the schema is loaded from (defaults to `ml/metadata/schema.json`, else `DEFAULT_SCHEMA`). |

### Retrain workflow (GitHub Actions secrets/vars)

Set these under the repo's **Settings → Secrets and variables → Actions**:

| Name | Kind | Description |
|---|---|---|
| `SUPABASE_URL` | secret | Supabase project URL (server-side). |
| `SUPABASE_SERVICE_ROLE_KEY` | secret | Supabase service-role key. |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | secret | Kaggle API credentials (dataset sync). |
| `KAGGLE_DATASET` | variable | `username/dataset-slug` of the Kaggle dataset. |
| `HF_TOKEN` / `HF_SPACE` | secret | Only if auto-deploying the ML service to a Hugging Face Space. |

The workflow reads the repo automatically via `${{ github.repository }}` — no
hardcoded owner/name.

## Verification checklist

- [ ] `ml`: venv created, `pip install -r requirements.txt` succeeds
- [ ] `uvicorn src.api.main:app --port 8000` serves `/health`
- [ ] `apps/web`: `npm install`, `npm run typecheck`, and `npm run build` pass
- [ ] `npm run dev` runs; uploading a photo returns a result (with the ML
      service running and `ML_API_URL` pointed at it)
- [ ] Supabase migrations applied; first admin promoted; `/member/login` works
- [ ] (optional) `python -m pytest` passes in `ml/`

## Docker (ML inference service)

The ML inference service ships a production [`ml/Dockerfile`](ml/Dockerfile) so
it can run on any container host (Hugging Face Spaces, Render, Fly.io, your own
box) with no Python setup. It builds from `requirements-serve.txt` (the slim,
serve-only dependency set — no training/eval libraries) to keep the image small
and boot fast.

> The **web app** is not containerized — it deploys to Vercel as a Next.js
> project (root directory `apps/web`). Docker here is for the ML service only.

**Build and run locally:**

```bash
cd ml
docker build -t mantis-vision-api .
docker run --rm -p 8000:8000 \
  -e MODEL_URL="https://.../best_model.pt" \
  -e RELOAD_TOKEN="$(openssl rand -hex 32)" \
  -e WEB_APP_ORIGIN="http://localhost:3000" \
  mantis-vision-api
```

Then `curl http://localhost:8000/health`. Without `MODEL_URL` the container
still starts and reports `model_loaded:false`; to serve a locally trained
checkpoint instead of downloading one, mount it:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mantis-vision-api
```

**What the image does / good to know:**

- Base `python:3.12-slim`; copies only `config.py` and `src/` (dataset,
  checkpoints, logs, and reports are excluded via
  [`.dockerignore`](ml/.dockerignore)).
- Serves with a **single** uvicorn worker (one model copy in memory) on
  `$PORT` (default `8000`).
- Native math libraries are pinned single-threaded and Grad-CAM is off by
  default (`ENABLE_GRADCAM=false`) so the container fits a 512 MB free tier.
  On a larger instance, raise the thread env vars and set `ENABLE_GRADCAM=true`.
- Runtime configuration is entirely via environment variables (see the [ML
  inference service](#ml-inference-service-host-env--secrets) table) — nothing
  host-specific is baked into the image.

Hugging Face Spaces and Render both build this same Dockerfile automatically;
the deploy guides below cover the platform-specific wiring.

## Deployment

- **Web app → Vercel.** Root directory `apps/web`. Set the web env vars above.
- **ML service → Hugging Face Spaces** (recommended, more free RAM) **or
  Render.** Same `ml/Dockerfile`. Full steps:
  [docs/DEPLOY_HUGGINGFACE.md](docs/DEPLOY_HUGGINGFACE.md),
  [docs/DEPLOY_ML_API.md](docs/DEPLOY_ML_API.md),
  [docs/DEV_LOCAL_ML_API.md](docs/DEV_LOCAL_ML_API.md).
- **Database → Supabase.** Apply `supabase/migrations/`.

After deploying, point the web app's `ML_API_URL` at the ML host and set the
matching `ML_API_ADMIN_TOKEN` / `RELOAD_TOKEN` pair so model promotion works.
