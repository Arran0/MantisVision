# Deploying the ML inference API (Render)

`ml/src/api` (FastAPI + PyTorch) can't run on Vercel — it needs a real
container host. These are the steps for Render, which is free-tier friendly
and builds straight from the `ml/Dockerfile` already in this repo.

> Render's free tier caps out at 512 MB RAM, which is tight for a PyTorch
> service. If you're hitting out-of-memory crashes, see
> [`DEPLOY_HUGGINGFACE.md`](DEPLOY_HUGGINGFACE.md) instead — same Dockerfile,
> free tier gives 16 GB RAM.

## 1. Host the checkpoint somewhere downloadable

`ml/checkpoints/best_model.pt` is gitignored (binary model weights don't
belong in git history), so Render's clone of this repo won't have it. The API
downloads it at startup instead, from a URL you provide via `MODEL_URL`.

Easiest option — attach it as a GitHub Release asset:

1. On GitHub: **Releases -> Draft a new release** (any tag name, e.g. `model-v1`).
2. Upload `best_model.pt` as a release asset and publish.
3. Copy the asset's direct download URL — looks like:
   `https://github.com/<owner>/<repo>/releases/download/model-v1/best_model.pt`

(Supabase Storage or any other host that gives a direct-download URL works
just as well, if you'd rather not use a GitHub Release.)

## 2. Create the Render service

1. [render.com](https://render.com) -> **New -> Web Service** -> connect this
   GitHub repo.
2. **Root Directory**: `ml`
3. **Environment**: Docker (Render auto-detects `ml/Dockerfile`).
4. **Instance type**: Free is fine for EfficientNet-B0 at this scale (note:
   free instances spin down after 15 min idle — the first request after that
   takes ~30-60s while it spins back up and reloads the model).
5. Environment variables:
   - `MODEL_URL` = the direct download URL from step 1.
   - `RELOAD_TOKEN` = a long random secret (e.g. `openssl rand -hex 32`).
     Enables zero-downtime model promotion: the admin "Promote" button calls
     `POST /admin/reload` on this service with this value as a bearer token to
     hot-swap the live model. Set the **same** value as `ML_API_ADMIN_TOKEN`
     in the Vercel web app's env (step 4). Leave unset only if you don't want
     promotion to work — with it unset, `/admin/reload` returns 503.
   - `WEB_APP_ORIGIN` = your Vercel app's origin (e.g.
     `https://mantis-vision.vercel.app`), once you know it — locks down CORS
     so only your PWA can call this API from a browser. Leave unset for now
     if you don't have the Vercel URL yet; it defaults to allowing all
     origins.
6. Deploy. Watch the build logs — first boot downloads the checkpoint and
   loads the model, so `/health` won't report `model_loaded: true` until that
   finishes.

## 3. Verify it

```bash
curl https://<your-service>.onrender.com/health
curl -F "file=@/path/to/photo.jpg" https://<your-service>.onrender.com/predict
```

## 4. Point the deployed web app at it

In the Vercel project (`apps/web`) -> **Settings -> Environment Variables**:

- `ML_API_URL` = `https://<your-service>.onrender.com`
- `ML_API_ADMIN_TOKEN` = the **same** value you set for `RELOAD_TOKEN` on
  Render (step 2.5). This is what lets the "Promote" button hot-swap the model.

Redeploy the Vercel project (env var changes require a new deploy to take
effect — push a commit or use "Redeploy" in the dashboard).

## 5. Updating the model later

Re-training produces a new `best_model.pt`, published as a GitHub Release
asset by the retrain workflow. Promotion is now fully automatic: in the admin
panel, review a completed run and click **Promote**. The web app calls
`POST /admin/reload` on this service, which downloads the new checkpoint,
verifies it loads, and hot-swaps the live model **with no restart and no env
var edits**. If the download or load fails, the previous model keeps serving
and the error is surfaced in the admin panel.

You no longer need to edit `MODEL_URL` or restart the service to update the
model. (`MODEL_URL` is still used for the very first cold boot; after that,
promotion also rewrites the on-disk checkpoint so a later restart serves the
most recently promoted version.)
