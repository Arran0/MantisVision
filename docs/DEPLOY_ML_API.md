# Deploying the ML inference API (Render)

`ml/src/api` (FastAPI + PyTorch) can't run on Vercel — it needs a real
container host. These are the steps for Render, which is free-tier friendly
and builds straight from the `ml/Dockerfile` already in this repo.

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

Redeploy the Vercel project (env var changes require a new deploy to take
effect — push a commit or use "Redeploy" in the dashboard).

## 5. Updating the model later

Re-training produces a new `best_model.pt`. Upload it as a new GitHub Release
asset, update `MODEL_URL` on Render to the new URL, and redeploy (or just
restart) the Render service — it re-downloads on the next boot.
