# Deploying the ML inference API (Hugging Face Spaces)

`ml/src/api` (FastAPI + PyTorch) needs a real container host — Vercel can't
run it. This is the recommended path: Hugging Face Spaces' free CPU Basic
tier gives **16 GB RAM** (vs. Render's free 512 MB), which is enough
headroom to leave Grad-CAM enabled and stop worrying about the service
getting OOM-killed. It builds straight from the `ml/Dockerfile` already in
this repo, the same as Render did.

Because this repo is a monorepo and a Space is its own git repo, deploys
work via a GitHub Action that mirrors the `ml/` subfolder to the Space on
every push — see step 3.

## 1. Host the checkpoint somewhere downloadable

`ml/checkpoints/best_model.pt` is gitignored, so the Space's clone of `ml/`
won't have it. The API downloads it at startup instead, from a URL you
provide via `MODEL_URL`. (Skip this if you already did it for Render — the
same URL works here.)

1. On GitHub: **Releases -> Draft a new release** (any tag, e.g. `model-v1`).
2. Upload `best_model.pt` as a release asset and publish.
3. Copy the asset's direct download URL — looks like:
   `https://github.com/<owner>/<repo>/releases/download/model-v1/best_model.pt`

## 2. Create the Space

1. [huggingface.co](https://huggingface.co) -> sign in (or create a free
   account) -> **New Space**.
2. **Space SDK**: Docker -> **Blank** template (the repo already has a
   Dockerfile; you don't need one of HF's presets).
3. Name it whatever you like, e.g. `mantis-vision-api`. Visibility: Public or
   Private both work — the API doesn't need to be public to be callable.
4. Once created, go to the Space's **Settings** tab and add these
   **Repository secrets**:
   - `MODEL_URL` = the direct download URL from step 1.
   - `RELOAD_TOKEN` = a long random secret (e.g. `openssl rand -hex 32`).
     Enables zero-downtime model promotion: the admin "Promote" button calls
     `POST /admin/reload` on the Space with this as a bearer token to hot-swap
     the live model. Set the **same** value as `ML_API_ADMIN_TOKEN` in the
     Vercel web app's env (step 5). With it unset, `/admin/reload` returns 503.
   - `WEB_APP_ORIGIN` = your Vercel app's origin (e.g.
     `https://mantis-vision.vercel.app`) — locks CORS down to just your PWA.
   - `ENABLE_GRADCAM` = `true` (optional — safe to enable here since 16 GB
     comfortably covers the extra memory Grad-CAM's backward pass needs;
     leave unset/`false` if you'd rather keep it off).
5. Get a write-access token: **Settings (your HF account, not the Space) ->
   Access Tokens -> New token**, role **Write**. Copy it — you won't see it
   again.

## 3. Wire up the auto-deploy GitHub Action

This repo has `.github/workflows/deploy-ml-to-huggingface.yml`, which pushes
`ml/` to the Space's git remote on every push to `main` that touches `ml/**`
(or on demand via **Actions -> Run workflow**).

In this GitHub repo -> **Settings -> Secrets and variables -> Actions ->
New repository secret**, add:

- `HF_TOKEN` = the write token from step 2.5.
- `HF_SPACE` = `<your-hf-username>/<space-name>` (e.g. `arran0/mantis-vision-api`).

Then push to `main` (or run the workflow manually) — the Action mirrors
`ml/` into the Space, which triggers a Docker build there. Watch progress
under the Space's **Logs** tab; first boot also downloads the checkpoint via
`MODEL_URL`, so `/health` won't report `model_loaded: true` until that's
done.

## 4. Verify it

Your Space is reachable at `https://<username>-<space-name>.hf.space` (dashes
replace underscores/spaces in the name — the exact URL is shown at the top
of the Space page).

```bash
curl https://<username>-<space-name>.hf.space/health
curl -F "file=@/path/to/photo.jpg" https://<username>-<space-name>.hf.space/predict
```

## 5. Point the deployed web app at it

In the Vercel project (`apps/web`) -> **Settings -> Environment Variables**:

- `ML_API_URL` = `https://<username>-<space-name>.hf.space`
- `ML_API_ADMIN_TOKEN` = the **same** value you set for `RELOAD_TOKEN` in the
  Space's secrets (step 2.4). This is what lets the "Promote" button hot-swap
  the model.

Redeploy the Vercel project for the env var change to take effect.

## 6. Updating the model later

Re-training produces a new `best_model.pt`, published as a GitHub Release
asset by the retrain workflow. Promotion is now fully automatic: in the admin
panel, review a completed run and click **Promote**. The web app calls
`POST /admin/reload` on the Space, which downloads the new checkpoint,
verifies it loads, and hot-swaps the live model **with no factory reboot and
no secret edits**. If the download or load fails, the previous model keeps
serving and the error is surfaced in the admin panel.

You no longer need to edit the `MODEL_URL` secret or reboot the Space to
update the model. (`MODEL_URL` is still used for the first cold boot; after
that, promotion also rewrites the on-disk checkpoint so a later reboot serves
the most recently promoted version.)

## Notes

- Free Spaces sleep after a period of inactivity, same as Render's free
  tier — expect a slower first request after idling while it wakes back up.
  The web app already waits up to ~60s for this (see
  `apps/web/src/app/api/predict/route.ts`).
- Render (`docs/DEPLOY_ML_API.md`) still works if you'd rather stick with
  it; the two aren't mutually exclusive; just point `ML_API_URL` at whichever
  one you want live.
