# Running the ML API locally (and why "endless loading" happens)

If analysis in the PWA seems to hang for a long time before returning a
result (or an error), it's almost certainly **not** a bottleneck in a
third-party model API — `ml/src/api/main.py` is your own FastAPI service
running your own PyTorch checkpoint, not a call to an external inference
provider. The likely cause is a **cold start**: free-tier hosts (Render free
tier, Hugging Face Spaces) spin the container down after ~15 minutes idle,
and the first request afterwards has to wait ~30-60s for it to boot back up
and reload the model before it can even start inferring. The client and
proxy timeouts (`apps/web/src/components/UploadCard.tsx`'s
`REQUEST_TIMEOUT_MS`, `apps/web/src/app/api/predict/route.ts`'s
`UPSTREAM_TIMEOUT_MS`) are already tuned to just barely wait that out, so a
cold start reads as a near-minute-long spinner before either a result or a
timeout error appears.

## Option A — run it locally for dev/demo use

Useful for local frontend development, or briefly pointing a deployed app at
your own machine for a demo. Not suitable for anything real users depend on
(see the tradeoff below).

1. Start the API on your machine:
   ```bash
   cd ml
   pip install -r requirements-serve.txt
   uvicorn src.api.main:app --reload --port 8000
   ```
   This needs `ml/checkpoints/best_model.pt` present locally (from a prior
   `python -m src.train`, or copied down from wherever it's hosted).

2. Point the web app at it. For local frontend dev, `apps/web/.env.local`:
   ```
   ML_API_URL=http://localhost:8000
   ```
   already works if both are running on the same machine.

3. To expose it beyond `localhost` (e.g. to a deployed Vercel app, or to test
   from a phone), tunnel it:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   (or `ngrok http 8000`). Copy the resulting HTTPS URL and set it as
   `ML_API_URL` — either in `.env.local`, or as a Vercel project environment
   variable if you're temporarily pointing a deployed app at your laptop.

Caveats: the tunnel URL can rotate on restart unless you set up a named
tunnel (Cloudflare) or reserved domain (ngrok, requires an account). The
whole setup breaks the moment the laptop sleeps, changes networks, or is
closed — there's no auto-reconnect.

## Option B — pay for an always-on tier (recommended for anything real)

Render Starter (~$7/mo) or Fly.io `shared-cpu-1x` always-on (~$5/mo) removes
the idle spin-down entirely — no more 30-60s cold starts — and gives a
stable URL that doesn't depend on a personal machine staying on and
reachable. Setup is the same `ml/Dockerfile` already in this repo; see
[`DEPLOY_ML_API.md`](DEPLOY_ML_API.md) and pick a non-free instance type
instead of the free tier described there.

| | Laptop + tunnel | Always-on paid host |
|---|---|---|
| Cost | $0 | ~$5-7/mo |
| Cold starts | None while running | None |
| Uptime | Only while the laptop is awake, online, and the tunnel is up | 24/7 |
| Stable URL | Only with a named tunnel/reserved domain | Yes |
| Suitable for | Local dev, quick demos | Anyone else actually using the app |

## Verifying

From a clean shell: run `uvicorn` locally, start a tunnel, point
`apps/web/.env.local`'s `ML_API_URL` at the tunnel URL, then run the PWA
(`npm run dev` in `apps/web`) and confirm an end-to-end analysis completes
against the tunneled local service.
