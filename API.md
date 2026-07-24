# API Reference

Mantis Vision exposes two HTTP surfaces:

1. **The ML inference service** (`ml/src/api`, FastAPI) — model predictions and
   model management. Called server-to-server by the web app, never directly by
   the browser.
2. **The web API routes** (`apps/web/src/app/api`, Next.js) — the browser-facing
   endpoints: a thin proxy to the ML service plus the role-gated admin API.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how they interact and
[INSTALL.md](INSTALL.md#environment-variables) for the environment variables
referenced here.

---

## 1. ML inference service (FastAPI)

Base URL: `ML_API_URL` (e.g. `http://localhost:8000` in dev). Runs from
`ml/src/api/main.py`.

### `GET /health`

Liveness + model status. Never fails even if the checkpoint is missing or
unloadable.

**Response** (`200`)

```json
{
  "status": "ok",
  "model_loaded": true,
  "species_classes": ["Kappaphycus_alvarezii"],
  "measurements": ["seaweed_presence", "health_status", "disease", "..."]
}
```

- `model_loaded` — `false` if no checkpoint is present or it failed to load.
- `species_classes` — every class the loaded schema's `species` measurement
  knows about (empty list if the schema has no `species` measurement).
- `measurements` — the measurement keys the loaded checkpoint's schema defines.

### `POST /predict`

Run inference on one image. `multipart/form-data` with a `file` field.

**Request**

```bash
curl -F "file=@/path/to/photo.jpg" $ML_API_URL/predict
```

**Response** (`200`) — legacy flat fields (kept stable for the current PWA)
plus a generic, forward-looking `measurements` map (one entry per schema
measurement):

```json
{
  "species": "Kappaphycus alvarezii",
  "is_seaweed": true,
  "condition": "Disease",
  "health": "Moderate",
  "health_score": 62.1,
  "confidence": 0.974,
  "disease_subtype": "IceIce",
  "dried_pct": null,
  "decayed_pct": null,
  "explanation": "Minor bleaching on branches and early tissue degradation observed.",
  "recommendation": "Increase water movement. Inspect for grazers and early disease signs.",
  "gradcam_png_base64": "...",
  "measurements": {
    "condition":      {"type": "classification", "value": "Disease", "confidence": 0.974, "explanation": "...", "recommendation": "...", "coverage": null, "mask_png_base64": null},
    "disease_subtype":{"type": "classification", "value": "IceIce",  "confidence": 0.88,  "explanation": null,  "recommendation": null, "coverage": null, "mask_png_base64": null},
    "health_score":   {"type": "regression",     "value": 62.1,      "confidence": null,  "explanation": null,  "recommendation": null, "coverage": null, "mask_png_base64": null}
  }
}
```

Each `measurements` entry (`MeasurementResultResponse`):

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `classification` \| `regression` \| `segmentation` |
| `value` | string \| number \| null | Predicted class name or scalar |
| `confidence` | number \| null | Softmax confidence (classification only) |
| `explanation` / `recommendation` | string \| null | Preset copy for the predicted class/range |
| `coverage` | object \| null | Per-class pixel coverage (segmentation) |
| `mask_png_base64` | string \| null | Overlay mask PNG (segmentation) |

**Errors:** `400` if the uploaded file is not an image; `503` if no trained
model is available.

### `POST /admin/reload`

Hot-swap the live model with a promoted checkpoint — no process restart.
Requires a bearer token.

**Request**

```
Authorization: Bearer <RELOAD_TOKEN>
Content-Type: application/json

{ "model_url": "https://.../best_model.pt" }
```

The new checkpoint is downloaded to a **staging file** and loaded into a fresh
predictor *before* anything swaps. Only on success is the live model replaced
and the on-disk checkpoint overwritten (so a later cold restart serves the same
version). On failure the currently-serving model is untouched.

**Responses**

- `200` — swapped; returns the same shape as `/health`.
- `401` — missing/invalid token.
- `503` — reload disabled (`RELOAD_TOKEN` not set on the host).
- `502` — download or load failed; previous model still serving.

---

## 2. Web API routes (Next.js)

Base path `/api`. `/api/member/*` routes are **role-gated**: `middleware.ts`
requires a session, and each handler calls `requireAdmin` /
`requireContributor` (`apps/web/src/lib/supabase/require-admin.ts`). Unauthorized
→ `401`; wrong role → `403`.

### `POST /api/predict` — public

Browser-facing proxy to the ML service's `/predict`. Accepts the same
`multipart/form-data` `file` field, forwards it to `${ML_API_URL}/predict`, and
returns a camelCase-normalized subset for the UI:

```json
{
  "species": "...", "isSeaweed": true, "condition": "...",
  "health": "...", "healthScore": 62.1, "confidence": 0.974,
  "diseaseSubtype": "...", "driedPct": null, "decayedPct": null,
  "explanation": "...", "recommendation": "...", "gradcamPngBase64": "..."
}
```

Tolerates free-tier cold starts (up to ~55s) before returning `502` with a
readable error. Keeps the ML service address off the browser.

### Admin routes

| Route | Methods | Role | Purpose |
|---|---|---|---|
| `/api/member/schema` | `GET`, `PUT` | admin | Read / replace the active measurement schema (stored in Supabase, mirrors `lib/schema.ts`) |
| `/api/member/dataset` | `GET`, `POST`, `PATCH` | contributor+ | List, upload, and label training photos against the active schema |
| `/api/member/team` | `GET`, `POST`, `PATCH` | admin | List teammates, invite a new one, change a role |
| `/api/member/retrain` | `GET`, `POST` | admin | List model runs; `POST` creates a run and dispatches the GitHub Actions retrain workflow |
| `/api/member/retrain/promote` | `POST` | admin | Hot-swap the live model to a completed run's checkpoint (calls ML `/admin/reload`, then records promotion) |

**Retrain (`POST /api/member/retrain`)** inserts a `model_runs` row (`queued`)
then dispatches `retrain.yml` via the GitHub API using `GITHUB_TOKEN` /
`GITHUB_REPO`. Returns `201 {"run": {...}}`. Missing token/repo → `502` with a
clear message.

**Promote (`POST /api/member/retrain/promote`)** — body `{ "modelRunId": "<uuid>" }`.
Order matters: it calls the ML API's `/admin/reload` **first** and only records
`promoted_at`/`promoted_by` in Supabase if the swap succeeds, so nothing is
marked "promoted" unless it is actually serving. Requires `ML_API_ADMIN_TOKEN`
(matching the ML host's `RELOAD_TOKEN`).

### Error conventions

JSON errors throughout: `{ "error": "message" }` with an appropriate status
(`400` bad input, `401` unauthenticated, `403` wrong role, `404` not found,
`500` server misconfiguration, `502` upstream/ML failure). The web routes
surface the ML service's own error text where useful.
