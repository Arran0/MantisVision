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

**Response** (`200`) — fully schema-driven: one entry per measurement the
active schema defines, in schema order. There are no flat/legacy fields (no
`species`, `is_seaweed`, `confidence`, ...) — a client renders whichever
measurements are present, using each entry's own `label`/`type`/`unit`/etc.
A measurement gated off by `applies_when` (e.g. `health_status` when
`seaweed_presence` predicted `"No"`) still appears, with `value: null`.

```json
{
  "measurements": {
    "seaweed_presence": {"type": "classification", "label": "Seaweed presence", "value": "Yes", "confidence": 0.99, "explanation": "...", "recommendation": "...", "unit": null, "min": null, "max": null, "coverage": null, "seg_colors": null, "mask_png_base64": null, "gradcam_png_base64": "..."},
    "health_status":    {"type": "classification", "label": "Health status",    "value": "Moderate", "confidence": 0.974, "explanation": "...", "recommendation": "...", "unit": null, "min": null, "max": null, "coverage": null, "seg_colors": null, "mask_png_base64": null, "gradcam_png_base64": null},
    "disease":          {"type": "classification", "label": "Disease",         "value": "IceIce", "confidence": 0.88, "explanation": null, "recommendation": "...", "unit": null, "min": null, "max": null, "coverage": null, "seg_colors": null, "mask_png_base64": null, "gradcam_png_base64": null},
    "dried":            {"type": "regression",     "label": "Dried",          "value": 12.4, "confidence": null, "explanation": null, "recommendation": null, "unit": "%", "min": 0.0, "max": 100.0, "coverage": null, "seg_colors": null, "mask_png_base64": null, "gradcam_png_base64": null}
  }
}
```

Each `measurements` entry (`MeasurementResultResponse`):

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `classification` \| `regression` \| `segmentation` |
| `label` | string | Admin-facing display label from the schema |
| `value` | string \| number \| null | Predicted class name or scalar (`null` if gated off by `applies_when`) |
| `confidence` | number \| null | Softmax confidence (classification only) |
| `explanation` / `recommendation` | string \| null | Preset copy for the predicted class/range |
| `unit` / `min` / `max` | string/number \| null | Regression only — unit label and value bounds for client-side scaling |
| `coverage` | object \| null | Per-class pixel coverage (segmentation) |
| `seg_colors` | object \| null | Per-class hex color, keyed the same as `coverage` (segmentation) |
| `mask_png_base64` | string \| null | Overlay mask PNG, `""` unless `ENABLE_SEGMENTATION_OVERLAY` (segmentation) |
| `gradcam_png_base64` | string \| null | Heatmap PNG for the one classification head Grad-CAM ran against (first in schema order), `""` unless `ENABLE_GRADCAM`, `null` for every other measurement |

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
returns the same schema-driven `measurements` map, camelCase-normalized:

```json
{
  "measurements": {
    "seaweed_presence": {"type": "classification", "label": "Seaweed presence", "value": "Yes", "confidence": 0.99, "explanation": "...", "recommendation": "...", "unit": null, "min": null, "max": null, "coverage": null, "segColors": null, "maskPngBase64": null, "gradcamPngBase64": "..."}
  }
}
```

Measurement dict keys (`seaweed_presence`, `health_status`, ...) are passed
through verbatim from the schema — they're admin-defined, not fixed field
names, so they aren't camelCased.

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
