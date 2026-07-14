-- Make the dataset structure admin-editable and let the model grow new
-- measurements without code changes, instead of hardcoding a fixed
-- condition/severity/subtype taxonomy across ml/config.py,
-- ml/src/inference/explanations.py, and apps/web/src/lib/taxonomy.ts.
--
-- The "measurement schema" is a versioned JSONB document describing every
-- per-image measurement the model predicts: its type (classification,
-- regression, or segmentation), its classes/range/mask-classes, loss weight,
-- an optional `applies_when` (so e.g. "disease_subtype" is only meaningful
-- when "condition" == "Disease"), and — for classification — preset
-- explanation/recommendation copy per class. The web app reads/writes it;
-- the retrain workflow exports the active version into the training run
-- (ml/scripts/export_schema.py) so the model, its head configuration, and its
-- preset copy all travel together in the checkpoint and hot-swap in on
-- promotion (ml/src/api/main.py's /admin/reload).
--
-- Append-only for auditability, mirroring model_runs: each admin edit inserts
-- a new row; the row with the latest created_at is the active schema.

create table measurement_schema (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  created_by uuid references profiles (id),
  doc jsonb not null
);

alter table measurement_schema enable row level security;

create policy "measurement_schema_admin_all" on measurement_schema
  for all using (
    exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );

-- Seed with a schema that reproduces today's fixed taxonomy, so behaviour is
-- unchanged until an admin edits it. Keep in sync with DEFAULT_SCHEMA in
-- apps/web/src/lib/schema.ts and the fallback in ml/config.py.
insert into measurement_schema (doc) values ('{
  "species": [{"name": "Kappaphycus alvarezii", "slug": "Kappaphycus_alvarezii"}],
  "active_species_slug": "Kappaphycus_alvarezii",
  "health_moderate_min": 45.0,
  "health_healthy_min": 75.0,
  "measurements": [
    {
      "key": "condition",
      "label": "Condition",
      "type": "classification",
      "loss_weight": 1.0,
      "background_class": "Background",
      "classes": [
        {"name": "Background",
         "explanation": "No seaweed specimen was detected in this image.",
         "recommendation": "Point the camera at a seaweed specimen, filling the frame, and try again."},
        {"name": "Healthy",
         "explanation": "Vivid, even coloration with intact branching and no whitening, lesions, or breakage detected.",
         "recommendation": "Continue routine monitoring. No action needed."},
        {"name": "Disease",
         "explanation": "Discrete lesions or spotting consistent with a disease outbreak, distinct from generalized decay.",
         "recommendation": "Isolate affected line segments and confirm the pathogen before treating."},
        {"name": "Decay",
         "explanation": "Tissue melting with dark, mushy patches and a breakdown of branch structure, consistent with decay.",
         "recommendation": "Remove affected fragments to prevent spread. Check water quality (temperature, salinity)."},
        {"name": "Dried",
         "explanation": "Tissue is brittle, bleached, and fully desiccated, with no living tissue remaining.",
         "recommendation": "Remove and dispose of dried-out material. Inspect the surrounding line for early damage."}
      ]
    },
    {
      "key": "disease_subtype",
      "label": "Disease subtype",
      "type": "classification",
      "loss_weight": 0.5,
      "applies_when": {"key": "condition", "equals": "Disease"},
      "classes": [
        {"name": "IceIce", "note": "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity."},
        {"name": "Epiphyte", "note": "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow."},
        {"name": "Bacterial", "note": "Possible bacterial infection: isolate and consult a specialist before any treatment."},
        {"name": "Bleaching", "note": "Bleaching suspected: check for temperature/light stress and relocate if possible."},
        {"name": "Unknown", "note": "Subtype unclear: photograph affected areas closely and consult a specialist."}
      ]
    },
    {
      "key": "health_score",
      "label": "Health score",
      "type": "regression",
      "loss_weight": 1.0,
      "unit": "score",
      "min": 0.0,
      "max": 100.0,
      "applies_when": {"key": "condition", "not_equals": "Background"}
    },
    {
      "key": "dried_extent",
      "label": "Dried extent",
      "type": "regression",
      "loss_weight": 0.5,
      "unit": "pct",
      "min": 0.0,
      "max": 100.0,
      "applies_when": {"key": "condition", "not_equals": "Background"}
    },
    {
      "key": "decayed_extent",
      "label": "Decayed extent",
      "type": "regression",
      "loss_weight": 0.5,
      "unit": "pct",
      "min": 0.0,
      "max": 100.0,
      "applies_when": {"key": "condition", "not_equals": "Background"}
    }
  ]
}'::jsonb);

-- Per-image annotations move from folder-name encoding to a column/CSV-style
-- map keyed by measurement key (value = class name string, numeric value, or
-- — for a segmentation measurement — a training-masks storage path). The old
-- condition/severity/subtype columns are kept for back-compat/query but are
-- no longer the source of truth for training.
alter table training_images
  add column measurements jsonb not null default '{}';

-- The old CHECK constraints froze the taxonomy at the DB layer — an
-- admin-added condition, subtype, or new measurement's class would be
-- rejected on insert. Drop them; the admin API now validates against the
-- active measurement_schema document instead (which can grow over time).
alter table training_images
  drop constraint if exists training_images_condition_check,
  drop constraint if exists training_images_severity_check,
  drop constraint if exists training_images_subtype_check;

-- Private bucket for segmentation ground-truth masks (e.g. a hand-painted or
-- uploaded biofouling/epiphyte coverage mask), mirroring the training-images
-- bucket''s admin-only access pattern.
insert into storage.buckets (id, name, public)
values ('training-masks', 'training-masks', false)
on conflict (id) do nothing;

create policy "training_masks_bucket_admin_all" on storage.objects
  for all
  using (
    bucket_id = 'training-masks'
    and exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  )
  with check (
    bucket_id = 'training-masks'
    and exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );
