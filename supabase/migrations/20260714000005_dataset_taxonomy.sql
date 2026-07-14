-- Make the dataset structure admin-editable instead of hardcoded in
-- ml/config.py + apps/web/src/lib/taxonomy.ts + explanations.py. The taxonomy
-- (active species, condition "buckets", severities, disease subtypes, the
-- heuristic regression anchors, and the preset explanation/recommendation text
-- shown to end users) now lives here as a versioned JSONB document. The web
-- app reads/writes it; the retrain workflow exports the active version into the
-- training run (scripts/export_taxonomy.py) so the model, its label taxonomy,
-- and its preset copy all travel together and hot-swap in on promotion.
--
-- Append-only for auditability, mirroring model_runs: each admin edit inserts a
-- new row; the row with the latest created_at is the active taxonomy.

create table dataset_taxonomy (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  created_by uuid references profiles (id),
  doc jsonb not null
);

alter table dataset_taxonomy enable row level security;

create policy "dataset_taxonomy_admin_all" on dataset_taxonomy
  for all using (
    exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );

-- Seed with the taxonomy that used to be hardcoded, so behaviour is unchanged
-- until an admin edits it. Keep in sync with DEFAULT_TAXONOMY in
-- apps/web/src/lib/taxonomy.ts and the fallbacks in ml/config.py.
insert into dataset_taxonomy (doc) values ('{
  "species": [{"name": "Kappaphycus alvarezii", "slug": "Kappaphycus_alvarezii"}],
  "active_species_slug": "Kappaphycus_alvarezii",
  "severities": ["Moderate", "Low"],
  "disease_moderate_min": 45.0,
  "conditions": [
    {"name": "Background", "is_background": true, "fixed_severity": null, "requires_subtype": false,
     "health_score_anchor": null, "health_score_anchors_by_severity": {}, "dried_extent_anchor": 0.0, "decayed_extent_anchor": 0.0,
     "explanation": "No seaweed specimen was detected in this image.",
     "recommendation": "Point the camera at a seaweed specimen, filling the frame, and try again."},
    {"name": "Healthy", "is_background": false, "fixed_severity": null, "requires_subtype": false,
     "health_score_anchor": 90.0, "health_score_anchors_by_severity": {}, "dried_extent_anchor": 0.0, "decayed_extent_anchor": 0.0,
     "explanation": "Vivid, even coloration with intact branching and no whitening, lesions, or breakage detected.",
     "recommendation": "Continue routine monitoring. No action needed."},
    {"name": "Disease", "is_background": false, "fixed_severity": null, "requires_subtype": true,
     "health_score_anchor": null, "health_score_anchors_by_severity": {"Moderate": 60.0, "Low": 30.0}, "dried_extent_anchor": 0.0, "decayed_extent_anchor": 20.0,
     "explanation": "Discrete lesions or spotting consistent with a disease outbreak, distinct from generalized decay.",
     "recommendation": "Isolate affected line segments and confirm the pathogen before treating."},
    {"name": "Decay", "is_background": false, "fixed_severity": "Low", "requires_subtype": false,
     "health_score_anchor": 20.0, "health_score_anchors_by_severity": {}, "dried_extent_anchor": 10.0, "decayed_extent_anchor": 80.0,
     "explanation": "Tissue melting with dark, mushy patches and a breakdown of branch structure, consistent with decay.",
     "recommendation": "Remove affected fragments to prevent spread. Check water quality (temperature, salinity)."},
    {"name": "Dried", "is_background": false, "fixed_severity": "Low", "requires_subtype": false,
     "health_score_anchor": 5.0, "health_score_anchors_by_severity": {}, "dried_extent_anchor": 90.0, "decayed_extent_anchor": 0.0,
     "explanation": "Tissue is brittle, bleached, and fully desiccated, with no living tissue remaining.",
     "recommendation": "Remove and dispose of dried-out material. Inspect the surrounding line for early damage."}
  ],
  "disease_subtypes": [
    {"name": "IceIce", "note": "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity."},
    {"name": "Epiphyte", "note": "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow."},
    {"name": "Bacterial", "note": "Possible bacterial infection: isolate and consult a specialist before any treatment."},
    {"name": "Bleaching", "note": "Bleaching suspected: check for temperature/light stress and relocate if possible."},
    {"name": "Unknown", "note": "Subtype unclear: photograph affected areas closely and consult a specialist."}
  ]
}'::jsonb);

-- The old CHECK constraints on training_images froze the taxonomy at the DB
-- layer — an admin-added condition or subtype would be rejected on insert.
-- Drop them; the admin API now validates condition/severity/subtype against
-- the active dataset_taxonomy document instead (which can grow over time).
alter table training_images
  drop constraint if exists training_images_condition_check,
  drop constraint if exists training_images_severity_check,
  drop constraint if exists training_images_subtype_check;
