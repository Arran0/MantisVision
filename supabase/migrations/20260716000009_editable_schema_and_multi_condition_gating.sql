-- Two changes to the measurement schema format:
--
-- 1. Drop the "locked"/"extensible_classes" flags. Every measurement
--    (including seaweed_presence, species, health_status, disease, colour,
--    and the lab-quality metrics) is now freely editable and removable from
--    the admin Structure editor — there is no more must-required backbone
--    that blocks the admin from removing or retyping a measurement.
--
-- 2. applies_when becomes a list of AND-combined conditions instead of a
--    single condition object, so a measurement can be gated on more than one
--    sibling classification at once (e.g. "only when seaweed_presence == Yes
--    AND disease != NoDisease"). A single condition is now wrapped in a
--    one-element list; the app also tolerates the old single-object shape on
--    read (see normalizeSchemaDoc / schema_from_dict's _applies_when), so an
--    existing row doesn't need to be migrated in place — this insert simply
--    makes the *new* active schema use the new shape going forward.
--
-- Append-only, mirroring model_runs: this inserts a new measurement_schema
-- row which (being newest) becomes the active schema. Keep in sync with
-- DEFAULT_SCHEMA in apps/web/src/lib/schema.ts and ml/config.py.

insert into measurement_schema (doc) values ('{
  "health_moderate_min": 45.0,
  "health_healthy_min": 75.0,
  "measurements": [
    {
      "key": "seaweed_presence",
      "label": "Seaweed presence",
      "type": "classification",
      "loss_weight": 1.0,
      "background_class": "No",
      "classes": [
        {"name": "Yes",
         "explanation": "A seaweed specimen was detected in this image.",
         "recommendation": "Continue with the assessment below."},
        {"name": "No",
         "explanation": "No seaweed specimen was detected in this image.",
         "recommendation": "Point the camera at a seaweed specimen, filling the frame, and try again."}
      ]
    },
    {
      "key": "species",
      "label": "Species",
      "type": "classification",
      "loss_weight": 1.0,
      "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}],
      "classes": [
        {"name": "Kappaphycus_alvarezii"}
      ]
    },
    {
      "key": "health_status",
      "label": "Health status",
      "type": "classification",
      "loss_weight": 1.0,
      "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}],
      "classes": [
        {"name": "Healthy",
         "explanation": "Vivid, even coloration with intact branching and no whitening, lesions, or breakage.",
         "recommendation": "Continue routine monitoring. No action needed."},
        {"name": "Moderate",
         "explanation": "Some discoloration or minor structural loss, but the specimen is largely intact.",
         "recommendation": "Increase monitoring frequency and check water quality (temperature, salinity)."},
        {"name": "Low",
         "explanation": "Extensive discoloration, tissue loss, or structural breakdown across the specimen.",
         "recommendation": "Remove affected fragments to prevent spread and investigate the cause promptly."}
      ]
    },
    {
      "key": "disease",
      "label": "Disease",
      "type": "classification",
      "loss_weight": 0.5,
      "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}],
      "classes": [
        {"name": "NoDisease", "explanation": "No disease detected."},
        {"name": "IceIce", "note": "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity."},
        {"name": "Epiphyte", "note": "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow."},
        {"name": "Bacterial", "note": "Possible bacterial infection: isolate and consult a specialist before any treatment."},
        {"name": "Bleaching", "note": "Bleaching suspected: check for temperature/light stress and relocate if possible."}
      ]
    },
    {
      "key": "disease_severity",
      "label": "Disease severity",
      "type": "regression",
      "loss_weight": 0.5,
      "unit": "score",
      "min": 0.0,
      "max": 100.0,
      "applies_when": [{"key": "disease", "not_equals": "NoDisease"}]
    },
    {"key": "dried", "label": "Dried", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "decayed", "label": "Decayed", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {
      "key": "colour",
      "label": "Colour",
      "type": "classification",
      "loss_weight": 0.5,
      "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}],
      "classes": [
        {"name": "Green"}, {"name": "Red"}, {"name": "Brown"}, {"name": "Yellow"},
        {"name": "Orange"}, {"name": "White"}, {"name": "Black"}
      ]
    },
    {"key": "carrageenan_yield", "label": "Carrageenan Yield", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "gel_strength", "label": "Gel Strength", "type": "regression", "loss_weight": 0.5, "unit": "g/cm²", "min": 0.0, "max": 2000.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "viscosity", "label": "Viscosity", "type": "regression", "loss_weight": 0.5, "unit": "cP", "min": 0.0, "max": 1000.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "daily_growth_rate", "label": "Daily Growth Rate", "type": "regression", "loss_weight": 0.5, "unit": "%/day", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "mineral_ca", "label": "Mineral Content — Ca", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "mineral_mg", "label": "Mineral Content — Mg", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "mineral_k", "label": "Mineral Content — K", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "mineral_na", "label": "Mineral Content — Na", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "caw", "label": "Clean Anhydrous Weed (CAW)", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "impurities", "label": "Impurities", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "sulfate_content", "label": "Sulfate Content", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "acid_insoluble_ash", "label": "Acid-Insoluble Ash", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]},
    {"key": "ash_content", "label": "Ash Content", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": [{"key": "seaweed_presence", "equals": "Yes"}]}
  ]
}'::jsonb);
