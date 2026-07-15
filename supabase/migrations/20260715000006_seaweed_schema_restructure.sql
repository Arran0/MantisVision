-- Restructure the measurement schema around a fixed, must-required column set.
--
-- What changes vs. the seed in 20260714000005_measurement_schema.sql:
--   * "Background" is renamed to "no seaweed" and split out of the old
--     condition column into its own primary classifier: seaweed_presence
--     (Yes/No), with "No" as the background / no-subject class.
--   * The old condition column becomes health_status — an explicit
--     Healthy/Moderate/Low classification (no longer a bucketed score).
--   * Dried and Decayed become their own 0–100 regression columns.
--   * Disease is its own classification (one class per named disease plus an
--     explicit NoDisease class) with a separate disease_severity 0–100 score.
--   * Colour becomes a fixed-palette classification.
--   * A block of lab/quality regression columns is added (carrageenan yield,
--     gel strength, viscosity, daily growth rate, per-mineral content, CAW,
--     impurities, sulfate content, acid-insoluble ash, ash content).
--
-- These are the required backbone every dataset collects: each carries
-- "locked": true so the admin Structure editor renders it read-only and the
-- schema API rejects removing/retyping it. "disease" additionally carries
-- "extensible_classes": true so the admin can still add a class per new
-- disease. Species stays admin-managed at the top level.
--
-- Append-only, mirroring model_runs: this inserts a new measurement_schema row
-- which (being newest) becomes the active schema. Keep in sync with
-- DEFAULT_SCHEMA in apps/web/src/lib/schema.ts and ml/config.py.

insert into measurement_schema (doc) values ('{
  "species": [{"name": "Kappaphycus alvarezii", "slug": "Kappaphycus_alvarezii"}],
  "active_species_slug": "Kappaphycus_alvarezii",
  "health_moderate_min": 45.0,
  "health_healthy_min": 75.0,
  "measurements": [
    {
      "key": "seaweed_presence",
      "label": "Seaweed presence",
      "type": "classification",
      "loss_weight": 1.0,
      "background_class": "No",
      "locked": true,
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
      "key": "health_status",
      "label": "Health status",
      "type": "classification",
      "loss_weight": 1.0,
      "applies_when": {"key": "seaweed_presence", "equals": "Yes"},
      "locked": true,
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
      "applies_when": {"key": "seaweed_presence", "equals": "Yes"},
      "locked": true,
      "extensible_classes": true,
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
      "applies_when": {"key": "disease", "not_equals": "NoDisease"},
      "locked": true
    },
    {
      "key": "dried",
      "label": "Dried",
      "type": "regression",
      "loss_weight": 0.5,
      "unit": "%",
      "min": 0.0,
      "max": 100.0,
      "applies_when": {"key": "seaweed_presence", "equals": "Yes"},
      "locked": true
    },
    {
      "key": "decayed",
      "label": "Decayed",
      "type": "regression",
      "loss_weight": 0.5,
      "unit": "%",
      "min": 0.0,
      "max": 100.0,
      "applies_when": {"key": "seaweed_presence", "equals": "Yes"},
      "locked": true
    },
    {
      "key": "colour",
      "label": "Colour",
      "type": "classification",
      "loss_weight": 0.5,
      "applies_when": {"key": "seaweed_presence", "equals": "Yes"},
      "locked": true,
      "classes": [
        {"name": "Green"}, {"name": "Red"}, {"name": "Brown"}, {"name": "Yellow"},
        {"name": "Orange"}, {"name": "White"}, {"name": "Black"}
      ]
    },
    {"key": "carrageenan_yield", "label": "Carrageenan Yield", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "gel_strength", "label": "Gel Strength", "type": "regression", "loss_weight": 0.5, "unit": "g/cm²", "min": 0.0, "max": 2000.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "viscosity", "label": "Viscosity", "type": "regression", "loss_weight": 0.5, "unit": "cP", "min": 0.0, "max": 1000.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "daily_growth_rate", "label": "Daily Growth Rate", "type": "regression", "loss_weight": 0.5, "unit": "%/day", "min": 0.0, "max": 100.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "mineral_ca", "label": "Mineral Content — Ca", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "mineral_mg", "label": "Mineral Content — Mg", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "mineral_k", "label": "Mineral Content — K", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "mineral_na", "label": "Mineral Content — Na", "type": "regression", "loss_weight": 0.5, "unit": "mg/kg", "min": 0.0, "max": 100000.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "caw", "label": "Clean Anhydrous Weed (CAW)", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "impurities", "label": "Impurities", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "sulfate_content", "label": "Sulfate Content", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "acid_insoluble_ash", "label": "Acid-Insoluble Ash", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true},
    {"key": "ash_content", "label": "Ash Content", "type": "regression", "loss_weight": 0.5, "unit": "%", "min": 0.0, "max": 100.0, "applies_when": {"key": "seaweed_presence", "equals": "Yes"}, "locked": true}
  ]
}'::jsonb);
