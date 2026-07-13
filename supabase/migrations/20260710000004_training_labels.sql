-- Move training_images from a single flat `health` class to the multi-head
-- taxonomy: condition (incl. Background), plus Disease severity/subtype/name,
-- plus optional numeric ground-truth for the regression heads (score/extents).
-- Matches the folder-naming convention in ml/src/data/labels.py.

alter table training_images
  add column condition text check (
    condition in ('Background', 'Healthy', 'Disease', 'Decay', 'Dried')
  ),
  add column severity text check (severity in ('Moderate', 'Low')),
  add column subtype text check (
    subtype in ('IceIce', 'Epiphyte', 'Bacterial', 'Bleaching', 'Unknown')
  ),
  add column disease_name text,
  add column health_score numeric check (health_score between 0 and 100),
  add column dried_pct numeric check (dried_pct between 0 and 100),
  add column decayed_pct numeric check (decayed_pct between 0 and 100),
  add column is_background boolean not null default false;

-- The old `health` CHECK constraint (Healthy/Moderate/Low/Decay/Dried/Disease)
-- no longer reflects the taxonomy. Drop it but keep the column nullable so any
-- historical rows are preserved; new rows populate `condition` instead.
alter table training_images
  alter column health drop not null,
  drop constraint if exists training_images_health_check;
