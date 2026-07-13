-- The admin-labeled training dataset: one row per uploaded+labeled photo.
-- Distinct from the (separately planned, not part of this migration set)
-- end-user `predictions` history table from docs/STEP_BY_STEP.md — that one
-- is about logging what end users analysed; this one is the growing corpus
-- admins use to retrain the model. Column set mirrors
-- ml/metadata/labels_template.csv. `health` is kept in sync by hand with
-- ml/config.py's CLASS_NAMES (see apps/web/src/lib/healthClasses.ts).

create table training_images (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  created_by uuid references profiles (id),
  storage_path text not null,
  species text not null default 'Kappaphycus alvarezii',
  colour text,
  health text not null check (health in ('Healthy', 'Moderate', 'Low', 'Decay', 'Dried', 'Disease')),
  notes text,
  farm text,
  gps point,
  water_temperature_c numeric,
  salinity_ppt numeric,
  depth_m numeric,
  camera text,
  captured_at timestamptz,
  split text check (split in ('train', 'validation', 'test')),
  status text not null default 'labeled' check (status in ('labeled', 'used_in_training', 'rejected'))
);

alter table training_images enable row level security;

create policy "training_images_admin_all" on training_images
  for all using (
    exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );

-- Private bucket for the actual photo bytes; the admin API always issues
-- short-lived signed URLs to display them, never a public URL.
insert into storage.buckets (id, name, public)
values ('training-images', 'training-images', false)
on conflict (id) do nothing;

create policy "training_images_bucket_admin_all" on storage.objects
  for all
  using (
    bucket_id = 'training-images'
    and exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  )
  with check (
    bucket_id = 'training-images'
    and exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );
