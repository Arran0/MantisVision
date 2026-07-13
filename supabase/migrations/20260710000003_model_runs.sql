-- Tracks manually-triggered retraining runs (see .github/workflows/retrain.yml
-- and ml/scripts/retrain_and_report.py). A row here is the audit trail an
-- admin reviews before deciding whether to promote a checkpoint to
-- production — promotion is always a deliberate, separate step, never automatic.

create table model_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  triggered_by uuid references profiles (id),
  status text not null default 'queued' check (status in ('queued', 'running', 'completed', 'failed')),
  github_run_id text,
  dataset_image_count int,
  metrics jsonb,
  checkpoint_url text,
  error text,
  promoted_at timestamptz,
  promoted_by uuid references profiles (id)
);

alter table model_runs enable row level security;

create policy "model_runs_admin_all" on model_runs
  for all using (
    exists (select 1 from profiles where id = auth.uid() and role = 'admin')
  );
