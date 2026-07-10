-- Admin roles for the MantisVision admin panel (upload/label dataset,
-- trigger retraining). Distinct from any future end-user account system —
-- there is no public signup route; admin accounts are provisioned manually:
--
--   1. Create the user in the Supabase Auth dashboard (or via
--      supabase.auth.admin.createUser in the SQL editor / a one-off script).
--      This fires the trigger below and creates a 'viewer' profile row.
--   2. Promote them:
--        update profiles set role = 'admin' where email = 'someone@example.com';

create table profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  role text not null default 'viewer' check (role in ('admin', 'viewer')),
  email text,
  created_at timestamptz not null default now()
);

alter table profiles enable row level security;

-- A user can read their own profile — needed so server-side code can check
-- "am I admin" using the caller's own session (not just the service role).
create policy "profiles_select_own" on profiles
  for select using (auth.uid() = id);

-- Auto-create a 'viewer' profile row whenever a new auth.users row appears,
-- so no authenticated user is ever missing a profiles row.
create function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, email) values (new.id, new.email);
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
