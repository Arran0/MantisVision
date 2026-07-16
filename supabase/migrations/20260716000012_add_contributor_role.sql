-- Two admin-panel account levels: 'admin' (full access — dataset, structure,
-- retrain, and team management) and 'contributor' (dataset labeling + their
-- own contribution stats only). 'viewer' is kept as the safe zero-access
-- default so any auto-provisioned profile row (see handle_new_user) can't see
-- the dashboard until an admin grants a real level.
--
-- Invited users are created with the level the inviting admin chose
-- (admin/contributor), set on the profiles row right after the invite.

alter table profiles drop constraint if exists profiles_role_check;

alter table profiles
  add constraint profiles_role_check check (role in ('admin', 'contributor', 'viewer'));
