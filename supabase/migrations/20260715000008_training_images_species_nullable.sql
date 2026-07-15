-- `species` on training_images was `not null default 'Kappaphycus alvarezii'`
-- from back when the schema had exactly one (implicitly "active") species.
-- Species is now a normal per-image classification measurement that only
-- applies when seaweed_presence == "Yes" (see the "species" measurement in
-- measurement_schema) — a background/no-seaweed photo legitimately has no
-- species value, and the admin dataset API inserts NULL for it. A fixed
-- single-species default no longer makes sense either now that the schema
-- supports any number of species.
alter table training_images
  alter column species drop not null,
  alter column species drop default;
