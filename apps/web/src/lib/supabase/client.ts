"use client";

import { createBrowserClient } from "@supabase/ssr";

// Browser-side Supabase client (anon key, respects RLS). Used only by the
// admin login page — nothing else in the app talks to Supabase from the client.
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}
