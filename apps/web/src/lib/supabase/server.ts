import { cookies } from "next/headers";
import { createServerClient, type SetAllCookies } from "@supabase/ssr";

// Server-side Supabase client bound to the request's cookies (anon key,
// respects RLS). Use in Server Components and /api/member/* route handlers to
// check "who is calling" — middleware.ts already refreshed the session cookie
// before this runs.
export function createClient() {
  const cookieStore = cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll: ((cookiesToSet) => {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            );
          } catch {
            // Called from a Server Component render — middleware.ts owns
            // refreshing the session cookie there, so this is safe to ignore.
          }
        }) satisfies SetAllCookies,
      },
    }
  );
}
