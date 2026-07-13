import { NextResponse, type NextRequest } from "next/server";
import { createServerClient, type SetAllCookies } from "@supabase/ssr";

// Coarse gate for everything under /admin and /api/admin: refreshes the
// Supabase session cookie and rejects unauthenticated requests outright.
// This only checks "is there a session" — the actual admin-role check lives
// in apps/web/src/app/admin/layout.tsx (pages) and requireAdmin() (API
// routes), since role membership needs a DB read that's worth keeping out
// of middleware, which runs on every matched request.
export async function middleware(request: NextRequest) {
  let response = NextResponse.next({ request: { headers: request.headers } });

  const path = request.nextUrl.pathname;
  const isAdminApi = path.startsWith("/api/admin");
  const isAdminPage = path.startsWith("/admin") && path !== "/admin/login";

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    // Supabase isn't configured (e.g. env vars not set yet) — fail closed on
    // admin routes rather than serving them with no auth check at all.
    if (isAdminApi) {
      return NextResponse.json({ error: "Admin auth is not configured." }, { status: 503 });
    }
    if (isAdminPage) {
      return NextResponse.redirect(new URL("/admin/login", request.url));
    }
    return response;
  }

  const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return request.cookies.getAll();
      },
      // Called once with every cookie that needs writing (there are usually
      // several — access + refresh token, sometimes split into chunks). The
      // old set()-per-cookie API rebuilt `response` on every call, which
      // discarded all but the last cookie and corrupted the session; setAll
      // gets the full batch in one call so nothing is lost.
      setAll: ((cookiesToSet) => {
        cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value));
        response = NextResponse.next({ request: { headers: request.headers } });
        cookiesToSet.forEach(({ name, value, options }) =>
          response.cookies.set(name, value, options)
        );
      }) satisfies SetAllCookies,
    },
  });

  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user && (isAdminApi || isAdminPage)) {
    return isAdminApi
      ? NextResponse.json({ error: "Unauthorized" }, { status: 401 })
      : NextResponse.redirect(new URL("/admin/login", request.url));
  }

  return response;
}

export const config = {
  matcher: ["/admin/:path*", "/api/admin/:path*"],
};
