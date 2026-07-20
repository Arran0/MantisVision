import { NextResponse, type NextRequest } from "next/server";
import { createServerClient, type SetAllCookies } from "@supabase/ssr";

// Coarse gate for everything under /member and /api/member: refreshes the
// Supabase session cookie and rejects unauthenticated requests outright.
// This only checks "is there a session" — the actual admin-role check lives
// in apps/web/src/app/member/(dashboard)/layout.tsx (pages) and
// requireAdmin() (API routes), since role membership needs a DB read that's
// worth keeping out of middleware, which runs on every matched request.
export async function middleware(request: NextRequest) {
  let response = NextResponse.next({ request: { headers: request.headers } });

  const path = request.nextUrl.pathname;
  const isMemberApi = path.startsWith("/api/member");
  // /member/login and /member/password-reset must render for signed-out
  // visitors: login is the entry point, and password-reset finishes an
  // invite whose token arrives in the URL hash (never sent to the server)
  // before any session cookie exists — gating it here would bounce the
  // invitee to login first.
  const isMemberPage =
    path.startsWith("/member") && path !== "/member/login" && path !== "/member/password-reset";

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    // Supabase isn't configured (e.g. env vars not set yet) — fail closed on
    // member routes rather than serving them with no auth check at all.
    if (isMemberApi) {
      return NextResponse.json({ error: "Admin auth is not configured." }, { status: 503 });
    }
    if (isMemberPage) {
      return NextResponse.redirect(new URL("/member/login", request.url));
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

  if (!user && (isMemberApi || isMemberPage)) {
    return isMemberApi
      ? NextResponse.json({ error: "Unauthorized" }, { status: 401 })
      : NextResponse.redirect(new URL("/member/login", request.url));
  }

  return response;
}

export const config = {
  matcher: ["/member/:path*", "/api/member/:path*"],
};
