"use client";

import { useEffect } from "react";

// Invite/recovery links are built with redirectTo=/member/password-reset,
// but Supabase only honors that when the URL is on the project's Redirect
// URLs allow-list — otherwise it silently swaps in the Site URL instead,
// which is this app's root, and the token (in the URL hash) lands here with
// nowhere to be picked up. Catch that fallback: if a page loads with an auth
// token or error in the hash, forward it to the page that knows what to do
// with it, hash intact.
export function AuthCallbackRedirect() {
  useEffect(() => {
    if (window.location.pathname === "/member/password-reset") return;

    const hash = window.location.hash.replace(/^#/, "");
    const hashParams = new URLSearchParams(hash);
    const isHashCallback =
      hashParams.has("access_token") || hashParams.has("error_description") || hashParams.has("token_hash");
    if (isHashCallback) {
      window.location.replace(`/member/password-reset#${hash}`);
      return;
    }

    // Same fallback, but for the PKCE flow (?code=...) instead of the
    // implicit flow's hash tokens, in case that's ever enabled project-side.
    const query = new URLSearchParams(window.location.search);
    if (query.has("code") || query.has("error_description")) {
      window.location.replace(`/member/password-reset${window.location.search}`);
    }
  }, []);

  return null;
}
