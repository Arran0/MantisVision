const CACHE_NAME = "mantis-vision-shell-v3";
const APP_SHELL = ["/manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
      )
  );
  self.clients.claim();
});

// API calls: always network, never cached (predictions must be fresh).
//
// Navigations (HTML documents, including "/"): network-first. A previous
// cache-first strategy here meant a visitor who had ever loaded the app kept
// getting served the exact HTML/JS shell from their first visit, silently,
// forever — no app update (including auth-flow fixes) ever reached them
// without an uninstall/hard-refresh. Network-first fixes that; the cache is
// now only a fallback for offline use.
//
// Everything else (hashed static assets): cache-first, since Next.js
// fingerprints these filenames and they're safe to treat as immutable.
self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(fetch(request));
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(() => caches.match(request).then((cached) => cached || caches.match("/")))
    );
    return;
  }

  event.respondWith(caches.match(request).then((cached) => cached || fetch(request)));
});
