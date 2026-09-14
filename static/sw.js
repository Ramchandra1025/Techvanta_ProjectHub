const CACHE = "techvanta-shell-v1";
const SHELL = ["/", "/static/style.css", "/static/script.js", "/static/manifest.json", "/static/icons/icon-192.png", "/static/icons/icon-512.png"];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", event => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  // API responses must stay network-backed so project data is never stale.
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/ws/")) return;
  event.respondWith(fetch(req).then(response => {
    const copy = response.clone();
    caches.open(CACHE).then(cache => cache.put(req, copy));
    return response;
  }).catch(() => caches.match(req).then(cached => cached || caches.match("/"))));
});
