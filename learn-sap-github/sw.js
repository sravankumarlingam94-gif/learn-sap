/* Learn SAP by running it: offline support */
var CACHE = 'sap-lab-v1';
var CORE = ['./', './index.html', './manifest.json', './icon-192.png', './icon-512.png', './apple-touch-icon.png'];

self.addEventListener('install', function(e){
  e.waitUntil(caches.open(CACHE).then(function(c){ return c.addAll(CORE); }).then(function(){ return self.skipWaiting(); }));
});

self.addEventListener('activate', function(e){
  e.waitUntil(caches.keys().then(function(keys){
    return Promise.all(keys.filter(function(k){ return k !== CACHE; }).map(function(k){ return caches.delete(k); }));
  }).then(function(){ return self.clients.claim(); }));
});

self.addEventListener('fetch', function(e){
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  var cacheable = url.origin === self.location.origin ||
    /(^|\.)cdnjs\.cloudflare\.com$|(^|\.)cdn\.jsdelivr\.net$|(^|\.)fonts\.googleapis\.com$|(^|\.)fonts\.gstatic\.com$/.test(url.hostname);
  if (!cacheable) return;

  /* pages: try the network first so updates show, fall back to the saved copy offline */
  if (req.mode === 'navigate'){
    e.respondWith(fetch(req).then(function(res){
      var copy = res.clone(); caches.open(CACHE).then(function(c){ c.put('./index.html', copy); });
      return res;
    }).catch(function(){ return caches.match('./index.html'); }));
    return;
  }

  /* scripts, fonts, icons: serve the saved copy at once and refresh it in the background */
  e.respondWith(caches.match(req).then(function(hit){
    var net = fetch(req).then(function(res){
      if (res && (res.ok || res.type === 'opaque')){ var copy = res.clone(); caches.open(CACHE).then(function(c){ c.put(req, copy); }); }
      return res;
    }).catch(function(){ return hit; });
    return hit || net;
  }));
});
