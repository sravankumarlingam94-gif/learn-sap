/* Sign-in gate: the page stays hidden until a signed-in member is confirmed. */
(function () {
  "use strict";

  var SB_URL = "https://kqahazjjjsvpkegeopgu.supabase.co";
  var SB_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtxYWhhempqanN2cGtlZ2VvcGd1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTExOTEwNjAsImV4cCI6MjEwNjc2NzA2MH0.gERW2EK5svGYoIxTLxIhGN2WckB-VRR9eVMu088Wm9A";
  var SESSION_KEY = "apgcoal_auth_v1";   /* shared on purpose: one sign-in works for every app on this site */
  var OK_KEY = "apg_gate_ok_v1";
  var GRACE_MS = 30 * 24 * 3600 * 1000;  /* offline use is allowed for 30 days after the last successful check */

  var root = document.documentElement;
  var sb = null, wall = null, form = null, msgEl = null, subEl = null, emailEl = null, passEl = null, btnEl = null;

  /* hide the whole page from the very first paint */
  var css = document.createElement("style");
  css.id = "apgGateCss";
  css.textContent =
    "html:not(.apg-ok){overflow:hidden!important;background:#05090F!important}" +
    "html:not(.apg-ok) body>*:not(#apgGate){visibility:hidden!important}" +
    "#apgGate{position:fixed;inset:0;z-index:2147483600;display:grid;place-items:center;padding:24px;" +
    "background:#05090F;color:#EDF3FC;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}" +
    "#apgGate form{width:100%;max-width:380px;display:flex;flex-direction:column;gap:12px;padding:30px 26px;" +
    "border:1px solid #1E3252;border-radius:18px;background:linear-gradient(180deg,#122038,#0D1626);box-sizing:border-box}" +
    "#apgGate .t{font-weight:700;font-size:22px;line-height:1.25}" +
    "#apgGate .s{font-size:13px;color:#93A9C8;margin-top:-6px}" +
    "#apgGate input{width:100%;box-sizing:border-box;padding:12px 14px;border-radius:10px;border:1px solid #2A4570;" +
    "background:#080E18;color:#EDF3FC;font-size:16px;font-family:inherit;outline:none}" +
    "#apgGate input:focus{border-color:#16E0C8}" +
    "#apgGate button{padding:12px 14px;border:0;border-radius:10px;background:#16E0C8;color:#03201C;" +
    "font-size:16px;font-weight:700;font-family:inherit;cursor:pointer}" +
    "#apgGate button:disabled{opacity:.6;cursor:default}" +
    "#apgGate .m{min-height:18px;font-size:13px;color:#FFB627;line-height:1.4}" +
    "#apgGate .f{font-size:12px;color:#5F7699;line-height:1.5}" +
    "#apgOut{position:fixed;left:10px;bottom:10px;z-index:2147483000;padding:5px 10px;border:1px solid rgba(128,128,128,.45);" +
    "border-radius:999px;background:rgba(10,15,25,.72);color:#C9D6EA;font:600 11px system-ui,sans-serif;cursor:pointer;opacity:.7}" +
    "#apgOut:hover{opacity:1}" +
    "@media print{#apgOut{display:none!important}}";
  (document.head || root).appendChild(css);

  function lsGet(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { window.localStorage.setItem(k, v); } catch (e) {} }
  function lsDel(k) { try { window.localStorage.removeItem(k); } catch (e) {} }

  function graceOk() {
    try {
      var o = JSON.parse(lsGet(OK_KEY) || "null");
      return !!(o && o.t && (Date.now() - o.t) < GRACE_MS);
    } catch (e) { return false; }
  }

  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  function buildWall() {
    if (wall) return;
    wall = document.createElement("div");
    wall.id = "apgGate";
    form = document.createElement("form");
    form.autocomplete = "on";
    var t = document.createElement("div"); t.className = "t"; t.textContent = (document.title || "Sign in").slice(0, 80);
    var s = document.createElement("div"); s.className = "s"; s.textContent = "Sign in to continue"; subEl = s;
    emailEl = document.createElement("input"); emailEl.type = "email"; emailEl.placeholder = "Email";
    emailEl.autocomplete = "username"; emailEl.required = true;
    passEl = document.createElement("input"); passEl.type = "password"; passEl.placeholder = "Password";
    passEl.autocomplete = "current-password"; passEl.required = true;
    btnEl = document.createElement("button"); btnEl.type = "submit"; btnEl.textContent = "Sign in";
    msgEl = document.createElement("div"); msgEl.className = "m"; msgEl.setAttribute("role", "status");
    var f = document.createElement("div"); f.className = "f";
    f.textContent = "Access is by invitation only. Please contact the administrator if you need an account.";
    form.appendChild(t); form.appendChild(s); form.appendChild(emailEl); form.appendChild(passEl);
    form.appendChild(btnEl); form.appendChild(msgEl); form.appendChild(f);
    wall.appendChild(form);
    document.body.appendChild(wall);
    form.addEventListener("submit", onSubmit);
  }

  function showWall(msg, withForm) {
    buildWall();
    wall.style.display = "grid";
    form.querySelectorAll("input,button").forEach(function (el) { el.style.display = withForm ? "" : "none"; });
    msgEl.textContent = msg || "";
    subEl.textContent = withForm ? "Sign in to continue" : "";
    if (withForm) { try { emailEl.focus(); } catch (e) {} }
  }

  function openApp() {
    root.classList.add("apg-ok");
    if (wall) wall.style.display = "none";
    if (!document.getElementById("apgOut")) {
      var b = document.createElement("button");
      b.id = "apgOut"; b.type = "button"; b.textContent = "Sign out"; b.title = "Sign out of this device";
      b.addEventListener("click", signOut);
      document.body.appendChild(b);
    }
  }

  async function signOut() {
    try { if (sb) await sb.auth.signOut(); } catch (e) {}
    lsDel(OK_KEY); lsDel(SESSION_KEY);
    location.reload();
  }
  window.apgSignOut = signOut;

  /* returns "ok" | "login" | "noaccess" | "neterr" | "expired" */
  async function verify() {
    var s;
    try { s = await sb.auth.getSession(); } catch (e) { return "neterr"; }
    var sess = s && s.data && s.data.session;
    if (!sess) return "login";
    var q;
    try {
      q = await sb.from("members").select("role").eq("user_id", sess.user.id).maybeSingle();
    } catch (e) { return "neterr"; }
    if (q.error) {
      var m = String(q.error.message || q.error);
      if (/jwt|expired|not authenticated|401/i.test(m)) {
        try { await sb.auth.signOut(); } catch (e) {}
        return "expired";
      }
      return "neterr";
    }
    if (!q.data || !q.data.role) {
      try { await sb.auth.signOut(); } catch (e) {}
      lsDel(OK_KEY);
      return "noaccess";
    }
    lsSet(OK_KEY, JSON.stringify({ id: sess.user.id, t: Date.now() }));
    return "ok";
  }

  function apply(state) {
    if (state === "ok") return openApp();
    if (state === "login") return showWall("", true);
    if (state === "expired") return showWall("Your session has expired. Please sign in again.", true);
    if (state === "noaccess") return showWall("This account has not been given access. Please contact the administrator.", true);
    if (graceOk()) return openApp();
    return showWall("Could not reach the sign-in service. Check the internet connection and refresh the page.", false);
  }

  async function onSubmit(ev) {
    ev.preventDefault();
    if (!sb) return;
    btnEl.disabled = true;
    msgEl.textContent = "Signing in...";
    var r;
    try {
      r = await sb.auth.signInWithPassword({ email: emailEl.value.trim(), password: passEl.value });
    } catch (e) {
      btnEl.disabled = false;
      msgEl.textContent = "Could not reach the sign-in service. Check the internet connection.";
      return;
    }
    passEl.value = "";
    btnEl.disabled = false;
    if (r.error) { msgEl.textContent = "Sign-in failed. Please check the email and password."; return; }
    apply(await verify());
  }

  async function start() {
    showWall("Checking access...", false);
    if (!(window.supabase && window.supabase.createClient)) {
      /* fail closed: no sign-in library, no app */
      if (navigator.onLine === false && graceOk()) return openApp();
      return showWall("Sign-in could not be loaded. Check the internet connection and refresh the page.", false);
    }
    sb = window.supabase.createClient(SB_URL, SB_KEY, {
      auth: { persistSession: true, autoRefreshToken: true, storageKey: SESSION_KEY }
    });
    if (navigator.onLine === false && graceOk()) return openApp();
    apply(await verify());
  }

  ready(function () { start().catch(function () { showWall("Could not start. Please refresh the page.", false); }); });
})();
