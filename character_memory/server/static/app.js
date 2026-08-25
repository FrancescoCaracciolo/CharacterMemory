/* Character Memory Browser.
 *
 * Talks to three JSON endpoints on the API server:
 *   GET /                              -> { characters: [...] }
 *   GET /api/memories/{character}      -> { memories: [...] }            (sidebar)
 *   GET /api/memories/{c}/{m}?page&size&user&q  -> paged records          (main pane)
 *
 * Each memory type renders through a function in RENDERERS (keyed by memory
 * name); unknown names fall back to a renderer keyed by the page's `kind`
 * (structured | rag | emotion | generic). Add a memory type -> add one entry
 * to RENDERERS; nothing else changes.
 */

const API = "";
const SIZE = 25;
const LIVE_GRAPH_EMBED_PATH = "/gui/embed/knowledge-graph";
const LIVE_GRAPH_EMBED = location.pathname.replace(/\/+$/, "") === LIVE_GRAPH_EMBED_PATH;

// ---------------------------------------------------------------- API key
// Optional server auth (CM_API_KEY / --api-key on the server). The key is
// stored locally and attached to every request as an X-API-Key header; the
// SSE EventSource cannot send headers, so its URL carries the key as an
// `api_key` query parameter instead. Best-effort, like cm_theme: ignored
// when localStorage is unavailable.
const API_KEY_STORAGE = "cm_api_key";
const CALENDAR_WORLD_ROUTINES_KEY = "cm_calendar_show_world_routines";
function storedApiKey() {
  try { return localStorage.getItem(API_KEY_STORAGE) || ""; } catch (error) { return ""; }
}
function embeddedUrlApiKey() {
  // A host may pass the key to the iframe because EventSource cannot attach
  // headers and third-party storage can be partitioned by the browser.  Keep
  // this scoped to the dedicated read-only surface; never persist the query
  // value or propagate it into the normal GUI's share link.
  return LIVE_GRAPH_EMBED ? (new URLSearchParams(location.search).get("api_key") || "") : "";
}
function activeApiKey() { return storedApiKey() || embeddedUrlApiKey(); }
function storedCalendarWorldRoutines() {
  try { return localStorage.getItem(CALENDAR_WORLD_ROUTINES_KEY) !== "false"; } catch (error) { return true; }
}
function authHeaders() {
  const key = activeApiKey();
  return key ? { "X-API-Key": key } : {};
}
function authQuery(url) {
  const key = activeApiKey();
  if (!key) return url;
  return url + (url.includes("?") ? "&" : "?") + "api_key=" + encodeURIComponent(key);
}

const state = {
  characters: [],
  character: null,
  memories: [],            // overview entries
  memory: null,            // selected memory name
  page: 1,
  size: SIZE,
  user: "",
  q: "",
  calendarOffset: 0,
  showWorldRoutines: storedCalendarWorldRoutines(),
  cache: new Map(),        // `${memory}|${user}|${q}|${calendarOffset}` -> Map(page -> data)
  inflight: null,          // AbortController for the active page fetch
  editor: { editable: false, fields: [], description: "" },
  editing: null,
  graph: { data: null, q: "", user: "", full: false, _gv: null, _wired: false }, // KG viz state
  live: {
    events: new Map(), selectedId: null, follow: true, source: null,
    connection: "closed", graph: {}, graphUser: "", openMemory: new Set(), openInitialized: false,
    embedOpenMemory: new Set(), embedOpenInitialized: false,
    _gv: null, _active: false,
  },
};

// ---------------------------------------------------------------- DOM shortcuts
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, children = []) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (typeof v === "boolean") n[k] = v; // disabled / hidden / checked … (truthy string via setAttribute would break these)
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return n;
};
const clear = (n) => { while (n && n.firstChild) n.removeChild(n.firstChild); };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ---------------------------------------------------------------- icons
// Inline Lucide icon (kebab-case name → SVG element). Degrades to an empty
// placeholder when the vendored bundle (static/vendor/lucide.min.js) is
// missing, so the GUI keeps working — just unadorned.
const icon = (name, cls = "") => {
  const classes = ("icon" + (cls ? " " + cls : "")).trim();
  const lib = window.lucide;
  if (lib) {
    const pascal = name.replace(/(^|-)([a-z0-9])/g, (_m, _dash, c) => c.toUpperCase());
    const def = lib[pascal];
    if (def) return lib.createElement(def, { class: classes, "aria-hidden": "true" });
  }
  if (!icon._warned) {
    icon._warned = true;
    console.warn("[Mnemosyne] Lucide is not loaded — icons render as empty placeholders. " +
      "Check that /gui/static/vendor/lucide.min.js is served, then hard-reload (Ctrl+Shift+R).");
  }
  return el("i", { class: classes, "aria-hidden": "true" });
};

// (Re)build an icon+label toggle button. Used instead of textContent swaps so
// the Lucide SVG children survive label changes (freeze/full-graph buttons).
const setButtonContent = (btn, iconName, label) => {
  if (!btn) return;
  clear(btn);
  btn.appendChild(icon(iconName));
  btn.appendChild(el("span", { class: "btn-label" }, label));
};

// ---------------------------------------------------------------- theme
const THEME_KEY = "cm_theme";
function applyTheme(theme, { persist = true } = {}) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  document.documentElement.style.colorScheme = next;
  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.checked = next === "dark";
    toggle.setAttribute("aria-label", `Dark theme ${toggle.checked ? "on" : "off"}`);
  }
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = next === "dark" ? "#19162b" : "#fff8e8";
  if (persist) {
    try { localStorage.setItem(THEME_KEY, next); } catch (error) { /* storage may be disabled */ }
  }
}

function wireThemeToggle() {
  const toggle = $("theme-toggle");
  if (!toggle) return;
  applyTheme(document.documentElement.dataset.theme, { persist: false });
  toggle.addEventListener("change", () => {
    const next = toggle.checked ? "dark" : "light";
    const change = () => applyTheme(next);
    const canAnimate = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (document.startViewTransition && canAnimate) document.startViewTransition(change);
    else change();
  });
  window.addEventListener("storage", (event) => {
    if (event.key === THEME_KEY && (event.newValue === "light" || event.newValue === "dark")) {
      applyTheme(event.newValue, { persist: false });
    }
  });
}

// ---------------------------------------------------------------- time helpers
function relTime(epoch) {
  if (!epoch) return "—";
  const s = Math.max(0, Date.now() / 1000 - epoch);
  if (s < 60) return "just now";
  const u = [[604800, "w"], [86400, "d"], [3600, "h"], [60, "m"]];
  for (const [sec, sym] of u) {
    if (s >= sec) return Math.floor(s / sec) + sym;
  }
  return Math.floor(s) + "s";
}
function absTime(epoch) {
  if (!epoch) return "";
  return new Date(epoch * 1000).toLocaleString();
}

// ---------------------------------------------------------------- small UI bits
function chip(text, cls = "") { return el("span", { class: "chip " + cls }, text); }
function kwChip(text) { return el("span", { class: "kw" }, text); }

/** 0..1 bar. */
function fieldBar(label, value, { fmt, klass = "" } = {}) {
  const v = clamp(Number(value) || 0, 0, 1);
  const wrap = el("div", { class: "field" }, [
    el("span", { class: "field-label" }, label),
    el("div", { class: "track" }, [el("div", { class: "fill " + klass, style: `width:${(v * 100).toFixed(1)}%` })]),
    el("span", { class: "field-val" }, fmt ? fmt(value) : v.toFixed(2)),
  ]);
  return wrap;
}

/** -1..1 bar centred at zero. */
function signedBar(label, value, { fmt } = {}) {
  let v = clamp(Number(value) || 0, -1, 1);
  const pos = v >= 0;
  const width = Math.abs(v) * 50; // half-track percent
  const color = pos ? "var(--good)" : "var(--bad)";
  const fill = el("div", {
    class: "signed-fill",
    style: pos
      ? `left:50%; width:${width}%; background:${color}`
      : `right:50%; width:${width}%; background:${color}`,
  });
  return el("div", { class: "field" }, [
    el("span", { class: "field-label" }, label),
    el("div", { class: "signed-track" }, [fill]),
    el("span", { class: "field-val" }, (fmt ? fmt(value) : (v >= 0 ? "+" : "") + v.toFixed(2))),
  ]);
}

/**
 * Activation score-breakdown: how much each factor contributed to a node's
 * final activation. Factor values are signed and unbounded (BLL is often
 * negative, seeds can be > 1), so each bar is drawn centred at zero and
 * scaled to the largest absolute contribution in the set — the relative
 * widths are what's meaningful.
 *
 * `comp` is the `score_breakdown` object from /api/graph:
 *   { bll, spread, seed, base, spread_w, emotion_mult, score }
 *   - base      = base_weight   * bll        (weighted BLL contribution)
 *   - spread_w  = spread_weight * spread     (weighted spreading contribution)
 *   - seed      = the RRF/SelfNode seed      ("did this node match the query?")
 *   - emotion_mult = multiplicative mood-alignment boost actually applied
 *   - score     = (base + spread_w) * emotion_mult   (final activation)
 */
function scoreBreakdownBlock(comp) {
  if (!comp || typeof comp !== "object") return null;
  const factors = [
    ["base", "BLL (base-weighted)"],
    ["spread_w", "spreading"],
    ["seed", "query seed"],
  ];
  const rows = [];
  // Scale every bar to the largest |weighted contribution| so the relative
  // share of each factor is legible at a glance.
  const scale = Math.max(
    1e-6,
    ...factors.map(([k]) => Math.abs(Number(comp[k]) || 0)),
    Math.abs(Number(comp.score) || 0),
  );
  for (const [key, label] of factors) {
    const v = Number(comp[key]) || 0;
    rows.push(breakdownRow(label, v, scale));
  }
  const mult = Number(comp.emotion_mult) || 1.0;
  // Show the emotion multiplier only when it actually changed the score
  // (i.e. this node has an emotional_shift aligned with the current mood).
  if (Math.abs(mult - 1.0) > 1e-3) {
    rows.push(el("div", { class: "field bd-mult" }, [
      el("span", { class: "field-label" }, "mood boost"),
      el("span", { class: "field-val bd-mult-val" }, "×" + mult.toFixed(2)),
    ]));
  }
  rows.push(breakdownRow("final score", Number(comp.score) || 0, scale, { strong: true }));
  return el("div", { class: "bd-block" }, [
    el("div", { class: "bd-title" }, "score breakdown"),
    ...rows,
  ]);
}

/** One centred-at-zero contribution bar, scaled by `scale` (max |value|). */
function breakdownRow(label, value, scale, { strong = false } = {}) {
  const v = Number(value) || 0;
  const pos = v >= 0;
  const half = Math.min(50, (Math.abs(v) / scale) * 50);
  const color = strong ? "var(--accent)" : pos ? "var(--good)" : "var(--bad)";
  const fill = el("div", {
    class: "signed-fill",
    style: pos
      ? `left:50%; width:${half}%; background:${color}`
      : `right:50%; width:${half}%; background:${color}`,
  });
  return el("div", { class: "field" + (strong ? " bd-total" : "") }, [
    el("span", { class: "field-label" }, label),
    el("div", { class: "signed-track" }, [fill]),
    el("span", { class: "field-val" }, (pos ? "+" : "") + v.toFixed(2)),
  ]);
}

function metaRow(rec) {
  const m = rec.meta || {};
  const parts = [];
  if (rec.user_id) parts.push(["user", rec.user_id]);
  if (rec.id != null) parts.push(["id", String(rec.id)]);
  if (m.created_at) parts.push(["created", relTime(m.created_at)]);
  if (m.last_recalled) parts.push(["recalled", relTime(m.last_recalled)]);
  if (m.recall_count != null) parts.push(["×", String(m.recall_count)]);
  return el("div", { class: "card-meta" },
    parts.map(([k, v]) => el("span", {}, [el("b", { style: "color:var(--text-faint)" }, k + ":"), " " + v])));
}

function scorePill(rec) {
  if (rec.score == null) return null;
  const label = state.q ? "relevance" : "effective";
  return el("span", { class: "score-pill", title: label }, `${label} ${Number(rec.score).toFixed(2)}`);
}

// ---------------------------------------------------------------- minimal markdown
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function inlineMd(s) {
  // Escape the text, but keep inline-code spans verbatim: their content must
  // not be mangled by `**`/`*`/`[..](..)` substitution. Escape happens once
  // here — callers feed the RAW text (escaping twice turns `'` into `&#39;`
  // into the literal text `&#39;`).
  const parts = String(s).split(/(`[^`]+`)/);
  return parts.map((part) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      const inner = esc(part.slice(1, -1));
      return `<code>${inner}</code>`;
    }
    return esc(part)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  }).join("");
}
function markdown(text) {
  const lines = String(text).split("\n");
  let html = "";
  let inUl = false, inOl = false, inCode = false, codeBuf = [];
  const closeLists = () => { if (inUl) { html += "</ul>"; inUl = false; } if (inOl) { html += "</ol>"; inOl = false; } };
  for (const raw of lines) {
    const line = raw;
    if (line.trim().startsWith("```")) {
      if (inCode) { html += `<pre><code>${esc(codeBuf.join("\n"))}</code></pre>`; codeBuf = []; inCode = false; }
      else { closeLists(); inCode = true; }
      continue;
    }
    if (inCode) { codeBuf.push(line); continue; }
    if (!line.trim()) { closeLists(); continue; }
    let m;
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { closeLists(); html += `<h${m[1].length}>${inlineMd(m[2])}</h${m[1].length}>`; continue; }
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) { closeLists(); html += "<hr>"; continue; }
    if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { if (!inUl) { closeLists(); html += "<ul>"; inUl = true; } html += `<li>${inlineMd(m[1])}</li>`; continue; }
    if ((m = line.match(/^\s*\d+\.\s+(.*)$/))) { if (!inOl) { closeLists(); html += "<ol>"; inOl = true; } html += `<li>${inlineMd(m[1])}</li>`; continue; }
    closeLists();
    html += `<p>${inlineMd(line)}</p>`;
  }
  if (inCode) html += `<pre><code>${esc(codeBuf.join("\n"))}</code></pre>`;
  closeLists();
  return html;
}

// ---------------------------------------------------------------- renderers (per memory type)
function wrapCard(children, rec) {
  const head = el("div", { class: "card-tags" }, []);
  const pill = scorePill(rec);
  const actions = state.editor && state.editor.editable && !(rec.fields && rec.fields.virtual)
    ? el("div", { class: "card-actions" }, [
        el("button", { class: "card-action", type: "button", title: "Edit memory", onclick: (event) => {
          event.stopPropagation(); openMemoryEditor(rec);
        } }, "Edit"),
        el("button", { class: "card-action danger", type: "button", title: "Delete memory", onclick: (event) => {
          event.stopPropagation(); removeMemory(rec);
        } }, "Delete"),
      ])
    : null;
  const card = el("div", { class: "card" }, [
    el("div", { class: "card-head" }, [head, el("div", { class: "card-head-end" }, [pill, actions])]),
    el("div", { class: "card-body" }, children),
    metaRow(rec),
  ]);
  card._head = head;
  return card;
}
function addTag(card, node) { if (node) card._head.appendChild(node); }

function renderUserFact(rec) {
  const f = rec.fields || {};
  const card = wrapCard([el("div", { class: "text" }, f.content || rec.text)], rec);
  addTag(card, chip(f.type || "general", "kind-chip"));
  card.appendChild(fieldBar("confidence", f.confidence, { klass: "good" }));
  card.appendChild(fieldBar("importance", f.importance));
  if (rec.meta && rec.meta.effective != null)
    card.appendChild(fieldBar("effective", rec.meta.effective, { klass: "warn" }));
  return card;
}

function renderDirective(rec) {
  const f = rec.fields || {};
  let kws = [];
  try { kws = JSON.parse(f.retrieval_keywords || "[]"); } catch (e) { kws = []; }
  const body = el("div", {}, [el("div", { class: "text" }, f.content || rec.text)]);
  if (kws.length) {
    body.appendChild(el("div", { class: "kw-chips" }, kws.map(kwChip)));
  }
  const card = wrapCard([body], rec);
  addTag(card, chip("directive"));
  card.appendChild(fieldBar("importance", f.importance));
  return card;
}

function renderEpisode(rec) {
  const f = rec.fields || {};
  const card = wrapCard([el("div", { class: "text" }, f.summary || rec.text)], rec);
  const shift = f.emotional_shift && typeof f.emotional_shift === "object" ? f.emotional_shift : {};
  const axes = Object.entries(shift).filter(([, value]) => Number(value) > 0);
  for (const [axis, value] of axes)
    addTag(card, chip(`${axis} ${Number(value).toFixed(2)}`, "kind-chip"));
  if (rec.meta && rec.meta.raw_emotional_impact != null)
    card.appendChild(fieldBar("emotional impact", rec.meta.raw_emotional_impact, { klass: "warn" }));
  if (rec.meta && rec.meta.emotion_similarity != null)
    card.appendChild(fieldBar("mood similarity", rec.meta.emotion_similarity, { klass: "good" }));
  card.appendChild(fieldBar("importance", f.importance));
  if (rec.meta && rec.meta.effective != null)
    card.appendChild(fieldBar("effective", rec.meta.effective, { klass: "warn" }));
  return card;
}

function renderHeartbeat(rec) {
  const f = rec.fields || {};
  const card = wrapCard([el("div", { class: "text" }, f.summary || rec.text)], rec);
  addTag(card, chip(f.kind || "discovery", "kind-chip"));
  card.appendChild(fieldBar("importance", f.importance));
  return card;
}

function renderUserSummary(rec) {
  const f = rec.fields || {};
  let aliases = [];
  try { aliases = JSON.parse(f.aliases || "[]"); } catch (e) { aliases = []; }
  const body = el("div", {}, [el("div", { class: "text" }, f.summary || rec.text)]);
  if (aliases.length) {
    body.appendChild(el("div", { class: "kw-chips" }, aliases.map(kwChip)));
  }
  const card = wrapCard([body], rec);
  addTag(card, chip(f.name || "user", "kind-chip"));
  if (f.importance != null) card.appendChild(fieldBar("importance", f.importance));
  return card;
}

function renderCalendar(rec) {
  const f = rec.fields || {};
  const start = f.start ? new Date(f.start) : null;
  const end = f.end ? new Date(f.end) : null;
  const time = start
    ? `${start.toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}${end ? ` – ${end.toLocaleTimeString([], { timeStyle: "short" })}` : ""}`
    : "unscheduled";
  const body = el("div", { class: "calendar-card-body" }, [
    el("div", { class: "calendar-time" }, time),
    el("div", { class: "text" }, f.title || rec.text || "(untitled)"),
    f.description ? el("div", { class: "calendar-description" }, f.description) : null,
    f.location ? el("div", { class: "calendar-location" }, [icon("map-pin"), f.location]) : null,
  ]);
  const card = wrapCard([body], rec);
  addTag(card, chip(f.virtual ? "world routine" : (f.kind || "event"), "kind-chip"));
  addTag(card, chip(f.owner_id === "_self" ? "character" : (f.owner_id || "user")));
  if (f.status === "cancelled") addTag(card, chip("cancelled", "kind-chip"));
  if (Array.isArray(f.attendees) && f.attendees.length) addTag(card, chip(`${f.attendees.length} attendee${f.attendees.length === 1 ? "" : "s"}`));
  return card;
}

function renderRag(rec) {
  const src = (rec.meta && rec.meta.source) || (rec.fields && rec.fields.source) || "";
  const card = wrapCard([el("div", { class: "md", html: markdown(rec.text || "") })], rec);
  if (src) addTag(card, chip(src.split("/").pop(), ""));
  return card;
}

function renderDialogue(rec) {
  const src = (rec.meta && rec.meta.source) || (rec.fields && rec.fields.source) || "";
  const card = wrapCard([el("div", { class: "md", html: markdown(rec.text || "") })], rec);
  addTag(card, chip("example", "kind-chip"));
  if (src) addTag(card, chip(src.split("/").pop(), ""));
  return card;
}

function renderEmotion(rec) {
  const state = (rec.fields && rec.fields.state) || {};
  const dims = Object.entries(state);
  const comment = (rec.fields && rec.fields.comment) || (rec.meta && rec.meta.comment) || "";
  const body = el("div", {}, dims.length
    ? dims.map(([k, v]) => signedBar(k, v))
    : [el("div", { class: "text" }, "no dimensions")]);
  const card = wrapCard([body], rec);
  addTag(card, chip("user", "kind-chip"));
  if (comment) addTag(card, chip(comment, ""));
  return card;
}

function renderGenericStructured(rec) {
  const f = rec.fields || {};
  const body = el("div", { class: "text" }, rec.text || "");
  // surface any remaining non-common columns.
  const skip = new Set(["id", "user_id", "importance", "created_at", "last_recalled", "recall_count"]);
  const extra = Object.entries(f).filter(([k]) => !skip.has(k));
  if (extra.length) {
    body.appendChild(el("div", { class: "kw-chips" },
      extra.map(([k, v]) => chip(`${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`))));
  }
  const card = wrapCard([body], rec);
  if (f.importance != null) card.appendChild(fieldBar("importance", f.importance));
  return card;
}

function renderGeneric(rec) {
  const card = wrapCard([el("div", { class: "text" }, rec.text || "(empty)")], rec);
  const f = rec.fields || {};
  const meta = Object.entries(f).filter(([k]) => !["text", "id", "user_id"].includes(k));
  if (meta.length) addTag(card, chip(`${meta.length} fields`));
  return card;
}

const RENDERERS = {
  user_facts: renderUserFact,
  user_directives: renderDirective,
  episodic: renderEpisode,
  heartbeat: renderHeartbeat,
  user_summary: renderUserSummary,
  calendar: renderCalendar,
  character_info: renderRag,
  dialogue_style: renderDialogue,
  emotion: renderEmotion,
};
const FALLBACK = {
  structured: renderGenericStructured,
  rag: renderRag,
  emotion: renderEmotion,
  generic: renderGeneric,
};
function pickRenderer(memName, kind) {
  return RENDERERS[memName] || FALLBACK[kind] || renderGeneric;
}

// ---------------------------------------------------------------- data fetching
async function getJSON(url, signal) {
  const r = await fetch(url, { headers: { Accept: "application/json", ...authHeaders() }, signal });
  if (r.status === 401) promptApiKey();
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

async function sendJSON(url, { method = "POST", body } = {}) {
  const response = await fetch(url, {
    method,
    headers: { Accept: "application/json", "Content-Type": "application/json", ...authHeaders() },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401) promptApiKey();
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

async function loadCharacters() {
  const data = await getJSON(`${API}/`);
  state.characters = data.characters || [];
  const sel = $("character");
  clear(sel);
  for (const c of state.characters) sel.appendChild(el("option", { value: c }, c));
  if (!state.characters.length) {
    const message = "No characters found. Start the server from the project root (so ./assets is visible), or set CM_ASSETS_DIR.";
    if (LIVE_GRAPH_EMBED) {
      liveSetConnection("error", "No character");
      if ($("live-embed-identity")) $("live-embed-identity").textContent = message;
    } else showError(message);
    return false;
  }
  state.character = state.character && state.characters.includes(state.character)
    ? state.character : state.characters[0];
  const requestedCharacter = new URLSearchParams(location.search).get("character");
  if (requestedCharacter && state.characters.includes(requestedCharacter)) state.character = requestedCharacter;
  sel.value = state.character;
  return true;
}

async function loadOverview() {
  // Never query endpoints with a null/empty character — that produces a
  // confusing "Unknown character 'null'" error from the server.
  if (!state.character) {
    renderSidebar();
    return;
  }
  const spin = $("refresh"); spin.classList.add("spinning");
  try {
    const data = await getJSON(`${API}/api/memories/${encodeURIComponent(state.character)}`);
    state.memories = data.memories || [];
    state.cache.clear();
    renderSidebar();
    // keep selection valid
    if (!state.memories.some((m) => m.name === state.memory)) {
      state.memory = (state.memories[0] && state.memories[0].name) || null;
    }
    await loadPage();
  } catch (e) {
    showError(e.message);
  } finally {
    spin.classList.remove("spinning");
  }
}

function cacheKey() { return `${state.memory}|${state.user || ""}|${state.q}|${state.calendarOffset}`; }

function calendarRange() {
  // Keep the navigation anchored to calendar months instead of a rolling
  // agenda window.  The API still receives the complete visible grid range,
  // including the leading/trailing days needed to draw a rectangular month.
  const monthStart = new Date();
  monthStart.setHours(0, 0, 0, 0);
  monthStart.setDate(1);
  monthStart.setMonth(monthStart.getMonth() + state.calendarOffset);

  const start = new Date(monthStart);
  start.setDate(start.getDate() - start.getDay()); // Sunday-first grid.

  const monthEnd = new Date(monthStart);
  monthEnd.setMonth(monthEnd.getMonth() + 1);
  const before = new Date(monthEnd);
  before.setDate(before.getDate() + (7 - before.getDay()) % 7);

  return {
    start,
    before,
    monthStart,
    monthEnd,
    label: monthStart.toLocaleDateString([], { month: "long", year: "numeric" }),
  };
}

const CALENDAR_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function calendarEditorFields(timezone = "UTC") {
  return [
    { name: "user_id", label: "Owner ID", type: "text", required: false, default: state.user || "_self", readonly_on_edit: true },
    { name: "title", label: "Title", type: "textarea", required: true },
    { name: "description", label: "Description", type: "textarea" },
    { name: "location", label: "Location", type: "text" },
    { name: "timezone", label: "Time zone", type: "timezone", default: timezone },
    { name: "kind", label: "Schedule", type: "select", options: ["event", "routine"], default: "event" },
    { name: "start_at", label: "Starts", type: "datetime-local", required: true, show_when: { field: "kind", equals: "event" } },
    { name: "end_at", label: "Ends", type: "datetime-local", required: true, show_when: { field: "kind", equals: "event" } },
    { name: "weekdays", label: "Repeats on", type: "weekdays", show_when: { field: "kind", equals: "routine" } },
    { name: "start_local", label: "Starts at", type: "time", required: true, show_when: { field: "kind", equals: "routine" } },
    { name: "duration_minutes", label: "Duration (minutes)", type: "number", required: true, min: 1, step: 5, show_when: { field: "kind", equals: "routine" } },
    { name: "attendees", label: "Attendees", type: "tags" },
  ];
}

function setWorldRoutinesVisible(visible) {
  state.showWorldRoutines = visible;
  try { localStorage.setItem(CALENDAR_WORLD_ROUTINES_KEY, String(visible)); } catch (error) { /* storage may be disabled */ }
}

async function loadPage() {
  if (!state.memory) { renderEmpty("Select a memory", ""); return; }
  $("status").classList.add("hidden");
  const key = cacheKey();
  let pages = state.cache.get(key);
  if (!pages) { pages = new Map(); state.cache.set(key, pages); }

  if (pages.has(state.page)) {
    renderPage(pages.get(state.page));
    return;
  }

  // cancel any in-flight fetch (debounced search / rapid paging).
  if (state.inflight) state.inflight.abort();
  const ac = new AbortController();
  state.inflight = ac;
  renderSkeletons();

  const params = new URLSearchParams({ page: state.page, size: state.size });
  if (state.user) params.set("user", state.user);
  if (state.q) params.set("q", state.q);
  let url;
  if (state.memory === "calendar") {
    const range = calendarRange();
    const calendarParams = new URLSearchParams({
      start: range.start.toISOString(), before: range.before.toISOString(), limit: "200",
    });
    if (state.user) calendarParams.append("owner", state.user);
    if (state.q) calendarParams.set("q", state.q);
    url = `${API}/api/calendar/${encodeURIComponent(state.character)}/events?${calendarParams}`;
  } else {
    url = `${API}/api/memories/${encodeURIComponent(state.character)}/${encodeURIComponent(state.memory)}?${params}`;
  }

  try {
    const data = await getJSON(url, ac.signal);
    const normalized = state.memory === "calendar"
      ? {
          memory: "calendar", kind: "calendar", search: !!state.q, q: state.q,
          page: 1, pages: 1, total: (data.events || []).length, size: SIZE,
          users: (state.memories.find((m) => m.name === "calendar") || {}).users || [],
          editor: { editable: true, fields: calendarEditorFields(data.timezone || "UTC"), description: "Create one-off events with the date picker, or weekly routines with the day selector. World routines are read-only live projections." },
          records: (data.events || []).map((event) => ({
            id: event.virtual ? event.id : (event.metadata && event.metadata.event_id) || event.series_id || event.id,
            user_id: event.owner_id,
            text: event.title,
            // The agenda API exposes both epoch fields and display-ready ISO
            // fields. Keep the editor's text inputs in ISO form so a round
            // trip through the generic memory editor remains valid.
            fields: { ...event, start_at: event.start || "", end_at: event.end || "" },
            meta: { virtual: event.virtual },
          })),
        }
      : data;
    pages.set(state.page, normalized);
    if (state.inflight === ac) state.inflight = null;
    renderPage(normalized);
  } catch (e) {
    if (e.name === "AbortError") return;
    showError(e.message);
  } finally {
    if (state.inflight === ac) state.inflight = null;
  }
}

// ---------------------------------------------------------------- rendering
function renderSidebar() {
  const nav = $("memories");
  clear(nav);
  const filter = ($("memory-filter").value || "").toLowerCase();
  const items = state.memories.filter((m) => !filter || m.title.toLowerCase().includes(filter) || m.name.includes(filter));
  if ($("memory-total")) $("memory-total").textContent = `${state.memories.length} active`;
  if (!items.length) { nav.appendChild(el("div", { class: "empty" }, "No memories.")); return; }
  for (const m of items) {
    // Show how many distinct users a per-user memory knows about. More than
    // one is the signature of a group chat's extracted memories.
    const userCount = Array.isArray(m.users) ? m.users.length : 0;
    const userChip = userCount > 1 ? chip(`${userCount} users`, "") : null;
    const btn = el("button", {
      class: "mem-item" + (m.name === state.memory ? " active" : "") + (m.enabled ? "" : " disabled"),
      onclick: () => selectMemory(m.name),
    }, [
      el("div", { class: "mem-row" }, [
        el("span", { class: "mem-title", title: m.name }, m.title),
        el("span", { class: "mem-count" }, String(m.count)),
      ]),
      el("div", { class: "mem-sub" }, [chip(m.kind), userChip, m.enabled ? null : chip("off")]),
    ]);
    nav.appendChild(btn);
  }
}

function renderHeader() {
  const m = state.memories.find((x) => x.name === state.memory) || {};
  $("mem-title").textContent = m.title || state.memory || "—";
  const kind = $("mem-kind"); clear(kind); kind.appendChild(document.createTextNode(m.kind || ""));
  $("mem-count").textContent = m.count != null ? `${m.count} records` : "";
  $("mem-disabled").classList.toggle("hidden", m.enabled !== false);
  $("add-memory").classList.toggle("hidden", !(state.editor && state.editor.editable));
}

function renderUserFilter(users) {
  const sel = $("user-filter");
  const prev = state.user;
  clear(sel);
  sel.appendChild(el("option", { value: "" }, "All users"));
  for (const u of users || []) sel.appendChild(el("option", { value: u }, u));
  sel.value = prev && (users || []).includes(prev) ? prev : "";
  state.user = sel.value;
  // Keep the graph view's user picker in sync with the main one.
  if (state.graph._wired) syncGraphUserPicker(users);
}

function renderExtra(extra) {
  const box = $("extra"); clear(box);
  if (!extra || !extra.baseline) return;
  const b = extra.baseline;
  const dims = Object.entries(b);
  if (!dims.length) return;
  const inner = dims.map(([k, v]) => fieldBar(k, v));
  box.appendChild(el("div", { class: "banner" }, [
    el("div", { class: "banner-title" }, "Baseline (resting state)"),
    ...inner,
  ]));
  const current = Object.entries(extra.current_mood || {});
  if (current.length) box.appendChild(el("div", { class: "banner" }, [
    el("div", { class: "banner-title" }, "Current mood"),
    ...current.map(([k, v]) => fieldBar(k, v, { klass: "good" })),
  ]));
}

function renderPage(data) {
  state.editor = data.editor || { editable: false, fields: [], description: "" };
  renderHeader();
  renderUserFilter(data.users);
  renderExtra(data.extra);
  const searchBox = $("q").parentElement;
  searchBox.classList.toggle("searching", !!data.search);
  $("q-clear").classList.toggle("hidden", !data.q);

  // The knowledge graph renders its own force-directed visualization
  // instead of the card list. Toggle the two views.
  const isGraph = data.memory === "knowledge_graph" || data.kind === "graph";
  const isCalendar = data.memory === "calendar" || data.kind === "calendar";
  $("graph-view").classList.toggle("hidden", !isGraph);
  $("records").classList.toggle("hidden", isGraph);
  $("paginator").classList.toggle("hidden", isGraph || isCalendar);
  if (isGraph) {
    renderGraphView();
    return;
  }

  if (isCalendar) {
    renderCalendarAgenda(data);
    return;
  }

  const list = $("records"); clear(list);
  const records = data.records || [];
  if (!records.length) {
    renderEmpty(data.search ? "No matches" : "Empty memory", data.search ? `Nothing matched “${data.q}”.` : "");
  } else {
    const render = pickRenderer(data.memory, data.kind);
    for (const rec of records) list.appendChild(render(rec));
  }
  renderPaginator(data);
}

function renderCalendarAgenda(data) {
  const list = $("records"); clear(list);
  const range = calendarRange();
  const worldRoutinesVisible = state.showWorldRoutines;
  const toolbar = el("div", { class: "calendar-toolbar" }, [
    el("div", { class: "calendar-nav" }, [
      el("button", { class: "btn ghost tight", type: "button", "aria-label": "Previous month", onclick: () => { state.calendarOffset -= 1; state.cache.clear(); loadPage(); } }, [icon("chevron-left"), "Previous"]),
      el("button", { class: "btn ghost tight", type: "button", onclick: () => { state.calendarOffset = 0; state.cache.clear(); loadPage(); } }, "Today"),
      el("button", { class: "btn ghost tight", type: "button", "aria-label": "Next month", onclick: () => { state.calendarOffset += 1; state.cache.clear(); loadPage(); } }, ["Next", icon("chevron-right")]),
    ]),
    el("div", { class: "calendar-range" }, [
      el("strong", { class: "calendar-month-title" }, range.label),
      el("span", { class: "calendar-range-detail" }, `${range.start.toLocaleDateString([], { dateStyle: "medium" })} – ${new Date(range.before.getTime() - 1).toLocaleDateString([], { dateStyle: "medium" })}`),
    ]),
    el("button", {
      class: "btn ghost tight calendar-filter-toggle", type: "button", role: "switch",
      "aria-checked": worldRoutinesVisible, title: "Show or hide read-only routines projected from the world",
      onclick: () => { setWorldRoutinesVisible(!state.showWorldRoutines); renderCalendarAgenda(data); },
    }, [icon(worldRoutinesVisible ? "eye" : "eye-off"), `World routines: ${worldRoutinesVisible ? "on" : "off"}`]),
  ]);
  list.appendChild(toolbar);

  const allRecords = data.records || [];
  const records = worldRoutinesVisible ? allRecords : allRecords.filter((rec) => !(rec.fields || {}).virtual);

  // Index each occurrence by its local day.  Multi-day events are shown in
  // every covered cell, which makes the month view useful for travel, leave,
  // and other events that cross midnight.  World routines arrive as concrete
  // occurrences from the API and therefore need no special rendering path.
  const eventsByDay = new Map();
  for (const rec of records) {
    const fields = rec.fields || {};
    const start = fields.start || fields.start_at;
    if (!start) continue;
    const startDate = new Date(start);
    if (Number.isNaN(startDate.getTime())) continue;
    const endDate = fields.end || fields.end_at ? new Date(fields.end || fields.end_at) : startDate;
    const lastDate = new Date(Number.isNaN(endDate.getTime()) ? startDate.getTime() : endDate.getTime());
    // An event ending exactly at midnight belongs to the preceding day.
    if (lastDate.getTime() > startDate.getTime()) lastDate.setTime(lastDate.getTime() - 1);
    const day = new Date(startDate);
    day.setHours(0, 0, 0, 0);
    lastDate.setHours(0, 0, 0, 0);
    if (day < range.start) day.setTime(range.start.getTime());
    while (day <= lastDate && day < range.before) {
      if (day >= range.start) {
        const key = calendarDateKey(day);
        if (!eventsByDay.has(key)) eventsByDay.set(key, []);
        eventsByDay.get(key).push(rec);
      }
      day.setDate(day.getDate() + 1);
    }
  }

  const weekdays = el("div", { class: "calendar-weekdays", role: "row" },
    ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((label) =>
      el("div", { class: "calendar-weekday", role: "columnheader" }, label)));
  const grid = el("div", { class: "calendar-grid", role: "grid", "aria-label": range.label });
  const todayKey = calendarDateKey(new Date());
  const monthKey = `${range.monthStart.getFullYear()}-${range.monthStart.getMonth()}`;
  const cursor = new Date(range.start);
  while (cursor < range.before) {
    const key = calendarDateKey(cursor);
    const dayMonthKey = `${cursor.getFullYear()}-${cursor.getMonth()}`;
    const outside = dayMonthKey !== monthKey;
    const today = key === todayKey;
    const events = (eventsByDay.get(key) || []).slice().sort(calendarEventSort);
    const cell = el("div", {
      class: "calendar-day" + (outside ? " is-outside" : "") + (today ? " is-today" : ""),
      role: "gridcell",
      "aria-label": cursor.toLocaleDateString([], { dateStyle: "full" }),
    }, [
      el("div", { class: "calendar-day-head" }, [
        el("span", { class: "calendar-day-number" }, String(cursor.getDate())),
        today ? el("span", { class: "calendar-day-today" }, "Today") : null,
      ]),
      el("div", { class: "calendar-events" }, events.map(renderCalendarEvent)),
    ]);
    grid.appendChild(cell);
    cursor.setDate(cursor.getDate() + 1);
  }

  const shell = el("div", { class: "calendar-shell" }, [weekdays, grid]);
  list.appendChild(shell);
  if (!records.length) {
    list.appendChild(el("div", { class: "calendar-empty-hint" }, [
      icon("calendar-days"), data.search
        ? `No events match “${data.q}” in this month.`
        : allRecords.length && !worldRoutinesVisible
          ? "Only world routines are scheduled this month. Turn their visibility back on to see them."
          : "No events in this month. Add one with the button above.",
    ]));
  }
}

function calendarDateKey(date) {
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function calendarEventSort(a, b) {
  const aTime = new Date((a.fields || {}).start || (a.fields || {}).start_at || 0).getTime();
  const bTime = new Date((b.fields || {}).start || (b.fields || {}).start_at || 0).getTime();
  return (Number.isNaN(aTime) ? Infinity : aTime) - (Number.isNaN(bTime) ? Infinity : bTime);
}

function renderCalendarEvent(rec) {
  const fields = rec.fields || {};
  const start = fields.start || fields.start_at ? new Date(fields.start || fields.start_at) : null;
  const validStart = start && !Number.isNaN(start.getTime());
  const time = validStart ? start.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
  const title = fields.title || rec.text || "(untitled)";
  const owner = fields.owner_id === "_self" ? "character" : (fields.owner_id || "user");
  const virtual = !!fields.virtual;
  const type = fields.kind === "routine" ? "routine" : "event";
  const classes = ["calendar-event", virtual ? "virtual" : "", type, fields.status === "cancelled" ? "cancelled" : ""]
    .filter(Boolean).join(" ");
  const description = [fields.description, fields.location].filter(Boolean).join(" · ");
  const node = el(virtual ? "div" : "button", {
    class: classes,
    ...(virtual ? {} : {
      type: "button",
      onclick: () => openMemoryEditor(rec),
    }),
    title: `${time ? `${time} ` : ""}${title}${description ? ` — ${description}` : ""}`,
    "aria-label": `${time ? `${time} ` : ""}${title} (${owner})`,
  }, [
    time ? el("span", { class: "calendar-event-time" }, time) : null,
    el("span", { class: "calendar-event-title" }, title),
  ]);
  if (!virtual) node.dataset.owner = owner;
  return node;
}

function renderSkeletons() {
  renderHeader();
  const list = $("records"); clear(list);
  for (let i = 0; i < 5; i++) {
    list.appendChild(el("div", { class: "skeleton-card" }, [
      el("div", { class: "sk-line med" }),
      el("div", { class: "sk-line" }),
      el("div", { class: "sk-line short" }),
    ]));
  }
  clear($("paginator"));
}

function renderEmpty(title, sub) {
  clear($("records"));
  $("records").appendChild(el("div", { class: "empty" }, [
    el("div", { class: "big" }, icon("inbox")),
    el("div", {}, title),
    sub ? el("div", { class: "text" }, sub) : null,
  ]));
  clear($("paginator"));
}

function renderPaginator(data) {
  const box = $("paginator"); clear(box);
  const { page, pages, total, size, search } = data;
  if (!total) return;
  box.appendChild(el("span", { class: "pg-info" },
    `${(page - 1) * size + 1}–${Math.min(page * size, total)} of ${total}${search ? " matches" : ""}`));

  const btn = (content, target, opts = {}) => el("button", {
    class: "pg-btn" + (opts.active ? " active" : ""),
    disabled: opts.disabled || false,
    "aria-label": opts.label,
    onclick: () => gotoPage(target),
  }, content);

  box.appendChild(btn(icon("chevron-left"), page - 1, { disabled: page <= 1, label: "Previous page" }));
  // windowed page numbers
  const win = 2;
  const from = Math.max(1, page - win), to = Math.min(pages, page + win);
  if (from > 1) { box.appendChild(btn("1", 1)); if (from > 2) box.appendChild(el("span", { class: "pg-info" }, "…")); }
  for (let p = from; p <= to; p++) box.appendChild(btn(String(p), p, { active: p === page }));
  if (to < pages) { if (to < pages - 1) box.appendChild(el("span", { class: "pg-info" }, "…")); box.appendChild(btn(String(pages), pages)); }
  box.appendChild(btn(icon("chevron-right"), page + 1, { disabled: page >= pages, label: "Next page" }));
}

function showError(msg) {
  const s = $("status");
  clear(s);
  s.appendChild(icon("triangle-alert"));
  s.appendChild(document.createTextNode(" " + msg));
  s.classList.remove("hidden");
}

let toastTimer = null;
function toast(message, bad = false) {
  const node = $("toast");
  clearTimeout(toastTimer);
  node.textContent = message;
  node.classList.remove("hidden", "bad");
  node.classList.toggle("bad", bad);
  toastTimer = setTimeout(() => node.classList.add("hidden"), 2600);
}

function recordValue(rec, field) {
  if (!rec) {
    if (field.name === "user_id" && state.user) return state.user;
    return field.default != null ? field.default : "";
  }
  if (field.name === "user_id") return rec.user_id || rec.id || "";
  const fields = rec.fields || {};
  if (fields[field.name] != null) return fields[field.name];
  if (fields.state && fields.state[field.name] != null) return fields.state[field.name];
  return field.default != null ? field.default : "";
}

function datetimeLocalValue(value) {
  if (!value) return "";
  const text = String(value);
  // Calendar API values are ISO strings in the event's selected timezone.
  // Truncating preserves that local wall time for the native control.
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(text)) return text.slice(0, 16);
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 16);
}

function timezoneChoices() {
  const fallback = ["UTC", "Europe/Rome", "Europe/London", "America/New_York", "America/Los_Angeles", "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney"];
  try {
    return typeof Intl.supportedValuesOf === "function" ? Intl.supportedValuesOf("timeZone") : fallback;
  } catch (_error) {
    return fallback;
  }
}

function browserTimezone() {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch (_error) { return "UTC"; }
}

function makeMemoryField(field, rec) {
  let value = recordValue(rec, field);
  if (field.type === "tags" && typeof value === "string") {
    try { value = JSON.parse(value || "[]"); } catch (e) { value = value.split(","); }
    value = Array.isArray(value) ? value.join(", ") : "";
  }
  const attrs = {
    id: `mf-${field.name}`,
    name: field.name,
    required: !!field.required,
    placeholder: field.placeholder || "",
    disabled: !!(rec && field.readonly_on_edit),
  };
  if (field.type === "weekdays") {
    const selected = new Set((Array.isArray(value) ? value : String(value || "").split(","))
      .map((day) => Number(day)).filter((day) => Number.isInteger(day) && day >= 0 && day < 7));
    const picker = el("div", { class: "weekday-picker", role: "group", "aria-label": field.label });
    CALENDAR_WEEKDAYS.forEach((day, index) => {
      const checkbox = el("input", { type: "checkbox", value: String(index), checked: selected.has(index) });
      picker.appendChild(el("label", { class: "weekday-option" }, [checkbox, el("span", {}, day)]));
    });
    const node = el("div", { class: "memory-field", "data-field": field.name }, [
      el("span", { class: "memory-field-label" }, field.label), picker,
    ]);
    if (field.show_when) node.dataset.showWhen = JSON.stringify(field.show_when);
    return node;
  }

  let input;
  if (field.type === "textarea" || field.type === "json") {
    input = el("textarea", { ...attrs, rows: "4", class: "memory-input memory-textarea" });
    input.value = field.type === "json" && typeof value !== "string"
      ? JSON.stringify(value || {}, null, 2) : value;
  } else if (field.type === "select") {
    input = el("select", { ...attrs, class: "memory-input" }, (field.options || []).map((option) =>
      el("option", { value: option }, option)));
    input.value = value;
  } else if (field.type === "range") {
    const output = el("output", { class: "range-output", for: attrs.id }, Number(value).toFixed(2));
    input = el("input", {
      ...attrs, type: "range", class: "memory-range", value,
      min: field.min, max: field.max, step: field.step || 0.05,
      oninput: () => { output.value = Number(input.value).toFixed(2); },
    });
    const node = el("label", { class: "memory-field range-field", for: attrs.id, "data-field": field.name }, [
      el("span", { class: "memory-field-label" }, [field.label, field.required ? el("i", {}, "required") : null]),
      el("div", { class: "range-wrap" }, [input, output]),
    ]);
    if (field.show_when) node.dataset.showWhen = JSON.stringify(field.show_when);
    return node;
  } else if (field.type === "timezone") {
    input = el("input", { ...attrs, type: "text", class: "memory-input", value, list: "calendar-timezone-options", autocomplete: "off" });
    const options = el("datalist", { id: "calendar-timezone-options" }, timezoneChoices().map((zone) => el("option", { value: zone })));
    const localButton = el("button", { type: "button", class: "timezone-local-btn", onclick: () => { input.value = browserTimezone(); input.dispatchEvent(new Event("change", { bubbles: true })); } }, [icon("locate-fixed"), " Use browser zone"]);
    const node = el("label", { class: "memory-field", for: attrs.id, "data-field": field.name }, [
      el("span", { class: "memory-field-label" }, [field.label, field.required ? el("i", {}, "required") : null]),
      el("div", { class: "timezone-control" }, [input, localButton]), options,
      el("small", {}, "Search for an IANA zone, such as Europe/Rome."),
    ]);
    if (field.show_when) node.dataset.showWhen = JSON.stringify(field.show_when);
    return node;
  } else {
    const inputType = ["number", "datetime-local", "time"].includes(field.type) ? field.type : "text";
    input = el("input", { ...attrs, type: inputType, class: "memory-input", value: field.type === "datetime-local" ? datetimeLocalValue(value) : value, min: field.min, step: field.step });
  }
  const node = el("label", { class: "memory-field", for: attrs.id, "data-field": field.name }, [
    el("span", { class: "memory-field-label" }, [field.label, field.required ? el("i", {}, "required") : null]),
    input,
    field.type === "tags" ? el("small", {}, "Separate values with commas") : null,
    field.type === "json" ? el("small", {}, "JSON object of emotion axis → intensity") : null,
  ]);
  if (field.show_when) node.dataset.showWhen = JSON.stringify(field.show_when);
  return node;
}

function syncConditionalEditorFields() {
  for (const node of document.querySelectorAll("#memory-fields [data-show-when]")) {
    try {
      const rule = JSON.parse(node.dataset.showWhen);
      const input = $(`mf-${rule.field}`);
      node.hidden = !input || input.value !== rule.equals;
      for (const control of node.querySelectorAll("input, select, textarea")) control.disabled = node.hidden;
    } catch (_error) {
      node.hidden = false;
    }
  }
}

function openMemoryEditor(rec = null) {
  if (!state.editor || !state.editor.editable) return;
  state.editing = rec;
  $("memory-editor-eyebrow").textContent = rec ? `Record ${rec.id}` : "New record";
  $("memory-editor-title").textContent = `${rec ? "Edit" : "Add"} ${$("mem-title").textContent}`;
  $("memory-editor-note").textContent = state.editor.description || "";
  $("memory-editor-error").classList.add("hidden");
  const fields = $("memory-fields"); clear(fields);
  for (const field of state.editor.fields || []) fields.appendChild(makeMemoryField(field, rec));
  syncConditionalEditorFields();
  for (const input of fields.querySelectorAll("select, input")) input.addEventListener("change", syncConditionalEditorFields);
  $("memory-editor-save").textContent = rec ? "Save changes" : "Add to memory";
  $("memory-editor").classList.remove("hidden");
  document.body.classList.add("modal-open");
  const first = fields.querySelector("input:not(:disabled), textarea:not(:disabled), select:not(:disabled)");
  if (first) setTimeout(() => first.focus(), 0);
}

function closeMemoryEditor() {
  $("memory-editor").classList.add("hidden");
  document.body.classList.remove("modal-open");
  state.editing = null;
}

function gatherMemoryValues() {
  const values = {};
  for (const field of state.editor.fields || []) {
    const input = $(`mf-${field.name}`);
    const container = input ? input.closest(".memory-field") : document.querySelector(`#memory-fields [data-field="${field.name}"]`);
    if ((!input && field.type !== "weekdays") || (input && input.disabled) || (container && container.hidden)) continue;
    if (field.type === "tags") values[field.name] = input.value.split(",").map((v) => v.trim()).filter(Boolean);
    else if (field.type === "weekdays") values[field.name] = [...container.querySelectorAll("input:checked")].map((day) => Number(day.value));
    else if (field.type === "json") values[field.name] = JSON.parse(input.value || "{}");
    else if (field.type === "range" || field.type === "number") values[field.name] = Number(input.value);
    else values[field.name] = input.value;
  }
  return values;
}

async function saveMemory(event) {
  event.preventDefault();
  const button = $("memory-editor-save");
  const error = $("memory-editor-error");
  const wasEditing = !!state.editing;
  button.disabled = true; button.textContent = "Saving…"; error.classList.add("hidden");
  const base = `${API}/api/memories/${encodeURIComponent(state.character)}/${encodeURIComponent(state.memory)}`;
  const url = state.editing ? `${base}/${encodeURIComponent(state.editing.id)}` : base;
  try {
    await sendJSON(url, { method: state.editing ? "PUT" : "POST", body: { values: gatherMemoryValues() } });
    closeMemoryEditor();
    toast(wasEditing ? "Memory updated" : "Memory added");
    await loadOverview();
  } catch (e) {
    error.textContent = e.message; error.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = wasEditing ? "Save changes" : "Add to memory";
  }
}

async function removeMemory(rec) {
  if (!confirm(`Delete memory ${rec.id}? This cannot be undone.`)) return;
  const url = `${API}/api/memories/${encodeURIComponent(state.character)}/${encodeURIComponent(state.memory)}/${encodeURIComponent(rec.id)}`;
  try {
    await sendJSON(url, { method: "DELETE" });
    toast("Memory deleted");
    state.page = 1;
    await loadOverview();
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------- actions
function selectMemory(name) {
  if (state.memory === name) return;
  state.memory = name;
  state.editor = { editable: false, fields: [], description: "" };
  state.page = 1;
  state.user = "";
  state.q = "";
  if (name === "calendar") state.calendarOffset = 0;
  $("q").value = "";
  $("user-filter").value = "";
  renderSidebar();
  loadPage();
}
function gotoPage(p) {
  state.page = p;
  loadPage();
  $("records").scrollTo({ top: 0 });
}

// debounced search
let qTimer = null;
function onSearchInput() {
  const v = $("q").value.trim();
  $("q-clear").classList.toggle("hidden", !v);
  clearTimeout(qTimer);
  qTimer = setTimeout(() => {
    if (v === state.q) return;
    state.q = v;
    state.page = 1;
    loadPage();
  }, 220);
}

// ---------------------------------------------------------------- knowledge-graph viz
// Obsidian-style interactive graph: a self-contained <canvas> force-renderer
// with no external dependencies. Glowing nodes coloured by kind, light
// "particles" flowing along edges, a continuously settling physics sim, and
// pan / zoom / node-drag with neighbour highlighting on hover.
// Names resolve through the theme's CSS custom properties.  Keeping the canvas
// palette here as tokens lets the light/dark clubhouse themes stay in charge.
const GRAPH_KIND_COLOR = {
  self: "graph-node-self", person: "graph-node-person", fact: "graph-node-fact",
  episode: "graph-node-episode", entity: "graph-node-entity",
};
const GRAPH_KIND_LABEL = {
  self: "self", person: "person", fact: "fact", episode: "episode", entity: "entity",
};
const GRAPH_EDGE_COLOR = {
  relation: "graph-edge-relation", fact: "graph-edge-fact", episode: "graph-edge-episode",
  transition: "graph-edge-transition", co_occurrence: "graph-edge-cooccurrence",
};
const GRAPH_FONT = '"JetBrains Mono", "Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace';

// Live, user-tunable graph settings (driven by the settings panel sliders).
// Shared by reference with the renderer so changes apply without a rebuild.
const GRAPH_SETTINGS = {
  repulsion: 3200,      // centrifugal force (node-node push-apart)
  linkDistance: 78,     // spring rest length
  gravity: 0.018,       // pull toward centre
  particleSpeed: 1.0,   // edge-particle flow speed
  labelZoom: 2.4,       // zoom level at which all labels appear
  nodeLimit: 50,        // nodes fetched (subgraph mode)
  edgeLimit: 300,       // edges fetched
  hideSparse: false,    // drop episode/fact with <2 structural edges
  hideEpFact: false,    // drop every episode/fact node
};
const GRAPH_SETTINGS_DEFAULTS = { ...GRAPH_SETTINGS };

// Persist graph UI choices (settings + full mode + panel visibility) so they
// survive reloads. Best-effort: ignored if localStorage is unavailable.
const GRAPH_UI_KEY = "cm_graph_ui_v1";
function loadGraphUI() {
  try {
    const raw = localStorage.getItem(GRAPH_UI_KEY);
    if (!raw) return;
    const o = JSON.parse(raw);
    if (o.settings) Object.assign(GRAPH_SETTINGS, o.settings);
    if (o.full != null) state.graph.full = !!o.full;
    setGraphSettingsOpen(o.panel === "open");
  } catch (e) { /* ignore */ }
}
function setGraphSettingsOpen(open, { persist = false } = {}) {
  const panel = $("graph-settings"), trigger = $("graph-settings-btn");
  if (!panel || !trigger) return;
  panel.classList.toggle("hidden", !open);
  panel.setAttribute("aria-hidden", open ? "false" : "true");
  trigger.setAttribute("aria-expanded", open ? "true" : "false");
  if (persist) saveGraphUI();
}
function saveGraphUI() {
  try {
    const open = !$("graph-settings").classList.contains("hidden");
    localStorage.setItem(GRAPH_UI_KEY, JSON.stringify({
      settings: GRAPH_SETTINGS, full: state.graph.full, panel: open ? "open" : "closed",
    }));
  } catch (e) { /* ignore */ }
}

function graphUrl() {
  const full = state.graph.full;
  const s = GRAPH_SETTINGS;
  const params = new URLSearchParams(
    full
      ? { limit: String(s.nodeLimit), hops: "2", max_edges: String(s.edgeLimit), include_co_occurrence: "1", full: "1", retrieve_k: "30" }
      : { limit: String(s.nodeLimit), hops: "1", max_edges: String(s.edgeLimit) }
  );
  if (state.graph.q) params.set("q", state.graph.q);
  if (state.graph.user) params.set("user", state.graph.user);
  return `${API}/api/graph/${encodeURIComponent(state.character)}?${params}`;
}

function graphNodeLabel(n) {
  if (n.kind === "fact") return n.content || n.text || n.id;
  if (n.kind === "episode") return n.summary || n.text || n.id;
  if (n.kind === "person") return n.name || n.user_id || n.id;
  if (n.kind === "entity") return n.name || n.id;
  if (n.kind === "self") return "self";
  return n.text || n.id;
}

// "#rrggbb" + alpha -> "rgba(...)" string.
function hexA(hex, a) {
  const h = hex.replace("#", "");
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${clamp(a, 0, 1)})`;
}

function graphCssColor(token, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--${token}`).trim();
  return /^#[0-9a-f]{6}$/i.test(value) ? value : fallback;
}

// Build one renderer bound to a <canvas>. The returned object exposes a small
// imperative API used by the rest of the app (setData / fit / focus / freeze).
function createGraphViz(canvas, opts) {
  const onSelect = opts && opts.onSelect;
  const overlay = (opts && opts.overlay) || document.getElementById("graph-overlay");
  const visibility = opts && opts.visibility;
  const settings = (opts && opts.settings) || GRAPH_SETTINGS;
  const ctx = canvas.getContext("2d");
  const dpr = () => window.devicePixelRatio || 1;

  let nodes = [], edges = [], byId = new Map(), adj = new Map();
  const prevPos = new Map();           // id -> {x,y} across reloads (stable layout)
  const spatialBins = new Map();       // reused by bounded large-graph repulsion
  const view = { scale: 1, panX: 0, panY: 0, w: 0, h: 0 };
  const target = { scale: 1, panX: 0, panY: 0 };
  let alpha = 1;                        // simulation temperature
  let frozen = false;
  let dimMode = false;                  // gray non-retrieved nodes (full-graph query)
  let hoveredId = null;
  let selectedId = null;                // clicked node (persistent highlight)
  let draggingId = null;
  let panning = false;
  let down = null;                      // pointer-down bookkeeping
  const pointers = new Map();           // active pointer id -> canvas coordinates
  let pinch = null;                     // two-finger camera gesture state
  const mouse = { x: 0, y: 0, inside: false };
  let lastT = performance.now();
  let raf = 0;
  let paletteTheme = "";
  let palette = null;
  let fitWhenSettled = false;

  // DOM tooltip living inside the overlay (sibling of the canvas).
  const tip = el("div", { class: "graph-tip-box" });
  (overlay || canvas.parentElement).appendChild(tip);

  // physics constants (world units) — REP/REST/GRAV are live-tunable via settings
  const SPRING = 0.045, DAMP = 0.82, MAXV = 28;
  const MIN_SCALE = 0.05;              // allow zooming/fitting way out for big graphs
  // Per-frame cost caps: radial-gradient halos and additive particles are the
  // expensive parts of the render. On big graphs (hundreds of nodes / thousands
  // of edges) they exhaust the GPU and trigger canvas context loss — which shows
  // up as the graph "disappearing" after a few seconds. Drop them past a threshold.
  const HALO_NODE_CAP = 150;
  const PARTICLE_EDGE_CAP = 500;
  const EXACT_REPULSION_CAP = 220;    // exact pairs for small, tactile subgraphs
  const MAX_REPULSION_NEIGHBORS = 48; // hard bound per node for large graphs
  const REPULSION_CELL = 170;

  function refreshPalette() {
    const theme = document.documentElement.dataset.theme || "light";
    if (theme === paletteTheme && palette) return false;
    paletteTheme = theme;
    const nodeColors = {}, edgeColors = {};
    for (const token of Object.values(GRAPH_KIND_COLOR)) nodeColors[token] = graphCssColor(token, "#408adf");
    for (const token of Object.values(GRAPH_EDGE_COLOR)) edgeColors[token] = graphCssColor(token, "#81798f");
    palette = {
      ink: graphCssColor("graph-ink", "#302947"),
      label: graphCssColor("graph-label", "#302947"),
      labelShadow: graphCssColor("graph-label-shadow", "#fff9e9"),
      muted: graphCssColor("graph-muted", "#81798f"),
      mutedEdge: graphCssColor("graph-muted-edge", "#9a91a7"),
      nodeDefault: graphCssColor("graph-node-default", "#408adf"),
      edgeDefault: graphCssColor("graph-edge-default", "#81798f"),
      nodeColors, edgeColors,
    };
    applyPalette();
    return true;
  }

  function applyPalette() {
    for (const n of nodes) n.color = palette.nodeColors[n.colorKey] || palette.nodeDefault;
    for (const e of edges) e.color = palette.edgeColors[e.colorKey] || palette.edgeDefault;
  }

  // -------------------------------------------------------------- sizing
  function resize() {
    const r = canvas.getBoundingClientRect();
    const d = dpr();
    canvas.width = Math.max(1, Math.round(r.width * d));
    canvas.height = Math.max(1, Math.round(r.height * d));
    view.w = r.width; view.h = r.height;
    if (!draggingId && !panning) { target.scale = view.scale; target.panX = view.panX; target.panY = view.panY; }
  }

  // -------------------------------------------------------------- data in
  function setData(data) {
    selectedId = null;
    if (onSelect) onSelect(null);
    const present = new Set(data.nodes.map(n => n.id));
    dimMode = (data.mode === "full" || data.mode === "trace") && !!data.query;
    const golden = Math.PI * (3 - Math.sqrt(5));
    const next = [];
    data.nodes.forEach((n, i) => {
      const act = clamp(Number(n.activation_norm) || 0, 0, 1);
      let pos = prevPos.get(n.id);
      if (!pos) {
        const ang = i * golden, rad = 26 + Math.sqrt(i) * 30;
        pos = { x: Math.cos(ang) * rad + (Math.random() - 0.5) * 18,
                y: Math.sin(ang) * rad + (Math.random() - 0.5) * 18 };
      }
      next.push({
        id: n.id, kind: n.kind, label: graphNodeLabel(n), raw: n,
        act, r: 4 + 9 * act + (n.kind === "self" ? 4 : 0),
        colorKey: GRAPH_KIND_COLOR[n.kind] || "graph-node-default",
        color: "#408adf",
        retrieved: n.retrieved !== false,
        x: pos.x, y: pos.y, vx: 0, vy: 0, _e: 1, _hot: true,
      });
    });
    byId = new Map(next.map(n => [n.id, n]));
    const nextEdges = [];
    adj = new Map(next.map(n => [n.id, new Set()]));
    for (const e of data.edges) {
      if (!present.has(e.src) || !present.has(e.dst)) continue;
      const w = clamp(Number(e.weight) || 0, 0, 1);
      nextEdges.push({
        src: e.src, dst: e.dst, kind: e.kind, w,
        colorKey: GRAPH_EDGE_COLOR[e.kind] || "graph-edge-default",
        color: "#81798f",
        p: Math.random(), _hot: true,
      });
      adj.get(e.src).add(e.dst);
      adj.get(e.dst).add(e.src);
    }
    prevPos.clear();
    for (const n of next) prevPos.set(n.id, { x: n.x, y: n.y });
    nodes = next; edges = nextEdges;
    // Palette values are cached by theme, not by dataset.
    if (!refreshPalette()) applyPalette();
    alpha = 1;
    // A deliberately tiny warmup removes the worst initial overlap without
    // holding the main thread.  The remaining settle happens over animation
    // frames, and a final fit catches the expanded graph when it cools.
    const warmup = nodes.length <= 80 ? 12 : nodes.length <= EXACT_REPULSION_CAP ? 6 : 2;
    for (let i = 0; i < warmup; i++) simulate();
    fitWhenSettled = true;
    resize();
    fitView(false);
  }

  // -------------------------------------------------------------- camera
  function applyCamera() {
    const d = dpr();
    ctx.setTransform(d * view.scale, 0, 0, d * view.scale, d * view.panX, d * view.panY);
  }
  function toScreen(x, y) { return { x: x * view.scale + view.panX, y: y * view.scale + view.panY }; }
  function toWorld(sx, sy) { return { x: (sx - view.panX) / view.scale, y: (sy - view.panY) / view.scale }; }

  function fitView(animate = true) {
    if (!nodes.length) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (!isFinite(n.x) || !isFinite(n.y)) return;   // never fit a poisoned layout
      minX = Math.min(minX, n.x); minY = Math.min(minY, n.y); maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y);
    }
    const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
    const vw = view.w || canvas.clientWidth || 1, vh = view.h || canvas.clientHeight || 1;
    const pad = Math.min(70, Math.max(18, Math.min(vw, vh) * 0.12));
    const s = clamp(Math.min(Math.max(1, vw - 2 * pad) / w, Math.max(1, vh - 2 * pad) / h), MIN_SCALE, 2.5);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const t = { scale: s, panX: vw / 2 - cx * s, panY: vh / 2 - cy * s };
    if (animate) Object.assign(target, t);
    else { Object.assign(view, t); Object.assign(target, t); }
  }

  function focusNode(id) {
    const n = byId.get(id); if (!n || !isFinite(n.x) || !isFinite(n.y)) return;
    fitWhenSettled = false;
    selectedId = id;
    if (onSelect) onSelect(n);
    const w = view.w || canvas.clientWidth || 1;
    const h = view.h || canvas.clientHeight || 1;
    const s = Math.max(view.scale, 1.1);
    const t = { scale: s, panX: w / 2 - n.x * s, panY: h / 2 - n.y * s };
    Object.assign(target, t);
  }

  function zoomBy(factor, sx = view.w / 2, sy = view.h / 2) {
    fitWhenSettled = false;
    // Use the target camera so repeated button presses compose even while the
    // previous animated zoom is still settling.
    const baseScale = target.scale || view.scale;
    const worldX = (sx - target.panX) / baseScale;
    const worldY = (sy - target.panY) / baseScale;
    const ns = clamp(baseScale * factor, MIN_SCALE, 4);
    target.scale = ns;
    target.panX = sx - worldX * ns;
    target.panY = sy - worldY * ns;
  }

  function relayout() {
    const golden = Math.PI * (3 - Math.sqrt(5));
    const spread = Math.max(26, settings.linkDistance * 0.68);
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i], angle = i * golden;
      const radius = spread + Math.sqrt(i + 1) * spread;
      n.x = Math.cos(angle) * radius;
      n.y = Math.sin(angle) * radius;
      n.vx = 0; n.vy = 0;
      prevPos.set(n.id, { x: n.x, y: n.y });
    }
    frozen = false;
    alpha = 1;
    fitWhenSettled = true;
    resize();
    fitView(false);
    return frozen;
  }

  // -------------------------------------------------------------- physics
  function reheat(a) { if (!frozen) alpha = Math.max(alpha, a == null ? 0.7 : a); }

  function applyRepulsion(a, b) {
    let dx = a.x - b.x, dy = a.y - b.y;
    let d2 = dx * dx + dy * dy;
    if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = dx * dx + dy * dy + 0.01; }
    const d = Math.sqrt(d2), f = settings.repulsion / d2;
    const fx = f * dx / d, fy = f * dy / d;
    a.ax += fx; a.ay += fy; b.ax -= fx; b.ay -= fy;
  }

  function applyBoundedRepulsion() {
    spatialBins.clear();
    for (const a of nodes) {
      const cx = Math.floor(a.x / REPULSION_CELL), cy = Math.floor(a.y / REPULSION_CELL);
      a._cx = cx; a._cy = cy;
      const key = cx * 1048576 + cy;
      let bin = spatialBins.get(key);
      if (!bin) { bin = []; spatialBins.set(key, bin); }
      bin.push(a);
    }
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i]; let checked = 0;
      for (let ox = -1; ox <= 1 && checked < MAX_REPULSION_NEIGHBORS; ox++) {
        for (let oy = -1; oy <= 1 && checked < MAX_REPULSION_NEIGHBORS; oy++) {
          const bin = spatialBins.get((a._cx + ox) * 1048576 + a._cy + oy);
          if (!bin || !bin.length) continue;
          const start = i % bin.length;
          for (let k = 0; k < bin.length && checked < MAX_REPULSION_NEIGHBORS; k++) {
            const b = bin[(start + k) % bin.length];
            if (b === a) continue;
            applyRepulsion(a, b); checked++;
          }
        }
      }
    }
  }

  function simulate() {
    if (alpha < 0.02 && !draggingId) return;
    const n = nodes.length;
    for (let i = 0; i < n; i++) { nodes[i].ax = 0; nodes[i].ay = 0; }
    if (n <= EXACT_REPULSION_CAP) {
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) applyRepulsion(nodes[i], nodes[j]);
    } else applyBoundedRepulsion();
    for (const e of edges) {
      const a = byId.get(e.src), b = byId.get(e.dst); if (!a || !b) continue;
      let dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = SPRING * (d - settings.linkDistance), fx = f * dx / d, fy = f * dy / d;
      a.ax += fx; a.ay += fy; b.ax -= fx; b.ay -= fy;
    }
    for (const a of nodes) {
      a.ax -= settings.gravity * a.x; a.ay -= settings.gravity * a.y;
      if (a.id === draggingId) { a.vx = 0; a.vy = 0; continue; }
      a.vx = (a.vx + a.ax) * DAMP; a.vy = (a.vy + a.ay) * DAMP;
      a.vx = clamp(a.vx, -MAXV, MAXV); a.vy = clamp(a.vy, -MAXV, MAXV);
      a.x += a.vx * alpha; a.y += a.vy * alpha;
    }
    alpha *= 0.985; if (alpha < 0.001) alpha = 0.001;
  }

  // which nodes/edges are "hot" (connected to the hovered OR selected node).
  function computeHighlights() {
    const activeId = hoveredId || selectedId;
    let hset = null;
    if (activeId) {
      hset = new Set([activeId]);
      const nb = adj.get(activeId); if (nb) for (const id of nb) hset.add(id);
    }
    for (const a of nodes) {
      const t = !activeId ? 1 : (hset.has(a.id) ? 1 : 0);
      a._hot = !!t;
      a._e += (t - a._e) * 0.15;
    }
    for (const e of edges) {
      e._hot = !activeId ? true : (e.src === activeId || e.dst === activeId);
    }
  }

  // -------------------------------------------------------------- render
  function draw(dt) {
    refreshPalette();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    applyCamera();
    const activeId = hoveredId || selectedId;

    // edges
    ctx.lineCap = "round";
    for (const e of edges) {
      const a = byId.get(e.src), b = byId.get(e.dst); if (!a || !b) continue;
      let baseA = 0.12 + 0.5 * e.w;
      let col = e.color, lw = (0.6 + 2.4 * e.w);
      if (dimMode) {
        const both = a.retrieved && b.retrieved;
        if (both) { baseA = 0.25 + 0.6 * e.w; lw *= 1.6; }
        else if (a.retrieved || b.retrieved) { baseA = 0.08 + 0.18 * e.w; col = palette.mutedEdge; }
        else { baseA = 0.04 + 0.06 * e.w; col = palette.muted; }
      }
      const aA = (activeId && !e._hot) ? baseA * 0.45 : baseA;
      ctx.strokeStyle = hexA(col, aA);
      ctx.lineWidth = (e._hot ? lw * 1.5 : lw);
      if (e.kind === "co_occurrence") ctx.setLineDash([4, 5]); else ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // flowing particles along edges (skipped past the edge cap — additive arcs
    // are expensive and big graphs overload the GPU).
    ctx.globalCompositeOperation = "lighter";
    if (edges.length <= PARTICLE_EDGE_CAP) for (const e of edges) {
      const a = byId.get(e.src), b = byId.get(e.dst); if (!a || !b) continue;
      e.p = (e.p + dt * settings.particleSpeed * (0.04 + 0.22 * e.w)) % 1;
      let pa = (activeId && !e._hot) ? 0 : (e._hot ? 0.95 : 0.3);
      if (dimMode) {
        const both = a.retrieved && b.retrieved;
        pa = both ? 0.95 : 0;            // only retrieved links carry the glow
      }
      if (pa <= 0) continue;
      const px = a.x + (b.x - a.x) * e.p, py = a.y + (b.y - a.y) * e.p;
      ctx.fillStyle = hexA(e.color, pa);
      ctx.beginPath(); ctx.arc(px, py, e._hot ? 2.6 : 1.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";

    // node halos (additive bloom) — skipped for grayed (non-retrieved) nodes and
    // skipped entirely on big graphs (gradient allocation is the heaviest cost).
    const GRAYC = palette.muted;
    const drawHalos = nodes.length <= HALO_NODE_CAP;
    if (drawHalos) for (const a of nodes) {
      const isGray = dimMode && !a.retrieved;
      const baseA = 0.4 + 0.6 * a.act;
      const na = (activeId && !a._hot) ? baseA * 0.4 : baseA;
      if (isGray || na <= 0.01) continue;
      const R = a.r * 3;
      const g = ctx.createRadialGradient(a.x, a.y, a.r * 0.2, a.x, a.y, R);
      g.addColorStop(0, hexA(a.color, na * 0.55));
      g.addColorStop(1, hexA(a.color, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(a.x, a.y, R, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";

    // node cores
    for (const a of nodes) {
      const isGray = dimMode && !a.retrieved;
      let baseA = 0.55 + 0.45 * a.act;
      let col = a.color;
      if (isGray) { baseA = 0.32; col = GRAYC; }
      const na = (activeId && !a._hot) ? baseA * 0.5 : baseA;
      ctx.fillStyle = hexA(col, na);
      ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2); ctx.fill();
      if (a.kind === "self") {
        ctx.lineWidth = 2.5; ctx.strokeStyle = hexA(palette.ink, na);
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + 2.5, 0, Math.PI * 2); ctx.stroke();
      } else if ((a._hot || (dimMode && a.retrieved)) && a._e > 0.05) {
        ctx.lineWidth = 2; ctx.strokeStyle = hexA(palette.ink, 0.4 + 0.5 * a._e);
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + 2, 0, Math.PI * 2); ctx.stroke();
      }
      // Persistent ring on the clicked (selected) node so the click is obvious.
      if (selectedId === a.id) {
        ctx.lineWidth = 3; ctx.strokeStyle = hexA(palette.ink, 0.95);
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + 4, 0, Math.PI * 2); ctx.stroke();
      }
    }

    // labels: hidden by default (Obsidian-style dots), shown on hover or when
    // zoomed in past a threshold. Kept short so the canvas stays clean.
    // Hovering a node also reveals its direct neighbours' labels.
    const showAll = view.scale > settings.labelZoom;
    const hoveredNeighbors = (activeId && adj.get(activeId)) || null;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.font = `10px ${GRAPH_FONT}`;
    for (const a of nodes) {
      const isActive = !!activeId && a.id === activeId;
      const show = showAll || isActive || (hoveredNeighbors && hoveredNeighbors.has(a.id));
      if (!show) continue;
      const txt = a.label.length > 22 ? a.label.slice(0, 21) + "…" : a.label;
      ctx.fillStyle = hexA(palette.label, isActive ? 0.95 : 0.72);
      ctx.shadowColor = hexA(palette.labelShadow, 0.95);
      ctx.shadowBlur = 5;
      ctx.fillText(txt, a.x, a.y + a.r + 3);
      ctx.shadowBlur = 0;
    }
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
    if (!frozen) simulate();
    if (fitWhenSettled && (!frozen && alpha <= 0.025)) {
      fitView(false);
      fitWhenSettled = false;
    }
    view.scale += (target.scale - view.scale) * 0.2;
    view.panX += (target.panX - view.panX) * 0.2;
    view.panY += (target.panY - view.panY) * 0.2;
    computeHighlights();
    draw(dt);
    raf = requestAnimationFrame(frame);
  }

  // -------------------------------------------------------------- picking
  function pick(mx, my, padding = 6) {
    let best = null, bestD = Infinity;
    for (const a of nodes) {
      const s = toScreen(a.x, a.y);
      const d = Math.hypot(s.x - mx, s.y - my);
      const hit = a.r * view.scale + padding;
      if (d <= hit && d < bestD) { bestD = d; best = a; }
    }
    return best;
  }

  // -------------------------------------------------------------- pointers
  function pointerPosition(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function startPinch() {
    const entries = Array.from(pointers.entries()).slice(0, 2);
    if (entries.length < 2) { pinch = null; return; }
    const [a, b] = entries.map(([, point]) => point);
    const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
    // Stop any camera easing first so the point between both fingers remains
    // anchored throughout the gesture.
    Object.assign(target, view);
    pinch = {
      ids: entries.map(([id]) => id),
      distance: Math.max(1, Math.hypot(b.x - a.x, b.y - a.y)),
      scale: view.scale,
      world: toWorld(cx, cy),
    };
    draggingId = null;
    panning = false;
    down = null;
    tip.classList.remove("show");
    canvas.classList.add("grabbing");
  }

  function onPointerDown(e) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.preventDefault();
    fitWhenSettled = false;
    const point = pointerPosition(e);
    pointers.set(e.pointerId, point);
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* unsupported capture */ }
    if (pointers.size >= 2) { startPinch(); return; }

    // Graph dots can be visually tiny after fitting a large graph. Give
    // coarse pointers a forgiving invisible hit area without changing the
    // tighter mouse-hover behaviour.
    const n = pick(point.x, point.y, e.pointerType === "mouse" ? 6 : 16);
    if (n) {
      draggingId = n.id;
      down = { pointerId: e.pointerId, mode: "node", nodeId: n.id, sx: point.x, sy: point.y, moved: false };
      reheat(0.5);
      canvas.classList.add("grabbing");
    } else {
      panning = true;
      down = { pointerId: e.pointerId, mode: "pan", sx: point.x, sy: point.y,
               panX: view.panX, panY: view.panY, moved: false };
      canvas.classList.add("grabbing");
    }
  }

  function updateHover(mx, my) {
    mouse.x = mx; mouse.y = my; mouse.inside = true;
    const n = pick(mx, my);
    hoveredId = n ? n.id : null;
    canvas.classList.toggle("node-hover", !!n);
    if (n) {
      tip.innerHTML = `<div class="tt-kind">${GRAPH_KIND_LABEL[n.kind] || n.kind}</div>` +
        `<div>${esc(n.label)}</div>` +
        `<div class="tt-act">activation ${Math.round(n.act * 100)}%</div>`;
      const s = toScreen(n.x, n.y);
      tip.style.left = clamp(s.x + 14, 4, view.w - tip.offsetWidth - 4) + "px";
      tip.style.top = clamp(s.y + 14, 4, view.h - tip.offsetHeight - 4) + "px";
      tip.classList.add("show");
    } else {
      tip.classList.remove("show");
    }
  }

  function onPointerMove(e) {
    const point = pointerPosition(e);
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, point);

    if (pointers.size >= 2) {
      e.preventDefault();
      if (!pinch || pinch.ids.some(id => !pointers.has(id))) startPinch();
      if (!pinch) return;
      const a = pointers.get(pinch.ids[0]), b = pointers.get(pinch.ids[1]);
      const distance = Math.max(1, Math.hypot(b.x - a.x, b.y - a.y));
      const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
      const ns = clamp(pinch.scale * distance / pinch.distance, MIN_SCALE, 4);
      view.scale = target.scale = ns;
      view.panX = target.panX = cx - pinch.world.x * ns;
      view.panY = target.panY = cy - pinch.world.y * ns;
      return;
    }

    if (draggingId || panning) {
      if (!down || down.pointerId !== e.pointerId) return;
      const mx = point.x, my = point.y;
      if (!down.moved && Math.hypot(mx - down.sx, my - down.sy) >= 6) down.moved = true;
      if (draggingId) {
        if (!down.moved) return;
        const n = byId.get(draggingId); const w = toWorld(mx, my);
        if (n) { n.x = w.x; n.y = w.y; n.vx = 0; n.vy = 0; prevPos.set(n.id, { x: n.x, y: n.y }); }
        reheat(0.4);
      } else if (panning) {
        if (!down.moved) return;
        view.panX = target.panX = down.panX + (mx - down.sx);
        view.panY = target.panY = down.panY + (my - down.sy);
      }
      return;
    }
    if (e.pointerType === "mouse") updateHover(point.x, point.y);
  }

  function finishPointer(e, cancelled = false) {
    const point = pointers.get(e.pointerId) || pointerPosition(e);
    const tapped = !cancelled && pointers.size === 1 && !pinch && down &&
      down.pointerId === e.pointerId && !down.moved;
    const tappedNodeId = tapped && down.mode === "node" ? down.nodeId : null;
    const tappedEmpty = tapped && down.mode === "pan";
    if (draggingId) reheat(0.3);
    pointers.delete(e.pointerId);
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }

    if (pointers.size >= 2) {
      startPinch();
      return;
    }
    if (pointers.size === 1) {
      // Continue naturally as a one-finger pan after ending a pinch, without a
      // jump and without treating the eventual release as a tap.
      const [pointerId, remaining] = pointers.entries().next().value;
      pinch = null;
      draggingId = null;
      panning = true;
      down = { pointerId, mode: "pan", sx: remaining.x, sy: remaining.y,
               panX: view.panX, panY: view.panY, moved: true };
      return;
    }

    pinch = null;
    draggingId = null;
    panning = false;
    down = null;
    canvas.classList.remove("grabbing");
    if (tappedNodeId) focusNode(tappedNodeId);
    else if (tappedEmpty) {
      selectedId = null;
      if (onSelect) onSelect(null);
    }
    if (e.pointerType === "mouse") updateHover(point.x, point.y);
  }
  function onPointerCancel(e) { finishPointer(e, true); }

  function onLeave() {
    hoveredId = null; mouse.inside = false; tip.classList.remove("show");
    if (!panning && !draggingId) canvas.classList.remove("node-hover");
  }
  function onWheel(e) {
    e.preventDefault();
    fitWhenSettled = false;
    const w = toWorld(e.offsetX, e.offsetY);
    const factor = Math.exp(-e.deltaY * 0.0015);
    const ns = clamp(view.scale * factor, MIN_SCALE, 4);
    view.scale = target.scale = ns;
    view.panX = target.panX = e.offsetX - w.x * ns;
    view.panY = target.panY = e.offsetY - w.y * ns;
  }
  function onResize() {
    if (!visibility || !visibility.classList.contains("hidden")) resize();
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", finishPointer);
  canvas.addEventListener("pointercancel", onPointerCancel);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("resize", onResize);
  let ro = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => resize());
    ro.observe(canvas);
  }

  resize();
  raf = requestAnimationFrame(frame);

  return {
    setData, resize, fit: () => { fitWhenSettled = false; resize(); fitView(false); }, focus: focusNode, relayout,
    zoomIn: () => zoomBy(1.35), zoomOut: () => zoomBy(1 / 1.35),
    reheat, freeze: () => { frozen = !frozen; return frozen; },
    isFrozen: () => frozen,
    debugView: () => {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of nodes) {
        if (!isFinite(n.x) || !isFinite(n.y)) continue;
        minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y);
      }
      return { scale: view.scale, w: view.w, h: view.h, panX: view.panX, panY: view.panY,
               minX, minY, maxX, maxY, n: nodes.length };
    },
    destroy() {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", finishPointer);
      canvas.removeEventListener("pointercancel", onPointerCancel);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
      if (ro) ro.disconnect();
      tip.remove();
    },
  };
}

// Drop episode/fact nodes based on two toggles:
//  - hideSparse: hide episode/fact with fewer than two *structural* edges
//    (co_occurrence edges are noisy and do not count toward the threshold).
//  - hideEpFact: hide every episode/fact node entirely.
// Applied client-side so toggling is instant (no refetch). The raw fetched
// payload is preserved in state.graph.data for re-filtering.
const GRAPH_LEAF_KINDS = new Set(["episode", "fact"]);
function filterGraphData(data) {
  if (!GRAPH_SETTINGS.hideSparse && !GRAPH_SETTINGS.hideEpFact) return data;
  // Degree counts only structural (non-co_occurrence) edges.
  const deg = new Map();
  for (const e of data.edges) {
    if (e.kind === "co_occurrence") continue;
    deg.set(e.src, (deg.get(e.src) || 0) + 1);
    deg.set(e.dst, (deg.get(e.dst) || 0) + 1);
  }
  const keep = new Set();
  for (const n of data.nodes) {
    if (GRAPH_LEAF_KINDS.has(n.kind)) {
      if (GRAPH_SETTINGS.hideEpFact) continue;
      if (GRAPH_SETTINGS.hideSparse && (deg.get(n.id) || 0) < 2) continue;
    }
    keep.add(n.id);
  }
  return {
    ...data,
    nodes: data.nodes.filter(n => keep.has(n.id)),
    edges: data.edges.filter(e => keep.has(e.src) && keep.has(e.dst)),
  };
}
function applyGraphFilter() {
  if (!state.graph.data || !state.graph._gv) return;
  state.graph._gv.setData(filterGraphData(state.graph.data));
  requestAnimationFrame(() => state.graph._gv.resize());
}

async function renderGraphView() {
  if (!graphControlsWired) wireGraphControls();
  $("graph-user").value = state.user || "";
  state.graph.user = state.user || "";
  $("graph-q").value = state.graph.q || "";
  $("graph-q-clear").classList.toggle("hidden", !state.graph.q);
  $("graph-full").classList.toggle("graph-freeze-on", !!state.graph.full);
  setButtonContent($("graph-full"), state.graph.full ? "circle-dot" : "circle-dashed", "full graph");

  let data;
  try {
    data = await getJSON(graphUrl());
  } catch (e) {
    $("graph-meta").textContent = "";
    clear($("graph-list"));
    showError(e.message);
    return;
  }
  state.graph.data = data;
  $("status").classList.add("hidden");
  const view = filterGraphData(data);
  const trunc = data.truncated ? " (truncated)" : "";
  $("graph-meta").textContent =
    `${view.nodes.length} nodes · ${view.edges.length} edges${trunc}` +
    (data.mode === "full" ? " · full" : "") +
    (data.query ? ` · query “${data.query}”` : "");

  const canvas = $("cy");
  if (!state.graph._gv) state.graph._gv = createGraphViz(canvas, {
    onSelect: renderNodeDetail,
    settings: GRAPH_SETTINGS,
    overlay: $("graph-overlay"),
    visibility: $("graph-view"),
  });
  state.graph._gv.resize();
  state.graph._gv.setData(view);
  requestAnimationFrame(() => state.graph._gv.resize());
  renderGraphList(view);
}

// Bind one settings slider to GRAPH_SETTINGS. `live` sliders only reheat the
// physics (no refetch); fetch sliders (node/edge limit) debounce a reload.
let _sliderTimer = null;
// These controls belong to the page, not a renderer instance. A character
// change replaces only the canvas renderer, so this guard survives it.
let graphControlsWired = false;
function wireSlider(id, key, opts) {
  opts = opts || {};
  const input = $(id), val = $(id + "-val");
  const fmt = opts.fmt || ((v) => String(v));
  input.addEventListener("input", () => {
    const v = Number(input.value);
    GRAPH_SETTINGS[key] = v;
    if (val) val.textContent = fmt(v);
    if (opts.live && state.graph._gv) state.graph._gv.reheat(0.6);
    else {
      clearTimeout(_sliderTimer);
      _sliderTimer = setTimeout(() => renderGraphView(), opts.debounce || 250);
    }
    saveGraphUI();
  });
}
function syncSlider(id, key, v) {
  const input = $(id), val = $(id + "-val");
  input.value = v; GRAPH_SETTINGS[key] = v;
  if (val) val.textContent = (id === "gs-gravity") ? v.toFixed(3)
    : (id === "gs-particles" || id === "gs-labelzoom") ? v.toFixed(1) : String(v);
}
function syncAllSliders() {
  syncSlider("gs-repulsion", "repulsion", GRAPH_SETTINGS.repulsion);
  syncSlider("gs-link", "linkDistance", GRAPH_SETTINGS.linkDistance);
  syncSlider("gs-gravity", "gravity", GRAPH_SETTINGS.gravity);
  syncSlider("gs-particles", "particleSpeed", GRAPH_SETTINGS.particleSpeed);
  syncSlider("gs-labelzoom", "labelZoom", GRAPH_SETTINGS.labelZoom);
  syncSlider("gs-nodes", "nodeLimit", GRAPH_SETTINGS.nodeLimit);
  syncSlider("gs-edges", "edgeLimit", GRAPH_SETTINGS.edgeLimit);
  syncCheckbox("gs-hide-sparse", "hideSparse");
  syncCheckbox("gs-hide-epfact", "hideEpFact");
}
function wireCheckbox(id, key) {
  const el = $(id);
  el.addEventListener("change", () => {
    GRAPH_SETTINGS[key] = el.checked;
    applyGraphFilter();
    saveGraphUI();
  });
}
function syncCheckbox(id, key) {
  const el = $(id); if (el) el.checked = !!GRAPH_SETTINGS[key];
}

function wireGraphControls() {
  if (graphControlsWired) return;
  graphControlsWired = true;
  state.graph._wired = true;
  let t = null;
  $("graph-q").addEventListener("input", (e) => {
    const v = e.target.value.trim();
    $("graph-q-clear").classList.toggle("hidden", !v);
    clearTimeout(t);
    t = setTimeout(() => {
      if (v === state.graph.q) return;
      state.graph.q = v;
      state.q = v; $("q").value = v;
      renderGraphView();
    }, 220);
  });
  $("graph-q-clear").addEventListener("click", () => {
    $("graph-q").value = ""; state.graph.q = ""; state.q = ""; $("q").value = "";
    $("graph-q-clear").classList.add("hidden");
    renderGraphView();
  });
  $("graph-user").addEventListener("change", (e) => {
    state.graph.user = e.target.value;
    state.user = e.target.value;
    $("user-filter").value = e.target.value;
    renderGraphView();
  });
  $("graph-repel").addEventListener("click", () => {
    if (!state.graph._gv) return;
    const frozen = state.graph._gv.relayout();
    $("graph-freeze").classList.toggle("graph-freeze-on", frozen);
    setButtonContent($("graph-freeze"), frozen ? "play" : "pause", frozen ? "resume" : "freeze");
  });
  $("graph-full").addEventListener("click", () => {
    state.graph.full = !state.graph.full;
    $("graph-full").classList.toggle("graph-freeze-on", state.graph.full);
    setButtonContent($("graph-full"), state.graph.full ? "circle-dot" : "circle-dashed", "full graph");
    // Full mode shows the whole graph by default; bump the caps so the slider
    // still lets the user dial it back down.
    if (state.graph.full) { syncSlider("gs-nodes", "nodeLimit", 6000); syncSlider("gs-edges", "edgeLimit", 9000); }
    else { syncSlider("gs-nodes", "nodeLimit", 50); syncSlider("gs-edges", "edgeLimit", 300); }
    saveGraphUI();
    renderGraphView();
  });
  $("graph-settings-btn").addEventListener("click", () => {
    const open = $("graph-settings").classList.contains("hidden");
    setGraphSettingsOpen(open, { persist: true });
  });
  $("gs-close").addEventListener("click", () => setGraphSettingsOpen(false, { persist: true }));
  $("gs-reset").addEventListener("click", () => {
    const limitsChanged = GRAPH_SETTINGS.nodeLimit !== GRAPH_SETTINGS_DEFAULTS.nodeLimit ||
      GRAPH_SETTINGS.edgeLimit !== GRAPH_SETTINGS_DEFAULTS.edgeLimit;
    Object.assign(GRAPH_SETTINGS, GRAPH_SETTINGS_DEFAULTS);
    syncAllSliders();
    saveGraphUI();
    // Fetch limits change server-side graph selection. Refetch directly rather
    // than doing a transient client-side filter on the stale larger dataset.
    if (limitsChanged) { renderGraphView(); return; }
    applyGraphFilter();
    if (state.graph._gv) state.graph._gv.reheat(1);
  });
  syncAllSliders();
  wireSlider("gs-repulsion", "repulsion", { live: true, fmt: (v) => String(v) });
  wireSlider("gs-link", "linkDistance", { live: true, fmt: (v) => String(v) });
  wireSlider("gs-gravity", "gravity", { live: true, fmt: (v) => v.toFixed(3) });
  wireSlider("gs-particles", "particleSpeed", { live: true, fmt: (v) => v.toFixed(1) });
  wireSlider("gs-labelzoom", "labelZoom", { live: true, fmt: (v) => v.toFixed(1) });
  wireSlider("gs-nodes", "nodeLimit", { fmt: (v) => String(v) });
  wireSlider("gs-edges", "edgeLimit", { fmt: (v) => String(v) });
  wireCheckbox("gs-hide-sparse", "hideSparse");
  wireCheckbox("gs-hide-epfact", "hideEpFact");
  $("graph-fit").addEventListener("click", () => {
    if (state.graph._gv) state.graph._gv.fit();
  });
  $("graph-zoom-in").addEventListener("click", () => {
    if (state.graph._gv) state.graph._gv.zoomIn();
  });
  $("graph-zoom-out").addEventListener("click", () => {
    if (state.graph._gv) state.graph._gv.zoomOut();
  });
  $("graph-freeze").addEventListener("click", () => {
    if (!state.graph._gv) return;
    const f = state.graph._gv.freeze();
    $("graph-freeze").classList.toggle("graph-freeze-on", f);
    setButtonContent($("graph-freeze"), f ? "play" : "pause", f ? "resume" : "freeze");
  });
  // Space toggles the physics freeze while the graph is on screen.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if ($("graph-view").classList.contains("hidden")) return;
    e.preventDefault();
    if (!state.graph._gv) return;
    const f = state.graph._gv.freeze();
    $("graph-freeze").classList.toggle("graph-freeze-on", f);
    setButtonContent($("graph-freeze"), f ? "play" : "pause", f ? "resume" : "freeze");
  });
}

function syncGraphUserPicker(users) {
  const sel = $("graph-user");
  const prev = state.graph.user;
  clear(sel);
  sel.appendChild(el("option", { value: "" }, "No user bias"));
  for (const u of users || []) sel.appendChild(el("option", { value: u }, u));
  sel.value = prev && (users || []).includes(prev) ? prev : "";
  state.graph.user = sel.value;
}

function renderGraphList(data) {
  const ol = $("graph-list"); clear(ol);
  const sorted = data.nodes.slice()
    .sort((a, b) => (+b.activation || 0) - (+a.activation || 0))
    .slice(0, 15);
  for (const n of sorted) {
    const a = Math.max(0, Math.min(1, +n.activation_norm || 0));
    ol.appendChild(el("li", { onclick: () => focusGraphNode(n.id) }, [
      el("div", {}, [
        el("span", { class: "gl-kind" }, GRAPH_KIND_LABEL[n.kind] || n.kind),
        " ",
        el("span", { class: "gl-act" }, (a * 100).toFixed(0) + "%"),
      ]),
      el("div", { class: "gl-text" }, graphNodeLabel(n)),
    ]));
  }
}

function focusGraphNode(id) {
  if (state.graph._gv) state.graph._gv.focus(id);
}

// Build the side detail card for a clicked node (called by the renderer's
// onSelect callback with the internal node object).
function renderNodeDetail(node) {
  renderNodeDetailInto(node, $("graph-detail"));
}

function renderNodeDetailInto(node, box) {
  clear(box);
  if (!node) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const raw = node.raw || {};
  const text = raw.content || raw.summary || raw.text || node.label;
  box.appendChild(el("div", { class: "gd-head" }, [chip(GRAPH_KIND_LABEL[node.kind] || node.kind, "kind-chip")]));
  box.appendChild(el("div", { class: "gd-title" }, node.label));
  if (text && text !== node.label) box.appendChild(el("div", { class: "gd-text" }, text));
  if (raw.confidence != null) box.appendChild(fieldBar("confidence", raw.confidence, { klass: "good" }));
  if (raw.importance != null) box.appendChild(fieldBar("importance", raw.importance));
  if (node.act != null) box.appendChild(fieldBar("activation", node.act, { klass: "warn" }));
  if (raw.score_breakdown && typeof raw.score_breakdown === "object") {
    const bd = scoreBreakdownBlock(raw.score_breakdown);
    if (bd) box.appendChild(bd);
  }
  if (raw.emotional_shift && typeof raw.emotional_shift === "object") {
    for (const [axis, value] of Object.entries(raw.emotional_shift)) {
      if (Number(value) > 0) box.appendChild(fieldBar(`emotion · ${axis}`, Number(value)));
    }
  }
  let aliases = [];
  try { aliases = Array.isArray(raw.aliases) ? raw.aliases : JSON.parse(raw.aliases || "[]"); } catch (e) { aliases = []; }
  if (aliases.length) box.appendChild(el("div", { class: "kw-chips" }, aliases.map(kwChip)));
  if (raw.comment) box.appendChild(el("div", { class: "gd-comment" }, "“" + raw.comment + "”"));
  const meta = [];
  if (raw.user_id) meta.push(["user", raw.user_id]);
  if (raw.name) meta.push(["name", raw.name]);
  if (raw.type) meta.push(["type", raw.type]);
  if (raw.created_at) meta.push(["created", relTime(raw.created_at)]);
  if (raw.last_recalled) meta.push(["recalled", relTime(raw.last_recalled)]);
  if (raw.recall_count != null) meta.push(["×", String(raw.recall_count)]);
  if (meta.length) box.appendChild(el("div", { class: "gd-meta" },
    meta.map(([k, v]) => el("span", {}, [el("b", {}, k + ":"), " " + v]))));
}

function renderLiveNodeDetail(node) {
  renderNodeDetailInto(node, $("live-graph-detail"));
}

// ---------------------------------------------------------------- live /context monitor
function liveUrl() {
  const params = new URLSearchParams(location.search);
  params.set("tab", "live");
  if (state.character) params.set("character", state.character);
  return `${location.pathname}?${params.toString()}${location.hash || ""}`;
}

function liveEmbedUrl() {
  const params = new URLSearchParams();
  if (state.character) params.set("character", state.character);
  return `${LIVE_GRAPH_EMBED_PATH}${params.toString() ? "?" + params.toString() : ""}`;
}

function updateLiveEmbedLink() {
  const link = $("live-embed-link");
  if (link) link.href = liveEmbedUrl();
}

function updateViewUrl({ replace = false } = {}) {
  const params = new URLSearchParams(location.search);
  const active = $("tab-live").classList.contains("active")
    ? "live" : $("tab-configure").classList.contains("active") ? "configure" : "browse";
  if (active === "browse") params.delete("tab"); else params.set("tab", active);
  if (state.character) params.set("character", state.character);
  const url = `${location.pathname}${params.toString() ? "?" + params.toString() : ""}${location.hash || ""}`;
  if (replace) history.replaceState({}, "", url); else history.pushState({}, "", url);
}

function liveSetConnection(kind, label) {
  state.live.connection = kind;
  const dot = $("live-status-dot"), text = $("live-status");
  if (dot) dot.className = `live-status-dot ${kind}`;
  if (text) text.textContent = label;
  const embedDot = $("live-embed-status-dot"), embedText = $("live-embed-status");
  if (embedDot) embedDot.className = `live-status-dot ${kind}`;
  if (embedText) embedText.textContent = label.replace("Listening for /context", "Live");
}

function closeLiveStream() {
  if (state.live.source) {
    state.live.source.close();
    state.live.source = null;
  }
  state.live.connection = "closed";
}

function connectLiveStream() {
  if (!state.live._active || !state.character) return;
  closeLiveStream();
  if (!window.EventSource) {
    liveSetConnection("error", "Live updates are not supported by this browser");
    return;
  }
  liveSetConnection("connecting", "Connecting to /context…");
  // EventSource cannot send headers, so the API key rides the query string.
  const source = new EventSource(authQuery(`${API}/api/context-events/${encodeURIComponent(state.character)}`));
  state.live.source = source;
  source.onopen = () => liveSetConnection("connected", "Listening for /context");
  source.onerror = () => {
    if (state.live._active) liveSetConnection("connecting", "Connection interrupted · retrying…");
  };
  const receiveContext = (event) => {
    let payload;
    try { payload = JSON.parse(event.data); } catch (error) { return; }
    const id = String(payload.id || payload.sequence || "");
    if (!id || state.live.events.has(id)) return;
    state.live.events.set(id, payload);
    while (state.live.events.size > 25) {
      const oldest = state.live.events.keys().next().value;
      state.live.events.delete(oldest);
    }
    renderLiveTimeline();
    const newest = latestLiveEvent();
    if (state.live.follow && newest && String(newest.id) === id) {
      state.live.selectedId = id;
      $("live-new").classList.add("hidden");
      renderLiveEvent(newest);
    } else if (state.live.selectedId !== id) {
      $("live-new").classList.remove("hidden");
    }
  };
  source.addEventListener("context", receiveContext);
  // Be tolerant of SSE relays that strip the named-event line.
  source.onmessage = receiveContext;
}

function sortedLiveEvents() {
  return Array.from(state.live.events.values()).sort((a, b) =>
    Number(a.sequence || a.id || 0) - Number(b.sequence || b.id || 0));
}

function latestLiveEvent() {
  const events = sortedLiveEvents();
  return events.length ? events[events.length - 1] : null;
}

function liveTime(epoch) {
  return epoch ? `${relTime(epoch)} · ${new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "—";
}

function liveMetaChips(event) {
  const out = [chip(`${event.memory_count || 0} memories`, "kind-chip"), chip(`${event.item_count || 0} items`)];
  if (event.user) out.push(chip(`speaker · ${event.user}`));
  if (event.chat_id) out.push(chip(`chat · ${event.chat_id}`));
  if ((event.participants || []).length > 1) out.push(chip(`participants · ${event.participants.join(", ")}`));
  if (Object.keys(event.graphs || {}).length) out.push(chip("graph active", "kind-chip"));
  return out;
}

function renderLiveTimeline() {
  const box = $("live-timeline"); clear(box);
  const events = sortedLiveEvents().reverse();
  $("live-history-count").textContent = `${events.length} / 25`;
  for (const event of events) {
    const id = String(event.id);
    const button = el("button", {
      class: "live-timeline-item" + (id === state.live.selectedId ? " active" : "") +
        (id === String(latestLiveEvent() && latestLiveEvent().id) ? " newest" : ""),
      type: "button",
      onclick: () => selectLiveEvent(id),
    }, [
      el("div", { class: "live-timeline-top" }, [
        el("span", { class: "live-timeline-speaker" }, event.user || "unknown speaker"),
        el("span", {}, liveTime(event.created_at)),
      ]),
      el("div", { class: "live-timeline-message" }, event.message || "(empty message)"),
      el("div", { class: "live-timeline-tags" }, [
        el("span", {}, `${event.memory_count || 0} recalled`),
        Object.keys(event.graphs || {}).length ? el("span", { class: "kg-on" }, [icon("share-2"), " graph"]) : null,
      ]),
    ]);
    box.appendChild(button);
  }
}

function liveItemMeta(item) {
  if (!item || typeof item !== "object") return [];
  const meta = item.metadata || {};
  const fields = [];
  for (const key of ["user_id", "source", "node_id", "id", "recall_count"]) {
    if (meta[key] !== undefined && meta[key] !== null && meta[key] !== "") fields.push(`${key}: ${typeof meta[key] === "object" ? JSON.stringify(meta[key]) : meta[key]}`);
  }
  return fields;
}

function liveMemoryItems(memory) {
  let raw = memory && (memory.items ?? memory.results ?? memory.records);
  if (typeof raw === "string") {
    try { raw = JSON.parse(raw); } catch (error) { return []; }
  }
  if (Array.isArray(raw)) return raw;
  if (raw && typeof raw === "object") return Object.values(raw);
  return [];
}

function liveMemories(event) {
  let raw = event && event.memories;
  if (typeof raw === "string") {
    try { raw = JSON.parse(raw); } catch (error) { raw = null; }
  }
  const usable = Array.isArray(raw) || (raw && typeof raw === "object");
  if ((!usable || (Array.isArray(raw) && raw.length === 0)) && event) {
    raw = event.recalls ?? event.recalled_memories ?? event.memory_sections ?? event.sections ?? event.context;
    if (typeof raw === "string") {
      try { raw = JSON.parse(raw); } catch (error) { raw = null; }
    }
  }
  if (Array.isArray(raw)) return raw.filter((memory) => memory && typeof memory === "object");
  if (raw && typeof raw === "object") {
    return Object.entries(raw).map(([name, memory]) => {
      if (Array.isArray(memory)) return { name, title: name, items: memory };
      return {
        name,
        ...(memory && typeof memory === "object" ? memory : { body: String(memory || "") }),
      };
    });
  }
  return [];
}

function renderLiveRecallsInto(event, box, empty) {
  if (!box) return;
  clear(box);
  const memories = liveMemories(event);
  const embedded = box.id === "live-embed-recalls";
  const openMemory = embedded ? state.live.embedOpenMemory : state.live.openMemory;
  const firstRender = embedded ? !state.live.embedOpenInitialized : !state.live.openInitialized;
  // The full Live recall tab keeps its established expanded-first behaviour.
  // The narrow iframe drawer starts as a navigable list of collapsed groups.
  if (firstRender && !embedded) for (const memory of memories) openMemory.add(memory.name);
  if (empty) empty.classList.toggle("hidden", !!memories.length);
  for (const memory of memories) {
    const known = openMemory.has(memory.name);
    const items = liveMemoryItems(memory);
    const details = el("details", { class: "live-memory", open: (!embedded && firstRender) || known }, []);
    const summary = el("summary", { class: "live-memory-summary" }, [
      icon("chevron-down", "live-memory-chev"),
      el("span", { class: "live-memory-name", title: memory.name }, memory.title || memory.name),
      el("span", { class: "live-memory-scope" }, memory.scope || "memory"),
      el("span", { class: "live-memory-count" }, `${items.length} recalled`),
    ]);
    details.appendChild(summary);
    const body = el("div", { class: "live-memory-body" });
    const promptText = typeof memory.body === "string" ? memory.body
      : typeof memory.section === "string" ? memory.section : "";
    if (promptText) body.appendChild(el("pre", { class: "live-memory-prompt" }, promptText));
    for (const item of items) {
      const safeItem = item && typeof item === "object" ? item : { text: String(item ?? "") };
      const score = Number(safeItem.score);
      body.appendChild(el("article", { class: "live-item" }, [
        el("div", { class: "live-item-top" }, [
          el("span", { class: "live-item-kind" }, safeItem.kind || "recalled item"),
          Number.isFinite(score) ? el("span", { class: "live-item-score" }, `score ${score.toFixed(2)}`) : null,
        ]),
        el("div", { class: "live-item-text" }, safeItem.text || "(empty)"),
        liveItemMeta(safeItem).length ? el("div", { class: "live-item-meta" }, liveItemMeta(safeItem).map((value) => el("span", {}, value))) : null,
      ]));
    }
    details.appendChild(body);
    details.addEventListener("toggle", () => {
      if (details.open) openMemory.add(memory.name);
      else openMemory.delete(memory.name);
      if (embedded) state.live.embedOpenInitialized = true;
      else state.live.openInitialized = true;
    });
    box.appendChild(details);
  }
  if (embedded) state.live.embedOpenInitialized = true;
  else state.live.openInitialized = true;
}

function renderLiveEmbedRecalls(event) {
  renderLiveRecallsInto(event, $("live-embed-recalls"), $("live-embed-recalls-empty"));
}

function renderLiveRecalls(event) {
  renderLiveRecallsInto(event, $("live-recalls"), $("live-empty"));
  renderLiveEmbedRecalls(event);
}

function graphNodeText(node) {
  if (node.kind === "fact") return node.content || node.text || node.id;
  if (node.kind === "episode") return node.summary || node.text || node.id;
  if (node.kind === "person") return node.name || node.user_id || node.id;
  if (node.kind === "entity") return node.name || node.id;
  if (node.kind === "self") return "self";
  return node.text || node.id;
}

function renderLiveGraphList(data) {
  const box = $("live-graph-list"); clear(box);
  const nodes = (data.nodes || []).slice().sort((a, b) => (+b.activation || 0) - (+a.activation || 0)).slice(0, 15);
  for (const node of nodes) {
    const normalized = clamp(Number(node.activation_norm) || 0, 0, 1);
    box.appendChild(el("li", { onclick: () => state.live._gv && state.live._gv.focus(node.id) }, [
      el("div", {}, [
        el("span", { class: "gl-kind" }, GRAPH_KIND_LABEL[node.kind] || node.kind),
        " ", el("span", { class: "gl-act" }, `${(normalized * 100).toFixed(0)}%`),
      ]),
      el("div", { class: "gl-text" }, graphNodeText(node)),
    ]));
  }
}

function renderLiveGraph(event) {
  let rawGraphs = event.graphs ?? event.graph ?? {};
  if (typeof rawGraphs === "string") {
    try { rawGraphs = JSON.parse(rawGraphs); } catch (error) { rawGraphs = {}; }
  }
  const graphs = rawGraphs && !Array.isArray(rawGraphs) && Array.isArray(rawGraphs.nodes)
    ? { [event.user || "default"]: rawGraphs }
    : Array.isArray(rawGraphs)
    ? Object.fromEntries(rawGraphs.map((graph, index) => [graph.user || graph.user_id || String(index), graph]))
    : rawGraphs && typeof rawGraphs === "object" ? rawGraphs : {};
  const users = Object.keys(graphs);
  const shell = $("live-graph-shell"), empty = $("live-graph-empty"), select = $("live-graph-user");
  if (!users.length) {
    shell.classList.add("hidden"); empty.classList.remove("hidden");
    select.classList.add("hidden"); $("live-graph-meta").textContent = "No graph in this request";
    clear($("live-graph-list")); clear($("live-graph-detail")); $("live-graph-detail").classList.add("hidden");
    if (state.live._gv) {
      try { state.live._gv.destroy(); } catch (error) { /* best effort */ }
      state.live._gv = null;
    }
    return;
  }
  const preferred = state.live.graphUser && users.includes(state.live.graphUser)
    ? state.live.graphUser
    : event.user && users.includes(event.user) ? event.user : users[0];
  state.live.graphUser = preferred;
  clear(select);
  for (const user of users) select.appendChild(el("option", { value: user }, user));
  select.value = state.live.graphUser;
  select.classList.toggle("hidden", users.length < 2);
  const data = graphs[state.live.graphUser];
  shell.classList.remove("hidden"); empty.classList.add("hidden");
  $("live-graph-meta").textContent = `${(data.nodes || []).length} nodes · ${(data.edges || []).length} edges · ${state.live.graphUser}`;
  if (!state.live._gv) {
    state.live._gv = createGraphViz($("live-cy"), {
      onSelect: renderLiveNodeDetail,
      settings: GRAPH_SETTINGS,
      overlay: $("live-graph-overlay"),
      visibility: shell,
    });
  }
  // The live monitor explains the exact captured activation. Browse-view
  // filters such as "hide all facts/episodes" must not erase that trace.
  state.live._gv.setData(data);
  renderLiveGraphList(data);
  requestAnimationFrame(() => state.live._gv && state.live._gv.resize());
}

function updateLiveEmbedSummary(event) {
  const identity = $("live-embed-identity");
  const label = $("live-embed-recalls-label");
  const count = $("live-embed-recalls-count");
  const request = $("live-embed-request");
  const memoryCount = Number(event && event.memory_count) || liveMemories(event).length;
  const itemCount = Number(event && event.item_count) || liveMemories(event)
    .reduce((total, memory) => total + liveMemoryItems(memory).length, 0);
  if (identity) {
    const speaker = event && event.user ? event.user : "unknown";
    identity.textContent = `${event && event.character || state.character || "Character"} · ${speaker} · ${liveTime(event && event.created_at)}`;
  }
  if (label) label.textContent = memoryCount === 1 ? "1 recalled memory" : `${memoryCount} recalled memories`;
  if (count) {
    count.textContent = String(itemCount);
    count.title = `${itemCount} recalled item${itemCount === 1 ? "" : "s"}`;
  }
  if (request) {
    const message = event && event.message ? event.message : "(empty message)";
    request.textContent = `${event && event.user || "unknown"}: ${message}`;
  }
}

function renderLiveEvent(event) {
  if (!event) return;
  const request = $("live-request"); clear(request); request.classList.remove("hidden");
  request.appendChild(el("div", { class: "live-request-title" }, [
    el("strong", {}, `${event.character || state.character} · /context`),
    el("span", { class: "live-request-time" }, liveTime(event.created_at)),
  ]));
  request.appendChild(el("div", { class: "live-request-message" }, event.message || "(empty message)"));
  request.appendChild(el("div", { class: "live-request-meta" }, liveMetaChips(event)));
  const query = (event.query || []).map((part) => {
    const weight = Number(part.weight);
    const prefix = Number.isFinite(weight) && weight !== 1 ? `${weight.toFixed(2)}× ` : "";
    return `${prefix}${part.text || ""}`;
  }).filter(Boolean).join(" · ");
  if (query) request.appendChild(el("div", { class: "live-request-query" }, [
    el("b", {}, "weighted retrieval"), " ", query,
  ]));
  updateLiveEmbedSummary(event);
  renderLiveRecalls(event);
  try {
    renderLiveGraph(event);
  } catch (error) {
    // A plugin-specific graph payload must not hide the recalled memories.
    renderLiveGraph({ graphs: {} });
  }
}

function selectLiveEvent(id, { follow = false } = {}) {
  const event = state.live.events.get(String(id));
  if (!event) return;
  if (state.live.selectedId !== String(id)) state.live.graphUser = event.user || "";
  state.live.selectedId = String(id);
  state.live.follow = !!follow;
  $("live-new").classList.add("hidden");
  $("live-follow-label").textContent = state.live.follow ? "Following latest" : "Inspecting a past request";
  $("live-follow-label").classList.toggle("paused", !state.live.follow);
  renderLiveTimeline(); renderLiveEvent(event);
}

function renderLiveLatest() {
  const newest = latestLiveEvent();
  if (!newest) {
    $("live-request").classList.add("hidden"); $("live-empty").classList.remove("hidden");
    if ($("live-embed-recalls-empty")) $("live-embed-recalls-empty").classList.remove("hidden");
    renderLiveTimeline(); return;
  }
  selectLiveEvent(String(newest.id), { follow: true });
}

function wireLiveControls() {
  $("live-follow").addEventListener("click", () => {
    state.live.follow = true;
    renderLiveLatest();
  });
  $("live-new-btn").addEventListener("click", () => {
    state.live.follow = true;
    renderLiveLatest();
  });
  $("live-expand").addEventListener("click", () => {
    for (const event of state.live.events.values()) for (const memory of liveMemories(event)) state.live.openMemory.add(memory.name);
    const selected = state.live.events.get(state.live.selectedId); if (selected) renderLiveRecalls(selected);
  });
  $("live-collapse").addEventListener("click", () => {
    state.live.openMemory.clear(); state.live.openInitialized = true;
    const selected = state.live.events.get(state.live.selectedId); if (selected) renderLiveRecalls(selected);
  });
  $("live-graph-user").addEventListener("change", (event) => {
    state.live.graphUser = event.target.value;
    const selected = state.live.events.get(state.live.selectedId); if (selected) renderLiveGraph(selected);
  });
  $("live-zoom-in").addEventListener("click", () => state.live._gv && state.live._gv.zoomIn());
  $("live-zoom-out").addEventListener("click", () => state.live._gv && state.live._gv.zoomOut());
}

function setLiveEmbedDrawer(open) {
  const drawer = $("live-embed-drawer");
  const backdrop = $("live-embed-backdrop");
  const button = $("live-embed-recalls-btn");
  if (!drawer || !backdrop || !button) return;
  drawer.classList.toggle("hidden", !open);
  drawer.setAttribute("aria-hidden", open ? "false" : "true");
  backdrop.classList.toggle("hidden", !open);
  button.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) requestAnimationFrame(() => $("live-embed-recalls") && $("live-embed-recalls").focus());
}

function wireLiveEmbedControls() {
  updateLiveEmbedLink();
  if (!LIVE_GRAPH_EMBED) return;
  const toolbar = $("live-embed-toolbar");
  const participant = $("live-graph-user");
  const recallsButton = $("live-embed-recalls-btn");
  if (toolbar && participant && recallsButton) toolbar.insertBefore(participant, recallsButton);
  recallsButton.addEventListener("click", () => {
    setLiveEmbedDrawer(recallsButton.getAttribute("aria-expanded") !== "true");
  });
  $("live-embed-drawer-close").addEventListener("click", () => setLiveEmbedDrawer(false));
  $("live-embed-backdrop").addEventListener("click", () => setLiveEmbedDrawer(false));
  $("live-embed-expand").addEventListener("click", () => {
    const selected = state.live.events.get(state.live.selectedId);
    if (!selected) return;
    for (const memory of liveMemories(selected)) state.live.embedOpenMemory.add(memory.name);
    state.live.embedOpenInitialized = true;
    renderLiveEmbedRecalls(selected);
    $("live-embed-recalls").focus();
  });
  $("live-embed-collapse").addEventListener("click", () => {
    const selected = state.live.events.get(state.live.selectedId);
    state.live.embedOpenMemory.clear();
    state.live.embedOpenInitialized = true;
    if (selected) renderLiveEmbedRecalls(selected);
    $("live-embed-recalls").focus();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && recallsButton.getAttribute("aria-expanded") === "true") {
      setLiveEmbedDrawer(false);
      recallsButton.focus();
    }
  });
}

function activateLive() {
  state.live._active = true;
  connectLiveStream();
  renderLiveLatest();
}

function deactivateLive() {
  state.live._active = false;
  closeLiveStream();
  if (state.live._gv) {
    try { state.live._gv.destroy(); } catch (error) { /* best effort */ }
    state.live._gv = null;
  }
}

function resetCharacterView() {
  state.memory = null; state.page = 1; state.q = ""; state.user = "";
  state.editor = { editable: false, fields: [], description: "" };
  if (state.graph._gv) { try { state.graph._gv.destroy(); } catch (error) {} }
  // Controls and persisted graph settings outlive a character. In particular,
  // do not keep Full mode while silently resetting its limits.
  state.graph = { data: null, q: "", user: "", full: state.graph.full, _gv: null, _wired: graphControlsWired };
  state.live.events.clear(); state.live.selectedId = null; state.live.graphUser = "";
  state.live.openMemory.clear(); state.live.openInitialized = false;
  state.live.embedOpenMemory.clear(); state.live.embedOpenInitialized = false;
  if (state.live._gv) { try { state.live._gv.destroy(); } catch (error) {} state.live._gv = null; }
  $("q").value = "";
  $("graph-q").value = "";
}

// ---------------------------------------------------------------- API key dialog
function setApiKeyIndicator() {
  const btn = $("api-key-btn");
  if (btn) btn.classList.toggle("has-key", Boolean(storedApiKey()));
}

function openApiKeyDialog(message) {
  const overlay = $("api-key-overlay");
  if (!overlay) return;
  const hint = $("api-key-hint");
  if (hint) {
    hint.textContent = message ||
      "Set the API key this server requires (CM_API_KEY or --api-key on the server side).";
  }
  const input = $("api-key-input");
  if (input) {
    input.value = storedApiKey();
    input.focus();
  }
  overlay.classList.remove("hidden");
}

function closeApiKeyDialog() {
  const overlay = $("api-key-overlay");
  if (overlay) overlay.classList.add("hidden");
}

// Prompt for the key after a 401. Throttled: parallel failed fetches
// (overview + page + graph) must not stack prompts or reset the input.
let apiKeyPromptedAt = 0;
function promptApiKey() {
  const now = Date.now();
  if (now - apiKeyPromptedAt < 1500) return;
  apiKeyPromptedAt = now;
  openApiKeyDialog(
    "The server rejected the request: the API key is missing or invalid. " +
    "Set it below and the page will retry."
  );
}

function persistApiKey(key) {
  try {
    if (key) localStorage.setItem(API_KEY_STORAGE, key);
    else localStorage.removeItem(API_KEY_STORAGE);
  } catch (error) { /* storage may be disabled */ }
  closeApiKeyDialog();
  // Re-bootstrap every tab with the new key applied; the selected tab and
  // character survive via the ?tab= / ?character= URL params.
  location.reload();
}

function wireApiKeyDialog() {
  const btn = $("api-key-btn");
  if (btn) btn.addEventListener("click", () => openApiKeyDialog());
  const form = $("api-key-form");
  if (form) form.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("api-key-input");
    persistApiKey(((input && input.value) || "").trim());
  });
  const clear = $("api-key-clear");
  if (clear) clear.addEventListener("click", () => persistApiKey(""));
  const close = $("api-key-close");
  if (close) close.addEventListener("click", closeApiKeyDialog);
  const overlay = $("api-key-overlay");
  if (overlay) overlay.addEventListener("mousedown", (event) => {
    if (event.target === overlay) closeApiKeyDialog();
  });
  setApiKeyIndicator();
}

// ---------------------------------------------------------------- wire up
async function init() {
  wireThemeToggle();
  // The markup starts closed; this selectively restores a saved graph panel
  // state and settings before the controls are first wired/rendered.
  loadGraphUI();
  wireLiveControls();
  wireLiveEmbedControls();
  wireApiKeyDialog();
  $("refresh").addEventListener("click", () => loadOverview());
  $("add-memory").addEventListener("click", () => openMemoryEditor());
  $("memory-form").addEventListener("submit", saveMemory);
  $("memory-editor-close").addEventListener("click", closeMemoryEditor);
  $("memory-editor-cancel").addEventListener("click", closeMemoryEditor);
  $("memory-editor").addEventListener("mousedown", (event) => {
    if (event.target === $("memory-editor")) closeMemoryEditor();
  });
  $("character").addEventListener("change", (e) => {
    state.character = e.target.value;
    updateLiveEmbedLink();
    updateViewUrl();
    resetCharacterView();
    if (state.live._active) connectLiveStream();
    loadOverview();
  });
  $("memory-filter").addEventListener("input", renderSidebar);
  $("q").addEventListener("input", onSearchInput);
  $("q-clear").addEventListener("click", () => { $("q").value = ""; onSearchInput(); });
  $("user-filter").addEventListener("change", (e) => {
    state.user = e.target.value; state.page = 1; loadPage();
  });
  // keyboard: "/" focuses search, Esc clears.
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "SELECT") {
      e.preventDefault(); $("q").focus();
    } else if (e.key === "Escape") {
      if (!$("api-key-overlay").classList.contains("hidden")) closeApiKeyDialog();
      else if (!$("memory-editor").classList.contains("hidden")) closeMemoryEditor();
      else if ($("q").value) { $("q").value = ""; onSearchInput(); }
    }
  });
  // Refit the canvas renderer when the window resizes (the viz also watches
  // its own canvas via ResizeObserver, this just forces a re-measure).
  window.addEventListener("resize", () => {
    if ($("graph-view") && !$("graph-view").classList.contains("hidden") && state.graph._gv) state.graph._gv.resize();
    if ($("live-pane") && !$("live-pane").classList.contains("hidden") && state.live._gv) state.live._gv.resize();
  });
  try {
    const ok = await loadCharacters();
    if (ok) {
      updateLiveEmbedLink();
      if (LIVE_GRAPH_EMBED) {
        // The iframe route needs only character discovery + the /context SSE.
        // Avoid loading the hidden archive/configurator surfaces entirely.
        setTab("live", { history: false });
      } else {
        await loadOverview();
        const initialTab = ["browse", "configure", "live"].includes(new URLSearchParams(location.search).get("tab"))
          ? new URLSearchParams(location.search).get("tab") : "browse";
        setTab(initialTab, { history: false });
      }
    }
  } catch (e) {
    if (LIVE_GRAPH_EMBED) {
      liveSetConnection("error", "Connection failed");
      if ($("live-embed-identity")) $("live-embed-identity").textContent = e.message;
    } else showError(e.message);
  }
}

// ----------------------------------------------------------------- tab switching
// Share DOM helpers with config.js (the Configure tab) through `window.cmUtil`
// so the configurator never duplicates el()/clear()/$()/markdown()/getJSON().
// authHeaders/authQuery carry the optional API key on config.js's own fetches.
window.cmUtil = { $, el, clear, icon, setButtonContent, getJSON, markdown, esc, authHeaders, authQuery };

function setTab(name, { history = true } = {}) {
  const browse = name === "browse";
  const configure = name === "configure";
  const live = name === "live";
  $("tab-browse").classList.toggle("active", browse);
  $("tab-browse").setAttribute("aria-selected", browse ? "true" : "false");
  $("tab-configure").classList.toggle("active", configure);
  $("tab-configure").setAttribute("aria-selected", configure ? "true" : "false");
  $("tab-live").classList.toggle("active", live);
  $("tab-live").setAttribute("aria-selected", live ? "true" : "false");
  const main = document.querySelector(".app > .main");
  const cfg = $("cfg-pane");
  if (main) main.classList.toggle("hidden", !browse);
  if (cfg) cfg.classList.toggle("hidden", !configure);
  $("live-pane").classList.toggle("hidden", !live);
  if (live) activateLive(); else deactivateLive();
  if (configure && window.cmConfig && typeof window.cmConfig.activate === "function") {
    window.cmConfig.activate();
  }
  if (history) updateViewUrl();
}

function wireTabs() {
  $("tab-browse").addEventListener("click", () => setTab("browse"));
  $("tab-configure").addEventListener("click", () => setTab("configure"));
  $("tab-live").addEventListener("click", () => setTab("live"));
  window.addEventListener("popstate", async () => {
    const requestedCharacter = new URLSearchParams(location.search).get("character");
    if (requestedCharacter && state.characters.includes(requestedCharacter) && requestedCharacter !== state.character) {
      state.character = requestedCharacter;
      $("character").value = requestedCharacter;
      updateLiveEmbedLink();
      resetCharacterView();
      if (state.live._active) connectLiveStream();
      if (!LIVE_GRAPH_EMBED) await loadOverview();
    }
    const tab = new URLSearchParams(location.search).get("tab");
    setTab(LIVE_GRAPH_EMBED ? "live" : (["browse", "configure", "live"].includes(tab) ? tab : "browse"), { history: false });
  });
}
wireTabs();

// Convert the static <i data-lucide="…"> placeholders in index.html into
// inline SVG. Dynamic DOM uses cmUtil.icon() directly, so no re-scan needed.
if (window.lucide && typeof window.lucide.createIcons === "function") {
  window.lucide.createIcons();
}

init();
