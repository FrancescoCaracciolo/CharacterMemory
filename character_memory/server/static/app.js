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

const state = {
  characters: [],
  character: null,
  memories: [],            // overview entries
  memory: null,            // selected memory name
  page: 1,
  size: SIZE,
  user: "",
  q: "",
  cache: new Map(),        // `${memory}|${user}|${q}` -> Map(page -> data)
  inflight: null,          // AbortController for the active page fetch
  graph: { data: null, q: "", user: "", full: false, _gv: null, _wired: false }, // KG viz state
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
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}
function markdown(text) {
  const lines = esc(text).split("\n");
  let html = "";
  let inUl = false, inOl = false, inCode = false, codeBuf = [];
  const closeLists = () => { if (inUl) { html += "</ul>"; inUl = false; } if (inOl) { html += "</ol>"; inOl = false; } };
  for (const raw of lines) {
    const line = raw;
    if (line.trim().startsWith("```")) {
      if (inCode) { html += `<pre><code>${codeBuf.join("\n")}</code></pre>`; codeBuf = []; inCode = false; }
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
  if (inCode) html += `<pre><code>${codeBuf.join("\n")}</code></pre>`;
  closeLists();
  return html;
}

// ---------------------------------------------------------------- renderers (per memory type)
function wrapCard(children, rec) {
  const head = el("div", { class: "card-tags" }, []);
  const pill = scorePill(rec);
  const card = el("div", { class: "card" }, [
    el("div", { class: "card-head" }, [head, pill ? el("div", {}, [pill]) : null]),
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
  const shift = Number(f.emotional_shift || 0);
  addTag(card, chip(shift >= 0 ? "positive" : "negative", shift >= 0 ? "kind-chip" : "warn-chip"));
  card.appendChild(signedBar("emotional shift", shift));
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
  const r = await fetch(url, { headers: { Accept: "application/json" }, signal });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

async function loadCharacters() {
  const data = await getJSON(`${API}/`);
  state.characters = data.characters || [];
  const sel = $("character");
  clear(sel);
  for (const c of state.characters) sel.appendChild(el("option", { value: c }, c));
  if (!state.characters.length) {
    showError("No characters found. Start the server from the project root (so ./assets is visible), or set CM_ASSETS_DIR.");
    return false;
  }
  state.character = state.character && state.characters.includes(state.character)
    ? state.character : state.characters[0];
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

function cacheKey() { return `${state.memory}|${state.user || ""}|${state.q}`; }

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
  const url = `${API}/api/memories/${encodeURIComponent(state.character)}/${encodeURIComponent(state.memory)}?${params}`;

  try {
    const data = await getJSON(url, ac.signal);
    pages.set(state.page, data);
    if (state.inflight === ac) state.inflight = null;
    renderPage(data);
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
}

function renderPage(data) {
  renderHeader();
  renderUserFilter(data.users);
  renderExtra(data.extra);
  const searchBox = $("q").parentElement;
  searchBox.classList.toggle("searching", !!data.search);
  $("q-clear").classList.toggle("hidden", !data.q);

  // The knowledge graph renders its own force-directed visualization
  // instead of the card list. Toggle the two views.
  const isGraph = data.memory === "knowledge_graph" || data.kind === "graph";
  $("graph-view").classList.toggle("hidden", !isGraph);
  $("records").classList.toggle("hidden", isGraph);
  $("paginator").classList.toggle("hidden", isGraph);
  if (isGraph) {
    renderGraphView();
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
    el("div", { class: "big" }, "∅"),
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

  const btn = (label, target, opts = {}) => el("button", {
    class: "pg-btn" + (opts.active ? " active" : ""),
    disabled: opts.disabled || false,
    onclick: () => gotoPage(target),
  }, label);

  box.appendChild(btn("‹", page - 1, { disabled: page <= 1 }));
  // windowed page numbers
  const win = 2;
  const from = Math.max(1, page - win), to = Math.min(pages, page + win);
  if (from > 1) { box.appendChild(btn("1", 1)); if (from > 2) box.appendChild(el("span", { class: "pg-info" }, "…")); }
  for (let p = from; p <= to; p++) box.appendChild(btn(String(p), p, { active: p === page }));
  if (to < pages) { if (to < pages - 1) box.appendChild(el("span", { class: "pg-info" }, "…")); box.appendChild(btn(String(pages), pages)); }
  box.appendChild(btn("›", page + 1, { disabled: page >= pages }));
}

function showError(msg) {
  const s = $("status");
  s.textContent = "⚠ " + msg;
  s.classList.remove("hidden");
}

// ---------------------------------------------------------------- actions
function selectMemory(name) {
  if (state.memory === name) return;
  state.memory = name;
  state.page = 1;
  state.user = "";
  state.q = "";
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
const GRAPH_KIND_COLOR = {
  self: "#e8c35a", person: "#8b8df0", fact: "#5cc88f",
  episode: "#6fb7ef", entity: "#b07bf0",
};
const GRAPH_KIND_LABEL = {
  self: "self", person: "person", fact: "fact", episode: "episode", entity: "entity",
};
const GRAPH_EDGE_COLOR = {
  relation: "#8b8df0", fact: "#5cc88f", episode: "#6fb7ef",
  transition: "#b07bf0", co_occurrence: "#67708d",
};
const GRAPH_FONT = '"JetBrains Mono", "Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace';

function graphUrl() {
  const full = state.graph.full;
  const params = new URLSearchParams(
    full
      ? { limit: "4000", hops: "2", max_edges: "9000", include_co_occurrence: "1", full: "1", retrieve_k: "30" }
      : { limit: "50", hops: "1", max_edges: "300" }
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

// Build one renderer bound to a <canvas>. The returned object exposes a small
// imperative API used by the rest of the app (setData / fit / focus / freeze).
function createGraphViz(canvas, opts) {
  const onSelect = opts && opts.onSelect;
  const ctx = canvas.getContext("2d");
  const dpr = () => window.devicePixelRatio || 1;

  let nodes = [], edges = [], byId = new Map(), adj = new Map();
  const prevPos = new Map();           // id -> {x,y} across reloads (stable layout)
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
  const mouse = { x: 0, y: 0, inside: false };
  let lastT = performance.now();
  let raf = 0;
  let first = true;                     // auto-fit on the very first dataset

  // DOM tooltip living inside the overlay (sibling of the canvas).
  const tip = el("div", { class: "graph-tip-box" });
  document.getElementById("graph-overlay").appendChild(tip);

  // physics constants (world units)
  const REP = 3200, SPRING = 0.045, REST = 78, GRAV = 0.018, DAMP = 0.82, MAXV = 28;

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
    dimMode = data.mode === "full" && !!data.query;
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
        color: GRAPH_KIND_COLOR[n.kind] || "#9aa3bd",
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
        color: GRAPH_EDGE_COLOR[e.kind] || "#67708d",
        p: Math.random(), _hot: true,
      });
      adj.get(e.src).add(e.dst);
      adj.get(e.dst).add(e.src);
    }
    prevPos.clear();
    for (const n of next) prevPos.set(n.id, { x: n.x, y: n.y });
    nodes = next; edges = nextEdges;
    alpha = 1;
    if (first) { resize(); fitView(false); first = false; }
    else resize();
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
    for (const n of nodes) { minX = Math.min(minX, n.x); minY = Math.min(minY, n.y); maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y); }
    const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
    const pad = 70;
    const s = clamp(Math.min((view.w - 2 * pad) / w, (view.h - 2 * pad) / h), 0.2, 2.5);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const t = { scale: s, panX: view.w / 2 - cx * s, panY: view.h / 2 - cy * s };
    if (animate) Object.assign(target, t);
    else { Object.assign(view, t); Object.assign(target, t); }
  }

  function focusNode(id) {
    const n = byId.get(id); if (!n || !isFinite(n.x) || !isFinite(n.y)) return;
    selectedId = id;
    if (onSelect) onSelect(n);
    const w = view.w || canvas.clientWidth || 1;
    const h = view.h || canvas.clientHeight || 1;
    const s = Math.max(view.scale, 1.1);
    const t = { scale: s, panX: w / 2 - n.x * s, panY: h / 2 - n.y * s };
    Object.assign(target, t);
  }

  // -------------------------------------------------------------- physics
  function reheat(a) { if (!frozen) alpha = Math.max(alpha, a == null ? 0.7 : a); }

  function simulate() {
    if (alpha < 0.02 && !draggingId) return;
    const n = nodes.length;
    for (let i = 0; i < n; i++) { nodes[i].ax = 0; nodes[i].ay = 0; }
    // O(n^2) repulsion — fine for the subgraph sizes this view serves.
    for (let i = 0; i < n; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < n; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = dx * dx + dy * dy + 0.01; }
        const d = Math.sqrt(d2);
        const f = REP / d2, fx = f * dx / d, fy = f * dy / d;
        a.ax += fx; a.ay += fy; b.ax -= fx; b.ay -= fy;
      }
    }
    for (const e of edges) {
      const a = byId.get(e.src), b = byId.get(e.dst); if (!a || !b) continue;
      let dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = SPRING * (d - REST), fx = f * dx / d, fy = f * dy / d;
      a.ax += fx; a.ay += fy; b.ax -= fx; b.ay -= fy;
    }
    for (const a of nodes) {
      a.ax -= GRAV * a.x; a.ay -= GRAV * a.y;
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
        else if (a.retrieved || b.retrieved) { baseA = 0.08 + 0.18 * e.w; col = "#55607d"; }
        else { baseA = 0.04 + 0.06 * e.w; col = "#3a4257"; }
      }
      const aA = (activeId && !e._hot) ? baseA * 0.45 : baseA;
      ctx.strokeStyle = hexA(col, aA);
      ctx.lineWidth = (e._hot ? lw * 1.5 : lw);
      if (e.kind === "co_occurrence") ctx.setLineDash([4, 5]); else ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // flowing particles along edges
    ctx.globalCompositeOperation = "lighter";
    for (const e of edges) {
      const a = byId.get(e.src), b = byId.get(e.dst); if (!a || !b) continue;
      e.p = (e.p + dt * (0.05 + 0.25 * e.w)) % 1;
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

    // node halos (additive bloom) — skipped for grayed (non-retrieved) nodes
    const GRAYC = "#5b647e";
    for (const a of nodes) {
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
        ctx.lineWidth = 2.5; ctx.strokeStyle = `rgba(255,255,255,${na})`;
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + 2.5, 0, Math.PI * 2); ctx.stroke();
      } else if ((a._hot || (dimMode && a.retrieved)) && a._e > 0.05) {
        ctx.lineWidth = 2; ctx.strokeStyle = `rgba(255,255,255,${0.4 + 0.5 * a._e})`;
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + 2, 0, Math.PI * 2); ctx.stroke();
      }
      // Persistent ring on the clicked (selected) node so the click is obvious.
      if (selectedId === a.id) {
        ctx.lineWidth = 3; ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + 4, 0, Math.PI * 2); ctx.stroke();
      }
    }

    // labels: hidden by default (Obsidian-style dots), shown on hover or when
    // zoomed in past a threshold. Kept short so the canvas stays clean.
    const showAll = view.scale > 2.4;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.font = `10px ${GRAPH_FONT}`;
    for (const a of nodes) {
      const show = showAll || (activeId && a.id === activeId);
      if (!show) continue;
      const txt = a.label.length > 22 ? a.label.slice(0, 21) + "…" : a.label;
      ctx.fillStyle = `rgba(230,233,242,${activeId && a.id === activeId ? 0.95 : 0.6})`;
      ctx.shadowColor = "rgba(8,10,16,0.95)"; ctx.shadowBlur = 4;
      ctx.fillText(txt, a.x, a.y + a.r + 3);
      ctx.shadowBlur = 0;
    }
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
    if (!frozen) simulate();
    view.scale += (target.scale - view.scale) * 0.2;
    view.panX += (target.panX - view.panX) * 0.2;
    view.panY += (target.panY - view.panY) * 0.2;
    computeHighlights();
    draw(dt);
    raf = requestAnimationFrame(frame);
  }

  // -------------------------------------------------------------- picking
  function pick(mx, my) {
    let best = null, bestD = Infinity;
    for (const a of nodes) {
      const s = toScreen(a.x, a.y);
      const d = Math.hypot(s.x - mx, s.y - my);
      const hit = a.r * view.scale + 6;
      if (d <= hit && d < bestD) { bestD = d; best = a; }
    }
    return best;
  }

  // -------------------------------------------------------------- pointers
  function onDown(e) {
    const n = pick(e.offsetX, e.offsetY);
    if (n) {
      draggingId = n.id; down = { mode: "node" }; reheat(0.5);
      canvas.classList.add("grabbing");
    } else {
      panning = true; down = { mode: "pan", sx: e.offsetX, sy: e.offsetY, panX: view.panX, panY: view.panY };
      canvas.classList.add("grabbing");
    }
  }
  function onMove(e) {
    if (draggingId || panning) {
      const r = canvas.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      if (draggingId) {
        const n = byId.get(draggingId); const w = toWorld(mx, my);
        if (n) { n.x = w.x; n.y = w.y; n.vx = 0; n.vy = 0; prevPos.set(n.id, { x: n.x, y: n.y }); }
        reheat(0.4);
      } else if (panning) {
        view.panX = target.panX = down.panX + (mx - down.sx);
        view.panY = target.panY = down.panY + (my - down.sy);
      }
      return;
    }
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
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
  function onUp() {
    if (draggingId) reheat(0.3);
    draggingId = null; panning = false; down = null;
    canvas.classList.remove("grabbing");
  }
  function onLeave() {
    hoveredId = null; mouse.inside = false; tip.classList.remove("show");
    if (!panning && !draggingId) canvas.classList.remove("node-hover");
  }
  function onWheel(e) {
    e.preventDefault();
    const w = toWorld(e.offsetX, e.offsetY);
    const factor = Math.exp(-e.deltaY * 0.0015);
    const ns = clamp(view.scale * factor, 0.2, 4);
    view.scale = target.scale = ns;
    view.panX = target.panX = e.offsetX - w.x * ns;
    view.panY = target.panY = e.offsetY - w.y * ns;
  }
  function onClick(e) {
    const r = canvas.getBoundingClientRect();
    const n = pick(e.clientX - r.left, e.clientY - r.top);
    if (n) {
      focusNode(n.id);                       // select + centre + show detail
    } else {
      selectedId = null;                     // click empty space clears selection
      if (onSelect) onSelect(null);
    }
  }
  function onResize() { if (!document.getElementById("graph-view").classList.contains("hidden")) resize(); }

  canvas.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  canvas.addEventListener("mouseleave", onLeave);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("click", onClick);
  window.addEventListener("resize", onResize);
  let ro = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => resize());
    ro.observe(canvas);
  }

  resize();
  raf = requestAnimationFrame(frame);

  return {
    setData, resize, fit: () => fitView(true), focus: focusNode,
    reheat, freeze: () => { frozen = !frozen; return frozen; },
    isFrozen: () => frozen,
    destroy() {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("mouseleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("click", onClick);
      window.removeEventListener("resize", onResize);
      if (ro) ro.disconnect();
      tip.remove();
    },
  };
}

async function renderGraphView() {
  if (!state.graph._wired) wireGraphControls();
  $("graph-user").value = state.user || "";
  state.graph.user = state.user || "";
  $("graph-q").value = state.graph.q || "";
  $("graph-q-clear").classList.toggle("hidden", !state.graph.q);
  $("graph-full").classList.toggle("graph-freeze-on", !!state.graph.full);
  $("graph-full").textContent = state.graph.full ? "● full graph" : "◯ full graph";

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
  const trunc = data.truncated ? " (truncated)" : "";
  $("graph-meta").textContent =
    `${data.nodes.length} nodes · ${data.edges.length} edges${trunc}` +
    (data.mode === "full" ? " · full" : "") +
    (data.query ? ` · query “${data.query}”` : "");

  const canvas = $("cy");
  if (!state.graph._gv) state.graph._gv = createGraphViz(canvas, { onSelect: renderNodeDetail });
  state.graph._gv.resize();
  state.graph._gv.setData(data);
  requestAnimationFrame(() => state.graph._gv.resize());
  renderGraphList(data);
}

function wireGraphControls() {
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
    if (state.graph._gv) { state.graph._gv.reheat(1); }
  });
  $("graph-full").addEventListener("click", () => {
    state.graph.full = !state.graph.full;
    $("graph-full").classList.toggle("graph-freeze-on", state.graph.full);
    $("graph-full").textContent = state.graph.full ? "● full graph" : "◯ full graph";
    renderGraphView();
  });
  $("graph-fit").addEventListener("click", () => {
    if (state.graph._gv) state.graph._gv.fit();
  });
  $("graph-freeze").addEventListener("click", () => {
    if (!state.graph._gv) return;
    const f = state.graph._gv.freeze();
    $("graph-freeze").classList.toggle("graph-freeze-on", f);
    $("graph-freeze").textContent = f ? "▶ resume" : "❚❚ freeze";
  });
  // Space toggles the physics freeze while the graph is on screen.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if ($("graph-view").classList.contains("hidden")) return;
    e.preventDefault();
    if (!state.graph._gv) return;
    const f = state.graph._gv.freeze();
    $("graph-freeze").classList.toggle("graph-freeze-on", f);
    $("graph-freeze").textContent = f ? "▶ resume" : "❚❚ freeze";
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
  const box = $("graph-detail");
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
  if (raw.emotional_shift != null) box.appendChild(signedBar("emotional shift", Number(raw.emotional_shift)));
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

// ---------------------------------------------------------------- wire up
async function init() {
  $("refresh").addEventListener("click", () => loadOverview());
  $("character").addEventListener("change", (e) => {
    state.character = e.target.value;
    state.memory = null; state.page = 1; state.q = ""; state.user = "";
    // Tear down the previous renderer when switching characters.
    if (state.graph._gv) { try { state.graph._gv.destroy(); } catch (err) {} }
    state.graph = { data: null, q: "", user: "", _gv: null, _wired: false };
    $("q").value = "";
    $("graph-q").value = "";
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
      if ($("q").value) { $("q").value = ""; onSearchInput(); }
    }
  });
  // Refit the canvas renderer when the window resizes (the viz also watches
  // its own canvas via ResizeObserver, this just forces a re-measure).
  window.addEventListener("resize", () => {
    if (!$("graph-view") || $("graph-view").classList.contains("hidden")) return;
    if (state.graph._gv) state.graph._gv.resize();
  });
  try {
    const ok = await loadCharacters();
    if (ok) await loadOverview();
  } catch (e) {
    showError(e.message);
  }
}
init();
