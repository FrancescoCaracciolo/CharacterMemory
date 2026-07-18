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
  if (!state.characters.length) { showError("No characters found under assets/."); return; }
  state.character = state.character && state.characters.includes(state.character)
    ? state.character : state.characters[0];
  sel.value = state.character;
}

async function loadOverview() {
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

// ---------------------------------------------------------------- wire up
async function init() {
  $("refresh").addEventListener("click", () => loadOverview());
  $("character").addEventListener("change", (e) => {
    state.character = e.target.value;
    state.memory = null; state.page = 1; state.q = ""; state.user = "";
    $("q").value = "";
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
  try {
    await loadCharacters();
    await loadOverview();
  } catch (e) {
    showError(e.message);
  }
}
init();
