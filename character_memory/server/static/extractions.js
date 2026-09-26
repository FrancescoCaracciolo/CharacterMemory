/* Extractions — the "Extractions" tab.
 *
 * A changelog of what each memory-extraction pass learned:
 *   GET /api/extractions/{character}?limit&offset&user  -> newest-first runs + counts
 *   GET /api/extractions/{character}/{run_id}           -> one run with its full report
 *
 * The report (see character_memory/memory/extraction_log.py) describes new
 * rows per structured memory (with their dedup fate), before/after user
 * summaries, new knowledge-graph nodes with their connections, and mood /
 * relationship deltas. Each block renders through one function below.
 *
 * Shares DOM helpers with app.js through `window.cmUtil`.
 */

"use strict";
(function () {
const U = window.cmUtil;
const { $, el, clear, icon, getJSON, relTime, absTime } = U;

const API = "";
const PAGE = 40;
const POLL_MS = 6000;
const FOLLOW_KEY = "cm_extractions_follow";
const HIDE_EMPTY_KEY = "cm_extractions_hide_empty";
// Word-level LCS is O(n·m); longer summaries fall back to before/after only.
const MAX_DIFF_CELLS = 400000;

const MEMORY_META = {
  user_facts: { title: "User facts", icon: "user-round" },
  user_directives: { title: "User directives", icon: "list-checks" },
  episodic: { title: "Episodes", icon: "book-open" },
  conversation_events: { title: "Source conversations", icon: "messages-square" },
  heartbeat: { title: "Heartbeat journal", icon: "heart-pulse" },
  world: { title: "Private world", icon: "map-pin" },
  calendar: { title: "Calendar", icon: "calendar-days" },
};
const STATUS_META = {
  new: { label: "New", cls: "s-new", hint: "Stored as a new memory." },
  merged: { label: "Merged", cls: "s-merged", hint: "Folded into an existing memory by deduplication." },
  duplicate: { label: "Duplicate", cls: "s-dup", hint: "Already known; the new copy was dropped." },
  removed: { label: "Discarded", cls: "s-removed", hint: "Removed by deduplication or contradiction handling." },
};
// Legacy dedup reports short codes; LLM reconciliation reports prose.
const REASON_TEXT = {
  dup: "It says the same thing as an existing memory.",
  contradict: "It contradicted an existing memory, so the newer version replaced it.",
};
const DIRECTION_ICON = { out: "arrow-right", in: "arrow-left", both: "arrow-left-right" };
const DIRECTION_HINT = {
  out: "Link from the new node to this one",
  in: "Link from this node to the new one",
  both: "Two-way link",
};
const KNOWN_KINDS = new Set(["self", "person", "fact", "episode", "entity"]);

const S = {
  character: null,
  active: false,
  runs: [],
  total: 0,
  users: [],
  retention: 0,
  user: "",
  selected: null,
  cache: new Map(),
  follow: readFlag(FOLLOW_KEY, true),
  hideEmpty: readFlag(HIDE_EMPTY_KEY, false),
  timer: null,
  listSeq: 0,
  detailSeq: 0,
  loading: false,
  disabled: false,
  error: "",
  lastPoll: 0,
  wired: false,
};

function readFlag(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : raw === "true";
  } catch (error) { return fallback; }
}
function writeFlag(key, value) {
  try { localStorage.setItem(key, String(value)); } catch (error) { /* best effort */ }
}

// ---------------------------------------------------------------- formatting
const pretty = (s) => String(s || "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
const cap = (s) => { const t = pretty(s); return t ? t[0].toUpperCase() + t.slice(1) : t; };
const plural = (n, one, many) => `${n} ${n === 1 ? one : (many || one + "s")}`;
const fmtNum = (v, digits = 2) => (v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(digits));
const fmtDelta = (d) => `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(2)}`;
const changed = (d) => Math.abs(Number(d) || 0) > 1e-6;
const memoryTitle = (name) => (MEMORY_META[name] || {}).title || cap(name);
const memoryIcon = (name) => (MEMORY_META[name] || {}).icon || "brain";
const kindClass = (kind) => (KNOWN_KINDS.has(kind) ? kind : "default");

function fmtDuration(ms) {
  if (ms == null) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} m ${Math.round(s % 60)} s`;
}
function whenLabel(epoch) {
  if (!epoch) return "";
  const rel = relTime(epoch);
  return rel === "just now" ? rel : `${rel} ago`;
}

function isEmptyRun(run) {
  return !run.error && !((run.counts || {}).total > 0);
}

// Display names from the report's summaries, so a user id like "1234" reads
// as the person's name when the profile knows it.
function nameMap(report) {
  const names = new Map();
  for (const s of report.summaries || []) {
    if (s.after && s.after.name && s.after.name !== s.user_id) names.set(s.user_id, s.after.name);
  }
  return names;
}

// ---------------------------------------------------------------- data
function listUrl(offset, limit) {
  const qs = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (S.user) qs.set("user", S.user);
  return `${API}/api/extractions/${encodeURIComponent(S.character)}?${qs}`;
}

async function fetchList({ offset = 0, limit = PAGE } = {}) {
  return getJSON(listUrl(offset, limit));
}

function mergeRuns(incoming, { replace = false } = {}) {
  const byId = new Map((replace ? [] : S.runs).map((r) => [r.id, r]));
  for (const r of incoming) byId.set(r.id, r);
  S.runs = [...byId.values()].sort((a, b) => b.id - a.id);
}

async function reload({ keepSelection = true } = {}) {
  if (!S.character) { render(); return; }
  const seq = ++S.listSeq;
  S.loading = true;
  setStatus("connecting", "Loading…");
  setRefreshing(true);
  try {
    const data = await fetchList({ offset: 0, limit: Math.max(PAGE, keepSelection ? S.runs.length : 0) });
    if (seq !== S.listSeq) return;
    S.disabled = false;
    S.error = "";
    applyListMeta(data);
    mergeRuns(data.runs || [], { replace: true });
    const stillThere = keepSelection && S.runs.some((r) => r.id === S.selected);
    if (!stillThere || S.follow) S.selected = firstVisible() ? firstVisible().id : null;
  } catch (e) {
    if (seq !== S.listSeq) return;
    S.runs = []; S.total = 0; S.selected = null;
    if (e.status === 409) { S.disabled = true; S.error = ""; }
    else { S.disabled = false; S.error = e.message || String(e); }
  } finally {
    if (seq === S.listSeq) {
      S.loading = false;
      S.lastPoll = Date.now();
      setRefreshing(false);
      render();
    }
  }
}

async function poll() {
  if (!S.active || !S.character || S.loading || S.disabled || document.hidden) return;
  const seq = S.listSeq;
  try {
    const data = await fetchList({ offset: 0, limit: PAGE });
    if (seq !== S.listSeq || !S.active) return;
    const incoming = data.runs || [];
    const known = new Set(S.runs.map((r) => r.id));
    const fresh = incoming.filter((r) => !known.has(r.id));
    // No overlap means more than a page arrived between polls; "load older"
    // offsets would skip rows, so start over from the top.
    const gap = S.runs.length && incoming.length && fresh.length === incoming.length;
    const previous = S.selected;
    applyListMeta(data);
    mergeRuns(incoming, { replace: gap });
    S.error = "";
    S.lastPoll = Date.now();
    if ((fresh.length && S.follow) || (gap && !S.runs.some((r) => r.id === S.selected))) {
      const top = firstVisible();
      S.selected = top ? top.id : null;
    }
    renderStatus();
    if (fresh.length || gap) renderList(); else renderRunTimes();
    if (S.selected !== previous) renderDetail();
  } catch (e) {
    if (seq !== S.listSeq) return;
    if (e.status === 409) { S.disabled = true; render(); return; }
    setStatus("error", `Update failed: ${e.message}`);
  }
}

async function loadMore() {
  if (!S.character || S.loading) return;
  const seq = S.listSeq;
  const btn = $("ex-more");
  btn.disabled = true;
  try {
    const data = await fetchList({ offset: S.runs.length, limit: PAGE });
    if (seq !== S.listSeq) return;
    applyListMeta(data);
    mergeRuns(data.runs || []);
    renderList();
    renderStatus();
  } catch (e) {
    setStatus("error", e.message);
  } finally {
    btn.disabled = false;
  }
}

function applyListMeta(data) {
  S.total = Number(data.total) || 0;
  S.retention = Number(data.retention) || 0;
  S.users = Array.isArray(data.users) ? data.users : [];
  renderUserFilter();
}

async function loadRun(id) {
  const key = `${S.character}|${id}`;
  if (S.cache.has(key)) return S.cache.get(key);
  const run = await getJSON(`${API}/api/extractions/${encodeURIComponent(S.character)}/${id}`);
  // Runs are immutable once recorded; bound the cache so a long session
  // clicking through history doesn't hold every report.
  if (S.cache.size > 60) S.cache.delete(S.cache.keys().next().value);
  S.cache.set(key, run);
  return run;
}

// ---------------------------------------------------------------- selection
function visibleRuns() {
  return S.hideEmpty ? S.runs.filter((r) => !isEmptyRun(r) || r.id === S.selected) : S.runs;
}
function firstVisible() {
  return S.hideEmpty ? S.runs.find((r) => !isEmptyRun(r)) || null : S.runs[0] || null;
}

function selectRun(id, { user = false } = {}) {
  S.selected = id;
  if (user) {
    const top = firstVisible();
    const follow = !!top && top.id === id;
    if (follow !== S.follow) setFollow(follow);
  }
  renderList();
  renderDetail();
  if (user && matchMedia("(max-width: 760px)").matches) {
    $("ex-detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function setFollow(on) {
  S.follow = on;
  writeFlag(FOLLOW_KEY, on);
  const btn = $("ex-follow");
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  btn.title = on
    ? "Following: the newest pass opens automatically. Click to pause."
    : "Paused: click to jump to the newest pass and keep following.";
  const label = btn.querySelector(".btn-label");
  if (label) label.textContent = on ? "Following latest" : "Follow latest";
}

// ---------------------------------------------------------------- chrome
function setStatus(kind, text) {
  $("ex-status-dot").className = `live-status-dot ${kind}`;
  $("ex-status").textContent = text;
}
function setRefreshing(on) {
  $("ex-refresh").classList.toggle("spinning", on);
}

function renderStatus() {
  if (!S.character) return setStatus("error", "No character selected");
  if (S.loading) return setStatus("connecting", "Loading…");
  if (S.disabled) return setStatus("error", "Extraction log disabled");
  if (S.error) return setStatus("error", S.error);
  const latest = S.runs[0];
  const parts = [plural(S.total, "pass", "passes")];
  if (latest) parts.push(`latest ${whenLabel(latest.created_at)}`);
  setStatus("connected", parts.join(" · "));
}

function renderUserFilter() {
  const sel = $("ex-user");
  const users = [...new Set([...(S.users || []), ...(S.user ? [S.user] : [])])].sort();
  const wanted = ["", ...users].join("\u0000");
  if (sel.dataset.options !== wanted) {
    clear(sel);
    sel.appendChild(el("option", { value: "" }, "All speakers"));
    for (const u of users) sel.appendChild(el("option", { value: u }, u));
    sel.dataset.options = wanted;
  }
  sel.value = S.user;
  sel.classList.toggle("hidden", users.length < 2 && !S.user);
}

function render() {
  renderStatus();
  renderList();
  renderDetail();
}

// ---------------------------------------------------------------- run list
function runTags(run) {
  const c = run.counts || {};
  const tags = [];
  const tag = (cls, iconName, text, title) =>
    tags.push(el("span", { class: `ex-tag ${cls}`, title }, [icon(iconName), text]));
  if (run.error) tag("t-error", "triangle-alert", "error", run.error);
  if (c.new_items) tag("t-mem", "brain", `+${c.new_items}`, plural(c.new_items, "new memory", "new memories"));
  const updated = Object.values(c.memories || {}).reduce((n, m) => n + (m.updated || 0), 0);
  if (updated) tag("t-mem", "git-merge", `${updated}`, plural(updated, "existing memory updated", "existing memories updated"));
  if (c.summaries) tag("t-profile", "id-card", c.summaries > 1 ? `${c.summaries}` : "profile", plural(c.summaries, "profile summary updated", "profile summaries updated"));
  if (c.kg_nodes || c.kg_links) {
    tag("t-kg", "share-2", c.kg_nodes ? `+${c.kg_nodes}` : `${c.kg_links} links`,
      `${plural(c.kg_nodes || 0, "new graph node")}, ${plural(c.kg_links || 0, "new link")}`);
  }
  if (c.emotion_users || c.mood_changed) {
    tag("t-emotion", "heart-pulse", c.mood_changed && c.emotion_users ? "mood + bond" : c.mood_changed ? "mood" : "bond",
      [c.mood_changed ? "Mood shifted" : "", c.emotion_users ? plural(c.emotion_users, "relationship changed", "relationships changed") : ""].filter(Boolean).join(" · "));
  }
  if (!tags.length) tags.push(el("span", { class: "ex-tag t-none" }, "no changes"));
  return tags;
}

function renderList() {
  const box = $("ex-runs");
  const hadFocus = box.contains(document.activeElement);
  clear(box);
  const runs = visibleRuns();
  const hidden = S.runs.length - runs.length;
  $("ex-count").textContent = S.hideEmpty && hidden
    ? `${runs.length} shown · ${S.total}`
    : `${S.runs.length < S.total ? `${S.runs.length} of ` : ""}${S.total}`;
  const newest = S.runs[0] ? S.runs[0].id : null;
  for (const run of runs) {
    const who = (run.participants && run.participants.length ? run.participants : [run.user_id]).filter(Boolean);
    const btn = el("button", {
      type: "button",
      class: "live-timeline-item ex-run" + (run.id === S.selected ? " active" : "")
        + (run.id === newest ? " newest" : "") + (isEmptyRun(run) ? " is-empty" : ""),
      "data-id": String(run.id),
      "aria-current": run.id === S.selected ? "true" : null,
      title: absTime(run.created_at),
      onclick: () => selectRun(run.id, { user: true }),
    }, [
      el("div", { class: "live-timeline-top" }, [
        el("span", { class: "live-timeline-speaker" }, who.join(", ") || "unknown"),
        el("span", { class: "ex-run-time", "data-epoch": String(run.created_at || "") }, whenLabel(run.created_at)),
      ]),
      run.preview ? el("div", { class: "live-timeline-message" }, run.preview) : null,
      el("div", { class: "ex-run-tags" }, runTags(run)),
    ]);
    box.appendChild(btn);
  }
  if (!runs.length && !S.loading) {
    box.appendChild(el("div", { class: "ex-list-empty" },
      S.runs.length ? "Every loaded pass is empty." : S.disabled ? "Logging is off." : S.error ? "Could not load passes." : "No passes yet."));
  }
  $("ex-more").classList.toggle("hidden", S.runs.length >= S.total);
  if (hadFocus) {
    const current = box.querySelector(".ex-run.active") || box.querySelector(".ex-run");
    if (current) current.focus({ preventScroll: true });
  }
}

function renderRunTimes() {
  document.querySelectorAll("#ex-runs .ex-run-time").forEach((n) => {
    const epoch = Number(n.dataset.epoch);
    if (epoch) n.textContent = whenLabel(epoch);
  });
}

// ---------------------------------------------------------------- detail
function emptyState(iconName, title, text, extra) {
  return el("div", { class: "live-empty ex-empty" }, [
    el("span", { class: "live-empty-mark" }, [icon(iconName)]),
    el("strong", {}, title),
    el("span", {}, text),
    extra || null,
  ]);
}

function detailPlaceholder() {
  if (!S.character) {
    return emptyState("sparkles", "No character selected", "Pick a character to see what it has been learning.");
  }
  if (S.disabled) {
    return emptyState("circle-slash", "Extraction log is turned off",
      "This character does not keep a changelog of its extraction passes.",
      el("span", {}, ["Set ", el("code", {}, "memory.extraction_log_limit"), " above 0 in its config to enable it."]));
  }
  if (S.error && !S.runs.length) return emptyState("triangle-alert", "Could not load extraction passes", S.error);
  if (S.selected != null) return null;
  if (S.loading) return emptyState("sparkles", "Loading…", "Fetching extraction passes.");
  return S.runs.length
    ? emptyState("sparkles", "Nothing to show", "Every loaded pass is empty. Turn off “Hide empty passes” to see them.")
    : emptyState("sparkles", "No extraction passes yet",
      "A pass runs after the character has chatted for a few turns. Each one will appear here with what it learned.");
}

async function renderDetail() {
  const box = $("ex-detail");
  const seq = ++S.detailSeq;
  const placeholder = detailPlaceholder();
  if (placeholder) {
    clear(box);
    box.dataset.shown = "";
    box.classList.remove("is-loading");
    box.appendChild(placeholder);
    return;
  }
  const id = S.selected;
  const cached = S.cache.get(`${S.character}|${id}`);
  if (!cached) {
    box.classList.add("is-loading");
    if (!box.firstChild) box.appendChild(emptyState("sparkles", "Loading pass…", ""));
  }
  try {
    const run = cached || await loadRun(id);
    if (seq !== S.detailSeq) return;
    const keepScroll = box.dataset.shown === `${S.character}|${id}` ? box.scrollTop : 0;
    clear(box);
    box.appendChild(renderReport(run));
    box.dataset.shown = `${S.character}|${id}`;
    box.scrollTop = keepScroll;
  } catch (e) {
    if (seq !== S.detailSeq) return;
    clear(box);
    box.dataset.shown = "";
    box.appendChild(emptyState("triangle-alert", `Could not load pass #${id}`, e.message || String(e)));
  } finally {
    if (seq === S.detailSeq) box.classList.remove("is-loading");
  }
}

function renderReport(run) {
  const report = run.report || {};
  const names = nameMap(report);
  const c = run.counts || {};
  const multi = (run.participants || []).length > 1;
  const ctx = { names, multi };

  const sections = [];
  const memSection = renderMemories(report.memories || {}, ctx);
  if (memSection) sections.push(memSection);
  if ((report.summaries || []).length) sections.push(renderSummaries(report.summaries, ctx));
  const kgSection = renderGraph(report.knowledge_graph || {}, ctx);
  if (kgSection) sections.push(kgSection);
  const emoSection = renderEmotion(report.emotion || {}, ctx);
  if (emoSection) sections.push(emoSection);

  const updated = Object.values(c.memories || {}).reduce((n, m) => n + (m.updated || 0), 0);
  const kgNodes = (report.knowledge_graph || {}).new_nodes_total ?? c.kg_nodes ?? 0;
  const tiles = [
    tile("brain", "New memories", c.new_items || 0, updated ? `${updated} updated` : "", "ex-sec-memories"),
    tile("id-card", "Profile updates", c.summaries || 0, "", "ex-sec-profile"),
    tile("share-2", "Graph nodes", kgNodes, c.kg_links ? plural(c.kg_links, "new link") : "", "ex-sec-graph"),
    tile("heart-pulse", "Emotion shifts", (c.emotion_users || 0) + (c.mood_changed ? 1 : 0),
      c.mood_changed ? "mood changed" : "", "ex-sec-emotion"),
  ];

  const who = (run.participants && run.participants.length ? run.participants : [run.user_id]).filter(Boolean);
  const meta = [
    chip("users", who.map((u) => names.get(u) || u).join(", ") || "unknown", "Speakers in this pass"),
    chip("messages-square", plural((report.messages || []).length, "message"), "Messages the extractor read"),
    run.duration_ms != null ? chip("clock", fmtDuration(run.duration_ms), "Time the pass took") : null,
    run.chat_id ? chip("tag", `chat ${String(run.chat_id).slice(0, 8)}`, `Chat ${run.chat_id}`) : null,
  ];

  return el("article", { class: "ex-report" }, [
    el("header", { class: "ex-report-head" }, [
      el("div", { class: "ex-report-title" }, [
        el("h3", {}, `Pass #${run.id}`),
        el("span", { class: "ex-report-time", title: absTime(run.created_at) },
          `${whenLabel(run.created_at)} · ${absTime(run.created_at)}`),
      ]),
      el("div", { class: "ex-report-meta" }, meta),
    ]),
    run.error || report.error ? el("div", { class: "ex-callout ex-callout-error", role: "alert" }, [
      icon("triangle-alert"),
      el("div", {}, [el("strong", {}, "The extractor reported an error"), el("span", {}, run.error || report.error)]),
    ]) : null,
    el("div", { class: "ex-tiles" }, tiles),
    sections.length ? null : el("div", { class: "ex-callout ex-callout-quiet" }, [
      icon("circle-dashed"),
      el("div", {}, [
        el("strong", {}, "Nothing new this time"),
        el("span", {}, "The extractor read the conversation below and found nothing worth remembering or changing."),
      ]),
    ]),
    ...sections,
    renderConversation(report.messages || [], ctx, !sections.length),
  ]);
}

function tile(iconName, label, value, sub, target) {
  const on = value > 0;
  return el(on ? "button" : "div", {
    type: on ? "button" : null,
    class: "ex-tile" + (on ? " on" : ""),
    title: on ? `Jump to ${label.toLowerCase()}` : `No ${label.toLowerCase()} in this pass`,
    onclick: on ? () => { const t = document.getElementById(target); if (t) t.scrollIntoView({ behavior: "smooth", block: "start" }); } : null,
  }, [
    el("span", { class: "ex-tile-icon" }, [icon(iconName)]),
    el("span", { class: "ex-tile-value" }, String(value)),
    el("span", { class: "ex-tile-label" }, label),
    sub ? el("span", { class: "ex-tile-sub" }, sub) : null,
  ]);
}

function chip(iconName, text, title) {
  return el("span", { class: "chip ex-chip", title }, [icon(iconName), text]);
}

function section(id, iconName, title, count, body, { hint } = {}) {
  return el("section", { class: "ex-section", id }, [
    el("header", { class: "ex-section-head" }, [
      el("span", { class: "ex-section-icon" }, [icon(iconName)]),
      el("h4", {}, title),
      count != null ? el("span", { class: "live-memory-count" }, String(count)) : null,
      hint ? el("span", { class: "ex-section-hint" }, hint) : null,
    ]),
    el("div", { class: "ex-section-body" }, body),
  ]);
}

function userChip(uid, ctx) {
  if (!uid) return null;
  const name = ctx.names.get(uid);
  return el("span", { class: "chip ex-user-chip", title: name ? `User ${uid}` : "User" }, [icon("user-round"), name || uid]);
}

// ---- new memories -----------------------------------------------------
function renderMemories(memories, ctx) {
  const groups = Object.entries(memories).filter(([, b]) => (b.items || []).length || (b.updated || []).length);
  if (!groups.length) return null;
  let total = 0;
  const body = groups.map(([name, block]) => {
    const items = [...(block.items || [])].sort((a, b) => (a.status === "new" ? 0 : 1) - (b.status === "new" ? 0 : 1));
    const updated = block.updated || [];
    const tally = {};
    for (const it of items) tally[it.status] = (tally[it.status] || 0) + 1;
    total += tally.new || 0;
    const summary = [
      tally.new ? `${tally.new} new` : "",
      tally.merged ? `${tally.merged} merged` : "",
      tally.duplicate ? `${tally.duplicate} duplicate` : "",
      tally.removed ? `${tally.removed} discarded` : "",
      updated.length ? `${updated.length} updated` : "",
    ].filter(Boolean).join(" · ");
    return el("div", { class: "ex-mem-group" }, [
      el("div", { class: "ex-mem-group-head" }, [
        icon(memoryIcon(name)),
        el("strong", {}, memoryTitle(name)),
        el("span", { class: "ex-mem-group-tally" }, summary),
      ]),
      ...items.map((it) => memoryItem(it, ctx)),
      ...updated.map((it) => memoryItem({ ...it, status: "updated" }, ctx)),
    ]);
  });
  return section("ex-sec-memories", "brain", "New memories", total, body,
    { hint: "What the extractor wrote, and what deduplication did with it." });
}

function memoryItem(it, ctx) {
  const status = it.status === "updated"
    ? { label: "Updated", cls: "s-updated", hint: "An existing memory was revised by this pass." }
    : STATUS_META[it.status] || STATUS_META.new;
  const folded = it.status === "merged" || it.status === "duplicate";
  return el("div", { class: `ex-mem ${status.cls}` }, [
    el("div", { class: "ex-mem-top" }, [
      el("span", { class: `ex-status ${status.cls}`, title: status.hint }, status.label),
      it.id != null ? el("span", { class: "ex-mem-id" }, `#${it.id}`) : null,
      ctx.multi ? userChip(it.user_id, ctx) : null,
      it.importance != null ? importance(it.importance) : null,
    ]),
    el("div", { class: "ex-mem-text" }, it.text || "(empty)"),
    folded && (it.survivor_text || it.survivor_id != null) ? el("div", { class: "ex-mem-survivor" }, [
      icon("git-merge"),
      el("div", {}, [
        el("span", { class: "ex-mem-survivor-label" },
          `${it.status === "merged" ? "Merged into" : "Duplicate of"} ${it.survivor_id != null ? `#${it.survivor_id}` : "an existing memory"}`),
        it.survivor_text ? el("span", { class: "ex-mem-survivor-text" }, it.survivor_text) : null,
      ]),
    ]) : null,
    it.reason ? el("div", { class: "ex-mem-reason" }, [el("b", {}, "Why: "), REASON_TEXT[it.reason] || it.reason]) : null,
    extras(it.extra),
  ]);
}

function importance(v) {
  const pct = Math.max(0, Math.min(1, Number(v) || 0));
  return el("span", { class: "ex-importance", title: `Importance ${fmtNum(v)}` }, [
    el("span", { class: "ex-importance-track" }, [el("span", { class: "ex-importance-fill", style: `width:${(pct * 100).toFixed(0)}%` })]),
    el("span", {}, fmtNum(v)),
  ]);
}

function extras(extra) {
  const parts = [];
  for (const [k, v] of Object.entries(extra || {})) {
    let text;
    if (Array.isArray(v)) text = v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ");
    // Numeric maps (e.g. an episode's emotional impact) are mostly zeros.
    else if (v && typeof v === "object") text = Object.entries(v).filter(([, n]) => changed(n)).map(([dk, n]) => `${dk} ${fmtNum(n)}`).join(", ");
    else if (typeof v === "number" && /_at$/.test(k)) text = absTime(v);
    else if (typeof v === "number") text = Number.isInteger(v) ? String(v) : fmtNum(v);
    else if (typeof v === "boolean") text = v ? "yes" : "no";
    else text = String(v);
    if (text) parts.push(el("span", {}, [el("b", {}, pretty(k)), " " + text]));
  }
  return parts.length ? el("div", { class: "ex-mem-extra" }, parts) : null;
}

// ---- profile summaries ------------------------------------------------
function renderSummaries(summaries, ctx) {
  return section("ex-sec-profile", "id-card", "Profile updates", summaries.length,
    summaries.map((s) => summaryCard(s, ctx)),
    { hint: "The rolling summary of who each person is." });
}

function summaryCard(s, ctx) {
  const before = s.before || null;
  const after = s.after || {};
  const isNew = !before;
  const nameChanged = before && before.name !== after.name;
  const textChanged = !before || before.summary !== after.summary;
  const body = el("div", { class: "ex-summary-body" });
  const views = {
    changes: () => diffView(before ? before.summary : "", after.summary || ""),
    before: () => el("p", { class: "ex-summary-text" }, before && before.summary ? before.summary : "(empty)"),
    after: () => el("p", { class: "ex-summary-text" }, after.summary || "(empty)"),
  };
  let current = isNew ? "after" : "changes";
  const seg = el("div", { class: "ex-seg", role: "group", "aria-label": "Summary view" });
  const show = (name) => {
    current = name;
    clear(body);
    body.appendChild(views[name]());
    for (const b of seg.children) b.setAttribute("aria-pressed", b.dataset.view === name ? "true" : "false");
  };
  if (!isNew && textChanged) {
    for (const [name, label] of [["changes", "Changes"], ["before", "Before"], ["after", "After"]]) {
      seg.appendChild(el("button", { type: "button", class: "ex-seg-btn", "data-view": name, onclick: () => show(name) }, label));
    }
  }
  show(current);
  return el("div", { class: "ex-summary" }, [
    el("div", { class: "ex-summary-head" }, [
      el("strong", { class: "ex-summary-name" }, after.name || s.user_id),
      after.name && after.name !== s.user_id ? el("span", { class: "ex-mem-id" }, s.user_id) : null,
      isNew ? el("span", { class: "ex-status s-new" }, "New profile") : null,
      nameChanged ? el("span", { class: "ex-status s-updated", title: `Was “${before.name}”` }, `Renamed from ${before.name}`) : null,
      seg.children.length ? seg : null,
    ]),
    (s.aliases_added || []).length ? el("div", { class: "ex-aliases" }, [
      el("span", { class: "ex-aliases-label" }, "New aliases"),
      ...s.aliases_added.map((a) => el("span", { class: "chip ex-alias" }, [icon("plus"), a])),
    ]) : null,
    textChanged ? body : el("p", { class: "ex-summary-text muted" }, "Summary text unchanged."),
  ]);
}

function tokenize(s) { return String(s || "").match(/\s+|[^\s]+/g) || []; }

function wordDiff(a, b) {
  const A = tokenize(a), B = tokenize(b);
  let start = 0;
  while (start < A.length && start < B.length && A[start] === B[start]) start++;
  let endA = A.length, endB = B.length;
  while (endA > start && endB > start && A[endA - 1] === B[endB - 1]) { endA--; endB--; }
  const a2 = A.slice(start, endA), b2 = B.slice(start, endB);
  const n = a2.length, m = b2.length;
  if (n * m > MAX_DIFF_CELLS) return null;
  const dp = new Uint32Array((n + 1) * (m + 1));
  const at = (i, j) => i * (m + 1) + j;
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[at(i, j)] = a2[i] === b2[j] ? dp[at(i + 1, j + 1)] + 1 : Math.max(dp[at(i + 1, j)], dp[at(i, j + 1)]);
    }
  }
  const ops = [];
  const push = (type, text) => {
    const last = ops[ops.length - 1];
    if (last && last.type === type) last.text += text; else ops.push({ type, text });
  };
  if (start) push("same", A.slice(0, start).join(""));
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a2[i] === b2[j]) { push("same", a2[i]); i++; j++; }
    else if (dp[at(i + 1, j)] >= dp[at(i, j + 1)]) { push("del", a2[i]); i++; }
    else { push("add", b2[j]); j++; }
  }
  while (i < n) push("del", a2[i++]);
  while (j < m) push("add", b2[j++]);
  if (endA < A.length) push("same", A.slice(endA).join(""));
  return ops;
}

function diffView(before, after) {
  const ops = wordDiff(before, after);
  if (!ops) {
    return el("div", { class: "ex-diff-split" }, [
      el("div", {}, [el("span", { class: "ex-diff-label" }, "Before"), el("p", { class: "ex-summary-text" }, before || "(empty)")]),
      el("div", {}, [el("span", { class: "ex-diff-label" }, "After"), el("p", { class: "ex-summary-text" }, after || "(empty)")]),
    ]);
  }
  return el("p", { class: "ex-summary-text ex-diff" }, ops.map((op) =>
    op.type === "same" ? document.createTextNode(op.text)
      : el(op.type === "add" ? "ins" : "del", { class: op.type === "add" ? "ex-ins" : "ex-del" }, op.text)));
}

// ---- knowledge graph --------------------------------------------------
function renderGraph(kg, ctx) {
  const nodes = kg.new_nodes || [];
  const links = kg.new_links || [];
  if (!kg.enabled || (!nodes.length && !links.length && !kg.removed_nodes)) return null;
  const body = [];
  if (nodes.length) {
    body.push(el("div", { class: "ex-nodes" }, nodes.map((n) => nodeCard(n, ctx))));
    const more = (kg.new_nodes_total || nodes.length) - nodes.length;
    if (more > 0) body.push(el("p", { class: "ex-more-note" }, `…and ${plural(more, "more new node")} not listed.`));
  }
  if (links.length) {
    body.push(el("div", { class: "ex-links" }, [
      el("div", { class: "ex-sub-label" }, "New links between existing nodes"),
      ...links.map((l) => el("div", { class: "ex-link" }, [
        nodeRef(l.src),
        el("span", { class: "ex-link-edge", title: `weight ${fmtNum(l.weight)}` }, [icon("arrow-right"), pretty(l.kind)]),
        nodeRef(l.dst),
      ])),
    ]));
  }
  if (kg.removed_nodes) {
    body.push(el("p", { class: "ex-more-note" }, `${plural(kg.removed_nodes, "node was", "nodes were")} removed from the graph during this pass.`));
  }
  const legend = el("div", { class: "graph-legend ex-legend" },
    ["self", "person", "fact", "episode", "entity"].map((k) => el("span", {}, [el("span", { class: `dot dot-${k}` }), " " + k])));
  body.push(legend);
  return section("ex-sec-graph", "share-2", "Knowledge graph", kg.new_nodes_total ?? nodes.length, body,
    { hint: `New nodes and what they connect to${kg.new_edge_count ? ` · ${plural(kg.new_edge_count, "new link")} in total` : ""}.` });
}

function kindDot(kind) {
  return el("span", { class: `ex-dot dot-${kindClass(kind)}`, title: kind });
}

function nodeRef(n) {
  return el("span", { class: "ex-node-ref", title: n.text && n.text !== n.label ? n.text : `${n.kind} node` }, [
    kindDot(n.kind),
    el("span", { class: "ex-node-ref-label" }, n.label || n.id),
  ]);
}

function nodeCard(n, ctx) {
  const neighbors = n.neighbors || [];
  const kindLabel = n.kind_label || n.type || "";
  const hiddenCount = (n.neighbor_total || neighbors.length) - neighbors.length;
  return el("div", { class: `ex-node kind-${kindClass(n.kind)}` }, [
    el("div", { class: "ex-node-head" }, [
      kindDot(n.kind),
      el("strong", { class: "ex-node-label" }, n.label || n.id),
      el("span", { class: "ex-node-kind" }, kindLabel && kindLabel !== n.kind ? `${n.kind} · ${pretty(kindLabel)}` : n.kind),
      ctx.multi && n.user_id ? userChip(n.user_id, ctx) : null,
    ]),
    n.text && n.text !== n.label ? el("div", { class: "ex-node-text" }, n.text) : null,
    neighbors.length ? el("ul", { class: "ex-neighbors", "aria-label": `Connections of ${n.label || n.id}` }, neighbors.map((nb) =>
      el("li", { class: "ex-neighbor" + (nb.new_edge ? "" : " old-edge") }, [
        el("span", { class: "ex-edge", title: `${DIRECTION_HINT[nb.direction] || ""} · weight ${fmtNum(nb.edge_weight)}` }, [
          icon(DIRECTION_ICON[nb.direction] || "arrow-right"),
          el("span", {}, pretty(nb.edge_kind)),
        ]),
        nodeRef(nb),
        el("span", { class: "ex-neighbor-badges" }, [
          nb.is_new ? el("span", { class: "ex-badge b-new", title: "This node was also created in this pass" }, "new node") : null,
          !nb.new_edge ? el("span", { class: "ex-badge b-old", title: "This link existed before the pass" }, "existing link") : null,
        ]),
      ]))) : el("div", { class: "ex-node-lonely" }, "Not connected to anything yet."),
    hiddenCount > 0 ? el("div", { class: "ex-more-note" }, `+${plural(hiddenCount, "more connection")}`) : null,
  ]);
}

// ---- emotion ----------------------------------------------------------
function renderEmotion(emo, ctx) {
  if (!emo.enabled) return null;
  const users = emo.users || [];
  if (!emo.mood_changed && !users.length) return null;
  const body = [];
  if (emo.mood_changed) {
    body.push(el("div", { class: "ex-emo-block" }, [
      el("div", { class: "ex-sub-label" }, "Character mood"),
      el("div", { class: "ex-bars" }, (emo.mood || []).map((m) => bar(m.axis, m.before, m.after, m.delta, { min: 0, max: 1, tone: "neutral" }))),
    ]));
  }
  for (const u of users) {
    body.push(el("div", { class: "ex-emo-block" }, [
      el("div", { class: "ex-sub-label" }, [`Feelings toward ${ctx.names.get(u.user_id) || u.user_id}`]),
      u.dims.length ? el("div", { class: "ex-bars" }, u.dims.map((d) => bar(d.dim, d.before, d.after, d.delta, { min: -1, max: 1, tone: "signed" }))) : null,
      u.comment_changed ? el("div", { class: "ex-relation" }, [
        el("span", { class: "ex-relation-label" }, "Relationship"),
        u.comment_before ? el("span", { class: "ex-relation-old" }, u.comment_before) : el("span", { class: "ex-relation-none" }, "unset"),
        icon("arrow-right"),
        u.comment_after ? el("span", { class: "ex-relation-new" }, u.comment_after) : el("span", { class: "ex-relation-none" }, "cleared"),
      ]) : null,
    ]));
  }
  const count = users.length + (emo.mood_changed ? 1 : 0);
  return section("ex-sec-emotion", "heart-pulse", "Emotion", count, body,
    { hint: "The faint marker is where each value was before the pass." });
}

function bar(label, before, after, delta, { min, max, tone }) {
  const span = max - min;
  const pos = (v) => ((Math.max(min, Math.min(max, Number(v) || 0)) - min) / span) * 100;
  const zero = pos(Math.max(min, 0));
  const a = pos(after), b = pos(before);
  const moved = changed(delta);
  const dir = delta > 0 ? "up" : "down";
  const fillLeft = Math.min(zero, a), fillWidth = Math.abs(a - zero);
  const deltaLeft = Math.min(a, b), deltaWidth = Math.abs(a - b);
  return el("div", { class: `ex-bar-row ${moved ? `moved ${tone}-${dir}` : "still"}` }, [
    el("span", { class: "ex-bar-label" }, cap(label)),
    el("span", {
      class: "ex-bar-track" + (min < 0 ? " signed" : ""),
      role: "img",
      "aria-label": `${cap(label)}: ${fmtNum(before)} to ${fmtNum(after)}`,
    }, [
      min < 0 ? el("span", { class: "ex-bar-zero", style: `left:${zero}%` }) : null,
      el("span", { class: "ex-bar-fill", style: `left:${fillLeft}%;width:${fillWidth}%` }),
      moved ? el("span", { class: "ex-bar-delta", style: `left:${deltaLeft}%;width:${deltaWidth}%` }) : null,
      moved ? el("span", { class: "ex-bar-before", style: `left:${b}%`, title: `Before ${fmtNum(before)}` }) : null,
    ]),
    el("span", { class: "ex-bar-value" }, fmtNum(after)),
    el("span", { class: "ex-bar-change" }, moved ? fmtDelta(delta) : "—"),
  ]);
}

// ---- conversation -----------------------------------------------------
function renderConversation(messages, ctx, open) {
  if (!messages.length) return null;
  return el("details", { class: "ex-section ex-convo", open }, [
    el("summary", { class: "ex-section-head" }, [
      el("span", { class: "ex-section-icon" }, [icon("messages-square")]),
      el("h4", {}, "Conversation read"),
      el("span", { class: "live-memory-count" }, String(messages.length)),
      el("span", { class: "ex-section-hint" }, "The messages this pass learned from."),
      icon("chevron-down", "ex-chev"),
    ]),
    el("div", { class: "ex-section-body ex-msgs" }, messages.map((m) => el("div", { class: `ex-msg role-${m.role === "user" ? "user" : "assistant"}` }, [
      el("div", { class: "ex-msg-top" }, [
        el("strong", {}, m.role === "user" ? (ctx.names.get(m.speaker) || m.speaker || "user") : (m.speaker || "assistant")),
        m.occurred_at ? el("span", { title: absTime(m.occurred_at) }, whenLabel(m.occurred_at)) : null,
      ]),
      el("div", { class: "ex-msg-text" }, (m.content || "") + (m.truncated ? " …" : "")),
    ]))),
  ]);
}

// ---------------------------------------------------------------- lifecycle
function wire() {
  if (S.wired) return;
  S.wired = true;
  setFollow(S.follow);
  $("ex-hide-empty").checked = S.hideEmpty;
  $("ex-hide-empty").addEventListener("change", (e) => {
    S.hideEmpty = e.target.checked;
    writeFlag(HIDE_EMPTY_KEY, S.hideEmpty);
    if (S.follow || S.selected == null) {
      const top = firstVisible();
      S.selected = top ? top.id : null;
    }
    renderList();
    renderDetail();
  });
  $("ex-follow").addEventListener("click", () => {
    setFollow(!S.follow);
    if (S.follow) {
      const top = firstVisible();
      if (top && top.id !== S.selected) selectRun(top.id);
      poll();
    }
  });
  $("ex-refresh").addEventListener("click", () => reload());
  $("ex-more").addEventListener("click", loadMore);
  $("ex-user").addEventListener("change", (e) => {
    S.user = e.target.value;
    S.runs = []; S.selected = null;
    reload({ keepSelection: false });
  });
  $("ex-runs").addEventListener("keydown", (e) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    const runs = visibleRuns();
    if (!runs.length) return;
    e.preventDefault();
    const idx = runs.findIndex((r) => r.id === S.selected);
    const next = e.key === "Home" ? 0 : e.key === "End" ? runs.length - 1
      : Math.max(0, Math.min(runs.length - 1, idx + (e.key === "ArrowDown" ? 1 : -1)));
    selectRun(runs[next].id, { user: true });
    const btn = $("ex-runs").querySelector(`[data-id="${runs[next].id}"]`);
    if (btn) { btn.focus({ preventScroll: true }); btn.scrollIntoView({ block: "nearest" }); }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && S.active) poll(); });
}

function resetForCharacter(name) {
  S.character = name || null;
  S.runs = []; S.total = 0; S.users = []; S.selected = null;
  S.user = ""; S.disabled = false; S.error = "";
  S.cache.clear();
  S.listSeq++; S.detailSeq++;
}

function startTimer() {
  stopTimer();
  S.timer = setInterval(poll, POLL_MS);
}
function stopTimer() {
  if (S.timer) clearInterval(S.timer);
  S.timer = null;
}

function activate(character) {
  wire();
  const changedCharacter = (character || null) !== S.character;
  S.active = true;
  if (changedCharacter) resetForCharacter(character);
  if (changedCharacter || !S.runs.length || Date.now() - S.lastPoll > POLL_MS) reload();
  else render();
  startTimer();
}

function deactivate() {
  S.active = false;
  stopTimer();
}

function setCharacter(name) {
  if ((name || null) === S.character) return;
  resetForCharacter(name);
  if (S.active) reload();
}

window.cmExtractions = { activate, deactivate, setCharacter };

// app.js picks the initial tab after an async character load, by which time
// this script has run; this covers a pane that is already visible regardless.
if ($("ex-pane") && !$("ex-pane").classList.contains("hidden")) {
  activate($("character") ? $("character").value : null);
}
})();
