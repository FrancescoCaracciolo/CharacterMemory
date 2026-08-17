/* Character Configurator — the "Configure" tab.
 *
 * Talks to the admin endpoints under /api/admin/*:
 *   GET    /api/admin/characters                       -> list
 *   POST   /api/admin/characters                       -> create
 *   DELETE /api/admin/characters/{name}                -> delete
 *   GET    /api/admin/characters/{name}/config         -> merged config
 *   PUT    /api/admin/characters/{name}/config         -> save config
 *   GET    /api/admin/characters/{name}/files?bucket=  -> file list
 *   GET    /api/admin/characters/{name}/files/{b}/{f}  -> read file
 *   PUT    /api/admin/characters/{name}/files/{b}/{f}  -> write file
 *   POST   /api/admin/characters/{name}/files/{b}      -> upload (multipart)
 *   DELETE /api/admin/characters/{name}/files/{b}/{f}  -> delete file
 *   POST   /api/admin/characters/{name}/rebuild        -> re-index
 *   POST   /api/admin/characters/{name}/chat           -> mini chat reply
 *
 * Shares DOM helpers with app.js through `window.cmUtil` so we never
 * duplicate the el()/clear()/$()/markdown()/getJSON() trio.
 */

"use strict";
(function () {
const U = window.cmUtil;
const { $, el, clear, icon, getJSON, markdown, esc } = U;

const API = "";

// Memory knobs the GUI exposes. Mirrors server `_TOGGLEABLE_MEMORIES`.
// `k` flags whether the memory has a retrieval-size number input.
const MEMORIES = [
  { name: "character_info", title: "Character info", k: true },
  { name: "dialogue_style", title: "Dialogue style", k: true },
  { name: "user_facts", title: "User facts", k: true },
  { name: "user_directives", title: "User directives", k: true },
  { name: "episodic", title: "Episodic", k: true },
  { name: "conversation_events", title: "Source conversations", k: true },
  { name: "heartbeat", title: "Heartbeat journal", k: true },
  { name: "user_summary", title: "User summary", k: true },
  { name: "emotion", title: "Emotion tracking", k: false },
  { name: "world", title: "Private world", k: true },
  { name: "calendar", title: "Calendar", k: true },
];

const DEFAULT_SECTION_ORDER = [
  "character_info", "emotion", "world", "calendar", "user_directives", "user_facts",
  "episodic", "conversation_events", "heartbeat", "user_summary",
  "knowledge_graph", "dialogue_style",
];
const SECTION_TITLES = Object.fromEntries([
  ...MEMORIES.map((memory) => [memory.name, memory.title]),
  ["knowledge_graph", "Knowledge graph"],
]);

const WORLD_FEATURES = ["locations", "activities", "routines", "hunger", "energy", "sleep", "autonomous_needs"];
const WORLD_ACTIVITY_KINDS = ["idle", "work", "school", "travel", "eat", "sleep", "leisure", "social", "other"];
const WORLD_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
// `icon` is a Lucide icon name rendered via cmUtil.icon().
const WORLD_SECTIONS = [
  { name: "locations", title: "Locations", singular: "location", icon: "map-pin" },
  { name: "actors", title: "Actors", singular: "actor", icon: "user-round" },
  { name: "routines", title: "Routines", singular: "routine", icon: "repeat" },
  { name: "facts", title: "Facts", singular: "fact", icon: "lightbulb" },
];

const cfgState = {
  characters: [],
  active: null,           // active character name
  config: null,           // merged config for the active character
  world: null,
  worldDraft: null,
  worldSourceDirty: false,
  worldOpen: new Set(),
  worldNew: null,
  bucket: "information",  // active files bucket
  file: null,             // {name} of the selected file
  chat: {                 // mini-chat state, per character
    // [charName]: { chat_id, messages: [{role, content}] }
  },
};

// --------------------------------------------------------------- fetch helpers
async function sendJSON(url, { method = "POST", body } = {}) {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json", ...U.authHeaders() },
    body: body ? JSON.stringify(body) : undefined,
  });
  const txt = await r.text();
  let data = null;
  try { data = txt ? JSON.parse(txt) : null; } catch (e) { data = { detail: txt }; }
  if (!r.ok) throw new Error((data && data.detail) || `HTTP ${r.status}`);
  return data;
}

// The label span flash() (and callers like rebuild()) swap. Created on first
// use by wrapping the button's loose text nodes, so icon SVGs and nested
// inputs (the upload labels) are never clobbered by a text swap.
function flashLabel(btn) {
  let label = btn.querySelector(":scope > .btn-label");
  if (!label) {
    label = el("span", { class: "btn-label" });
    for (const node of [...btn.childNodes]) if (node.nodeType === Node.TEXT_NODE) label.appendChild(node);
    btn.prepend(label);
  }
  return label;
}

function flash(btn, msg, ok = true) {
  if (!btn) return;
  const label = flashLabel(btn);
  // Snapshot only when idle: during a pending flash the label holds the
  // transient message, not the button's real text.
  if (!btn.dataset.flashing) btn.dataset.label = label.textContent.trim();
  btn.dataset.flashing = "1";
  label.textContent = msg;
  btn.classList.toggle("flash-ok", ok);
  btn.classList.toggle("flash-bad", !ok);
  clearTimeout(btn._flashTimer);
  btn._flashTimer = setTimeout(() => {
    label.textContent = btn.dataset.label;
    delete btn.dataset.flashing;
    btn.classList.remove("flash-ok", "flash-bad");
  }, 1400);
}

function showError(msg) {
  const activeEditor = document.querySelector(".editor-tabs .etab.active")?.dataset.edit;
  const log = $("chat-log");
  if (log && activeEditor === "chat") {
    log.appendChild(el("div", { class: "chat-error" }, [icon("triangle-alert"), ` ${msg}`]));
    log.scrollTop = log.scrollHeight;
    return;
  }
  const node = $("toast");
  if (!node) { alert(msg); return; }
  node.textContent = msg;
  node.classList.remove("hidden");
  node.classList.add("bad");
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => node.classList.add("hidden"), 3600);
}

// --------------------------------------------------------------- character list
async function loadCharacters() {
  const data = await getJSON(`${API}/api/admin/characters`);
  cfgState.characters = data.characters || [];
  renderCharList();
}

function renderCharList() {
  const nav = $("cfg-char-list");
  clear(nav);
  if (!cfgState.characters.length) {
    nav.appendChild(el("div", { class: "cfg-empty-sm" }, "No characters. Click + New."));
    return;
  }
  for (const c of cfgState.characters) {
    const tags = [];
    if (c.has_info) tags.push("info");
    if (c.has_dialogue) tags.push("dialogue");
    if (c.has_kg) tags.push("kg");
    nav.appendChild(el("button", {
      class: "cfg-char" + (c.name === cfgState.active ? " active" : ""),
      onclick: () => selectCharacter(c.name),
    }, [
      el("div", { class: "cfg-char-name" }, c.name),
      tags.length ? el("div", { class: "cfg-char-tags" }, tags.map((t) => el("span", { class: "chip" }, t))) : null,
    ]));
  }
}

async function selectCharacter(name) {
  cfgState.active = name;
  cfgState.file = null;
  renderCharList();
  // Lazy-build a chat-cfgState slot per character.
  if (!cfgState.chat[name]) cfgState.chat[name] = { chat_id: null, messages: [] };
  $("cfg-empty").classList.add("hidden");
  $("cfg-pane-main").classList.remove("hidden");
  $("cfg-name").textContent = name;
  await loadConfig();
  await loadFiles();
  renderChat();
}

async function loadWorld() {
  if (!cfgState.active) return;
  try {
    cfgState.world = await getJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/world`);
    $("cfg-world-disabled").classList.add("hidden");
    $("cfg-world-editor").classList.remove("hidden");
    renderWorld();
  } catch (e) {
    cfgState.world = null;
    $("cfg-world-disabled").classList.remove("hidden");
    $("cfg-world-editor").classList.add("hidden");
  }
}

function selectedWorldActor() { return $("cfg-world-actor").value || null; }

function cloneWorldSeed(seed) {
  const draft = JSON.parse(JSON.stringify(seed || {}));
  for (const { name } of WORLD_SECTIONS) {
    if (!Array.isArray(draft[name])) draft[name] = [];
  }
  return draft;
}

function markWorldDirty() {
  $("cfg-world-save-seed").classList.add("dirty");
}

function syncWorldSource({ dirty = true } = {}) {
  if (!cfgState.worldDraft) return;
  $("cfg-world-seed").value = JSON.stringify(cfgState.worldDraft, null, 2);
  cfgState.worldSourceDirty = false;
  if (dirty) markWorldDirty();
  else $("cfg-world-save-seed").classList.remove("dirty");
}

function renderWorld() {
  const data = cfgState.world;
  if (!data) return;
  cfgState.worldDraft = cloneWorldSeed(data.seed);
  cfgState.worldSourceDirty = false;
  cfgState.worldNew = null;
  const actorSelect = $("cfg-world-actor");
  const selected = actorSelect.value;
  clear(actorSelect);
  actorSelect.appendChild(el("option", { value: "" }, "World defaults"));
  for (const actor of data.actors || []) actorSelect.appendChild(el("option", { value: actor.id }, actor.name));
  actorSelect.value = (data.actors || []).some((a) => a.id === selected) ? selected : "";
  renderWorldFeatures();
  $("cfg-world-state").textContent = JSON.stringify(data.snapshot, null, 2);
  $("cfg-world-events").textContent = JSON.stringify(data.events || [], null, 2);
  renderWorldElements();
  syncWorldSource({ dirty: false });
}

function worldOptions(items, { blank = "None" } = {}) {
  return [
    { value: "", label: blank },
    ...items.map((item) => ({ value: item.id, label: item.name ? `${item.name} · ${item.id}` : item.id })),
  ];
}

function worldField(item, field, label, {
  type = "text", options = null, wide = false, placeholder = "", min = null,
  max = null, step = null, transform = null, hint = "", onCommit = null,
  afterInput = null,
} = {}) {
  const wrap = el("div", { class: `world-field${wide ? " wide" : ""}` });
  wrap.appendChild(el("span", { class: "world-field-label" }, label));
  let control;
  if (type === "textarea") {
    control = el("textarea", { class: "world-input world-textarea", placeholder, rows: 3 });
    control.value = item[field] == null ? "" : String(item[field]);
  } else if (type === "select") {
    control = el("select", { class: "world-input" });
    for (const option of options || []) {
      control.appendChild(el("option", { value: option.value }, option.label));
    }
    control.value = item[field] == null ? "" : String(item[field]);
  } else if (type === "checkbox") {
    control = el("input", { type: "checkbox", checked: !!item[field] });
    const check = el("label", { class: "world-check" }, [control, el("span", {}, hint || label)]);
    wrap.appendChild(check);
  } else {
    control = el("input", {
      class: "world-input", type, placeholder, min, max, step,
      value: item[field] == null ? "" : String(item[field]),
    });
  }
  if (type !== "checkbox") wrap.appendChild(control);
  if (hint && type !== "checkbox") wrap.appendChild(el("small", {}, hint));

  const initialValue = item[field];
  const read = () => {
    if (type === "checkbox") return control.checked;
    const raw = control.value;
    if (transform) return transform(raw);
    if (type === "number") return raw === "" ? null : Number(raw);
    if (type === "select") return raw || null;
    return raw;
  };
  control.addEventListener("input", () => {
    const previous = item[field];
    item[field] = read();
    if (afterInput) afterInput(previous, item[field]);
    syncWorldSource();
  });
  if (onCommit) {
    control.addEventListener("change", () => {
      onCommit(initialValue, item[field]);
      renderWorldElements();
      syncWorldSource();
    });
  }
  return wrap;
}

function worldDaysField(item) {
  const wrap = el("div", { class: "world-field wide" }, [
    el("span", { class: "world-field-label" }, "Active days"),
  ]);
  const days = new Set((item.days || []).map(Number));
  const choices = el("div", { class: "world-days" });
  WORLD_DAYS.forEach((name, day) => {
    const input = el("input", { type: "checkbox", checked: days.has(day) });
    input.addEventListener("change", () => {
      if (input.checked) days.add(day); else days.delete(day);
      item.days = [...days].sort();
      syncWorldSource();
    });
    choices.appendChild(el("label", {}, [input, el("span", {}, name)]));
  });
  wrap.appendChild(choices);
  return wrap;
}

function rewriteWorldReferences(collection, previous, next) {
  if (!previous || previous === next) return;
  const draft = cfgState.worldDraft;
  if (collection === "locations") {
    for (const location of draft.locations) if (location.parent_id === previous) location.parent_id = next || null;
    for (const actor of draft.actors) {
      if (actor.home_location === previous) actor.home_location = next || null;
      if (actor.location_id === previous) actor.location_id = next || null;
    }
    for (const routine of draft.routines) if (routine.location_id === previous) routine.location_id = next || null;
    for (const fact of draft.facts) if (fact.location_id === previous) fact.location_id = next || null;
  } else if (collection === "actors") {
    if (draft.observer_id === previous) draft.observer_id = next;
    for (const routine of draft.routines) if (routine.actor_id === previous) routine.actor_id = next || null;
    for (const fact of draft.facts) if (fact.subject_id === previous) fact.subject_id = next || null;
  }
}

function worldElementSummary(section, item, index) {
  let title = item.name;
  if (section.name === "routines") title = item.activity;
  if (section.name === "facts") title = item.content;
  title = String(title || `New ${section.singular}`).trim();
  if (title.length > 72) title = title.slice(0, 69) + "…";
  return el("summary", { class: "world-element-summary" }, [
    el("span", { class: "world-element-icon", "aria-hidden": "true" }, icon(section.icon)),
    el("span", { class: "world-element-title" }, title),
    item.id ? el("code", {}, item.id) : el("span", { class: "world-missing-id" }, `#${index + 1}`),
    el("span", { class: "world-chevron", "aria-hidden": "true" }, icon("chevron-down")),
  ]);
}

function renderLocationFields(item) {
  const locations = cfgState.worldDraft.locations;
  return [
    worldField(item, "name", "Name", { placeholder: "Research lab" }),
    worldField(item, "id", "ID", {
      placeholder: "research_lab", hint: "Stable identifier used by actors and routines.",
      transform: (value) => value.trim(),
      onCommit: (previous, next) => rewriteWorldReferences("locations", previous, next),
    }),
    worldField(item, "description", "Description", {
      type: "textarea", wide: true, placeholder: "What this place is like…",
    }),
    worldField(item, "parent_id", "Inside / parent", {
      type: "select", options: worldOptions(locations.filter((location) => location !== item)),
    }),
    worldField(item, "timezone", "Timezone override", { placeholder: "Europe/Rome", transform: (value) => value.trim() || null }),
    worldField(item, "tags", "Tags", {
      wide: true, placeholder: "indoors, science, private",
      transform: (value) => value.split(",").map((tag) => tag.trim()).filter(Boolean),
    }),
  ];
}

function renderActorFields(item) {
  const locations = cfgState.worldDraft.locations;
  return [
    worldField(item, "name", "Name", { placeholder: "Mayuri Shiina" }),
    worldField(item, "id", "ID", {
      placeholder: "mayuri", hint: "Stable identifier used by routines and facts.",
      transform: (value) => value.trim(),
      onCommit: (previous, next) => rewriteWorldReferences("actors", previous, next),
    }),
    worldField(item, "home_location", "Home location", {
      type: "select", options: worldOptions(locations),
      afterInput: (previous, next) => { if (!item.location_id || item.location_id === previous) item.location_id = next; },
    }),
    worldField(item, "timezone", "Timezone", { placeholder: cfgState.worldDraft.timezone || "UTC", transform: (value) => value.trim() || null }),
    worldField(item, "public_activity", "Public activity", {
      type: "checkbox", hint: "Other actors can see this actor's activity from elsewhere.", wide: true,
    }),
  ];
}

function renderRoutineFields(item) {
  const draft = cfgState.worldDraft;
  return [
    worldField(item, "activity", "Activity", { placeholder: "working in the lab" }),
    worldField(item, "id", "ID", { placeholder: "weekday_lab", transform: (value) => value.trim() }),
    worldField(item, "actor_id", "Actor", { type: "select", options: worldOptions(draft.actors, { blank: "Choose an actor" }) }),
    worldField(item, "activity_kind", "Kind", {
      type: "select", options: WORLD_ACTIVITY_KINDS.map((kind) => ({ value: kind, label: kind })),
    }),
    worldField(item, "location_id", "Location", { type: "select", options: worldOptions(draft.locations) }),
    worldField(item, "start_local", "Starts at", { type: "time" }),
    worldField(item, "duration_minutes", "Duration (minutes)", { type: "number", min: 1, step: 5 }),
    worldField(item, "priority", "Priority", { type: "number", step: 1, hint: "Higher wins when routines overlap." }),
    worldDaysField(item),
    worldField(item, "enabled", "Enabled", { type: "checkbox", hint: "Use this routine in the simulation." }),
    worldField(item, "interruptible", "Interruptible", { type: "checkbox", hint: "Commands may interrupt this routine." }),
  ];
}

function renderFactFields(item) {
  const draft = cfgState.worldDraft;
  return [
    worldField(item, "content", "Fact", { type: "textarea", wide: true, placeholder: "The lab's front door locks after midnight." }),
    worldField(item, "id", "ID", { placeholder: "lab_door", transform: (value) => value.trim() }),
    worldField(item, "visibility", "Visibility", {
      type: "select", options: [
        { value: "known", label: "Known · always known by this character" },
        { value: "public", label: "Public · visible to everyone" },
        { value: "local", label: "Local · visible only at its location" },
        { value: "private", label: "Private · stored but not perceived" },
      ],
    }),
    worldField(item, "subject_id", "Subject actor", { type: "select", options: worldOptions(draft.actors) }),
    worldField(item, "location_id", "Location", { type: "select", options: worldOptions(draft.locations) }),
    worldField(item, "importance", "Importance", { type: "number", min: 0, max: 1, step: 0.05 }),
  ];
}

function renderWorldElement(section, item, index) {
  const key = `${section.name}:${index}`;
  const isNew = cfgState.worldNew && cfgState.worldNew.collection === section.name && cfgState.worldNew.index === index;
  const card = el("details", {
    class: `world-element-card${isNew ? " is-new" : ""}`,
    open: isNew || cfgState.worldOpen.has(key),
  });
  card.appendChild(worldElementSummary(section, item, index));
  const body = el("div", { class: "world-element-body" });
  const tools = el("div", { class: "world-element-tools" }, [
    el("span", { class: "cfg-hint" }, `Edit ${section.singular}`),
    el("button", {
      class: "btn danger ghost tight", type: "button",
      disabled: section.name === "actors" && item.id === cfgState.worldDraft.observer_id,
      title: section.name === "actors" && item.id === cfgState.worldDraft.observer_id ? "The observer actor cannot be removed" : "",
      onclick: () => removeWorldElement(section.name, index),
    }, "Remove"),
  ]);
  body.appendChild(tools);
  const fields = el("div", { class: "world-element-fields" });
  let children = [];
  if (section.name === "locations") children = renderLocationFields(item);
  else if (section.name === "actors") children = renderActorFields(item);
  else if (section.name === "routines") children = renderRoutineFields(item);
  else children = renderFactFields(item);
  for (const child of children) fields.appendChild(child);
  body.appendChild(fields);
  card.appendChild(body);
  card.addEventListener("toggle", () => {
    if (card.open) cfgState.worldOpen.add(key); else cfgState.worldOpen.delete(key);
  });
  return card;
}

function renderWorldElements() {
  const root = $("cfg-world-elements");
  clear(root);
  if (!cfgState.worldDraft) return;
  for (const section of WORLD_SECTIONS) {
    const items = cfgState.worldDraft[section.name];
    const group = el("section", { class: "world-element-group" });
    group.appendChild(el("div", { class: "world-element-group-head" }, [
      el("div", {}, [
        el("span", { class: "world-element-group-icon", "aria-hidden": "true" }, icon(section.icon)),
        el("strong", {}, section.title),
        el("span", { class: "world-count" }, String(items.length)),
      ]),
      el("button", {
        class: "btn ghost tight", type: "button", onclick: () => addWorldElement(section.name),
      }, [icon("plus"), `Add ${section.singular}`]),
    ]));
    const list = el("div", { class: "world-element-list" });
    if (!items.length) {
      list.appendChild(el("button", {
        class: "world-element-empty", type: "button", onclick: () => addWorldElement(section.name),
      }, `No ${section.name} yet — add one`));
    } else {
      items.forEach((item, index) => list.appendChild(renderWorldElement(section, item, index)));
    }
    group.appendChild(list);
    root.appendChild(group);
  }
  if (cfgState.worldNew) {
    requestAnimationFrame(() => {
      const card = root.querySelector(".world-element-card.is-new");
      if (!card) return;
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      const input = card.querySelector("input:not([type=checkbox]), textarea");
      if (input) { input.focus(); if (input.select) input.select(); }
      cfgState.worldNew = null;
    });
  }
}

function uniqueWorldId(collection, base) {
  const used = new Set((cfgState.worldDraft[collection] || []).map((item) => item.id));
  let id = base; let suffix = 2;
  while (used.has(id)) id = `${base}_${suffix++}`;
  return id;
}

function applyWorldSource() {
  let parsed;
  try { parsed = JSON.parse($("cfg-world-seed").value); }
  catch (e) { throw new Error("The visual editor can apply JSON directly. Save valid YAML to import it."); }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("World source must be an object.");
  cfgState.worldDraft = cloneWorldSeed(parsed);
  cfgState.worldSourceDirty = false;
  renderWorldElements();
  syncWorldSource();
  flash($("cfg-world-apply-source"), "Applied", true);
}

function ensureWorldDraft() {
  if (cfgState.worldSourceDirty) applyWorldSource();
  if (!cfgState.worldDraft) throw new Error("Load a world before adding elements.");
}

function addWorldElement(collection) {
  try { ensureWorldDraft(); } catch (e) { showError(e.message); return; }
  const draft = cfgState.worldDraft;
  let item;
  if (collection === "locations") {
    const id = uniqueWorldId(collection, "new_location");
    item = { id, name: "New location", description: "", parent_id: null, timezone: null, tags: [] };
  } else if (collection === "actors") {
    const id = uniqueWorldId(collection, "new_actor");
    const home = draft.locations[0]?.id || null;
    item = {
      id, name: "New actor", home_location: home, location_id: home,
      timezone: draft.timezone || "UTC", public_activity: false, features: {}, metadata: {},
    };
  } else if (collection === "routines") {
    const id = uniqueWorldId(collection, "new_routine");
    item = {
      id, actor_id: draft.observer_id || draft.actors[0]?.id || null,
      days: [0, 1, 2, 3, 4, 5, 6], start_local: "09:00", duration_minutes: 60,
      location_id: draft.locations[0]?.id || null, activity_kind: "other",
      activity: "new activity", priority: 0, interruptible: true, enabled: true,
    };
  } else {
    const id = uniqueWorldId(collection, "new_fact");
    item = { id, content: "", subject_id: null, location_id: null, visibility: "known", importance: 0.8 };
  }
  draft[collection].push(item);
  cfgState.worldNew = { collection, index: draft[collection].length - 1 };
  syncWorldSource();
  renderWorldElements();
}

function removeWorldElement(collection, index) {
  const draft = cfgState.worldDraft;
  const item = draft[collection][index];
  if (!item) return;
  if (collection === "actors" && item.id === draft.observer_id) {
    showError("The observer actor cannot be removed. Choose another observer in the advanced source first.");
    return;
  }
  const label = item.name || item.activity || item.content || item.id || `this ${collection.slice(0, -1)}`;
  if (!confirm(`Remove “${String(label).slice(0, 80)}” from the world? The change is applied when you save.`)) return;
  draft[collection].splice(index, 1);
  if (collection === "locations") rewriteWorldReferences("locations", item.id, null);
  if (collection === "actors") {
    draft.routines = draft.routines.filter((routine) => routine.actor_id !== item.id);
    for (const fact of draft.facts) if (fact.subject_id === item.id) fact.subject_id = null;
  }
  cfgState.worldOpen.clear();
  syncWorldSource();
  renderWorldElements();
}

function renderWorldFeatures() {
  const box = $("cfg-world-features"); clear(box);
  const actorId = selectedWorldActor();
  const actor = actorId && (cfgState.world.actors || []).find((a) => a.id === actorId);
  const defaults = cfgState.world.seed.features || {};
  const overrides = actor ? ((cfgState.world.seed.actors || []).find((a) => a.id === actorId)?.features || {}) : {};
  for (const name of WORLD_FEATURES) {
    const select = el("select", { class: "user-filter", "data-world-feature": name });
    if (actorId) {
      select.appendChild(el("option", { value: "" }, `inherit (${defaults[name] === false ? "off" : "on"})`));
      select.appendChild(el("option", { value: "true" }, "on"));
      select.appendChild(el("option", { value: "false" }, "off"));
      select.value = Object.prototype.hasOwnProperty.call(overrides, name) ? String(!!overrides[name]) : "";
    } else {
      select.appendChild(el("option", { value: "true" }, "on"));
      select.appendChild(el("option", { value: "false" }, "off"));
      select.value = String(defaults[name] !== false);
    }
    box.appendChild(el("label", { class: "mem-row" }, [el("span", { class: "mem-row-title" }, name.replaceAll("_", " ")), select]));
  }
  const resolved = actor ? actor.resolved_features : defaults;
  const commands = [];
  if (resolved.locations !== false) commands.push("move");
  if (resolved.activities !== false) commands.push("start_activity");
  if (resolved.hunger !== false || resolved.activities !== false) commands.push("eat");
  if (resolved.sleep !== false) commands.push("sleep", "wake");
  const command = $("cfg-world-command"); clear(command);
  for (const name of commands) command.appendChild(el("option", {}, name));
  command.disabled = commands.length === 0;
  $("cfg-world-command-run").disabled = commands.length === 0;
}

async function saveWorldFeatures() {
  const actorId = selectedWorldActor();
  const values = {};
  document.querySelectorAll("[data-world-feature]").forEach((input) => {
    values[input.dataset.worldFeature] = input.value === "" ? null : input.value === "true";
  });
  await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/world/features`, {
    method: "PATCH", body: { actor_id: actorId, values },
  });
  await loadWorld();
}

async function saveWorldSeed() {
  if (cfgState.worldSourceDirty) {
    let parsed;
    try { parsed = JSON.parse($("cfg-world-seed").value); }
    catch (e) {
      await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/world/import`, {
        body: { content: $("cfg-world-seed").value },
      });
      await loadWorld();
      flash($("cfg-world-save-seed"), "Saved", true);
      return;
    }
    cfgState.worldDraft = cloneWorldSeed(parsed);
  }
  validateWorldDraft(cfgState.worldDraft);
  await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/world/seed`, {
    method: "PUT", body: cfgState.worldDraft,
  });
  await loadWorld();
  flash($("cfg-world-save-seed"), "Saved", true);
}

function validateWorldDraft(seed) {
  if (!seed || Array.isArray(seed) || typeof seed !== "object") throw new Error("World source must be an object.");
  for (const { name, singular } of WORLD_SECTIONS) {
    if (!Array.isArray(seed[name])) throw new Error(`${name} must be a list.`);
    const ids = new Set();
    seed[name].forEach((item, index) => {
      if (!item || typeof item !== "object") throw new Error(`${singular} #${index + 1} must be an object.`);
      const id = String(item.id || "").trim();
      if (!id) throw new Error(`Every ${singular} needs an ID.`);
      if (ids.has(id)) throw new Error(`Duplicate ${singular} ID: ${id}`);
      ids.add(id);
    });
  }
  const locationIds = new Set(seed.locations.map((item) => item.id));
  const actorIds = new Set(seed.actors.map((item) => item.id));
  if (!actorIds.has(seed.observer_id)) throw new Error(`Observer actor “${seed.observer_id || ""}” does not exist.`);
  for (const location of seed.locations) {
    if (!String(location.name || "").trim()) throw new Error(`Location “${location.id}” needs a name.`);
    if (location.parent_id && !locationIds.has(location.parent_id)) throw new Error(`Location “${location.id}” has an unknown parent.`);
    if (location.parent_id === location.id) throw new Error(`Location “${location.id}” cannot be its own parent.`);
  }
  for (const actor of seed.actors) {
    if (!String(actor.name || "").trim()) throw new Error(`Actor “${actor.id}” needs a name.`);
    if (actor.home_location && !locationIds.has(actor.home_location)) throw new Error(`Actor “${actor.id}” has an unknown home location.`);
  }
  for (const routine of seed.routines) {
    if (!actorIds.has(routine.actor_id)) throw new Error(`Routine “${routine.id}” needs a valid actor.`);
    if (routine.location_id && !locationIds.has(routine.location_id)) throw new Error(`Routine “${routine.id}” has an unknown location.`);
    if (!WORLD_ACTIVITY_KINDS.includes(routine.activity_kind)) throw new Error(`Routine “${routine.id}” has an unknown activity kind.`);
    if (!/^([01]\d|2[0-3]):[0-5]\d$/.test(String(routine.start_local || ""))) throw new Error(`Routine “${routine.id}” needs a valid start time.`);
    if (!Number.isFinite(Number(routine.duration_minutes)) || Number(routine.duration_minutes) < 1) throw new Error(`Routine “${routine.id}” needs a positive duration.`);
  }
  for (const fact of seed.facts) {
    if (!String(fact.content || "").trim()) throw new Error(`Fact “${fact.id}” needs content.`);
    if (fact.subject_id && !actorIds.has(fact.subject_id)) throw new Error(`Fact “${fact.id}” has an unknown subject actor.`);
    if (fact.location_id && !locationIds.has(fact.location_id)) throw new Error(`Fact “${fact.id}” has an unknown location.`);
  }
}

function importWorldSource(content) {
  $("cfg-world-seed").value = content;
  cfgState.worldSourceDirty = true;
  markWorldDirty();
  try {
    const parsed = JSON.parse(content);
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("World source must be an object.");
    cfgState.worldDraft = cloneWorldSeed(parsed);
    cfgState.worldSourceDirty = false;
    renderWorldElements();
    syncWorldSource();
  } catch (e) {
    $("cfg-world-source-panel").open = true;
  }
}

function exportWorld() {
  const content = cfgState.worldSourceDirty
    ? $("cfg-world-seed").value
    : JSON.stringify(cfgState.worldDraft || {}, null, 2);
  const blob = new Blob([content], { type: "application/json" });
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob);
  link.download = `${cfgState.active || "character"}-world.json`; link.click(); URL.revokeObjectURL(link.href);
}

async function advanceWorld() {
  await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/world/advance`);
  await loadWorld();
}

async function applyWorldCommand() {
  const kind = $("cfg-world-command").value;
  const value = $("cfg-world-command-value").value.trim();
  const body = { kind, actor_id: selectedWorldActor() || cfgState.world.snapshot.observer_id };
  if (kind === "move") body.location_id = value;
  if (kind === "start_activity") { body.activity_kind = "other"; body.activity = value; }
  await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/world/commands`, { body });
  await loadWorld();
}

// --------------------------------------------------------------- new / delete
function newCharacter() {
  // The 5-step wizard owns creation; opens a modal.
  openWizard();
}

async function deleteCharacter() {
  if (!cfgState.active) return;
  if (!confirm(`Delete "${cfgState.active}"? This wipes its files, learned memory and indexes. Irreversible.`)) return;
  try {
    await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}`, { method: "DELETE" });
    cfgState.active = null;
    cfgState.config = null;
    $("cfg-empty").classList.remove("hidden");
    $("cfg-pane-main").classList.add("hidden");
    await loadCharacters();
  } catch (e) { showError(e.message); }
}

// --------------------------------------------------------------- config (persona + memories)
async function loadConfig() {
  try {
    cfgState.config = await getJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/config`);
    renderConfig();
  } catch (e) { showError(e.message); }
}

function renderConfig() {
  const cfg = cfgState.config;
  $("cfg-persona").value = cfg.persona || "";
  renderMemoryList(cfg.memory || {});
  $("cfg-kg").checked = !!cfg.kg_enabled;
  $("cfg-kg-token-budget").disabled = !cfg.kg_enabled;
  $("cfg-kg-token-budget").value = String(
    cfg.memory && cfg.memory.knowledge_graph_token_budget != null
      ? cfg.memory.knowledge_graph_token_budget
      : 1000
  );
  renderSectionOrder(cfg.section_order || DEFAULT_SECTION_ORDER);
}

function renderMemoryList(mem) {
  const box = $("cfg-memory-list"); clear(box);
  for (const m of MEMORIES) {
    const enabled = mem[`enabled_${m.name}`] !== false;
    const kVal = mem[`${m.name}_k`] != null ? mem[`${m.name}_k`] : "";
    const kInput = el("input", {
      type: "number", min: "1", max: "50", value: String(kVal),
      class: "k-input", "data-mem": m.name, disabled: !enabled,
    });
    kInput.addEventListener("input", () => {
      const v = parseInt(kInput.value, 10);
      if (!Number.isNaN(v)) dirtyMemory();
    });
    const toggle = el("input", {
      type: "checkbox", class: "switch-input", checked: enabled, "data-mem": m.name,
    });
    toggle.addEventListener("change", () => {
      kInput.disabled = !toggle.checked;
      if (toggle.checked) ensureSectionIncluded(m.name);
      refreshSectionOrderRows();
      dirtyMemory();
    });
    box.appendChild(el("label", { class: "mem-row" }, [
      el("span", { class: "switch" }, [toggle]),
      el("span", { class: "mem-row-title" }, m.title),
      m.k ? el("span", { class: "k-wrap" }, [
        el("span", { class: "k-label" }, "k"),
        kInput,
      ]) : null,
    ]));
  }
}

function sectionEnabled(name) {
  if (name === "knowledge_graph") return !!$("cfg-kg").checked;
  const toggle = document.querySelector(`.switch-input[data-mem="${name}"]`);
  return !!(toggle && toggle.checked);
}

function renderSectionOrder(order) {
  const box = $("cfg-section-order"); clear(box);
  const requested = Array.isArray(order)
    ? order.filter((name, index) => SECTION_TITLES[name] && order.indexOf(name) === index)
    : [...DEFAULT_SECTION_ORDER];
  const included = new Set(requested);
  const names = [...requested, ...DEFAULT_SECTION_ORDER.filter((name) => !included.has(name))];
  for (const name of names) {
    const include = el("input", {
      type: "checkbox", checked: included.has(name), "data-section-include": name,
      title: `Include ${SECTION_TITLES[name]} in /context`,
      "aria-label": `Include ${SECTION_TITLES[name]} in /context`,
    });
    include.addEventListener("change", () => { refreshSectionOrderRows(); dirtyMemory(); });
    const row = el("div", { class: "cfg-section-row", "data-section-name": name }, [
      el("span", { class: "cfg-section-position", "aria-hidden": "true" }),
      include,
      el("span", { class: "cfg-section-title" }, SECTION_TITLES[name]),
      el("span", { class: "cfg-section-status" }),
      el("span", { class: "cfg-section-actions" }, [
        el("button", {
          type: "button", class: "cfg-section-move", "data-direction": "up",
          title: `Move ${SECTION_TITLES[name]} up`, "aria-label": `Move ${SECTION_TITLES[name]} up`,
          onclick: () => moveSectionRow(row, -1),
        }, icon("chevron-up")),
        el("button", {
          type: "button", class: "cfg-section-move", "data-direction": "down",
          title: `Move ${SECTION_TITLES[name]} down`, "aria-label": `Move ${SECTION_TITLES[name]} down`,
          onclick: () => moveSectionRow(row, 1),
        }, icon("chevron-down")),
      ]),
    ]);
    box.appendChild(row);
  }
  refreshSectionOrderRows();
}

function moveSectionRow(row, direction) {
  const sibling = direction < 0 ? row.previousElementSibling : row.nextElementSibling;
  if (!sibling) return;
  if (direction < 0) row.parentNode.insertBefore(row, sibling);
  else row.parentNode.insertBefore(sibling, row);
  refreshSectionOrderRows();
  dirtyMemory();
}

function refreshSectionOrderRows() {
  const rows = [...document.querySelectorAll("#cfg-section-order .cfg-section-row")];
  let includedPosition = 0;
  rows.forEach((row, index) => {
    const name = row.dataset.sectionName;
    const enabled = sectionEnabled(name);
    const included = row.querySelector("[data-section-include]").checked;
    row.classList.toggle("is-disabled", !enabled);
    row.classList.toggle("is-excluded", !included);
    row.querySelector(".cfg-section-position").textContent = included
      ? String(++includedPosition).padStart(2, "0") : "—";
    const status = row.querySelector(".cfg-section-status");
    status.textContent = !enabled ? "memory off" : included ? "included" : "omitted";
    row.querySelector('[data-direction="up"]').disabled = index === 0;
    row.querySelector('[data-direction="down"]').disabled = index === rows.length - 1;
  });
}

function ensureSectionIncluded(name) {
  const include = document.querySelector(`[data-section-include="${name}"]`);
  if (include && !include.checked) include.checked = true;
}

function dirtyMemory() {
  const btn = $("cfg-save-memory");
  btn.classList.add("dirty");
}

function gatherConfig() {
  const mem = {};
  for (const m of MEMORIES) {
    const tog = document.querySelector(`.switch-input[data-mem="${m.name}"]`);
    if (tog) mem[`enabled_${m.name}`] = tog.checked;
    if (m.k) {
      const k = document.querySelector(`.k-input[data-mem="${m.name}"]`);
      const v = parseInt(k && k.value, 10);
      if (!Number.isNaN(v)) mem[`${m.name}_k`] = v;
    }
  }
  const kgTokenBudget = parseInt($("cfg-kg-token-budget").value, 10);
  if (!Number.isNaN(kgTokenBudget)) {
    mem.knowledge_graph_token_budget = Math.max(0, kgTokenBudget);
  }
  const sectionOrder = [...document.querySelectorAll("#cfg-section-order .cfg-section-row")]
    .filter((row) => row.querySelector("[data-section-include]").checked)
    .map((row) => row.dataset.sectionName);
  return {
    persona: $("cfg-persona").value,
    kg_enabled: $("cfg-kg").checked,
    memory: mem,
    section_order: sectionOrder,
  };
}

async function savePersona() {
  if (!cfgState.active) return;
  try {
    const persona = $("cfg-persona").value;
    cfgState.config = await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/config`, {
      method: "PUT", body: { persona },
    });
    flash($("cfg-save-persona"), "Saved", true);
  } catch (e) { showError(e.message); flash($("cfg-save-persona"), "Failed", false); }
}

async function saveMemory() {
  if (!cfgState.active) return;
  try {
    const cfg = gatherConfig();
    cfgState.config = await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/config`, {
      method: "PUT", body: cfg,
    });
    renderConfig();
    $("cfg-save-memory").classList.remove("dirty");
    flash($("cfg-save-memory"), "Saved", true);
  } catch (e) { showError(e.message); flash($("cfg-save-memory"), "Failed", false); }
}

// --------------------------------------------------------------- files
async function loadFiles() {
  if (!cfgState.active) return;
  try {
    const data = await getJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/files?bucket=${cfgState.bucket}`);
    renderFileList(data.files || []);
    if (cfgState.file) await loadFileContent(cfgState.file.name);
    else clearFileEditor();
  } catch (e) { showError(e.message); }
}

function renderFileList(files) {
  const ul = $("cfg-file-list"); clear(ul);
  if (!files.length) {
    ul.appendChild(el("li", { class: "cfg-empty-sm" }, "No files. Upload or create one."));
    return;
  }
  for (const f of files) {
    const isActive = cfgState.file && cfgState.file.name === f.name;
    ul.appendChild(el("li", {
      class: "file-item" + (isActive ? " active" : ""),
      onclick: () => selectFile(f.name),
    }, [
      el("span", { class: "file-name", title: f.name }, f.name),
      el("span", { class: "file-size" }, humanSize(f.size)),
    ]));
  }
}

function humanSize(n) {
  if (n == null) return "";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
}

async function selectFile(name) {
  cfgState.file = { name };
  await loadFiles();  // refresh highlight
}

async function loadFileContent(name) {
  try {
    const r = await fetch(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/files/${cfgState.bucket}/${encodeURIComponent(name)}`, { headers: U.authHeaders() });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    $("cfg-filename").value = name;
    $("cfg-file-content").value = await r.text();
  } catch (e) { showError(e.message); }
}

function clearFileEditor() {
  $("cfg-filename").value = "";
  $("cfg-file-content").value = "";
}

async function saveFile() {
  if (!cfgState.active) return;
  const name = $("cfg-filename").value.trim();
  if (!name) { showError("Filename required."); return; }
  const content = $("cfg-file-content").value;
  try {
    await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/files/${cfgState.bucket}/${encodeURIComponent(name)}`, {
      method: "PUT", body: { content },
    });
    cfgState.file = { name };
    await loadFiles();
    flash($("cfg-save-file"), "Saved", true);
  } catch (e) { showError(e.message); flash($("cfg-save-file"), "Failed", false); }
}

async function deleteFile() {
  if (!cfgState.active || !cfgState.file) return;
  if (!confirm(`Delete ${cfgState.file.name}?`)) return;
  try {
    await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/files/${cfgState.bucket}/${encodeURIComponent(cfgState.file.name)}`, { method: "DELETE" });
    cfgState.file = null;
    clearFileEditor();
    await loadFiles();
  } catch (e) { showError(e.message); }
}

async function newFile() {
  const name = prompt("New filename (must end in .md or .txt):", "new.md");
  if (!name) return;
  if (!/\.(md|txt)$/i.test(name)) { showError("Filename must end in .md or .txt"); return; }
  cfgState.file = { name };
  $("cfg-filename").value = name;
  $("cfg-file-content").value = "";
  await saveFile();
}

async function uploadFiles(fileList) {
  if (!cfgState.active || !fileList || !fileList.length) return;
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f, f.name);
  try {
    const r = await fetch(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/files/${cfgState.bucket}`, {
      method: "POST", body: fd, headers: U.authHeaders(),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    await loadFiles();
    flash(document.querySelector(".files-toolbar .btn.ghost"), "Uploaded", true);
  } catch (e) { showError(e.message); }
}

async function rebuild() {
  if (!cfgState.active) return;
  if (!confirm(`Rebuild all indexes for "${cfgState.active}"? Re-chunks files and (if enabled) re-runs KG extraction. May take a while.`)) return;
  const btn = $("cfg-rebuild");
  const label = flashLabel(btn);
  btn.disabled = true; label.textContent = "Rebuilding…";
  const status = $("cfg-build-status");
  try {
    const data = await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/rebuild`);
    await pollJob(data.job_id, (snap) => {
      const pct = Math.round((snap.progress || 0) * 100);
      label.textContent = `Rebuilding… ${pct}%`;
      if (status) {
        clear(status);
        const line = el("div", { class: "build-line" }, `${snap.cfgState} · ${snap.stage}${snap.detail ? " — " + snap.detail : ""}`);
        const bar = el("div", { class: "build-bar" }, [el("div", { class: "build-bar-fill", style: `width:${pct}%` })]);
        status.append(bar, line);
      }
    });
    flash(btn, "Done", true);
    if (status) status.appendChild(el("div", { class: "build-done" }, [icon("check"), " Indexes rebuilt."]));
    await loadCharacters();
  } catch (e) { showError(e.message); flash(btn, "Failed", false); }
  finally {
    btn.disabled = false;
    label.textContent = "Rebuild indexes";
    // A pending flash timer restores dataset.label — keep it in sync so the
    // button settles on its real label, not the last progress text.
    if (btn.dataset.label) btn.dataset.label = "Rebuild indexes";
  }
}

// Poll a rebuild job until it reaches a terminal cfgState. `onUpdate` is called
// for every snapshot (including the final one). Throws if the job errored.
async function pollJob(jobId, onUpdate) {
  let snap = { cfgState: "pending", stage: "queued", progress: 0, detail: "" };
  while (snap.cfgState !== "done" && snap.cfgState !== "error") {
    await new Promise((r) => setTimeout(r, 700));
    let r;
    try {
      r = await fetch(`${API}/api/jobs/${encodeURIComponent(jobId)}`, { headers: U.authHeaders() });
    } catch (e) {
      // transient fetch error: keep polling
      continue;
    }
    if (r.status === 404) throw new Error("Job vanished.");
    if (r.status === 401) throw new Error("Unauthorized: set the API key with the key button in the top bar.");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    snap = await r.json();
    onUpdate(snap || {});
  }
  if (snap.cfgState === "error") throw new Error(snap.detail || "Build failed.");
  return snap;
}

// --------------------------------------------------------------- mini chat
function renderChat() {
  const log = $("chat-log"); clear(log);
  const slot = cfgState.chat[cfgState.active] || { messages: [] };
  if (!slot.messages.length) {
    log.appendChild(el("div", { class: "chat-empty" }, `Chat with ${cfgState.active}. Messages are remembered by the character.`));
    return;
  }
  for (const m of slot.messages) {
    log.appendChild(el("div", { class: `chat-bubble ${m.role}` }, [
      el("div", { class: "chat-role" }, m.role === "user" ? "you" : cfgState.active),
      m.role === "assistant"
        ? el("div", { class: "chat-md", html: markdown(m.content || "") })
        : el("div", { class: "chat-text" }, m.content || ""),
    ]));
  }
  log.scrollTop = log.scrollHeight;
}

async function sendMessage(text) {
  if (!cfgState.active || !text.trim()) return;
  const slot = cfgState.chat[cfgState.active];
  slot.messages.push({ role: "user", content: text });
  renderChat();
  $("chat-input").value = "";
  const sendBtn = $("chat-send"); sendBtn.disabled = true;
  // Thinking placeholder.
  const log = $("chat-log");
  const think = el("div", { class: "chat-bubble assistant thinking" }, [
    el("div", { class: "chat-role" }, cfgState.active),
    el("div", { class: "chat-text" }, "…"),
  ]);
  log.appendChild(think);
  log.scrollTop = log.scrollHeight;
  try {
    const user = $("cfg-chat-user").value || "user";
    const data = await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/chat`, {
      body: { message: text, user, chat_id: slot.chat_id },
    });
    slot.chat_id = data.chat_id;
    slot.messages.push({ role: "assistant", content: data.reply });
    think.remove();
    renderChat();
  } catch (e) {
    think.remove();
    slot.messages.pop();  // roll back the optimistic user turn
    showError(e.message);
    renderChat();
  } finally {
    sendBtn.disabled = false;
  }
}

function clearChat() {
  if (!cfgState.active) return;
  // Just drop the local conversation view + chat_id; the server keeps history.
  cfgState.chat[cfgState.active] = { chat_id: null, messages: [] };
  renderChat();
}

// --------------------------------------------------------------- editor tabs
function setEditTab(name) {
  document.querySelectorAll(".editor-tabs .etab").forEach((b) => {
    b.classList.toggle("active", b.dataset.edit === name);
  });
  document.querySelectorAll(".edit-pane").forEach((p) => {
    p.classList.toggle("hidden", p.dataset.edit !== name);
  });
  if (name === "world") loadWorld();
}

function setBucket(name) {
  cfgState.bucket = name;
  cfgState.file = null;
  document.querySelectorAll(".bucket-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.bucket === name);
  });
  clearFileEditor();
  loadFiles();
}

// --------------------------------------------------------------- wizard (5-step create)
// State held only while the modal is open. Reset on open.
const wiz = {
  step: 1,
  name: "",
  desc: "",
  wikiFiles: [],      // [{name, file}]
  dialogueFiles: [],  // [{name, file}]
  memories: {},       // {name: {enabled, k}}
  kg: false,
  building: false,
};

function defaultWizardMemories() {
  // Same defaults as the library (everything except KG, which is its own toggle).
  const out = {};
  for (const m of MEMORIES) out[m.name] = { enabled: true, k: null };
  return out;
}

function openWizard() {
  wiz.step = 1;
  wiz.name = ""; wiz.desc = "";
  wiz.wikiFiles = []; wiz.dialogueFiles = [];
  wiz.memories = defaultWizardMemories();
  wiz.kg = false; wiz.building = false;
  $("wiz-name").value = "";
  $("wiz-desc").value = "";
  $("wiz-kg").checked = false;
  renderWizFileLists();
  renderWizMemories();
  setWizStep(1);
  setStatus("Ready to build.");
  $("wiz-overlay").classList.remove("hidden");
  $("wiz-name").focus();
}

function closeWizard() {
  $("wiz-overlay").classList.add("hidden");
}

function setWizStep(n) {
  wiz.step = n;
  // Stepper dots
  document.querySelectorAll("#wiz-stepper .wiz-step").forEach((li) => {
    const s = Number(li.dataset.step);
    li.classList.toggle("active", s === n);
    li.classList.toggle("done", s < n);
  });
  // Panes
  document.querySelectorAll(".wiz-pane").forEach((p) => {
    p.classList.toggle("hidden", Number(p.dataset.step) !== n);
    p.classList.toggle("active", Number(p.dataset.step) === n);
  });
  // Foot buttons
  $("wiz-back").disabled = n === 1;
  $("wiz-next").classList.toggle("hidden", n === 5);
  $("wiz-build").classList.toggle("hidden", n !== 5);
  // Validate current step
  hideWizError();
  if (n === 1) validateStep1();
}

function hideWizError() {
  for (let i = 1; i <= 5; i++) {
    const e = $(`wiz-err-${i}`);
    if (e) { e.textContent = ""; e.classList.add("hidden"); }
  }
}

function showWizError(step, msg) {
  const e = $(`wiz-err-${step}`);
  if (e) { e.textContent = msg; e.classList.remove("hidden"); }
  else showError(msg);
}

function validateStep1() {
  const name = $("wiz-name").value.trim();
  const ok = name.length > 0;
  $("wiz-next").disabled = !ok;
  if (!ok && document.activeElement !== $("wiz-name")) {
    showWizError(1, "A name is required.");
  }
  return ok;
}

function renderWizFileLists() {
  for (const [key, files, listId] of [
    ["wiki", wiz.wikiFiles, "wiz-wiki-list"],
    ["dialogue", wiz.dialogueFiles, "wiz-dialogue-list"],
  ]) {
    const ul = $(listId); clear(ul);
    if (!files.length) {
      ul.appendChild(el("li", { class: "wiz-file-empty" },
        key === "wiki" ? "No wiki files added yet." : "No dialogue files added yet."));
      continue;
    }
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      ul.appendChild(el("li", { class: "wiz-file-item" }, [
        el("span", { class: "wiz-file-name", title: f.name }, f.name),
        el("span", { class: "wiz-file-size" }, humanSize(f.file.size)),
        el("button", { class: "wiz-file-rm", onclick: () => {
          files.splice(i, 1); renderWizFileLists();
        } }, icon("x")),
      ]));
    }
  }
}

function renderWizMemories() {
  const box = $("wiz-memories"); clear(box);
  for (const m of MEMORIES) {
    const st = wiz.memories[m.name] || { enabled: true, k: null };
    const toggle = el("input", {
      type: "checkbox", class: "switch-input", checked: st.enabled, "data-mem": m.name,
    });
    toggle.addEventListener("change", () => {
      wiz.memories[m.name].enabled = toggle.checked;
    });
    box.appendChild(el("label", { class: "mem-row" }, [
      el("span", { class: "switch" }, [toggle]),
      el("span", { class: "mem-row-title" }, m.title),
    ]));
  }
}

function addWizFiles(which, fileList) {
  const target = which === "wiki" ? wiz.wikiFiles : wiz.dialogueFiles;
  for (const f of Array.from(fileList || [])) target.push({ name: f.name, file: f });
  renderWizFileLists();
}

function setStatus(msg, kind) {
  const box = $("wiz-build-status"); clear(box);
  box.appendChild(el("div", { class: "wiz-status-line " + (kind || "") },
    kind === "ok" ? [icon("check"), ` ${msg}`]
      : kind === "bad" ? [icon("x"), ` ${msg}`]
      : msg));
}

async function runWizardBuild() {
  if (wiz.building) return;
  wiz.building = true;
  $("wiz-build").disabled = true;
  $("wiz-back").disabled = true;
  const name = wiz.name.trim();
  const enc = encodeURIComponent;
  try {
    // 1. Create the character.
    setStatus(`Creating character "${name}"…`);
    await sendJSON(`${API}/api/admin/characters`, { body: { name } });
    // 2. Persona.
    setStatus("Saving persona…");
    await sendJSON(`${API}/api/admin/characters/${enc(name)}/config`, {
      method: "PUT", body: { persona: wiz.desc.trim() },
    });
    // 3. Memory toggles + KG.
    setStatus("Applying memory settings…");
    const mem = {};
    for (const m of MEMORIES) mem[`enabled_${m.name}`] = !!wiz.memories[m.name]?.enabled;
    await sendJSON(`${API}/api/admin/characters/${enc(name)}/config`, {
      method: "PUT", body: { memory: mem, kg_enabled: wiz.kg },
    });
    // 4. Upload files (parallel buckets).
    setStatus("Uploading files…");
    await uploadWizBucket(name, "information", wiz.wikiFiles);
    await uploadWizBucket(name, "dialogues", wiz.dialogueFiles);
    // 5. Build (async job).
    setStatus("Starting build…");
    const { job_id } = await sendJSON(`${API}/api/admin/characters/${enc(name)}/rebuild`);
    await pollJob(job_id, (snap) => {
      const pct = Math.round((snap.progress || 0) * 100);
      setStatus(`${snap.stage}${snap.detail ? " — " + snap.detail : ""} (${pct}%)`);
    });
    setStatus(`"${name}" is ready.`, "ok");
    await loadCharacters();
    await selectCharacter(name);
    // Brief beat so the user sees the success line before close.
    await new Promise((r) => setTimeout(r, 600));
    closeWizard();
  } catch (e) {
    setStatus(e.message, "bad");
    showError(e.message);
  } finally {
    wiz.building = false;
    $("wiz-build").disabled = false;
    $("wiz-back").disabled = false;
  }
}

async function uploadWizBucket(name, bucket, files) {
  if (!files || !files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f.file, f.name);
  const r = await fetch(`${API}/api/admin/characters/${encodeURIComponent(name)}/files/${bucket}`, {
    method: "POST", body: fd, headers: U.authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
}

// --------------------------------------------------------------- wiring
function wire() {
  $("cfg-new").addEventListener("click", newCharacter);
  $("cfg-delete").addEventListener("click", deleteCharacter);

  $("cfg-save-persona").addEventListener("click", savePersona);
  $("cfg-save-memory").addEventListener("click", saveMemory);
  $("cfg-kg").addEventListener("change", () => {
    $("cfg-kg-token-budget").disabled = !$("cfg-kg").checked;
    if ($("cfg-kg").checked) ensureSectionIncluded("knowledge_graph");
    refreshSectionOrderRows();
    dirtyMemory();
  });
  $("cfg-kg-token-budget").addEventListener("input", dirtyMemory);

  document.querySelectorAll(".editor-tabs .etab").forEach((b) => {
    b.addEventListener("click", () => setEditTab(b.dataset.edit));
  });
  document.querySelectorAll(".bucket-btn").forEach((b) => {
    b.addEventListener("click", () => setBucket(b.dataset.bucket));
  });

  $("cfg-save-file").addEventListener("click", saveFile);
  $("cfg-delete-file").addEventListener("click", deleteFile);
  $("cfg-new-file").addEventListener("click", newFile);
  $("cfg-upload").addEventListener("change", (e) => {
    uploadFiles(e.target.files);
    e.target.value = "";  // allow re-uploading the same file
  });
  $("cfg-rebuild").addEventListener("click", rebuild);
  $("cfg-world-actor").addEventListener("change", renderWorldFeatures);
  $("cfg-world-save-features").addEventListener("click", () => saveWorldFeatures().catch((e) => showError(e.message)));
  $("cfg-world-save-seed").addEventListener("click", () => saveWorldSeed().catch((e) => showError(e.message)));
  document.querySelectorAll("[data-world-add]").forEach((button) => {
    button.addEventListener("click", () => addWorldElement(button.dataset.worldAdd));
  });
  $("cfg-world-apply-source").addEventListener("click", () => {
    try { applyWorldSource(); } catch (e) { showError(e.message); flash($("cfg-world-apply-source"), "Invalid", false); }
  });
  $("cfg-world-seed").addEventListener("input", () => {
    cfgState.worldSourceDirty = true;
    markWorldDirty();
  });
  $("cfg-world-advance").addEventListener("click", () => advanceWorld().catch((e) => showError(e.message)));
  $("cfg-world-command-run").addEventListener("click", () => applyWorldCommand().catch((e) => showError(e.message)));
  $("cfg-world-export").addEventListener("click", exportWorld);
  $("cfg-world-import").addEventListener("click", () => $("cfg-world-import-file").click());
  $("cfg-world-import-file").addEventListener("change", async (e) => {
    const file = e.target.files[0]; if (!file) return;
    importWorldSource(await file.text()); e.target.value = "";
  });

  // Chat
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage($("chat-input").value);
  });
  $("cfg-chat-clear").addEventListener("click", clearChat);

  // Wizard
  $("wiz-close").addEventListener("click", closeWizard);
  $("wiz-overlay").addEventListener("click", (e) => {
    if (e.target === $("wiz-overlay") && !wiz.building) closeWizard();
  });
  $("wiz-back").addEventListener("click", () => { if (wiz.step > 1 && !wiz.building) setWizStep(wiz.step - 1); });
  $("wiz-next").addEventListener("click", () => {
    if (wiz.step === 1) {
      const name = $("wiz-name").value.trim();
      if (!name) { showWizError(1, "A name is required."); return; }
      wiz.name = name; wiz.desc = $("wiz-desc").value;
    }
    if (wiz.step < 5) setWizStep(wiz.step + 1);
  });
  $("wiz-name").addEventListener("input", validateStep1);
  $("wiz-build").addEventListener("click", runWizardBuild);
  $("wiz-wiki-upload").addEventListener("change", (e) => {
    addWizFiles("wiki", e.target.files); e.target.value = "";
  });
  $("wiz-dialogue-upload").addEventListener("change", (e) => {
    addWizFiles("dialogue", e.target.files); e.target.value = "";
  });
  $("wiz-kg").addEventListener("change", (e) => { wiz.kg = e.target.checked; });
}

// Public hook: app.js calls this when the Configure tab is activated so the
// character list refreshes (it may have changed via another tab/window).
function activate() {
  loadCharacters();
}

window.cmConfig = { activate };

wire();
})();
