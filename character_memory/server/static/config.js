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
const { $, el, clear, getJSON, markdown, esc } = U;

const API = "";

// Memory knobs the GUI exposes. Mirrors server `_TOGGLEABLE_MEMORIES`.
// `k` flags whether the memory has a retrieval-size number input.
const MEMORIES = [
  { name: "character_info", title: "Character info", k: true },
  { name: "dialogue_style", title: "Dialogue style", k: true },
  { name: "user_facts", title: "User facts", k: true },
  { name: "user_directives", title: "User directives", k: true },
  { name: "episodic", title: "Episodic", k: true },
  { name: "heartbeat", title: "Heartbeat journal", k: true },
  { name: "user_summary", title: "User summary", k: true },
  { name: "emotion", title: "Emotion tracking", k: false },
];

const cfgState = {
  characters: [],
  active: null,           // active character name
  config: null,           // merged config for the active character
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
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const txt = await r.text();
  let data = null;
  try { data = txt ? JSON.parse(txt) : null; } catch (e) { data = { detail: txt }; }
  if (!r.ok) throw new Error((data && data.detail) || `HTTP ${r.status}`);
  return data;
}

function flash(btn, msg, ok = true) {
  if (!btn) return;
  const orig = btn.dataset.label || btn.textContent;
  btn.dataset.label = orig;
  btn.textContent = msg;
  btn.classList.toggle("flash-ok", ok);
  btn.classList.toggle("flash-bad", !ok);
  setTimeout(() => { btn.textContent = orig; btn.classList.remove("flash-ok", "flash-bad"); }, 1400);
}

function showError(msg) {
  const log = $("chat-log");
  if (log) {
    log.appendChild(el("div", { class: "chat-error" }, "⚠ " + msg));
    log.scrollTop = log.scrollHeight;
  } else {
    alert(msg);
  }
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
  return { persona: $("cfg-persona").value, kg_enabled: $("cfg-kg").checked, memory: mem };
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
    const r = await fetch(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/files/${cfgState.bucket}/${encodeURIComponent(name)}`);
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
      method: "POST", body: fd,
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
  const btn = $("cfg-rebuild"); btn.disabled = true; btn.textContent = "Rebuilding…";
  const status = $("cfg-build-status");
  try {
    const data = await sendJSON(`${API}/api/admin/characters/${encodeURIComponent(cfgState.active)}/rebuild`);
    await pollJob(data.job_id, (snap) => {
      const pct = Math.round((snap.progress || 0) * 100);
      btn.textContent = `Rebuilding… ${pct}%`;
      if (status) {
        clear(status);
        const line = el("div", { class: "build-line" }, `${snap.cfgState} · ${snap.stage}${snap.detail ? " — " + snap.detail : ""}`);
        const bar = el("div", { class: "build-bar" }, [el("div", { class: "build-bar-fill", style: `width:${pct}%` })]);
        status.append(bar, line);
      }
    });
    flash(btn, "Done", true);
    if (status) status.appendChild(el("div", { class: "build-done" }, "✓ Indexes rebuilt."));
    await loadCharacters();
  } catch (e) { showError(e.message); flash(btn, "Failed", false); }
  finally { btn.disabled = false; btn.textContent = "⟳ Rebuild indexes"; }
}

// Poll a rebuild job until it reaches a terminal cfgState. `onUpdate` is called
// for every snapshot (including the final one). Throws if the job errored.
async function pollJob(jobId, onUpdate) {
  let snap = { cfgState: "pending", stage: "queued", progress: 0, detail: "" };
  while (snap.cfgState !== "done" && snap.cfgState !== "error") {
    await new Promise((r) => setTimeout(r, 700));
    let r;
    try {
      r = await fetch(`${API}/api/jobs/${encodeURIComponent(jobId)}`);
    } catch (e) {
      // transient fetch error: keep polling
      continue;
    }
    if (r.status === 404) throw new Error("Job vanished.");
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
        } }, "×"),
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
  box.appendChild(el("div", { class: "wiz-status-line " + (kind || "") }, msg));
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
    setStatus(`✓ "${name}" is ready.`, "ok");
    await loadCharacters();
    await selectCharacter(name);
    // Brief beat so the user sees the success line before close.
    await new Promise((r) => setTimeout(r, 600));
    closeWizard();
  } catch (e) {
    setStatus("✗ " + e.message, "bad");
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
    method: "POST", body: fd,
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
