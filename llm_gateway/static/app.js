"use strict";

const $ = (s) => document.querySelector(s);
const store = {
  get(k, d) { try { const v = localStorage.getItem("llmgw." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("llmgw." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};

let profiles = [];
let status = { active: null, status: "idle" };
let convs = store.get("convs", []);
let currentId = store.get("current", null);
let abortCtl = null;
let pendingActivate = null;
const BUSY_MSG = "The current model is in use or loading (a chat or API request is still running). Switching is possible once it finishes.";

// ── helpers ───────────────────────────────────────────────────────────────
function headers(json = true) {
  const h = json ? { "Content-Type": "application/json" } : {};
  const key = store.get("key", "");
  if (key) h.Authorization = "Bearer " + key;
  return h;
}
async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
  if (r.status === 401) { showAuth(); throw new Error("API key required"); }
  const data = await r.json().catch(() => ({}));
  if (r.status === 409) throw new Error(BUSY_MSG);
  if (!r.ok) throw new Error(data?.error?.message || r.statusText);
  return data;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtCtx = (n) => (n == null ? "—" : n >= 1024 ? Math.round(n / 1024) + "K" : String(n));
const prof = (id) => profiles.find((p) => p.id === id);

function banner(text, kind = "error") {
  const b = $("#banner");
  if (!text) { b.hidden = true; return; }
  b.textContent = text; b.className = "banner" + (kind === "info" ? " info" : ""); b.hidden = false;
}
function showAuth() {
  $("#keyRow").hidden = false;
  banner("The server requires an API key. Enter it on the API page and save.");
}

// Minimal, safe markdown: HTML is escaped first; only a small set of constructs is rendered.
function md(src) {
  const parts = String(src).split(/```/);
  return parts.map((seg, i) => {
    if (i % 2 === 1) {
      const nl = seg.indexOf("\n");
      const code = nl >= 0 ? seg.slice(nl + 1) : seg;
      return `<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`;
    }
    return seg.split(/\n{2,}/).map(block).join("");
  }).join("");
}
function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
// Line-based: headings, rules and list runs can sit anywhere inside a paragraph block.
function block(b) {
  let html = "", para = [], list = null;
  const flushPara = () => { if (para.length) { html += `<p>${para.map(inline).join("<br>")}</p>`; para = []; } };
  const flushList = () => { if (list) { html += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`; list = null; } };
  for (const line of b.split("\n")) {
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    const li = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(line);
    if (!line.trim()) { flushPara(); flushList(); }
    else if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flushPara(); flushList(); html += "<hr>"; }
    else if (h) { flushPara(); flushList(); const n = Math.min(h[1].length + 1, 4); html += `<h${n}>${inline(h[2])}</h${n}>`; }
    else if (li) {
      flushPara();
      const tag = /\d/.test(li[1]) ? "ol" : "ul";
      if (list && list.tag !== tag) flushList();
      if (!list) list = { tag, items: [] };
      list.items.push(li[2]);
    } else { flushList(); para.push(line); }
  }
  flushPara(); flushList();
  return html;
}

// "Working" indicator (chat and agent): bouncing dots plus what is happening right now.
function workingEl(label) {
  const d = document.createElement("div");
  d.className = "working";
  d.setAttribute("role", "status");
  d.innerHTML = `<span class="dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="lbl"></span>`;
  d.querySelector(".lbl").textContent = label;
  return d;
}

// ── tabs ──────────────────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === "view-" + name));
  store.set("view", name);
}
$("#tabs").addEventListener("click", (e) => { const v = e.target.closest("button")?.dataset.view; if (v) showView(v); });

// ── status ────────────────────────────────────────────────────────────────
function renderStatus() {
  const p = prof(status.active);
  const loadingP = prof(status.loading);
  const dot = $("#activeDot");
  dot.className = "dot " + (status.status === "ready" ? "ready" : status.status === "loading" ? "loading" : status.status === "error" ? "error" : "");
  if (status.status === "loading" && loadingP) {
    $("#activeName").textContent = "Loading: " + loadingP.name;
    $("#activeMeta").textContent = loadingP.backend + " · context " + fmtCtx(loadingP.context);
  } else if (p) {
    $("#activeName").textContent = p.name;
    const sp = p.speed_zh_tps || p.speed_tps;
    $("#activeMeta").textContent = `${p.backend} · context ${fmtCtx(p.context)}${sp ? " · ~" + sp + " tok/s" : ""}`;
  } else {
    $("#activeName").textContent = status.status === "error" ? "Load failed" : "No model loaded";
    $("#activeMeta").textContent = "";
  }
  $("#unloadBtn").hidden = !status.active;
  $("#unloadBtn").disabled = !!status.busy;
  $("#unloadBtn").title = status.busy ? "The model is in use; it can be released when it finishes" : "Release the current model and free the GPU";
  if (status.status === "error" && status.error) banner("Model failed to load: " + status.error);
  else if ($("#banner").textContent.startsWith("Model failed to load")) banner("");
  renderResources();
  renderModels();
  renderChatHead();
}
// CPU, RAM, GPU and VRAM of the machine running the gateway (refreshed with the 2-second status poll).
function renderResources() {
  const s = status.system, g = status.gpu;
  const gb = (mb) => (mb / 1024).toFixed(1);
  const meters = [];
  if (s) {
    meters.push({ k: "CPU", pct: s.cpu_percent, v: `${Math.round(s.cpu_percent)}%`, t: `${s.cpu_count} threads` });
    meters.push({ k: "RAM", pct: 100 * s.memory_used_mb / s.memory_total_mb, v: `${gb(s.memory_used_mb)}/${gb(s.memory_total_mb)} GB`, t: "System memory" });
  }
  if (g) {
    meters.push({ k: "GPU", pct: g.utilization, v: `${g.utilization}%`, t: g.name });
    meters.push({ k: "VRAM", pct: 100 * g.memory_used_mb / g.memory_total_mb, v: `${gb(g.memory_used_mb)}/${gb(g.memory_total_mb)} GB`, t: g.name + " memory" });
  }
  const box = $("#res");
  const open = store.get("resOpen", false);      // collapsed until the user opens it
  $("#resToggle").hidden = !meters.length;
  $("#resToggle").textContent = open ? "Resources ▾" : "Resources ▸";
  $("#resToggle").setAttribute("aria-expanded", String(open));
  box.hidden = !meters.length || !open;
  box.innerHTML = meters.map((m) => `<div class="res-item" title="${esc(m.t)}">
    <span class="res-k">${m.k}</span><span class="res-v">${m.v}</span>
    <span class="meter"><span class="meter-fill${m.pct >= 90 ? " hot" : ""}" style="width:${Math.min(100, m.pct).toFixed(0)}%"></span></span></div>`).join("")
    + renderQueue();
  const n = (status.queue?.length || 0) + (status.tasks?.waiting?.length || 0);
  $("#resToggle").innerHTML = (open ? "Resources ▾" : "Resources ▸") +
    (n ? `<span class="badge" title="${n} waiting for the GPU or to start">${n}</span>` : "");
}

// The GPU scheduler's state (manager.py + agent/dispatch.py), shown only in the Resources panel.
function who(item) {
  const name = prof(item.model)?.name || item.model;
  const by = item.title ? `Agent “${item.title}”` : { chat: "Chat", agent: "Agent", api: "API client", admin: "Model load" }[item.source] || item.source;
  return `${esc(by)} · ${esc(name)}`;
}
// The GPU queue as one ordered list: the running job boxed on top (or Empty), then the jobs
// waiting for the GPU in the order they will run, then agent tasks waiting to start.
function jobRows() {
  const run = status.running, queue = status.queue || [], leases = status.leases || [];
  const waiting = status.tasks?.waiting || [];
  const rows = [];
  if (run) {
    const kept = leases.find((l) => l.id === run.lease);
    rows.push(`<li class="job running"><span class="job-n">Running</span><span class="job-what">${who(run)}
      <small>${run.running_s ?? 0}s${kept ? " · keeps the model until the task ends" : ""}</small></span></li>`);
  } else {
    const kept = leases[0];
    rows.push(`<li class="job running idle"><span class="job-n">Running</span><span class="job-what">Empty${kept
      ? `<small>${esc(kept.title ? `Agent “${kept.title}”` : kept.id)} keeps ${esc(prof(kept.model)?.name || kept.model)} between its steps</small>` : ""}</span></li>`);
  }
  let n = 0;
  for (const q of queue) {
    rows.push(`<li class="job"><span class="job-n">${++n}</span><span class="job-what">${who(q)}<small>waiting ${q.waiting_s}s</small></span></li>`);
  }
  for (const w of waiting) {
    rows.push(`<li class="job"><span class="job-n">${++n}</span><span class="job-what">Agent “${esc(w.title)}”<small>task waiting to start · ${w.waiting_s}s</small></span></li>`);
  }
  return rows.join("");
}
function renderQueue() {
  return `<ol class="jobs res-queue">${jobRows()}</ol>`;
}

$("#resToggle").addEventListener("click", () => { store.set("resOpen", !store.get("resOpen", false)); renderResources(); });

let offline = false;
async function pollStatus() {
  try {
    status = await api("/api/status");
    if (offline) { offline = false; banner(""); }
    renderStatus();
  } catch (e) {
    if (e instanceof TypeError) { offline = true; banner("Cannot reach llm-gateway, retrying…", "info"); }
  }
}
$("#unloadBtn").addEventListener("click", async () => {
  try { status = await api("/api/unload", { method: "POST" }); renderStatus(); } catch (e) { banner(e.message); }
});

// ── models page ───────────────────────────────────────────────────────────
function renderModels() {
  const list = $("#modelList");
  list.innerHTML = profiles.map((p) => {
    const active = p.id === status.active;
    const loading = p.id === status.loading || p.id === pendingActivate;
    const bar = (label, v) => v ? `<div class="speed"><span class="lbl">${label}</span><span class="track"><span class="fill${v < 30 ? " slow" : ""}" style="width:${Math.min(100, v / 80 * 100)}%"></span><span class="goal"></span></span><span class="val">${v} tok/s</span></div>` : "";
    const busyNote = status.loading ? "Another model is loading; switching is possible once it finishes" : "The current model is in use; switching is possible once it finishes";
    const btn = active ? `<button type="button" class="ghost" disabled>Loaded</button>`
      : loading ? `<button type="button" class="ghost" disabled>Loading…</button>`
      : status.busy ? `<button type="button" class="ghost" disabled title="${busyNote}">In use</button>`
      : `<button type="button" class="primary" data-activate="${esc(p.id)}">Load</button>`;
    const quant = p.quant ? `<span>Quant <b>${esc(p.quant)}</b>${p.bits ? ` · avg ${p.bits} bit` : ""}</span>` : "";
    const details = [
      ["Model file", p.model],
      ["Parameters", p.params_b ? `${p.params_b}B` : ""],
      ["Avg bits", p.bits ? `${p.bits} bits per parameter (file size ÷ parameters, including unquantized parts such as embeddings)` : ""],
      ["Launch flags", p.args?.length ? p.args.join(" ") : ""],
      ["Notes", p.notes],
    ].filter(([, v]) => v);
    return `<article class="model${active ? " active" : ""}">
      <div class="name">${esc(p.name)} <span class="id">${esc(p.id)}</span>
        <span class="tag">${esc(p.backend)}</span>${p.type ? `<span class="tag">${p.type === "moe" ? "MoE" : "Dense"}</span>` : ""}
        ${p.placement ? `<span class="tag ${p.placement === "gpu" ? "on-gpu" : "on-cpu"}" title="${p.placement === "gpu" ? "The whole model is on the GPU" : "Too big for the GPU: part of it runs on the CPU / system RAM"}">${p.placement === "gpu" ? "GPU" : "GPU+CPU"}</span>` : ""}
        ${active ? '<span class="tag on">Loaded</span>' : ""}</div>
      <div class="facts"><span>context <b>${fmtCtx(p.context)}</b>${p.max_context ? ` / max ${fmtCtx(p.max_context)}` : ""}</span>
        ${quant}${p.size_gb ? `<span>Size <b>${p.size_gb} GB</b></span>` : ""}</div>
      <div class="act">${btn}</div>
      ${bar("Chinese", p.speed_zh_tps)}${bar("English code", p.speed_tps)}
      ${details.length ? `<details class="more" data-id="${esc(p.id)}"${openDetails.has(p.id) ? " open" : ""}><summary>Details</summary>
        <dl>${details.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl></details>` : ""}
    </article>`;
  }).join("");
}
const openDetails = new Set();  // survive the 2-second re-render
$("#modelList").addEventListener("toggle", (e) => {
  const id = e.target.dataset?.id;
  if (id) e.target.open ? openDetails.add(id) : openDetails.delete(id);
}, true);
$("#modelList").addEventListener("click", async (e) => {
  const id = e.target.closest("[data-activate]")?.dataset.activate;
  if (!id) return;
  pendingActivate = id; renderModels(); banner("");
  pollStatus();
  try { status = await api("/api/activate", { method: "POST", body: JSON.stringify({ id }) }); }
  catch (err) { banner(err.message === BUSY_MSG ? BUSY_MSG : "Model failed to load: " + err.message, err.message === BUSY_MSG ? "info" : "error"); }
  pendingActivate = null; pollStatus();
});

// ── chat ──────────────────────────────────────────────────────────────────
function saveConvs() { store.set("convs", convs); store.set("current", currentId); }
function current() { return convs.find((c) => c.id === currentId); }
function newConv() {
  const c = { id: Date.now().toString(36), title: "New chat", model: $("#chatModel").value || status.active, messages: [] };
  convs.unshift(c); currentId = c.id; saveConvs(); renderConvs(); renderMessages();
}
function renderConvs() {
  $("#convList").innerHTML = convs.map((c) => `<li class="${c.id === currentId ? "on" : ""}">
    <button type="button" class="title" data-open="${c.id}">${esc(c.title)}</button>
    <button type="button" class="ren" data-rename="${c.id}" aria-label="Rename">✎</button>
    <button type="button" class="del" data-del="${c.id}" aria-label="Delete">✕</button></li>`).join("");
}
// Rename in place (chat and agent lists): the title becomes a text box; Enter or tapping elsewhere
// saves, Esc cancels.
function inlineRename(li, current, save) {
  const btn = li.querySelector(".title");
  const input = document.createElement("input");
  input.className = "rename";
  input.value = current;
  input.maxLength = 120;
  input.setAttribute("aria-label", "New name");
  btn.replaceWith(input);
  input.focus(); input.select();
  let done = false;
  const finish = async (keep) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (keep && name && name !== current) {
      try { await save(name); return; } catch (e) { banner(e.message); }
    }
    input.replaceWith(btn);
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { e.stopPropagation(); finish(false); }   // don't also close the drawer
  });
  input.addEventListener("blur", () => finish(true));
}
$("#convList").addEventListener("click", (e) => {
  const open = e.target.closest("[data-open]")?.dataset.open;
  const del = e.target.closest("[data-del]")?.dataset.del;
  const ren = e.target.closest("[data-rename]")?.dataset.rename;
  if (ren) {
    const c = convs.find((x) => x.id === ren);
    if (c) inlineRename(e.target.closest("li"), c.title, (name) => { c.title = name; saveConvs(); renderConvs(); });
    return;
  }
  if (open) { currentId = open; const c = current(); if (c?.model && prof(c.model)) $("#chatModel").value = c.model; saveConvs(); renderConvs(); renderMessages(); renderChatHead(); setConvsOpen(false); }
  if (del) { convs = convs.filter((c) => c.id !== del); if (currentId === del) currentId = convs[0]?.id || null; saveConvs(); renderConvs(); renderMessages(); }
});
function setConvsOpen(open) {
  $("#convs").classList.toggle("open", open);
  $("#scrim").hidden = !open;
}
$("#newChat").addEventListener("click", () => { newConv(); setConvsOpen(false); $("#input").focus(); });
$("#toggleConvs").addEventListener("click", () => setConvsOpen(!$("#convs").classList.contains("open")));
$("#closeConvs").addEventListener("click", () => setConvsOpen(false));
$("#scrim").addEventListener("click", () => setConvsOpen(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("#convs").classList.contains("open")) setConvsOpen(false); });

function renderChatHead() {
  const sel = $("#chatModel");
  if (!sel.options.length && profiles.length) {
    sel.innerHTML = byName(profiles).map((p) => modelOption(p.id, p.name, p.context,
      [p.backend, p.quant, p.speed_zh_tps ? `~${p.speed_zh_tps} tok/s` : ""].filter(Boolean).join(" · "))).join("");
    const saved = store.get("model", null);
    sel.value = (saved && prof(saved)) ? saved : (status.active || profiles[0].id);
  }
  const p = prof(sel.value);
  if (!p) return;
  const note = p.id === status.active ? "loaded" : "switches on send";
  const used = estimateTokens(current()?.messages || []);
  const usedTxt = used ? `~${used >= 1024 ? (used / 1024).toFixed(1) + "K" : used} / ` : "";
  $("#chatCtx").textContent = `context ${usedTxt}${fmtCtx(p.context)} · ${note}`;
}
// Model menus list models alphabetically ("Qwen3 8B" before "Qwen3 14B": numbers compare as numbers).
const byName = (list) => [...list].sort((x, y) => x.name.localeCompare(y.name, undefined, { numeric: true }));
// Closed menu: "Qwen3 8B · 32K"; open menu: the full name plus a hint line.
function modelOption(id, name, context, hint) {
  const short = `${name.split(" · ")[0]} · ${fmtCtx(context)}`;
  return `<option value="${esc(id)}" data-short="${esc(short)}" data-hint="${esc(`context ${fmtCtx(context)}${hint ? " · " + hint : ""}`)}">${esc(name)}</option>`;
}
// Same rough estimate the server uses: CJK chars ≈ 1 token, other text ≈ 3.5 chars per token.
function estimateTokens(messages) {
  return messages.reduce((sum, m) => {
    const t = (m.content || "");
    const cjk = (t.match(/[　-鿿가-힯豈-﫿＀-￯]/g) || []).length;
    return sum + 6 + cjk + Math.round((t.length - cjk) / 3.5);
  }, 0);
}
$("#thinkToggle").checked = store.get("think", false);
$("#thinkToggle").addEventListener("change", () => store.set("think", $("#thinkToggle").checked));
$("#chatModel").addEventListener("change", () => { store.set("model", $("#chatModel").value); const c = current(); if (c) { c.model = $("#chatModel").value; saveConvs(); } renderChatHead(); });

function msgEl(m) {
  const div = document.createElement("div");
  div.className = "msg " + m.role;
  if (m.role === "user") {
    div.innerHTML = `<div class="body"></div>`;
    div.querySelector(".body").textContent = m.content;
  } else {
    div.innerHTML = `${m.reasoning ? `<details class="think"><summary>Reasoning</summary><div></div></details>` : ""}<div class="body${m.error ? " error" : ""}"></div><div class="meta"></div>`;
    if (m.reasoning) div.querySelector(".think div").textContent = m.reasoning;
    div.querySelector(".body").innerHTML = m.error ? esc(m.error) : md(m.content || "");
    div.querySelector(".meta").textContent = m.meta || "";
  }
  return div;
}
function renderMessages() {
  const box = $("#messages");
  const c = current();
  box.innerHTML = "";
  if (!c || !c.messages.length) { box.innerHTML = $("#emptyTpl").innerHTML; return; }
  c.messages.forEach((m) => box.appendChild(msgEl(m)));
  box.scrollTop = box.scrollHeight;
}

async function send(text) {
  let c = current();
  if (!c) { newConv(); c = current(); }
  const model = $("#chatModel").value;
  c.model = model;
  if (!c.messages.length) c.title = text.slice(0, 40);
  c.messages.push({ role: "user", content: text });
  const reply = { role: "assistant", content: "", reasoning: "", meta: "" };
  c.messages.push(reply);
  saveConvs(); renderConvs(); renderMessages();

  const box = $("#messages");
  const el = box.lastElementChild;
  const body = el.querySelector(".body");
  const metaEl = el.querySelector(".meta");
  const p = prof(model);
  if (model !== status.active) { metaEl.textContent = `Loading ${p?.name || model}…`; pollStatus(); }
  const working = workingEl("Thinking…");       // until the answer is complete
  el.insertBefore(working, metaEl);

  const history = c.messages.slice(0, -1).filter((m) => !m.error).map((m) => ({ role: m.role, content: m.content }));
  abortCtl = new AbortController();
  $("#sendBtn").hidden = true; $("#stopBtn").hidden = false;
  const t0 = performance.now();
  let tFirst = null, nChunks = 0, usage = null, timings = null;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: { ...headers(), "X-LLM-Gateway-Source": "chat" }, signal: abortCtl.signal,
      body: JSON.stringify({ model, messages: history, stream: true, think: $("#thinkToggle").checked }),
    });
    if (r.status === 401) { showAuth(); throw new Error("API key required"); }
    if (r.status === 409) throw new Error(BUSY_MSG);
    if (!r.ok || !r.body) { const d = await r.json().catch(() => ({})); throw new Error(d?.error?.message || r.statusText); }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 1);
        if (line.startsWith(": llm-gateway ")) {  // server-side auto-compaction notice
          try { reply.compacted = JSON.parse(line.slice(14)); } catch { /* ignore */ }
          continue;
        }
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j; try { j = JSON.parse(data); } catch { continue; }
        if (j.error) throw new Error(j.error.message || "Backend error");
        const d = j.choices?.[0]?.delta || {};
        if (d.content || d.reasoning_content) {
          if (tFirst == null) { tFirst = performance.now(); metaEl.textContent = ""; }
          if (d.content) working.querySelector(".lbl").textContent = "Writing…";
          nChunks++;
        }
        if (d.reasoning_content) reply.reasoning += d.reasoning_content;
        if (d.content) reply.content += d.content;
        if (j.usage) usage = j.usage;
        if (j.timings) timings = j.timings;
        if (d.content || d.reasoning_content) {
          if (reply.reasoning && !el.querySelector(".think")) {
            const det = document.createElement("details"); det.className = "think";
            det.innerHTML = "<summary>Reasoning</summary><div></div>"; el.insertBefore(det, body);
          }
          if (reply.reasoning) {
            el.querySelector(".think div").textContent = reply.reasoning;
            el.querySelector(".think summary").textContent = reply.content ? "Reasoning" : `Thinking… (${reply.reasoning.length} chars)`;
          }
          body.innerHTML = md(reply.content);
          box.scrollTop = box.scrollHeight;
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") reply.meta = "Stopped";
    else { reply.error = "Error: " + e.message; }
  } finally {
    const secs = tFirst ? (performance.now() - tFirst) / 1000 : 0;
    const n = timings?.predicted_n || usage?.completion_tokens || nChunks;
    const tps = timings?.predicted_per_second || (secs > 0 ? n / secs : null);
    if (!reply.error && !reply.meta) {
      reply.meta = `${p?.name || model} · ${n} tokens${tps ? ` · ${tps.toFixed(1)} tok/s` : ""}${tFirst ? ` · first token ${((tFirst - t0) / 1000).toFixed(1)}s` : ""}`;
    }
    if (reply.compacted) {
      reply.meta = `${reply.meta ? reply.meta + " · " : ""}the chat outgrew the context: ${reply.compacted.compacted_messages} earlier messages were summarized`;
    }
    abortCtl = null;
    $("#sendBtn").hidden = false; $("#stopBtn").hidden = true;
    saveConvs(); renderMessages(); pollStatus();
  }
}
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#input").value.trim();
  if (!text || abortCtl) return;
  $("#input").value = ""; autosize();
  send(text);
});
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
});
function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, window.innerHeight * 0.4) + "px"; }
$("#input").addEventListener("input", autosize);
$("#stopBtn").addEventListener("click", () => abortCtl?.abort());

// ── API page ──────────────────────────────────────────────────────────────
function renderApi() {
  const base = location.origin;
  $("#baseUrl").textContent = base + "/v1";
  const auth = store.get("key", "") ? ' -H "Authorization: Bearer $LLM_GATEWAY_KEY"' : "";
  const sel = $("#apiModel");
  if (!sel.options.length && profiles.length) {
    sel.innerHTML = byName(profiles).map((p) => modelOption(p.id, p.name, p.context,
      [p.tools ? "agent / coding tools" : "", p.speed_zh_tps ? `~${p.speed_zh_tps} tok/s` : ""].filter(Boolean).join(" · "))).join("");
    sel.value = store.get("apiModel", status.active || profiles[0].id);
    if (!sel.value) sel.value = profiles[0].id;
  }
  const example = sel.value || profiles[0]?.id || "qwen3-8b";
  $("#apiModelNote").textContent = `model id: ${example}`;
  const eps = [
    { m: "GET", p: "/v1/models", d: "List all models, with context size and whether each is loaded.", c: `curl ${base}/v1/models${auth}` },
    { m: "POST", p: "/v1/chat/completions", d: "Chat. If model differs from the loaded one, the gateway switches to it. While the GPU is busy the request waits in the queue (see Resources). Streaming supported.", c: `curl ${base}/v1/chat/completions${auth} \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${example}", "stream": true,\n       "messages": [{"role": "user", "content": "Hello"}]}'` },
    { m: "GET", p: "/api/status", d: "Loaded model, state (idle / loading / ready / error) and GPU memory.", c: `curl ${base}/api/status${auth}` },
    { m: "POST", p: "/api/activate", d: "Load a model ahead of time (unloads the current one).", c: `curl -X POST ${base}/api/activate${auth} \\\n  -H "Content-Type: application/json" -d '{"id": "${example}"}'` },
    { m: "POST", p: "/api/unload", d: "Unload the current model and free the GPU.", c: `curl -X POST ${base}/api/unload${auth}` },
    { m: "ENV", p: "Claude Code (/v1/messages)", d: "Anthropic Messages API. Pick a model with a large context (64K): Claude Code's prompt and tools alone are ~20K tokens.", c: `ANTHROPIC_BASE_URL=${base}
ANTHROPIC_AUTH_TOKEN=local
ANTHROPIC_MODEL=${example}
ANTHROPIC_DEFAULT_HAIKU_MODEL=${example}
CLAUDE_CODE_MAX_CONTEXT_TOKENS=${prof(example)?.context || 65536}
CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192
CLAUDE_CODE_ATTRIBUTION_HEADER=0` },
    { m: "TOML", p: "Codex (/v1/responses)", d: "OpenAI Responses API, in ~/.codex/config.toml.", c: `model = "${example}"
model_provider = "llmgw"
model_context_window = ${prof(example)?.context || 65536}

[model_providers.llmgw]
name = "llm-gateway"
base_url = "${base}/v1"
wire_api = "responses"` },
    { m: "PY", p: "OpenAI Python SDK", d: "Any OpenAI-compatible tool connects the same way.", c: `from openai import OpenAI\nclient = OpenAI(base_url="${base}/v1", api_key="${store.get("key", "") ? "<your key>" : "none"}")\nr = client.chat.completions.create(\n    model="${example}",\n    messages=[{"role": "user", "content": "Hello"}],\n)\nprint(r.choices[0].message.content)` },
  ];
  $("#endpoints").innerHTML = eps.map((e, i) => `<div class="ep">
    <div class="ep-head"><span class="method${e.m === "POST" ? " post" : ""}">${e.m}</span><code>${esc(e.p)}</code></div>
    <p>${esc(e.d)}</p><pre id="ep${i}">${esc(e.c)}</pre>
    <button type="button" class="ghost copy" data-copy="#ep${i}">Copy</button></div>`).join("");
}
document.addEventListener("click", async (e) => {
  const sel = e.target.closest("[data-copy]")?.dataset.copy;
  if (!sel) return;
  const text = $(sel).textContent;
  const btn = e.target.closest("button");
  try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
  catch { const r = document.createRange(); r.selectNodeContents($(sel)); const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = "Selected, press Ctrl+C"; }
  setTimeout(() => (btn.textContent = "Copy"), 1500);
});
$("#saveKey").addEventListener("click", () => { store.set("key", $("#apiKey").value.trim()); banner(""); init(); });

// ── boot ──────────────────────────────────────────────────────────────────
async function init() {
  const tpl = document.createElement("template"); tpl.id = "emptyTpl"; tpl.innerHTML = $("#emptyState")?.outerHTML || ""; if (!$("#emptyTpl")) document.body.appendChild(tpl);
  try {
    profiles = (await api("/api/profiles")).profiles;
    status = await api("/api/status");
    $("#keyRow").hidden = !status.auth_required;
  } catch (e) { if (e.message !== "API key required") banner("Cannot reach llm-gateway: " + e.message); return; }
  renderChatHead(); renderStatus(); renderConvs(); renderMessages(); renderApi();
}
enhanceSelect($("#chatModel"), "Model");
enhanceSelect($("#apiModel"), "Model");
$("#apiModel").addEventListener("change", () => { store.set("apiModel", $("#apiModel").value); renderApi(); });
showView(store.get("view", "chat"));
init();
setInterval(pollStatus, 2000);
