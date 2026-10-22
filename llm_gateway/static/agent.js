"use strict";
// Agent page. The agent runs on the server; this page starts turns, streams the turn's events
// (GET .../events, Server-Sent Events) and answers permission requests. It reuses $, esc, md,
// headers, store and showView from app.js.

const AG_MODES = { plan: "Plan (read only)", default: "Default (ask before edits and commands)", accept_edits: "Accept edits (ask before commands)", bypass: "Bypass (never ask)" };
const ag = { info: null, cur: null, sessions: null, reader: null, text: null, cards: new Map() };

async function agApi(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
  const data = await r.json().catch(() => ({}));
  if (r.status === 401) { showAuth(); throw new Error("API key required"); }
  if (!r.ok) throw new Error(data?.error?.message || r.statusText);
  return data;
}
const agBox = () => $("#agMessages");
const agPods = () => ag.info?.sandbox === "k8s";     // every agent in its own Kubernetes pod and folder
function agAdd(el) {
  // new output goes above the "working" indicator, which stays last while a turn runs
  if (ag.working?.isConnected) agBox().insertBefore(el, ag.working); else agBox().appendChild(el);
  agBox().scrollTop = agBox().scrollHeight;
  return el;
}
function agWorking(label) {
  if (!ag.working?.isConnected) { ag.working = workingEl(label); agBox().appendChild(ag.working); }
  ag.working.querySelector(".lbl").textContent = label;
  agBox().scrollTop = agBox().scrollHeight;
}
function agWorkingOff() { ag.working?.remove(); ag.working = null; }
function agEl(cls, html) { const d = document.createElement("div"); d.className = cls; d.innerHTML = html; return d; }

function agEmpty(extra = "") {
  const where = agPods() ? "Each agent runs in its own Kubernetes pod with its own folder." : "It only works inside the workspace folder.";
  agBox().innerHTML = `<div class="empty"><h2>Agent</h2><p>Give it a task: the agent reads and searches files, edits code, runs commands and git
    until it is done. Depending on the permission mode it asks before changing anything. ${where}</p>${extra}</div>`;
}

// ── setup ─────────────────────────────────────────────────────────────────
async function agInit() {
  try {
    ag.info = await agApi("/api/agent/info");
  } catch (e) {
    agEmpty(`<p class="notice error">${esc(e.message)}</p>`);
    $("#agComposer").hidden = true; $("#agWorkspaceRow").hidden = true; $(".agent-head").hidden = true;
    return;
  }
  $("#agModel").innerHTML = ag.info.models.map((m) => modelOption(m.id, m.name, m.context, "")).join("")
    || `<option value="">No model is marked "tools": true</option>`;
  $("#agModel").value = store.get("agModel", ag.info.models[0]?.id || "");
  if (!$("#agModel").value && ag.info.models[0]) $("#agModel").value = ag.info.models[0].id;
  // closed menu shows "default"; the open menu explains each mode
  $("#agMode").innerHTML = ag.info.modes.map((m) => `<option value="${m}" data-short="${m}">${AG_MODES[m] || m}</option>`).join("");
  $("#agMode").value = store.get("agMode", ag.info.default_mode);
  agRenderRoots();
  for (const err of ag.info.mcp_errors) banner(err);
  if (ag.info.error) banner(ag.info.error);
  await agLoadSessions();
  const last = store.get("agSession", null);
  if (last && ag.sessions.some((s) => s.id === last)) await agOpen(last);
  else agNewTask();
}
enhanceSelect($("#agModel"), "Model");
enhanceSelect($("#agMode"), "Permission mode");

// ── task list (drawer, like the chat list) ────────────────────────────────
async function agLoadSessions() {
  ag.sessions = (await agApi("/api/agent/sessions")).sessions;
  agRenderList();
}
function agRenderList() {
  const cur = ag.cur?.id;
  $("#agList").innerHTML = ag.sessions.map((s) => `<li class="${s.id === cur ? "on" : ""}">
    <button type="button" class="title" data-open="${esc(s.id)}" title="${esc(s.title || s.id)}">${esc(s.title || s.id)}</button>
    <button type="button" class="ren" data-rename="${esc(s.id)}" aria-label="Rename task">✎</button>
    <button type="button" class="del" data-del="${esc(s.id)}" aria-label="Delete task">✕</button></li>`).join("")
    || `<li class="none">No tasks yet</li>`;
}
function agSetListOpen(open) {
  $("#agConvs").classList.toggle("open", open);
  $("#agScrim").hidden = !open;
}
$("#agList").addEventListener("click", async (e) => {
  const open = e.target.closest("[data-open]")?.dataset.open;
  const del = e.target.closest("[data-del]")?.dataset.del;
  const ren = e.target.closest("[data-rename]")?.dataset.rename;
  if (ren) {
    const s = ag.sessions.find((x) => x.id === ren);
    inlineRename(e.target.closest("li"), s?.title || ren, async (title) => {
      await agApi(`/api/agent/sessions/${ren}`, { method: "PATCH", body: JSON.stringify({ title }) });
      await agLoadSessions();
    });
    return;
  }
  if (open) { agSetListOpen(false); try { await agOpen(open); } catch (err) { banner(err.message); } }
  if (del) {
    const what = agPods() ? "its pod and its whole folder" : "its saved conversation (files in the workspace stay)";
    if (!confirm(`Delete this task: ${what}?`)) return;
    try { await agApi(`/api/agent/sessions/${del}`, { method: "DELETE" }); } catch (err) { banner(err.message); return; }
    if (ag.cur?.id === del) agNewTask();
    await agLoadSessions();
  }
});
$("#agToggleList").addEventListener("click", () => agSetListOpen(!$("#agConvs").classList.contains("open")));
$("#agCloseList").addEventListener("click", () => agSetListOpen(false));
$("#agScrim").addEventListener("click", () => agSetListOpen(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("#agConvs").classList.contains("open")) agSetListOpen(false); });

// ── workspace row (collapsed by default) ──────────────────────────────────
// Suggestions for the workspace box: the empty default folder, the allowed roots, recently used folders.
function agRenderRoots() {
  if (agPods()) return;
  const all = [...ag.info.workspaces, ...store.get("agRecent", [])];
  $("#agRoots").innerHTML = [...new Set(all)].map((w) => `<option value="${esc(w)}">`).join("");
}
function agRemember(ws) {
  store.set("agRecent", [ws, ...store.get("agRecent", []).filter((w) => w !== ws)].slice(0, 8));
  agRenderRoots();
}
const agWsHint = () => agPods()
  ? `Each agent runs in its own Kubernetes pod and can only touch its own folder under ${ag.info.agents_dir}.`
  : "Change it any time (Enter applies). Allowed: " + ag.info.workspaces.join(", ");
function agSetWorkspace(path) {
  $("#agWorkspace").value = path;
  $("#agWsShort").textContent = path.split(/[\\/]/).filter(Boolean).slice(-2).join("/");
  $("#agWsNote").textContent = agWsHint();
}
$("#agWorkspaceRow").open = store.get("agWsOpen", false);
$("#agWorkspaceRow").addEventListener("toggle", () => store.set("agWsOpen", $("#agWorkspaceRow").open));

function agNewTask() {
  agDetach();
  ag.cur = null; store.set("agSession", null);
  // new tasks start in an empty folder (with pods: a new folder of their own)
  agSetWorkspace(agPods() ? `${ag.info.agents_dir}\\<new agent>\\workspace` : ag.info.default_workspace);
  $("#agTodos").hidden = true;
  if (ag.sessions) agRenderList();
  agEmpty();
  agBusy(false);
}

async function agOpen(id) {
  agDetach();
  ag.cur = await agApi(`/api/agent/sessions/${encodeURIComponent(id)}`);
  store.set("agSession", id);
  agRenderList();
  $("#agModel").value = ag.cur.model;
  $("#agMode").value = ag.cur.mode;
  agSetWorkspace(ag.cur.host_folder || ag.cur.workspace);
  agRenderTodos(ag.cur.todos);
  if (ag.cur.queued) {
    agRenderHistory(ag.cur.messages);
    agWaitStart(id);
  } else if (ag.cur.running) {
    // Show the history before the running turn, then replay the turn's events from the start.
    const msgs = ag.cur.messages;
    let start = msgs.length;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user" && !/^(\[Stop hook\]|\(Continue the task)/.test(msgs[i].content)) { start = i; break; }
    }
    agRenderHistory(msgs.slice(0, start + 1));
    agAttach(0);
  } else {
    agRenderHistory(ag.cur.messages);
    agBusy(false);
  }
}

// ── rendering ─────────────────────────────────────────────────────────────
function agUser(text) {
  const d = agEl("msg user", `<div class="body"></div>`);
  d.querySelector(".body").textContent = text.replace(/\n*<hook_context>[\s\S]*<\/hook_context>\s*$/, "");
  return agAdd(d);
}
function agAssistantText() {
  if (!ag.text) {
    const d = agAdd(agEl("msg assistant", `<div class="body"></div>`));
    ag.text = { el: d.querySelector(".body"), src: "" };
  }
  return ag.text;
}
function agToolCard(id, summary) {
  const det = document.createElement("details");
  det.className = "tool";
  det.innerHTML = `<summary><span class="st run">●</span><span class="sum"></span><span class="dur"></span></summary><pre>Running…</pre>`;
  det.querySelector(".sum").textContent = summary;
  ag.cards.set(id, det);
  ag.text = null;           // text after a tool call starts a new block
  return agAdd(det);
}
function agToolResult(id, ok, content, dur) {
  const det = ag.cards.get(id) || agToolCard(id, "(tool)");
  const st = det.querySelector(".st");
  st.className = "st " + (ok ? "ok" : "fail"); st.textContent = ok ? "✓" : "✗";
  det.querySelector("pre").textContent = content;
  if (dur) det.querySelector(".dur").textContent = dur + "s";
  if (!ok) det.open = true;
}
function agRenderTodos(todos) {
  const box = $("#agTodos");
  box.hidden = !todos?.length;
  box.innerHTML = (todos || []).map((t) => {
    const cls = t.status === "completed" ? "done" : t.status === "in_progress" ? "doing" : "";
    const mark = t.status === "completed" ? "✔" : t.status === "in_progress" ? "▶" : "○";
    return `<div class="${cls}">${mark} ${esc(t.content)}</div>`;
  }).join("");
}

function agRenderHistory(messages) {
  agBox().innerHTML = ""; ag.cards.clear(); ag.text = null;
  if (!messages.length) { agEmpty(); return; }
  for (const m of messages) {
    if (m.role === "user") { ag.text = null; agUser(m.content); }
    else if (m.role === "assistant") {
      if (m.content) { const t = agAssistantText(); t.src = m.content; t.el.innerHTML = md(m.content); }
      for (const c of m.tool_calls || []) {
        let args = c.function.arguments;
        try { args = JSON.stringify(JSON.parse(args)); } catch { /* keep raw */ }
        agToolCard(c.id, `${c.function.name} ${args}`.slice(0, 160));
      }
    } else if (m.role === "tool") {
      const failed = /^(Error \(|The user denied)/.test(m.content || "");
      agToolResult(m.tool_call_id, !failed, m.content || "");
    }
  }
}

function agPermission(ev) {
  const d = agEl("perm", `<div class="q">Allow this ${ev.risk === "exec" ? "command" : "change"}?</div><pre></pre>
    <div class="btns"><button type="button" class="primary" data-a="yes">Allow</button>
    <button type="button" class="ghost" data-a="always">Always allow this tool</button>
    <button type="button" class="danger" data-a="no">Deny</button></div>`);
  let detail = ev.summary;
  if (ev.name === "edit_file") detail += `\n\n- ${ev.arguments.old_string}\n+ ${ev.arguments.new_string}`;
  else if (ev.name === "write_file") detail += `\n\n${String(ev.arguments.content).slice(0, 2000)}`;
  d.querySelector("pre").textContent = detail;
  d.dataset.id = ev.id;
  d.querySelector(".btns").addEventListener("click", async (e) => {
    const a = e.target.closest("button")?.dataset.a;
    if (!a || d.classList.contains("answered")) return;
    d.classList.add("answered");
    d.querySelectorAll("button").forEach((b) => (b.disabled = true));
    d.querySelector(".q").textContent = a === "no" ? "Denied" : "Allowed";
    try {
      await agApi(`/api/agent/sessions/${ag.cur.id}/permission`, { method: "POST",
        body: JSON.stringify({ id: ev.id, allow: a !== "no", always: a === "always" }) });
    } catch (err) { d.querySelector(".q").textContent = "Could not send: " + err.message; }
  });
  agAdd(d);
}

function agHandle(ev) {
  switch (ev.type) {
    case "text": { const t = agAssistantText(); t.src += ev.text; t.el.innerHTML = md(t.src); agWorking("Writing…"); break; }
    case "reasoning": agWorking("Thinking…"); break;   // the reasoning text itself is not shown
    case "tool_call": agToolCard(ev.id, ev.summary); agWorking(`Running ${ev.name}…`); break;
    case "permission_request": agPermission(ev); agWorking("Waiting for your approval"); break;
    case "tool_result": {
      agWorking("Thinking…");
      agToolResult(ev.id, ev.ok, ev.content, ev.duration_s);
      const perm = agBox().querySelector(`.perm[data-id="${CSS.escape(ev.id)}"]:not(.answered)`);
      if (perm) { perm.classList.add("answered"); perm.querySelectorAll("button").forEach((b) => (b.disabled = true)); }
      break;
    }
    case "todos": agRenderTodos(ev.todos); break;
    case "notice": ag.text = null; agAdd(agEl("notice " + ev.level, esc(ev.text))); break;
    case "turn_end": {
      ag.text = null;
      agWorkingOff();
      const used = (ev.usage?.prompt_tokens || 0) + (ev.usage?.completion_tokens || 0);
      const why = { done: "Done", max_steps: "Step limit reached", loop: "Stopped: repeated calls", cancelled: "Stopped", error: "Error", blocked: "Blocked by a hook" }[ev.reason] || ev.reason;
      agAdd(agEl("turn-end", `${why} · ${ev.steps} steps${used ? ` · ${used} tokens` : ""}`));
      break;
    }
  }
}

// ── streaming a turn ──────────────────────────────────────────────────────
// A task waiting for its turn (agent/dispatch.py): show only a typing cursor and check every
// 2 s; the queue order itself is shown in the Resources panel.
async function agWaitStart(id) {
  agBusy(true);
  agWorking("Starting…");
  const pending = { remove: agWorkingOff };
  try {
    for (;;) {
      await new Promise((res) => setTimeout(res, 2000));
      if (ag.cur?.id !== id) return;
      const s = await agApi(`/api/agent/sessions/${id}`);
      if (s.queued) continue;
      pending.remove();
      if (s.start_error) { agAdd(agEl("notice error", esc(s.start_error))); agBusy(false); return; }
      if (s.running) { agAttach(0); return; }
      agBusy(false); return;                            // cancelled while waiting
    }
  } catch (e) { pending.remove(); agAdd(agEl("notice error", esc(e.message))); agBusy(false); }
}

function agBusy(on) {
  if (on) agWorking(ag.working?.querySelector(".lbl").textContent || "Thinking…"); else agWorkingOff();
  $("#agSend").hidden = on; $("#agStop").hidden = !on; $("#agModel").disabled = on; $("#agWorkspace").disabled = on || agPods(); }
function agDetach() { try { ag.reader?.cancel(); } catch { /* already closed */ } ag.reader = null; }

async function agAttach(start) {
  agBusy(true);
  const id = ag.cur.id;
  let i = start;
  try {
    for (let attempt = 0; attempt < 30; attempt++) {    // reconnect if the connection drops mid-turn
      const r = await fetch(`/api/agent/sessions/${id}/events?start=${i}`, { headers: headers(false) });
      if (!r.ok || !r.body) throw new Error((await r.json().catch(() => ({})))?.error?.message || r.statusText);
      ag.reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "", ended = false;
      for (;;) {
        const { value, done } = await ag.reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let k;
        while ((k = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, k); buf = buf.slice(k + 2);
          const data = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (!data) continue;
          const ev = JSON.parse(data.slice(6));
          i++;
          if (ag.cur?.id !== id) return;       // user switched session meanwhile
          agHandle(ev);
          if (ev.type === "turn_end") ended = true;
        }
      }
      if (ended || ag.cur?.id !== id || !ag.reader) break;
      await new Promise((res) => setTimeout(res, 1000));
    }
  } catch (e) {
    if (ag.cur?.id === id) agAdd(agEl("notice error", esc("Connection lost: " + e.message)));
  } finally {
    if (ag.cur?.id === id) { ag.reader = null; agBusy(false); agLoadSessions().catch(() => {}); }
  }
}

async function agSend(prompt) {
  try {
    if (!ag.cur) {
      if (agPods()) agBox().innerHTML = `<div class="notice">Starting this agent's pod…</div>`;
      ag.cur = await agApi("/api/agent/sessions", { method: "POST", body: JSON.stringify({
        workspace: $("#agWorkspace").value.trim(), model: $("#agModel").value, mode: $("#agMode").value }) });
      store.set("agSession", ag.cur.id);
      if (!agPods()) agRemember(ag.cur.workspace);
      agSetWorkspace(ag.cur.host_folder || ag.cur.workspace);
      agBox().innerHTML = "";
      await agLoadSessions();
    }
    ag.text = null;
    agUser(prompt);
    const r = await agApi(`/api/agent/sessions/${ag.cur.id}/turn`, { method: "POST", body: JSON.stringify({ prompt }) });
    if (r.position > 0) agWaitStart(ag.cur.id);        // another agent task runs first (see Resources)
    else agAttach(0);
  } catch (e) {
    agAdd(agEl("notice error", esc(e.message)));
  }
}

// ── controls ──────────────────────────────────────────────────────────────
$("#agComposer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#agInput").value.trim();
  if (!text || !$("#agStop").hidden) return;
  $("#agInput").value = "";
  agSend(text);
});
$("#agInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#agComposer").requestSubmit(); }
});
$("#agStop").addEventListener("click", () => ag.cur && agApi(`/api/agent/sessions/${ag.cur.id}/cancel`, { method: "POST" }).catch(() => {}));
$("#agNew").addEventListener("click", () => { agNewTask(); agSetListOpen(false); $("#agInput").focus(); });
// Changing the folder of an existing task: the conversation stays, the agent works in the new folder.
$("#agWorkspace").addEventListener("change", async () => {
  const ws = $("#agWorkspace").value.trim();
  if (!ag.cur || !ws || ws === ag.cur.workspace) return;
  try {
    ag.cur = await agApi(`/api/agent/sessions/${ag.cur.id}`, { method: "PATCH", body: JSON.stringify({ workspace: ws }) });
    agSetWorkspace(ag.cur.workspace); agRemember(ag.cur.workspace);
    agAdd(agEl("notice", esc("Workspace changed to " + ag.cur.workspace)));
  } catch (e) {
    agSetWorkspace(ag.cur.workspace);
    agAdd(agEl("notice error", esc(e.message)));
  }
});

$("#agMode").addEventListener("change", async () => {
  store.set("agMode", $("#agMode").value);
  if (ag.cur) { try { ag.cur = { ...ag.cur, ...(await agApi(`/api/agent/sessions/${ag.cur.id}`, { method: "PATCH", body: JSON.stringify({ mode: $("#agMode").value }) })) }; } catch (e) { banner(e.message); } }
});
$("#agModel").addEventListener("change", async () => {
  store.set("agModel", $("#agModel").value);
  if (ag.cur) { try { await agApi(`/api/agent/sessions/${ag.cur.id}`, { method: "PATCH", body: JSON.stringify({ model: $("#agModel").value }) }); } catch (e) { banner(e.message); } }
});

agInit();
