// TESAR-Live dashboard. Vanilla JS, no external resources. The DOM is built with
// createElement/textContent only (the Content-Security-Policy forbids inline code).
"use strict";

(() => {
  const PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
                   "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#d62728"];
  const STATE_COLORS = { ACTIVE: "ok", SUSPECT: "warn", DEAD: "off", QUARANTINED: "bad", LEFT: "gone" };
  const WARN_EVENTS = new Set(["attempt_timeout", "attempt_lost", "worker_state", "task_fallback_local",
                               "energy_budget_exhausted", "worker_ineligible_battery", "attempt_late"]);
  const ERROR_EVENTS = new Set(["verification_result", "worker_quarantined", "verification_skipped"]);
  const MAX_EVENTS = 300;

  let csrf = "";
  let demo = false;
  let grid = null;
  let tiles = [];
  let workers = [];
  let events = [];
  const colors = new Map();
  const flashes = new Map(); // "row,col" -> time until which the tile flashes red

  const $ = (id) => document.getElementById(id);

  function el(tag, text, cls) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (cls) node.className = cls;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function colorOf(workerId) {
    if (!workerId) return "#bbbbbb";
    if (!colors.has(workerId)) colors.set(workerId, PALETTE[colors.size % PALETTE.length]);
    return colors.get(workerId);
  }

  async function api(path, options) {
    const opts = Object.assign({ credentials: "same-origin" }, options || {});
    const response = await fetch(path, opts);
    if (response.status === 401) {
      window.location.assign("/login");
      throw new Error("not logged in");
    }
    return response;
  }

  function post(path, body) {
    return api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-MDC-CSRF": csrf },
      body: JSON.stringify(body),
    });
  }

  // ------------------------------------------------------------------ header

  function renderHeader(c, metrics) {
    $("address").textContent = "Workers connect to " + c.address;
    $("job-status").textContent = c.job_status;
    $("policy").textContent = c.policy;
    $("alpha").textContent = c.alpha;
    $("elapsed").textContent = metrics && metrics.elapsed_s !== undefined ? metrics.elapsed_s.toFixed(1) + " s" : "–";
    const total = metrics && metrics.tiles_total ? metrics.tiles_total : 0;
    const done = metrics && metrics.tiles_done ? metrics.tiles_done : 0;
    $("progress-bar").style.width = total ? (100 * done / total).toFixed(1) + "%" : "0%";
    $("tile-count").textContent = total ? done + " / " + total : "";
  }

  // ------------------------------------------------------------------- tiles

  function renderTiles() {
    const canvas = $("grid");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!grid) return;
    const cell = Math.max(2, Math.floor(Math.min(canvas.width / grid.cols, canvas.height / grid.rows)));
    canvas.height = cell * grid.rows;
    const now = Date.now();
    for (const [row, col, status, producer] of tiles) {
      const x = col * cell;
      const y = row * cell;
      const key = row + "," + col;
      if (flashes.has(key) && flashes.get(key) > now) {
        ctx.fillStyle = "#d62728";
      } else if (status === "pending" || status === "invalidated") {
        ctx.fillStyle = "#dddddd";
      } else {
        ctx.fillStyle = colorOf(producer);
      }
      ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      if (status === "running") {
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.beginPath();
        for (let d = -cell; d < cell; d += 6) {
          ctx.moveTo(x + d, y + cell);
          ctx.lineTo(x + d + cell, y);
        }
        ctx.stroke();
      } else if (status === "verifying") {
        ctx.strokeStyle = "#f2c200";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4);
        ctx.lineWidth = 1;
      } else if (status === "invalidated") {
        ctx.strokeStyle = "#d62728";
        ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
      } else if (status === "verified" && cell >= 12) {
        ctx.strokeStyle = "#ffffff";
        ctx.beginPath();
        ctx.moveTo(x + cell * 0.3, y + cell * 0.55);
        ctx.lineTo(x + cell * 0.45, y + cell * 0.7);
        ctx.lineTo(x + cell * 0.72, y + cell * 0.32);
        ctx.stroke();
      }
    }
    renderLegend();
  }

  function renderLegend() {
    const legend = $("legend");
    clear(legend);
    for (const w of workers) {
      const item = el("li", w.name);
      const swatch = el("span", "", "swatch");
      swatch.style.backgroundColor = colorOf(w.worker_id);
      item.prepend(swatch);
      legend.appendChild(item);
    }
  }

  // ----------------------------------------------------------------- workers

  function controlButton(workerId, label, action, value) {
    const button = el("button", label, "small");
    button.type = "button";
    button.addEventListener("click", async () => {
      const body = { action: action };
      if (value !== undefined) body.value = value;
      const r = await post("/api/workers/" + encodeURIComponent(workerId) + "/control", body);
      if (!r.ok) button.textContent = label + " ✗";
    });
    return button;
  }

  function renderWorkers() {
    const body = $("workers");
    clear(body);
    $("controls-head").hidden = !demo;
    for (const w of workers) {
      const tr = el("tr");
      const name = el("td", w.name);
      const swatch = el("span", "", "swatch");
      swatch.style.backgroundColor = colorOf(w.worker_id);
      name.prepend(swatch);
      tr.appendChild(name);
      tr.appendChild(el("td", w.device_class + (w.trusted ? " (local)" : "")));
      const state = el("td");
      state.appendChild(el("span", w.state, "badge " + (STATE_COLORS[w.state] || "off")));
      tr.appendChild(state);
      const rep = el("td");
      const bar = el("div", "", "rep");
      const fill = el("div");
      fill.style.width = (100 * (w.rep || 0)).toFixed(0) + "%";
      bar.appendChild(fill);
      rep.appendChild(bar);
      rep.appendChild(el("span", (w.rep || 0).toFixed(2), "muted"));
      tr.appendChild(rep);
      tr.appendChild(el("td", w.lambda !== undefined ? w.lambda.toExponential(1) : "–"));
      tr.appendChild(el("td", w.rtt_ms));
      tr.appendChild(el("td", w.bw_kbps !== undefined ? (w.bw_kbps / 8).toFixed(0) : "–"));
      const bat = w.battery || {};
      tr.appendChild(el("td", bat.pct === undefined || bat.pct === null ? "–" : bat.pct.toFixed(0) + "%" + (bat.charging ? " ⚡" : "")));
      tr.appendChild(el("td", w.tiles_done + " / " + w.failed + " / " + w.wrong));
      const meas = w.energy_measured_j ? w.energy_measured_j.toFixed(1) : "–";
      tr.appendChild(el("td", (w.energy_spent_j || 0).toFixed(2) + " / " + meas));
      tr.appendChild(el("td", w.inflight));
      tr.appendChild(el("td", w.avg_chunk_tiles ? w.avg_chunk_tiles.toFixed(1) : "–"));
      if (demo) {
        const cell = el("td", "", "controls");
        if (!w.trusted) {
          cell.appendChild(controlButton(w.worker_id, "Crash", "crash"));
          cell.appendChild(controlButton(w.worker_id, "Make malicious", "set_malicious", 1.0));
          cell.appendChild(controlButton(w.worker_id, "Slow ×3", "set_slowdown", 3.0));
          cell.appendChild(controlButton(w.worker_id, "Normal", "set_slowdown", 1.0));
        }
        tr.appendChild(cell);
      }
      body.appendChild(tr);
    }
  }

  // ----------------------------------------------------------------- metrics

  function renderMetrics(m, result) {
    const list = $("metrics");
    clear(list);
    const rows = [
      ["Elapsed", m.elapsed_s !== undefined ? m.elapsed_s + " s" : "–"],
      ["Attempts", m.attempts],
      ["ok / timeout / lost / error / late",
       [m.ok, m.timeout, m.lost, m.error, m.late].map((v) => (v === undefined ? "–" : v)).join(" / ")],
      ["Replicas", m.replicas],
      ["Verifications (mismatches)", m.verifications === undefined ? "–" : m.verifications + " (" + m.mismatches + ")"],
    ];
    if (result) {
      rows.push(["Makespan", result.makespan_s + " s"]);
      rows.push(["Speedup vs local", result.speedup]);
    }
    for (const [k, v] of rows) {
      const wrap = el("div");
      wrap.appendChild(el("dt", k));
      wrap.appendChild(el("dd", v === undefined || v === null ? "–" : v));
      list.appendChild(wrap);
    }
    const box = $("result");
    if (result) {
      box.hidden = false;
      const badge = $("result-badge");
      badge.textContent = result.output_correct ? "correct ✓" : "incorrect ✗";
      badge.className = "badge " + (result.output_correct ? "ok" : "bad");
      const img = $("output");
      const src = "/api/output.png?t=" + encodeURIComponent(result.makespan_s);
      if (img.getAttribute("src") !== src) img.setAttribute("src", src);
    } else {
      box.hidden = true;
    }
  }

  // ------------------------------------------------------------------ events

  function severity(e) {
    if (ERROR_EVENTS.has(e.type)) return e.type === "verification_result" && e.match ? "info" : "error";
    if (WARN_EVENTS.has(e.type)) return "warn";
    return "info";
  }

  function describe(e) {
    const skip = new Set(["type", "t", "wall"]);
    const parts = [];
    for (const [k, v] of Object.entries(e)) {
      if (!skip.has(k)) parts.push(k + "=" + (typeof v === "object" ? JSON.stringify(v) : v));
    }
    return parts.join(" ");
  }

  function renderEvents() {
    const list = $("events");
    clear(list);
    const filter = $("filter").value.trim().toLowerCase();
    for (let i = events.length - 1; i >= 0 && list.childElementCount < 200; i--) {
      const e = events[i];
      const text = describe(e);
      if (filter && !(e.type + " " + text).toLowerCase().includes(filter)) continue;
      const item = el("li", "", severity(e));
      item.appendChild(el("span", (e.t !== undefined ? e.t.toFixed(2) : "") + " ", "muted"));
      item.appendChild(el("strong", e.type + " "));
      item.appendChild(el("span", text));
      list.appendChild(item);
    }
  }

  function onEvent(e) {
    events.push(e);
    if (events.length > MAX_EVENTS) events = events.slice(-MAX_EVENTS);
    if (e.type === "verification_result" && e.replaced) {
      for (const tid of e.replaced) {
        const m = /^s1_r(\d+)_c(\d+)$/.exec(tid);
        if (m) flashes.set(Number(m[1]) + "," + Number(m[2]), Date.now() + 1000);
      }
    }
  }

  // ------------------------------------------------------------------- state

  function apply(data, full) {
    if (full) {
      grid = data.grid;
      events = (data.events || []).slice(-MAX_EVENTS);
    }
    if (data.grid !== undefined) grid = data.grid;
    tiles = data.tiles || [];
    workers = data.workers || [];
    renderHeader(data.coordinator, data.metrics);
    renderTiles();
    renderWorkers();
    renderMetrics(data.metrics || {}, data.result);
    renderEvents();
  }

  function connect() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + window.location.host + "/ws");
    ws.addEventListener("message", (msg) => {
      const m = JSON.parse(msg.data);
      if (m.kind === "state_full") apply(m.data, true);
      else if (m.kind === "state") apply(m.data, false);
      else if (m.kind === "event") onEvent(m.data);
    });
    ws.addEventListener("close", () => window.setTimeout(connect, 2000));
  }

  async function loadImages() {
    const r = await api("/api/images");
    const data = await r.json();
    const select = $("f-image");
    clear(select);
    for (const name of data.images) select.appendChild(el("option", name));
  }

  async function startJob(ev) {
    ev.preventDefault();
    const body = {
      policy: $("f-policy").value,
      alpha: Number($("f-alpha").value),
      image: $("f-image").value,
      blur_iterations: Number($("f-iters").value),
      verification: $("f-verify").checked,
      adaptive: $("f-adaptive").checked,
    };
    const r = await post("/api/job/start", body);
    const msg = $("form-msg");
    if (r.status === 202) msg.textContent = "Job started.";
    else if (r.status === 409) msg.textContent = "A job is already running.";
    else msg.textContent = "Rejected (" + r.status + ").";
  }

  async function init() {
    const r = await api("/api/state");
    const state = await r.json();
    csrf = state.csrf;
    demo = state.demo_controls;
    apply(state, true);
    await loadImages();
    $("f-alpha").addEventListener("input", () => { $("f-alpha-out").textContent = $("f-alpha").value; });
    $("job-form").addEventListener("submit", startJob);
    $("filter").addEventListener("input", renderEvents);
    $("logout").addEventListener("click", async () => {
      await post("/logout", {});
      window.location.assign("/login");
    });
    connect();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
