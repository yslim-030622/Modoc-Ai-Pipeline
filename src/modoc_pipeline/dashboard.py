"""Local dashboard for watching agentic pipeline runs."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def serve_dashboard(*, host: str, port: int, logs_dir: Path) -> None:
    handler = _make_handler(logs_dir=logs_dir)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard listening at http://{host}:{port}")
    print(f"Watching logs in {logs_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


def build_runs_payload(logs_dir: Path, *, limit: int = 30) -> dict[str, Any]:
    runs = []
    if logs_dir.exists():
        run_dirs = [path for path in logs_dir.iterdir() if path.is_dir()]
        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for run_dir in run_dirs[:limit]:
            runs.append(_run_summary(run_dir))
    return {"runs": runs}


def build_run_detail(logs_dir: Path, run_id: str) -> dict[str, Any]:
    safe_id = Path(run_id).name
    run_dir = logs_dir / safe_id
    if not run_dir.exists() or not run_dir.is_dir():
        return {"error": "run_not_found", "run_id": safe_id}
    summary = _run_summary(run_dir)
    summary["pipeline_state"] = _read_json(run_dir / "pipeline_state.json", {})
    summary["reviews"] = {
        path.stem: _read_json(path, {})
        for path in sorted(run_dir.glob("*review_report.json"))
    }
    return summary


def _make_handler(*, logs_dir: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_html(_HTML)
                return
            if path == "/api/runs":
                self._send_json(build_runs_payload(logs_dir))
                return
            if path.startswith("/api/runs/"):
                run_id = unquote(path.removeprefix("/api/runs/"))
                self._send_json(build_run_detail(logs_dir, run_id))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib API
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def _run_summary(run_dir: Path) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json", {})
    meme_status = _read_json(run_dir / "meme_status.json", {})
    source = _read_json(run_dir / "source.json", {})
    trace = _read_json(run_dir / "agent_trace.json", [])
    pipeline_state = _read_json(run_dir / "pipeline_state.json", {})
    latest = trace[-1] if isinstance(trace, list) and trace else {}
    rendered = pipeline_state.get("rendered_videos") or meme_status.get("videos") or []
    return {
        "run_id": run_dir.name,
        "row_number": source.get("row_number") or status.get("source_row"),
        "status": status.get("status") or meme_status.get("status") or "running",
        "failure_status": pipeline_state.get("failure_status"),
        "failure_message": pipeline_state.get("failure_message") or status.get("error"),
        "latest_agent": latest.get("agent"),
        "latest_status": latest.get("status"),
        "latest_message": latest.get("message"),
        "agent_trace": trace if isinstance(trace, list) else [],
        "rendered_videos": rendered if isinstance(rendered, list) else [],
        "post_render_attempts": pipeline_state.get("post_render_repair_attempts", 0),
        "max_post_render_repairs": pipeline_state.get("max_post_render_repairs"),
        "updated_at": latest.get("recorded_at") or status.get("generated_at"),
    }


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MoDoc Agent Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d7dce2;
      --text: #171b21;
      --muted: #68717d;
      --ok: #176f45;
      --warn: #9a5b00;
      --bad: #b42318;
      --info: #2459a6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { font-size: 17px; margin: 0; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 420px) 1fr;
      min-height: calc(100vh - 56px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      padding: 12px;
      overflow: auto;
    }
    section { padding: 16px; overflow: auto; }
    button.run {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 10px;
      cursor: pointer;
    }
    button.run.active { border-color: var(--info); outline: 2px solid rgba(36, 89, 166, 0.12); }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .title { font-weight: 700; overflow-wrap: anywhere; }
    .muted { color: var(--muted); font-size: 12px; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      background: #f9fafb;
      font-size: 12px;
      white-space: nowrap;
    }
    .ok { color: var(--ok); border-color: rgba(23, 111, 69, 0.28); }
    .bad { color: var(--bad); border-color: rgba(180, 35, 24, 0.28); }
    .warn { color: var(--warn); border-color: rgba(154, 91, 0, 0.28); }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfd; }
    .metric b { display: block; font-size: 18px; margin-top: 4px; overflow-wrap: anywhere; }
    .trace { display: grid; gap: 8px; }
    .step {
      display: grid;
      grid-template-columns: 190px 92px 1fr 72px;
      gap: 10px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      padding: 9px 0;
    }
    .step:last-child { border-bottom: 0; }
    code { background: #eef1f5; padding: 2px 5px; border-radius: 5px; }
    a { color: var(--info); text-decoration: none; overflow-wrap: anywhere; }
    a:hover { text-decoration: underline; }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 42vh; }
      .grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .step { grid-template-columns: 1fr; gap: 4px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MoDoc Agent Dashboard</h1>
    <span class="muted" id="last-refresh">Loading</span>
  </header>
  <main>
    <aside id="runs"></aside>
    <section id="detail">
      <div class="panel">Select a run to inspect agent state.</div>
    </section>
  </main>
<script>
let selectedRun = null;
let latestRuns = [];

function cls(status) {
  const text = String(status || "").toLowerCase();
  if (text.includes("succeed") || text === "passed") return "ok";
  if (text.includes("fail")) return "bad";
  if (text.includes("running") || text.includes("repair")) return "warn";
  return "";
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

async function loadRuns() {
  const res = await fetch("/api/runs", {cache: "no-store"});
  const data = await res.json();
  latestRuns = data.runs || [];
  if (!selectedRun && latestRuns.length) selectedRun = latestRuns[0].run_id;
  renderRuns();
  renderDetail();
  document.getElementById("last-refresh").textContent = "Updated " + new Date().toLocaleTimeString();
}

function renderRuns() {
  const root = document.getElementById("runs");
  root.innerHTML = latestRuns.map(run => `
    <button class="run ${run.run_id === selectedRun ? "active" : ""}" onclick="selectRun('${esc(run.run_id)}')">
      <div class="row">
        <span class="title">${esc(run.run_id)}</span>
        <span class="pill ${cls(run.status || run.latest_status)}">${esc(run.status || run.latest_status || "running")}</span>
      </div>
      <div class="muted">Row ${esc(run.row_number || "?")} · ${esc(run.latest_agent || "pending")}</div>
      <div class="muted">${esc(run.latest_message || run.failure_message || "")}</div>
    </button>
  `).join("") || '<div class="panel">No runs found.</div>';
}

function selectRun(runId) {
  selectedRun = runId;
  renderRuns();
  renderDetail();
}

function renderDetail() {
  const run = latestRuns.find(item => item.run_id === selectedRun);
  const root = document.getElementById("detail");
  if (!run) {
    root.innerHTML = '<div class="panel">No run selected.</div>';
    return;
  }
  const videos = (run.rendered_videos || []).map(video => `
    <div><span class="pill">${esc(video.language)}</span> <a href="file://${esc(video.path)}">${esc(video.path)}</a></div>
  `).join("") || '<div class="muted">No rendered videos yet.</div>';
  const trace = (run.agent_trace || []).map(step => `
    <div class="step">
      <div><b>${esc(step.agent)}</b><div class="muted">${esc(step.recorded_at || "")}</div></div>
      <div><span class="pill ${cls(step.status)}">${esc(step.status)}</span></div>
      <div>${esc(step.message || "")}</div>
      <div class="muted">${esc(step.duration_seconds ?? "")}s</div>
    </div>
  `).join("");
  root.innerHTML = `
    <div class="panel">
      <div class="row">
        <div>
          <div class="title">${esc(run.run_id)}</div>
          <div class="muted">Row ${esc(run.row_number || "?")}</div>
        </div>
        <span class="pill ${cls(run.status || run.latest_status)}">${esc(run.status || run.latest_status || "running")}</span>
      </div>
      ${run.failure_message ? `<p class="bad">${esc(run.failure_message)}</p>` : ""}
    </div>
    <div class="grid">
      <div class="metric"><span class="muted">Latest agent</span><b>${esc(run.latest_agent || "pending")}</b></div>
      <div class="metric"><span class="muted">Latest status</span><b>${esc(run.latest_status || "pending")}</b></div>
      <div class="metric"><span class="muted">Post-render repairs</span><b>${esc(run.post_render_attempts || 0)} / ${esc(run.max_post_render_repairs ?? "-")}</b></div>
      <div class="metric"><span class="muted">Steps</span><b>${esc((run.agent_trace || []).length)}</b></div>
    </div>
    <div class="panel"><h2 style="font-size:15px;margin:0 0 10px;">Rendered Videos</h2>${videos}</div>
    <div class="panel"><h2 style="font-size:15px;margin:0 0 10px;">Agent Trace</h2><div class="trace">${trace}</div></div>
  `;
}

loadRuns().catch(console.error);
setInterval(() => loadRuns().catch(console.error), 2000);
</script>
</body>
</html>
"""
