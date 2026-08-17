"""Zero-dependency development server.

Serves the same handlers as the FastAPI app using only the standard library, so
the project can be run and demonstrated before anything is pip-installed:

    python scripts/devserver.py            # http://127.0.0.1:8000

Production uses atex/app.py on Vercel. This exists so development never blocks
on an install.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex import api  # noqa: E402

PORT = 8000


class Handler(BaseHTTPRequestHandler):
    server_version = "ATEXDev/0.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path == "/":
            status, html = api.index_html()
            self._send(status, html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/team_info":
            self._send_json(*api.team_info())
        elif path == "/api/agent_info":
            self._send_json(*api.agent_info())
        elif path == "/api/health":
            from atex.config import settings

            self._send_json(200, {"status": "ok", "backends": settings().backend_summary()})
        elif path == "/api/model_architecture":
            status, body = api.model_architecture()
            if status == 200 and isinstance(body, bytes):
                self._send(200, body, "image/png")
            else:
                self._send_json(status, body)
        else:
            self._send_json(404, {"status": "error", "error": "not found",
                                  "response": None, "steps": []})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/execute":
            self._send_json(404, {"status": "error", "error": "not found",
                                  "response": None, "steps": []})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None

        debug = parse_qs(parsed.query).get("debug", ["0"])[0] in ("1", "true", "yes")
        self._send_json(*api.execute(payload, debug=debug))

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.command} {self.path}\n")


def main() -> None:
    from atex.config import settings

    backends = settings().backend_summary()
    print(f"ATEX dev server on http://127.0.0.1:{PORT}")
    print(f"  backends: {backends}")
    if backends["llm"] == "fake":
        print("  (offline mode: no LLMOD_API_KEY set, using the fake backend)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
