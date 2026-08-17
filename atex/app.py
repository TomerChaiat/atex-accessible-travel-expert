"""FastAPI adapter.

Deliberately thin: every handler delegates to atex/api.py and does nothing but
translate the returned (status, payload) into an HTTP response. Keeping the
logic out of here is what lets the test suite cover the API without FastAPI
installed.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import api

app = FastAPI(
    title="ATEX - Accessible Travel Expert",
    description="Supervisor multi-agent system for accessible trip planning.",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# No authentication of any kind, by requirement. Open CORS so the endpoints can
# be exercised from anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    status, html = api.index_html()
    return HTMLResponse(content=html, status_code=status)


@app.get("/api/team_info")
def team_info() -> JSONResponse:
    status, body = api.team_info()
    return JSONResponse(content=body, status_code=status)


@app.get("/api/agent_info")
def agent_info() -> JSONResponse:
    status, body = api.agent_info()
    return JSONResponse(content=body, status_code=status)


@app.get("/api/model_architecture")
def model_architecture() -> Response:
    status, body = api.model_architecture()
    if status != 200:
        return JSONResponse(content=body, status_code=status)
    return Response(content=body, media_type="image/png")


@app.post("/api/execute")
async def execute(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is a client error, not a crash
        payload = None

    debug = request.query_params.get("debug") in ("1", "true", "yes")
    status, body = api.execute(payload, debug=debug)
    return JSONResponse(content=body, status_code=status)


@app.get("/api/health")
def health() -> JSONResponse:
    from .config import settings

    return JSONResponse(content={"status": "ok", "backends": settings().backend_summary()})
