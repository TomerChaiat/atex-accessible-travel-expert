"""Endpoint logic, framework-free.

Every handler is a plain function returning `(status_code, payload)`. FastAPI in
app.py is a thin adapter over these, which means the whole API contract is
testable -- and is tested -- without installing a web framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import MODULE_NAMES
from .config import DATA_DIR, STATIC_DIR, Settings, settings as load_settings
from .graph import run_agent
from .sessions import build_session_store, sanitize_session_id

TEAM_FILE = DATA_DIR / "team.json"
AGENT_INFO_EXAMPLE_FILE = DATA_DIR / "agent_info_example.json"
ARCHITECTURE_PNG = STATIC_DIR / "architecture.png"

DESCRIPTION = (
    "ATEX (Accessible Travel Expert) is a supervisor multi-agent system that builds "
    "day-by-day travel itineraries for people with disabilities. The user describes their "
    "trip and their access needs in ordinary language; ATEX extracts a structured profile, "
    "searches a curated catalogue of places, checks each candidate against an accessibility "
    "knowledge base, and returns a paced itinerary in which every place carries an explicit "
    "accessibility verdict.\n\n"
    "What it CAN do: interpret mobility and sensory needs, select activities, hotels and "
    "restaurants for a supported city, judge each place as verified / concerns / unverified "
    "with cited evidence, pace days around a stated activity limit with built-in rest, and "
    "answer follow-up turns that adjust an existing plan.\n\n"
    "What it CANNOT do (constraints): it books nothing and takes no payment; it covers only "
    "the cities present in its curated catalogue; it has no live opening hours, prices or "
    "availability; and it will not assert that a place is accessible without supporting "
    "evidence. When the knowledge base is silent it returns 'unverified' and says so, rather "
    "than guessing. Unverified is a genuine answer, not a failure."
)

PURPOSE = (
    "Remove the research burden that makes accessible travel planning exhausting, by "
    "assembling a realistic itinerary in which accessibility is checked per place and "
    "clearly separated into verified, flagged, and unknown."
)

PROMPT_TEMPLATE = (
    "Destination: <city>\n"
    "Trip length: <number of days>\n"
    "Travellers: <who is going, e.g. 'family of four'>\n"
    "Access needs: <e.g. 'manual wheelchair, step-free entrances, accessible toilet'>\n"
    "Pace: <relaxed | moderate | packed, and max activities per day>\n"
    "Interests: <e.g. museums, parks, art, food>\n"
    "Accommodation: <needed or not>"
)

PROMPT_TEMPLATE_EXAMPLE = (
    "Destination: Amsterdam\n"
    "Trip length: 3 days\n"
    "Travellers: family of four\n"
    "Access needs: our daughter uses a manual wheelchair, we need step-free entrances\n"
    "Pace: relaxed, no more than two activities per day\n"
    "Interests: museums, parks\n"
    "Accommodation: we need a verified accessible hotel"
)

EXAMPLE_PROMPTS = (
    "We are a family of four visiting Amsterdam for three days. Our daughter uses a manual "
    "wheelchair. We prefer a relaxed pace, no more than two activities per day. We need a "
    "verified accessible hotel.",
    "Two days in Berlin, I use a powered wheelchair and I am sensitive to noise. I like "
    "history and quiet parks.",
)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message, "response": None, "steps": []}


# --------------------------------------------------------- GET /api/team_info
def team_info() -> tuple[int, dict[str, Any]]:
    data = _read_json(TEAM_FILE)
    if not isinstance(data, dict):
        return 500, _error("data/team.json is missing or unreadable")
    return 200, {
        "group_batch_order_number": data.get("group_batch_order_number", ""),
        "team_name": data.get("team_name", "ATEX"),
        "students": data.get("students", []),
    }


# -------------------------------------------------------- GET /api/agent_info
def _generate_example(prompt: str, settings: Settings) -> dict[str, Any]:
    """Run one example through the offline backend so the docs are never stale.

    Forced onto the fake LLM regardless of configuration: /api/agent_info is
    documentation and must not spend the project's token budget when called.
    """
    from dataclasses import replace

    offline = replace(settings, llm_backend="fake", embedding_backend="fake",
                      vector_backend="memory", repository_backend="local")
    result = run_agent(prompt, settings=offline)
    return {"prompt": prompt, "full_response": result.response, "steps": result.steps}


def agent_info(settings: Settings | None = None) -> tuple[int, dict[str, Any]]:
    cfg = settings or load_settings()

    cached = _read_json(AGENT_INFO_EXAMPLE_FILE)
    if isinstance(cached, dict) and cached.get("prompt_examples"):
        examples = cached["prompt_examples"]
    else:
        examples = [_generate_example(EXAMPLE_PROMPTS[0], cfg)]

    return 200, {
        "description": DESCRIPTION,
        "purpose": PURPOSE,
        "prompt_template": {
            "template": PROMPT_TEMPLATE,
            "example": PROMPT_TEMPLATE_EXAMPLE,
            "note": (
                "Plain prose works too; the template only makes the request easier to parse. "
                "Send {\"prompt\": \"...\"} to POST /api/execute. An optional \"session_id\" "
                "enables follow-up turns."
            ),
        },
        "modules": list(MODULE_NAMES),
        "prompt_examples": examples,
    }


# ------------------------------------------------ GET /api/model_architecture
def model_architecture() -> tuple[int, bytes | dict[str, Any]]:
    if not ARCHITECTURE_PNG.exists():
        return 500, _error(
            "architecture.png is missing. Run: python scripts/build_architecture_png.py"
        )
    try:
        return 200, ARCHITECTURE_PNG.read_bytes()
    except OSError as exc:
        return 500, _error(f"could not read architecture.png: {exc}")


# --------------------------------------------------------- POST /api/execute
def execute(
    payload: Any, settings: Settings | None = None, debug: bool = False
) -> tuple[int, dict[str, Any]]:
    cfg = settings or load_settings()

    if not isinstance(payload, dict):
        return 400, _error("request body must be a JSON object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return 400, _error("'prompt' is required and must be a non-empty string")
    if len(prompt) > 8000:
        return 400, _error("'prompt' is too long (limit 8000 characters)")

    session_id = sanitize_session_id(payload.get("session_id"))
    store = build_session_store(cfg)
    saved = store.get(session_id) if session_id else None

    try:
        result = run_agent(prompt.strip(), session_id=session_id, saved_state=saved, settings=cfg)
    except Exception as exc:  # noqa: BLE001 - the contract requires a JSON error
        return 200, _error(f"agent run failed: {type(exc).__name__}: {exc}")

    if result.error:
        body = _error(result.error)
        body["steps"] = result.steps  # keep the partial trace; it is the useful part
        return 200, body

    if session_id:
        state = result.session_state()
        state["turn_index"] = result.state.turn_index
        store.put(session_id, state)

    body: dict[str, Any] = {
        "status": "ok",
        "error": None,
        "response": result.response,
        "steps": result.steps,
    }
    if debug:
        # Non-standard, opt-in via ?debug=1. The default response keeps exactly
        # the four top-level fields the assignment specifies.
        body["_diagnostics"] = {
            "usage": result.usage,
            "notes": result.notes,
            "session_id": session_id,
        }
    return 200, body


# ------------------------------------------------------------------ index
def index_html() -> tuple[int, str]:
    from .config import WEB_DIR

    path = WEB_DIR / "index.html"
    try:
        return 200, path.read_text(encoding="utf-8")
    except OSError:
        return 500, "<h1>ATEX</h1><p>public/index.html is missing.</p>"
