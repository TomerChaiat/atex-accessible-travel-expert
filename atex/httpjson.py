"""Minimal JSON-over-HTTP helper built on urllib.

Deliberately stdlib-only: it keeps httpx/requests out of the Vercel bundle and
lets the whole agent core be imported and tested with zero installs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:400]}")
        self.status = status
        self.body = body
        self.url = url


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """POST JSON and parse a JSON response, retrying transient failures.

    Retries 429 and 5xx with exponential backoff; 4xx errors are raised
    immediately since retrying a bad request only burns time we do not have.
    """
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    hdrs.update(headers or {})

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            if exc.code not in (429, 500, 502, 503, 504) or attempt == max_retries:
                raise HttpError(exc.code, text, url) from exc
            last_error = HttpError(exc.code, text, url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == max_retries:
                raise
            last_error = exc
        time.sleep(min(2.0 * (2**attempt), 8.0))

    raise last_error or RuntimeError("request failed")


def get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    hdrs = {"Accept": "application/json"}
    hdrs.update(headers or {})
    request = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise HttpError(exc.code, text, url) from exc


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Models wrap JSON in prose or fences more often than we would like, so fall
    back to slicing the outermost braces before giving up.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"could not parse a JSON object from model output: {text[:200]!r}")
