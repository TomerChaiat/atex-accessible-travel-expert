"""LLM access: backend interface, LLMod.ai client, and the traced wrapper.

`TracedLLM` is the only way any module talks to a model. Because it owns both
the trace and the budget check, it is structurally impossible for a module to
make an untraced or over-budget call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .httpjson import HttpError, extract_json_object, post_json
from .tracing import RunTrace, estimate_tokens


class ModuleOutputError(RuntimeError):
    """A module's response could not be parsed into the expected shape."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMBackend(Protocol):
    name: str

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        module: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        json_object: bool = True,
        timeout: float = 60.0,
    ) -> LLMResult: ...


class LLModBackend:
    """OpenAI-compatible chat client for the course's LLMod.ai gateway.

    The gateway fronts an Azure deployment whose exact parameter support we
    cannot probe offline, so the request adapts: if it rejects `temperature` or
    `max_tokens`, we retry once with that parameter dropped or renamed rather
    than failing the run.
    """

    name = "llmod"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._drop_temperature = False
        self._use_max_completion_tokens = False

    def _payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        json_object: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.text_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not self._drop_temperature:
            payload["temperature"] = temperature
        key = "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
        payload[key] = max_tokens
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        module: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        json_object: bool = True,
        timeout: float = 60.0,
    ) -> LLMResult:
        url = f"{self._settings.llmod_base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._settings.llmod_api_key}"}

        for _ in range(3):
            payload = self._payload(
                system_prompt, user_prompt, temperature, max_tokens, json_object
            )
            try:
                data = post_json(
                    url,
                    payload,
                    headers=headers,
                    timeout=timeout,
                    max_retries=self._settings.budget.llm_max_retries,
                )
                break
            except HttpError as exc:
                body = exc.body.lower()
                if exc.status == 400 and "temperature" in body and not self._drop_temperature:
                    self._drop_temperature = True
                    continue
                if (
                    exc.status == 400
                    and "max_tokens" in body
                    and not self._use_max_completion_tokens
                ):
                    self._use_max_completion_tokens = True
                    continue
                raise
        else:
            raise RuntimeError("LLMod request failed after parameter adaptation")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLMod returned no choices: {str(data)[:300]}")
        text = (choices[0].get("message") or {}).get("content") or ""

        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


class TracedLLM:
    """Records every call into the run trace and enforces the budget."""

    def __init__(self, backend: LLMBackend, trace: RunTrace, settings: Settings):
        self.backend = backend
        self.trace = trace
        self.settings = settings

    def complete_json(
        self,
        module: str,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        repair: bool = True,
    ) -> dict[str, Any]:
        """Call the model and parse a JSON object, tracing the step either way.

        On a parse failure we spend at most one extra call on a repair attempt,
        and only when the budget still allows it.
        """
        result = self._call(module, system_prompt, user_prompt, temperature, max_tokens)

        try:
            parsed = extract_json_object(result.text)
        except ValueError as exc:
            self.trace.record(
                module,
                system_prompt,
                user_prompt,
                {"_unparsed_text": result.text, "_error": str(exc)},
                result.prompt_tokens,
                result.completion_tokens,
            )
            if not repair or self.trace.soft_exhausted():
                raise ModuleOutputError(f"{module}: {exc}") from exc

            repair_prompt = (
                f"{user_prompt}\n\nYour previous reply was not valid JSON. "
                "Reply with a single JSON object and nothing else."
            )
            retry = self._call(
                module, system_prompt, repair_prompt, temperature, max_tokens
            )
            try:
                parsed = extract_json_object(retry.text)
            except ValueError as exc2:
                self.trace.record(
                    module,
                    system_prompt,
                    repair_prompt,
                    {"_unparsed_text": retry.text, "_error": str(exc2)},
                    retry.prompt_tokens,
                    retry.completion_tokens,
                )
                raise ModuleOutputError(f"{module}: {exc2}") from exc2

            self.trace.record(
                module,
                system_prompt,
                repair_prompt,
                parsed,
                retry.prompt_tokens,
                retry.completion_tokens,
            )
            return parsed

        self.trace.record(
            module,
            system_prompt,
            user_prompt,
            parsed,
            result.prompt_tokens,
            result.completion_tokens,
        )
        return parsed

    def _call(
        self,
        module: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        self.trace.check_hard_limit()
        timeout = min(
            self.settings.budget.llm_timeout_s, max(5.0, self.trace.remaining_s())
        )
        result = self.backend.complete(
            system_prompt,
            user_prompt,
            module=module,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        # Fall back to an estimate when the provider reports no usage, so the
        # token budget still moves and cannot be silently bypassed.
        if not result.prompt_tokens:
            result.prompt_tokens = estimate_tokens(system_prompt + user_prompt)
        if not result.completion_tokens:
            result.completion_tokens = estimate_tokens(result.text)
        return result


def build_llm_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "llmod":
        return LLModBackend(settings)
    from .fake_backend import FakeLLMBackend

    return FakeLLMBackend()
