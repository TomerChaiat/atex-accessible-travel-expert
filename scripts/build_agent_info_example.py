"""Pre-compute the worked examples served by GET /api/agent_info.

Caching them means /api/agent_info is a file read rather than an agent run, so
documenting the agent never spends the project's token budget.

    python scripts/build_agent_info_example.py           # offline fake backend
    python scripts/build_agent_info_example.py --real    # real LLM (costs money)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.api import AGENT_INFO_EXAMPLE_FILE, EXAMPLE_PROMPTS  # noqa: E402
from atex.config import settings  # noqa: E402
from atex.graph import run_agent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="use the configured LLM backend")
    parser.add_argument("--count", type=int, default=1, help="how many examples to generate")
    args = parser.parse_args()

    cfg = settings(refresh=True)
    if not args.real:
        cfg = replace(
            cfg,
            llm_backend="fake",
            embedding_backend="fake",
            vector_backend="memory",
            repository_backend="local",
        )

    print(f"Generating with backends: {cfg.backend_summary()}")

    examples = []
    for prompt in EXAMPLE_PROMPTS[: max(1, args.count)]:
        print(f"  running: {prompt[:70]}...")
        result = run_agent(prompt, settings=cfg)
        if result.error:
            print(f"  ! run failed: {result.error}")
            return 1
        examples.append({
            "prompt": prompt,
            "full_response": result.response,
            "steps": result.steps,
        })
        print(f"    {len(result.steps)} steps, {result.usage['total_tokens']} tokens")

    AGENT_INFO_EXAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AGENT_INFO_EXAMPLE_FILE.write_text(
        json.dumps(
            {
                "_generated_with": cfg.backend_summary(),
                "_note": (
                    "Regenerate with scripts/build_agent_info_example.py whenever prompts or "
                    "module names change, so /api/agent_info never shows a stale trace."
                ),
                "prompt_examples": examples,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote data/{AGENT_INFO_EXAMPLE_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
