"""Generate atex/static/architecture.png for GET /api/model_architecture.

The module labels are imported from atex.MODULE_NAMES rather than typed here,
so the diagram cannot drift from the names used in the execution trace. A test
asserts the same thing from the other direction.

    python scripts/build_architecture_png.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from atex import (  # noqa: E402
    ACCESSIBILITY_VALIDATOR,
    ACTIVITY_LOGISTICS_FINDER,
    SCHEDULE_PLANNER,
    SUPERVISOR,
    USER_PROFILE_AGENT,
)
from atex.config import STATIC_DIR  # noqa: E402

W, H = 1680, 1120
BG = (255, 255, 255)
INK = (22, 25, 29)
MUTED = (105, 114, 126)
LINE = (176, 184, 193)

BLUE = (31, 111, 235)
BLUE_FILL = (233, 241, 254)
GREEN = (26, 127, 69)
GREEN_FILL = (232, 246, 237)
PURPLE = (122, 78, 203)
PURPLE_FILL = (243, 238, 253)
AMBER = (154, 100, 0)
AMBER_FILL = (253, 245, 227)
GREY_FILL = (245, 247, 249)


def font(size: int, bold: bool = False):
    candidates = ["arialbd.ttf", "seguisb.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    candidates += ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font(38, True)
F_SUB = font(19)
F_BOX = font(23, True)
F_SMALL = font(16)
F_TINY = font(14)
F_LABEL = font(15, True)


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def centered(draw, text, fnt, cx, y, fill=INK):
    w, _ = text_size(draw, text, fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


def wrap(draw, text, fnt, max_width):
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if text_size(draw, trial, fnt)[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def box(draw, x, y, w, h, title, subtitle="", *, outline=LINE, fill=GREY_FILL, tag=""):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=fill, outline=outline, width=3)
    cx = x + w / 2
    ty = y + (18 if subtitle or tag else h / 2 - 14)

    if tag:
        centered(draw, tag, F_TINY, cx, ty, fill=outline)
        ty += 22

    centered(draw, title, F_BOX, cx, ty)
    ty += 32

    if subtitle:
        for line in wrap(draw, subtitle, F_SMALL, w - 32):
            centered(draw, line, F_SMALL, cx, ty, fill=MUTED)
            ty += 22
    return (x, y, x + w, y + h)


def arrow(draw, start, end, color=LINE, width=3, head=11, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        total = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(int(total / 12), 1)
        for i in range(steps):
            if i % 2:
                continue
            a = (x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps)
            b = (x1 + (x2 - x1) * (i + 1) / steps, y1 + (y2 - y1) * (i + 1) / steps)
            draw.line([a, b], fill=color, width=width)
    else:
        draw.line([start, end], fill=color, width=width)

    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    for sign in (1, -1):
        theta = angle + sign * math.radians(150)
        draw.line(
            [(x2, y2), (x2 + head * math.cos(theta), y2 + head * math.sin(theta))],
            fill=color,
            width=width,
        )


def main() -> Path:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    centered(draw, "ATEX - Accessible Travel Expert", F_TITLE, W / 2, 34)
    centered(
        draw,
        "Supervisor multi-agent architecture. Module names match every steps[].module in POST /api/execute.",
        F_SUB, W / 2, 84, fill=MUTED,
    )

    # --- Layer 1: client -------------------------------------------------
    box(draw, W / 2 - 230, 140, 460, 78,
        "Web GUI  /  POST /api/execute",
        "prompt in, response + full steps trace out",
        outline=LINE, fill=GREY_FILL)

    # --- Layer 2: supervisor ---------------------------------------------
    sup = box(draw, W / 2 - 330, 268, 660, 122, SUPERVISOR,
              "LLM-driven routing every turn. Enforces invariants, "
              "budget guardrails, and the forced-finalize path.",
              outline=BLUE, fill=BLUE_FILL, tag="ORCHESTRATOR")

    arrow(draw, (W / 2, 218), (W / 2, 266))
    arrow(draw, (W / 2 - 40, 268), (W / 2 - 40, 220))

    # --- Layer 3: the four specialists ------------------------------------
    top, height, width = 500, 190, 350
    gap = (W - 80 - 4 * width) / 3
    xs = [40 + i * (width + gap) for i in range(4)]

    specs = [
        (USER_PROFILE_AGENT, "Extracts a structured profile: mobility, sensory needs, pace, budget.",
         GREEN, GREEN_FILL, "1 LLM call"),
        (ACTIVITY_LOGISTICS_FINDER, "ReAct agent over live place search. Bounded to 4 iterations.",
         AMBER, AMBER_FILL, "ReAct loop"),
        (ACCESSIBILITY_VALIDATOR, "RAG verdicts: supported / flagged / unknown, with cited evidence.",
         PURPLE, PURPLE_FILL, "RAG, batched"),
        (SCHEDULE_PLANNER, "Day-by-day itinerary, geographic grouping, built-in rest.",
         BLUE, BLUE_FILL, "1 LLM call"),
    ]

    for x, (name, desc, colour, fill, tag) in zip(xs, specs):
        box(draw, x, top, width, height, name, desc, outline=colour, fill=fill, tag=tag)
        cx = x + width / 2
        arrow(draw, (cx - 14, 390), (cx - 14, top - 2), color=colour)   # dispatch
        arrow(draw, (cx + 14, top - 2), (cx + 14, 390), color=LINE)      # result

    # --- Layer 4: data and tools ------------------------------------------
    data_top = 762
    box(draw, xs[1], data_top, width, 130, "Live place discovery",
        "Google Places API (New). Local JSON sample when unset.",
        outline=AMBER, fill=GREY_FILL, tag="PLACE SEARCH TOOLS")
    box(draw, xs[2], data_top, width, 130, "Accessibility knowledge base",
        "Pinecone + text-embedding-3-small. In-memory cosine when unset.",
        outline=PURPLE, fill=GREY_FILL, tag="VECTOR STORE")

    arrow(draw, (xs[1] + width / 2, top + height), (xs[1] + width / 2, data_top - 2), color=AMBER)
    arrow(draw, (xs[2] + width / 2, top + height), (xs[2] + width / 2, data_top - 2), color=PURPLE)

    # --- Layer 5: cross-cutting -------------------------------------------
    box(draw, 40, 940, W - 80, 128,
        "TracedLLM  ->  LLMod.ai  (MB5R2CF-azure/gpt-5.4-mini)",
        "Every module's only route to a model. Records {module, system_prompt, user_prompt, response} "
        "into steps[] and enforces the run budget: 8 supervisor turns, 20 LLM calls, 60k tokens, 240s. "
        "Swaps to an offline fake backend when no API key is present.",
        outline=BLUE, fill=BLUE_FILL, tag="CROSS-CUTTING: TRACING + BUDGET")

    for x in xs:
        arrow(draw, (x + width / 2, top + height if x in (xs[0], xs[3]) else data_top + 130),
              (x + width / 2, 938), color=LINE, width=2, dashed=True)

    draw.text((40, H - 26), "Generated by scripts/build_architecture_png.py", font=F_TINY, fill=MUTED)

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    out = STATIC_DIR / "architecture.png"
    image.save(out, "PNG")
    return out


if __name__ == "__main__":
    path = main()
    # Print a relative path: the absolute one may contain characters the
    # Windows console codepage cannot encode.
    print(f"wrote atex/static/{path.name} ({path.stat().st_size / 1024:.0f} KB)")
