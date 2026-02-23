"""Shared utilities for Claude API calls across Athanor pipeline stages.

Provides JSON-retry logic: if Claude returns malformed JSON, the parse error
is fed back as a follow-up user message so Claude can self-correct.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

log = logging.getLogger(__name__)


def strip_json_fences(text: str) -> str:
    """Remove markdown code fences Claude sometimes emits despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def call_llm_json(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    system: str,
    prompt: str,
    max_retries: int = 2,
) -> tuple[Optional[dict], str]:
    """Call Claude and retry if the response is not valid JSON.

    Returns ``(parsed_dict, raw_text)``.
    ``parsed_dict`` is ``None`` if all retries are exhausted.
    ``raw_text`` is the last raw response from Claude.

    On a JSON parse failure the error message is fed back to Claude as a
    follow-up user turn so it can self-correct in-context.
    """
    messages: list[dict] = [{"role": "user", "content": prompt}]
    raw = ""

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        raw = response.content[0].text
        text = strip_json_fences(raw)
        try:
            data = json.loads(text)
            if attempt > 0:
                log.info("JSON parse succeeded on retry %d/%d", attempt, max_retries)
            return data, raw
        except json.JSONDecodeError as exc:
            if attempt < max_retries:
                log.warning(
                    "JSON parse failed (attempt %d/%d): %s — feeding error back to Claude",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response was not valid JSON. "
                        f"Parse error at position {exc.pos}: {exc.msg}. "
                        f"Reply with ONLY valid JSON matching the schema — "
                        f"no prose, no markdown fences, no trailing commas."
                    ),
                })
            else:
                log.error(
                    "JSON parse failed after %d attempt(s). Giving up.\nRaw: %s",
                    max_retries + 1,
                    raw[:400],
                )

    return None, raw
