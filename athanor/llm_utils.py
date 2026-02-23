"""Shared utilities for Claude API calls across Athanor pipeline stages.

Provides JSON-retry logic: if Claude returns malformed JSON, the parse error
is fed back as a follow-up user message so Claude can self-correct.

Also handles RateLimitError with exponential backoff + jitter so parallel
workers survive org-level TPM limits without crashing the pipeline.

Prompt caching: long system prompts (≥ 1024 tokens, estimated by char count)
are automatically sent with cache_control=ephemeral so repeated calls within
the 5-minute TTL reuse the cached prefix — saving ~80 % of input token cost
for stages that call Claude many times with the same system prompt.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Optional, Union

import anthropic

log = logging.getLogger(__name__)

# Maximum total wait across all rate-limit retries (seconds).
# 8 attempts at 2,4,8,16,32,60,60,60 = ~4 min worst case.
_MAX_RATE_RETRIES = 8
_MAX_RATE_SLEEP = 60.0

# Anthropic requires ≥ 1024 tokens in a cached block.
# We estimate 1 token ≈ 4 characters (conservative for scientific English).
_CACHE_MIN_CHARS = 1024 * 4


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
    use_cache: bool = True,
) -> tuple[Optional[dict], str]:
    """Call Claude and retry automatically on rate limits and bad JSON.

    Rate-limit handling
    -------------------
    On ``RateLimitError`` the call is retried with exponential backoff + jitter
    (up to ``_MAX_RATE_RETRIES`` times, sleeping at most ``_MAX_RATE_SLEEP`` s).
    This keeps parallel workers alive through TPM bursts instead of crashing.

    JSON-retry handling
    -------------------
    On a JSON parse failure the error message is fed back to Claude as a
    follow-up user turn so it can self-correct in-context (up to ``max_retries``
    rounds).

    Prompt caching
    --------------
    When ``use_cache=True`` (default) and the system prompt is ≥ 1024 tokens
    (estimated), it is sent with ``cache_control={"type": "ephemeral"}`` so
    Anthropic caches it for 5 minutes.  Saves ~80 % of input token cost for
    stages that call Claude many times with the same system prompt (Stage 1
    processes ~20 papers; Stage 2 analyses ~20 gaps; Stage 3 generates ~10
    hypotheses — all with identical multi-KB system prompts).

    Returns ``(parsed_dict, raw_text)``.
    ``parsed_dict`` is ``None`` only if all retries are exhausted.
    """
    messages: list[dict] = [{"role": "user", "content": prompt}]
    raw = ""

    # Build the system parameter — plain string or cached block list
    if use_cache and len(system) >= _CACHE_MIN_CHARS:
        system_param: Union[str, list] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        log.debug("Prompt caching enabled for %d-char system prompt", len(system))
    else:
        system_param = system

    for attempt in range(max_retries + 1):
        # ── rate-limit-aware call ─────────────────────────────────────────────
        rate_attempt = 0
        while True:
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_param,
                    messages=messages,
                )
                break  # success
            except anthropic.RateLimitError as exc:
                rate_attempt += 1
                if rate_attempt > _MAX_RATE_RETRIES:
                    log.error(
                        "Rate limit exceeded after %d retries — giving up: %s",
                        _MAX_RATE_RETRIES, exc,
                    )
                    return None, ""
                # Exponential backoff with ±20 % jitter
                sleep_s = min(_MAX_RATE_SLEEP, 2 ** rate_attempt)
                sleep_s *= 0.8 + 0.4 * random.random()
                log.warning(
                    "Rate limited (attempt %d/%d) — sleeping %.1f s",
                    rate_attempt, _MAX_RATE_RETRIES, sleep_s,
                )
                time.sleep(sleep_s)

        raw = response.content[0].text
        text = strip_json_fences(raw)

        # Log cache stats when available (Anthropic returns these in usage)
        usage = response.usage
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            log.debug(
                "Prompt cache HIT — saved %d input tokens (created: %d)",
                usage.cache_read_input_tokens,
                getattr(usage, "cache_creation_input_tokens", 0),
            )

        # ── JSON parse / in-context retry ─────────────────────────────────────
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
