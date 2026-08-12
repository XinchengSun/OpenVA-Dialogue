#!/usr/bin/env python3
"""Minimal streaming LLM probe that never prints credentials or response text."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipecat_dystream.custom_cascade import _llm_extra_body, _llm_provider


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def main() -> None:
    model = _required("PIPECAT_LLM_MODEL")
    base_url = _required("PIPECAT_LLM_BASE_URL")
    client = OpenAI(
        api_key=_required("PIPECAT_LLM_API_KEY"),
        base_url=base_url,
        max_retries=0,
        timeout=float(os.getenv("PIPECAT_LLM_TIMEOUT_SEC", "3.0")),
    )
    started = time.perf_counter()
    first_content_ms: float | None = None
    character_count = 0
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "只回复两个字：收到"}],
        "stream": True,
        "max_tokens": 8,
        "temperature": 0,
    }
    extra_body = _llm_extra_body(model, base_url)
    if extra_body:
        request["extra_body"] = extra_body
    stream = client.chat.completions.create(**request)
    try:
        for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if not content:
                continue
            if first_content_ms is None:
                first_content_ms = (time.perf_counter() - started) * 1000
            character_count += len(content)
    finally:
        stream.close()
    if first_content_ms is None or character_count <= 0:
        raise SystemExit("LLM stream completed without answer content")
    print(
        f"LLM_STREAM_OK provider={_llm_provider(model, base_url)} "
        f"model={model} first_content_ms={first_content_ms:.1f} "
        f"characters={character_count} response_text_printed=no"
    )


if __name__ == "__main__":
    main()
