#!/usr/bin/env python3
"""Probe the production Qwen streaming path with optional real-time search."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipecat.processors.aggregators.llm_context import LLMContext

from pipecat_dystream.custom_cascade import _llm_extra_body
from pipecat_dystream.resilient_llm import RecoveringOpenAILLMService


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


async def _probe(args: argparse.Namespace) -> None:
    model = _required("PIPECAT_LLM_MODEL")
    base_url = _required("PIPECAT_LLM_BASE_URL")
    settings_extra = {}
    extra_body = _llm_extra_body(model, base_url)
    if extra_body:
        settings_extra["extra_body"] = extra_body

    service = RecoveringOpenAILLMService(
        api_key=_required("PIPECAT_LLM_API_KEY"),
        base_url=base_url,
        settings=RecoveringOpenAILLMService.Settings(
            model=model,
            system_instruction=(
                "你是中文实时信息助手。直接回答问题，不要输出Markdown链接，"
                "若查询不到可靠的最新数据就明确说明。"
            ),
            temperature=0,
            max_tokens=args.max_tokens,
            extra=settings_extra,
        ),
        timeout_fallback_text="网络查询超时。",
        hedge_model="",
        realtime_search_mode=args.mode,
        realtime_search_strategy=args.strategy,
    )
    service._client = service._client.with_options(
        max_retries=0,
        timeout=args.timeout,
    )

    for round_index in range(args.rounds):
        context = LLMContext([{"role": "user", "content": args.query}])
        started_at = time.perf_counter()
        first_content_ms = None
        answer_parts: list[str] = []
        stream = await service.get_chat_completions(context)
        async for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if not content:
                continue
            if first_content_ms is None:
                first_content_ms = (time.perf_counter() - started_at) * 1000.0
            answer_parts.append(content)

        answer = "".join(answer_parts).strip()
        if not answer or first_content_ms is None:
            raise SystemExit("provider stream completed without answer text")
        payload = {
            "round": round_index + 1,
            "mode": args.mode,
            "strategy": args.strategy,
            "first_content_ms": round(first_content_ms, 1),
            "total_ms": round((time.perf_counter() - started_at) * 1000.0, 1),
            "answer_chars": len(answer),
            "route": service.realtime_search_snapshot(),
        }
        if args.show_text:
            payload["answer"] = answer
        print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--mode",
        choices=("off", "smart", "auto", "always"),
        default="smart",
    )
    parser.add_argument(
        "--strategy",
        choices=("turbo", "max", "agent"),
        default="turbo",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        raise SystemExit("rounds, max-tokens, and timeout must be positive")
    asyncio.run(_probe(args))


if __name__ == "__main__":
    main()
