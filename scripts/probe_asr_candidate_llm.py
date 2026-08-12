#!/usr/bin/env python3
"""Verify that the live LLM resolves ASR candidates without leaking metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat_dystream.custom_cascade import DEFAULT_SYSTEM_INSTRUCTION


def main() -> None:
    client = OpenAI(
        api_key=os.environ["PIPECAT_LLM_API_KEY"],
        base_url=os.environ["PIPECAT_LLM_BASE_URL"],
        max_retries=0,
        timeout=8.0,
    )
    message = """<asr_candidates same_utterance="true">
A: 腰解放一下自己
B: 简单介绍一下自己
</asr_candidates>"""
    response = client.chat.completions.create(
        model=os.environ["PIPECAT_LLM_MODEL"],
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": message},
        ],
        stream=False,
        max_tokens=128,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    answer = response.choices[0].message.content or ""
    if any(term in answer for term in ("候选", "识别结果", "asr_candidates")):
        raise SystemExit("candidate metadata leaked into the answer")
    if not any(term in answer for term in ("助手", "帮助", "帮你", "解答", "陪你")):
        raise SystemExit("model did not resolve the self-introduction intent")
    print(
        "ASR_CANDIDATE_LLM_OK "
        + json.dumps({"answer": answer}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
