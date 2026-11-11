"""Opt-in stage timestamps. Never writes speech/text contents or credentials."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

_lock = threading.Lock()


def trace(stage: str, **fields) -> None:
    directory = os.getenv("OPENVA_LATENCY_TRACE_DIR", "").strip()
    if not directory:
        return
    record = {"stage": stage, "monotonic_ns": time.monotonic_ns(), "pid": os.getpid(), **fields}
    # Diagnosis must never stop a live conversation when its output is unavailable.
    try:
        with _lock:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"stages-{os.getpid()}.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        pass


def text_fields(text: str) -> dict:
    if not os.getenv("OPENVA_LATENCY_TRACE_DIR", "").strip():
        return {}
    return {"characters": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def media_fields(data: bytes) -> dict:
    """Match encoded chunks across transport boundaries without retaining media."""
    if not os.getenv("OPENVA_LATENCY_TRACE_DIR", "").strip():
        return {}
    return {"bytes": len(data), "media_sha256": hashlib.sha256(data).hexdigest()}
