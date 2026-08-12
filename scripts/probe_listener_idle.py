import argparse
import asyncio
import json
import time

import websockets


async def probe(url: str, duration_sec: float):
    started = time.monotonic()
    binary_messages = 0
    binary_bytes = 0
    text_messages = 0
    max_binary_gap_sec = 0.0
    last_binary_at = None

    async with websockets.connect(url, max_size=None) as ws:
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= duration_sec:
                break
            try:
                message = await asyncio.wait_for(
                    ws.recv(),
                    timeout=min(5.0, duration_sec - elapsed),
                )
            except asyncio.TimeoutError:
                continue

            now = time.monotonic()
            if isinstance(message, bytes):
                binary_messages += 1
                binary_bytes += len(message)
                if last_binary_at is not None:
                    max_binary_gap_sec = max(
                        max_binary_gap_sec,
                        now - last_binary_at,
                    )
                last_binary_at = now
            else:
                text_messages += 1

    result = {
        "duration_sec": round(time.monotonic() - started, 3),
        "binary_messages": binary_messages,
        "binary_bytes": binary_bytes,
        "text_messages": text_messages,
        "max_binary_gap_sec": round(max_binary_gap_sec, 3),
    }
    if binary_messages == 0 or binary_bytes == 0:
        raise RuntimeError(f"no media received: {result}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:7860/ws/media")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    args = parser.parse_args()
    asyncio.run(probe(args.url, max(1.0, args.duration_sec)))


if __name__ == "__main__":
    main()
