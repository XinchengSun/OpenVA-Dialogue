"""Reproduce the slow-consumer log leak against a Git baseline on CPU.

This measures retained queue tasks, not GPU performance or network latency.
The extracted classes are the production implementation, without model imports.
"""

import argparse
import ast
import asyncio
import __future__
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_client(source):
    tree = ast.parse(source)
    namespace = dict(asyncio=asyncio, threading=threading, queue=queue, os=os)
    names = {"MediaPipeClient", "_BoundedLoopQueueWriter"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "server_mse.py", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace["MediaPipeClient"]


async def measure(source, messages):
    client_class = load_client(source)
    engine = SimpleNamespace(loop=asyncio.get_running_loop(), log=lambda msg: None)
    client = client_class(engine, 0)
    tasks_before = asyncio.all_tasks()
    send_log = getattr(client, "put_log", client.put_text)
    for index in range(messages):
        send_log(json.dumps({"type": "log", "message": f"synthetic-{index}"}))
    # Flush scheduling callbacks and their spawned coroutines with no consumer.
    for _ in range(5):
        await asyncio.sleep(0)
    pending = asyncio.all_tasks() - tasks_before
    result = {
        "submitted_log_messages": messages,
        "queued_messages": client.out_q.qsize(),
        "pending_queue_putters": len(client.out_q._putters),
        "pending_async_tasks": len(pending),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    if hasattr(client, "transport_snapshot"):
        result["transport"] = client.transport_snapshot()
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="73a2d4846172ad1e1ea1037b5294b922113e8de9")
    parser.add_argument("--messages", type=int, default=10000)
    args = parser.parse_args()
    if args.messages < 1:
        parser.error("--messages must be positive")
    baseline_sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", args.baseline + "^{commit}"],
        cwd=ROOT, text=True,
    ).strip()
    baseline = subprocess.check_output(
        ["git", "show", baseline_sha + ":server_mse.py"], cwd=ROOT,
    ).decode("utf-8")
    current = (ROOT / "server_mse.py").read_text(encoding="utf-8")
    result = {
        "scope": "synthetic CPU log overload, no consumer; no GPU/network claim",
        "baseline_commit": baseline_sha,
        "baseline": asyncio.run(measure(baseline, args.messages)),
        "current": asyncio.run(measure(current, args.messages)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
