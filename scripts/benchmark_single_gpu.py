#!/usr/bin/env python3
"""Measure a running local server; does not launch models or read credentials.

Example (run ON the GPU server, in the server's PID namespace):
  python scripts/benchmark_single_gpu.py --base ws://127.0.0.1:7871 \
    --gpu 1 --server-pid 1234 --duration-sec 60 --output-dir /data/bench/run01

Repeat --server-pid for separately launched speech services. The GPU is a
physical nvidia-smi index, unaffected by CUDA_VISIBLE_DEVICES. A remote URL
cannot establish ownership of locally sampled GPUs. --input-wav must be mono
16-kHz PCM16; it claims the server's microphone session. Output must be a NEW
directory. All media, samples and reports are stored there, without server logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import struct
import subprocess
import time
import urllib.request
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


HEALTH_FIELDS = {
    "status", "engine_version", "feed_thread_alive", "frame_ready", "state",
    "latest_frame_seq", "latest_frame_age_sec", "latest_visible_frame_seq",
    "latest_visible_frame_age_sec", "turn_id", "stream_generation",
    "speaker_samples", "tail_samples", "user_samples", "listener_audio_enabled",
    "listener_virtual_samples", "listener_other_source", "media_clients", "time",
}
WORKER_FIELDS = {
    "generation", "motion_alive", "render_alive", "motion_exitcode",
    "render_exitcode", "motion_pid", "render_pid", "motion_device", "render_device",
}
EVENT_FIELDS = {"type", "turn_id", "generation", "epoch", "mime", "media_time"}


def run(command: list[str], timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace", check=False)


def checked(command: list[str], timeout: float = 10) -> str:
    result = run(command, timeout)
    if result.returncode:
        # Do not copy process output: it might contain service credentials.
        raise RuntimeError(f"{Path(command[0]).name} exited {result.returncode}")
    return result.stdout


def number(value: str):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
            "max": ordered[-1]}


def intervals(times: list[float]) -> dict:
    return distribution([b - a for a, b in zip(times, times[1:])])


def process_table() -> dict[int, tuple[int, str]]:
    """PID -> (parent PID, creation identity), without reading command lines/env."""
    if os.name == "nt":
        output = checked([
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,"
            "CreationDate | ConvertTo-Json -Compress",
        ], 15)
        rows = json.loads(output)
        if isinstance(rows, dict):
            rows = [rows]
        return {int(p["ProcessId"]): (int(p["ParentProcessId"]), str(p["CreationDate"]))
                for p in rows}
    if not Path("/proc").is_dir():
        raise RuntimeError("PID ownership sampling requires Linux /proc or Windows CIM")
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm may contain spaces or parentheses; fields start after its last ')'.
            fields = path.read_text().rsplit(")", 1)[1].split()
            result[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return result


def linux_process_identity(pid: int) -> tuple[int, str] | None:
    """Read a process/thread's TGID and start time without its command or env."""
    try:
        path = Path("/proc") / str(pid)
        before = path.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        tgid = next(int(line.split()[1]) for line in path.joinpath("status").read_text().splitlines()
                    if line.startswith("Tgid:"))
        after = path.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        return (tgid, before) if before == after else None
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def service_task_snapshot(owned: set[int], known: dict[int, str]) -> dict[int, tuple[int, str]]:
    """Anchor thread identities before nvidia-smi, which can report CUDA TIDs."""
    result = {}
    if os.name == "nt":
        return result
    for owner in owned:
        if linux_process_identity(owner) != (owner, known[owner]):
            continue
        for path in (Path("/proc") / str(owner) / "task").glob("[0-9]*"):
            try:
                start = path.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
                result[int(path.name)] = (owner, start)
            except (OSError, ValueError, IndexError):
                continue
    return result


def compute_process_owner(pid: int, owned: set[int], known: dict[int, str],
                          tasks: dict[int, tuple[int, str]]) -> dict:
    if os.name == "nt":
        # Windows exposes process IDs rather than Linux light-weight thread IDs.
        return {"owner_pid": pid, "is_test_service": pid in owned,
                "service_identity_unverified": False}
    identity = linux_process_identity(pid)
    if identity is None:
        return {"owner_pid": None, "is_test_service": False,
                "service_identity_unverified": pid in tasks or pid in owned}
    owner, _ = identity
    candidate = owner in owned
    verified = (candidate and tasks.get(pid) == identity
                and linux_process_identity(owner) == (owner, known[owner]))
    return {"owner_pid": owner, "is_test_service": verified,
            "service_identity_unverified": candidate and not verified}


def gpu_snapshot(roots: list[int], known: dict[int, str], target: str) -> dict:
    table = process_table() if roots else {}
    for pid in roots:
        if pid in table and pid not in known:
            known[pid] = table[pid][1]
    owned = {pid for pid, identity in known.items()
             if pid in table and table[pid][1] == identity}
    while True:
        children = {pid for pid, (parent, _) in table.items() if parent in owned}
        additional = children - owned
        if not additional:
            break
        owned.update(additional)
        known.update({pid: table[pid][1] for pid in additional})
    tasks = service_task_snapshot(owned, known)
    # query-compute-apps omits graphics-only EGL/OpenGL contexts, which still
    # violate strict single-card placement. XML includes C, G and mixed types.
    xml = ET.fromstring(checked(["nvidia-smi", "-q", "-x"]))
    devices = []
    apps = []
    for index, gpu in enumerate(xml.findall("gpu")):
        def metric(path):
            value = gpu.findtext(path, "N/A").split()
            return number(value[0]) if value else None

        uuid = gpu.findtext("uuid", "")
        devices.append({
            "index": str(index), "uuid": uuid,
            "name": gpu.findtext("product_name", ""),
            "memory_used_mib": metric("fb_memory_usage/used"),
            "memory_total_mib": metric("fb_memory_usage/total"),
            "utilization_pct": metric("utilization/gpu_util"),
        })
        for process in gpu.findall("processes/process_info"):
            reported_pid = process.findtext("pid", "")
            if not reported_pid.isdigit():
                continue
            pid = int(reported_pid)
            memory = process.findtext("used_memory", "N/A").split()
            apps.append({"gpu_uuid": uuid, "pid": pid,
                         "type": process.findtext("type", "unknown"),
                         "process_name": Path(process.findtext("process_name", "unknown")).name,
                         "memory_used_mib": number(memory[0]) if memory else None,
                         **compute_process_owner(pid, owned, known, tasks)})
    selected = next((d for d in devices if target in (d["index"], d["uuid"])), None)
    if selected is None:
        raise RuntimeError("requested physical GPU was not returned by nvidia-smi")
    return {"target": selected, "all_gpus": devices, "gpu_processes": apps,
            "process_query": "nvidia-smi -q -x (compute and graphics)",
            "live_service_pids": sorted(owned),
            "missing_root_pids": [pid for pid in roots if pid not in owned]}


def health_snapshot(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.load(response)
    # The endpoint currently includes launch_token. Never persist the raw object.
    result = {key: payload[key] for key in HEALTH_FIELDS if key in payload}
    workers = payload.get("workers", {})
    if isinstance(workers, dict):
        result["workers"] = {key: workers[key] for key in WORKER_FIELDS if key in workers}
    session = payload.get("dialog_session", {})
    if isinstance(session, dict):
        result["dialog_session"] = {key: session[key] for key in
            ("ready", "closed", "task_done", "tts_active", "mode", "custom_cascade_ready")
            if key in session}
    return result


class Boxes:
    """Incrementally parse top-level MP4 boxes, never scanning compressed bytes."""
    def __init__(self):
        self.buffer = bytearray()
        self.counts: dict[str, int] = {}
        self.fragment_times: list[float] = []
        self.awaiting_mdat = False
        self.parsed_bytes = 0
        self.decodable_prefix_bytes = 0

    def feed(self, data: bytes, at: float):
        self.buffer.extend(data)
        while len(self.buffer) >= 8:
            size, kind = struct.unpack(">I4s", self.buffer[:8])
            header = 8
            if size == 1:
                if len(self.buffer) < 16:
                    break
                size = struct.unpack(">Q", self.buffer[8:16])[0]
                header = 16
            if size == 0:
                break  # box extends to EOF; a complete live fragment is not established
            if size < header or size > 256 * 1024 * 1024:
                raise ValueError("invalid or excessively large top-level MP4 box")
            if len(self.buffer) < size:
                break
            name = kind.decode("ascii", errors="replace")
            self.counts[name] = self.counts.get(name, 0) + 1
            self.parsed_bytes += size
            if kind == b"moof":
                self.awaiting_mdat = True
            elif kind == b"mdat" and self.awaiting_mdat:
                self.fragment_times.append(at)
                self.awaiting_mdat = False
                self.decodable_prefix_bytes = self.parsed_bytes
            elif kind == b"moov":
                self.decodable_prefix_bytes = self.parsed_bytes
            del self.buffer[:size]


def inspect_video(path: Path, ffprobe: str, ffmpeg: str | None) -> dict:
    result = run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_frames", "-show_streams", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames:"
        "frame=best_effort_timestamp_time,pkt_duration_time,duration_time",
        "-of", "json", str(path),
    ], 180)
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    frames = payload.get("frames", [])
    stamps = [float(f["best_effort_timestamp_time"]) for f in frames
              if "best_effort_timestamp_time" in f]
    deltas = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    last_duration = number(frames[-1].get("duration_time", frames[-1].get("pkt_duration_time"))) if frames else None
    if last_duration is None or last_duration <= 0:
        last_duration = statistics.median(deltas) if deltas else 0
    span = stamps[-1] - stamps[0] + last_duration if stamps else 0
    report = {
        "path": str(path), "ffprobe_returncode": result.returncode,
        "decoder_error_present": bool(result.stderr.strip()), "streams": streams,
        "decoded_frame_count": len(frames), "timestamped_frame_count": len(stamps),
        "presentation_duration_sec": span,
        "decoded_presentation_fps": len(frames) / span if span > 0 else None,
        "frame_pts_step_sec": distribution(deltas),
        "nonincreasing_frame_pts_count": sum(b <= a for a, b in zip(stamps, stamps[1:])),
        "model_generation_fps": None,
    }
    if ffmpeg:
        hashes_path = path.with_suffix(".framemd5")
        # Explicit software decode: postprocessing must not create new GPU workloads.
        decode = run([ffmpeg, "-v", "error", "-hwaccel", "none", "-i", str(path),
                      "-map", "0:v:0", "-an", "-vsync", "0", "-f", "framemd5", "-"], 180)
        hashes_path.write_text(decode.stdout, encoding="utf-8")
        hashes = [line.rsplit(",", 1)[-1].strip() for line in decode.stdout.splitlines()
                  if line and not line.startswith("#") and "," in line]
        repeats = sum(a == b for a, b in zip(hashes, hashes[1:]))
        report["decoded_image_changes"] = {
            "ffmpeg_returncode": decode.returncode,
            "decoder_error_present": bool(decode.stderr.strip()),
            "hashed_frames": len(hashes), "exact_adjacent_repeated_frames": repeats,
            "exact_adjacent_repeat_ratio": repeats / (len(hashes) - 1) if len(hashes) > 1 else None,
            "note": "Exact decoded pixel equality only. Lossy encoding may alter repeated "
                    "source images; this is not a model-generation frame count.",
        }
    return report


class AssistantTurnLifecycle:
    """Accept one ordered reply lifecycle after the input's final voiced sample."""
    def __init__(self, voice_end: float):
        self.voice_end = voice_end
        self.key: tuple[int, int] | None = None
        self.marks: dict[str, float] = {}

    def observe(self, at: float, event: dict) -> bool:
        turn_id, generation = event.get("turn_id"), event.get("generation")
        if type(turn_id) is not int or type(generation) is not int:
            return False
        key = (turn_id, generation)
        kind = event.get("type")
        if kind == "assistant_turn_started":
            # A VAD partial response can begin while the input WAV is still
            # speaking. It cannot establish this benchmark turn's lifecycle.
            if at < self.voice_end:
                return False
            if key != self.key:
                self.key = key
                self.marks.clear()
            self.marks.setdefault(kind, at)
        elif key != self.key:
            return False
        elif kind == "assistant_media_boundary":
            if at < self.marks["assistant_turn_started"]:
                return False
            self.marks.setdefault(kind, at)
        elif kind == "assistant_media_ended":
            if "assistant_media_boundary" not in self.marks or at < self.marks["assistant_media_boundary"]:
                return False
            self.marks.setdefault(kind, at)
        return "assistant_media_ended" in self.marks


async def capture(args, output: Path) -> dict:
    from websockets.asyncio.client import connect

    parsed = urlsplit(args.base)
    http_base = urlunsplit(("https" if parsed.scheme == "wss" else "http",
                          parsed.netloc, parsed.path.rstrip("/"), "", ""))
    origin = time.perf_counter()
    elapsed = lambda: time.perf_counter() - origin
    stop = asyncio.Event()
    samples, health, errors, epochs, events = [], [], [], [], []
    known: dict[int, str] = {}
    turn_events: asyncio.Queue = asyncio.Queue()
    dialogue = {"requested": bool(args.input_wav), "turns": []}

    async def sampler():
        with (output / "samples.jsonl").open("x", encoding="utf-8") as file:
            while not stop.is_set():
                batch = await asyncio.gather(
                    asyncio.to_thread(gpu_snapshot, args.server_pid, known, args.gpu),
                    asyncio.to_thread(health_snapshot, http_base + "/health"),
                    return_exceptions=True,
                )
                row = {"at_sec": elapsed()}
                for name, value in zip(("gpu", "health"), batch):
                    if isinstance(value, Exception):
                        row[name + "_error"] = type(value).__name__
                    else:
                        row[name] = value
                samples.append(row)
                if "health" in row:
                    health.append({"at_sec": row["at_sec"], **row["health"]})
                file.write(json.dumps(row) + "\n")
                file.flush()
                try:
                    await asyncio.wait_for(stop.wait(), args.sample_sec)
                except asyncio.TimeoutError:
                    pass

    async def talk():
        if not args.input_wav:
            return
        await asyncio.sleep(args.input_delay_sec)
        with wave.open(str(args.input_wav), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("--input-wav requires mono 16-kHz uncompressed PCM16")
            raw = wav.readframes(wav.getnframes())
        if not raw:
            raise ValueError("input WAV is empty")
        async with connect(args.base.rstrip("/") + "/ws/mic", max_size=None) as mic:
            await mic.send(json.dumps({"type": "config", "instruction": args.instruction}))
            ack = json.loads(await asyncio.wait_for(mic.recv(), 10))
            if ack.get("type") != "ok":
                raise RuntimeError("microphone session configuration was rejected")
            for turn in range(1, args.turns + 1):
                while not turn_events.empty():
                    turn_events.get_nowait()
                voice_end = None
                for offset in range(0, len(raw), 1280):
                    chunk = raw[offset:offset + 1280]
                    chunk_start = elapsed()
                    voiced = [i for i, (v,) in enumerate(struct.iter_unpack("<h", chunk))
                              if abs(v) >= args.voice_threshold]
                    if voiced:
                        voice_end = chunk_start + (voiced[-1] + 1) / 16000
                    await mic.send(chunk.ljust(1280, b"\0"))
                    await asyncio.sleep(.04)
                if voice_end is None:
                    raise ValueError("input WAV contains no speech above voice threshold")
                for _ in range(30):
                    await mic.send(bytes(1280))
                    await asyncio.sleep(.04)
                lifecycle = AssistantTurnLifecycle(voice_end)
                deadline = time.perf_counter() + args.turn_timeout_sec
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise asyncio.TimeoutError("reply did not complete a matching turn lifecycle")
                    at, event = await asyncio.wait_for(turn_events.get(), remaining)
                    if lifecycle.observe(at, event):
                        break
                marks = lifecycle.marks
                dialogue["turns"].append({
                    "turn": turn, "input_duration_sec": len(raw) / 32000,
                    "engine_turn_id": lifecycle.key[0], "generation": lifecycle.key[1],
                    "lifecycle_complete": True,
                    "voice_end_sec": voice_end, "events_at_sec": marks,
                    "voice_end_to_event_ms": {key: (at - voice_end) * 1000 for key, at in marks.items()},
                })
                await asyncio.sleep(.5)

    sample_task = asyncio.create_task(sampler())
    talk_task = None
    connected_at = None
    captured_until = None
    try:
        async with connect(args.base.rstrip("/") + "/ws/media", max_size=None,
                           open_timeout=10, close_timeout=3) as ws:
            connected_at = elapsed()
            deadline = time.perf_counter() + args.duration_sec
            epoch = None
            file = None
            boxes = None

            def close_epoch():
                nonlocal file
                if file is not None:
                    # A timed capture can stop halfway through moof/mdat. Preserve
                    # complete fragments only, so an intentional cutoff does not
                    # look like an encoder failure in ffprobe.
                    file.truncate(boxes.decodable_prefix_bytes)
                    file.close()
                    file = None
                    epoch["box_counts"] = boxes.counts
                    epoch["complete_fragment_arrival_sec"] = boxes.fragment_times
                    epoch["trailing_incomplete_box_bytes"] = len(boxes.buffer)
                    epoch["saved_bytes"] = boxes.decodable_prefix_bytes
                    epoch["discarded_incomplete_fragment_bytes"] = epoch["bytes"] - boxes.decodable_prefix_bytes

            talk_task = asyncio.create_task(talk())
            try:
                while time.perf_counter() < deadline:
                    try:
                        message = await asyncio.wait_for(ws.recv(), deadline - time.perf_counter())
                    except asyncio.TimeoutError:
                        break
                    at = elapsed()
                    if isinstance(message, bytes):
                        if file is None:
                            path = output / f"media-{len(epochs):03d}.mp4"
                            file = path.open("xb")
                            boxes = Boxes()
                            epoch = {"path": str(path), "binary_message_arrival_sec": [], "bytes": 0}
                            epochs.append(epoch)
                        file.write(message)
                        epoch["bytes"] += len(message)
                        epoch["binary_message_arrival_sec"].append(at)
                        boxes.feed(message, at)
                    else:
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        event = {key: payload[key] for key in EVENT_FIELDS if key in payload}
                        events.append({"at_sec": at, **event})
                        if event.get("type") == "stream_reset":
                            close_epoch()
                        if event.get("type") in {"assistant_turn_started", "assistant_media_boundary", "assistant_media_ended"}:
                            turn_events.put_nowait((at, event))
            finally:
                close_epoch()
                captured_until = elapsed()
    except Exception as exc:
        errors.append({"component": "media", "error": type(exc).__name__})
    finally:
        stop.set()
        await sample_task
        if talk_task:
            if not talk_task.done():
                talk_task.cancel()
                if args.input_wav:
                    errors.append({"component": "dialogue", "error": "capture ended before dialogue completed"})
            outcome = (await asyncio.gather(talk_task, return_exceptions=True))[0]
            if isinstance(outcome, Exception):
                errors.append({"component": "dialogue", "error": type(outcome).__name__})
    return {"samples": samples, "health": health, "epochs": epochs, "events": events,
            "dialogue": dialogue, "errors": errors, "connected_at_sec": connected_at,
            "captured_until_sec": captured_until}


def summarize(args, captured: dict, videos: list[dict]) -> dict:
    samples = captured["samples"]
    gpu_samples = [sample["gpu"] for sample in samples if "gpu" in sample]
    # Accept legacy compute-only reports for inspection, but new captures use
    # the accurately named gpu_processes field and include graphics contexts.
    processes = lambda sample: sample.get("gpu_processes", sample.get("compute_processes", []))
    service_apps = [app for sample in gpu_samples for app in processes(sample)
                    if app["is_test_service"]]
    target_uuid = gpu_samples[0]["target"]["uuid"] if gpu_samples else None
    off_target = [app for app in service_apps if app["gpu_uuid"] != target_uuid]
    unresolved_service_identity = any(
        app.get("service_identity_unverified", False)
        for sample in gpu_samples for app in processes(sample)
    )
    health = captured["health"]
    duration = ((captured["captured_until_sec"] - captured["connected_at_sec"])
                if captured["connected_at_sec"] is not None and captured["captured_until_sec"] is not None else 0)
    decoded = sum(video.get("decoded_frame_count", 0) for video in videos)
    errors = captured["errors"]
    checks = {
        "full_capture_duration": duration >= args.duration_sec * .99,
        "media_received_and_decoded": decoded > 0,
        "software_decode_clean": bool(videos) and all(
            v.get("ffprobe_returncode") == 0 and not v.get("decoder_error_present")
            and v.get("decoded_frame_count", 0) > 0 for v in videos),
        "gpu_sampling_complete": bool(samples) and all("gpu" in s for s in samples),
        "health_sampling_complete": bool(samples) and all("health" in s for s in samples),
        "health_ok": bool(health) and all(s.get("status") in {"ok", "ready"} for s in health),
        "service_single_gpu": (False if off_target else None
                               if unresolved_service_identity or not service_apps else True),
        "service_roots_live": (all(not s["missing_root_pids"] for s in gpu_samples)
                               if args.server_pid and gpu_samples else None),
        "no_capture_errors": not errors,
    }
    if args.input_wav:
        turns = captured["dialogue"]["turns"]
        reply_keys = {(turn.get("engine_turn_id"), turn.get("generation")) for turn in turns}
        checks["dialogue_completed"] = (
            len(turns) == args.turns and len(reply_keys) == args.turns
            and all(turn.get("lifecycle_complete") is True for turn in turns)
        )
    memory_values = [s["target"]["memory_used_mib"] for s in gpu_samples
                     if s["target"]["memory_used_mib"] is not None]
    if args.max_memory_mib is not None:
        checks["total_target_memory_within_budget"] = (max(memory_values) <= args.max_memory_mib if memory_values else None)
    if args.min_delivered_fps is not None:
        checks["minimum_delivered_fps"] = duration > 0 and decoded / duration >= args.min_delivered_fps
    epochs = captured["epochs"]
    for epoch in epochs:
        epoch["binary_message_arrival_interval_sec"] = intervals(epoch["binary_message_arrival_sec"])
        epoch["complete_fragment_arrival_interval_sec"] = intervals(epoch["complete_fragment_arrival_sec"])
    seq_delta = None
    if len(health) >= 2 and all("latest_frame_seq" in h for h in (health[0], health[-1])):
        seq_delta = health[-1]["latest_frame_seq"] - health[0]["latest_frame_seq"]
    return {
        "status": "fail" if False in checks.values() else "inconclusive" if None in checks.values() else "pass",
        "checks": checks, "base": args.base, "gpu": args.gpu, "server_root_pids": args.server_pid,
        "requested_duration_sec": args.duration_sec, "capture_duration_sec": duration,
        "total_target_gpu_memory_mib": distribution(memory_values),
        "target_gpu_utilization_pct": distribution([s["target"]["utilization_pct"] for s in gpu_samples
                                                    if s["target"]["utilization_pct"] is not None]),
        "service_gpu_uuids_observed": sorted({p["gpu_uuid"] for p in service_apps}),
        "off_target_service_process_observations": off_target,
        "decoded_frames_delivered_per_wall_second": decoded / duration if duration else None,
        "health_latest_frame_seq_delta": seq_delta, "model_generation_fps": None,
        "videos": videos, "capture": captured,
        "measurement_notes": [
            "Binary WebSocket message rate and completed fMP4 fragment rate are NOT video FPS.",
            "Decoded presentation FPS uses decoded frame timestamps. Delivered FPS uses capture wall time; both may include repeated/held frames.",
            "No instrumented model-inference frame counter is available; model_generation_fps is intentionally null. Health frame_seq is only a queue-consumption counter.",
            "nvidia-smi XML reports total target GPU memory including other users and individually lists compute (C), graphics (G), and mixed processes where supported.",
            "Only explicit server PIDs and their observed descendants count as test services. Supply every independent service root. Polling can miss very short GPU allocations/processes.",
            "On Linux nvidia-smi may report CUDA thread IDs: owner_pid is resolved through /proc/TID/status Tgid. Thread and owning-process start identities are checked across sampling; unresolved service identities make single-GPU status inconclusive.",
            "This benchmark must run on the server in the same PID namespace. It does not measure browser playback or display latency.",
            "Voice-end latency uses the last PCM sample above the amplitude threshold; media boundary is not browser playback onset.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="ws://127.0.0.1:7871")
    parser.add_argument("--gpu", required=True, help="physical nvidia-smi index or full GPU UUID")
    parser.add_argument("--server-pid", type=int, action="append", default=[], help="repeat for independently launched TTS/LLM services")
    parser.add_argument("--duration-sec", type=float, default=60)
    parser.add_argument("--sample-sec", type=float, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-wav", type=Path)
    parser.add_argument("--instruction", default="Answer briefly in Chinese.",
                        help="dialogue system instruction sent with microphone configuration")
    parser.add_argument("--input-delay-sec", type=float, default=2)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--turn-timeout-sec", type=float, default=40)
    parser.add_argument("--voice-threshold", type=int, default=500)
    parser.add_argument("--max-memory-mib", type=float)
    parser.add_argument("--min-delivered-fps", type=float)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="optional exact decoded-frame repeat audit")
    args = parser.parse_args()
    parsed = urlsplit(args.base)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
        parser.error("--base must be ws(s)://host[:port] without credentials, query or fragment")
    if min(args.duration_sec, args.sample_sec, args.turn_timeout_sec, args.turns) <= 0 or args.input_delay_sec < 0:
        parser.error("durations and turns must be positive; input delay must be nonnegative")
    if any(pid <= 0 for pid in args.server_pid):
        parser.error("server PIDs must be positive")
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be absolute")
    output = args.output_dir.resolve()
    if os.name == "nt" and output.drive.lower() == "c:":
        parser.error("choose a non-C output directory")
    if output.exists():
        parser.error("--output-dir already exists; use a new directory to preserve previous results")
    ffprobe = shutil.which(args.ffprobe)
    if not ffprobe:
        parser.error("ffprobe is required for real decoded-frame measurement")
    if not shutil.which("nvidia-smi"):
        parser.error("nvidia-smi is required; run this benchmark on the GPU server")
    if args.input_wav and not args.input_wav.is_file():
        parser.error("--input-wav does not exist")
    output.mkdir(parents=True, exist_ok=False)
    captured = asyncio.run(capture(args, output))
    videos = []
    for epoch in captured["epochs"]:
        try:
            videos.append(inspect_video(Path(epoch["path"]), ffprobe, shutil.which(args.ffmpeg)))
        except Exception as exc:
            captured["errors"].append({"component": "video_inspection", "error": type(exc).__name__})
    report = summarize(args, captured, videos)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
