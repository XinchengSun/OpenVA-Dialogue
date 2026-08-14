#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import audioop
import fcntl
import json
import math
import os
import shutil
import shlex
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, NamedTuple


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def update_status(
    path: Path, *, clear_errors: bool = False, **updates: Any
) -> dict[str, Any]:
    status = read_json(path) if path.is_file() else {}
    if clear_errors:
        status.pop("error_code", None)
        status.pop("rollback_succeeded", None)
    status.update(updates)
    status["updated_at"] = time.time()
    write_json(path, status)
    return status


def patch_env_file(path: Path, updates: dict[str, str]) -> None:
    original = path.read_text(encoding="utf-8")
    applied: set[str] = set()
    lines: list[str] = []
    for line in original.splitlines():
        key = line.split("=", 1)[0] if "=" in line else ""
        if key in updates and key and key.replace("_", "A").isalnum():
            if key not in applied:
                value = shlex.quote(updates[key]) if updates[key] else ""
                lines.append(f"{key}={value}")
                applied.add(key)
        else:
            lines.append(line)
    remaining = {key: value for key, value in updates.items() if key not in applied}
    if remaining:
        if lines and lines[-1]:
            lines.append("")
        lines.append("# Managed by the custom avatar upload flow.")
        lines.extend(
            f"{key}={shlex.quote(value) if value else ''}"
            for key, value in remaining.items()
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(temporary, path.stat().st_mode & 0o777)
    os.replace(temporary, path)


def build_restart_environment(
    repo_root: Path, main_env: Path, port: int, job_id: str
) -> tuple[dict[str, str], str]:
    environment = os.environ.copy()
    environment["ENV_FILE"] = str(main_env)
    environment["PORT"] = str(port)
    inherited_pythonpath = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not inherited_pythonpath
        else f"{repo_root}{os.pathsep}{inherited_pythonpath}"
    )
    environment.pop("PYTHONSAFEPATH", None)
    launch_token = f"custom-{job_id[:16]}"
    environment["DEMO_LAUNCH_TOKEN"] = launch_token
    return environment, launch_token


def run_demo_restart(repo_root: Path, main_env: Path, port: int, job_id: str) -> str:
    environment, launch_token = build_restart_environment(
        repo_root, main_env, port, job_id
    )
    print(f"[CUSTOMIZE] restarting managed demo for job={job_id}", flush=True)
    subprocess.run(
        ["bash", str(repo_root / "scripts" / "run_demo.sh"), "restart-mse"],
        cwd=str(repo_root),
        env=environment,
        check=True,
        timeout=600,
    )
    return launch_token


def run_demo_full_restart(
    repo_root: Path, main_env: Path, port: int, job_id: str
) -> str:
    """Compatibility lifecycle for the legacy managed VoxCPM2 backend."""
    environment, launch_token = build_restart_environment(
        repo_root, main_env, port, job_id
    )
    print(f"[CUSTOMIZE] restarting managed VoxCPM2 stack for job={job_id}", flush=True)
    subprocess.run(
        ["bash", str(repo_root / "scripts" / "run_demo.sh"), "restart"],
        cwd=str(repo_root),
        env=environment,
        check=True,
        timeout=600,
    )
    return launch_token


def restart_fish_bridge(
    repo_root: Path, bridge_env: Path, bridge_instance: str = "realtime"
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONSAFEPATH", None)
    inherited_pythonpath = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not inherited_pythonpath
        else f"{repo_root}{os.pathsep}{inherited_pythonpath}"
    )
    environment["OPENAI_SPEECH_INSTANCE"] = bridge_instance
    print("[CUSTOMIZE] reloading Fish Speech reference voice", flush=True)
    subprocess.run(
        [
            "bash",
            str(repo_root / "voice_service" / "run_openai_speech_bridge.sh"),
            "restart",
            str(bridge_env),
        ],
        cwd=str(repo_root),
        env=environment,
        check=True,
        timeout=180,
    )


def transcribe_reference(
    repo_root: Path,
    main_env: Path,
    voice_path: Path,
    reference_language: str = "auto",
) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["MODELSCOPE_DISABLE_AUTO_UPDATE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [
            read_env_value(main_env, "PIPECAT_PYTHON")
            or os.environ.get("PIPECAT_PYTHON", sys.executable),
            str(repo_root / "scripts" / "transcribe_reference.py"),
            "--audio",
            str(voice_path),
            "--model",
            "iic/SenseVoiceSmall",
            "--device",
            "cpu",
            "--hub",
            "ms",
            "--language",
            reference_language,
        ],
        cwd=str(repo_root),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    transcript = str(payload.get("text", "")).strip()
    if not transcript:
        raise RuntimeError("automatic reference transcription returned no text")
    return transcript


def should_auto_transcribe(manifest: dict[str, Any]) -> bool:
    source = str(manifest.get("transcript_source") or "").strip()
    return not str(manifest.get("transcript_path") or "").strip() or source in {
        "none",
        "legacy_paraformer_auto",
    }


def read_env_value(path: Path, key: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            raw = line.partition("=")[2].strip()
            if not raw.startswith(("'", '"')):
                return raw
            values = shlex.split(raw, comments=False)
            return values[0] if values else ""
    return ""


def stage_fish_reference(
    bridge_env: Path,
    voice_path: Path,
    job_id: str,
) -> Path:
    current_reference = read_env_value(bridge_env, "OPENAI_SPEECH_REFERENCE_AUDIO")
    if not current_reference:
        raise RuntimeError("Fish bridge environment has no reference-audio path")
    reference_dir = Path(current_reference).expanduser().resolve().parent
    if not reference_dir.is_dir():
        raise RuntimeError("Fish reference allowlist directory is missing")
    managed_reference = reference_dir / f"custom-{job_id}.wav"
    temporary = managed_reference.with_name(f".{managed_reference.name}.{os.getpid()}.tmp")
    shutil.copyfile(voice_path, temporary)
    os.chmod(temporary, 0o600)
    os.replace(temporary, managed_reference)
    return managed_reference


def apply_fish_reference_env(
    bridge_env: Path,
    managed_reference: Path,
    transcript: str,
    reference_language: str = "auto",
    target_language: str = "zh-CN",
) -> None:
    patch_env_file(
        bridge_env,
        {
            "OPENAI_SPEECH_REFERENCE_AUDIO": str(managed_reference),
            "OPENAI_SPEECH_REFERENCE_TEXT": transcript,
            "OPENAI_SPEECH_BACKEND": "fish_s2_pro",
            "OPENAI_SPEECH_REFERENCE_LANGUAGE": reference_language,
            "OPENAI_SPEECH_TARGET_LANGUAGE": target_language,
        },
    )


def install_fish_reference(
    bridge_env: Path,
    voice_path: Path,
    transcript: str,
    job_id: str,
) -> Path:
    managed_reference = stage_fish_reference(bridge_env, voice_path, job_id)
    apply_fish_reference_env(bridge_env, managed_reference, transcript)
    return managed_reference


class PCMProbeMetrics(NamedTuple):
    rms_dbfs: float
    peak_dbfs: float
    clipping_samples: int
    samples: int


def measure_pcm16(pcm: bytes) -> PCMProbeMetrics:
    if not pcm or len(pcm) % 2:
        raise RuntimeError("Fish preflight returned invalid PCM16 audio")
    rms = audioop.rms(pcm, 2)
    peak = audioop.max(pcm, 2)
    clipping_samples = sum(
        1
        for offset in range(0, len(pcm), 2)
        if abs(int.from_bytes(pcm[offset : offset + 2], "little", signed=True)) >= 32767
    )
    return PCMProbeMetrics(
        rms_dbfs=20.0 * math.log10(max(rms, 1) / 32768.0),
        peak_dbfs=20.0 * math.log10(max(peak, 1) / 32768.0),
        clipping_samples=clipping_samples,
        samples=len(pcm) // 2,
    )


def validate_probe_pcm(
    pcm: bytes,
    sample_rate: int,
) -> PCMProbeMetrics:
    if sample_rate <= 0:
        raise RuntimeError("Fish preflight sample rate is invalid")
    if len(pcm) > 20 * sample_rate * 2:
        raise RuntimeError("Fish preflight audio exceeds 20 seconds")
    duration = len(pcm) / float(sample_rate * 2)
    if not 0.3 <= duration <= 20.0:
        raise RuntimeError(
            f"Fish preflight duration is unsafe: {duration:.3f} seconds"
        )
    return measure_pcm16(pcm)


def validate_candidate_probe(
    candidate: PCMProbeMetrics,
    current: PCMProbeMetrics,
) -> None:
    validate_absolute_probe(candidate)
    rms_delta = candidate.rms_dbfs - current.rms_dbfs
    if not -12.0 <= rms_delta <= 4.0:
        raise RuntimeError(
            "Fish candidate relative RMS is outside the -12 to +4 dB safety range"
        )


def validate_absolute_probe(candidate: PCMProbeMetrics) -> None:
    if not -42.0 <= candidate.rms_dbfs <= -16.0:
        raise RuntimeError(
            f"Fish candidate RMS is unsafe: {candidate.rms_dbfs:.2f} dBFS"
        )
    if candidate.peak_dbfs > -1.0:
        raise RuntimeError(
            f"Fish candidate peak is unsafe: {candidate.peak_dbfs:.2f} dBFS"
        )
    if candidate.clipping_samples / max(candidate.samples, 1) >= 0.0001:
        raise RuntimeError("Fish candidate output contains clipped samples")


def probe_fish_reference(
    bridge_env: Path,
    reference_audio: Path,
    reference_text: str,
) -> PCMProbeMetrics:
    base_url = read_env_value(bridge_env, "OPENAI_SPEECH_BASE_URL").rstrip("/")
    endpoint = read_env_value(bridge_env, "OPENAI_SPEECH_ENDPOINT") or "/v1/audio/speech"
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    provider = (read_env_value(bridge_env, "OPENAI_SPEECH_PROVIDER") or "sglang").lower()
    payload: dict[str, Any] = {
        "model": read_env_value(bridge_env, "OPENAI_SPEECH_MODEL"),
        "voice": read_env_value(bridge_env, "OPENAI_SPEECH_VOICE") or "default",
        "input": "你好，这是音色启用前的安全检测。",
        "response_format": "pcm",
        "stream": True,
    }
    extra_raw = read_env_value(bridge_env, "OPENAI_SPEECH_EXTRA_BODY_JSON")
    if extra_raw:
        extra = json.loads(extra_raw)
        if not isinstance(extra, dict):
            raise RuntimeError("Fish extra request body must be a JSON object")
        payload = {**extra, **payload}
    if provider == "vllm":
        payload.update(
            {
                "ref_audio": reference_audio.resolve().as_uri(),
                "ref_text": reference_text,
                "stream_format": "audio",
            }
        )
    else:
        payload["references"] = [
            {"audio_path": str(reference_audio.resolve()), "text": reference_text}
        ]
    request = urllib.request.Request(
        f"{base_url}{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "audio/pcm"},
        method="POST",
    )
    api_key = read_env_value(bridge_env, "OPENAI_SPEECH_API_KEY")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    timeout = float(
        read_env_value(bridge_env, "OPENAI_SPEECH_REQUEST_TIMEOUT_SEC") or "120"
    )
    sample_rate = int(read_env_value(bridge_env, "OPENAI_SPEECH_SAMPLE_RATE") or "44100")
    max_bytes = 20 * sample_rate * 2
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type().lower()
        if content_type not in {
            "audio/pcm",
            "application/octet-stream",
            "binary/octet-stream",
        }:
            raise RuntimeError(f"Fish preflight returned {content_type!r}, not PCM")
        header_rate = response.headers.get("x-sample-rate")
        if header_rate is not None and int(header_rate) != sample_rate:
            raise RuntimeError("Fish preflight response sample rate changed")
        pcm = response.read(max_bytes + 1)
    if len(pcm) > max_bytes:
        raise RuntimeError("Fish preflight audio exceeds 20 seconds")
    return validate_probe_pcm(pcm, sample_rate)


def preflight_fish_reference(
    bridge_env: Path,
    candidate_audio: Path,
    candidate_text: str,
) -> tuple[PCMProbeMetrics, PCMProbeMetrics]:
    current_audio_raw = read_env_value(bridge_env, "OPENAI_SPEECH_REFERENCE_AUDIO")
    current_text = read_env_value(bridge_env, "OPENAI_SPEECH_REFERENCE_TEXT")
    if not current_audio_raw or not current_text:
        raise RuntimeError("current Fish reference is unavailable for preflight")
    current = probe_fish_reference(bridge_env, Path(current_audio_raw), current_text)
    candidate = probe_fish_reference(bridge_env, candidate_audio, candidate_text)
    validate_candidate_probe(candidate, current)
    return current, candidate


def probe_restarted_fish_bridge(bridge_env: Path) -> PCMProbeMetrics:
    host = read_env_value(bridge_env, "OPENAI_SPEECH_BRIDGE_HOST") or "127.0.0.1"
    port = int(read_env_value(bridge_env, "OPENAI_SPEECH_BRIDGE_PORT") or "8773")
    configured_rate = int(
        read_env_value(bridge_env, "OPENAI_SPEECH_SAMPLE_RATE") or "44100"
    )
    max_bytes = 20 * configured_rate * 2

    async def synthesize() -> tuple[bytes, int]:
        from websockets.asyncio.client import connect

        request_id = uuid.uuid4().hex
        pcm = bytearray()
        sample_rate = 0
        async with asyncio.timeout(120.0):
            async with connect(
                f"ws://{host}:{port}", max_size=None, ping_interval=20, ping_timeout=20
            ) as websocket:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "synthesize",
                            "request_id": request_id,
                            "text": "你好，这是音色启用后的安全检测。",
                        },
                        ensure_ascii=False,
                    )
                )
                async for message in websocket:
                    if isinstance(message, bytes):
                        pcm.extend(message)
                        if len(pcm) > max_bytes:
                            raise RuntimeError("Fish bridge probe audio exceeds 20 seconds")
                        continue
                    event = json.loads(message)
                    if str(event.get("request_id", "")) not in {"", request_id}:
                        continue
                    if event.get("type") == "start":
                        sample_rate = int(event.get("sample_rate", 0))
                        if (
                            sample_rate != configured_rate
                            or int(event.get("channels", 0)) != 1
                            or int(event.get("sample_width", 0)) != 2
                            or event.get("audio_format") != "pcm_s16le"
                        ):
                            raise RuntimeError("Fish bridge returned invalid PCM metadata")
                    elif event.get("type") == "error":
                        raise RuntimeError(str(event.get("error") or "Fish bridge error"))
                    elif event.get("type") == "done":
                        if event.get("status") != "completed":
                            raise RuntimeError("Fish bridge synthesis did not complete")
                        break
        return bytes(pcm), sample_rate

    pcm, sample_rate = asyncio.run(synthesize())
    metrics = validate_probe_pcm(pcm, sample_rate)
    validate_absolute_probe(metrics)
    return metrics


def verify_health(port: int, expected_launch_token: str) -> None:
    deadline = time.monotonic() + 15.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=3,
            ) as response:
                health = json.loads(response.read().decode("utf-8"))
            dialog = health.get("dialog_session") or {}
            if (
                health.get("status") == "ok"
                and health.get("frame_ready") is True
                and health.get("launch_ready") is True
                and health.get("launch_token") == expected_launch_token
                and dialog.get("custom_cascade_ready") is True
            ):
                return
            raise RuntimeError("health endpoint is not fully ready")
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"customized demo did not become healthy: {last_error!r}")


def process_environment(pid_file: Path) -> dict[str, str]:
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8", errors="replace"
        )
    return values


def verify_loaded_assets(
    repo_root: Path,
    image_path: Path,
    bridge_reference_path: Path,
    bridge_instance: str = "realtime",
) -> None:
    mse_env = process_environment(repo_root / "logs" / "pipecat_mse.pid")
    suffix = "" if bridge_instance == "default" else f".{bridge_instance}"
    bridge_env = process_environment(
        repo_root / "logs" / f"openai_speech_bridge{suffix}.pid"
    )
    if Path(mse_env.get("DYSTREAM_REF_IMAGE", "")).resolve() != image_path:
        raise RuntimeError("managed MSE process did not load the selected reference image")
    if (
        Path(bridge_env.get("OPENAI_SPEECH_REFERENCE_AUDIO", "")).resolve()
        != bridge_reference_path
    ):
        raise RuntimeError("managed Fish bridge did not load the selected reference audio")


def verify_loaded_voxcpm2_assets(
    repo_root: Path, image_path: Path, voice_path: Path
) -> None:
    mse_env = process_environment(repo_root / "logs" / "pipecat_mse.pid")
    bridge_env = process_environment(repo_root / "logs" / "voxcpm2_bridge.pid")
    if Path(mse_env.get("DYSTREAM_REF_IMAGE", "")).resolve() != image_path:
        raise RuntimeError("managed MSE process did not load the selected reference image")
    if Path(bridge_env.get("VOXCPM2_PROMPT_WAV", "")).resolve() != voice_path:
        raise RuntimeError("managed VoxCPM2 bridge did not load the selected reference audio")


def activate_voxcpm2(
    *,
    args: argparse.Namespace,
    job_dir: Path,
    runtime_root: Path,
    repo_root: Path,
    main_env: Path,
    bridge_env: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    status_path: Path,
    job_id: str,
    image_path: Path,
    voice_path: Path,
    transcript_path: Path | None,
    reference_language: str,
    target_language: str,
) -> int:
    """Preserve the established VoxCPM2 customization/lifecycle contract."""
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "activation.lock"
    rollback_dir = job_dir / "rollback"
    rollback_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(rollback_dir, 0o700)

    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            update_status(
                status_path,
                state="failed",
                message="另一个定制任务正在激活，请稍后重试",
            )
            return 2

        main_backup = rollback_dir / "custom_cascade.env"
        bridge_backup = rollback_dir / "voxcpm2.env"
        active_path = runtime_root / "active.json"
        active_backup = rollback_dir / "active.json"
        active_existed = active_path.is_file()
        shutil.copy2(main_env, main_backup)
        shutil.copy2(bridge_env, bridge_backup)
        if active_existed:
            shutil.copy2(active_path, active_backup)
        os.chmod(main_backup, 0o600)
        os.chmod(bridge_backup, 0o600)
        update_status(
            status_path,
            clear_errors=True,
            state="activating",
            message="正在写入头像与克隆音色配置",
        )
        try:
            patch_env_file(main_env, {
                "DYSTREAM_REF_IMAGE": str(image_path),
                "PIPECAT_TTS_BACKEND": "voxcpm2",
                "PIPECAT_TTS_REFERENCE_LANGUAGE": reference_language,
                "PIPECAT_TTS_TARGET_LANGUAGE": target_language,
            })
            patch_env_file(bridge_env, {
                "VOXCPM2_PROMPT_WAV": str(voice_path),
                "VOXCPM2_PROMPT_TEXT": "",
                "VOXCPM2_PROMPT_TEXT_FILE": str(transcript_path) if transcript_path else "",
            })
            update_status(
                status_path,
                state="restarting",
                message="模型正在重启并缓存新头像、新音色，请等待",
            )
            time.sleep(1.5)
            launch_token = run_demo_full_restart(
                repo_root, main_env, args.port, job_id
            )
            verify_health(args.port, launch_token)
            verify_loaded_voxcpm2_assets(repo_root, image_path, voice_path)
            active = {
                "job_id": job_id,
                "activated_at": time.time(),
                "image_url": f"/api/customization/{job_id}/image",
                "voice_source_type": (manifest.get("voice") or {}).get("source_type"),
                "tts_backend": "voxcpm2",
                "reference_language": reference_language,
                "target_language": target_language,
            }
            write_json(active_path, active)
            update_status(
                status_path,
                clear_errors=True,
                state="ready",
                message="定制数字人已就绪，可以进入实时对话",
                activated_at=active["activated_at"],
                rollback_available=True,
            )
            print(f"[CUSTOMIZE] activation ready job={job_id}", flush=True)
            return 0
        except Exception as activation_error:
            print(f"[CUSTOMIZE] activation failed: {activation_error!r}", flush=True)
            update_status(
                status_path,
                state="rolling_back",
                message="激活失败，正在自动恢复上一个可运行版本",
                error_code="activation_failed",
            )
            rollback_error: Exception | None = None
            try:
                shutil.copy2(main_backup, main_env)
                shutil.copy2(bridge_backup, bridge_env)
                if active_existed:
                    shutil.copy2(active_backup, active_path)
                elif active_path.exists():
                    active_path.unlink()
                rollback_token = run_demo_full_restart(
                    repo_root, main_env, args.port, f"rollback-{job_id}"[:32]
                )
                verify_health(args.port, rollback_token)
                previous_image = Path(
                    read_env_value(main_env, "DYSTREAM_REF_IMAGE")
                ).resolve()
                previous_voice = Path(
                    read_env_value(bridge_env, "VOXCPM2_PROMPT_WAV")
                ).resolve()
                verify_loaded_voxcpm2_assets(
                    repo_root, previous_image, previous_voice
                )
            except Exception as exc:
                rollback_error = exc
                print(f"[CUSTOMIZE] rollback failed: {exc!r}", flush=True)
            if rollback_error is None:
                update_status(
                    status_path,
                    state="failed",
                    message="激活失败；已自动恢复此前版本，原实时对话仍可使用",
                    rollback_succeeded=True,
                )
            else:
                update_status(
                    status_path,
                    state="failed",
                    message="激活和自动回退都失败，请按 README 的手动回退命令处理",
                    rollback_succeeded=False,
                    error_code="rollback_failed",
                )
            return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--main-env", type=Path, required=True)
    parser.add_argument(
        "--tts-provider",
        choices=("fish_s2pro", "fish_s2_pro", "voxcpm2"),
    )
    parser.add_argument(
        "--tts-env", "--bridge-env", "--vox-env",
        dest="tts_env", type=Path, required=True,
    )
    parser.add_argument("--bridge-instance", default="realtime")
    # Accepted for CLI compatibility with the former Fish manager contract.
    parser.add_argument("--fish-env", type=Path)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def activate(args: argparse.Namespace) -> int:
    job_dir = args.job_dir.resolve()
    runtime_root = args.runtime_root.resolve()
    repo_root = args.repo_root.resolve()
    main_env = args.main_env.resolve()
    bridge_env = args.tts_env.resolve()
    if job_dir.parent != runtime_root or job_dir.name != job_dir.name.lower():
        raise RuntimeError("job directory is outside the configured runtime root")
    manifest_path = job_dir / "manifest.json"
    status_path = job_dir / "status.json"
    manifest = read_json(manifest_path)
    job_id = str(manifest.get("job_id") or "")
    if job_id != job_dir.name:
        raise RuntimeError("job manifest does not match its directory")
    backend = str(
        manifest.get("tts_backend")
        or manifest.get("tts_provider")
        or args.tts_provider
        or "fish_s2_pro"
    ).strip().lower()
    if backend == "fish_s2pro":
        backend = "fish_s2_pro"
    reference_language = str(
        manifest.get("reference_language") or "auto"
    ).strip()
    target_language = str(manifest.get("target_language") or "zh-CN").strip()
    if backend not in {"fish_s2_pro", "voxcpm2"}:
        raise RuntimeError(f"selected TTS backend is not available: {backend}")
    if reference_language not in {"auto", "zh-CN", "en-US", "ja-JP"}:
        raise RuntimeError(f"unsupported reference language: {reference_language}")
    if target_language not in {"zh-CN", "en-US", "ja-JP"}:
        raise RuntimeError(f"unsupported target language: {target_language}")
    image_path = Path(str(manifest.get("image_path") or "")).resolve()
    voice_path = Path(str(manifest.get("voice_path") or "")).resolve()
    transcript_raw = str(manifest.get("transcript_path") or "").strip()
    transcript_path = Path(transcript_raw).resolve() if transcript_raw else None
    for path in (image_path, voice_path, main_env, bridge_env):
        if not path.is_file():
            raise RuntimeError(f"required activation file is missing: {path}")
    if image_path.parent.parent != job_dir or voice_path.parent.parent != job_dir:
        raise RuntimeError("prepared assets are outside the selected job")
    if transcript_path is not None:
        if not transcript_path.is_file() or transcript_path.parent.parent != job_dir:
            raise RuntimeError("prepared transcript is outside the selected job")

    if backend == "voxcpm2":
        return activate_voxcpm2(
            args=args,
            job_dir=job_dir,
            runtime_root=runtime_root,
            repo_root=repo_root,
            main_env=main_env,
            bridge_env=bridge_env,
            manifest=manifest,
            manifest_path=manifest_path,
            status_path=status_path,
            job_id=job_id,
            image_path=image_path,
            voice_path=voice_path,
            transcript_path=transcript_path,
            reference_language=reference_language,
            target_language=target_language,
        )

    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "activation.lock"
    rollback_dir = job_dir / "rollback"
    rollback_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(rollback_dir, 0o700)

    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            update_status(
                status_path,
                state="failed",
                message="另一个定制任务正在激活，请稍后重试",
            )
            return 2

        main_backup = rollback_dir / "custom_cascade.env"
        bridge_backup = rollback_dir / "fish_bridge.env"
        active_path = runtime_root / "active.json"
        active_backup = rollback_dir / "active.json"
        active_existed = active_path.is_file()
        shutil.copy2(main_env, main_backup)
        shutil.copy2(bridge_env, bridge_backup)
        if active_existed:
            shutil.copy2(active_path, active_backup)
        os.chmod(main_backup, 0o600)
        os.chmod(bridge_backup, 0o600)
        update_status(
            status_path,
            clear_errors=True,
            state="activating",
            message="正在写入头像与克隆音色配置",
        )
        bridge_reference: Path | None = None
        online_configuration_changed = False
        try:
            if should_auto_transcribe(manifest):
                transcript = transcribe_reference(
                    repo_root,
                    main_env,
                    voice_path,
                    reference_language,
                )
                transcript_path = voice_path.with_name("voice_reference.txt")
                transcript_path.write_text(transcript + "\n", encoding="utf-8")
                os.chmod(transcript_path, 0o600)
                manifest["transcript_path"] = str(transcript_path)
                manifest["transcript_source"] = "sensevoice_auto"
                write_json(manifest_path, manifest)
            else:
                transcript = transcript_path.read_text(encoding="utf-8").strip()
            if not transcript:
                raise RuntimeError("Fish Speech requires a non-empty reference transcript")
            bridge_reference = stage_fish_reference(bridge_env, voice_path, job_id)
            current_probe, candidate_probe = preflight_fish_reference(
                bridge_env, bridge_reference, transcript
            )
            print(
                "[CUSTOMIZE] Fish preflight passed "
                f"current_rms_dbfs={current_probe.rms_dbfs:.2f} "
                f"candidate_rms_dbfs={candidate_probe.rms_dbfs:.2f} "
                f"candidate_peak_dbfs={candidate_probe.peak_dbfs:.2f}",
                flush=True,
            )
            apply_fish_reference_env(
                bridge_env,
                bridge_reference,
                transcript,
                reference_language,
                target_language,
            )
            online_configuration_changed = True
            patch_env_file(main_env, {
                "DYSTREAM_REF_IMAGE": str(image_path),
                "PIPECAT_TTS_BACKEND": backend,
                "PIPECAT_TTS_REFERENCE_LANGUAGE": reference_language,
                "PIPECAT_TTS_TARGET_LANGUAGE": target_language,
            })
            update_status(
                status_path,
                state="restarting",
                message="模型正在重启并缓存新头像、新音色，请等待",
            )
            time.sleep(1.5)
            restart_fish_bridge(repo_root, bridge_env, args.bridge_instance)
            bridge_probe = probe_restarted_fish_bridge(bridge_env)
            print(
                "[CUSTOMIZE] restarted Fish bridge probe passed "
                f"rms_dbfs={bridge_probe.rms_dbfs:.2f} "
                f"peak_dbfs={bridge_probe.peak_dbfs:.2f}",
                flush=True,
            )
            launch_token = run_demo_restart(repo_root, main_env, args.port, job_id)
            verify_health(args.port, launch_token)
            verify_loaded_assets(
                repo_root, image_path, bridge_reference, args.bridge_instance
            )
            active = {
                "job_id": job_id,
                "activated_at": time.time(),
                "image_url": f"/api/customization/{job_id}/image",
                "voice_source_type": (manifest.get("voice") or {}).get("source_type"),
                "tts_backend": backend,
                "reference_language": reference_language,
                "target_language": target_language,
            }
            write_json(runtime_root / "active.json", active)
            update_status(
                status_path,
                clear_errors=True,
                state="ready",
                message="定制数字人已就绪，可以进入实时对话",
                activated_at=active["activated_at"],
                rollback_available=True,
            )
            print(f"[CUSTOMIZE] activation ready job={job_id}", flush=True)
            return 0
        except Exception as activation_error:
            print(f"[CUSTOMIZE] activation failed: {activation_error!r}", flush=True)
            if not online_configuration_changed:
                if bridge_reference is not None and bridge_reference.is_file():
                    bridge_reference.unlink()
                update_status(
                    status_path,
                    state="failed",
                    message="Voice safety preflight failed; the current avatar remains active.",
                    error_code="voice_preflight_failed",
                    rollback_succeeded=True,
                )
                return 1
            update_status(
                status_path,
                state="rolling_back",
                message="激活失败，正在自动恢复上一个可运行版本",
                error_code="activation_failed",
            )
            rollback_error: Exception | None = None
            try:
                shutil.copy2(main_backup, main_env)
                shutil.copy2(bridge_backup, bridge_env)
                if active_existed:
                    shutil.copy2(active_backup, active_path)
                elif active_path.exists():
                    active_path.unlink()
                restart_fish_bridge(repo_root, bridge_env, args.bridge_instance)
                probe_restarted_fish_bridge(bridge_env)
                rollback_token = run_demo_restart(
                    repo_root, main_env, args.port, f"rollback-{job_id}"[:32]
                )
                verify_health(args.port, rollback_token)
                previous_image = Path(
                    read_env_value(main_env, "DYSTREAM_REF_IMAGE")
                ).resolve()
                previous_voice = Path(
                    read_env_value(bridge_env, "OPENAI_SPEECH_REFERENCE_AUDIO")
                ).resolve()
                verify_loaded_assets(
                    repo_root,
                    previous_image,
                    previous_voice,
                    args.bridge_instance,
                )
                if (
                    bridge_reference is not None
                    and bridge_reference != previous_voice
                    and bridge_reference.name == f"custom-{job_id}.wav"
                    and bridge_reference.is_file()
                ):
                    bridge_reference.unlink()
            except Exception as exc:
                rollback_error = exc
                print(f"[CUSTOMIZE] rollback failed: {exc!r}", flush=True)
            if rollback_error is None:
                update_status(
                    status_path,
                    state="failed",
                    message="激活失败；已自动恢复此前版本，原实时对话仍可使用",
                    rollback_succeeded=True,
                )
            else:
                update_status(
                    status_path,
                    state="failed",
                    message="激活和自动回退都失败，请按 README 的手动回退命令处理",
                    rollback_succeeded=False,
                    error_code="rollback_failed",
                )
            return 1


def main() -> int:
    args = parse_args()
    status_path = args.job_dir.expanduser().resolve() / "status.json"
    try:
        return activate(args)
    except Exception as exc:
        print(f"[CUSTOMIZE] controller failed before completion: {exc!r}", flush=True)
        try:
            update_status(
                status_path,
                state="failed",
                message="激活控制器异常退出；现有服务配置未被确认切换，请查看激活日志",
                error_code="controller_failed",
            )
        except Exception as status_error:
            print(f"[CUSTOMIZE] could not persist controller failure: {status_error!r}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
