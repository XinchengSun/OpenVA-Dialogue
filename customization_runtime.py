from __future__ import annotations

import asyncio
import audioop
import ipaddress
import json
import math
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from aiohttp import web
from PIL import Image, ImageOps

from pipecat_dystream.public_access import is_loopback_host


REPO_ROOT = Path(__file__).resolve().parent
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".mp4", ".mov", ".webm", ".mkv",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_MEDIA_BYTES = 200 * 1024 * 1024
MAX_TRANSCRIPT_CHARS = 1000
MAX_REFERENCE_SECONDS = 30.0
REFERENCE_TARGET_DBFS = -29.0
REFERENCE_MAX_PEAK_DBFS = -6.0
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + MAX_MEDIA_BYTES + 1024 * 1024

DEFAULT_TTS_SELECTION = {
    "tts_backend": "fish_s2_pro",
    "reference_language": "auto",
    "target_language": "zh-CN",
}
TTS_BACKEND_ALIASES = {"fish_s2pro": "fish_s2_pro"}
TTS_BACKEND_SPECS = (
    {
        "id": "fish_s2_pro",
        "label": "Fish Speech S2 Pro",
        "disabled_reason": "fish_runtime_unavailable",
    },
    {
        "id": "qwen3_tts_1_7b_base",
        "label": "Qwen3-TTS 1.7B Base",
        "disabled_reason": "dependency_and_weights_missing",
    },
    {
        "id": "cosyvoice3_0_5b",
        "label": "CosyVoice3 0.5B",
        "disabled_reason": "runtime_and_weights_missing_on_host",
    },
    {
        "id": "voxcpm2",
        "label": "VoxCPM2",
        "disabled_reason": "installed_stopped_gpu_conflict",
    },
)
TTS_BACKEND_IDS = {item["id"] for item in TTS_BACKEND_SPECS}
TRANSCRIPT_REQUIRED_BACKENDS = {
    "fish_s2_pro",
    "qwen3_tts_1_7b_base",
    "cosyvoice3_0_5b",
}
TTS_BACKEND_ENV_KEYS = {
    "fish_s2_pro": (
        "CUSTOMIZATION_FISH_S2_PRO_ENV_FILE",
    ),
    "qwen3_tts_1_7b_base": (
        "CUSTOMIZATION_QWEN3_TTS_1_7B_ENV_FILE",
        "CUSTOMIZATION_QWEN3_TTS_ENV_FILE",
    ),
    "cosyvoice3_0_5b": (
        "CUSTOMIZATION_COSYVOICE3_ENV_FILE",
        "CUSTOMIZATION_COSYVOICE3_TTS_ENV_FILE",
    ),
    "voxcpm2": ("CUSTOMIZATION_VOXCPM2_ENV_FILE", "VOXCPM2_ENV_FILE"),
}
TTS_BACKEND_BRIDGE_INSTANCE_KEYS = {
    "fish_s2_pro": "CUSTOMIZATION_FISH_S2_PRO_BRIDGE_INSTANCE",
    "qwen3_tts_1_7b_base": "CUSTOMIZATION_QWEN3_TTS_1_7B_BRIDGE_INSTANCE",
    "cosyvoice3_0_5b": "CUSTOMIZATION_COSYVOICE3_BRIDGE_INSTANCE",
    "voxcpm2": "CUSTOMIZATION_VOXCPM2_BRIDGE_INSTANCE",
}
TTS_LANGUAGE_OPTIONS = {
    "reference": (
        {"value": "auto", "label": "自动识别"},
        {"value": "zh-CN", "label": "中文"},
        {"value": "en-US", "label": "英语"},
        {"value": "ja-JP", "label": "日语"},
    ),
    "target": (
        {"value": "zh-CN", "label": "中文"},
        {"value": "en-US", "label": "英语"},
        {"value": "ja-JP", "label": "日语"},
    ),
}
MAX_TTS_OPTION_CHARS = 64


class CustomizationInputError(ValueError):
    pass


def _canonical_tts_backend(value: Any) -> str:
    backend = str(value or DEFAULT_TTS_SELECTION["tts_backend"]).strip().lower()
    backend = TTS_BACKEND_ALIASES.get(backend, backend)
    if backend not in TTS_BACKEND_IDS:
        raise CustomizationInputError(f"unknown TTS backend: {backend}")
    return backend


def _normalize_language(value: Any, *, field: str, default: str) -> str:
    language = default if value in (None, "") else str(value).strip()
    allowed = {
        "reference_language": {"auto", "zh-CN", "en-US", "ja-JP"},
        "target_language": {"zh-CN", "en-US", "ja-JP"},
    }[field]
    if language not in allowed:
        raise CustomizationInputError(f"unsupported {field}: {language}")
    return language


def _normalize_tts_selection(payload: dict[str, Any] | None) -> dict[str, str]:
    source = payload or {}
    backend = source.get("tts_backend")
    if backend in (None, ""):
        backend = source.get("tts_provider")
    return {
        "tts_backend": _canonical_tts_backend(backend),
        "reference_language": _normalize_language(
            source.get("reference_language"),
            field="reference_language",
            default=DEFAULT_TTS_SELECTION["reference_language"],
        ),
        "target_language": _normalize_language(
            source.get("target_language"),
            field="target_language",
            default=DEFAULT_TTS_SELECTION["target_language"],
        ),
    }


def _normalize_runtime_tts_selection(
    payload: dict[str, Any] | None, current_backend: str
) -> dict[str, str]:
    source = dict(payload or {})
    if source.get("tts_backend") in (None, "") and source.get("tts_provider") in (
        None, "",
    ):
        source["tts_backend"] = current_backend
    return _normalize_tts_selection(source)


def _explicit_backend_env(backend: str) -> str:
    for key in TTS_BACKEND_ENV_KEYS[backend]:
        configured = os.getenv(key, "").strip()
        if configured:
            return configured
    return ""


def _configured_ready_backends() -> set[str]:
    ready: set[str] = set()
    for item in os.getenv("CUSTOMIZATION_TTS_READY_BACKENDS", "").split(","):
        value = item.strip().lower()
        if not value:
            continue
        value = TTS_BACKEND_ALIASES.get(value, value)
        if value in TTS_BACKEND_IDS:
            ready.add(value)
    return ready


def _bridge_instance_for_backend(backend: str, default: str) -> str:
    key = TTS_BACKEND_BRIDGE_INSTANCE_KEYS[backend]
    value = os.getenv(key, "").strip() or default
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise RuntimeError(f"{key} must contain only letters, digits, _ or -")
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        if not re.fullmatch(r"[A-Z0-9_]+", key):
            continue
        parsed = shlex.split(raw, comments=False)
        values[key] = parsed[0] if parsed else ""
    return values


def _loopback_port_is_open(host: str, port: int) -> bool:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _configured_backend_is_live(backend: str, env_path: Path) -> bool:
    try:
        values = _read_env_file(env_path)
        if backend == "voxcpm2":
            host = values.get("VOXCPM2_BRIDGE_HOST", "127.0.0.1")
            port = int(values.get("VOXCPM2_BRIDGE_PORT", "8770"))
            return _loopback_port_is_open(host, port)
        base_url = values.get("OPENAI_SPEECH_BASE_URL", "")
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1", "localhost", "::1",
        }:
            return False
        health_path = values.get("OPENAI_SPEECH_HEALTH_PATH", "/health")
        if not health_path.startswith("/"):
            health_path = f"/{health_path}"
        with urlopen(f"{base_url.rstrip('/')}{health_path}", timeout=0.5) as response:
            if not 200 <= response.status < 300:
                return False
        bridge_host = values.get("OPENAI_SPEECH_BRIDGE_HOST", "127.0.0.1")
        bridge_port = int(values.get("OPENAI_SPEECH_BRIDGE_PORT", "8771"))
        return _loopback_port_is_open(bridge_host, bridge_port)
    except (OSError, ValueError, UnicodeError):
        return False


def _openai_bridge_endpoint(env_path: Path) -> tuple[str, int] | None:
    try:
        values = _read_env_file(env_path)
        host = values.get("OPENAI_SPEECH_BRIDGE_HOST", "127.0.0.1")
        port = int(values.get("OPENAI_SPEECH_BRIDGE_PORT", "8771"))
        return host, port
    except (OSError, ValueError, UnicodeError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _patch_env_file(path: Path, updates: dict[str, str]) -> None:
    """Atomically update selected shell-env assignments without exposing others."""
    original = path.read_text(encoding="utf-8")
    applied: set[str] = set()
    lines: list[str] = []
    assignment = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
    for line in original.splitlines():
        match = assignment.match(line)
        if match and match.group(1) in updates:
            key = match.group(1)
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


def _safe_job_id(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if not JOB_ID_RE.fullmatch(value):
        raise web.HTTPNotFound(text="unknown customization job")
    return value


def _current_server_port() -> int:
    configured = os.getenv("CUSTOMIZATION_SERVER_PORT", "").strip()
    if configured:
        return int(configured)
    for index, item in enumerate(sys.argv[:-1]):
        if item == "--port":
            try:
                return int(sys.argv[index + 1])
            except ValueError:
                break
    return int(os.getenv("PORT", "7860"))


def _runtime_paths() -> dict[str, Any]:
    legacy_provider = os.getenv("PIPECAT_TTS_PROVIDER", "").strip().lower()
    provider = (
        os.getenv("PIPECAT_TTS_BACKEND", "").strip().lower()
        or legacy_provider
        or DEFAULT_TTS_SELECTION["tts_backend"]
    )
    provider = TTS_BACKEND_ALIASES.get(provider, provider)
    if provider not in TTS_BACKEND_IDS:
        provider = "unsupported"
    configured_main = os.getenv("CUSTOMIZATION_MAIN_ENV_FILE", "").strip()
    if configured_main:
        main_env = Path(configured_main).expanduser().resolve()
    else:
        main_env = (REPO_ROOT / ".env").resolve()

    configured_tts = (
        _explicit_backend_env(provider) if provider in TTS_BACKEND_IDS else ""
    )
    if not configured_tts:
        configured_tts = os.getenv("CUSTOMIZATION_TTS_ENV_FILE", "").strip()
    if not configured_tts:
        configured_tts = os.getenv("PIPECAT_TTS_BRIDGE_ENV_FILE", "").strip()
    if not configured_tts and provider == "voxcpm2":
        configured_tts = os.getenv("VOXCPM2_ENV_FILE", "").strip()
    tts_env = Path(configured_tts).expanduser().resolve() if configured_tts else None
    if tts_env is None and provider == "fish_s2_pro":
        port = "8773"
        if main_env.is_file():
            match = re.search(
                r"^(?:PIPECAT_TTS_BRIDGE_URI|VOXCPM2_BRIDGE_URI)=.*:(\d+)\s*$",
                main_env.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            if match:
                port = match.group(1)
        local_candidate = main_env.parent / f"fish_bridge_{port}.env"
        if local_candidate.is_file():
            tts_env = local_candidate.resolve()
        elif main_env.parent.name == "config":
            candidates_root = main_env.parent.parent.parent
            matches = sorted(candidates_root.glob(f"*/config/fish_bridge_{port}.env"))
            if matches:
                tts_env = matches[-1].resolve()
    if tts_env is None:
        fallback = ".env.voxcpm2" if provider == "voxcpm2" else ".env.openai_speech"
        tts_env = (REPO_ROOT / "voice_service" / fallback).resolve()

    configured_root = os.getenv("CUSTOMIZATION_ROOT", "").strip()
    if configured_root:
        runtime_root = Path(configured_root).expanduser().resolve()
    elif main_env.parent.name == "config":
        runtime_root = (main_env.parent.parent / "customizations").resolve()
    else:
        runtime_root = (REPO_ROOT / "runtime" / "customizations").resolve()
    bridge_instance = os.getenv(
        "CUSTOMIZATION_TTS_BRIDGE_INSTANCE", "flashav2av"
    ).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", bridge_instance):
        raise RuntimeError(
            "CUSTOMIZATION_TTS_BRIDGE_INSTANCE must contain only letters, digits, _ or -"
        )
    backend_envs: dict[str, Path | None] = {}
    backend_bridge_instances: dict[str, str] = {}
    default_instances = {
        "fish_s2_pro": bridge_instance,
        "qwen3_tts_1_7b_base": "qwen3",
        "cosyvoice3_0_5b": "cosyvoice3",
        "voxcpm2": "voxcpm2",
    }
    for backend in TTS_BACKEND_IDS:
        configured_backend_env = _explicit_backend_env(backend)
        backend_env = (
            Path(configured_backend_env).expanduser().resolve()
            if configured_backend_env
            else None
        )
        if backend == provider and backend_env is None:
            backend_env = tts_env
        backend_envs[backend] = backend_env
        backend_bridge_instances[backend] = _bridge_instance_for_backend(
            backend,
            default_instances[backend],
        )
    return {
        "tts_provider": provider,
        "legacy_tts_provider": legacy_provider or (
            "fish_s2pro" if provider == "fish_s2_pro" else provider
        ),
        "runtime_root": runtime_root,
        "main_env": main_env,
        "tts_env": tts_env,
        "bridge_instance": bridge_instance,
        "backend_envs": backend_envs,
        "backend_bridge_instances": backend_bridge_instances,
        "run_demo": (REPO_ROOT / "scripts" / "run_demo.sh").resolve(),
        "bridge_manager": (
            REPO_ROOT / "voice_service" / "run_openai_speech_bridge.sh"
        ).resolve(),
        "transcriber": (REPO_ROOT / "scripts" / "transcribe_reference.py").resolve(),
        "controller": (REPO_ROOT / "scripts" / "activate_customization.py").resolve(),
        "validator": (REPO_ROOT / "scripts" / "validate_custom_avatar.py").resolve(),
    }


def _runtime_missing(paths: dict[str, Any]) -> list[str]:
    missing = [
        key
        for key in (
            "main_env", "tts_env", "run_demo", "controller", "validator",
        )
        if not paths[key].is_file()
    ]
    if paths["tts_provider"] in TRANSCRIPT_REQUIRED_BACKENDS:
        for key in ("bridge_manager", "transcriber"):
            if not paths[key].is_file():
                missing.append(key)
    if paths["tts_provider"] not in TTS_BACKEND_IDS:
        missing.append("tts_provider")
    return missing


def _tts_options(paths: dict[str, Any]) -> dict[str, Any]:
    current_backend = paths["tts_provider"]
    current_ready = not _runtime_missing(paths)
    configured_ready = _configured_ready_backends()
    openai_backends = {
        "fish_s2_pro", "qwen3_tts_1_7b_base", "cosyvoice3_0_5b",
    }
    instance_counts: dict[str, int] = {}
    endpoint_counts: dict[tuple[str, int], int] = {}
    for backend in openai_backends:
        env_path = paths["backend_envs"].get(backend)
        if env_path is None or not env_path.is_file():
            continue
        instance = paths["backend_bridge_instances"][backend]
        instance_counts[instance] = instance_counts.get(instance, 0) + 1
        endpoint = _openai_bridge_endpoint(env_path)
        if endpoint is not None:
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
    backends: list[dict[str, Any]] = []
    for spec in TTS_BACKEND_SPECS:
        backend = spec["id"]
        backend_env = paths["backend_envs"].get(backend)
        configured = bool(backend_env and backend_env.is_file())
        operator_ready = backend in configured_ready and configured
        live_health = bool(
            configured
            and backend_env is not None
            and _configured_backend_is_live(backend, backend_env)
        )
        instance_conflict = bool(
            backend in openai_backends
            and configured
            and (
                instance_counts.get(paths["backend_bridge_instances"][backend], 0) > 1
                or (
                    backend_env is not None
                    and (endpoint := _openai_bridge_endpoint(backend_env)) is not None
                    and endpoint_counts.get(endpoint, 0) > 1
                )
            )
            and backend != current_backend
        )
        ready = (
            backend == current_backend and current_ready and live_health
        ) or (operator_ready and live_health and not instance_conflict)
        disabled_reason = spec["disabled_reason"]
        if configured:
            disabled_reason = (
                "bridge_instance_conflict"
                if instance_conflict
                else "service_not_ready"
                if (backend == current_backend or backend in configured_ready)
                and not live_health
                else "installed_not_started"
            )
        backends.append({
            "id": backend,
            "label": spec["label"],
            "available": ready,
            "configured": configured,
            "ready": ready,
            "selectable": ready,
            "disabled_reason": None if ready else disabled_reason,
            "supports_voice_clone": True,
            "transcript_required": backend in TRANSCRIPT_REQUIRED_BACKENDS,
            "reference_languages": [
                dict(item) for item in TTS_LANGUAGE_OPTIONS["reference"]
            ],
            "target_languages": [
                dict(item) for item in TTS_LANGUAGE_OPTIONS["target"]
            ],
        })
    defaults = dict(DEFAULT_TTS_SELECTION)
    if current_backend in TTS_BACKEND_IDS:
        defaults["tts_backend"] = current_backend
    return {
        "defaults": defaults,
        "backends": backends,
        "language_options": {
            key: [dict(item) for item in values]
            for key, values in TTS_LANGUAGE_OPTIONS.items()
        },
    }


def _backend_status(paths: dict[str, Any], backend: str) -> dict[str, Any]:
    for item in _tts_options(paths)["backends"]:
        if item["id"] == backend:
            return item
    raise CustomizationInputError(f"unknown TTS backend: {backend}")


def _backend_env_path(paths: dict[str, Any], backend: str) -> Path:
    value = paths["backend_envs"].get(backend)
    if value is None or not value.is_file():
        raise CustomizationInputError(f"TTS backend environment is missing: {backend}")
    return value


def _probe_media(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise CustomizationInputError("无法读取这段音频/视频，请换一个常见格式的文件")
    try:
        probe = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CustomizationInputError("ffprobe 返回了无效媒体信息") from exc
    streams = probe.get("streams") or []
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    if not audio_streams:
        raise CustomizationInputError("文件里没有可用音轨，无法克隆声音")
    duration_raw = (probe.get("format") or {}).get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        for stream in audio_streams:
            try:
                duration = max(duration, float(stream.get("duration", 0)))
            except (TypeError, ValueError):
                pass
    return {
        "source_type": "video" if video_streams else "audio",
        "source_duration_seconds": round(duration, 3) if duration > 0 else None,
        "audio_codec": audio_streams[0].get("codec_name") or "unknown",
    }


def _normalize_image(source: Path, destination: Path) -> dict[str, Any]:
    Image.MAX_IMAGE_PIXELS = 40_000_000
    try:
        with Image.open(source) as opened:
            opened.verify()
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.width < 512 or image.height < 512:
                raise CustomizationInputError("照片至少需要 512×512 像素")
            if image.width * image.height > 16_000_000:
                raise CustomizationInputError("照片分辨率过大，请控制在 1600 万像素以内")
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                image = background.convert("RGB")
            else:
                image = image.convert("RGB")
            original_size = [image.width, image.height]
            if max(image.size) > 4096:
                image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            if image.width < 512 or image.height < 512:
                raise CustomizationInputError(
                    "照片缩放后短边不足 512 像素，请换一张构图更完整的清晰正脸"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG", compress_level=3)
            return {
                "original_size": original_size,
                "normalized_size": [image.width, image.height],
            }
    except CustomizationInputError:
        raise
    except Exception as exc:
        raise CustomizationInputError("无法读取照片，请上传清晰的 JPG、PNG 或 WebP") from exc


def _validate_official_face_path(image_path: Path) -> dict[str, Any]:
    validator = REPO_ROOT / "scripts" / "validate_custom_avatar.py"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(validator), "--image", str(image_path)],
        cwd=str(REPO_ROOT),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if proc.returncode != 0:
        raise CustomizationInputError(
            "官方人脸预检失败：请上传单人、正脸、五官清晰且头部完整的照片"
        )
    try:
        last_line = proc.stdout.strip().splitlines()[-1]
        result = json.loads(last_line)
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("official face validator returned invalid output") from exc
    if not isinstance(result, dict):
        raise RuntimeError("official face validator returned invalid data")
    return result


def _normalize_voice(source: Path, destination: Path) -> dict[str, Any]:
    probe = _probe_media(source)
    source_duration = probe.get("source_duration_seconds")
    if source_duration is None:
        raise CustomizationInputError("无法确定参考声音时长，请先裁成 3–30 秒")
    if source_duration > MAX_REFERENCE_SECONDS + 0.05:
        raise CustomizationInputError("参考声音不能超过 30 秒，请先裁剪后再上传")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:a:0", "-vn", "-t",
        f"{MAX_REFERENCE_SECONDS:g}", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(temporary),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, timeout=120)
        if proc.returncode != 0 or not temporary.is_file():
            raise CustomizationInputError("音轨提取失败，请换一段清晰、无损坏的音频或视频")
        with wave.open(str(temporary), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            pcm = wav.readframes(frames)
    finally:
        if temporary.exists():
            temporary.unlink()
    duration = frames / float(sample_rate)
    if duration < 3.0:
        raise CustomizationInputError("有效参考语音至少需要 3 秒，建议提供 5–15 秒")
    rms = audioop.rms(pcm, 2) if pcm else 0
    dbfs = 20.0 * math.log10(max(rms, 1) / 32768.0)
    input_dbfs = dbfs
    peak = audioop.max(pcm, 2) if pcm else 0
    input_peak_dbfs = 20.0 * math.log10(max(peak, 1) / 32768.0)
    sample_count = len(pcm) // 2
    clipped_samples = sum(
        1 for (sample,) in struct.iter_unpack("<h", pcm) if abs(sample) >= 32700
    )
    clipping_ratio = clipped_samples / max(sample_count, 1)
    dc_offset = audioop.avg(pcm, 2) / 32768.0 if pcm else 0.0
    if clipping_ratio >= 0.001:
        raise CustomizationInputError("reference audio contains clipped samples")
    if abs(dc_offset) > 0.02:
        raise CustomizationInputError("reference audio has excessive DC offset")
    if dbfs < -60.0:
        raise CustomizationInputError("参考音轨几乎是静音，无法克隆声音")
    analysis_frame_samples = max(1, int(sample_rate * 0.02))
    analysis_frame_bytes = analysis_frame_samples * 2
    active_frames = 0
    active_threshold = 10.0 ** (-50.0 / 20.0) * 32768.0
    for offset in range(0, len(pcm) - analysis_frame_bytes + 1, analysis_frame_bytes):
        if audioop.rms(pcm[offset:offset + analysis_frame_bytes], 2) >= active_threshold:
            active_frames += 1
    active_seconds = active_frames * analysis_frame_samples / float(sample_rate)
    required_active_seconds = min(3.0, duration * 0.60)
    if active_seconds < required_active_seconds:
        raise CustomizationInputError(
            "参考音轨里的有效人声太短，请提供 5–15 秒连续、清晰的单人语音"
        )
    peak_dbfs = input_peak_dbfs
    requested_gain_db = min(8.0, REFERENCE_TARGET_DBFS - dbfs)
    peak_limited_gain_db = REFERENCE_MAX_PEAK_DBFS - peak_dbfs
    applied_gain_db = min(requested_gain_db, peak_limited_gain_db)
    gain = 10.0 ** (applied_gain_db / 20.0)
    pcm = audioop.mul(pcm, 2, gain)
    rms = audioop.rms(pcm, 2) if pcm else 0
    peak = audioop.max(pcm, 2) if pcm else 0
    dbfs = 20.0 * math.log10(max(rms, 1) / 32768.0)
    peak_dbfs = 20.0 * math.log10(max(peak, 1) / 32768.0)
    normalized = destination.with_name(f".{destination.name}.{os.getpid()}.normalized.wav")
    try:
        with wave.open(str(normalized), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        os.replace(normalized, destination)
    finally:
        if normalized.exists():
            normalized.unlink()
    warnings: list[str] = []
    if dbfs < -35.0:
        warnings.append("参考音量偏低，克隆效果可能受影响")
    probe.update({
        "reference_duration_seconds": round(duration, 3),
        "reference_sample_rate": sample_rate,
        "reference_channels": 1,
        "reference_dbfs": round(dbfs, 1),
        "reference_peak_dbfs": round(peak_dbfs, 1),
        "reference_input_dbfs": round(input_dbfs, 1),
        "reference_input_peak_dbfs": round(input_peak_dbfs, 1),
        "reference_input_clipping_ratio": round(clipping_ratio, 7),
        "reference_input_dc_offset": round(dc_offset, 6),
        "normalization_gain_db": round(applied_gain_db, 1),
        "active_voice_seconds": round(active_seconds, 3),
        "warnings": warnings,
    })
    return probe


def _write_private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.strip() + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _transcribe_reference(voice_path: Path, language: str) -> str:
    transcriber = REPO_ROOT / "scripts" / "transcribe_reference.py"
    if not transcriber.is_file():
        raise CustomizationInputError("automatic reference transcription is unavailable")
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONSAFEPATH", None)
    inherited_pythonpath = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not inherited_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{inherited_pythonpath}"
    )
    environment["MODELSCOPE_DISABLE_AUTO_UPDATE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [
                os.getenv("PIPECAT_PYTHON", "").strip() or sys.executable,
                str(transcriber),
                "--audio", str(voice_path),
                "--language", language,
            ],
            cwd=str(REPO_ROOT),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CustomizationInputError(
            "automatic reference transcription is unavailable"
        ) from exc
    if proc.returncode != 0:
        raise CustomizationInputError(
            "automatic reference transcription failed; enter the exact transcript manually"
        )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        transcript = str(payload.get("text") or "").strip()
    except (IndexError, json.JSONDecodeError, AttributeError) as exc:
        raise CustomizationInputError(
            "automatic reference transcription returned invalid output"
        ) from exc
    if not transcript or len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise CustomizationInputError(
            "automatic reference transcription was empty; enter the exact transcript manually"
        )
    return transcript


def _prepare_job(
    job_dir: Path,
    image_source: Path,
    media_source: Path,
    transcript: str,
    tts_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection = _normalize_tts_selection(tts_selection)
    assets_dir = job_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(assets_dir, 0o700)
    image_path = assets_dir / "reference.png"
    voice_path = assets_dir / "voice_reference.wav"
    image_info = _normalize_image(image_source, image_path)
    image_info["official_face_preflight"] = _validate_official_face_path(image_path)
    media_info = _normalize_voice(media_source, voice_path)
    os.chmod(image_path, 0o600)
    os.chmod(voice_path, 0o600)
    transcript = transcript.strip()
    transcript_source = "user" if transcript else "none"
    if not transcript and selection["tts_backend"] in TRANSCRIPT_REQUIRED_BACKENDS:
        transcript = _transcribe_reference(
            voice_path, selection["reference_language"]
        )
        transcript_source = "sensevoice_auto"
    transcript_path: Path | None = None
    if transcript:
        transcript_path = assets_dir / "voice_reference.txt"
        _write_private_text(transcript_path, transcript)
    manifest = {
        "job_id": job_dir.name,
        "created_at": time.time(),
        "image_path": str(image_path.resolve()),
        "voice_path": str(voice_path.resolve()),
        "transcript_path": str(transcript_path.resolve()) if transcript_path else "",
        "transcript": transcript,
        "transcript_source": transcript_source,
        **selection,
        "image": image_info,
        "voice": media_info,
    }
    _write_json(job_dir / "manifest.json", manifest)
    return manifest


async def _save_part(part: Any, destination: Path, limit: int) -> int:
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while True:
            chunk = await part.read_chunk(size=1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise CustomizationInputError("上传文件超过大小限制")
            output.write(chunk)
    os.chmod(destination, 0o600)
    return written


async def _read_transcript_part(part: Any) -> str:
    data = bytearray()
    while True:
        chunk = await part.read_chunk(size=4096)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > 8192:
            raise CustomizationInputError("参考文本不能超过 1000 个字符")
    try:
        value = data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise CustomizationInputError("参考文本必须使用 UTF-8") from exc
    if len(value) > MAX_TRANSCRIPT_CHARS:
        raise CustomizationInputError("参考文本不能超过 1000 个字符")
    return value


async def _read_tts_option_part(part: Any, field: str) -> str:
    data = bytearray()
    while True:
        chunk = await part.read_chunk(size=1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_TTS_OPTION_CHARS * 4:
            raise CustomizationInputError(f"invalid {field}")
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise CustomizationInputError(f"{field} must use UTF-8") from exc


def _manifest_transcript(manifest: dict[str, Any]) -> str:
    direct = manifest.get("transcript")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    transcript_path = str(manifest.get("transcript_path") or "").strip()
    if not transcript_path:
        return ""
    path = Path(transcript_path)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


async def _activation_overrides(request: web.Request) -> dict[str, str]:
    if not request.can_read_body:
        return {}
    body = await request.read()
    if not body.strip():
        return {}
    if len(body) > 16 * 1024:
        raise web.HTTPRequestEntityTooLarge(max_size=16 * 1024, actual_size=len(body))
    if request.content_type != "application/json":
        raise web.HTTPBadRequest(text="activation overrides must use application/json")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise web.HTTPBadRequest(text="invalid activation JSON") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="activation JSON must be an object")
    allowed = {
        "tts_backend", "reference_language", "target_language", "transcript",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise web.HTTPBadRequest(text=f"unknown activation field: {unknown[0]}")
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str):
            raise web.HTTPBadRequest(text=f"{key} must be a string")
        normalized[key] = value.strip()
    if "transcript" in normalized:
        transcript = normalized["transcript"]
        if not transcript or len(transcript) > MAX_TRANSCRIPT_CHARS:
            raise web.HTTPBadRequest(text="invalid transcript")
    return normalized


def _is_loopback_request(request: web.Request) -> bool:
    if os.getenv("CUSTOMIZATION_ALLOW_REMOTE", "0").lower() in {"1", "true", "yes"}:
        return True
    # A same-host reverse tunnel connects from loopback, so the TCP peer alone
    # is not enough to distinguish it from the local SSH-forwarded UI.
    if not is_loopback_host(request.host):
        return False
    peer = request.transport.get_extra_info("peername") if request.transport else None
    if not peer:
        return False
    try:
        return ipaddress.ip_address(peer[0]).is_loopback
    except ValueError:
        return False


@web.middleware
async def _local_customization_only(request: web.Request, handler: Any):
    if request.path == "/customize" or request.path.startswith("/api/customization"):
        if not _is_loopback_request(request):
            raise web.HTTPForbidden(text="customization is available through the local SSH tunnel only")
        if (
            request.method == "POST"
            and request.headers.get("X-DyStream-Customize") != "1"
        ):
            raise web.HTTPForbidden(text="missing customization request header")
    return await handler(request)


def register_customization_routes(app: web.Application) -> None:
    paths = _runtime_paths()
    try:
        paths["runtime_root"].mkdir(parents=True, exist_ok=True)
        os.chmod(paths["runtime_root"], 0o700)
        runtime_root_error = ""
    except OSError as exc:
        runtime_root_error = repr(exc)
    app["customization_paths"] = paths
    app["customization_runtime_root_error"] = runtime_root_error
    app.middlewares.append(_local_customization_only)

    async def customize_page(request: web.Request):
        return web.FileResponse(REPO_ROOT / "static" / "customize.html")

    async def active_status(request: web.Request):
        active_path = paths["runtime_root"] / "active.json"
        active = _read_json(active_path) if active_path.is_file() else None
        if active is not None:
            active = {
                **active,
                **_normalize_runtime_tts_selection(active, paths["tts_provider"]),
            }
        missing = _runtime_missing(paths)
        return web.json_response({
            "status": "ok",
            "available": not missing and not runtime_root_error,
            "missing_components": missing,
            "tts_provider": paths["legacy_tts_provider"],
            "transcript_required": paths["tts_provider"] in TRANSCRIPT_REQUIRED_BACKENDS,
            "active": active,
            "tts_options": await asyncio.to_thread(_tts_options, paths),
            "requirements": {
                "image": "JPG/PNG/WebP，至少 512×512，单人正脸且清晰",
                "voice": "WAV/MP3/M4A/FLAC/OGG 或含音轨的视频，至少 3 秒，建议 5–15 秒",
                "activation": "需要受控重启并重新缓存头像与音色，通常约 1–3 分钟",
            },
        })

    async def prepare(request: web.Request):
        if runtime_root_error:
            raise web.HTTPServiceUnavailable(text="customization storage is unavailable")
        if not paths["validator"].is_file():
            raise web.HTTPServiceUnavailable(text="official face validator is unavailable")
        if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
            raise web.HTTPRequestEntityTooLarge(
                max_size=MAX_REQUEST_BYTES,
                actual_size=request.content_length,
            )
        if not request.content_type.startswith("multipart/"):
            raise web.HTTPBadRequest(text="expected multipart form data")
        job_id = uuid.uuid4().hex
        job_dir = paths["runtime_root"] / job_id
        uploads_dir = job_dir / "uploads"
        job_dir.mkdir(parents=True, exist_ok=False)
        uploads_dir.mkdir(parents=True, exist_ok=False)
        os.chmod(job_dir, 0o700)
        os.chmod(uploads_dir, 0o700)
        _write_json(job_dir / "status.json", {
            "job_id": job_id,
            "state": "uploading",
            "message": "正在接收素材",
            "updated_at": time.time(),
        })
        image_source: Path | None = None
        media_source: Path | None = None
        transcript = ""
        text_fields = {
            "transcript": "",
            **_normalize_runtime_tts_selection({}, paths["tts_provider"]),
        }
        fields_seen: set[str] = set()
        try:
            reader = await request.multipart()
            async for part in reader:
                if part.name == "transcript":
                    if "transcript" in fields_seen:
                        raise CustomizationInputError("参考文本只能提交一次")
                    fields_seen.add("transcript")
                    transcript = await _read_transcript_part(part)
                    text_fields["transcript"] = transcript
                    continue
                if part.name in DEFAULT_TTS_SELECTION:
                    if part.name in fields_seen:
                        raise CustomizationInputError(
                            f"{part.name} may only be submitted once"
                        )
                    fields_seen.add(part.name)
                    text_fields[part.name] = await _read_tts_option_part(
                        part, part.name
                    )
                    continue
                if part.name not in {"image", "voice_media"} or not part.filename:
                    continue
                suffix = Path(part.filename).suffix.lower()
                if part.name == "image":
                    if image_source is not None:
                        raise CustomizationInputError("照片只能上传一张")
                    if suffix not in IMAGE_SUFFIXES:
                        raise CustomizationInputError("照片仅支持 JPG、PNG 或 WebP")
                    image_source = uploads_dir / f"source_image{suffix}"
                    await _save_part(part, image_source, MAX_IMAGE_BYTES)
                else:
                    if media_source is not None:
                        raise CustomizationInputError("音频或视频只能上传一个")
                    if suffix not in MEDIA_SUFFIXES:
                        raise CustomizationInputError("不支持这个音频/视频格式")
                    media_source = uploads_dir / f"source_voice{suffix}"
                    await _save_part(part, media_source, MAX_MEDIA_BYTES)
            if image_source is None or media_source is None:
                raise CustomizationInputError("必须同时上传一张照片和一段音频或视频")
            selection = _normalize_tts_selection(text_fields)
            options = await asyncio.to_thread(_tts_options, paths)
            backend = next(
                item for item in options["backends"]
                if item["id"] == selection["tts_backend"]
            )
            if not backend["selectable"]:
                raise CustomizationInputError(
                    f"selected TTS backend is unavailable: {backend['disabled_reason']}"
                )
            _write_json(job_dir / "status.json", {
                "job_id": job_id,
                "state": "processing",
                "message": "正在规范化照片并提取参考音轨",
                "updated_at": time.time(),
            })
            manifest = await asyncio.to_thread(
                _prepare_job,
                job_dir,
                image_source,
                media_source,
                transcript,
                selection,
            )
            status = {
                "job_id": job_id,
                "state": "prepared",
                "message": "素材处理完成，可以激活",
                "updated_at": time.time(),
                "image": manifest["image"],
                "voice": manifest["voice"],
                **selection,
                "transcript": manifest["transcript"],
                "transcript_source": manifest["transcript_source"],
                "requires_restart": True,
            }
            _write_json(job_dir / "status.json", status)
            return web.json_response(status, status=201)
        except CustomizationInputError as exc:
            shutil.rmtree(job_dir / "assets", ignore_errors=True)
            status = {
                "job_id": job_id,
                "state": "failed",
                "message": str(exc),
                "updated_at": time.time(),
            }
            _write_json(job_dir / "status.json", status)
            return web.json_response(status, status=400)
        except asyncio.CancelledError:
            shutil.rmtree(job_dir / "assets", ignore_errors=True)
            _write_json(job_dir / "status.json", {
                "job_id": job_id,
                "state": "failed",
                "message": "上传连接已中断，请重新提交",
                "updated_at": time.time(),
            })
            raise
        except Exception as exc:
            shutil.rmtree(job_dir / "assets", ignore_errors=True)
            status = {
                "job_id": job_id,
                "state": "failed",
                "message": "素材处理失败，请查看服务日志",
                "updated_at": time.time(),
            }
            _write_json(job_dir / "status.json", status)
            request.app["engine"].log(f"[CUSTOMIZE ERROR] prepare job={job_id} error={exc!r}")
            return web.json_response(status, status=500)
        finally:
            if uploads_dir.parent == job_dir and uploads_dir.exists():
                await asyncio.to_thread(shutil.rmtree, uploads_dir, True)

    async def job_status(request: web.Request):
        job_id = _safe_job_id(request.match_info["job_id"])
        status_path = paths["runtime_root"] / job_id / "status.json"
        if not status_path.is_file():
            raise web.HTTPNotFound(text="unknown customization job")
        return web.json_response(_read_json(status_path))

    async def job_image(request: web.Request):
        job_id = _safe_job_id(request.match_info["job_id"])
        image_path = paths["runtime_root"] / job_id / "assets" / "reference.png"
        if not image_path.is_file():
            raise web.HTTPNotFound(text="customization preview is not ready")
        return web.FileResponse(image_path)

    async def activate(request: web.Request):
        missing = _runtime_missing(paths)
        if missing or runtime_root_error:
            return web.json_response({
                "state": "unavailable",
                "message": "当前运行模式未配置本地音色克隆，无法激活定制素材",
                "missing_components": missing,
            }, status=503)
        job_id = _safe_job_id(request.match_info["job_id"])
        job_dir = paths["runtime_root"] / job_id
        status_path = job_dir / "status.json"
        if not status_path.is_file() or not (job_dir / "manifest.json").is_file():
            raise web.HTTPNotFound(text="unknown customization job")
        current = _read_json(status_path)
        if current.get("state") == "ready":
            return web.json_response(current)
        if current.get("state") == "failed" and current.get("rollback_succeeded") is False:
            return web.json_response(current, status=409)
        if current.get("state") not in {"prepared", "failed"}:
            return web.json_response(current, status=409)
        overrides = await _activation_overrides(request)
        manifest_path = job_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        try:
            selection = _normalize_tts_selection({
                **_normalize_runtime_tts_selection(
                    manifest, paths["tts_provider"]
                ),
                **{
                    key: value
                    for key, value in overrides.items()
                    if key in DEFAULT_TTS_SELECTION
                },
            })
        except CustomizationInputError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        options = await asyncio.to_thread(_tts_options, paths)
        backend = next(
            item for item in options["backends"]
            if item["id"] == selection["tts_backend"]
        )
        if not backend["selectable"]:
            return web.json_response({
                **current,
                **selection,
                "state": "unavailable",
                "message": "selected TTS backend is unavailable",
                "disabled_reason": backend["disabled_reason"],
            }, status=409)
        try:
            selected_tts_env = _backend_env_path(paths, selection["tts_backend"])
        except CustomizationInputError as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        selected_bridge_instance = paths["backend_bridge_instances"][
            selection["tts_backend"]
        ]
        transcript = overrides.get("transcript", _manifest_transcript(manifest)).strip()
        if selection["tts_backend"] in TRANSCRIPT_REQUIRED_BACKENDS and not transcript:
            raise web.HTTPBadRequest(
                text="selected TTS backend requires an exact reference transcript"
            )
        if "transcript" in overrides:
            transcript_path = job_dir / "assets" / "voice_reference.txt"
            _write_private_text(transcript_path, transcript)
            manifest["transcript_path"] = str(transcript_path.resolve())
            manifest["transcript"] = transcript
            manifest["transcript_source"] = "user"
        manifest.update(selection)
        _write_json(manifest_path, manifest)
        queued = {
            **current,
            **selection,
            "transcript": transcript,
            "state": "queued",
            "message": "已提交激活；服务将短暂重启",
            "updated_at": time.time(),
        }
        _write_json(status_path, queued)
        log_path = job_dir / "activation.log"
        command = [
            sys.executable, str(paths["controller"]),
            "--job-dir", str(job_dir),
            "--runtime-root", str(paths["runtime_root"]),
            "--repo-root", str(REPO_ROOT),
            "--main-env", str(paths["main_env"]),
            "--tts-provider", selection["tts_backend"],
            "--tts-env", str(selected_tts_env),
            "--bridge-instance", selected_bridge_instance,
            "--previous-tts-provider", paths["tts_provider"],
            "--previous-tts-env", str(paths["tts_env"]),
            "--previous-bridge-instance", paths["backend_bridge_instances"][
                paths["tts_provider"]
            ],
            "--port", str(_current_server_port()),
        ]
        try:
            with log_path.open("ab", buffering=0) as log_output:
                subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log_output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except Exception as exc:
            queued.update({
                "state": "failed",
                "message": "无法启动激活控制器",
                "updated_at": time.time(),
            })
            _write_json(status_path, queued)
            request.app["engine"].log(f"[CUSTOMIZE ERROR] activate job={job_id} error={exc!r}")
            return web.json_response(queued, status=500)
        return web.json_response(queued, status=202)

    app.router.add_get("/customize", customize_page)
    app.router.add_get("/api/customization/active", active_status)
    app.router.add_post("/api/customization/prepare", prepare)
    app.router.add_get("/api/customization/{job_id}", job_status)
    app.router.add_get("/api/customization/{job_id}/image", job_image)
    app.router.add_post("/api/customization/{job_id}/activate", activate)
