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
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any

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
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + MAX_MEDIA_BYTES + 1024 * 1024


class CustomizationInputError(ValueError):
    pass


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


def _runtime_paths() -> dict[str, Path]:
    configured_vox = os.getenv("VOXCPM2_ENV_FILE", "").strip()
    vox_env = Path(configured_vox).expanduser().resolve() if configured_vox else None
    configured_main = os.getenv("CUSTOMIZATION_MAIN_ENV_FILE", "").strip()
    if configured_main:
        main_env = Path(configured_main).expanduser().resolve()
    elif vox_env is not None and (vox_env.parent / "custom_cascade.env").is_file():
        main_env = (vox_env.parent / "custom_cascade.env").resolve()
    else:
        main_env = (REPO_ROOT / ".env").resolve()
    if vox_env is None:
        vox_env = (REPO_ROOT / "voice_service" / ".env.voxcpm2").resolve()
    configured_root = os.getenv("CUSTOMIZATION_ROOT", "").strip()
    if configured_root:
        runtime_root = Path(configured_root).expanduser().resolve()
    elif vox_env.parent.name == "config":
        runtime_root = (vox_env.parent.parent / "customizations").resolve()
    else:
        runtime_root = (REPO_ROOT / "runtime" / "customizations").resolve()
    return {
        "runtime_root": runtime_root,
        "main_env": main_env,
        "vox_env": vox_env,
        "run_demo": (REPO_ROOT / "scripts" / "run_demo.sh").resolve(),
        "controller": (REPO_ROOT / "scripts" / "activate_customization.py").resolve(),
        "validator": (REPO_ROOT / "scripts" / "validate_custom_avatar.py").resolve(),
    }


def _runtime_missing(paths: dict[str, Path]) -> list[str]:
    return [
        key
        for key in ("main_env", "vox_env", "run_demo", "controller", "validator")
        if not paths[key].is_file()
    ]


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
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    duration = frames / float(sample_rate)
    if duration < 3.0:
        raise CustomizationInputError("有效参考语音至少需要 3 秒，建议提供 5–15 秒")
    rms = audioop.rms(pcm, 2) if pcm else 0
    dbfs = 20.0 * math.log10(max(rms, 1) / 32768.0)
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
    warnings: list[str] = []
    if dbfs < -35.0:
        warnings.append("参考音量偏低，克隆效果可能受影响")
    probe.update({
        "reference_duration_seconds": round(duration, 3),
        "reference_sample_rate": sample_rate,
        "reference_channels": 1,
        "reference_dbfs": round(dbfs, 1),
        "active_voice_seconds": round(active_seconds, 3),
        "warnings": warnings,
    })
    return probe


def _prepare_job(job_dir: Path, image_source: Path, media_source: Path, transcript: str) -> dict[str, Any]:
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
    transcript_path: Path | None = None
    if transcript:
        transcript_path = assets_dir / "voice_reference.txt"
        transcript_path.write_text(transcript.strip() + "\n", encoding="utf-8")
        os.chmod(transcript_path, 0o600)
    manifest = {
        "job_id": job_dir.name,
        "created_at": time.time(),
        "image_path": str(image_path.resolve()),
        "voice_path": str(voice_path.resolve()),
        "transcript_path": str(transcript_path.resolve()) if transcript_path else "",
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
        missing = _runtime_missing(paths)
        return web.json_response({
            "status": "ok",
            "available": not missing and not runtime_root_error,
            "missing_components": missing,
            "active": active,
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
        transcript_seen = False
        try:
            reader = await request.multipart()
            async for part in reader:
                if part.name == "transcript":
                    if transcript_seen:
                        raise CustomizationInputError("参考文本只能提交一次")
                    transcript_seen = True
                    transcript = await _read_transcript_part(part)
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
            _write_json(job_dir / "status.json", {
                "job_id": job_id,
                "state": "processing",
                "message": "正在规范化照片并提取参考音轨",
                "updated_at": time.time(),
            })
            manifest = await asyncio.to_thread(
                _prepare_job, job_dir, image_source, media_source, transcript
            )
            status = {
                "job_id": job_id,
                "state": "prepared",
                "message": "素材处理完成，可以激活",
                "updated_at": time.time(),
                "image": manifest["image"],
                "voice": manifest["voice"],
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
        queued = {
            **current,
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
            "--vox-env", str(paths["vox_env"]),
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
