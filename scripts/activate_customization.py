#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


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


def update_status(path: Path, **updates: Any) -> dict[str, Any]:
    status = read_json(path) if path.is_file() else {}
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
        ["bash", str(repo_root / "scripts" / "run_demo.sh"), "restart"],
        cwd=str(repo_root),
        env=environment,
        check=True,
        timeout=600,
    )
    return launch_token


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
    voice_path: Path,
    tts_provider: str,
) -> None:
    mse_env = process_environment(repo_root / "logs" / "pipecat_mse.pid")
    if Path(mse_env.get("DYSTREAM_REF_IMAGE", "")).resolve() != image_path:
        raise RuntimeError("managed MSE process did not load the selected reference image")
    if tts_provider == "fish_s2pro":
        bridge_env = process_environment(
            repo_root / "logs" / "openai_speech_bridge.flashav2av.pid"
        )
        loaded_voice = bridge_env.get("OPENAI_SPEECH_REFERENCE_AUDIO", "")
    else:
        bridge_env = process_environment(repo_root / "logs" / "voxcpm2_bridge.pid")
        loaded_voice = bridge_env.get("VOXCPM2_PROMPT_WAV", "")
    if Path(loaded_voice).resolve() != voice_path:
        raise RuntimeError(
            f"managed {tts_provider} process did not load the selected reference audio"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--main-env", type=Path, required=True)
    parser.add_argument(
        "--tts-provider", choices=("fish_s2pro", "voxcpm2"), required=True
    )
    parser.add_argument("--tts-env", type=Path, required=True)
    parser.add_argument("--fish-env", type=Path)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def activate(args: argparse.Namespace) -> int:
    job_dir = args.job_dir.resolve()
    runtime_root = args.runtime_root.resolve()
    repo_root = args.repo_root.resolve()
    main_env = args.main_env.resolve()
    tts_env = args.tts_env.resolve()
    fish_env = args.fish_env.resolve() if args.fish_env else None
    if job_dir.parent != runtime_root or job_dir.name != job_dir.name.lower():
        raise RuntimeError("job directory is outside the configured runtime root")
    manifest_path = job_dir / "manifest.json"
    status_path = job_dir / "status.json"
    manifest = read_json(manifest_path)
    job_id = str(manifest.get("job_id") or "")
    if job_id != job_dir.name:
        raise RuntimeError("job manifest does not match its directory")
    image_path = Path(str(manifest.get("image_path") or "")).resolve()
    voice_path = Path(str(manifest.get("voice_path") or "")).resolve()
    transcript_raw = str(manifest.get("transcript_path") or "").strip()
    transcript_path = Path(transcript_raw).resolve() if transcript_raw else None
    for path in (image_path, voice_path, main_env, tts_env):
        if not path.is_file():
            raise RuntimeError(f"required activation file is missing: {path}")
    if image_path.parent.parent != job_dir or voice_path.parent.parent != job_dir:
        raise RuntimeError("prepared assets are outside the selected job")
    if transcript_path is not None:
        if not transcript_path.is_file() or transcript_path.parent.parent != job_dir:
            raise RuntimeError("prepared transcript is outside the selected job")
    transcript = (
        transcript_path.read_text(encoding="utf-8").strip()
        if transcript_path is not None
        else ""
    )
    if args.tts_provider == "fish_s2pro":
        if not transcript:
            raise RuntimeError("Fish Speech S2 Pro requires an exact reference transcript")
        if fish_env is None or not fish_env.is_file():
            raise RuntimeError("Fish S2 Pro environment file is missing")

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
        tts_backup = rollback_dir / "tts_bridge.env"
        fish_backup = rollback_dir / "fish_s2pro.env"
        active_path = runtime_root / "active.json"
        active_backup = rollback_dir / "active.json"
        active_existed = active_path.is_file()
        shutil.copy2(main_env, main_backup)
        shutil.copy2(tts_env, tts_backup)
        if fish_env is not None:
            shutil.copy2(fish_env, fish_backup)
        if active_existed:
            shutil.copy2(active_path, active_backup)
        os.chmod(main_backup, 0o600)
        os.chmod(tts_backup, 0o600)
        if fish_env is not None:
            os.chmod(fish_backup, 0o600)
        update_status(
            status_path,
            state="activating",
            message="正在写入头像与克隆音色配置",
        )
        try:
            patch_env_file(main_env, {"DYSTREAM_REF_IMAGE": str(image_path)})
            if args.tts_provider == "fish_s2pro":
                patch_env_file(tts_env, {
                    "OPENAI_SPEECH_REFERENCE_AUDIO": str(voice_path),
                    "OPENAI_SPEECH_REFERENCE_TEXT": transcript,
                })
                assert fish_env is not None
                patch_env_file(
                    fish_env, {"FISH_REFERENCE_DIR": str(voice_path.parent)}
                )
            else:
                patch_env_file(tts_env, {
                    "VOXCPM2_PROMPT_WAV": str(voice_path),
                    "VOXCPM2_PROMPT_TEXT": "",
                    "VOXCPM2_PROMPT_TEXT_FILE": (
                        str(transcript_path) if transcript_path else ""
                    ),
                })
            update_status(
                status_path,
                state="restarting",
                message="模型正在重启并缓存新头像、新音色，请等待",
            )
            time.sleep(1.5)
            launch_token = run_demo_restart(repo_root, main_env, args.port, job_id)
            verify_health(args.port, launch_token)
            verify_loaded_assets(
                repo_root, image_path, voice_path, args.tts_provider
            )
            active = {
                "job_id": job_id,
                "activated_at": time.time(),
                "image_url": f"/api/customization/{job_id}/image",
                "voice_source_type": (manifest.get("voice") or {}).get("source_type"),
            }
            write_json(runtime_root / "active.json", active)
            update_status(
                status_path,
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
                shutil.copy2(tts_backup, tts_env)
                if fish_env is not None:
                    shutil.copy2(fish_backup, fish_env)
                if active_existed:
                    shutil.copy2(active_backup, active_path)
                elif active_path.exists():
                    active_path.unlink()
                rollback_token = run_demo_restart(
                    repo_root, main_env, args.port, f"rollback-{job_id}"[:32]
                )
                verify_health(args.port, rollback_token)
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
