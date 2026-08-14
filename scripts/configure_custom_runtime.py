#!/usr/bin/env python3
"""Create private custom-cascade env files without printing credentials."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_RUNTIME_ROOT = Path(
    os.environ.get(
        "FLASHAV2AV_DATA_ROOT",
        Path.home() / ".local" / "share" / "flashav2av",
    )
)


def _parse_env(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        if key.replace("_", "").isalnum():
            values[key] = raw_value
    return lines, values


def _plain(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _upsert(lines: list[str], replacements: dict[str, str]) -> list[str]:
    output: list[str] = []
    pending = dict(replacements)
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0]
            if key in pending:
                output.append(f"{key}={pending.pop(key)}")
                continue
        output.append(line)
    if pending:
        if output and output[-1]:
            output.append("")
        output.append("# Generated custom-cascade overrides.")
        output.extend(f"{key}={value}" for key, value in pending.items())
    return output


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            if not text.endswith("\n"):
                stream.write("\n")
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _raw_first(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        raw_value = values.get(key, "")
        if _plain(raw_value):
            return raw_value
    return ""


def _parser() -> argparse.ArgumentParser:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create isolated custom-cascade and TTS runtime env files."
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--source-env", type=Path)
    parser.add_argument("--pipecat-python", type=Path)
    parser.add_argument("--prompt-wav", type=Path)
    parser.add_argument("--prompt-text-file", type=Path)
    parser.add_argument("--dystream-gpus")
    parser.add_argument(
        "--tts-backend",
        choices=("fish_s2pro", "voxcpm2"),
        default="fish_s2pro",
    )
    parser.add_argument("--fish-gpus", default="2,3")
    parser.add_argument("--fish-profile", choices=("balanced", "low_ttfa_gapless"), default="low_ttfa_gapless")
    parser.add_argument("--fish-http-port", type=int, default=8001)
    parser.add_argument("--tts-gpu", default="2", help="Physical GPU used only by the VoxCPM2 fallback.")
    parser.add_argument("--llm-model", default="qwen3.7-flash")
    parser.add_argument(
        "--realtime-search-mode",
        choices=("off", "smart", "auto", "always"),
        default="off",
    )
    parser.add_argument(
        "--realtime-search-strategy",
        choices=("turbo", "max", "agent"),
        default="turbo",
    )
    parser.add_argument(
        "--realtime-search-model",
        default="",
        help="Optional model used only for real-time search turns.",
    )
    parser.add_argument("--bridge-uri")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()
    runtime_root = args.runtime_root.resolve()
    source_env = (args.source_env or repo_root / ".env").resolve()
    prompt_wav = (args.prompt_wav or repo_root / "wav_files" / "11.wav").resolve()
    prompt_text_file = args.prompt_text_file.resolve() if args.prompt_text_file else None

    if not source_env.is_file():
        raise SystemExit(f"source env is missing: {source_env}")
    if not prompt_wav.is_file():
        raise SystemExit(f"prompt wav is missing: {prompt_wav}")
    if prompt_text_file is not None and not prompt_text_file.is_file():
        raise SystemExit(f"prompt text file is missing: {prompt_text_file}")
    if runtime_root == Path("/"):
        raise SystemExit("runtime root must not be /")

    source_lines, values = _parse_env(source_env)
    llm_key = _raw_first(
        values,
        "PIPECAT_LLM_API_KEY",
        "OPENAI_API_KEY",
        "PIPECAT_S2S_API_KEY",
        "DASHSCOPE_API_KEY",
    )
    if not llm_key:
        raise SystemExit("source env contains no reusable LLM/DashScope API key")

    custom_env = runtime_root / "config" / "custom_cascade.env"
    asr_cache = runtime_root / "cache" / "pipecat" / "modelscope"
    dystream_gpus = args.dystream_gpus or _plain(values.get("CUDA_VISIBLE_DEVICES", "0,1"))
    gpu_items = [item.strip() for item in dystream_gpus.split(",") if item.strip()]
    if len(gpu_items) != 2 or gpu_items[0] == gpu_items[1]:
        raise SystemExit("DyStream requires exactly two distinct CUDA_VISIBLE_DEVICES")
    if not all(item.isdigit() for item in gpu_items):
        raise SystemExit("--dystream-gpus must contain two physical numeric GPU ids")

    if args.tts_backend == "fish_s2pro":
        fish_gpu_items = [
            item.strip() for item in args.fish_gpus.split(",") if item.strip()
        ]
        if (
            len(fish_gpu_items) != 2
            or fish_gpu_items[0] == fish_gpu_items[1]
            or not all(item.isdigit() for item in fish_gpu_items)
        ):
            raise SystemExit("--fish-gpus must contain two distinct physical numeric GPU ids")
        if set(fish_gpu_items) & set(gpu_items):
            raise SystemExit("Fish physical GPUs must not overlap the two DyStream GPUs")
        if prompt_text_file is None:
            raise SystemExit("Fish S2 Pro requires --prompt-text-file")
        prompt_text = prompt_text_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise SystemExit("Fish S2 Pro prompt transcript must not be empty")
        if not 1 <= args.fish_http_port <= 65535:
            raise SystemExit("--fish-http-port must be between 1 and 65535")
        bridge_uri = args.bridge_uri or "ws://127.0.0.1:8771"
        bridge_env = runtime_root / "config" / "fish_speech_bridge.env"
        upstream_env = runtime_root / "config" / "fish_s2pro.env"
    else:
        fish_gpu_items = []
        prompt_text = ""
        if not args.tts_gpu.isdigit():
            raise SystemExit("--tts-gpu must be one physical numeric GPU id")
        if args.tts_gpu in gpu_items:
            raise SystemExit("VoxCPM2 physical GPU must not overlap the two DyStream GPUs")
        bridge_uri = args.bridge_uri or "ws://127.0.0.1:8770"
        bridge_env = runtime_root / "config" / "voxcpm2.env"
        upstream_env = None

    bridge_url = urlsplit(bridge_uri)
    try:
        bridge_port = bridge_url.port
    except ValueError as exc:
        raise SystemExit(f"invalid local bridge URI: {exc}") from exc
    if (
        bridge_url.scheme != "ws"
        or bridge_url.hostname != "127.0.0.1"
        or bridge_port is None
        or bridge_url.path not in {"", "/"}
        or bridge_url.query
        or bridge_url.fragment
    ):
        raise SystemExit("--bridge-uri must use ws://127.0.0.1:PORT")
    if args.tts_backend == "fish_s2pro" and bridge_port == args.fish_http_port:
        raise SystemExit("Fish HTTP port and PCM bridge port must be different")

    pipecat_python = (
        args.pipecat_python.resolve()
        if args.pipecat_python
        else runtime_root / "venvs" / "pipecat" / "bin" / "python"
    )

    replacements = {
        "PIPECAT_MSE_DIALOG_MODE": "custom_cascade",
        "PIPECAT_LLM_API_KEY": llm_key,
        "PIPECAT_LLM_BASE_URL": _raw_first(values, "PIPECAT_LLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "PIPECAT_LLM_MODEL": shlex.quote(args.llm_model),
        "PIPECAT_LLM_ENABLE_THINKING": "false",
        "PIPECAT_REALTIME_SEARCH_MODE": args.realtime_search_mode,
        "PIPECAT_REALTIME_SEARCH_STRATEGY": args.realtime_search_strategy,
        "PIPECAT_REALTIME_SEARCH_MODEL": shlex.quote(args.realtime_search_model),
        "PIPECAT_LLM_MAX_RETRIES": "0",
        "PIPECAT_LLM_TIMEOUT_SEC": "8.0",
        "PIPECAT_ASR_MODEL": "paraformer-zh-streaming",
        "PIPECAT_ASR_DEVICE": "cpu",
        "PIPECAT_ASR_HUB": "ms",
        "PIPECAT_ASR_CHUNK_SIZE": "0,10,5",
        "PIPECAT_ASR_ENCODER_CHUNK_LOOK_BACK": "4",
        "PIPECAT_ASR_DECODER_CHUNK_LOOK_BACK": "1",
        "PIPECAT_ASR_PRE_ROLL_SEC": "0.30",
        "MODELSCOPE_CACHE": shlex.quote(str(asr_cache)),
        "PIPECAT_TTS_PROVIDER": args.tts_backend,
        "PIPECAT_TTS_BACKEND": (
            "fish_s2_pro" if args.tts_backend == "fish_s2pro" else "voxcpm2"
        ),
        "PIPECAT_TTS_REFERENCE_LANGUAGE": "auto",
        "PIPECAT_TTS_TARGET_LANGUAGE": "zh-CN",
        "PIPECAT_TTS_BRIDGE_URI": bridge_uri,
        "PIPECAT_TTS_MODEL": "fishaudio/s2-pro" if args.tts_backend == "fish_s2pro" else "VoxCPM2",
        "PIPECAT_TTS_VOICE": "zero-shot-cloned" if args.tts_backend == "fish_s2pro" else "cloned",
        "PIPECAT_TTS_CONNECT_TIMEOUT_SEC": "45",
        "PIPECAT_TTS_BRIDGE_ENV_FILE": shlex.quote(str(bridge_env)),
        "FISH_S2PRO_ENV_FILE": shlex.quote(str(upstream_env)) if upstream_env else "",
        # Keep legacy keys for the provider-neutral adapter and older snapshots.
        "VOXCPM2_BRIDGE_URI": bridge_uri,
        "VOXCPM2_CONNECT_TIMEOUT_SEC": "45",
        "VOXCPM2_ENV_FILE": shlex.quote(str(bridge_env)) if args.tts_backend == "voxcpm2" else "",
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_items),
        "MOTION_GPU": "0",
        "RENDER_GPU": "1",
        "ALLOW_SHARED_DYSTREAM_GPU": "0",
    }
    custom_lines = _upsert(source_lines, replacements)
    _write_private(custom_env, "\n".join(custom_lines))

    if args.tts_backend == "fish_s2pro":
        fish_root = runtime_root / "runtime" / "fish-s2-pro"
        bridge_values = {
            "OPENAI_SPEECH_PYTHON": str(pipecat_python),
            "OPENAI_SPEECH_BRIDGE_HOST": "127.0.0.1",
            "OPENAI_SPEECH_BRIDGE_PORT": str(bridge_port),
            "OPENAI_SPEECH_BASE_URL": f"http://127.0.0.1:{args.fish_http_port}",
            "OPENAI_SPEECH_HEALTH_PATH": "/health",
            "OPENAI_SPEECH_ENDPOINT": "/v1/audio/speech",
            "OPENAI_SPEECH_PROVIDER": "sglang",
            "OPENAI_SPEECH_BACKEND": "fish_s2_pro",
            "OPENAI_SPEECH_MODEL": "fishaudio/s2-pro",
            "OPENAI_SPEECH_VOICE": "default",
            "OPENAI_SPEECH_SAMPLE_RATE": "44100",
            "OPENAI_SPEECH_REFERENCE_LANGUAGE": "auto",
            "OPENAI_SPEECH_TARGET_LANGUAGE": "zh-CN",
            "OPENAI_SPEECH_REFERENCE_AUDIO": str(prompt_wav),
            "OPENAI_SPEECH_REFERENCE_TEXT": prompt_text,
            "OPENAI_SPEECH_EXTRA_BODY_JSON": '{"seed":20260811}',
            "OPENAI_SPEECH_WARMUP_TEXT": "\u4f60\u597d\u3002",
            "OPENAI_SPEECH_REQUEST_TIMEOUT_SEC": "120",
            "OPENAI_SPEECH_CONNECT_TIMEOUT_SEC": "10",
        }
        upstream_values = {
            "FISH_ROOT": str(fish_root),
            "SGLANG_OMNI_SRC": str(fish_root / "src" / "sglang-omni-ghproxy"),
            "FISH_VENV": str(fish_root / "venv"),
            "FISH_MODEL": str(fish_root / "models" / "fishaudio-s2-pro"),
            "FISH_REFERENCE_DIR": str(prompt_wav.parent),
            "FISH_CUDA_VISIBLE_DEVICES": ",".join(fish_gpu_items),
            "FISH_HTTP_HOST": "127.0.0.1",
            "FISH_HTTP_PORT": str(args.fish_http_port),
            "FISH_PROFILE": args.fish_profile,
            "FISH_START_TIMEOUT_SEC": "900",
            "FISH_STOP_TIMEOUT_SEC": "120",
        }
        upstream_text = "\n".join(
            f"{key}={shlex.quote(value)}" for key, value in upstream_values.items()
        )
        assert upstream_env is not None
        _write_private(upstream_env, upstream_text)
    else:
        model_path = runtime_root / "models" / "VoxCPM2"
        voxcpm_python = runtime_root / "venvs" / "voxcpm2-nano-2.0.3" / "bin" / "python"
        bridge_values = {
            "CUDA_VISIBLE_DEVICES": args.tts_gpu,
            "VOXCPM2_DEVICES": "0",
            "VOXCPM2_PYTHON": str(voxcpm_python),
            "VOXCPM2_MODEL_PATH": str(model_path),
            "VOXCPM2_PROMPT_WAV": str(prompt_wav),
            "VOXCPM2_BRIDGE_HOST": "127.0.0.1",
            "VOXCPM2_BRIDGE_PORT": str(bridge_port),
            "VOXCPM2_INFERENCE_TIMESTEPS": "10",
            "VOXCPM2_GPU_MEMORY_UTILIZATION": "0.90",
            "VOXCPM2_MAX_BATCHED_TOKENS": "8192",
            "VOXCPM2_MAX_NUM_SEQS": "16",
            "VOXCPM2_WARMUP_TEXT": "\u4f60\u597d\u3002",
            "VOXCPM2_WARMUP_TIMEOUT_SEC": "120",
            "VOXCPM2_START_TIMEOUT_SEC": "600",
            "VOXCPM2_STOP_TIMEOUT_SEC": "120",
            "VOXCPM2_LOG_LEVEL": "INFO",
        }
        if prompt_text_file is not None:
            bridge_values["VOXCPM2_PROMPT_TEXT_FILE"] = str(prompt_text_file)
    bridge_text = "\n".join(
        f"{key}={shlex.quote(value)}" for key, value in bridge_values.items()
    )
    _write_private(bridge_env, bridge_text)

    print(f"CUSTOM_ENV_READY={custom_env}")
    print(f"TTS_BRIDGE_ENV_READY={bridge_env}")
    if upstream_env is not None:
        print(f"FISH_S2PRO_ENV_READY={upstream_env}")
    print(
        "GPU_MAP_READY="
        + (
            f"dystream:{','.join(gpu_items)} fish:{','.join(fish_gpu_items)}"
            if args.tts_backend == "fish_s2pro"
            else f"dystream:{','.join(gpu_items)} voxcpm2:{args.tts_gpu}"
        )
    )
    print("CREDENTIALS_COPIED=yes values_printed=no")


if __name__ == "__main__":
    main()
