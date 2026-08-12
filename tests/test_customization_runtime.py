import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

try:
    import fcntl  # noqa: F401
except ModuleNotFoundError:
    import sys
    import types

    fcntl_stub = types.ModuleType("fcntl")
    fcntl_stub.LOCK_EX = 2
    fcntl_stub.LOCK_NB = 4
    fcntl_stub.flock = lambda *_args, **_kwargs: None
    sys.modules["fcntl"] = fcntl_stub

from PIL import Image

from customization_runtime import (
    CustomizationInputError,
    _normalize_image,
    _normalize_voice,
    _patch_env_file,
    _runtime_missing,
    _runtime_paths,
)


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "activate_customization.py"
CONTROLLER_SPEC = importlib.util.spec_from_file_location("activate_customization", CONTROLLER_PATH)
CONTROLLER = importlib.util.module_from_spec(CONTROLLER_SPEC)
CONTROLLER_SPEC.loader.exec_module(CONTROLLER)
STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"


class CustomizationRuntimeTests(unittest.TestCase):
    def _activation_fixture(self, root, provider="fish_s2pro", transcript="参考文本。"):
        runtime_root = root / "customizations"
        job_dir = runtime_root / ("a" * 32)
        assets = job_dir / "assets"
        assets.mkdir(parents=True)
        image = assets / "reference.png"
        voice = assets / "voice_reference.wav"
        image.write_bytes(b"image")
        voice.write_bytes(b"voice")
        transcript_path = assets / "voice_reference.txt"
        if transcript:
            transcript_path.write_text(transcript + "\n", encoding="utf-8")
        main_env = root / "custom.env"
        tts_env = root / "tts.env"
        fish_env = root / "fish.env"
        main_env.write_text("DYSTREAM_REF_IMAGE=/old.png\n", encoding="utf-8")
        if provider == "fish_s2pro":
            tts_env.write_text(
                "OPENAI_SPEECH_REFERENCE_AUDIO=/old.wav\n"
                "OPENAI_SPEECH_REFERENCE_TEXT='old text'\n",
                encoding="utf-8",
            )
            fish_env.write_text("FISH_REFERENCE_DIR=/old/references\n", encoding="utf-8")
        else:
            tts_env.write_text("VOXCPM2_PROMPT_WAV=/old.wav\n", encoding="utf-8")
        manifest = {
            "job_id": job_dir.name,
            "image_path": str(image),
            "voice_path": str(voice),
            "transcript_path": str(transcript_path) if transcript else "",
            "voice": {"source_type": "audio"},
        }
        (job_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (job_dir / "status.json").write_text("{}\n", encoding="utf-8")
        args = SimpleNamespace(
            job_dir=job_dir,
            runtime_root=runtime_root,
            repo_root=root / "repo",
            main_env=main_env,
            tts_provider=provider,
            tts_env=tts_env,
            fish_env=fish_env if provider == "fish_s2pro" else None,
            port=7860,
        )
        return args, image, voice, main_env, tts_env, fish_env

    def test_fish_runtime_paths_use_provider_neutral_bridge_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            main_env = config / "custom_cascade.env"
            tts_env = config / "fish_speech_bridge.env"
            fish_env = config / "fish_s2pro.env"
            for path in (main_env, tts_env, fish_env):
                path.write_text("READY=1\n", encoding="utf-8")
            environment = {
                "PIPECAT_TTS_PROVIDER": "fish_s2pro",
                "PIPECAT_TTS_BRIDGE_ENV_FILE": str(tts_env),
                "FISH_S2PRO_ENV_FILE": str(fish_env),
                "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                paths = _runtime_paths()
            self.assertEqual(paths["tts_provider"], "fish_s2pro")
            self.assertEqual(paths["tts_env"], tts_env.resolve())
            self.assertEqual(paths["fish_env"], fish_env.resolve())
            self.assertEqual(paths["runtime_root"], (root / "customizations").resolve())
            self.assertNotIn("tts_env", _runtime_missing(paths))
            self.assertNotIn("fish_env", _runtime_missing(paths))

    def test_fish_runtime_requires_upstream_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_env = root / "main.env"
            tts_env = root / "bridge.env"
            for path in (main_env, tts_env):
                path.write_text("READY=1\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {
                "PIPECAT_TTS_PROVIDER": "fish_s2pro",
                "PIPECAT_TTS_BRIDGE_ENV_FILE": str(tts_env),
                "FISH_S2PRO_ENV_FILE": str(root / "missing-fish.env"),
                "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                "CUSTOMIZATION_ROOT": str(root / "customizations"),
            }, clear=True):
                paths = _runtime_paths()
            self.assertIn("fish_env", _runtime_missing(paths))

    def test_product_pages_use_current_names_and_switch_flow(self):
        customize = (STATIC_ROOT / "customize.html").read_text(encoding="utf-8")
        realtime = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn("生成并启用", customize)
        self.assertIn("更换数字人", customize)
        self.assertIn("新人物启用时会自动替换旧模型", customize)
        self.assertIn("立即进入对话", customize)
        self.assertIn("重新检查状态", customize)
        self.assertIn("ACTIVATION_HARD_LIMIT_MS = 5 * 60 * 1000", customize)
        self.assertIn("Math.min(95, 42 +", customize)
        self.assertIn("await fetchActiveSnapshot()", customize)
        self.assertIn("String(authoritativeActive.job_id) !== String(readyJob)", customize)
        self.assertIn("window.location.assign('/')", customize)
        self.assertNotIn("激活并重启模型", customize)
        self.assertNotIn("renderActive({\n        job_id: readyJob", customize)
        self.assertIn("<title>FlashAV2AV · 实时数字人</title>", realtime)
        self.assertIn("更换数字人", realtime)
        self.assertIn("运行日志", realtime)
        self.assertNotIn("Doubao SeedDuplex + DyStream Live AV2AV", realtime)

    def test_env_patch_changes_only_requested_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.write_text(
                "API_KEY=keep-me\nDYSTREAM_REF_IMAGE=/old.png\n# note\n",
                encoding="utf-8",
            )
            _patch_env_file(path, {
                "DYSTREAM_REF_IMAGE": "/new.png",
                "NEW_EMPTY_VALUE": "",
            })
            value = path.read_text(encoding="utf-8")
            self.assertIn("API_KEY=keep-me", value)
            self.assertIn("DYSTREAM_REF_IMAGE=/new.png", value)
            self.assertIn("NEW_EMPTY_VALUE=", value)
            self.assertNotIn("/old.png", value)

    def test_controller_env_patch_deduplicates_and_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.write_text(
                "KEEP=1\nDYSTREAM_REF_IMAGE=/old-a.png\n"
                "DYSTREAM_REF_IMAGE=/old-b.png\n",
                encoding="utf-8",
            )
            CONTROLLER.patch_env_file(
                path, {"DYSTREAM_REF_IMAGE": "/path with spaces/reference.png"}
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            selected = [line for line in lines if line.startswith("DYSTREAM_REF_IMAGE=")]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0], "DYSTREAM_REF_IMAGE='/path with spaces/reference.png'")
            self.assertIn("KEEP=1", lines)

    def test_controller_restart_environment_can_import_repo_modules(self):
        repo = Path("/srv/dystream")
        with mock.patch.dict(
            CONTROLLER.os.environ,
            {"PYTHONPATH": "/unrelated", "PYTHONSAFEPATH": "1"},
            clear=False,
        ):
            environment, token = CONTROLLER.build_restart_environment(
                repo, Path("/runtime/custom.env"), 7860, "a" * 32
            )
        self.assertEqual(environment["PYTHONPATH"].split(CONTROLLER.os.pathsep)[0], str(repo))
        self.assertNotIn("PYTHONSAFEPATH", environment)
        self.assertEqual(environment["ENV_FILE"], str(Path("/runtime/custom.env")))
        self.assertEqual(token, "custom-aaaaaaaaaaaaaaaa")

    def test_loaded_asset_check_uses_fish_bridge_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "reference.png"
            voice = Path(directory) / "reference.wav"
            calls = []

            def fake_environment(pid_path):
                calls.append(pid_path.name)
                if pid_path.name == "pipecat_mse.pid":
                    return {"DYSTREAM_REF_IMAGE": str(image)}
                return {"OPENAI_SPEECH_REFERENCE_AUDIO": str(voice)}

            with mock.patch.object(CONTROLLER, "process_environment", fake_environment):
                CONTROLLER.verify_loaded_assets(
                    Path(directory) / "repo", image, voice, "fish_s2pro"
                )
            self.assertEqual(calls, [
                "pipecat_mse.pid", "openai_speech_bridge.flashav2av.pid"
            ])

    def test_loaded_asset_check_preserves_voxcpm2_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "reference.png"
            voice = Path(directory) / "reference.wav"
            calls = []

            def fake_environment(pid_path):
                calls.append(pid_path.name)
                if pid_path.name == "pipecat_mse.pid":
                    return {"DYSTREAM_REF_IMAGE": str(image)}
                return {"VOXCPM2_PROMPT_WAV": str(voice)}

            with mock.patch.object(CONTROLLER, "process_environment", fake_environment):
                CONTROLLER.verify_loaded_assets(
                    Path(directory) / "repo", image, voice, "voxcpm2"
                )
            self.assertEqual(calls, ["pipecat_mse.pid", "voxcpm2_bridge.pid"])

    def test_fish_activation_updates_all_provider_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            args, image, voice, main_env, tts_env, fish_env = (
                self._activation_fixture(Path(directory))
            )
            with mock.patch.object(
                CONTROLLER, "run_demo_restart", return_value="launch-token"
            ), mock.patch.object(CONTROLLER, "verify_health"), mock.patch.object(
                CONTROLLER, "verify_loaded_assets"
            ) as loaded, mock.patch.object(CONTROLLER.time, "sleep"):
                result = CONTROLLER.activate(args)
            self.assertEqual(result, 0)
            self.assertIn(str(image), main_env.read_text(encoding="utf-8"))
            tts_value = tts_env.read_text(encoding="utf-8")
            self.assertIn(str(voice), tts_value)
            self.assertIn("参考文本。", tts_value)
            self.assertIn(str(voice.parent), fish_env.read_text(encoding="utf-8"))
            loaded.assert_called_once_with(
                args.repo_root.resolve(), image.resolve(), voice.resolve(), "fish_s2pro"
            )

    def test_fish_activation_rejects_missing_transcript_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, main_env, tts_env, fish_env = self._activation_fixture(
                Path(directory), transcript=""
            )
            originals = tuple(
                path.read_text(encoding="utf-8")
                for path in (main_env, tts_env, fish_env)
            )
            with mock.patch.object(CONTROLLER, "run_demo_restart") as restart:
                with self.assertRaisesRegex(RuntimeError, "exact reference transcript"):
                    CONTROLLER.activate(args)
            restart.assert_not_called()
            self.assertEqual(originals, tuple(
                path.read_text(encoding="utf-8")
                for path in (main_env, tts_env, fish_env)
            ))

    def test_fish_activation_rolls_back_all_three_env_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, main_env, tts_env, fish_env = self._activation_fixture(
                Path(directory)
            )
            originals = tuple(
                path.read_text(encoding="utf-8")
                for path in (main_env, tts_env, fish_env)
            )
            with mock.patch.object(
                CONTROLLER, "run_demo_restart", return_value="launch-token"
            ), mock.patch.object(
                CONTROLLER, "verify_health", side_effect=[RuntimeError("bad"), None]
            ), mock.patch.object(CONTROLLER.time, "sleep"):
                result = CONTROLLER.activate(args)
            self.assertEqual(result, 1)
            self.assertEqual(originals, tuple(
                path.read_text(encoding="utf-8")
                for path in (main_env, tts_env, fish_env)
            ))

    def test_image_is_exif_safe_rgb_png_without_custom_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jpg"
            destination = Path(directory) / "reference.png"
            Image.new("RGB", (900, 700), (32, 64, 96)).save(source, quality=90)
            info = _normalize_image(source, destination)
            self.assertEqual(info["original_size"], [900, 700])
            self.assertEqual(info["normalized_size"], [900, 700])
            with Image.open(destination) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (900, 700))

    def test_small_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "small.png"
            Image.new("RGB", (511, 700)).save(source)
            with self.assertRaises(CustomizationInputError):
                _normalize_image(source, Path(directory) / "out.png")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_audio_is_normalized_to_16k_mono_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            destination = Path(directory) / "reference.wav"
            sample_rate = 44100
            samples = bytearray()
            for index in range(int(sample_rate * 3.2)):
                value = int(0.2 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
                samples.extend(struct.pack("<hh", value, value))
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(bytes(samples))
            info = _normalize_voice(source, destination)
            self.assertEqual(info["source_type"], "audio")
            with wave.open(str(destination), "rb") as wav:
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertGreater(wav.getnframes(), 3 * 16000)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_video_audio_track_is_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            destination = Path(directory) / "reference.wav"
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=512x512:d=3.2",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=3.2",
                "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source),
            ], check=True, timeout=60)
            info = _normalize_voice(source, destination)
            self.assertEqual(info["source_type"], "video")
            self.assertGreaterEqual(info["reference_duration_seconds"], 3.0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_near_silent_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silent.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * (16000 * 3))
            with self.assertRaises(CustomizationInputError):
                _normalize_voice(source, Path(directory) / "out.wav")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_reference_over_30_seconds_is_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "long.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x20\x03" * int(16000 * 30.2))
            with self.assertRaisesRegex(CustomizationInputError, "30 秒"):
                _normalize_voice(source, Path(directory) / "out.wav")


if __name__ == "__main__":
    unittest.main()
