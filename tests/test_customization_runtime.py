import importlib.util
import math
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from PIL import Image

from customization_runtime import (
    CustomizationInputError,
    _normalize_image,
    _normalize_voice,
    _patch_env_file,
)


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "activate_customization.py"
CONTROLLER_SPEC = importlib.util.spec_from_file_location("activate_customization", CONTROLLER_PATH)
CONTROLLER = importlib.util.module_from_spec(CONTROLLER_SPEC)
CONTROLLER_SPEC.loader.exec_module(CONTROLLER)
STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"


class CustomizationRuntimeTests(unittest.TestCase):
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
        self.assertIn("<title>实时数字人</title>", realtime)
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
        self.assertEqual(environment["ENV_FILE"], "/runtime/custom.env")
        self.assertEqual(token, "custom-aaaaaaaaaaaaaaaa")

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
