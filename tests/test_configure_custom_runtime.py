import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "configure_custom_runtime.py"


def _env_value(text: str, key: str) -> str:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            values = shlex.split(line[len(prefix) :], posix=True)
            return values[0] if values else ""
    raise AssertionError(f"missing env key: {key}")


class ConfigureCustomRuntimeTests(unittest.TestCase):
    def _run(self, root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        source_env = root / "source.env"
        prompt_wav = root / "reference.wav"
        source_env.write_text(
            "PIPECAT_LLM_API_KEY=secret-test-key\n"
            "PIPECAT_LLM_BASE_URL=https://example.invalid/v1\n"
            "CUDA_VISIBLE_DEVICES=0,1\n",
            encoding="utf-8",
        )
        prompt_wav.write_bytes(b"RIFF-test-fixture")
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(ROOT),
                "--runtime-root",
                str(root / "runtime"),
                "--source-env",
                str(source_env),
                "--prompt-wav",
                str(prompt_wav),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_fish_is_default_and_generates_two_private_runtime_envs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "reference.txt"
            transcript.write_text("这是与参考声音逐字一致的文本。", encoding="utf-8")
            result = self._run(
                root,
                "--prompt-text-file",
                str(transcript),
                "--dystream-gpus",
                "0,1",
                "--fish-gpus",
                "2,3",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("secret-test-key", result.stdout)

            config = root / "runtime" / "config"
            main_env = (config / "custom_cascade.env").read_text(encoding="utf-8")
            bridge_env = (config / "fish_speech_bridge.env").read_text(
                encoding="utf-8"
            )
            upstream_env = (config / "fish_s2pro.env").read_text(encoding="utf-8")
            self.assertIn("PIPECAT_TTS_PROVIDER=fish_s2pro", main_env)
            self.assertIn("PIPECAT_TTS_BACKEND=fish_s2_pro", main_env)
            self.assertIn("PIPECAT_TTS_REFERENCE_LANGUAGE=auto", main_env)
            self.assertIn("PIPECAT_TTS_TARGET_LANGUAGE=zh-CN", main_env)
            self.assertIn("PIPECAT_TTS_MODEL=fishaudio/s2-pro", main_env)
            self.assertIn("PIPECAT_TTS_BRIDGE_URI=ws://127.0.0.1:8771", main_env)
            self.assertIn("OPENAI_SPEECH_MODEL=fishaudio/s2-pro", bridge_env)
            self.assertEqual(
                _env_value(bridge_env, "OPENAI_SPEECH_PYTHON"),
                str(root / "runtime" / "venvs" / "pipecat" / "bin" / "python"),
            )
            self.assertIn("OPENAI_SPEECH_REFERENCE_TEXT=", bridge_env)
            self.assertIn("FISH_CUDA_VISIBLE_DEVICES=2,3", upstream_env)
            if os.name != "nt":
                for path in config.iterdir():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_fish_requires_exact_reference_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory), "--fish-gpus", "2,3")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --prompt-text-file", result.stderr)

    def test_fish_gpus_must_not_overlap_dystream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "reference.txt"
            transcript.write_text("测试。", encoding="utf-8")
            result = self._run(
                root,
                "--prompt-text-file",
                str(transcript),
                "--dystream-gpus",
                "0,1",
                "--fish-gpus",
                "1,2",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overlap", result.stderr)

    def test_custom_pipecat_python_is_written_to_fish_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "reference.txt"
            transcript.write_text("测试。", encoding="utf-8")
            custom_python = root / "custom-venv" / "bin" / "python"
            result = self._run(
                root,
                "--prompt-text-file",
                str(transcript),
                "--fish-gpus",
                "2,3",
                "--pipecat-python",
                str(custom_python),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            bridge_env = (
                root / "runtime" / "config" / "fish_speech_bridge.env"
            ).read_text(encoding="utf-8")
            self.assertEqual(
                _env_value(bridge_env, "OPENAI_SPEECH_PYTHON"),
                str(custom_python),
            )

    def test_fish_http_and_bridge_ports_must_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "reference.txt"
            transcript.write_text("测试。", encoding="utf-8")
            result = self._run(
                root,
                "--prompt-text-file",
                str(transcript),
                "--fish-gpus",
                "2,3",
                "--fish-http-port",
                "8771",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be different", result.stderr)

    def test_bridge_uri_must_match_the_ipv4_loopback_bind_host(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "reference.txt"
            transcript.write_text("测试。", encoding="utf-8")
            result = self._run(
                root,
                "--prompt-text-file",
                str(transcript),
                "--fish-gpus",
                "2,3",
                "--bridge-uri",
                "ws://localhost:8771",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ws://127.0.0.1:PORT", result.stderr)

    def test_voxcpm2_remains_an_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._run(
                root,
                "--tts-backend",
                "voxcpm2",
                "--dystream-gpus",
                "0,1",
                "--tts-gpu",
                "2",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = root / "runtime" / "config"
            main_env = (config / "custom_cascade.env").read_text(encoding="utf-8")
            self.assertIn("PIPECAT_TTS_PROVIDER=voxcpm2", main_env)
            self.assertIn("PIPECAT_TTS_BACKEND=voxcpm2", main_env)
            self.assertIn("PIPECAT_TTS_MODEL=VoxCPM2", main_env)
            self.assertTrue((config / "voxcpm2.env").is_file())
            self.assertFalse((config / "fish_s2pro.env").exists())


if __name__ == "__main__":
    unittest.main()
