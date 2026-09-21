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
    def _run(self, root: Path, *extra: str, source_extra: str = "") -> subprocess.CompletedProcess[str]:
        source_env = root / "source.env"
        prompt_wav = root / "reference.wav"
        source_env.write_text(
            "PIPECAT_LLM_API_KEY=secret-test-key\n"
            "PIPECAT_LLM_BASE_URL=https://example.invalid/v1\n"
            "CUDA_VISIBLE_DEVICES=0,1\n" + source_extra,
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
            self.assertEqual(_env_value(main_env, "CUDA_VISIBLE_DEVICES"), "0,1")
            self.assertEqual(_env_value(main_env, "RENDER_GPU"), "1")
            self.assertEqual(_env_value(main_env, "ALLOW_SHARED_DYSTREAM_GPU"), "0")
            self.assertEqual(_env_value(main_env, "DYSTREAM_SINGLE_GPU"), "")
            self.assertNotIn("DYSTREAM_FOLD_EMA=", main_env)
            self.assertNotIn("DYSTREAM_PRUNE_CFG=", main_env)
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

    def test_single_gpu_maps_physical_six_to_logical_zero_for_every_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._run(root, "--single-gpu", "6")
            self.assertEqual(result.returncode, 0, result.stderr)
            config = root / "runtime" / "config"
            main_env = (config / "custom_cascade.env").read_text(encoding="utf-8")
            bridge_env = (config / "voxcpm2.env").read_text(encoding="utf-8")
            for key, value in {
                "DYSTREAM_SINGLE_GPU": "6",
                "CUDA_VISIBLE_DEVICES": "6",
                "DYSTREAM_AUDIO_HISTORY_KEEP_SEC": "4.0",
                "DYSTREAM_AUDIO_HISTORY_MAX_SEC": "8.0",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "MOTION_GPU": "0",
                "RENDER_GPU": "0",
                "ALLOW_SHARED_DYSTREAM_GPU": "1",
                "PIPECAT_TTS_PROVIDER": "voxcpm2",
                "PIPECAT_TTS_LIFECYCLE": "managed",
                "PIPECAT_ASR_DEVICE": "cpu",
                "DYSTREAM_FOLD_EMA": "1",
                "DYSTREAM_PRUNE_CFG": "1",
            }.items():
                self.assertEqual(_env_value(main_env, key), value, key)
            self.assertEqual(_env_value(bridge_env, "CUDA_VISIBLE_DEVICES"), "6")
            self.assertEqual(_env_value(bridge_env, "CUDA_DEVICE_ORDER"), "PCI_BUS_ID")
            self.assertEqual(_env_value(bridge_env, "VOXCPM2_DEVICES"), "0")
            self.assertEqual(_env_value(bridge_env, "VOXCPM2_OFFICIAL_DEVICE"), "cuda:0")
            self.assertEqual(_env_value(bridge_env, "VOXCPM2_BACKEND"), "official_prompt_cache")
            self.assertNotIn("VOXCPM2_GPU_MEMORY_UTILIZATION", bridge_env)
            self.assertNotIn("VOXCPM2_MAX_NUM_SEQS", bridge_env)
            self.assertNotIn("secret-test-key", result.stdout)
            self.assertFalse((config / "fish_s2pro.env").exists())

    def test_single_gpu_rejects_invalid_ids_and_conflicting_options_before_writing(self):
        cases = [
            ("--single-gpu", "-1"),
            ("--single-gpu", "0,1"),
            ("--single-gpu", "cuda:0"),
            ("--single-gpu", "6", "--tts-backend", "fish_s2pro"),
            ("--single-gpu", "6", "--fish-gpus", "2,3"),
            ("--single-gpu", "6", "--dystream-gpus", "0,1"),
            ("--single-gpu", "6", "--tts-gpu", "2"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self._run(root, *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--single-gpu", result.stderr)
                self.assertFalse((root / "runtime").exists())

    def test_single_gpu_accepts_consistent_explicit_mapping_and_official_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "official-venv" / "bin" / "python"
            source = root / "VoxCPM" / "src"
            result = self._run(
                root, "--single-gpu", "6", "--dystream-gpus", "6",
                "--tts-gpu", "6", "--tts-backend", "voxcpm2",
                "--voxcpm-python", str(python), "--voxcpm-official-source", str(source),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            bridge_env = (root / "runtime" / "config" / "voxcpm2.env").read_text(encoding="utf-8")
            self.assertEqual(_env_value(bridge_env, "VOXCPM2_PYTHON"), str(python))
            self.assertEqual(_env_value(bridge_env, "VOXCPM2_OFFICIAL_SOURCE"), str(source))

    def test_source_duplicates_cannot_override_the_single_gpu_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._run(
                root, "--single-gpu", "6",
                source_extra="CUDA_VISIBLE_DEVICES=2,3\nRENDER_GPU=1\nRENDER_GPU=2\nPIPECAT_TTS_LIFECYCLE=external\nEXPECTED_ENGINE_VERSION=legacy-production-build\n",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            main_env = (root / "runtime" / "config" / "custom_cascade.env").read_text(encoding="utf-8")
            self.assertEqual(main_env.count("CUDA_VISIBLE_DEVICES="), 1)
            self.assertEqual(main_env.count("RENDER_GPU="), 1)
            self.assertEqual(_env_value(main_env, "CUDA_VISIBLE_DEVICES"), "6")
            self.assertEqual(_env_value(main_env, "RENDER_GPU"), "0")
            self.assertEqual(_env_value(main_env, "PIPECAT_TTS_LIFECYCLE"), "managed")
            self.assertEqual(_env_value(main_env, "EXPECTED_ENGINE_VERSION"), "")

    @unittest.skipIf(os.name == "nt", "Linux venv interpreter symlink regression")
    def test_interpreter_paths_preserve_virtualenv_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv_python = root / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(Path(sys.executable).resolve())
            self.assertNotEqual(venv_python.absolute(), venv_python.resolve())
            transcript = root / "reference.txt"
            transcript.write_text("Test.", encoding="utf-8")
            for arguments, filename, key in (
                (("--single-gpu", "6", "--voxcpm-python", str(venv_python)),
                 "voxcpm2.env", "VOXCPM2_PYTHON"),
                (("--pipecat-python", str(venv_python), "--prompt-text-file", str(transcript)),
                 "fish_speech_bridge.env", "OPENAI_SPEECH_PYTHON"),
            ):
                with self.subTest(key=key):
                    result = self._run(root, *arguments)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    generated = (root / "runtime" / "config" / filename).read_text(encoding="utf-8")
                    self.assertEqual(_env_value(generated, key), str(venv_python.absolute()))

    def test_multigpu_voxcpm2_still_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory), "--tts-backend", "voxcpm2", "--tts-gpu", "1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overlap", result.stderr)


if __name__ == "__main__":
    unittest.main()
