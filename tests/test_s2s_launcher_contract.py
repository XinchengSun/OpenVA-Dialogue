import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE_VERSION = "FLASHAV2AV_0.1.0"


class S2SLauncherContractTests(unittest.TestCase):
    def test_all_demo_entrypoints_expect_the_current_engine_version(self):
        sources = (
            ROOT / "scripts" / "run_demo.sh",
            ROOT / "scripts" / "connect_demo.ps1",
            ROOT / "scripts" / "check_demo_ready.py",
        )

        for path in sources:
            with self.subTest(path=path.name):
                self.assertIn(
                    EXPECTED_ENGINE_VERSION,
                    path.read_text(encoding="utf-8"),
                )

    def test_production_launchers_require_native_s2s_health(self):
        run_demo = (ROOT / "scripts" / "run_demo.sh").read_text(encoding="utf-8")
        connect_demo = (ROOT / "scripts" / "connect_demo.ps1").read_text(
            encoding="utf-8"
        )

        for source in (run_demo, connect_demo):
            self.assertIn("s2s_ready", source)
            self.assertIn("native_s2s", source)
            self.assertNotIn("llm_ready", source)
            self.assertNotIn("tts_ready", source)

    def test_production_preflight_keeps_dual_gpu_guard(self):
        run_demo = (ROOT / "scripts" / "run_demo.sh").read_text(encoding="utf-8")
        start_demo = (ROOT / "scripts" / "start_pipecat_mse.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("at least two NVIDIA GPUs are required", run_demo)
        self.assertIn(
            "CUDA_VISIBLE_DEVICES must expose exactly two different GPUs",
            start_demo,
        )

    def test_workers_preserve_launcher_cuda_mapping(self):
        worker_source = (
            ROOT / "dual_gpu_mic_realtime_ui_v8_playbuffer.py"
        ).read_text(encoding="utf-8")
        start_demo = (ROOT / "scripts" / "start_pipecat_mse.sh").read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.motion_gpu)',
            worker_source,
        )
        self.assertNotIn(
            'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.render_gpu)',
            worker_source,
        )
        self.assertIn("torch.cuda.set_device(args.motion_gpu)", worker_source)
        self.assertIn("torch.cuda.set_device(args.render_gpu)", worker_source)
        self.assertIn('CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"', start_demo)

    def test_fish_launcher_preserves_model_and_reference_contract(self):
        launcher = (
            ROOT / "scripts" / "run_fish_s2pro_dual_gpu.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--model-name fishaudio/s2-pro", launcher)
        self.assertIn('--allowed-local-media-path "$reference_dir"', launcher)
        self.assertIn('CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"', launcher)

    def test_primary_env_contract_uses_qwen_audio_s2s(self):
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn("PIPECAT_S2S_API_KEY=", env_example)
        self.assertIn(
            "PIPECAT_S2S_MODEL=qwen-audio-3.0-realtime-flash",
            env_example,
        )
        self.assertIn("PIPECAT_S2S_VOICE=longanqian", env_example)
        self.assertIn("PIPECAT_S2S_TURN_DETECTION=smart_turn", env_example)
        self.assertIn("PIPECAT_S2S_VAD_SILENCE_MS=500", env_example)
        self.assertIn("PIPECAT_S2S_VAD_THRESHOLD=0.5", env_example)
        self.assertIn("ENGINE_TTS_STREAM_RESET=0", env_example)
        self.assertIn("ENGINE_LISTENER_AUDIO=1", env_example)
        self.assertIn("ENGINE_LISTENER_VIRTUAL_AUDIO=", env_example)
        self.assertIn("ENGINE_INTERRUPT_GRACE_SEC=0.40", env_example)
        self.assertIn("ENGINE_INTERRUPT_BRIDGE_SEC=0.20", env_example)
        self.assertIn("PIPECAT_S2S_WORKSPACE_ID=", env_example)
        self.assertIn("PIPECAT_S2S_BASE_URL=", env_example)
        self.assertNotIn(
            "PIPECAT_S2S_BASE_URL=wss://dashscope.aliyuncs.com",
            env_example,
        )

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_shell_launchers_are_syntax_valid(self):
        for script in (
            ROOT / "scripts" / "run_demo.sh",
            ROOT / "scripts" / "start_pipecat_mse.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(script)],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
