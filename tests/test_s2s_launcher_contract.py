import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE_VERSION = "FLASHAV2AV_0.1.0"


class S2SLauncherContractTests(unittest.TestCase):
    def test_fish_start_failure_keeps_unified_ownership_for_recovery(self):
        run_demo = (ROOT / "scripts" / "run_demo.sh").read_text(encoding="utf-8")
        start_marker = 'if ! bash "$FISH_MANAGER_SCRIPT" start "$TTS_UPSTREAM_ENV"; then'
        bridge_marker = 'if ! OPENAI_SPEECH_INSTANCE=flashav2av'
        failure_block = run_demo.split(start_marker, 1)[1].split(bridge_marker, 1)[0]
        self.assertIn("return 1", failure_block)
        self.assertNotIn('rm -f -- "$TTS_STATE_FILE"', failure_block)
        stop_block = run_demo.split("stop_owned_tts_stack_if_present()", 1)[1]
        self.assertIn('bash "$FISH_MANAGER_SCRIPT" stop "${state[2]}"', stop_block)

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

    def test_lifecycle_validates_native_and_custom_fish_health(self):
        run_demo = (ROOT / "scripts" / "run_demo.sh").read_text(encoding="utf-8")
        connect_demo = (ROOT / "scripts" / "connect_demo.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("s2s_ready", run_demo)
        self.assertIn("custom_cascade_ready", run_demo)
        self.assertIn("fishaudio/s2-pro", run_demo)
        self.assertIn("manage_fish_s2pro.sh", run_demo)
        self.assertIn("native_s2s", connect_demo)

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

    def test_env_contract_exposes_fish_as_primary_cascade_tts(self):
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
        self.assertIn("ENGINE_LISTENER_AUDIO=0", env_example)
        self.assertIn("ENGINE_LISTENER_VIRTUAL_AUDIO=", env_example)
        self.assertIn("ENGINE_INTERRUPT_GRACE_SEC=0.40", env_example)
        self.assertIn("ENGINE_INTERRUPT_BRIDGE_SEC=0.20", env_example)
        self.assertIn("PIPECAT_S2S_WORKSPACE_ID=", env_example)
        self.assertIn("PIPECAT_S2S_BASE_URL=", env_example)
        self.assertNotIn(
            "PIPECAT_S2S_BASE_URL=wss://dashscope.aliyuncs.com",
            env_example,
        )
        self.assertIn("PIPECAT_TTS_PROVIDER=fish_s2pro", env_example)
        self.assertIn("PIPECAT_TTS_BRIDGE_URI=ws://127.0.0.1:8771", env_example)
        self.assertIn("PIPECAT_TTS_MODEL=fishaudio/s2-pro", env_example)

    @unittest.skipIf(
        os.name == "nt",
        "Windows bash launchers do not preserve this workspace path; CI validates shell syntax",
    )
    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_shell_launchers_are_syntax_valid(self):
        for script in (
            ROOT / "scripts" / "run_demo.sh",
            ROOT / "scripts" / "start_pipecat_mse.sh",
            ROOT / "scripts" / "manage_fish_s2pro.sh",
        ):
            subprocess.run(
                ["bash", "-n"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                input=script.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
