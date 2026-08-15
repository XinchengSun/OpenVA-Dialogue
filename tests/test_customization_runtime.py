import asyncio
import importlib.util
import inspect
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
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
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

import customization_runtime as customization

from customization_runtime import (
    CustomizationInputError,
    _normalize_image,
    _normalize_tts_selection,
    _normalize_voice,
    _patch_env_file,
    _prepare_job,
    _runtime_missing,
    _runtime_paths,
    _tts_options,
    register_customization_routes,
)


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "activate_customization.py"
CONTROLLER_SPEC = importlib.util.spec_from_file_location("activate_customization", CONTROLLER_PATH)
CONTROLLER = importlib.util.module_from_spec(CONTROLLER_SPEC)
CONTROLLER_SPEC.loader.exec_module(CONTROLLER)
STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"


class CustomizationRuntimeTests(unittest.TestCase):
    def test_backend_live_probe_is_cached_for_five_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "bridge.env"
            env_path.write_text("OPENAI_SPEECH_BASE_URL=http://127.0.0.1:8003\n")
            with customization._BACKEND_LIVE_CACHE_LOCK:
                customization._BACKEND_LIVE_CACHE.clear()
            with mock.patch.object(
                customization,
                "_probe_configured_backend_is_live",
                return_value=True,
            ) as probe:
                self.assertTrue(
                    customization._configured_backend_is_live(
                        "qwen3_tts_1_7b_base", env_path
                    )
                )
                self.assertTrue(
                    customization._configured_backend_is_live(
                        "qwen3_tts_1_7b_base", env_path
                    )
                )
            probe.assert_called_once_with("qwen3_tts_1_7b_base", env_path)

    def test_verify_health_accepts_ready_marker_provider_suffix(self):
        health = {
            "status": "ok",
            "frame_ready": True,
            "launch_ready": True,
            "launch_token": "custom-abc qwen3_tts_1_7b_base",
            "dialog_session": {"custom_cascade_ready": True},
        }
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(health).encode()
        with mock.patch.object(
            CONTROLLER.urllib.request, "urlopen", return_value=response
        ):
            CONTROLLER.verify_health(7862, "custom-abc")

    def test_verify_health_rejects_different_ready_marker_token(self):
        health = {
            "status": "ok",
            "frame_ready": True,
            "launch_ready": True,
            "launch_token": "custom-other qwen3_tts_1_7b_base",
            "dialog_session": {"custom_cascade_ready": True},
        }
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(health).encode()
        with mock.patch.object(
            CONTROLLER.urllib.request, "urlopen", return_value=response
        ), mock.patch.object(
            CONTROLLER.time, "monotonic", side_effect=[0.0, 0.0, 16.0]
        ), mock.patch.object(CONTROLLER.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not become healthy"):
                CONTROLLER.verify_health(7862, "custom-abc")

    def test_product_pages_use_current_names_and_switch_flow(self):
        customize = (STATIC_ROOT / "customize.html").read_text(encoding="utf-8")
        realtime = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn("生成素材", customize)
        self.assertIn("确认文本并启用", customize)
        self.assertIn("TTS 后端", customize)
        self.assertIn("参考声音语言", customize)
        self.assertIn("输出语言", customize)
        self.assertIn("更换数字人", customize)
        self.assertIn("新人物启用时会自动替换旧模型", customize)
        self.assertIn("立即进入对话", customize)
        self.assertIn("立即检查状态", customize)
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

    def test_success_status_clears_previous_failure_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "status.json"
            CONTROLLER.write_json(status_path, {
                "state": "failed",
                "error_code": "controller_failed",
                "rollback_succeeded": False,
            })
            status = CONTROLLER.update_status(
                status_path,
                clear_errors=True,
                state="ready",
                message="ready",
            )
            self.assertEqual(status["state"], "ready")
            self.assertNotIn("error_code", status)
            self.assertNotIn("rollback_succeeded", status)
            persisted = CONTROLLER.read_json(status_path)
            self.assertNotIn("error_code", persisted)
            self.assertNotIn("rollback_succeeded", persisted)

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
        self.assertEqual(Path(environment["ENV_FILE"]), Path("/runtime/custom.env"))
        self.assertEqual(token, "custom-aaaaaaaaaaaaaaaa")

    def test_runtime_paths_are_derived_from_main_env_not_legacy_vox_env(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate"
            config = candidate / "config"
            config.mkdir(parents=True)
            main_env = config / "custom_cascade.env"
            main_env.write_text(
                "VOXCPM2_BRIDGE_URI=ws://127.0.0.1:8773\n", encoding="utf-8"
            )
            tts_env = config / "fish_bridge_8773.env"
            tts_env.write_text("OPENAI_SPEECH_BRIDGE_PORT=8773\n", encoding="utf-8")
            with mock.patch.dict(
                "os.environ",
                {
                    "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                    "CUSTOMIZATION_TTS_BRIDGE_INSTANCE": "flashav2av",
                    "VOXCPM2_ENV_FILE": "/legacy/voxcpm2.env",
                },
                clear=False,
            ):
                paths = _runtime_paths()
            self.assertEqual(paths["runtime_root"], candidate / "customizations")
            self.assertEqual(paths["tts_env"], tts_env)
            self.assertEqual(paths["bridge_instance"], "flashav2av")

    def test_runtime_paths_preserve_explicit_legacy_voxcpm2_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate"
            config = candidate / "config"
            config.mkdir(parents=True)
            main_env = config / "custom_cascade.env"
            vox_env = config / "voxcpm2.env"
            main_env.write_text("PIPECAT_TTS_PROVIDER=voxcpm2\n", encoding="utf-8")
            vox_env.write_text("VOXCPM2_BRIDGE_PORT=8770\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "PIPECAT_TTS_PROVIDER": "voxcpm2",
                    "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                    "VOXCPM2_ENV_FILE": str(vox_env),
                },
                clear=True,
            ), mock.patch.object(
                customization, "_configured_backend_is_live", return_value=True
            ):
                paths = _runtime_paths()
                options = _tts_options(paths)
            self.assertEqual(paths["tts_provider"], "voxcpm2")
            self.assertEqual(paths["tts_env"], vox_env.resolve())
            selected = {
                item["id"]: item for item in options["backends"]
            }
            self.assertTrue(selected["voxcpm2"]["selectable"])
            self.assertFalse(selected["fish_s2_pro"]["selectable"])
            self.assertEqual(options["defaults"]["tts_backend"], "voxcpm2")

    def test_configured_sidecar_is_selectable_only_after_explicit_ready_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            main_env = config / "custom_cascade.env"
            fish_env = config / "fish.env"
            qwen_env = config / "qwen.env"
            main_env.write_text("READY=1\n", encoding="utf-8")
            fish_env.write_text(
                "OPENAI_SPEECH_BRIDGE_PORT=8773\n", encoding="utf-8"
            )
            qwen_env.write_text(
                "OPENAI_SPEECH_BRIDGE_PORT=8774\n", encoding="utf-8"
            )
            environment = {
                "PIPECAT_TTS_BACKEND": "fish_s2_pro",
                "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                "CUSTOMIZATION_FISH_S2_PRO_ENV_FILE": str(fish_env),
                "CUSTOMIZATION_QWEN3_TTS_1_7B_ENV_FILE": str(qwen_env),
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                customization, "_configured_backend_is_live", return_value=True
            ):
                options = {
                    item["id"]: item for item in _tts_options(_runtime_paths())["backends"]
                }
            self.assertTrue(options["fish_s2_pro"]["selectable"])
            self.assertFalse(options["qwen3_tts_1_7b_base"]["selectable"])
            self.assertEqual(
                options["qwen3_tts_1_7b_base"]["disabled_reason"],
                "installed_not_started",
            )

            environment["CUSTOMIZATION_TTS_READY_BACKENDS"] = (
                "qwen3_tts_1_7b_base"
            )
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                customization, "_configured_backend_is_live", return_value=True
            ):
                options = {
                    item["id"]: item for item in _tts_options(_runtime_paths())["backends"]
                }
            self.assertTrue(options["qwen3_tts_1_7b_base"]["selectable"])
            self.assertTrue(options["qwen3_tts_1_7b_base"]["configured"])

    def test_current_backend_is_not_selectable_when_live_health_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            main_env = config / "custom_cascade.env"
            fish_env = config / "fish.env"
            main_env.write_text("READY=1\n", encoding="utf-8")
            fish_env.write_text(
                "OPENAI_SPEECH_BASE_URL=http://127.0.0.1:8002\n"
                "OPENAI_SPEECH_BRIDGE_PORT=8773\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "PIPECAT_TTS_BACKEND": "fish_s2_pro",
                    "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                    "CUSTOMIZATION_FISH_S2_PRO_ENV_FILE": str(fish_env),
                },
                clear=True,
            ), mock.patch.object(
                customization, "_configured_backend_is_live", return_value=False
            ):
                options = {
                    item["id"]: item
                    for item in _tts_options(_runtime_paths())["backends"]
                }
            self.assertFalse(options["fish_s2_pro"]["selectable"])
            self.assertEqual(
                options["fish_s2_pro"]["disabled_reason"], "service_not_ready"
            )

    def test_candidate_with_distinct_instance_but_duplicate_bridge_port_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            main_env = config / "custom_cascade.env"
            fish_env = config / "fish.env"
            qwen_env = config / "qwen.env"
            main_env.write_text("READY=1\n", encoding="utf-8")
            for path in (fish_env, qwen_env):
                path.write_text(
                    "OPENAI_SPEECH_BASE_URL=http://127.0.0.1:8002\n"
                    "OPENAI_SPEECH_BRIDGE_PORT=8773\n",
                    encoding="utf-8",
                )
            with mock.patch.dict(
                os.environ,
                {
                    "PIPECAT_TTS_BACKEND": "fish_s2_pro",
                    "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                    "CUSTOMIZATION_FISH_S2_PRO_ENV_FILE": str(fish_env),
                    "CUSTOMIZATION_QWEN3_TTS_1_7B_ENV_FILE": str(qwen_env),
                    "CUSTOMIZATION_FISH_S2_PRO_BRIDGE_INSTANCE": "fish",
                    "CUSTOMIZATION_QWEN3_TTS_1_7B_BRIDGE_INSTANCE": "qwen3",
                    "CUSTOMIZATION_TTS_READY_BACKENDS": "qwen3_tts_1_7b_base",
                },
                clear=True,
            ), mock.patch.object(
                customization, "_configured_backend_is_live", return_value=True
            ):
                options = {
                    item["id"]: item
                    for item in _tts_options(_runtime_paths())["backends"]
                }
            self.assertFalse(options["qwen3_tts_1_7b_base"]["selectable"])
            self.assertEqual(
                options["qwen3_tts_1_7b_base"]["disabled_reason"],
                "bridge_instance_conflict",
            )

    def test_new_fish_backend_keeps_legacy_provider_api_value(self):
        with mock.patch.dict(
            os.environ, {"PIPECAT_TTS_BACKEND": "fish_s2_pro"}, clear=True
        ):
            paths = _runtime_paths()
        self.assertEqual(paths["tts_provider"], "fish_s2_pro")
        self.assertEqual(paths["legacy_tts_provider"], "fish_s2pro")

    def test_controller_cli_accepts_legacy_provider_arguments(self):
        argv = [
            "activate_customization.py",
            "--job-dir", "/runtime/customizations/" + "a" * 32,
            "--runtime-root", "/runtime/customizations",
            "--repo-root", "/srv/dystream",
            "--main-env", "/runtime/config/custom.env",
            "--vox-env", "/runtime/config/voxcpm2.env",
            "--tts-provider", "voxcpm2",
            "--fish-env", "/runtime/config/fish.env",
            "--port", "7860",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = CONTROLLER.parse_args()
        self.assertEqual(args.tts_provider, "voxcpm2")
        self.assertEqual(args.tts_env, Path("/runtime/config/voxcpm2.env"))
        self.assertEqual(args.fish_env, Path("/runtime/config/fish.env"))

    def test_controller_dispatches_legacy_voxcpm2_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            runtime_root = root / "customizations"
            job_id = "a" * 32
            job_dir = runtime_root / job_id
            assets = job_dir / "assets"
            assets.mkdir(parents=True)
            image = assets / "reference.png"
            voice = assets / "voice_reference.wav"
            image.write_bytes(b"image")
            voice.write_bytes(b"voice")
            main_env = root / "custom_cascade.env"
            vox_env = root / "voxcpm2.env"
            main_env.write_text("DYSTREAM_REF_IMAGE=/old.png\n", encoding="utf-8")
            vox_env.write_text("VOXCPM2_PROMPT_WAV=/old.wav\n", encoding="utf-8")
            CONTROLLER.write_json(job_dir / "manifest.json", {
                "job_id": job_id,
                "image_path": str(image),
                "voice_path": str(voice),
                "transcript_path": "",
                "tts_provider": "voxcpm2",
                "reference_language": "auto",
                "target_language": "zh-CN",
                "voice": {"source_type": "audio"},
            })
            CONTROLLER.write_json(
                job_dir / "status.json", {"job_id": job_id, "state": "prepared"}
            )
            args = SimpleNamespace(
                job_dir=job_dir,
                runtime_root=runtime_root,
                repo_root=repo,
                main_env=main_env,
                tts_env=vox_env,
                tts_provider=None,
                fish_env=None,
                bridge_instance="flashav2av",
                port=7860,
            )
            metrics = CONTROLLER.PCMProbeMetrics(-30.0, -10.0, 0, 1000)
            with mock.patch.object(
                CONTROLLER, "restart_voxcpm2_bridge"
            ) as restart_bridge, mock.patch.object(
                CONTROLLER, "probe_restarted_voxcpm2_bridge", return_value=metrics
            ), mock.patch.object(
                CONTROLLER, "run_demo_restart", return_value="custom-token"
            ) as restart, mock.patch.object(
                CONTROLLER, "verify_health"
            ), mock.patch.object(
                CONTROLLER, "verify_loaded_voxcpm2_assets"
            ), mock.patch.object(CONTROLLER.time, "sleep"):
                result = CONTROLLER.activate(args)
            self.assertEqual(result, 0)
            restart_bridge.assert_called_once_with(repo.resolve(), vox_env.resolve())
            restart.assert_called_once_with(repo.resolve(), main_env.resolve(), 7860, job_id)
            self.assertEqual(
                CONTROLLER.read_env_value(main_env, "PIPECAT_TTS_BACKEND"),
                "voxcpm2",
            )
            self.assertEqual(
                Path(CONTROLLER.read_env_value(vox_env, "VOXCPM2_PROMPT_WAV")),
                voice.resolve(),
            )
            active = CONTROLLER.read_json(runtime_root / "active.json")
            self.assertEqual(active["tts_backend"], "voxcpm2")
            self.assertEqual(active["job_id"], job_id)

    def test_controller_uses_fish_bridge_and_mse_only(self):
        repo = Path("/srv/dystream")
        bridge_env = Path("/runtime/fish_bridge_8773.env")
        with mock.patch.dict(
            CONTROLLER.os.environ,
            {"PYTHONSAFEPATH": "1", "PYTHONPATH": "/unrelated"},
            clear=False,
        ), mock.patch.object(CONTROLLER.subprocess, "run") as run:
            CONTROLLER.restart_fish_bridge(repo, bridge_env, "realtime")
            bridge_environment = run.call_args.kwargs["env"]
            CONTROLLER.run_demo_restart(repo, Path("/runtime/main.env"), 7862, "a" * 32)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertNotIn("PYTHONSAFEPATH", bridge_environment)
        self.assertEqual(
            bridge_environment["PYTHONPATH"].split(CONTROLLER.os.pathsep)[0],
            str(repo),
        )
        self.assertEqual(commands[0][-2:], ["restart", str(bridge_env)])
        self.assertEqual(commands[1][-1], "restart-mse")
        flattened = " ".join(item for command in commands for item in command)
        self.assertNotIn("voxcpm2", flattened.lower())
        self.assertNotIn("8002", flattened)
        self.assertNotRegex(flattened, r"run_demo\.sh restart(?:\s|$)")

    def test_voxcpm2_bridge_restart_invokes_its_manager(self):
        repo = Path("/srv/dystream")
        bridge_env = Path("/runtime/voxcpm2.env")
        with mock.patch.object(CONTROLLER.subprocess, "run") as run:
            CONTROLLER.restart_voxcpm2_bridge(repo, bridge_env)
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ["restart", str(bridge_env)])
        self.assertEqual(Path(command[1]).name, "run_bridge.sh")
        self.assertEqual(run.call_args.kwargs["cwd"], str(repo))

    def test_backend_switch_persists_previous_fish_and_external_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_env = root / "fish.env"
            selected_env = root / "qwen.env"
            args = SimpleNamespace(
                previous_tts_provider="fish_s2_pro",
                previous_tts_env=previous_env,
                previous_bridge_instance="fish",
                bridge_instance="qwen3",
            )
            with mock.patch.dict(
                CONTROLLER.os.environ,
                {"CUSTOMIZATION_TTS_READY_BACKENDS": "cosyvoice3_0_5b"},
                clear=False,
            ):
                updates = CONTROLLER.backend_selection_env_updates(
                    args,
                    "qwen3_tts_1_7b_base",
                    selected_env,
                )
            self.assertEqual(updates["PIPECAT_TTS_LIFECYCLE"], "external")
            self.assertEqual(
                set(updates["CUSTOMIZATION_TTS_READY_BACKENDS"].split(",")),
                {
                    "cosyvoice3_0_5b",
                    "fish_s2_pro",
                    "qwen3_tts_1_7b_base",
                },
            )
            self.assertEqual(
                updates["PIPECAT_TTS_PROVIDER"], "qwen3_tts_1_7b_base"
            )
            self.assertEqual(
                updates["CUSTOMIZATION_FISH_S2_PRO_ENV_FILE"],
                str(previous_env.resolve()),
            )
            self.assertEqual(
                updates["CUSTOMIZATION_FISH_S2_PRO_BRIDGE_INSTANCE"], "fish"
            )
            self.assertEqual(
                updates["CUSTOMIZATION_QWEN3_TTS_1_7B_ENV_FILE"],
                str(selected_env),
            )

    def test_controller_rejects_cross_backend_instance_or_port_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fish_env = root / "fish.env"
            qwen_env = root / "qwen.env"
            fish_env.write_text(
                "OPENAI_SPEECH_BRIDGE_PORT=8773\n", encoding="utf-8"
            )
            qwen_env.write_text(
                "OPENAI_SPEECH_BRIDGE_PORT=8774\n", encoding="utf-8"
            )
            with mock.patch.object(CONTROLLER, "restart_backend_bridge") as restart:
                with self.assertRaisesRegex(RuntimeError, "instance collides"):
                    CONTROLLER.assert_backend_switch_is_isolated(
                        "qwen3_tts_1_7b_base",
                        qwen_env,
                        "shared",
                        "fish_s2_pro",
                        fish_env,
                        "shared",
                    )
                restart.assert_not_called()

            qwen_env.write_text(
                "OPENAI_SPEECH_BRIDGE_PORT=8773\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "port collides"):
                CONTROLLER.assert_backend_switch_is_isolated(
                    "qwen3_tts_1_7b_base",
                    qwen_env,
                    "qwen3",
                    "fish_s2_pro",
                    fish_env,
                    "fish",
                )

    def test_same_backend_rollback_restores_bridge_before_mse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            main_env = root / "main.env"
            main_backup = root / "main.backup.env"
            bridge_env = root / "fish.env"
            bridge_backup = root / "fish.backup.env"
            active_path = root / "active.json"
            active_backup = root / "active.backup.json"
            previous_image = root / "previous.png"
            previous_voice = root / "previous.wav"
            main_env.write_text("BROKEN=1\n", encoding="utf-8")
            main_backup.write_text(
                f"DYSTREAM_REF_IMAGE={previous_image}\n"
                "PIPECAT_TTS_PROVIDER=fish_s2pro\n",
                encoding="utf-8",
            )
            bridge_env.write_text("NEW=1\n", encoding="utf-8")
            bridge_backup.write_text(
                f"OPENAI_SPEECH_REFERENCE_AUDIO={previous_voice}\n"
                "OPENAI_SPEECH_BRIDGE_PORT=8773\n",
                encoding="utf-8",
            )
            active_path.write_text('{"job_id":"new"}\n', encoding="utf-8")
            active_backup.write_text('{"job_id":"old"}\n', encoding="utf-8")
            args = SimpleNamespace(
                previous_tts_provider="fish_s2_pro",
                previous_tts_env=bridge_env,
                previous_bridge_instance="fish",
                bridge_instance="fish",
                port=7862,
            )
            events = []

            def restart_bridge(*_args):
                events.append("restart-old-bridge")

            def probe_bridge(*_args):
                events.append("probe-old-bridge")
                return CONTROLLER.PCMProbeMetrics(-30.0, -10.0, 0, 1000)

            def restart_mse(*_args):
                events.append("restart-old-mse")
                return "rollback-token"

            with mock.patch.object(
                CONTROLLER, "restart_backend_bridge", side_effect=restart_bridge
            ), mock.patch.object(
                CONTROLLER, "probe_backend_bridge", side_effect=probe_bridge
            ), mock.patch.object(
                CONTROLLER, "run_demo_restart", side_effect=restart_mse
            ), mock.patch.object(
                CONTROLLER, "verify_health"
            ), mock.patch.object(
                CONTROLLER, "verify_loaded_assets"
            ):
                CONTROLLER.rollback_backend_switch(
                    args=args,
                    repo_root=repo,
                    main_env=main_env,
                    main_backup=main_backup,
                    selected_backend="fish_s2_pro",
                    selected_bridge_env=bridge_env,
                    selected_bridge_backup=bridge_backup,
                    active_path=active_path,
                    active_backup=active_backup,
                    active_existed=True,
                    job_id="b" * 32,
                )
            self.assertEqual(
                events,
                ["restart-old-bridge", "probe-old-bridge", "restart-old-mse", "probe-old-bridge"],
            )

    def test_cross_backend_rollback_restores_old_mse_before_candidate_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            main_env = root / "main.env"
            main_backup = root / "main.backup.env"
            previous_env = root / "fish.env"
            candidate_env = root / "qwen.env"
            candidate_backup = root / "qwen.backup.env"
            active_path = root / "active.json"
            active_backup = root / "active.backup.json"
            previous_image = root / "previous.png"
            previous_voice = root / "previous.wav"
            main_env.write_text("BROKEN=1\n", encoding="utf-8")
            main_backup.write_text(
                f"DYSTREAM_REF_IMAGE={previous_image}\n"
                "PIPECAT_TTS_BACKEND=fish_s2_pro\n",
                encoding="utf-8",
            )
            previous_env.write_text(
                f"OPENAI_SPEECH_REFERENCE_AUDIO={previous_voice}\n"
                "OPENAI_SPEECH_BRIDGE_PORT=8773\n",
                encoding="utf-8",
            )
            candidate_env.write_text("NEW=1\n", encoding="utf-8")
            candidate_backup.write_text("OLD=1\n", encoding="utf-8")
            active_path.write_text('{"job_id":"new"}\n', encoding="utf-8")
            active_backup.write_text('{"job_id":"old"}\n', encoding="utf-8")
            args = SimpleNamespace(
                previous_tts_provider="fish_s2_pro",
                previous_tts_env=previous_env,
                previous_bridge_instance="fish",
                bridge_instance="qwen3",
                port=7862,
            )
            metrics = CONTROLLER.PCMProbeMetrics(-30.0, -10.0, 0, 1000)
            events = []

            def restart_mse(*_args):
                events.append("restart-old-mse")
                return "rollback-token"

            def fail_candidate_cleanup(*_args):
                events.append("cleanup-candidate")
                raise RuntimeError("candidate remains unavailable")

            with mock.patch.object(
                CONTROLLER, "run_demo_restart", side_effect=restart_mse
            ), mock.patch.object(
                CONTROLLER, "verify_health"
            ), mock.patch.object(
                CONTROLLER, "probe_backend_bridge", return_value=metrics
            ), mock.patch.object(
                CONTROLLER, "verify_loaded_assets"
            ), mock.patch.object(
                CONTROLLER,
                "restart_backend_bridge",
                side_effect=fail_candidate_cleanup,
            ):
                CONTROLLER.rollback_backend_switch(
                    args=args,
                    repo_root=repo,
                    main_env=main_env,
                    main_backup=main_backup,
                    selected_backend="qwen3_tts_1_7b_base",
                    selected_bridge_env=candidate_env,
                    selected_bridge_backup=candidate_backup,
                    active_path=active_path,
                    active_backup=active_backup,
                    active_existed=True,
                    job_id="a" * 32,
                )
            self.assertEqual(events, ["restart-old-mse", "cleanup-candidate"])
            self.assertEqual(main_env.read_bytes(), main_backup.read_bytes())
            self.assertEqual(candidate_env.read_bytes(), candidate_backup.read_bytes())
            self.assertEqual(active_path.read_bytes(), active_backup.read_bytes())

    def test_fish_reference_is_copied_next_to_current_allowlisted_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references"
            references.mkdir()
            current = references / "ref.wav"
            current.write_bytes(b"old")
            bridge_env = root / "fish.env"
            bridge_env.write_text(
                f"OPENAI_SPEECH_REFERENCE_AUDIO={current}\n"
                "OPENAI_SPEECH_REFERENCE_TEXT=old\n",
                encoding="utf-8",
            )
            upload = root / "upload.wav"
            upload.write_bytes(b"new voice")
            installed = CONTROLLER.install_fish_reference(
                bridge_env, upload, "new transcript", "b" * 32
            )
            self.assertEqual(installed.parent, references)
            self.assertEqual(installed.read_bytes(), b"new voice")
            self.assertEqual(
                CONTROLLER.read_env_value(
                    bridge_env, "OPENAI_SPEECH_REFERENCE_AUDIO"
                ),
                str(installed),
            )
            self.assertEqual(
                CONTROLLER.read_env_value(
                    bridge_env, "OPENAI_SPEECH_REFERENCE_TEXT"
                ),
                "new transcript",
            )

    def test_fish_reference_can_be_staged_without_patching_live_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references"
            references.mkdir()
            current = references / "ref.wav"
            current.write_bytes(b"old")
            bridge_env = root / "fish.env"
            bridge_env.write_text(
                f"OPENAI_SPEECH_REFERENCE_AUDIO={current}\n"
                "OPENAI_SPEECH_REFERENCE_TEXT=old\n",
                encoding="utf-8",
            )
            upload = root / "upload.wav"
            upload.write_bytes(b"new voice")

            staged = CONTROLLER.stage_fish_reference(
                bridge_env, upload, "c" * 32
            )

            self.assertEqual(staged.read_bytes(), b"new voice")
            self.assertEqual(
                CONTROLLER.read_env_value(
                    bridge_env, "OPENAI_SPEECH_REFERENCE_AUDIO"
                ),
                str(current),
            )

    def test_qwen_sidecar_env_keeps_its_backend_and_dedicated_bridge_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "ref.wav"
            reference.write_bytes(b"voice")
            bridge_env = root / "qwen.env"
            bridge_env.write_text(
                "OPENAI_SPEECH_BRIDGE_HOST=127.0.0.1\n"
                "OPENAI_SPEECH_BRIDGE_PORT=8774\n",
                encoding="utf-8",
            )
            CONTROLLER.apply_fish_reference_env(
                bridge_env,
                reference,
                "exact transcript",
                "en-US",
                "zh-CN",
                "qwen3_tts_1_7b_base",
            )
            self.assertEqual(
                CONTROLLER.read_env_value(bridge_env, "OPENAI_SPEECH_BACKEND"),
                "qwen3_tts_1_7b_base",
            )
            self.assertEqual(CONTROLLER.openai_bridge_uri(bridge_env), "ws://127.0.0.1:8774")

    def test_qwen_activation_switches_main_env_to_dedicated_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            runtime_root = root / "customizations"
            job_id = "9" * 32
            job = runtime_root / job_id
            assets = job / "assets"
            assets.mkdir(parents=True)
            image = assets / "reference.png"
            voice = assets / "voice_reference.wav"
            transcript = assets / "voice_reference.txt"
            image.write_bytes(b"image")
            voice.write_bytes(b"voice")
            transcript.write_text("exact transcript\n", encoding="utf-8")
            CONTROLLER.write_json(
                job / "manifest.json",
                {
                    "job_id": job_id,
                    "image_path": str(image),
                    "voice_path": str(voice),
                    "transcript_path": str(transcript),
                    "transcript_source": "user",
                    "tts_backend": "qwen3_tts_1_7b_base",
                    "reference_language": "en-US",
                    "target_language": "zh-CN",
                },
            )
            CONTROLLER.write_json(job / "status.json", {"state": "prepared"})
            main_env = root / "main.env"
            main_env.write_text("DYSTREAM_REF_IMAGE=/old.png\n", encoding="utf-8")
            references = root / "references"
            references.mkdir()
            current = references / "ref.wav"
            current.write_bytes(b"old")
            bridge_env = root / "qwen.env"
            bridge_env.write_text(
                f"OPENAI_SPEECH_REFERENCE_AUDIO={current}\n"
                "OPENAI_SPEECH_REFERENCE_TEXT=old transcript\n"
                "OPENAI_SPEECH_PROVIDER=vllm\n"
                "OPENAI_SPEECH_BACKEND=qwen3_tts_1_7b_base\n"
                "OPENAI_SPEECH_BRIDGE_HOST=127.0.0.1\n"
                "OPENAI_SPEECH_BRIDGE_PORT=8774\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                job_dir=job,
                runtime_root=runtime_root,
                repo_root=repo,
                main_env=main_env,
                tts_env=bridge_env,
                tts_provider="qwen3_tts_1_7b_base",
                fish_env=None,
                bridge_instance="qwen3",
                port=7862,
            )
            metrics = CONTROLLER.PCMProbeMetrics(-30.0, -10.0, 0, 1000)
            with mock.patch.object(
                CONTROLLER, "preflight_fish_reference", return_value=(metrics, metrics)
            ), mock.patch.object(
                CONTROLLER, "restart_fish_bridge"
            ), mock.patch.object(
                CONTROLLER, "probe_restarted_fish_bridge", return_value=metrics
            ), mock.patch.object(
                CONTROLLER, "run_demo_restart", return_value="custom-token"
            ), mock.patch.object(
                CONTROLLER, "verify_health"
            ), mock.patch.object(
                CONTROLLER, "verify_loaded_assets"
            ), mock.patch.object(CONTROLLER.time, "sleep"):
                result = CONTROLLER.activate(args)
            self.assertEqual(result, 0)
            self.assertEqual(
                CONTROLLER.read_env_value(main_env, "PIPECAT_TTS_BACKEND"),
                "qwen3_tts_1_7b_base",
            )
            self.assertEqual(
                CONTROLLER.read_env_value(main_env, "PIPECAT_TTS_BRIDGE_URI"),
                "ws://127.0.0.1:8774",
            )
            self.assertEqual(
                CONTROLLER.read_env_value(bridge_env, "OPENAI_SPEECH_BACKEND"),
                "qwen3_tts_1_7b_base",
            )

    def test_candidate_probe_rejects_large_rms_jump(self):
        current = CONTROLLER.PCMProbeMetrics(
            rms_dbfs=-30.0, peak_dbfs=-12.0, clipping_samples=0, samples=1000
        )
        candidate = CONTROLLER.PCMProbeMetrics(
            rms_dbfs=-23.0, peak_dbfs=-9.0, clipping_samples=0, samples=1000
        )
        with self.assertRaisesRegex(RuntimeError, "relative RMS"):
            CONTROLLER.validate_candidate_probe(candidate, current)

    def test_candidate_probe_accepts_safe_pcm(self):
        current = CONTROLLER.PCMProbeMetrics(
            rms_dbfs=-30.0, peak_dbfs=-12.0, clipping_samples=0, samples=1000
        )
        candidate = CONTROLLER.PCMProbeMetrics(
            rms_dbfs=-29.0, peak_dbfs=-10.0, clipping_samples=0, samples=1000
        )
        CONTROLLER.validate_candidate_probe(candidate, current)

    def test_candidate_probe_rejects_pcm_that_is_too_quiet_relative_to_current(self):
        current = CONTROLLER.PCMProbeMetrics(
            rms_dbfs=-30.0, peak_dbfs=-12.0, clipping_samples=0, samples=1000
        )
        candidate = CONTROLLER.PCMProbeMetrics(
            rms_dbfs=-42.0, peak_dbfs=-14.0, clipping_samples=0, samples=1000
        )
        current = current._replace(rms_dbfs=-29.0)
        with self.assertRaisesRegex(RuntimeError, "relative RMS"):
            CONTROLLER.validate_candidate_probe(candidate, current)

    def test_reference_transcription_always_uses_multilingual_sensevoice(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps({"text": "hello world"}) + "\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            main_env = Path(directory) / "main.env"
            main_env.write_text(
                "PIPECAT_ASR_MODEL=paraformer-zh-streaming\n", encoding="utf-8"
            )
            with mock.patch.object(
                CONTROLLER.subprocess, "run", return_value=completed
            ) as run:
                transcript = CONTROLLER.transcribe_reference(
                    Path(directory), main_env, Path(directory) / "voice.wav"
                )
        command = run.call_args.args[0]
        self.assertEqual(transcript, "hello world")
        self.assertIn("iic/SenseVoiceSmall", command)
        self.assertNotIn("paraformer-zh-streaming", command)

    def test_only_legacy_paraformer_or_missing_transcript_is_retranscribed(self):
        self.assertTrue(CONTROLLER.should_auto_transcribe({"transcript_path": ""}))
        self.assertTrue(
            CONTROLLER.should_auto_transcribe(
                {
                    "transcript_path": "/job/voice_reference.txt",
                    "transcript_source": "legacy_paraformer_auto",
                }
            )
        )
        self.assertFalse(
            CONTROLLER.should_auto_transcribe(
                {
                    "transcript_path": "/job/voice_reference.txt",
                    "transcript_source": "user",
                }
            )
        )
        self.assertFalse(
            CONTROLLER.should_auto_transcribe(
                {"transcript_path": "/job/voice_reference.txt"}
            )
        )

    def test_probe_pcm_enforces_duration_and_even_pcm16(self):
        sample_rate = 100
        safe = struct.pack("<h", 1000) * 30
        metrics = CONTROLLER.validate_probe_pcm(safe, sample_rate)
        self.assertEqual(metrics.samples, 30)
        with self.assertRaisesRegex(RuntimeError, "duration"):
            CONTROLLER.validate_probe_pcm(struct.pack("<h", 1000) * 29, sample_rate)
        with self.assertRaisesRegex(RuntimeError, "PCM16"):
            CONTROLLER.validate_probe_pcm(safe + b"x", sample_rate)

    def test_new_manifest_records_transcript_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            voice = root / "voice.wav"
            image.write_bytes(b"image")
            voice.write_bytes(b"voice")
            job = root / ("d" * 32)
            def normalize(source, destination):
                destination.write_bytes(source.read_bytes())
                return {"ok": True}
            with mock.patch(
                "customization_runtime._normalize_image", side_effect=normalize
            ), mock.patch(
                "customization_runtime._validate_official_face_path",
                return_value={"ok": True},
            ), mock.patch(
                "customization_runtime._normalize_voice", side_effect=normalize
            ):
                from customization_runtime import _prepare_job

                user_manifest = _prepare_job(job, image, voice, "written by user")
            self.assertEqual(user_manifest["transcript_source"], "user")

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
                value = int(0.1 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
                samples.extend(struct.pack("<hh", value, value))
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(bytes(samples))
            info = _normalize_voice(source, destination)
            self.assertEqual(info["source_type"], "audio")
            self.assertAlmostEqual(info["reference_dbfs"], -29.0, delta=0.2)
            self.assertLessEqual(info["reference_peak_dbfs"], -6.0)
            self.assertAlmostEqual(info["reference_input_dbfs"], -23.0, delta=0.3)
            self.assertLessEqual(info["normalization_gain_db"], 8.0)
            with wave.open(str(destination), "rb") as wav:
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertGreater(wav.getnframes(), 3 * 16000)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_clipped_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clipped.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(struct.pack("<h", 32767) * (16000 * 3))
            with self.assertRaisesRegex(CustomizationInputError, "clip"):
                _normalize_voice(source, Path(directory) / "out.wav")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_reference_with_large_dc_offset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dc.wav"
            samples = bytearray()
            for index in range(16000 * 3):
                value = 2000 + int(600 * math.sin(2 * math.pi * 220 * index / 16000))
                samples.extend(struct.pack("<h", value))
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(bytes(samples))
            with self.assertRaisesRegex(CustomizationInputError, "DC"):
                _normalize_voice(source, Path(directory) / "out.wav")

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


class ActivationProcessMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_polls_without_calling_blocking_wait(self):
        process = mock.Mock()
        process.poll.side_effect = [None, 0]
        process.wait.side_effect = AssertionError("blocking wait must not be used")

        returncode = await customization._wait_for_process_exit(
            process, poll_interval=0
        )

        self.assertEqual(returncode, 0)
        process.wait.assert_not_called()

    def test_activation_monitor_uses_cancellable_wait(self):
        source = inspect.getsource(customization.register_customization_routes)
        self.assertIn("await _wait_for_process_exit(process)", source)
        self.assertNotIn("asyncio.to_thread(process.wait)", source)

    async def test_wait_is_cancellable_while_child_process_remains_alive(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        try:
            monitor = asyncio.create_task(
                customization._wait_for_process_exit(process, poll_interval=0.01)
            )
            await asyncio.sleep(0.03)
            monitor.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(monitor, timeout=0.5)
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)


class CustomizationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config = self.root / "config"
        config.mkdir()
        main_env = config / "custom_cascade.env"
        tts_env = config / "fish_speech_bridge.env"
        fish_env = config / "fish_s2_pro.env"
        main_env.write_text("READY=1\n", encoding="utf-8")
        tts_env.write_text(
            "OPENAI_SPEECH_BRIDGE_PORT=8773\n", encoding="utf-8"
        )
        fish_env.write_text("READY=1\n", encoding="utf-8")
        self.environment = {
            "PIPECAT_TTS_PROVIDER": "fish_s2pro",
            "PIPECAT_TTS_BRIDGE_ENV_FILE": str(tts_env),
            "FISH_S2PRO_ENV_FILE": str(fish_env),
            "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
            "CUSTOMIZATION_ROOT": str(self.root / "customizations"),
            "CUSTOMIZATION_ALLOW_REMOTE": "1",
        }
        self.environment_patch = mock.patch.dict(
            os.environ, self.environment, clear=True
        )
        self.environment_patch.start()
        self.live_health_patch = mock.patch.object(
            customization, "_configured_backend_is_live", return_value=True
        )
        self.live_health_patch.start()
        app = web.Application()
        app["engine"] = SimpleNamespace(log=lambda _message: None)
        register_customization_routes(app)
        self.runtime_root = app["customization_paths"]["runtime_root"]
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.live_health_patch.stop()
        self.environment_patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def _form(
        *, backend=None, reference_language=None, target_language=None,
        transcript="reference text",
    ):
        form = FormData()
        form.add_field(
            "image", io.BytesIO(b"image"), filename="avatar.png", content_type="image/png"
        )
        form.add_field(
            "voice_media",
            io.BytesIO(b"voice"),
            filename="voice.wav",
            content_type="audio/wav",
        )
        form.add_field("transcript", transcript)
        if backend is not None:
            form.add_field("tts_backend", backend)
        if reference_language is not None:
            form.add_field("reference_language", reference_language)
        if target_language is not None:
            form.add_field("target_language", target_language)
        return form

    @staticmethod
    def _normalization_patches():
        def fake_image(_source, destination):
            destination.write_bytes(b"normalized-image")
            return {"normalized_size": [1024, 1024]}

        def fake_voice(_source, destination):
            destination.write_bytes(b"normalized-voice")
            return {"reference_duration_seconds": 5.0}

        return (
            mock.patch.object(customization, "_normalize_image", side_effect=fake_image),
            mock.patch.object(
                customization, "_validate_official_face_path", return_value={"ok": True}
            ),
            mock.patch.object(customization, "_normalize_voice", side_effect=fake_voice),
        )

    def _create_prepared_job(self, *, job_id, provider="fish_s2pro"):
        job_dir = self.runtime_root / job_id
        assets = job_dir / "assets"
        assets.mkdir(parents=True)
        image = assets / "reference.png"
        voice = assets / "voice_reference.wav"
        transcript = assets / "voice_reference.txt"
        image.write_bytes(b"image")
        voice.write_bytes(b"voice")
        transcript.write_text("old transcript\n", encoding="utf-8")
        manifest = {
            "job_id": job_id,
            "image_path": str(image),
            "voice_path": str(voice),
            "transcript_path": str(transcript),
            "tts_provider": provider,
        }
        (job_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (job_dir / "status.json").write_text(
            json.dumps({"job_id": job_id, "state": "prepared"}), encoding="utf-8"
        )
        return job_dir

    async def test_active_adds_registry_and_normalizes_legacy_selection(self):
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "active.json").write_text(
            json.dumps({"job_id": "a" * 32, "tts_provider": "fish_s2pro"}),
            encoding="utf-8",
        )
        with mock.patch.object(customization.subprocess, "Popen") as popen:
            response = await self.client.get("/api/customization/active")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["tts_provider"], "fish_s2pro")
        self.assertEqual(payload["active"]["tts_backend"], "fish_s2_pro")
        self.assertEqual(payload["active"]["reference_language"], "auto")
        self.assertEqual(payload["active"]["target_language"], "zh-CN")
        self.assertEqual(
            [item["id"] for item in payload["tts_options"]["backends"]],
            ["fish_s2_pro", "qwen3_tts_1_7b_base", "cosyvoice3_0_5b", "voxcpm2"],
        )
        popen.assert_not_called()

    async def test_prepare_old_request_uses_defaults_and_persists_status(self):
        image_patch, face_patch, voice_patch = self._normalization_patches()
        with image_patch, face_patch, voice_patch:
            response = await self.client.post(
                "/api/customization/prepare",
                data=self._form(),
                headers={"X-DyStream-Customize": "1"},
            )
        self.assertEqual(response.status, 201)
        status = await response.json()
        expected = {
            "tts_backend": "fish_s2_pro",
            "reference_language": "auto",
            "target_language": "zh-CN",
        }
        self.assertEqual({key: status[key] for key in expected}, expected)
        manifest = json.loads(
            (self.runtime_root / status["job_id"] / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        persisted_status = json.loads(
            (self.runtime_root / status["job_id"] / "status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual({key: manifest[key] for key in expected}, expected)
        self.assertEqual({key: persisted_status[key] for key in expected}, expected)

    async def test_prepare_rejects_unknown_backend(self):
        with mock.patch.object(customization, "_prepare_job") as prepare_job:
            response = await self.client.post(
                "/api/customization/prepare",
                data=self._form(backend="unknown_backend"),
                headers={"X-DyStream-Customize": "1"},
            )
        self.assertEqual(response.status, 400)
        payload = await response.json()
        self.assertIn("unknown TTS backend", payload["message"])
        prepare_job.assert_not_called()

    async def test_prepare_auto_transcribes_blank_reference_for_confirmation(self):
        image_patch, face_patch, voice_patch = self._normalization_patches()
        with image_patch, face_patch, voice_patch, mock.patch.object(
            customization, "_transcribe_reference", return_value="detected transcript"
        ) as transcribe:
            response = await self.client.post(
                "/api/customization/prepare",
                data=self._form(transcript="", reference_language="en-US"),
                headers={"X-DyStream-Customize": "1"},
            )
        self.assertEqual(response.status, 201)
        status = await response.json()
        self.assertEqual(status["transcript"], "detected transcript")
        self.assertEqual(status["transcript_source"], "sensevoice_auto")
        transcribe.assert_called_once()
        self.assertEqual(transcribe.call_args.args[1], "en-US")
        manifest = json.loads(
            (self.runtime_root / status["job_id"] / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["transcript"], "detected transcript")
        self.assertEqual(manifest["transcript_source"], "sensevoice_auto")

    async def test_activation_override_is_validated_and_persisted_before_spawn(self):
        job_id = "c" * 32
        job_dir = self._create_prepared_job(job_id=job_id)
        with mock.patch.object(customization.subprocess, "Popen") as popen:
            response = await self.client.post(
                f"/api/customization/{job_id}/activate",
                json={
                    "tts_backend": "fish_s2pro",
                    "reference_language": "en-US",
                    "target_language": "zh-CN",
                    "transcript": "corrected transcript",
                },
                headers={"X-DyStream-Customize": "1"},
            )
        self.assertEqual(response.status, 202)
        payload = await response.json()
        self.assertEqual(payload["tts_backend"], "fish_s2_pro")
        self.assertEqual(payload["transcript"], "corrected transcript")
        manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tts_backend"], "fish_s2_pro")
        self.assertEqual(manifest["reference_language"], "en-US")
        self.assertEqual(
            (job_dir / "assets" / "voice_reference.txt").read_text(encoding="utf-8"),
            "corrected transcript\n",
        )
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(
            command[command.index("--bridge-instance") + 1], "flashav2av"
        )
        self.assertEqual(
            command[command.index("--tts-provider") + 1], "fish_s2_pro"
        )
        self.assertNotIn("--fish-env", command)

    async def test_activation_rejects_unavailable_backend_without_mutation(self):
        job_id = "d" * 32
        job_dir = self._create_prepared_job(job_id=job_id)
        original = (job_dir / "manifest.json").read_bytes()
        with mock.patch.object(customization.subprocess, "Popen") as popen:
            response = await self.client.post(
                f"/api/customization/{job_id}/activate",
                json={"tts_backend": "qwen3_tts_1_7b_base"},
                headers={"X-DyStream-Customize": "1"},
            )
        self.assertEqual(response.status, 409)
        payload = await response.json()
        self.assertEqual(payload["state"], "unavailable")
        self.assertEqual(payload["disabled_reason"], "dependency_and_weights_missing")
        self.assertEqual((job_dir / "manifest.json").read_bytes(), original)
        popen.assert_not_called()

    async def test_activation_routes_ready_sidecar_to_its_own_env_and_instance(self):
        job_id = "f" * 32
        self._create_prepared_job(job_id=job_id)
        qwen_env = self.root / "config" / "qwen_bridge.env"
        qwen_env.write_text(
            "OPENAI_SPEECH_BRIDGE_PORT=8774\n", encoding="utf-8"
        )
        paths = self.client.server.app["customization_paths"]
        paths["backend_envs"]["qwen3_tts_1_7b_base"] = qwen_env
        paths["backend_bridge_instances"]["qwen3_tts_1_7b_base"] = "qwen3"
        with mock.patch.dict(
            os.environ,
            {"CUSTOMIZATION_TTS_READY_BACKENDS": "qwen3_tts_1_7b_base"},
            clear=False,
        ), mock.patch.object(
            customization, "_configured_backend_is_live", return_value=True
        ), mock.patch.object(customization.subprocess, "Popen") as popen:
            response = await self.client.post(
                f"/api/customization/{job_id}/activate",
                json={"tts_backend": "qwen3_tts_1_7b_base"},
                headers={"X-DyStream-Customize": "1"},
            )
        self.assertEqual(response.status, 202)
        command = popen.call_args.args[0]
        self.assertEqual(
            command[command.index("--tts-provider") + 1],
            "qwen3_tts_1_7b_base",
        )
        self.assertEqual(command[command.index("--tts-env") + 1], str(qwen_env))
        self.assertEqual(command[command.index("--bridge-instance") + 1], "qwen3")

    async def test_activation_empty_body_preserves_old_client_contract(self):
        job_id = "e" * 32
        self._create_prepared_job(job_id=job_id)
        with mock.patch.object(customization.subprocess, "Popen") as popen:
            response = await self.client.post(
                f"/api/customization/{job_id}/activate",
                headers={"X-DyStream-Customize": "1"},
            )
        self.assertEqual(response.status, 202)
        payload = await response.json()
        self.assertEqual(payload["tts_backend"], "fish_s2_pro")
        self.assertEqual(payload["reference_language"], "auto")
        self.assertEqual(payload["target_language"], "zh-CN")
        self.assertEqual(payload["transcript"], "old transcript")
        popen.assert_called_once()

    async def test_legacy_voxcpm2_client_defaults_to_current_runtime(self):
        vox_root = self.root / "legacy_vox"
        config = vox_root / "config"
        config.mkdir(parents=True)
        main_env = config / "custom_cascade.env"
        vox_env = config / "voxcpm2.env"
        main_env.write_text("PIPECAT_TTS_PROVIDER=voxcpm2\n", encoding="utf-8")
        vox_env.write_text("VOXCPM2_BRIDGE_PORT=8770\n", encoding="utf-8")
        environment = {
            "PIPECAT_TTS_PROVIDER": "voxcpm2",
            "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
            "VOXCPM2_ENV_FILE": str(vox_env),
            "CUSTOMIZATION_ROOT": str(vox_root / "customizations"),
            "CUSTOMIZATION_ALLOW_REMOTE": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            app = web.Application()
            app["engine"] = SimpleNamespace(log=lambda _message: None)
            register_customization_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            image_patch, face_patch, voice_patch = self._normalization_patches()
            with image_patch, face_patch, voice_patch, mock.patch.object(
                customization, "_transcribe_reference"
            ) as transcribe:
                response = await client.post(
                    "/api/customization/prepare",
                    data=self._form(transcript=""),
                    headers={"X-DyStream-Customize": "1"},
                )
            self.assertEqual(response.status, 201)
            prepared = await response.json()
            self.assertEqual(prepared["tts_backend"], "voxcpm2")
            self.assertEqual(prepared["transcript"], "")
            transcribe.assert_not_called()

            with mock.patch.object(customization.subprocess, "Popen") as popen:
                response = await client.post(
                    f"/api/customization/{prepared['job_id']}/activate",
                    headers={"X-DyStream-Customize": "1"},
                )
            self.assertEqual(response.status, 202)
            payload = await response.json()
            self.assertEqual(payload["tts_backend"], "voxcpm2")
            command = popen.call_args.args[0]
            self.assertEqual(
                command[command.index("--tts-provider") + 1], "voxcpm2"
            )
            self.assertEqual(command[command.index("--tts-env") + 1], str(vox_env))
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
