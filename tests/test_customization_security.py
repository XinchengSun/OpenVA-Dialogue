import asyncio
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import customization_runtime as customization


ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_SCRIPT = ROOT / "scripts" / "configure_customization_access.py"


def _env_value(text: str, key: str) -> str:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    raise AssertionError(f"missing environment key: {key}")


class _FakePart:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read_chunk(self, size):
        del size
        return self._chunks.pop(0) if self._chunks else b""


class CustomizationSecurityUnitTests(unittest.IsolatedAsyncioTestCase):
    def test_signed_session_hides_admin_secret_and_expires(self):
        token = "a" * 64
        session = customization._issue_admin_session(
            token,
            now=1000,
            ttl_sec=300,
            nonce="b" * 32,
        )
        self.assertNotIn(token, session)
        self.assertTrue(customization._admin_session_valid(session, token, now=1299))
        self.assertFalse(customization._admin_session_valid(session, token, now=1300))
        self.assertFalse(
            customization._admin_session_valid(session[:-1] + "0", token, now=1100)
        )

    def test_legacy_allow_remote_flag_cannot_enable_admin_routes(self):
        with mock.patch.dict(
            os.environ,
            {"CUSTOMIZATION_ALLOW_REMOTE": "1", "CUSTOMIZATION_REMOTE_ENABLED": "0"},
            clear=False,
        ):
            self.assertFalse(customization._remote_customization_enabled())

    def test_customization_path_matching_has_path_boundaries(self):
        self.assertTrue(customization.is_customization_path("/customize"))
        self.assertTrue(customization.is_customization_path("/customize/login"))
        self.assertTrue(customization.is_customization_path("/api/customization/active"))
        self.assertFalse(customization.is_customization_path("/customize-evil"))
        self.assertFalse(customization.is_customization_path("/api/customizations"))

    def test_upload_limits_match_public_resource_budget(self):
        self.assertEqual(customization.MAX_IMAGE_BYTES, 10 * 1024 * 1024)
        self.assertEqual(customization.MAX_MEDIA_BYTES, 50 * 1024 * 1024)
        self.assertLessEqual(customization.MAX_REQUEST_BYTES, 61 * 1024 * 1024)

    async def test_file_limit_stops_before_writing_overflow_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "upload.bin"
            with self.assertRaises(web.HTTPRequestEntityTooLarge):
                await customization._save_part(
                    _FakePart([b"ab", b"cd"]),
                    destination,
                    3,
                    customization._UploadBudget(10),
                )
            self.assertEqual(destination.read_bytes(), b"ab")

    def test_login_page_does_not_persist_or_query_encode_admin_token(self):
        page = (ROOT / "static" / "customize_login.html").read_text(encoding="utf-8")
        self.assertIn("/api/customization/session", page)
        self.assertIn("'X-DyStream-Customize': '1'", page)
        self.assertIn("window.location.replace('/customize')", page)
        self.assertNotIn("localStorage", page)
        self.assertNotIn("access_token", page)

    def test_customize_page_marks_global_switch_and_handles_busy_409(self):
        page = (ROOT / "static" / "customize.html").read_text(encoding="utf-8")
        self.assertIn("这是全局切换", page)
        self.assertIn("所有访客都会看到", page)
        self.assertIn("response.status === 409 && status.state === 'busy'", page)
        self.assertIn("response.headers.get('Retry-After')", page)

    def test_activation_lock_probe_fails_closed_when_controller_owns_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "activation.lock").touch()

            def locked_flock(_descriptor, operation):
                if operation == 3:
                    raise BlockingIOError

            fake_fcntl = SimpleNamespace(
                LOCK_EX=1,
                LOCK_NB=2,
                LOCK_UN=4,
                flock=locked_flock,
            )
            with mock.patch.dict(sys.modules, {"fcntl": fake_fcntl}):
                self.assertTrue(customization._activation_lock_is_held(root))


class PublicGateCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_paths_bypass_public_token_but_keep_admin_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            main_env = config / "custom_cascade.env"
            main_env.write_text("PIPECAT_TTS_PROVIDER=fish_s2pro\n", encoding="utf-8")

            @web.middleware
            async def public_gate(request, handler):
                if customization.is_customization_path(request.path):
                    return await handler(request)
                if request.cookies.get("dystream_public_access") != "public-secret":
                    return web.Response(status=401)
                return await handler(request)

            async def chat_page(_request):
                return web.Response(text="chat")

            with mock.patch.dict(
                os.environ,
                {
                    "CUSTOMIZATION_REMOTE_ENABLED": "1",
                    "CUSTOMIZATION_ADMIN_TOKEN": "c" * 64,
                    "CUSTOMIZATION_PUBLIC_ORIGIN": "https://avatar.example",
                    "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                    "CUSTOMIZATION_ROOT": str(root / "customizations"),
                },
                clear=False,
            ):
                app = web.Application(middlewares=[public_gate])
                app["engine"] = SimpleNamespace(log=lambda _message: None, media_clients=set())
                app.router.add_get("/", chat_page)
                customization.register_customization_routes(app)
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    login_page = await client.get(
                        "/customize/login",
                        headers={"Host": "avatar.example"},
                    )
                    self.assertEqual(login_page.status, 200)
                    ordinary_chat = await client.get(
                        "/",
                        headers={"Host": "avatar.example"},
                    )
                    self.assertEqual(ordinary_chat.status, 401)
                    custom_api = await client.get(
                        "/api/customization/active",
                        headers={"Host": "avatar.example"},
                    )
                    self.assertEqual(custom_api.status, 401)
                    csrf_failure = await client.post(
                        "/api/customization/session",
                        json={"token": "c" * 64},
                        headers={"Host": "avatar.example"},
                    )
                    self.assertEqual(csrf_failure.status, 403)
                finally:
                    await client.close()


class RemoteCustomizationAccessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config = self.root / "config"
        config.mkdir()
        main_env = config / "custom_cascade.env"
        main_env.write_text("PIPECAT_TTS_PROVIDER=fish_s2pro\n", encoding="utf-8")
        self.admin_token = "c" * 64
        self.environment_patch = mock.patch.dict(
            os.environ,
            {
                "CUSTOMIZATION_REMOTE_ENABLED": "1",
                "CUSTOMIZATION_ADMIN_TOKEN": self.admin_token,
                "CUSTOMIZATION_PUBLIC_ORIGIN": "https://avatar.example",
                "CUSTOMIZATION_ADMIN_SESSION_TTL_SEC": "600",
                "CUSTOMIZATION_ALLOW_REMOTE": "1",
                "CUSTOMIZATION_MAIN_ENV_FILE": str(main_env),
                "CUSTOMIZATION_ROOT": str(self.root / "customizations"),
            },
            clear=False,
        )
        self.environment_patch.start()
        self.busy = False
        app = web.Application()
        app["engine"] = SimpleNamespace(log=lambda _message: None, media_clients=set())
        app["workload_admission_lock"] = asyncio.Lock()
        app[customization.CUSTOMIZATION_BUSY_CHECK_APP_KEY] = lambda _operation: {
            "active": self.busy,
            "reason": "conversation_active",
            "retry_after_seconds": 7,
        }
        customization.register_customization_routes(app)
        self.app = app
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.environment_patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def _external_headers(**extra):
        return {
            "Host": "avatar.example",
            "Origin": "https://avatar.example",
            "X-DyStream-Customize": "1",
            **extra,
        }

    async def _login(self):
        response = await self.client.post(
            "/api/customization/session",
            json={"token": self.admin_token},
            headers=self._external_headers(),
        )
        self.assertEqual(response.status, 200)
        cookie = response.cookies[customization.CUSTOMIZATION_ADMIN_COOKIE_NAME]
        return response, cookie.value

    async def test_remote_page_redirects_to_post_login(self):
        response = await self.client.get(
            "/customize",
            headers={"Host": "avatar.example"},
            allow_redirects=False,
        )
        self.assertEqual(response.status, 302)
        self.assertEqual(response.headers["Location"], "/customize/login")
        login = await self.client.get(
            "/customize/login",
            headers={"Host": "avatar.example"},
        )
        self.assertEqual(login.status, 200)
        self.assertEqual(login.headers["Cache-Control"], "no-store")

    async def test_login_requires_exact_origin_and_custom_header(self):
        missing_origin = await self.client.post(
            "/api/customization/session",
            json={"token": self.admin_token},
            headers={"Host": "avatar.example", "X-DyStream-Customize": "1"},
        )
        self.assertEqual(missing_origin.status, 403)
        evil_origin = await self.client.post(
            "/api/customization/session",
            json={"token": self.admin_token},
            headers=self._external_headers(Origin="https://evil.example"),
        )
        self.assertEqual(evil_origin.status, 403)

    async def test_login_sets_only_a_signed_hardened_session_cookie(self):
        response, cookie_value = await self._login()
        self.assertEqual(
            set(response.cookies),
            {customization.CUSTOMIZATION_ADMIN_COOKIE_NAME},
        )
        cookie = response.cookies[customization.CUSTOMIZATION_ADMIN_COOKIE_NAME]
        self.assertNotIn(self.admin_token, cookie_value)
        self.assertTrue(cookie["secure"])
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Strict")
        self.assertEqual(cookie["path"], "/")
        self.assertEqual(cookie["max-age"], "600")
        authorized = await self.client.get(
            "/customize",
            headers={
                "Host": "avatar.example",
                "Cookie": (
                    f"{customization.CUSTOMIZATION_ADMIN_COOKIE_NAME}={cookie_value}"
                ),
            },
        )
        self.assertEqual(authorized.status, 200)
        self.assertEqual(authorized.headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", authorized.headers["Content-Security-Policy"])
        self.assertEqual(authorized.headers["X-Frame-Options"], "DENY")

        active = await self.client.get(
            "/api/customization/active",
            headers={
                "Host": "avatar.example",
                "Cookie": (
                    f"{customization.CUSTOMIZATION_ADMIN_COOKIE_NAME}={cookie_value}"
                ),
            },
        )
        self.assertEqual(active.status, 200)
        self.assertEqual(active.headers["Cache-Control"], "no-store")

    async def test_wrong_or_public_only_token_cannot_access_admin_api(self):
        wrong = await self.client.post(
            "/api/customization/session",
            json={"token": "d" * 64},
            headers=self._external_headers(),
        )
        self.assertEqual(wrong.status, 401)
        self.assertNotIn("Set-Cookie", wrong.headers)
        public_only = await self.client.get(
            "/api/customization/active",
            headers={
                "Host": "avatar.example",
                "Cookie": "dystream_public_access=public-chat-token",
            },
        )
        self.assertEqual(public_only.status, 401)

    async def test_forwarded_public_request_cannot_claim_localhost_bypass(self):
        response = await self.client.get(
            "/customize",
            headers={
                "Host": "127.0.0.1",
                "CF-Connecting-IP": "203.0.113.8",
            },
            allow_redirects=False,
        )
        self.assertEqual(response.status, 302)
        local = await self.client.get(
            "/customize",
            headers={"Host": "127.0.0.1"},
        )
        self.assertEqual(local.status, 200)

    async def test_prepare_and_activate_fail_closed_while_conversation_is_busy(self):
        self.busy = True
        prepare = await self.client.post(
            "/api/customization/prepare",
            data=b"",
            headers={"Host": "127.0.0.1", "X-DyStream-Customize": "1"},
        )
        self.assertEqual(prepare.status, 409)
        self.assertEqual((await prepare.json())["reason"], "conversation_active")
        activate = await self.client.post(
            f"/api/customization/{'a' * 32}/activate",
            data=b"",
            headers={"Host": "127.0.0.1", "X-DyStream-Customize": "1"},
        )
        self.assertEqual(activate.status, 409)
        self.assertEqual(activate.headers["Retry-After"], "7")

    async def test_prepare_gate_rejects_parallel_work_without_queueing_body(self):
        lock = self.app["customization_prepare_lock"]
        await lock.acquire()
        try:
            response = await self.client.post(
                "/api/customization/prepare",
                data=io.BytesIO(b"not-read"),
                headers={"Host": "127.0.0.1", "X-DyStream-Customize": "1"},
            )
        finally:
            lock.release()
        self.assertEqual(response.status, 409)
        self.assertEqual(
            (await response.json())["reason"], "customization_prepare_active"
        )

    async def test_nonterminal_persisted_job_blocks_after_http_process_restart(self):
        job_dir = self.app["customization_paths"]["runtime_root"] / ("b" * 32)
        job_dir.mkdir(parents=True)
        (job_dir / "status.json").write_text(
            '{"state":"restarting"}\n',
            encoding="utf-8",
        )
        self.assertTrue(customization.customization_workload_active(self.app))
        response = await self.client.post(
            f"/api/customization/{'a' * 32}/activate",
            data=b"",
            headers={"Host": "127.0.0.1", "X-DyStream-Customize": "1"},
        )
        self.assertEqual(response.status, 409)
        self.assertEqual(
            (await response.json())["reason"],
            "customization_activation_persisted",
        )


class ConfigureCustomizationAccessTests(unittest.TestCase):
    def _run(self, env_file: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CONFIGURE_SCRIPT),
                *args,
                "--env-file",
                str(env_file),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_enable_rotate_disable_are_atomic_and_do_not_print_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "runtime.env"
            env_file.write_text(
                "KEEP=value\n"
                "PUBLIC_ACCESS_TOKEN=keep-public-secret\n"
                "CUSTOMIZATION_ALLOW_REMOTE=1\n",
                encoding="utf-8",
            )
            enabled = self._run(
                env_file,
                "enable",
                "--public-origin",
                "https://Avatar.Example/",
                "--session-ttl",
                "900",
            )
            self.assertEqual(enabled.returncode, 0, enabled.stderr)
            configured = env_file.read_text(encoding="utf-8")
            token = _env_value(configured, "CUSTOMIZATION_ADMIN_TOKEN")
            self.assertRegex(token, r"^[0-9a-f]{64}$")
            self.assertNotIn(token, enabled.stdout)
            self.assertNotIn("keep-public-secret", enabled.stdout + enabled.stderr)
            self.assertIn("KEEP=value", configured)
            self.assertEqual(
                _env_value(configured, "PUBLIC_ACCESS_TOKEN"),
                "keep-public-secret",
            )
            self.assertEqual(_env_value(configured, "CUSTOMIZATION_ALLOW_REMOTE"), "0")
            self.assertEqual(_env_value(configured, "CUSTOMIZATION_REMOTE_ENABLED"), "1")
            self.assertEqual(
                _env_value(configured, "CUSTOMIZATION_PUBLIC_ORIGIN"),
                "https://avatar.example",
            )

            status = self._run(env_file, "status")
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertNotIn(token, status.stdout + status.stderr)
            self.assertNotIn("keep-public-secret", status.stdout + status.stderr)

            rotated = self._run(env_file, "rotate-token")
            self.assertEqual(rotated.returncode, 0, rotated.stderr)
            rotated_text = env_file.read_text(encoding="utf-8")
            rotated_token = _env_value(rotated_text, "CUSTOMIZATION_ADMIN_TOKEN")
            self.assertNotEqual(rotated_token, token)
            self.assertNotIn(rotated_token, rotated.stdout)
            self.assertNotIn("keep-public-secret", rotated.stdout + rotated.stderr)

            shown = self._run(env_file, "show-token")
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertEqual(shown.stdout.strip(), rotated_token)

            disabled = self._run(env_file, "disable")
            self.assertEqual(disabled.returncode, 0, disabled.stderr)
            disabled_text = env_file.read_text(encoding="utf-8")
            self.assertEqual(
                _env_value(disabled_text, "CUSTOMIZATION_REMOTE_ENABLED"), "0"
            )
            self.assertEqual(
                _env_value(disabled_text, "CUSTOMIZATION_ADMIN_TOKEN"), rotated_token
            )
            self.assertEqual(
                _env_value(disabled_text, "PUBLIC_ACCESS_TOKEN"),
                "keep-public-secret",
            )
            self.assertNotIn("keep-public-secret", disabled.stdout + disabled.stderr)
            if os.name != "nt":
                self.assertEqual(env_file.stat().st_mode & 0o777, 0o600)

    def test_status_redacts_token_and_invalid_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "runtime.env"
            token = "f" * 64
            env_file.write_text(
                "CUSTOMIZATION_REMOTE_ENABLED=1\n"
                f"CUSTOMIZATION_ADMIN_TOKEN={token}\n"
                "CUSTOMIZATION_PUBLIC_ORIGIN=https://avatar.example\n",
                encoding="utf-8",
            )
            status = self._run(env_file, "status")
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn("admin_token_configured=1", status.stdout)
            self.assertNotIn(token, status.stdout)
            invalid = self._run(
                env_file,
                "enable",
                "--public-origin",
                "http://avatar.example/path",
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertNotIn(token, invalid.stdout + invalid.stderr)

    def test_manage_script_never_puts_admin_token_in_login_url(self):
        script = (ROOT / "scripts" / "manage_customization_access.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('URL_FILE="$STATE_DIR/public_url"', script)
        self.assertIn('configure enable --public-origin "$(public_origin)"', script)
        self.assertIn("%s/customize/login", script)
        self.assertIn("share-url", script)
        self.assertIn("#token=%s", script)
        self.assertIn("urllib.parse.quote", script)
        self.assertIn('origin_value="$(public_origin)"', script)
        self.assertIn("show-token", script)
        self.assertNotIn("admin_token=", script)
        self.assertNotIn("access_token=", script)
        self.assertNotIn("PUBLIC_ACCESS_TOKEN", script)


if __name__ == "__main__":
    unittest.main()
