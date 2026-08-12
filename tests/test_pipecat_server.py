import os
import unittest
from unittest.mock import patch

from pipecat_dystream.ice import (
    browser_ice_servers_from_env,
    ice_servers_from_env,
)


class IceServerConfigurationTests(unittest.TestCase):
    def test_empty_value_disables_external_ice_servers(self):
        with patch.dict(os.environ, {"PIPECAT_ICE_SERVERS": ""}):
            self.assertEqual(ice_servers_from_env(), [])

    def test_comma_separated_stun_urls_are_trimmed(self):
        with patch.dict(
            os.environ,
            {
                "PIPECAT_ICE_SERVERS": (
                    " stun:stun.miwifi.com:3478,"
                    "stun:stun.chat.bilibili.com:3478 "
                )
            },
        ):
            servers = ice_servers_from_env()
        self.assertEqual([server.urls for server in servers], [
            "stun:stun.miwifi.com:3478",
            "stun:stun.chat.bilibili.com:3478",
        ])

    def test_browser_config_includes_domestic_stun_without_turn(self):
        with patch.dict(os.environ, {"PIPECAT_TURN_URL": ""}):
            servers = browser_ice_servers_from_env()
        self.assertEqual(
            servers,
            [{
                "urls": [
                    "stun:stun.miwifi.com:3478",
                    "stun:stun.chat.bilibili.com:3478",
                ],
            }],
        )

    def test_browser_config_includes_tunneled_turn(self):
        env = {
            "PIPECAT_TURN_URL": "turn:127.0.0.1:3478?transport=tcp",
            "PIPECAT_TURN_USERNAME": "test-user",
            "PIPECAT_TURN_CREDENTIAL": "test-credential",
        }
        with patch.dict(os.environ, env):
            servers = browser_ice_servers_from_env()
        self.assertEqual(servers[-1], {
            "urls": env["PIPECAT_TURN_URL"],
            "username": env["PIPECAT_TURN_USERNAME"],
            "credential": env["PIPECAT_TURN_CREDENTIAL"],
        })


if __name__ == "__main__":
    unittest.main()
