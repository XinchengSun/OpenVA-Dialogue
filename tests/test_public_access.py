import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pipecat_dystream.public_access import (
    RedactedAccessLogger,
    host_without_port,
    is_loopback_host,
    token_matches,
)


def test_host_without_port_supports_ipv4_names_and_bracketed_ipv6():
    assert host_without_port("127.0.0.1:7860") == "127.0.0.1"
    assert host_without_port("LOCALHOST:6008") == "localhost"
    assert host_without_port("[::1]:7860") == "::1"


def test_only_loopback_hosts_bypass_public_access():
    assert is_loopback_host("localhost:6008")
    assert is_loopback_host("127.0.0.1:7860")
    assert is_loopback_host("[::1]:7860")
    assert not is_loopback_host("avatar.example.com")
    assert not is_loopback_host("203.0.113.10:7860")
    assert not is_loopback_host("")


def test_access_token_requires_an_exact_nonempty_match():
    assert token_matches("secret-token", "secret-token")
    assert not token_matches("secret-token-x", "secret-token")
    assert not token_matches(None, "secret-token")
    assert not token_matches("secret-token", "")


def test_access_logger_never_records_query_strings_or_tokens():
    logger = mock.Mock()
    access_logger = RedactedAccessLogger(logger, "%r")
    request = SimpleNamespace(
        remote="203.0.113.10",
        method="GET",
        path="/demo",
        path_qs="/demo?access_token=must-not-appear",
    )
    response = SimpleNamespace(status=302, body_length=17)

    access_logger.log(request, response, 0.125)

    logger.info.assert_called_once_with(
        '%s "%s %s" %s %s %.6f',
        "203.0.113.10",
        "GET",
        "/demo",
        302,
        17,
        0.125,
    )


def test_server_uses_the_redacted_access_logger():
    source = (Path(__file__).resolve().parents[1] / "server_mse.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "web"
        and node.func.attr == "run_app"
    ]
    assert len(calls) == 1
    kwargs = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    access_logger = kwargs.get("access_log_class")
    assert isinstance(access_logger, ast.Name)
    assert access_logger.id == "RedactedAccessLogger"


if __name__ == "__main__":
    test_host_without_port_supports_ipv4_names_and_bracketed_ipv6()
    test_only_loopback_hosts_bypass_public_access()
    test_access_token_requires_an_exact_nonempty_match()
    test_access_logger_never_records_query_strings_or_tokens()
    test_server_uses_the_redacted_access_logger()
    print("public access logic: 5/5 passed")
