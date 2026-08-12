from pipecat_dystream.public_access import (
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


if __name__ == "__main__":
    test_host_without_port_supports_ipv4_names_and_bracketed_ipv6()
    test_only_loopback_hosts_bypass_public_access()
    test_access_token_requires_an_exact_nonempty_match()
    print("public access logic: 3/3 passed")
