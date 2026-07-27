"""SSRF guard for web_crawler: only publicly-routable destinations.

The resolver is injected in most tests so these stay hermetic (no DNS), which
also lets us pin the cases that matter: a public name that resolves to a private
address, and a name that resolves to a mix.
"""

import pytest

from mnemoai.server.tools.safety import classify_url


def fixed_resolver(*addresses):
    """A resolver returning a fixed address list, ignoring the hostname."""
    return lambda host: list(addresses)


class TestSchemeValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/x",
            "gopher://example.com",
            "data:text/plain,hello",
        ],
    )
    def test_non_http_schemes_blocked(self, url):
        verdict = classify_url(url)
        assert verdict.blocked
        assert "http" in verdict.reason

    def test_empty_url_blocked(self):
        assert classify_url("").blocked
        assert classify_url("   ").blocked

    def test_no_hostname_blocked(self):
        assert classify_url("http://").blocked


class TestLocalAndPrivateDestinations:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/admin",
            "http://localhost:8080/",
            "https://LOCALHOST/x",
        ],
    )
    def test_localhost_blocked_by_name(self, url):
        verdict = classify_url(url)
        assert verdict.blocked
        assert "localhost" in verdict.reason.lower()

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "127.1.2.3",
            "10.0.0.5",
            "192.168.1.1",
            "172.16.4.2",
            "169.254.169.254",  # cloud instance metadata
            "100.64.0.1",  # CGNAT
            "0.0.0.0",
            "224.0.0.1",  # multicast
        ],
    )
    def test_private_literal_ips_blocked(self, ip):
        verdict = classify_url(f"http://{ip}/latest/meta-data/")
        assert verdict.blocked, ip
        assert "non-public" in verdict.reason

    def test_metadata_endpoint_gets_explicit_warning(self):
        verdict = classify_url("http://169.254.169.254/latest/meta-data/iam/")
        assert verdict.blocked
        assert "instance-metadata" in verdict.reason
        assert "IAM credentials" in verdict.reason

    @pytest.mark.parametrize(
        "ip",
        ["::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1"],
    )
    def test_private_ipv6_blocked(self, ip):
        assert classify_url(f"http://[{ip}]/").blocked

    def test_public_name_resolving_to_private_is_blocked(self):
        """The actual DNS-based bypass: a public-looking name pointing inward."""
        verdict = classify_url(
            "http://totally-innocent.example.com/",
            resolver=fixed_resolver("169.254.169.254"),
        )
        assert verdict.blocked
        assert "169.254.169.254" in verdict.reason

    def test_mixed_resolution_is_blocked(self):
        """If any address is private we can't control which one gets used."""
        verdict = classify_url(
            "http://mixed.example.com/",
            resolver=fixed_resolver("93.184.216.34", "10.0.0.1"),
        )
        assert verdict.blocked


class TestPublicDestinations:
    def test_public_name_allowed(self):
        verdict = classify_url(
            "https://example.com/page", resolver=fixed_resolver("93.184.216.34")
        )
        assert not verdict.blocked
        assert verdict.host == "example.com"
        assert verdict.addresses == ("93.184.216.34",)

    def test_public_literal_ip_allowed(self):
        assert not classify_url("https://93.184.216.34/").blocked

    def test_public_ipv6_allowed(self):
        assert not classify_url("https://[2606:2800:220:1:248:1893:25c8:1946]/").blocked

    def test_url_with_port_and_path_allowed(self):
        verdict = classify_url(
            "https://docs.example.com:8443/a/b?c=d#e",
            resolver=fixed_resolver("93.184.216.34"),
        )
        assert not verdict.blocked


class TestResolutionFailures:
    def test_unresolvable_host_blocked(self):
        def boom(host):
            raise OSError("Name or service not known")

        verdict = classify_url("http://nope.invalid/", resolver=boom)
        assert verdict.blocked
        assert "resolve" in verdict.reason

    def test_empty_resolution_blocked(self):
        verdict = classify_url("http://void.example.com/", resolver=fixed_resolver())
        assert verdict.blocked
