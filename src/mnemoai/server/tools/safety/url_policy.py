"""SSRF guard for the URL-fetching tools.

``web_crawler`` validated only the scheme, which leaves the whole internal
network reachable from a prompt. That matters more here than in a normal HTTP
client for two reasons:

* the *fetched page text becomes model instructions*, so a crawl of
  ``http://169.254.169.254/latest/meta-data/iam/security-credentials/`` doesn't
  just read cloud instance credentials, it feeds them to the model;
* the URL often comes from the model itself (or from a search result), not from
  the user, so "the user wouldn't type that" is not a control.

The check resolves the hostname and refuses any address that isn't publicly
routable: loopback, private (RFC1918), link-local (incl. the cloud metadata
endpoint at 169.254.169.254), CGNAT, multicast, reserved, and the IPv6
equivalents. Resolution happens BEFORE the fetch, and every address a name maps
to must pass -- a name resolving to both a public and a private address is
refused, since we can't control which one the browser will connect to.

Known, accepted limitation: this is a pre-flight check, so it does not close a
DNS-rebinding race (the name could re-resolve between this check and the
browser's own lookup) and it does not follow redirects -- crawl4ai does its own
fetching, so a public URL that 302s to a private one is not caught here. Closing
those requires a proxy at the socket layer; this raises the floor from "no
control at all" without pretending to be that.
"""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

# Hostnames that resolve locally by convention. Checked as a fast path (and to
# give a clearer message) before DNS resolution.
_LOCAL_HOSTNAMES: frozenset[str] = frozenset(
    {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


@dataclass(frozen=True)
class UrlPolicyResult:
    """Outcome of classifying a URL for fetching.

    Attributes:
        blocked: True if the URL must not be fetched.
        reason: Human-readable explanation (empty when not blocked).
        host: The hostname extracted from the URL (empty if unparseable).
        addresses: The resolved IP addresses considered (empty if unresolved).
    """

    blocked: bool
    reason: str = ""
    host: str = ""
    addresses: tuple[str, ...] = ()


def _is_public_address(ip: ipaddress._BaseAddress) -> bool:
    """True if ``ip`` is a publicly routable unicast address.

    Deliberately allowlist-shaped: anything not positively known to be public is
    rejected, so a category we didn't think of fails closed.

    ``is_global`` is the primary test because the individual category flags are
    NOT sufficient and vary by Python version -- CGNAT (100.64.0.0/10) reports
    ``is_private=False`` and ``is_reserved=False`` on 3.13 while still being
    unroutable. The explicit flags are kept as a second, self-documenting layer.
    """
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    # IPv4-mapped/compatible IPv6 (::ffff:127.0.0.1) would otherwise sneak a
    # loopback address past the checks above.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return _is_public_address(mapped)
    sixtofour = getattr(ip, "sixtofour", None)
    if sixtofour is not None:
        return _is_public_address(sixtofour)
    return bool(ip.is_global)


def classify_url(url: str, resolver=None) -> UrlPolicyResult:
    """Classify a URL as safe or forbidden to fetch.

    Args:
        url: The absolute URL to check.
        resolver: Optional callable taking a hostname and returning a list of IP
            strings, injected by tests. Defaults to DNS via
            ``socket.getaddrinfo``.

    Returns:
        A :class:`UrlPolicyResult`; ``blocked`` is True for non-HTTP(S) schemes,
        unparseable URLs, unresolvable hosts, and any host that resolves to a
        non-public address.
    """
    if not url or not url.strip():
        return UrlPolicyResult(blocked=True, reason="No URL provided.")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return UrlPolicyResult(
            blocked=True,
            reason=(
                f"Refusing to fetch a '{parsed.scheme}' URL — only http and "
                f"https are allowed."
            ),
        )

    host = (parsed.hostname or "").strip()
    if not host:
        return UrlPolicyResult(
            blocked=True, reason="Could not extract a hostname from the URL."
        )

    if host.lower() in _LOCAL_HOSTNAMES:
        return UrlPolicyResult(
            blocked=True,
            reason=(
                "Refusing to fetch from localhost. This tool reaches the public "
                "web; a local address would expose services on this machine to "
                "the model."
            ),
            host=host,
        )

    # A literal IP in the URL still has to pass, and needs no DNS lookup.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        addresses = [str(literal)]
    else:
        try:
            resolve = resolver or _resolve
            addresses = resolve(host)
        except Exception as e:
            return UrlPolicyResult(
                blocked=True,
                reason=f"Could not resolve host '{host}': {e}",
                host=host,
            )
        if not addresses:
            return UrlPolicyResult(
                blocked=True,
                reason=f"Host '{host}' did not resolve to any address.",
                host=host,
            )

    for raw in addresses:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return UrlPolicyResult(
                blocked=True,
                reason=f"Host '{host}' resolved to an unparseable address: {raw}",
                host=host,
                addresses=tuple(addresses),
            )
        if not _is_public_address(ip):
            # Name the metadata endpoint explicitly: it's the case with real
            # blast radius, and a generic message makes it look like a bug.
            extra = (
                " That is the cloud instance-metadata endpoint, which serves "
                "IAM credentials."
                if raw.startswith("169.254.169.254")
                else ""
            )
            return UrlPolicyResult(
                blocked=True,
                reason=(
                    f"Refusing to fetch '{host}': it resolves to the "
                    f"non-public address {raw}.{extra} This tool only reaches "
                    f"the public internet — page content becomes model input, "
                    f"so internal endpoints are blocked."
                ),
                host=host,
                addresses=tuple(addresses),
            )

    return UrlPolicyResult(blocked=False, host=host, addresses=tuple(addresses))


def _resolve(host: str) -> list[str]:
    """Resolve ``host`` to every A/AAAA address DNS returns."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    # sockaddr[0] is the address for both AF_INET and AF_INET6.
    return sorted({info[4][0].split("%")[0] for info in infos})
