"""Retire idle AWS HTTP pools before reusing potentially stale connections."""

import threading
import time

from mnemoai.utils.logger import logger

# urllib3's pools have no idle expiry. Proxies/NATs can silently forget a
# connection while it remains locally open. No heartbeat or model call is needed:
# retire idle pools immediately before the next real request instead.
IDLE_POOL_SECONDS = 60.0
_GUARD_ATTRIBUTE = "_mnemoai_idle_pool_guard"


class _IdlePoolGuard:
    def __init__(self, client, idle_seconds, clock):
        self.client = client
        self.idle_seconds = idle_seconds
        self.clock = clock
        self.last_request = None
        self.lock = threading.Lock()

    def before_request(self, **kwargs):
        logger.info("AWS request preparation", extra={"console": False})
        now = self.clock()
        with self.lock:
            previous = self.last_request
            self.last_request = now
            if previous is None or now - previous < self.idle_seconds:
                return
            try:
                # Botocore's public close() clears its urllib3 pools. PoolManager
                # documents that in-flight connections are unaffected; they simply
                # won't be reused when returned. The client and credentials remain.
                self.client.close()
                logger.debug("Retired idle AWS connection pools before a new request")
            except Exception:
                # An unfamiliar/custom transport retains its existing behavior.
                logger.debug("Could not retire idle AWS connection pools", exc_info=True)

    def before_send(self, **kwargs):
        logger.info("AWS HTTP request dispatch", extra={"console": False})


def recycle_idle_connections(client, *, idle_seconds=IDLE_POOL_SECONDS, clock=None):
    """Install one pool-retirement hook on a long-lived botocore-compatible client.

    Covers both streaming and ordinary calls without shortening the legitimate
    first-token/request timeout, changing retries, or sending extra requests.
    """
    if client is None or isinstance(getattr(client, _GUARD_ATTRIBUTE, None), _IdlePoolGuard):
        return client
    try:
        register = client.meta.events.register
        if not callable(register) or not callable(client.close):
            return client
        guard = _IdlePoolGuard(client, idle_seconds, clock or time.monotonic)
        register("before-call.*.*", guard.before_request, unique_id="mnemoai-idle-pool")
        register("before-send.*.*", guard.before_send, unique_id="mnemoai-request-stage")
        setattr(client, _GUARD_ATTRIBUTE, guard)
    except Exception:
        logger.debug("AWS client does not support idle-pool retirement", exc_info=True)
    return client
