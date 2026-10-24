"""Idle keep-alive recovery with real botocore and a local HTTP server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ReadTimeoutError

from mnemoai.models.aws_connections import recycle_idle_connections
from mnemoai.models.chat_models.bedrock_stream_compat import harden_converse_stream


def test_idle_hook_is_idempotent_and_preserves_recent_pools():
    callbacks = []
    client = SimpleNamespace(
        meta=SimpleNamespace(events=SimpleNamespace(register=lambda *a, **kw: callbacks.append(a[1]))),
        close=Mock(),
    )
    clock = [0.0]
    assert recycle_idle_connections(client, clock=lambda: clock[0]) is client
    recycle_idle_connections(client)
    assert len(callbacks) == 2  # pool retirement and file-only dispatch diagnostics
    callbacks[0]()
    clock[0] = 30
    callbacks[0]()
    client.close.assert_not_called()
    clock[0] = 100
    callbacks[0]()
    client.close.assert_called_once()
    callbacks[0]()
    client.close.assert_called_once()


def test_unknown_clients_and_failed_pool_close_do_not_change_request_behavior():
    assert recycle_idle_connections(None) is None
    custom = object()
    assert recycle_idle_connections(custom) is custom
    callbacks = []
    client = SimpleNamespace(
        meta=SimpleNamespace(events=SimpleNamespace(register=lambda *a, **kw: callbacks.append(a[1]))),
        close=Mock(side_effect=RuntimeError("custom transport")),
    )
    clock = [0]
    recycle_idle_connections(client, clock=lambda: clock[0])
    callbacks[0]()
    clock[0] = 100
    callbacks[0]()  # failure is diagnostic only, never an extra failed request


@pytest.fixture
def idle_endpoint(request):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.requests = 0
            with self.server.lock:
                self.server.connections += 1
                self.connection_number = self.server.connections

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.requests += 1
            self.server.requests.append(self.connection_number)
            if self.server.block_first and self.connection_number == 1:
                self.server.first_entered.set()
                self.server.release_first.wait(3)
            if self.requests > 1:
                # Simulate an idle connection blackholed by an intermediary.
                # The test releases this handler; no dangling server thread.
                self.server.release.wait(3)
                self.close_connection = True
                return
            body = json.dumps({
                "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.connections, server.requests = 0, []
    server.lock, server.release = threading.Lock(), threading.Event()
    server.block_first = False
    server.first_entered, server.release_first = threading.Event(), threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = boto3.client(
        "bedrock-runtime", region_name="us-east-1",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        aws_access_key_id="test", aws_secret_access_key="test",
        config=Config(read_timeout=getattr(request, "param", 0.3), connect_timeout=1, retries={"max_attempts": 0}),
    )
    try:
        yield client, server
    finally:
        server.release.set()
        server.release_first.set()
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(client):
    return client.converse(modelId="test-model", messages=[
        {"role": "user", "content": [{"text": "temporary connectivity check"}]},
    ])


def test_unprotected_idle_pool_reproduces_stalled_next_request(idle_endpoint):
    client, server = idle_endpoint
    assert request(client)["stopReason"] == "end_turn"
    with pytest.raises(ReadTimeoutError):
        request(client)
    assert server.requests == [1, 1]


def test_idle_recovery_uses_new_socket_without_retry_or_reducing_timeouts(idle_endpoint):
    client, server = idle_endpoint
    clock = [0.0]
    recycle_idle_connections(client, clock=lambda: clock[0])
    # The existing model hardener also keeps the guard when wrapping the client.
    model = harden_converse_stream(SimpleNamespace(client=client))
    before_timeout = client.meta.config.read_timeout
    assert request(model.client)["stopReason"] == "end_turn"
    clock[0] = 601  # simulate ten minutes idle without sleeping
    assert request(model.client)["stopReason"] == "end_turn"
    assert server.requests == [1, 2]
    assert client.meta.config.read_timeout == before_timeout


@pytest.mark.parametrize("idle_endpoint", [2.0], indirect=True)
def test_recycling_does_not_interrupt_an_inflight_request(idle_endpoint):
    client, server = idle_endpoint
    clock = [0.0]
    recycle_idle_connections(client, clock=lambda: clock[0])
    server.block_first = True
    results, errors = [], []
    def first_request():
        try:
            results.append(request(client))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=first_request)
    thread.start()
    try:
        assert server.first_entered.wait(2)
        clock[0] = 601
        second = request(client)
    finally:
        server.release_first.set()
        thread.join(timeout=2)
    assert not errors and results[0]["stopReason"] == second["stopReason"] == "end_turn"
    assert server.requests == [1, 2]
