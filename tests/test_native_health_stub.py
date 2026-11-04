import io
from pathlib import Path
import runpy
from unittest.mock import Mock, patch
import urllib.error

import pytest


@pytest.fixture
def handler():
    with patch("socketserver.ThreadingTCPServer"):
        module = runpy.run_path(
            str(Path(__file__).parents[1] / "scripts/health_stub.py")
        )
    result = module["H"].__new__(module["H"])
    result.path = "/health"
    result.headers = {}
    result.wfile = io.BytesIO()
    result.send_response = Mock()
    result.send_header = Mock()
    result.end_headers = Mock()
    result.send_error = Mock()
    return result


def test_native_provider_health_uses_live_readiness(handler, monkeypatch):
    monkeypatch.setenv("NATIVE_DEPLOYMENT_CONFIG", "/image/deployment.json")
    response = Mock(status=200, headers={})
    response.read.return_value = b'{"ready":true}'
    with patch("urllib.request.urlopen", return_value=response) as fetch:
        handler.do_GET()
    assert fetch.call_args.args[0].full_url == "http://127.0.0.1:8080/readyz"


def test_native_loading_does_not_return_success(handler, monkeypatch):
    monkeypatch.setenv("NATIVE_DEPLOYMENT_CONFIG", "/image/deployment.json")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("starting")):
        handler.do_GET()
    handler.send_response.assert_not_called()
    handler.send_error.assert_called_once_with(503, "loading")


def test_python_loading_health_is_unchanged(handler, monkeypatch):
    monkeypatch.delenv("NATIVE_DEPLOYMENT_CONFIG", raising=False)
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("starting")):
        handler.do_GET()
    handler.send_response.assert_called_once_with(200)
    assert handler.wfile.getvalue() == b'{"status":"loading"}'
