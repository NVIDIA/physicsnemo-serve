# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import os
import pathlib
import pprint
import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CappedRetry(Retry):
    """Retry with a lower backoff ceiling to avoid 2-minute waits."""

    BACKOFF_MAX = 10


logger = logging.getLogger(__name__)


def log_http(response: requests.Response, *args, **kwargs):
    MAX_BODY = 10000

    def short(data: str) -> str:
        if len(data) <= MAX_BODY:
            return data
        return data[:MAX_BODY] + f"... <truncated {len(data) - MAX_BODY} chars>"

    def format_json(data) -> str:
        try:
            return pprint.pformat(data, width=100, compact=False)
        except Exception:
            return "<failed to serialize JSON>"

    request = response.request
    logger.debug(f"[REQUEST] {request.method} {request.url}\nPayload: {request.body}")

    try:
        body = format_json(response.json())
    except Exception:
        body = short(response.text)

    logger.debug(
        f"[RESPONSE] {request.method} {request.url} -> {response.status_code}\nBody:\n{body}"
    )


@pytest.fixture
def user_token(request):
    token = request.config.getoption("--token") or os.environ.get(
        "LEPTON_ENDPOINT_TOKEN", ""
    )
    if not token:
        raise pytest.UsageError(
            "LEPTON_ENDPOINT_TOKEN not set (use --token or LEPTON_ENDPOINT_TOKEN env var)"
        )
    return token


# These retries and timeouts are only for curl, to get past Lepton's TLS issues
# and cold-start delays when a model is loading for the first time.
MAX_RETRIES = 10
REQUEST_TIMEOUT = 10  # seconds (covers both connect and read)

# Make urllib3 retries visible in logs
logging.getLogger("urllib3.util.retry").setLevel(logging.DEBUG)


class TimeoutHTTPAdapter(HTTPAdapter):
    """HTTPAdapter with a default timeout and retry logging."""

    def __init__(self, *args, timeout=None, **kwargs):
        self.timeout = timeout
        self._attempt_count = {}
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout
        url = request.url
        self._attempt_count[url] = self._attempt_count.get(url, 0) + 1
        attempt = self._attempt_count[url]
        if attempt > 1:
            logger.warning(f"[RETRY] attempt #{attempt} for {request.method} {url}")
        else:
            logger.info(f"[REQUEST] {request.method} {url} (timeout={self.timeout})")
        try:
            response = super().send(request, **kwargs)
            self._attempt_count.pop(url, None)
            return response
        except Exception as e:
            logger.warning(
                f"[TIMEOUT/ERROR] {request.method} {url} attempt #{attempt}: {type(e).__name__}: {e}"
            )
            raise


@pytest.fixture
def client(user_token, base_url):
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {user_token}"
    session.hooks["response"].append(log_http)
    session.base_url = base_url

    retry = CappedRetry(
        total=MAX_RETRIES,
        backoff_factor=0.3,
        status_forcelist=[502, 503, 504],
        allowed_methods=None,  # retry all HTTP methods including POST
    )
    adapter = TimeoutHTTPAdapter(max_retries=retry, timeout=REQUEST_TIMEOUT)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    logger.info(f"Using {session.base_url} (max_retries={MAX_RETRIES})")
    return session


@pytest.fixture
def adapter(request):
    from service_adapter import get_adapter

    return get_adapter(request.config.getoption("--service"))


def pytest_addoption(parser):
    parser.addoption(
        "--urls",
        action="store",
        default=os.getenv("BASE_URLS", ""),
        help="Comma-separated base URLs (or BASE_URLS env var).",
    )
    parser.addoption(
        "--token",
        action="store",
        default=os.getenv("LEPTON_ENDPOINT_TOKEN", ""),
        help="Bearer token for service auth (or LEPTON_ENDPOINT_TOKEN env var).",
    )
    parser.addoption(
        "--service",
        action="store",
        default=os.getenv("SERVICE_TYPE", "python"),
        choices=["python", "rust"],
        help="Service backend type: 'python' or 'rust' (or SERVICE_TYPE env var).",
    )


def get_base_urls(config):
    raw = (config.getoption("--urls") or os.environ.get("BASE_URLS", "")).strip()
    return [u.strip() for u in raw.split(",") if u.strip()]


@pytest.fixture
def base_url(request):
    base_urls = get_base_urls(request.config)
    workerid = getattr(request.config, "workerinput", {}).get("workerid", "master")
    if workerid == "master":
        return base_urls[0]
    idx = int(workerid.replace("gw", ""))
    return base_urls[idx % len(base_urls)]


def pytest_collection_modifyitems(config, items):
    service = config.getoption("--service")
    if service == "rust":
        return
    skip_rust = pytest.mark.skip(reason="rust_only test (--service is not rust)")
    for item in items:
        if "rust_only" in item.keywords:
            item.add_marker(skip_rust)


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: fast sanity check for the test harness")
    config.addinivalue_line(
        "markers", "cicd: CI/CD pipeline suite (deploy-test-teardown)"
    )
    config.addinivalue_line(
        "markers",
        "basic: parameterized QA suite (thorough per-workflow coverage)",
    )
    config.addinivalue_line(
        "markers",
        "negative: negative/invalid-input tests (expect 422 errors)",
    )
    config.addinivalue_line(
        "markers",
        "rust_only: tests that only apply to the Rust/physicsnemo-serve service",
    )
    config.addinivalue_line(
        "markers", "multigpu: tests that require an endpoint with multiple visible GPUs"
    )
    config.addinivalue_line(
        "markers", "stress: load/stability tests that intentionally sustain concurrency"
    )

    pathlib.Path("reports").mkdir(exist_ok=True)
    pathlib.Path("logs").mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not config.option.log_file:
        config.option.log_file = f"logs/pytest_{ts}.log"

    if hasattr(config.option, "htmlpath") and not config.option.htmlpath:
        config.option.htmlpath = f"reports/report_{ts}.html"

    urls = get_base_urls(config)
    num_proc = getattr(config.option, "numprocesses", None) or 0
    if len(urls) < num_proc:
        raise pytest.UsageError(
            f"Only {len(urls)} URL provided, but -n {num_proc} was requested. "
            f"Remove -n or provide more URLs."
        )
