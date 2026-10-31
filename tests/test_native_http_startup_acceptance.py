from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
from native_release_http_job import ServiceJob


def readiness(proof):
    job = object.__new__(ServiceJob)
    job.args = SimpleNamespace(startup_timeout=0.01)
    job.workflow = "test-workflow"
    job.release = {"models": {"primary": {}}}
    job.stack = SimpleNamespace(poll=lambda: None)
    job.stack_tail = lambda: ""
    worker = {
        "status": "available",
        "workflow_id": job.workflow,
        "model_cache": {"warmup": {"self_test": proof}},
    }
    http = SimpleNamespace(
        json=lambda path: (
            {"status": "ready"} if path == "/readyz" else {"readiness": {"ready": True}}
        )
    )
    redis = SimpleNamespace(registry=lambda: {"worker": worker})
    return job, http, redis


def test_finite_only_legacy_smoke_cannot_satisfy_generic_numerical_startup_gate():
    job, http, redis = readiness(
        {"status": "passed", "policy": "finite-static-inputs-v1"}
    )
    with pytest.raises(TimeoutError):
        job.ready(http, redis)


def test_numerical_startup_requires_every_fixed_alias():
    job, http, redis = readiness(
        {
            "status": "passed",
            "policy": "expected-fp32-v1",
            "models": [],
            "session_load_count": 0,
        }
    )
    with pytest.raises(TimeoutError):
        job.ready(http, redis)


def test_matching_numerical_startup_alias_inventory_can_be_admitted():
    job, http, redis = readiness(
        {
            "status": "passed",
            "policy": "expected-fp32-v1",
            "models": [{"alias": "primary"}],
            "session_load_count": 1,
        }
    )
    assert job.ready(http, redis)["worker"]["status"] == "available"


def test_managed_release_cannot_use_only_cached_http_readiness():
    job, http, redis = readiness(
        {
            "status": "passed",
            "policy": "expected-fp32-v1",
            "models": [{"alias": "primary"}],
            "session_load_count": 1,
        }
    )
    job.release["management_capabilities"] = ["live-native-readiness-v1"]
    with pytest.raises(TimeoutError):
        job.ready(http, redis)
