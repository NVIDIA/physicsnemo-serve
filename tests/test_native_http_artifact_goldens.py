"""HTTP acceptance must bind artifact bytes to independently authored goldens."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
from native_release_http_job import ServiceJob


def fixture_job(tmp_path, artifact):
    job = object.__new__(ServiceJob)
    job.root, job.workflow = tmp_path, "raw-workflow"
    job.args = SimpleNamespace(
        image="registry.example/native@sha256:" + "c" * 64, request_timeout=1
    )
    job.release = {"source_sha": "a" * 40, "models": {"primary": {}}}
    job.runs, job.results = [], {}
    payload = {
        "values": {
            "artifact": "output.f32",
            "dtype": "float32",
            "shape": [4],
            "byte_order": "little",
            "layout": "contiguous",
            "size_bytes": 16,
        }
    }
    result = {
        "payload": payload,
        "execution": {
            "run_id": "run-1",
            "status": "succeeded",
            "outputs": [artifact],
            "native": {
                "source_sha": "a" * 40,
                "release_image": job.args.image,
                "runtime": "cpp",
                "metadata": {"session_load_count": 1},
            },
        },
    }

    class HTTP:
        def json(self, path, **kwargs):
            if path.endswith("/run"):
                return {
                    "workflow": job.workflow,
                    "pipeline": ["schedule", "execute", "results"],
                    "run_id": "run-1",
                }
            if path.endswith("/status"):
                return {"status": "succeeded"}
            return result

    case = {
        "request": {},
        "expected": payload,
        "expected_artifacts": {"output.f32": {"sha256": "b" * 64, "size_bytes": 16}},
    }
    return job, HTTP(), case


def test_http_rejects_artifact_identity_that_disagrees_with_authored_golden(tmp_path):
    artifact = {"name": "output.f32", "sha256": "d" * 64, "size_bytes": 16}
    job, http, case = fixture_job(tmp_path, artifact)
    with pytest.raises(ValueError, match="authored acceptance"):
        job.request(http, "mismatch", case)
    assert (tmp_path / "mismatch/results.json").is_file()


def test_http_accepts_matching_authored_artifact_identity(tmp_path):
    artifact = {"name": "output.f32", "sha256": "b" * 64, "size_bytes": 16}
    job, http, case = fixture_job(tmp_path, artifact)
    assert job.request(http, "matching", case)[0] == "run-1"
