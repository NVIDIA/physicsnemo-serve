from __future__ import annotations

import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qa" / "inference"))

from output_publication_helpers import (  # noqa: E402
    COMPARE_JOB_SCRIPT,
    _compare_job_command_for_comparisons,
    _compare_job_name_for_runs,
    _extract_compare_summary,
    _local_path_for_published_artifact,
    _submit_compare_job,
    compare_publication_with_lustre_job,
    fetch_publication_results,
    load_request_payload,
    load_request_payloads,
    local_artifact_file_map,
    publication_results_payload_from_status,
    submit_workflow,
    wait_for_publication,
)


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")

    def json(self):
        return json.loads(self.content.decode("utf-8"))


class FakeClient:
    base_url = "https://example.test"

    def __init__(self):
        self.urls: list[str] = []

    def get(self, url: str, *, timeout: int):
        self.urls.append(url)
        if "format=zarr_zip" in url:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, mode="w") as archive:
                archive.writestr("zarr.json", b"local-bytes")
            return FakeResponse(200, buffer.getvalue())
        if "artifact=forecast_dataset" in url or "artifact=primary" in url:
            return FakeResponse(404)
        return FakeResponse(500)


class FakeAdapter:
    def result_file_url(self, workflow: str, exec_id: str, artifact: str) -> str:
        return f"/v1/infer/{workflow}/{exec_id}/results?artifact={artifact}"


class ResultsAdapter:
    def submit_url(self, workflow: str) -> str:
        return f"/v1/infer/{workflow}/run"

    def format_submit_body(self, params: dict) -> dict:
        return {"parameters": params}

    def parse_submit_response(self, payload: dict) -> dict:
        return {"execution_id": payload["run_id"]}

    def results_url(self, workflow: str, exec_id: str) -> str:
        return f"/v1/infer/{workflow}/{exec_id}/results"

    def status_url(self, workflow: str, exec_id: str) -> str:
        return f"/v1/infer/{workflow}/{exec_id}/status"

    def parse_status_response(self, payload: dict) -> dict:
        return payload


def _compare_job_namespace() -> dict[str, object]:
    namespace = {"__name__": "publication_compare_test"}
    exec(COMPARE_JOB_SCRIPT, namespace)
    return namespace


def test_submit_workflow_bypasses_retrying_session(monkeypatch):
    class SubmitClient:
        base_url = "https://example.test"
        headers = {"Authorization": "Bearer token"}

        def post(self, *_args, **_kwargs):
            raise AssertionError("retrying session should not be used for submit")

    def fake_post(url: str, *, headers: dict, json: dict, timeout: float):
        assert url.endswith("/v1/infer/demo/run")
        assert headers["Authorization"] == "Bearer token"
        assert json == {"parameters": {"value": 1}}
        assert timeout == 7
        return FakeResponse(202, b'{"run_id":"run-1"}')

    monkeypatch.setenv("QA_PUBLICATION_SUBMIT_REQUEST_TIMEOUT_SECS", "7")
    monkeypatch.setattr("output_publication_helpers.requests.post", fake_post)

    exec_id = submit_workflow(SubmitClient(), ResultsAdapter(), "demo", {"value": 1})

    assert exec_id == "run-1"


def test_submit_workflow_retries_transient_503(monkeypatch):
    calls = []

    class SubmitClient:
        base_url = "https://example.test"
        headers = {"Authorization": "Bearer token"}

    def fake_post(url: str, *, headers: dict, json: dict, timeout: float):
        calls.append((url, headers, json, timeout))
        if len(calls) == 1:
            return FakeResponse(503, b'{"error":"not ready"}')
        return FakeResponse(202, b'{"run_id":"run-2"}')

    monkeypatch.setenv("QA_PUBLICATION_SUBMIT_REQUEST_TIMEOUT_SECS", "5")
    monkeypatch.setenv("QA_PUBLICATION_SUBMIT_ATTEMPTS", "2")
    monkeypatch.setenv("QA_PUBLICATION_SUBMIT_RETRY_INTERVAL_SECS", "10")
    monkeypatch.setattr("output_publication_helpers.requests.post", fake_post)
    monkeypatch.setattr("output_publication_helpers.time.sleep", lambda _seconds: None)

    exec_id = submit_workflow(SubmitClient(), ResultsAdapter(), "demo", {"value": 1})

    assert exec_id == "run-2"
    assert len(calls) == 2


def test_local_artifact_file_map_falls_back_to_primary_for_published_source_name(
    tmp_path,
):
    client = FakeClient()
    files = local_artifact_file_map(
        client,
        FakeAdapter(),
        "deterministic_workflow",
        "run-1",
        {
            "source_artifact": "forecast_dataset",
            "filename": "forecast.zarr",
        },
        tmp_path / "local",
    )

    assert "artifact=forecast_dataset" in client.urls[0]
    assert "artifact=primary" in client.urls[1]
    assert "artifact=forecast_dataset" in client.urls[2]
    assert "format=zarr_zip" in client.urls[2]
    assert files == {
        "zarr.json": {
            "sha256": "57745905a87f748134c5cfd5db849be387ed629c5b8297290f1c0cb18d964016",
            "size": 11,
        }
    }


def test_local_path_for_published_artifact_prefers_named_output():
    path = _local_path_for_published_artifact(
        {
            "execution": {
                "output_path": "/outputs/run-1/fallback.zarr",
                "outputs": [
                    {
                        "name": "forecast_dataset",
                        "storage_path": "/outputs/run-1/forecast.zarr",
                        "primary": True,
                    }
                ],
            }
        },
        {"source_artifact": "forecast_dataset"},
    )

    assert path == "/outputs/run-1/forecast.zarr"


def test_publication_results_payload_from_status_avoids_full_results_fetch():
    payload = publication_results_payload_from_status(
        {
            "run_id": "run-1",
            "workflow": "earth2-ensemble-fanout",
            "status": "running",
            "published_artifacts": [
                {
                    "provider": "s3",
                    "source_artifact": "forecast_dataset",
                    "destination_uri": "s3://bucket/prefix/forecast.zarr",
                    "status": "uploaded",
                }
            ],
            "outputs": [
                {
                    "name": "forecast_dataset",
                    "storage_path": "/outputs/run-1/forecast.zarr",
                    "primary": True,
                }
            ],
        }
    )

    assert payload is not None
    assert payload["execution"]["published_artifacts"][0]["provider"] == "s3"
    assert (
        _local_path_for_published_artifact(
            payload, payload["execution"]["published_artifacts"][0]
        )
        == "/outputs/run-1/forecast.zarr"
    )


def test_publication_results_payload_from_status_accepts_json_strings():
    payload = publication_results_payload_from_status(
        {
            "run_id": "run-1",
            "workflow_id": "demo",
            "status": "running",
            "published_artifacts": json.dumps(
                [
                    {
                        "provider": "s3",
                        "source_artifact": "primary",
                        "destination_uri": "s3://bucket/result.json",
                        "status": "uploaded",
                    }
                ]
            ),
            "output_path": "/outputs/run-1/result.json",
        }
    )

    assert payload is not None
    assert payload["workflow"] == "demo"
    assert payload["execution"]["output_path"] == "/outputs/run-1/result.json"


def test_fetch_publication_results_prefers_status_payload():
    class NoResultsClient:
        base_url = "https://example.test"

        def get(self, *_args, **_kwargs):
            raise AssertionError("full results endpoint should not be fetched")

    runs = [
        {
            "workflow": "demo",
            "exec_id": "run-1",
            "status_payload": {
                "run_id": "run-1",
                "workflow": "demo",
                "status": "completed",
                "published_artifacts": [
                    {
                        "provider": "s3",
                        "source_artifact": "primary",
                        "destination_uri": "s3://bucket/result.json",
                        "status": "uploaded",
                    }
                ],
                "output_path": "/outputs/run-1/result.json",
            },
        }
    ]

    fetch_publication_results(NoResultsClient(), ResultsAdapter(), runs)

    assert runs[0]["published_artifact"]["destination_uri"] == "s3://bucket/result.json"


def test_wait_for_publication_accepts_rust_succeeded_status(monkeypatch):
    class StatusClient:
        base_url = "https://example.test"
        headers = {"Authorization": "Bearer token"}

        def get(self, url: str, *, timeout: int):
            raise AssertionError(
                "retrying session should not be used for status polling"
            )

    def fake_get(url: str, *, headers: dict, timeout: float):
        assert url.endswith("/v1/infer/demo/run-1/status")
        assert headers["Authorization"] == "Bearer token"
        assert timeout == 15
        return FakeResponse(
            200,
            json.dumps(
                {
                    "run_id": "run-1",
                    "workflow": "demo",
                    "status": "succeeded",
                    "output_publication_status": "uploaded",
                }
            ).encode("utf-8"),
        )

    monkeypatch.setenv("QA_PUBLICATION_TIMEOUT_SECS", "1")
    monkeypatch.setattr("output_publication_helpers.requests.get", fake_get)

    payload = wait_for_publication(StatusClient(), ResultsAdapter(), "demo", "run-1")

    assert payload["status"] == "succeeded"


def test_wait_for_publication_tolerates_transient_status_timeout(monkeypatch):
    calls = []

    class StatusClient:
        base_url = "https://example.test"
        headers = {"Authorization": "Bearer token"}

        def get(self, *_args, **_kwargs):
            raise AssertionError(
                "retrying session should not be used for status polling"
            )

    def fake_get(url: str, *, headers: dict, timeout: float):
        calls.append((url, headers, timeout))
        if len(calls) == 1:
            raise requests.ReadTimeout("temporary lepton timeout")
        return FakeResponse(
            200,
            json.dumps(
                {
                    "run_id": "run-1",
                    "workflow": "demo",
                    "status": "completed",
                    "output_publication_status": "uploaded",
                }
            ).encode("utf-8"),
        )

    monkeypatch.setenv("QA_PUBLICATION_TIMEOUT_SECS", "30")
    monkeypatch.setenv("QA_PUBLICATION_POLL_INTERVAL_SECS", "10")
    monkeypatch.setenv("QA_PUBLICATION_STATUS_REQUEST_TIMEOUT_SECS", "3")
    monkeypatch.setattr("output_publication_helpers.requests.get", fake_get)
    monkeypatch.setattr("output_publication_helpers.time.sleep", lambda _seconds: None)

    payload = wait_for_publication(StatusClient(), ResultsAdapter(), "demo", "run-1")

    assert payload["output_publication_status"] == "uploaded"
    assert len(calls) == 2
    assert calls[0][2] == 3


def test_extract_compare_summary_reads_marked_json():
    logs = """
noise
PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_BEGIN
{"matched": true, "local_count": 2}
PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_END
more noise
"""

    assert _extract_compare_summary(logs) == {"matched": True, "local_count": 2}


def test_compare_job_script_writes_summary_on_failure():
    assert "def write_summary" in COMPARE_JOB_SCRIPT
    assert '"error_type": type(error).__name__' in COMPARE_JOB_SCRIPT
    assert "except Exception as error" in COMPARE_JOB_SCRIPT


def test_compare_publication_accepts_successful_job_without_summary(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "output_publication_helpers._submit_compare_job",
        lambda job_name, command: "job-1",
    )
    monkeypatch.setattr(
        "output_publication_helpers._poll_compare_job", lambda job_id: 0
    )
    monkeypatch.setattr(
        "output_publication_helpers._capture_compare_job_logs", lambda job_id: ""
    )
    monkeypatch.setattr(
        "output_publication_helpers._read_compare_summary_with_job",
        lambda job_name, summary_path: None,
    )

    def remove_job(job_id):
        calls.append(("remove", job_id))

    monkeypatch.setattr("output_publication_helpers._remove_compare_job", remove_job)

    summary = compare_publication_with_lustre_job(
        {
            "run_id": "run-1",
            "execution": {
                "outputs": [
                    {
                        "name": "forecast_dataset",
                        "storage_path": "/outputs/run-1/forecast.zarr",
                    }
                ]
            },
        },
        {
            "source_artifact": "forecast_dataset",
            "provider": "s3",
            "destination_uri": "s3://bucket/prefix/forecast.zarr",
        },
    )

    assert summary == {"matched": True, "summary_unavailable": True}
    assert calls == [("remove", "job-1")]


def test_load_request_payloads_expands_all_defaults(monkeypatch):
    monkeypatch.setenv("QA_PUBLICATION_WORKFLOWS", "all")
    monkeypatch.delenv("QA_PUBLICATION_REQUEST_JSON", raising=False)

    requests = load_request_payloads()

    assert len(requests) > 1
    workflows = [workflow for workflow, _payload in requests]
    assert "deterministic_workflow" in workflows
    assert "earth2-ensemble-fanout" in workflows
    assert "example_user_workflow" not in workflows
    payloads = dict(requests)
    assert payloads["earth2-ensemble-fanout"]["batch_size"] == 16
    assert payloads["earth2-ensemble-fanout"]["nensemble"] == 64


def test_load_request_payloads_uses_direct_payload_for_one_plural_selected_workflow(
    monkeypatch, tmp_path
):
    request_file = tmp_path / "request.json"
    payload = {"model": "custom", "nsteps": 7}
    request_file.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("QA_PUBLICATION_WORKFLOWS", "earth2-deterministic")
    monkeypatch.setenv("QA_PUBLICATION_REQUEST_JSON", str(request_file))

    assert load_request_payloads() == [("earth2-deterministic", payload)]


def test_load_request_payloads_preserves_multi_workflow_payload_map(
    monkeypatch, tmp_path
):
    request_file = tmp_path / "requests.json"
    payloads = {
        "earth2-deterministic": {"model": "dlwp", "nsteps": 2},
        "earth2-ensemble": {"model": "fcn", "nensemble": 3},
    }
    request_file.write_text(json.dumps(payloads), encoding="utf-8")
    monkeypatch.setenv(
        "QA_PUBLICATION_WORKFLOWS", "earth2-deterministic,earth2-ensemble"
    )
    monkeypatch.setenv("QA_PUBLICATION_REQUEST_JSON", str(request_file))

    assert load_request_payloads() == list(payloads.items())


def test_load_request_payloads_rejects_direct_payload_for_multiple_workflows(
    monkeypatch, tmp_path
):
    request_file = tmp_path / "request.json"
    request_file.write_text(
        json.dumps({"model": "dlwp", "nsteps": 2}), encoding="utf-8"
    )
    monkeypatch.setenv(
        "QA_PUBLICATION_WORKFLOWS", "earth2-deterministic,earth2-ensemble"
    )
    monkeypatch.setenv("QA_PUBLICATION_REQUEST_JSON", str(request_file))

    with pytest.raises(ValueError, match="workflow-to-payload map"):
        load_request_payloads()


def test_load_request_payload_uses_workflow_specific_default(monkeypatch):
    monkeypatch.setenv("QA_PUBLICATION_WORKFLOW", "deterministic_fcn_workflow")
    monkeypatch.delenv("QA_PUBLICATION_REQUEST_JSON", raising=False)

    workflow, payload = load_request_payload()

    assert workflow == "deterministic_fcn_workflow"
    assert payload["data_source"] == "gfs"
    assert payload["forecast_times"] == ["2024-01-01T00:00:00"]
    assert "model" not in payload


def test_compare_job_command_encodes_multiple_comparisons(monkeypatch):
    monkeypatch.setenv("QA_PUBLICATION_S3_ENDPOINT", "https://object.example")
    monkeypatch.setenv("QA_PUBLICATION_S3_REGION", "region-1")

    command = _compare_job_command_for_comparisons(
        comparisons=[
            {
                "label": "workflow-a:run-a",
                "local_path": "/outputs/run-a/forecast.zarr",
                "destination_uri": "s3://bucket/prefix-a/forecast.zarr",
            },
            {
                "label": "workflow-b:run-b",
                "local_path": "/outputs/run-b/forecast.zarr",
                "destination_uri": "s3://bucket/prefix-b/forecast.zarr",
            },
        ],
        summary_path="/outputs/publication-compare/summary.json",
    )

    assert "--comparisons-b64" in command
    assert "--endpoint-url https://object.example" in command
    assert "--region region-1" in command
    assert "import base64" in COMPARE_JOB_SCRIPT


def test_compare_job_name_uses_provider_prefix():
    name = _compare_job_name_for_runs(
        [
            {
                "exec_id": "run-1234",
                "published_artifact": {
                    "provider": "azure",
                },
            }
        ]
    )

    assert name == "ff-pubcmp-azure-run-1r"


def test_submit_compare_job_forwards_azure_credentials(monkeypatch):
    captured = {}

    monkeypatch.setenv("QA_PUBLICATION_NFS_PATH", "/PhysicsNeMo/platform/test")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "account")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "key")
    monkeypatch.setenv("AZURE_STORAGE_ACCESS_KEY", "access-key")

    def fake_run(args):
        captured["args"] = args
        return 0, "ID: job-123\n"

    monkeypatch.setattr("output_publication_helpers._run_lepton_command", fake_run)

    assert _submit_compare_job("job-name", "python3 -c pass") == "job-123"
    args = captured["args"]
    assert "--env" in args
    assert "AZURE_STORAGE_ACCOUNT_NAME=account" in args
    assert "AZURE_STORAGE_ACCOUNT_KEY=key" in args
    assert "AZURE_STORAGE_ACCESS_KEY=access-key" in args


def test_compare_job_s3_manifest_only_empty_directory_is_empty(monkeypatch):
    manifest_key = "prefix/empty/_physicsnemo_serve_publish_manifest.json"

    class Body:
        def read(self):
            return b'{"object_count": 0, "total_bytes": 0}'

    class Client:
        def get_paginator(self, _operation):
            return types.SimpleNamespace(
                paginate=lambda **_kwargs: [{"Contents": [{"Key": manifest_key}]}]
            )

        def get_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            if Key == manifest_key:
                return {"Body": Body()}
            raise AssertionError(f"must not download exact object {Key!r}")

    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *_args, **_kwargs: Client()
    botocore = types.ModuleType("botocore")
    botocore.__path__ = []
    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)

    namespace = _compare_job_namespace()
    s3_file_map = namespace["s3_file_map"]

    assert s3_file_map("s3://bucket/prefix/empty", None, None) == {}


def test_compare_job_s3_only_excludes_root_manifest(monkeypatch):
    root_manifest_key = "prefix/output/_physicsnemo_serve_publish_manifest.json"
    nested_key = "prefix/output/sub/_physicsnemo_serve_publish_manifest.json"
    nested_body = b"legitimate nested file"

    class Body:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

    class Client:
        def get_paginator(self, _operation):
            return types.SimpleNamespace(
                paginate=lambda **_kwargs: [
                    {"Contents": [{"Key": root_manifest_key}, {"Key": nested_key}]}
                ]
            )

        def get_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            if Key == root_manifest_key:
                return {"Body": Body(b'{"object_count": 1}')}
            if Key == nested_key:
                return {"Body": Body(nested_body)}
            raise AssertionError(f"unexpected object download {Key!r}")

    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *_args, **_kwargs: Client()
    botocore = types.ModuleType("botocore")
    botocore.__path__ = []
    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)

    namespace = _compare_job_namespace()
    s3_file_map = namespace["s3_file_map"]

    assert s3_file_map("s3://bucket/prefix/output", None, None) == {
        "sub/_physicsnemo_serve_publish_manifest.json": {
            "sha256": hashlib.sha256(nested_body).hexdigest(),
            "size": len(nested_body),
        }
    }


def test_compare_job_azure_manifest_only_empty_directory_is_empty(monkeypatch):
    manifest_name = "prefix/empty/_physicsnemo_serve_publish_manifest.json"

    class Download:
        def readall(self):
            return b'{"object_count": 0, "total_bytes": 0}'

    class ContainerClient:
        def list_blobs(self, *, name_starts_with):
            assert name_starts_with == "prefix/empty/"
            return [types.SimpleNamespace(name=manifest_name)]

        def download_blob(self, name):
            if name == manifest_name:
                return Download()
            raise AssertionError(f"must not download exact blob {name!r}")

    container_client = ContainerClient()

    class BlobServiceClient:
        def __init__(self, *, account_url, credential):
            assert account_url == "https://account.blob.core.windows.net"
            assert credential is not None

        def get_container_client(self, container):
            assert container == "container"
            return container_client

    azure = types.ModuleType("azure")
    azure.__path__ = []
    azure_identity = types.ModuleType("azure.identity")
    azure_identity.DefaultAzureCredential = lambda: object()
    azure_storage = types.ModuleType("azure.storage")
    azure_storage.__path__ = []
    azure_blob = types.ModuleType("azure.storage.blob")
    azure_blob.BlobServiceClient = BlobServiceClient
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.identity", azure_identity)
    monkeypatch.setitem(sys.modules, "azure.storage", azure_storage)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", azure_blob)
    for name in (
        "AZURE_STORAGE_SAS_TOKEN",
        "AZURE_STORAGE_ACCESS_KEY",
        "AZURE_STORAGE_ACCOUNT_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    namespace = _compare_job_namespace()
    azure_file_map = namespace["azure_file_map"]

    assert (
        azure_file_map("https://account.blob.core.windows.net/container/prefix/empty")
        == {}
    )


def test_compare_job_azure_only_excludes_root_manifest(monkeypatch):
    root_manifest_name = "prefix/output/_physicsnemo_serve_publish_manifest.json"
    nested_name = "prefix/output/sub/_physicsnemo_serve_publish_manifest.json"
    nested_body = b"legitimate nested file"

    class Download:
        def __init__(self, body):
            self.body = body

        def readall(self):
            return self.body

    class ContainerClient:
        def list_blobs(self, *, name_starts_with):
            assert name_starts_with == "prefix/output/"
            return [
                types.SimpleNamespace(name=root_manifest_name),
                types.SimpleNamespace(name=nested_name),
            ]

        def download_blob(self, name):
            if name == root_manifest_name:
                return Download(b'{"object_count": 1}')
            if name == nested_name:
                return Download(nested_body)
            raise AssertionError(f"unexpected blob download {name!r}")

    container_client = ContainerClient()

    class BlobServiceClient:
        def __init__(self, *, account_url, credential):
            assert account_url == "https://account.blob.core.windows.net"
            assert credential is not None

        def get_container_client(self, container):
            assert container == "container"
            return container_client

    azure = types.ModuleType("azure")
    azure.__path__ = []
    azure_identity = types.ModuleType("azure.identity")
    azure_identity.DefaultAzureCredential = lambda: object()
    azure_storage = types.ModuleType("azure.storage")
    azure_storage.__path__ = []
    azure_blob = types.ModuleType("azure.storage.blob")
    azure_blob.BlobServiceClient = BlobServiceClient
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.identity", azure_identity)
    monkeypatch.setitem(sys.modules, "azure.storage", azure_storage)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", azure_blob)
    for name in (
        "AZURE_STORAGE_SAS_TOKEN",
        "AZURE_STORAGE_ACCESS_KEY",
        "AZURE_STORAGE_ACCOUNT_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    namespace = _compare_job_namespace()
    azure_file_map = namespace["azure_file_map"]

    assert azure_file_map(
        "https://account.blob.core.windows.net/container/prefix/output"
    ) == {
        "sub/_physicsnemo_serve_publish_manifest.json": {
            "sha256": hashlib.sha256(nested_body).hexdigest(),
            "size": len(nested_body),
        }
    }
