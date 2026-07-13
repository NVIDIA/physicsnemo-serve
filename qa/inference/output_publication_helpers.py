from __future__ import annotations

import base64
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from helpers import RequestStatus

REQUEST_TIMEOUT = 60
DEFAULT_COMPARE_IMAGE_NAME = "your-registry.example.com/your-org/physicsnemo-serve"
DEFAULT_COMPARE_NODE_GROUP = "your-node-group"
DEFAULT_COMPARE_RESOURCE_SHAPE = "cpu.large"
DEFAULT_COMPARE_PULL_SECRET = "your-pull-secret"
DEFAULT_COMPARE_MOUNT_TARGET = "/outputs"
DEFAULT_COMPARE_LUSTRE_STORAGE = "lustre"
COMPARE_JOB_TIMEOUT = 1800
COMPARE_JOB_POLL_INTERVAL = 15
COMPARE_JOB_CREDENTIAL_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_STORAGE_ACCOUNT",
    "AZURE_STORAGE_ACCOUNT_NAME",
    "AZURE_STORAGE_ACCOUNT_KEY",
    "AZURE_STORAGE_ACCESS_KEY",
    "AZURE_STORAGE_SAS_TOKEN",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
)

DEFAULT_PUBLICATION_REQUESTS: dict[str, dict] = {
    "deterministic_workflow": {
        "model_type": "fcn",
        "data_source": "gfs",
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 1,
        "output_format": "zarr",
        "create_plots": False,
    },
    "deterministic_fcn_workflow": {
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 1,
        "data_source": "gfs",
        "output_format": "zarr",
        "create_plots": False,
    },
    "diagnostic_workflow": {
        "data_source": "gfs",
        "forecast_times": ["2024-01-01T00:00:00"],
        "prognostic_model_type": "fcn",
        "diagnostic_model_type": "precipitation_afno",
        "nsteps": 1,
        "output_format": "zarr",
        "create_plots": False,
    },
    "ensemble_workflow": {
        "forecast_times": ["2024-01-01T00:00:00"],
        "nsteps": 1,
        "nensemble": 2,
        "noise_amplitude": 0.05,
        "model_type": "fcn",
        "data_source": "gfs",
        "output_format": "zarr",
        "create_plots": False,
    },
    "deterministic_earth2_workflow": {
        "start_time": ["2024-01-01T00:00:00"],
        "num_steps": 1,
    },
    "stormcast_fcn3_workflow": {
        "start_time": "2024-01-01T00:00:00",
        "num_hours": 6,
        "run_stormcast": True,
    },
    "earth2-deterministic": {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 1,
    },
    "earth2-deterministic-batch": {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 1,
    },
    "earth2-ensemble": {
        "model": "dlwp",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 1,
        "nensemble": 2,
        "perturbation": "gaussian",
        "noise_amplitude": 0.05,
    },
    "earth2-ensemble-fanout": {
        "model": "fcn",
        "start_time": "2024-01-01T00:00:00",
        "nsteps": 4,
        "nensemble": 64,
        "batch_size": 16,
        "max_in_flight": 4,
        "perturbation": "gaussian",
        "noise_amplitude": 0.15,
        "seed_base": 1000,
        "perturbation_materialization_mode": "scheduled_gpu",
    },
}


COMPARE_JOB_SCRIPT = r"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


def write_summary(path: str, summary: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def local_file_map(root: Path) -> dict[str, dict[str, object]]:
    if root.is_file():
        return {root.name: file_fingerprint(root)}
    if not root.is_dir():
        raise FileNotFoundError(f"local artifact does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): file_fingerprint(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def s3_file_map(destination_uri: str, endpoint_url: str | None, region: str | None):
    import boto3
    from botocore.config import Config

    parsed = urlparse(destination_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 destination URI: {destination_uri}")
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        region_name=region or None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    out: dict[str, dict[str, object]] = {}
    manifest_key = None
    expected_manifest_key = (
        f"{prefix.rstrip('/')}/_physicsnemo_serve_publish_manifest.json"
    )
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            if key == expected_manifest_key:
                manifest_key = key
                continue
            rel = key.removeprefix(f"{prefix.rstrip('/')}/")
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            digest = hashlib.sha256(body).hexdigest()
            out[rel] = {"sha256": digest, "size": len(body)}
    if not out and manifest_key:
        manifest_body = client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
        if json.loads(manifest_body).get("object_count") == 0:
            return out
    if not out:
        body = client.get_object(Bucket=bucket, Key=prefix)["Body"].read()
        out[Path(prefix).name] = {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}
    return out


def azure_file_map(destination_uri: str):
    import os
    import subprocess
    import sys

    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "azure-identity",
                "azure-storage-blob",
            ]
        )
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

    parsed = urlparse(destination_uri)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"invalid Azure destination URI: {destination_uri}")
    parts = parsed.path.strip("/").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Azure destination URI must include container and blob: {destination_uri}")
    container, prefix = parts
    credential = (
        os.environ.get("AZURE_STORAGE_SAS_TOKEN")
        or os.environ.get("AZURE_STORAGE_ACCESS_KEY")
        or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        or DefaultAzureCredential()
    )
    service = BlobServiceClient(
        account_url=f"{parsed.scheme}://{parsed.netloc}",
        credential=credential,
    )
    container_client = service.get_container_client(container)
    out: dict[str, dict[str, object]] = {}
    manifest_name = None
    expected_manifest_name = (
        f"{prefix.rstrip('/')}/_physicsnemo_serve_publish_manifest.json"
    )
    for blob in container_client.list_blobs(name_starts_with=f"{prefix.rstrip('/')}/"):
        name = blob.name
        if name.endswith("/"):
            continue
        if name == expected_manifest_name:
            manifest_name = name
            continue
        rel = name.removeprefix(f"{prefix.rstrip('/')}/")
        body = container_client.download_blob(name).readall()
        digest = hashlib.sha256(body).hexdigest()
        out[rel] = {"sha256": digest, "size": len(body)}
    if not out and manifest_name:
        manifest_body = container_client.download_blob(manifest_name).readall()
        if json.loads(manifest_body).get("object_count") == 0:
            return out
    if not out:
        body = container_client.download_blob(prefix).readall()
        out[Path(prefix).name] = {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--destination-uri", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--comparisons-b64", default="")
    parser.add_argument("--endpoint-url", default="")
    parser.add_argument("--region", default="")
    args = parser.parse_args()

    if args.comparisons_b64:
        comparisons = json.loads(base64.b64decode(args.comparisons_b64).decode("utf-8"))
    else:
        comparisons = [{"local_path": args.local_path, "destination_uri": args.destination_uri}]

    try:
        comparison_summaries = []
        all_matched = True
        for comparison in comparisons:
            local = local_file_map(Path(comparison["local_path"]))
            destination_uri = comparison["destination_uri"]
            if destination_uri.startswith("s3://"):
                remote = s3_file_map(destination_uri, args.endpoint_url, args.region)
            else:
                remote = azure_file_map(destination_uri)
            matched = local == remote
            all_matched = all_matched and matched
            comparison_summaries.append({
                "label": comparison.get("label"),
                "local_path": comparison["local_path"],
                "destination_uri": comparison["destination_uri"],
                "local_count": len(local),
                "remote_count": len(remote),
                "local_total_bytes": sum(int(item["size"]) for item in local.values()),
                "remote_total_bytes": sum(int(item["size"]) for item in remote.values()),
                "matched": matched,
                "local_only": sorted(set(local) - set(remote)),
                "remote_only": sorted(set(remote) - set(local)),
                "mismatched": sorted(
                    key for key in set(local) & set(remote) if local[key] != remote[key]
                ),
            })
        summary = {
            "comparison_count": len(comparison_summaries),
            "comparisons": comparison_summaries,
            "matched": all_matched,
        }
        write_summary(args.summary_path, summary)
        print("PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_BEGIN")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_END")
        return 0 if all_matched else 1
    except Exception as error:
        summary = {
            "comparison_count": len(comparisons),
            "comparisons": [],
            "matched": False,
            "error": repr(error),
            "error_type": type(error).__name__,
        }
        write_summary(args.summary_path, summary)
        print("PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_BEGIN")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_END")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
"""


def load_request_payload() -> tuple[str, dict]:
    workflow = os.environ.get("QA_PUBLICATION_WORKFLOW", "earth2-deterministic").strip()
    request_file = os.environ.get("QA_PUBLICATION_REQUEST_JSON", "").strip()
    if request_file:
        with open(request_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"request payload must be a JSON object: {request_file}")
        return workflow, payload
    payload = DEFAULT_PUBLICATION_REQUESTS.get(workflow)
    if payload is None:
        raise ValueError(
            f"no default publication request for workflow {workflow!r}; "
            "provide it in QA_PUBLICATION_REQUEST_JSON"
        )
    return workflow, copy.deepcopy(payload)


def load_request_payloads() -> list[tuple[str, dict]]:
    request_file = os.environ.get("QA_PUBLICATION_REQUEST_JSON", "").strip()
    workflow_spec = os.environ.get("QA_PUBLICATION_WORKFLOWS", "").strip()
    if not workflow_spec:
        workflow, payload = load_request_payload()
        return [(workflow, payload)]

    if workflow_spec.lower() == "all":
        workflow_names = list(DEFAULT_PUBLICATION_REQUESTS)
    else:
        workflow_names = [
            item.strip() for item in workflow_spec.split(",") if item.strip()
        ]
    if not workflow_names:
        raise ValueError("QA_PUBLICATION_WORKFLOWS did not include any workflow names")

    payloads = {}
    direct_payload = None
    if request_file:
        with open(request_file, "r", encoding="utf-8") as file:
            request_data = json.load(file)
        if not isinstance(request_data, dict):
            raise ValueError(
                "QA_PUBLICATION_REQUEST_JSON must be an object when QA_PUBLICATION_WORKFLOWS is set"
            )
        if len(workflow_names) == 1:
            workflow = workflow_names[0]
            if workflow in request_data:
                if len(request_data) != 1:
                    raise ValueError(
                        "QA_PUBLICATION_REQUEST_JSON is ambiguous: use either a direct "
                        "request object or a workflow-to-payload map"
                    )
                payloads = request_data
            elif any(name in DEFAULT_PUBLICATION_REQUESTS for name in request_data):
                raise ValueError(
                    "QA_PUBLICATION_REQUEST_JSON looks like a workflow-to-payload map "
                    f"but does not contain selected workflow {workflow!r}"
                )
            else:
                direct_payload = request_data
        else:
            if not any(workflow in request_data for workflow in workflow_names):
                raise ValueError(
                    "QA_PUBLICATION_REQUEST_JSON must be a workflow-to-payload map "
                    "when multiple workflows are selected"
                )
            payloads = request_data

    requests = []
    for workflow in workflow_names:
        if direct_payload is not None:
            payload = direct_payload
        elif workflow in payloads:
            payload = payloads[workflow]
        else:
            payload = DEFAULT_PUBLICATION_REQUESTS.get(workflow)
        if payload is None:
            raise ValueError(
                f"no default publication request for workflow {workflow!r}; "
                "provide it in QA_PUBLICATION_REQUEST_JSON"
            )
        if not isinstance(payload, dict):
            raise ValueError(
                f"request payload for workflow {workflow!r} must be an object"
            )
        requests.append((workflow, copy.deepcopy(payload)))
    return requests


def submit_workflow(client, adapter, workflow: str, params: dict) -> str:
    request_timeout = float(
        os.environ.get("QA_PUBLICATION_SUBMIT_REQUEST_TIMEOUT_SECS", "180")
    )
    attempts = max(1, int(os.environ.get("QA_PUBLICATION_SUBMIT_ATTEMPTS", "3")))
    interval = float(os.environ.get("QA_PUBLICATION_SUBMIT_RETRY_INTERVAL_SECS", "10"))
    submit_url = client.base_url + adapter.submit_url(workflow)
    headers = dict(getattr(client, "headers", {}) or {})
    body = adapter.format_submit_body(params)
    last_error: Exception | None = None
    response = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                submit_url,
                headers=headers,
                json=body,
                timeout=request_timeout,
            )
            if response.status_code != 503:
                break
            last_error = requests.HTTPError(
                f"submit returned transient 503 on attempt {attempt}/{attempts}",
                response=response,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(interval)
    if response is None:
        assert last_error is not None
        raise last_error
    response.raise_for_status()
    parsed = adapter.parse_submit_response(response.json())
    return parsed["execution_id"]


def submit_workflows_parallel(
    client, adapter, requests: list[tuple[str, dict]]
) -> list[dict]:
    max_workers = int(
        os.environ.get("QA_PUBLICATION_SUBMIT_WORKERS", str(len(requests)))
    )
    max_workers = max(1, min(max_workers, len(requests)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(submit_workflow, client, adapter, workflow, payload): (
                workflow,
                payload,
            )
            for workflow, payload in requests
        }
        runs = []
        for future in concurrent.futures.as_completed(futures):
            workflow, payload = futures[future]
            runs.append(
                {
                    "workflow": workflow,
                    "request_payload": payload,
                    "exec_id": future.result(),
                }
            )
    return sorted(runs, key=lambda item: item["workflow"])


def wait_for_publication(client, adapter, workflow: str, exec_id: str) -> dict:
    timeout = int(os.environ.get("QA_PUBLICATION_TIMEOUT_SECS", "900"))
    interval = int(os.environ.get("QA_PUBLICATION_POLL_INTERVAL_SECS", "10"))
    request_timeout = float(
        os.environ.get("QA_PUBLICATION_STATUS_REQUEST_TIMEOUT_SECS", "15")
    )
    status_url = client.base_url + adapter.status_url(workflow, exec_id)
    headers = dict(getattr(client, "headers", {}) or {})
    deadline = time.time() + timeout
    last_payload = None
    while time.time() < deadline:
        try:
            response = requests.get(
                status_url,
                headers=headers,
                timeout=request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            last_payload = {
                "status_poll_error": f"{type(exc).__name__}: {exc}",
                "status_url": status_url,
            }
            time.sleep(interval)
            continue
        last_payload = payload
        parsed = adapter.parse_status_response(payload)
        status = parsed["status"]
        publication_status = payload.get("output_publication_status")
        if (
            status
            in {
                RequestStatus.COMPLETED.value,
                "succeeded",
                "success",
            }
            and publication_status == "uploaded"
        ):
            return payload
        if status in {RequestStatus.FAILED.value, RequestStatus.CANCELLED.value}:
            raise RuntimeError(
                f"execution {exec_id} ended with status={status}: {payload}"
            )
        time.sleep(interval)
    raise TimeoutError(
        f"execution {exec_id} did not publish within {timeout}s; last status: {last_payload}"
    )


def wait_for_publications(client, adapter, runs: list[dict]) -> list[dict]:
    for run in runs:
        run["status_payload"] = wait_for_publication(
            client, adapter, run["workflow"], run["exec_id"]
        )
    return runs


def fetch_results(client, adapter, workflow: str, exec_id: str) -> dict:
    response = client.get(
        client.base_url + adapter.results_url(workflow, exec_id),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_publication_results(client, adapter, runs: list[dict]) -> list[dict]:
    for run in runs:
        run["results_payload"] = publication_results_payload_from_status(
            run.get("status_payload")
        ) or fetch_results(client, adapter, run["workflow"], run["exec_id"])
        run["published_artifact"] = published_artifact(run["results_payload"])
    return runs


def publication_results_payload_from_status(status_payload: dict | None) -> dict | None:
    if not isinstance(status_payload, dict):
        return None
    published_artifacts = _coerce_json_array(status_payload.get("published_artifacts"))
    if not published_artifacts:
        return None

    execution = {
        "run_id": status_payload.get("run_id"),
        "status": status_payload.get("status"),
        "workflow": status_payload.get("workflow") or status_payload.get("workflow_id"),
        "published_artifacts": published_artifacts,
    }
    outputs = _coerce_json_array(status_payload.get("outputs"))
    if outputs:
        execution["outputs"] = outputs
    artifacts = _coerce_json_array(status_payload.get("artifacts"))
    if artifacts:
        execution["artifacts"] = artifacts
    for key in ("output_path", "output_archive"):
        value = status_payload.get(key)
        if isinstance(value, str) and value:
            execution[key] = value

    return {
        "run_id": status_payload.get("run_id"),
        "status": status_payload.get("status"),
        "workflow": execution["workflow"],
        "execution": execution,
        "payload": {},
    }


def _coerce_json_array(value) -> list | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def published_artifact(results_payload: dict) -> dict:
    execution = results_payload.get("execution")
    if not isinstance(execution, dict):
        raise AssertionError("results payload did not include execution metadata")
    artifacts = execution.get("published_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AssertionError("results payload did not include published_artifacts")
    artifact = artifacts[0]
    if artifact.get("status") != "uploaded":
        raise AssertionError(f"published artifact was not uploaded: {artifact}")
    return artifact


def compare_publication_with_lustre_job(results_payload: dict, artifact: dict) -> dict:
    return compare_publications_with_lustre_job(
        [{"results_payload": results_payload, "published_artifact": artifact}]
    )


def compare_publications_with_lustre_job(runs: list[dict]) -> dict:
    comparisons = []
    for run in runs:
        results_payload = run["results_payload"]
        artifact = run["published_artifact"]
        comparisons.append(
            {
                "label": f"{run.get('workflow', results_payload.get('workflow'))}:{run.get('exec_id', results_payload.get('run_id'))}",
                "local_path": _local_path_for_published_artifact(
                    results_payload, artifact
                ),
                "destination_uri": artifact["destination_uri"],
            }
        )
    if not comparisons:
        raise AssertionError("no publication comparisons were provided")

    job_name = _compare_job_name_for_runs(runs)
    summary_path = (
        f"{_compare_mount_target().rstrip('/')}/publication-compare/{job_name}.json"
    )
    command = _compare_job_command_for_comparisons(
        comparisons=comparisons,
        summary_path=summary_path,
    )
    return _run_compare_job(job_name, command, summary_path)


def _run_compare_job(job_name: str, command: str, summary_path: str) -> dict:
    job_id = _submit_compare_job(job_name, command)
    try:
        exit_code = _poll_compare_job(job_id)
        logs = _capture_compare_job_logs(job_id)
        summary = _extract_compare_summary(logs)
        if summary is None:
            summary = _read_compare_summary_with_job(job_name, summary_path)
    finally:
        _remove_compare_job(job_id)
    if exit_code != 0:
        raise AssertionError(
            f"publication compare job failed with exit code {exit_code}: {summary or logs}"
        )
    if summary is None:
        return {"matched": True, "summary_unavailable": True}
    if not summary or not summary.get("matched"):
        raise AssertionError(f"publication compare did not match: {summary}")
    return summary


def _compare_job_name_for_runs(runs: list[dict]) -> str:
    first = runs[0] if runs else {}
    first_results = first.get("results_payload") if isinstance(first, dict) else None
    first_artifact = (
        first.get("published_artifact") if isinstance(first, dict) else None
    )
    provider = (
        str(first_artifact.get("provider") or "").strip().lower()
        if isinstance(first_artifact, dict)
        else ""
    )
    provider = provider if provider in {"s3", "azure"} else "object"
    run_id = str(
        first.get("exec_id")
        or first.get("run_id")
        or (first_results.get("run_id") if isinstance(first_results, dict) else None)
        or int(time.time())
    )
    return f"ff-pubcmp-{provider}-{run_id.split('-')[0]}-{len(runs)}r"[:63].rstrip("-")


def _compare_job_command_for_comparisons(
    *, comparisons: list[dict], summary_path: str
) -> str:
    encoded_script = base64.b64encode(COMPARE_JOB_SCRIPT.encode("utf-8")).decode(
        "ascii"
    )
    encoded_comparisons = base64.b64encode(
        json.dumps(comparisons).encode("utf-8")
    ).decode("ascii")
    args = [
        "python3",
        "-",
        "--local-path",
        comparisons[0]["local_path"],
        "--destination-uri",
        comparisons[0]["destination_uri"],
        "--summary-path",
        summary_path,
        "--comparisons-b64",
        encoded_comparisons,
        "--endpoint-url",
        os.environ.get("QA_PUBLICATION_S3_ENDPOINT")
        or os.environ.get("S3_ENDPOINT_URL", ""),
        "--region",
        os.environ.get("QA_PUBLICATION_S3_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION", ""),
    ]
    return f"printf %s {shlex.quote(encoded_script)} | base64 -d | {' '.join(shlex.quote(arg) for arg in args)}"


def _local_path_for_published_artifact(results_payload: dict, artifact: dict) -> str:
    execution = results_payload.get("execution")
    if not isinstance(execution, dict):
        raise AssertionError("results payload did not include execution metadata")
    source_name = artifact.get("source_artifact")
    outputs = execution.get("outputs") or execution.get("artifacts") or []
    if isinstance(outputs, list):
        for entry in outputs:
            if not isinstance(entry, dict):
                continue
            if entry.get("name") == source_name or entry.get("primary") is True:
                path = (
                    entry.get("storage_path")
                    or entry.get("path")
                    or entry.get("output_path")
                )
                if isinstance(path, str) and path:
                    return path
    output_path = execution.get("output_path")
    if isinstance(output_path, str) and output_path:
        return output_path
    raise AssertionError(
        f"could not resolve local path for published artifact {source_name!r}"
    )


def _compare_image() -> str:
    image = os.environ.get("QA_PUBLICATION_COMPARE_IMAGE", "").strip()
    if image:
        return image
    image_tag = os.environ.get("QA_PUBLICATION_COMPARE_IMAGE_TAG", "").strip()
    image_name = os.environ.get(
        "QA_PUBLICATION_COMPARE_IMAGE_NAME", DEFAULT_COMPARE_IMAGE_NAME
    ).strip()
    if image_tag:
        if ":" in image_tag.rsplit("/", 1)[-1]:
            return image_tag
        return f"{image_name}:{image_tag}"
    return f"{image_name}:v0.1.0"


def _compare_mount_target() -> str:
    return os.environ.get("QA_PUBLICATION_MOUNT_TARGET", DEFAULT_COMPARE_MOUNT_TARGET)


def _run_lepton_command(args: list[str]) -> tuple[int, str]:
    env = {**os.environ}
    process = subprocess.run(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout or ""


def _submit_compare_job(job_name: str, command: str) -> str:
    nfs_path = os.environ.get("QA_PUBLICATION_NFS_PATH", "").strip()
    if not nfs_path:
        raise AssertionError("QA_PUBLICATION_NFS_PATH is required for compare job")
    job_args = [
        "lep",
        "job",
        "create",
        "--name",
        job_name,
        "--container-image",
        _compare_image(),
        "--node-group",
        os.environ.get("QA_PUBLICATION_NODE_GROUP", DEFAULT_COMPARE_NODE_GROUP),
        "--resource-shape",
        os.environ.get("QA_PUBLICATION_RESOURCE_SHAPE", DEFAULT_COMPARE_RESOURCE_SHAPE),
        "--image-pull-secrets",
        os.environ.get("QA_PUBLICATION_PULL_SECRET", DEFAULT_COMPARE_PULL_SECRET),
        "--mount",
        f"{nfs_path}:{_compare_mount_target()}:node-nfs:{os.environ.get('QA_PUBLICATION_LUSTRE_STORAGE', DEFAULT_COMPARE_LUSTRE_STORAGE)}",
        "--command",
        command,
    ]
    for name in COMPARE_JOB_CREDENTIAL_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            job_args[-2:-2] = ["--env", f"{name}={value}"]
    returncode, output = _run_lepton_command(job_args)
    if returncode != 0:
        raise AssertionError(f"failed to submit compare job: {output}")
    match = re.search(r"^\s*ID:\s*(\S+)", output, re.MULTILINE)
    if not match:
        raise AssertionError(f"could not parse compare job id: {output}")
    return match.group(1)


def _read_compare_summary_with_job(job_name: str, summary_path: str) -> dict | None:
    reader_name = f"{job_name}-read"[:63].rstrip("-")
    reader_python = (
        "import json\n"
        f"payload=json.load(open({json.dumps(summary_path)}, encoding='utf-8'))\n"
        "print('PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_BEGIN')\n"
        "print(json.dumps(payload, indent=2, sort_keys=True))\n"
        "print('PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_END')\n"
    )
    encoded = base64.b64encode(reader_python.encode("utf-8")).decode("ascii")
    command = f"printf %s {shlex.quote(encoded)} | base64 -d | python3"
    reader_id = _submit_compare_job(reader_name, command)
    try:
        exit_code = _poll_compare_job(reader_id)
        logs = _capture_compare_job_logs(reader_id)
    finally:
        _remove_compare_job(reader_id)
    if exit_code != 0:
        return None
    return _extract_compare_summary(logs)


def _job_succeeded(status_output: str) -> bool:
    return bool(
        re.search(r'"state":\s*"(Completed|Succeeded|Success)"', status_output, re.I)
    )


def _job_failed(status_output: str) -> bool:
    return bool(
        re.search(r'"state":\s*"(Failed|Cancelled|Stopped|Error)"', status_output, re.I)
    )


def _poll_compare_job(job_id: str) -> int:
    deadline = time.time() + int(
        os.environ.get("QA_PUBLICATION_COMPARE_TIMEOUT_SECS", str(COMPARE_JOB_TIMEOUT))
    )
    interval = int(
        os.environ.get(
            "QA_PUBLICATION_COMPARE_POLL_INTERVAL_SECS",
            str(COMPARE_JOB_POLL_INTERVAL),
        )
    )
    last_output = ""
    while time.time() < deadline:
        returncode, output = _run_lepton_command(["lep", "job", "get", "-i", job_id])
        if returncode == 0:
            last_output = output
            if _job_succeeded(output):
                return 0
            if _job_failed(output):
                return 1
        time.sleep(interval)
    raise TimeoutError(f"compare job {job_id} timed out; last status: {last_output}")


def _capture_compare_job_logs(job_id: str) -> str:
    returncode, output = _run_lepton_command(["lep", "job", "log", "-i", job_id])
    if returncode != 0:
        return output
    return output


def _remove_compare_job(job_id: str) -> None:
    _run_lepton_command(["lep", "job", "remove", "-i", job_id])


def _extract_compare_summary(logs: str) -> dict | None:
    match = re.search(
        r"PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_BEGIN\s*(\{.*?\})\s*PHYSICSNEMO_SERVE_PUBLICATION_COMPARE_END",
        logs,
        re.S,
    )
    if not match:
        return None
    return json.loads(match.group(1))


def local_artifact_file_map(
    client, adapter, workflow: str, exec_id: str, artifact: dict, work_dir: Path
):
    artifact_name = artifact["source_artifact"]
    artifact_names = [artifact_name]
    if artifact_name != "primary":
        artifact_names.append("primary")

    urls = [
        client.base_url + adapter.result_file_url(workflow, exec_id, name)
        for name in artifact_names
    ]
    filename = artifact.get("filename") or artifact_name
    if str(filename).endswith(".zarr"):
        urls.extend(f"{url}&format=zarr_zip" for url in list(urls))

    response = None
    for url in urls:
        response = client.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 404:
            break
    if response is None:
        raise AssertionError("no local artifact download URL was attempted")
    response.raise_for_status()
    work_dir.mkdir(parents=True, exist_ok=True)
    download_path = work_dir / "local-artifact"
    download_path.write_bytes(response.content)
    if zipfile.is_zipfile(download_path):
        extract_dir = work_dir / "local-extracted"
        with zipfile.ZipFile(download_path, "r") as zip_file:
            zip_file.extractall(extract_dir)
        root = _find_artifact_root(extract_dir, filename)
        return file_map(root)
    return {filename: file_fingerprint(download_path)}


def remote_artifact_file_map(artifact: dict, work_dir: Path):
    provider = artifact["provider"]
    destination_uri = artifact["destination_uri"]
    if provider == "s3":
        return _download_s3_artifact(destination_uri, work_dir / "remote")
    if provider == "azure":
        return _download_azure_artifact(destination_uri, work_dir / "remote")
    raise AssertionError(f"unsupported publication provider: {provider}")


def file_map(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): file_fingerprint(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def assert_manifest_matches(
    manifest: dict | None, files: dict[str, dict[str, object]]
) -> None:
    if manifest is None:
        return
    total_bytes = sum(int(item["size"]) for item in files.values())
    assert manifest.get("object_count") == len(files)
    assert manifest.get("total_bytes") == total_bytes


def _find_artifact_root(extract_dir: Path, filename: str) -> Path:
    for path in extract_dir.rglob(filename):
        if path.is_dir():
            return path
    return extract_dir


def _download_s3_artifact(destination_uri: str, target_dir: Path):
    import boto3

    parsed = urlparse(destination_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise AssertionError(f"invalid S3 destination URI: {destination_uri}")
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    client = boto3.client("s3")
    manifest = None
    files = _download_s3_prefix(client, bucket, prefix, target_dir)
    if not files:
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / Path(prefix).name
        client.download_file(bucket, prefix, str(output_path))
        return {output_path.name: file_fingerprint(output_path)}, None
    manifest_path = target_dir / "_physicsnemo_serve_publish_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_path.unlink()
    return file_map(target_dir), manifest


def _download_s3_prefix(
    client, bucket: str, prefix: str, target_dir: Path
) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    downloaded = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            rel = key.removeprefix(f"{prefix.rstrip('/')}/")
            output_path = target_dir / rel
            output_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(output_path))
            downloaded.append(rel)
    return downloaded


def _download_azure_artifact(destination_uri: str, target_dir: Path):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    parsed = urlparse(destination_uri)
    if parsed.scheme != "https" or not parsed.netloc:
        raise AssertionError(f"invalid Azure destination URI: {destination_uri}")
    parts = parsed.path.strip("/").split("/", 1)
    if len(parts) != 2:
        raise AssertionError(
            f"Azure destination URI must include container and blob: {destination_uri}"
        )
    container, prefix = parts
    credential = (
        os.environ.get("AZURE_STORAGE_SAS_TOKEN")
        or os.environ.get("AZURE_STORAGE_ACCESS_KEY")
        or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        or DefaultAzureCredential()
    )
    service = BlobServiceClient(
        account_url=f"{parsed.scheme}://{parsed.netloc}",
        credential=credential,
    )
    container_client = service.get_container_client(container)
    downloaded = []
    for blob in container_client.list_blobs(name_starts_with=f"{prefix.rstrip('/')}/"):
        rel = blob.name.removeprefix(f"{prefix.rstrip('/')}/")
        output_path = target_dir / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            file.write(container_client.download_blob(blob.name).readall())
        downloaded.append(rel)
    if not downloaded:
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / Path(prefix).name
        with output_path.open("wb") as file:
            file.write(container_client.download_blob(prefix).readall())
        return {output_path.name: file_fingerprint(output_path)}, None
    manifest = None
    manifest_path = target_dir / "_physicsnemo_serve_publish_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_path.unlink()
    return file_map(target_dir), manifest
