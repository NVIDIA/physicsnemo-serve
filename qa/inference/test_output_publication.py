import os

import pytest

from helpers import (
    enabled_physicsnemo_serve_plugin_id,
    physicsnemo_serve_plugin_id_for_workflow,
)
from output_publication_helpers import (
    assert_manifest_matches,
    compare_publications_with_lustre_job,
    fetch_results,
    fetch_publication_results,
    load_request_payloads,
    local_artifact_file_map,
    published_artifact,
    remote_artifact_file_map,
    submit_workflows_parallel,
    submit_workflow,
    wait_for_publications,
    wait_for_publication,
)

pytestmark = [
    pytest.mark.publication,
    pytest.mark.rust_only,
    pytest.mark.skipif(
        os.environ.get("QA_PUBLICATION_STORAGE_TYPE", "").strip().lower()
        not in {"s3", "azure"},
        reason="QA_PUBLICATION_STORAGE_TYPE must be set to 's3' or 'azure'",
    ),
]


def _publication_requests_for_enabled_plugin(adapter, requests):
    enabled_plugin_id = enabled_physicsnemo_serve_plugin_id()
    if enabled_plugin_id is None:
        return requests

    filtered = [
        (workflow, request_payload)
        for workflow, request_payload in requests
        if physicsnemo_serve_plugin_id_for_workflow(adapter, workflow)
        == enabled_plugin_id
    ]
    if not filtered:
        raise AssertionError(
            "no publication request payload resolves to "
            f"PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID={enabled_plugin_id!r}"
        )
    return filtered


def test_output_publication_remote_matches_local(client, adapter, tmp_path):
    storage_type = os.environ.get("QA_PUBLICATION_STORAGE_TYPE", "").strip().lower()
    requests = _publication_requests_for_enabled_plugin(
        adapter, load_request_payloads()
    )

    if storage_type == "s3":
        runs = submit_workflows_parallel(client, adapter, requests)
        wait_for_publications(client, adapter, runs)
        fetch_publication_results(client, adapter, runs)
        compare_publications_with_lustre_job(runs)
        for run in runs:
            assert run["published_artifact"]["provider"] == storage_type
        return

    if storage_type == "azure":
        if len(requests) != 1:
            raise AssertionError(
                "multi-workflow publication QA is only supported for S3"
            )
        workflow, request_payload = requests[0]
        exec_id = submit_workflow(client, adapter, workflow, request_payload)
        status_payload = wait_for_publication(client, adapter, workflow, exec_id)
        results_payload = fetch_results(client, adapter, workflow, exec_id)
        artifact = published_artifact(results_payload)
        compare_publications_with_lustre_job(
            [
                {
                    "workflow": workflow,
                    "exec_id": exec_id,
                    "status_payload": status_payload,
                    "results_payload": results_payload,
                    "published_artifact": artifact,
                }
            ]
        )
        assert artifact["provider"] == storage_type
        return

    if len(requests) != 1:
        raise AssertionError("multi-workflow publication QA is only supported for S3")
    workflow, request_payload = requests[0]
    exec_id = submit_workflow(client, adapter, workflow, request_payload)
    wait_for_publication(client, adapter, workflow, exec_id)
    results_payload = fetch_results(client, adapter, workflow, exec_id)
    artifact = published_artifact(results_payload)

    local_files = local_artifact_file_map(
        client,
        adapter,
        workflow,
        exec_id,
        artifact,
        tmp_path / "local",
    )
    remote_files, manifest = remote_artifact_file_map(artifact, tmp_path / "remote")

    assert artifact["provider"] == storage_type
    assert remote_files == local_files
    assert_manifest_matches(manifest, remote_files)
