from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qa" / "scripts"))

import run_qa  # noqa: E402


def _resolved_publication_requests(monkeypatch, extra_env):
    from output_publication_helpers import load_request_payloads
    from service_adapter import RustAdapter

    with monkeypatch.context() as request_env:
        for name, value in extra_env.items():
            request_env.setenv(name, value)
        adapter = RustAdapter()
        return [
            adapter._resolve_workflow(workflow)
            for workflow, _payload in load_request_payloads()
        ]


def test_publication_defaults_to_plugin_selected_by_request(monkeypatch):
    for name in (
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID",
        "QA_PUBLICATION_WORKFLOW",
        "QA_PUBLICATION_WORKFLOWS",
        "QA_PUBLICATION_REQUEST_JSON",
    ):
        monkeypatch.delenv(name, raising=False)

    assert run_qa.determine_workflows(
        service="rust",
        workflows_arg=None,
        suite="publication",
    ) == ["earth2-deterministic"]


def test_publication_all_selects_only_packaged_rust_plugins(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    monkeypatch.delenv("QA_PUBLICATION_REQUEST_JSON", raising=False)
    monkeypatch.setenv("QA_PUBLICATION_WORKFLOWS", "all")

    selected = run_qa.determine_workflows(
        service="rust",
        workflows_arg=None,
        suite="publication",
    )

    missing = [
        plugin_id
        for plugin_id in selected
        if not (REPO_ROOT / "plugins" / plugin_id / "plugin.yaml").is_file()
    ]
    assert missing == []


def test_publication_workflows_argument_selects_matching_pytest_requests(
    monkeypatch,
    tmp_path,
):
    resolved_requests = []
    for name in (
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID",
        "QA_PUBLICATION_WORKFLOW",
        "QA_PUBLICATION_WORKFLOWS",
        "QA_PUBLICATION_REQUEST_JSON",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(
        run_qa,
        "deploy",
        lambda **_kwargs: ("https://endpoint.example.test", "test-endpoint"),
    )
    monkeypatch.setattr(run_qa, "health_check", lambda _url, _token: True)
    monkeypatch.setattr(run_qa, "post_health_wait", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qa, "teardown", lambda **_kwargs: None)

    def fake_run_pytest(**kwargs):
        resolved_requests.append(
            _resolved_publication_requests(monkeypatch, kwargs["extra_env"])
        )
        return 0

    monkeypatch.setattr(run_qa, "run_pytest", fake_run_pytest)

    workflow_ids = run_qa.determine_workflows(
        service="rust",
        workflows_arg="e2s-deterministic,earth2-deterministic",
        suite="publication",
    )
    for workflow_id in workflow_ids:
        assert (
            run_qa.run_one_workflow(
                workflow_id=workflow_id,
                source="physicsnemo-serve",
                service="rust",
                image_tag="test-image",
                workspace_id="test-workspace",
                workspace_token="test-token",
                endpoint_token="endpoint-token",
                nfs_path="/mnt/shared/test",
                suite="publication",
                test_filter=None,
                num_proc=1,
                skip_teardown=False,
                stream_logs=False,
                log_interval=30,
                artifact_dir=tmp_path,
                post_health_wait_secs=0,
                container_envs=[],
            )
            == 0
        )

    assert resolved_requests == [
        ["e2s-deterministic"],
        ["earth2-deterministic"],
    ]


def test_publication_pytest_selection_preserves_explicit_request_env(
    monkeypatch,
    tmp_path,
):
    request_file = tmp_path / "requests.json"
    payloads = {
        "deterministic_workflow": {"model_type": "fcn", "nsteps": 2},
        "earth2-deterministic": {"model": "dlwp", "nsteps": 3},
    }
    request_file.write_text(json.dumps(payloads), encoding="utf-8")
    monkeypatch.setenv(
        "QA_PUBLICATION_WORKFLOWS",
        "deterministic_workflow,earth2-deterministic",
    )
    monkeypatch.setenv("QA_PUBLICATION_REQUEST_JSON", str(request_file))
    monkeypatch.setattr(
        run_qa,
        "deploy",
        lambda **_kwargs: ("https://endpoint.example.test", "test-endpoint"),
    )
    monkeypatch.setattr(run_qa, "health_check", lambda _url, _token: True)
    monkeypatch.setattr(run_qa, "post_health_wait", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qa, "teardown", lambda **_kwargs: None)

    observed_requests = []

    def fake_run_pytest(**kwargs):
        from output_publication_helpers import load_request_payloads

        with monkeypatch.context() as request_env:
            for name, value in kwargs["extra_env"].items():
                request_env.setenv(name, value)
            observed_requests.extend(load_request_payloads())
        return 0

    monkeypatch.setattr(run_qa, "run_pytest", fake_run_pytest)

    assert (
        run_qa.run_one_workflow(
            workflow_id="e2s-deterministic",
            source="physicsnemo-serve",
            service="rust",
            image_tag="test-image",
            workspace_id="test-workspace",
            workspace_token="test-token",
            endpoint_token="endpoint-token",
            nfs_path="/mnt/shared/test",
            suite="publication",
            test_filter=None,
            num_proc=1,
            skip_teardown=False,
            stream_logs=False,
            log_interval=30,
            artifact_dir=tmp_path,
            post_health_wait_secs=0,
            container_envs=[],
        )
        == 0
    )

    assert observed_requests == list(payloads.items())


def test_main_exits_after_per_workflow_results_without_second_pytest(
    monkeypatch,
    tmp_path,
):
    workflow_calls = []

    monkeypatch.setattr(run_qa, "_ensure_line_buffered", lambda: None)
    monkeypatch.setattr(
        run_qa,
        "load_deploy_config",
        lambda: {
            "lepton_workspace_id": "test-workspace",
            "nfs_mount_base": "/mnt/shared",
        },
    )
    monkeypatch.setattr(
        run_qa.signal,
        "signal",
        lambda _signum, _handler: None,
    )
    monkeypatch.setenv("LEPTON_WORKSPACE_TOKEN", "test-token")
    monkeypatch.setenv("LEPTON_ENDPOINT_TOKEN", "endpoint-token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qa.py",
            "--service",
            "rust",
            "--image-tag",
            "test-image",
            "--suite",
            "smoke",
            "--workflows",
            "e2s-deterministic",
            "--artifact-dir",
            str(tmp_path),
            "--no-endpoint-logs",
        ],
    )

    def fake_run_one_workflow(**kwargs):
        workflow_calls.append(kwargs)
        return 0

    monkeypatch.setattr(run_qa, "run_one_workflow", fake_run_one_workflow)
    monkeypatch.setattr(
        run_qa,
        "run_pytest",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("main must not run a second pytest pass")
        ),
    )

    try:
        run_qa.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("main() must exit with the QA result code")

    assert len(workflow_calls) == 1
    assert workflow_calls[0]["workflow_id"] == "e2s-deterministic"


def test_publication_container_envs_are_passed_to_workflow_deployment(
    monkeypatch,
    tmp_path,
):
    workflow_calls = []
    publication_envs = [
        "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG=/container/runtime.json",
        "WORKER_RUNTIME_CONFIG=/container/runtime.json",
        "AZURE_STORAGE_ACCOUNT_NAME=test-account",
        "AZURE_STORAGE_ACCOUNT_KEY=test-key",
    ]

    monkeypatch.setattr(run_qa, "_ensure_line_buffered", lambda: None)
    monkeypatch.setattr(
        run_qa,
        "load_deploy_config",
        lambda: {
            "lepton_workspace_id": "test-workspace",
            "nfs_mount_base": "/mnt/shared",
        },
    )
    monkeypatch.setattr(
        run_qa.signal,
        "signal",
        lambda _signum, _handler: None,
    )
    monkeypatch.setattr(
        run_qa,
        "write_publication_runtime_config",
        lambda **_kwargs: (tmp_path / "runtime.json", "/container/runtime.json"),
    )
    monkeypatch.setattr(
        run_qa,
        "publication_container_envs",
        lambda _config_path: list(publication_envs),
    )
    monkeypatch.setenv("LEPTON_WORKSPACE_TOKEN", "test-token")
    monkeypatch.setenv("LEPTON_ENDPOINT_TOKEN", "endpoint-token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qa.py",
            "--service",
            "rust",
            "--image-tag",
            "test-image",
            "--suite",
            "publication",
            "--workflows",
            "e2s-deterministic",
            "--artifact-dir",
            str(tmp_path),
            "--no-endpoint-logs",
        ],
    )

    def fake_run_one_workflow(**kwargs):
        workflow_calls.append(kwargs)
        return 0

    monkeypatch.setattr(run_qa, "run_one_workflow", fake_run_one_workflow)

    try:
        run_qa.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("main() must exit with the QA result code")

    assert len(workflow_calls) == 1
    assert workflow_calls[0]["container_envs"] == publication_envs


def test_run_one_workflow_merges_publication_envs_into_deploy(
    monkeypatch,
    tmp_path,
):
    deploy_calls = []
    publication_envs = [
        "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG=/container/runtime.json",
        "WORKER_RUNTIME_CONFIG=/container/runtime.json",
        "AZURE_STORAGE_ACCOUNT_KEY=test-key",
    ]

    def fake_deploy(**kwargs):
        deploy_calls.append(kwargs)
        return "https://endpoint.example.test", "test-endpoint"

    monkeypatch.setattr(run_qa, "deploy", fake_deploy)
    monkeypatch.setattr(run_qa, "health_check", lambda _url, _token: True)
    monkeypatch.setattr(run_qa, "post_health_wait", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qa, "run_pytest", lambda **_kwargs: 0)
    monkeypatch.setattr(run_qa, "teardown", lambda **_kwargs: None)

    exit_code = run_qa.run_one_workflow(
        workflow_id="e2s-deterministic",
        source="physicsnemo-serve",
        service="rust",
        image_tag="test-image",
        workspace_id="test-workspace",
        workspace_token="test-token",
        endpoint_token="endpoint-token",
        nfs_path="/mnt/shared/test",
        suite="publication",
        test_filter=None,
        num_proc=1,
        skip_teardown=False,
        stream_logs=False,
        log_interval=30,
        artifact_dir=tmp_path,
        post_health_wait_secs=0,
        container_envs=publication_envs,
    )

    assert exit_code == 0
    assert deploy_calls[0]["container_envs"] == [
        *publication_envs,
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID=e2s-deterministic",
    ]


def test_run_one_workflow_passes_publication_compare_env_to_pytest(
    monkeypatch,
    tmp_path,
):
    pytest_calls = []

    def fake_deploy(**_kwargs):
        return "https://endpoint.example.test", "test-endpoint"

    def fake_run_pytest(**kwargs):
        pytest_calls.append(kwargs)
        return 0

    monkeypatch.setenv("LEPTON_NODE_GROUP", "test-node")
    monkeypatch.setenv("QA_PUBLICATION_COMPARE_RESOURCE_SHAPE", "cpu.test")
    monkeypatch.setenv("LEPTON_PULL_SECRET", "test-pull-secret")
    monkeypatch.setenv("LEPTON_LUSTRE_STORAGE", "test-lustre")
    monkeypatch.setattr(run_qa, "deploy", fake_deploy)
    monkeypatch.setattr(run_qa, "health_check", lambda _url, _token: True)
    monkeypatch.setattr(run_qa, "post_health_wait", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qa, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(run_qa, "teardown", lambda **_kwargs: None)

    exit_code = run_qa.run_one_workflow(
        workflow_id="e2s-deterministic",
        source="physicsnemo-serve",
        service="rust",
        image_tag="nvcr.io/test/physicsnemo-serve:current",
        workspace_id="test-workspace",
        workspace_token="test-token",
        endpoint_token="endpoint-token",
        nfs_path="/mnt/shared/test",
        suite="publication",
        test_filter=None,
        num_proc=1,
        skip_teardown=False,
        stream_logs=False,
        log_interval=30,
        artifact_dir=tmp_path,
        post_health_wait_secs=0,
        container_envs=[],
    )

    assert exit_code == 0
    assert pytest_calls[0]["extra_env"] == {
        "LEPTON_WORKSPACE_ID": "test-workspace",
        "LEPTON_WORKSPACE_TOKEN": "test-token",
        "QA_PUBLICATION_WORKFLOW": "deterministic_workflow",
        "QA_PUBLICATION_NFS_PATH": "/mnt/shared/test",
        "QA_PUBLICATION_MOUNT_TARGET": "/outputs",
        "QA_PUBLICATION_COMPARE_IMAGE": "nvcr.io/test/physicsnemo-serve:current",
        "QA_PUBLICATION_NODE_GROUP": "test-node",
        "QA_PUBLICATION_RESOURCE_SHAPE": "cpu.test",
        "QA_PUBLICATION_PULL_SECRET": "test-pull-secret",
        "QA_PUBLICATION_LUSTRE_STORAGE": "test-lustre",
    }


def test_publication_compare_env_uses_custom_deploy_config(monkeypatch):
    for name in (
        "QA_PUBLICATION_COMPARE_IMAGE",
        "QA_PUBLICATION_NODE_GROUP",
        "LEPTON_NODE_GROUP",
        "QA_PUBLICATION_PULL_SECRET",
        "LEPTON_PULL_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    compare_env = run_qa.publication_compare_env(
        image_tag="release-tag",
        workspace_id="test-workspace",
        workspace_token="test-token",
        nfs_path="/remote/nfs/test",
        deploy_config={
            "docker_registry": "registry.example.test/team",
            "image_name": "custom-physicsnemo-serve",
            "lepton_node_group": "custom-node-group",
            "pull_secret": "custom-pull-secret",
        },
    )

    assert compare_env["QA_PUBLICATION_COMPARE_IMAGE"] == (
        "registry.example.test/team/custom-physicsnemo-serve:release-tag"
    )
    assert compare_env["QA_PUBLICATION_NODE_GROUP"] == "custom-node-group"
    assert compare_env["QA_PUBLICATION_PULL_SECRET"] == "custom-pull-secret"


def test_publication_compare_env_prefers_qa_overrides(monkeypatch):
    monkeypatch.setenv(
        "QA_PUBLICATION_COMPARE_IMAGE", "registry.example.test/qa/compare:override"
    )
    monkeypatch.setenv("QA_PUBLICATION_NODE_GROUP", "qa-node-group")
    monkeypatch.setenv("LEPTON_NODE_GROUP", "generic-node-group")
    monkeypatch.setenv("QA_PUBLICATION_PULL_SECRET", "qa-pull-secret")
    monkeypatch.setenv("LEPTON_PULL_SECRET", "generic-pull-secret")

    compare_env = run_qa.publication_compare_env(
        image_tag="release-tag",
        workspace_id="test-workspace",
        workspace_token="test-token",
        nfs_path="/remote/nfs/test",
        deploy_config={
            "docker_registry": "config-registry.example.test",
            "image_name": "config-image",
            "lepton_node_group": "config-node-group",
            "pull_secret": "config-pull-secret",
        },
    )

    assert (
        compare_env["QA_PUBLICATION_COMPARE_IMAGE"]
        == "registry.example.test/qa/compare:override"
    )
    assert compare_env["QA_PUBLICATION_NODE_GROUP"] == "qa-node-group"
    assert compare_env["QA_PUBLICATION_PULL_SECRET"] == "qa-pull-secret"


def test_publication_s3_region_endpoint_go_into_config_not_container_env(
    monkeypatch,
):
    monkeypatch.setenv("QA_PUBLICATION_STORAGE_TYPE", "s3")
    monkeypatch.setenv("QA_PUBLICATION_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("QA_PUBLICATION_S3_REGION", "us-ashburn-1")
    monkeypatch.setenv(
        "QA_PUBLICATION_S3_ENDPOINT", "https://objectstorage.example.test"
    )
    monkeypatch.setenv("QA_PUBLICATION_PREFIX", "test-prefix")
    monkeypatch.setenv("AWS_REGION", "ambient-region")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://ambient-endpoint.example.test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-key")

    storage = run_qa.build_publication_storage_config()
    envs = run_qa.publication_container_envs("/app/runtime.json")

    assert storage == {
        "type": "s3",
        "bucket": "test-bucket",
        "prefix": "test-prefix",
        "region": "us-ashburn-1",
        "endpoint": "https://objectstorage.example.test",
    }
    assert "AWS_ACCESS_KEY_ID=access-key" in envs
    assert "AWS_SECRET_ACCESS_KEY=secret-key" in envs
    assert "AWS_REGION=ambient-region" not in envs
    assert "S3_ENDPOINT_URL=https://ambient-endpoint.example.test" not in envs


def test_publication_upload_perf_env_goes_into_publish_role_config(monkeypatch):
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_MAX_CONCURRENT_FILES", "16")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_MULTIPART_THRESHOLD_BYTES", "67108864")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_MULTIPART_PART_SIZE_BYTES", "16777216")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_MULTIPART_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_TIMEOUT_SECS", "300")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_CONNECT_TIMEOUT_SECS", "10")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_POOL_MAX_IDLE_PER_HOST", "64")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_RETRY_MAX_RETRIES", "10")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_RETRY_TIMEOUT_SECS", "300")

    assert run_qa.build_publication_publish_role_config() == {
        "max_concurrent_files": 16,
        "multipart_threshold_bytes": 67108864,
        "multipart_part_size_bytes": 16777216,
        "multipart_max_concurrency": 4,
        "client_options": {
            "timeout_secs": 300,
            "connect_timeout_secs": 10,
            "pool_max_idle_per_host": 64,
        },
        "retry": {
            "max_retries": 10,
            "timeout_secs": 300,
        },
    }


def test_publication_upload_perf_env_is_optional(monkeypatch):
    for name in (
        "QA_PUBLICATION_UPLOAD_MAX_CONCURRENT_FILES",
        "QA_PUBLICATION_UPLOAD_MULTIPART_THRESHOLD_BYTES",
        "QA_PUBLICATION_UPLOAD_MULTIPART_PART_SIZE_BYTES",
        "QA_PUBLICATION_UPLOAD_MULTIPART_MAX_CONCURRENCY",
        "QA_PUBLICATION_UPLOAD_TIMEOUT_SECS",
        "QA_PUBLICATION_UPLOAD_CONNECT_TIMEOUT_SECS",
        "QA_PUBLICATION_UPLOAD_POOL_MAX_IDLE_PER_HOST",
        "QA_PUBLICATION_UPLOAD_RETRY_MAX_RETRIES",
        "QA_PUBLICATION_UPLOAD_RETRY_TIMEOUT_SECS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert run_qa.build_publication_publish_role_config() is None


def test_publication_azure_account_key_is_forwarded_to_container(monkeypatch):
    monkeypatch.setenv("QA_PUBLICATION_STORAGE_TYPE", "azure")
    monkeypatch.setenv(
        "QA_PUBLICATION_AZURE_ENDPOINT",
        "https://exampleaccount.blob.core.windows.net",
    )
    monkeypatch.setenv("QA_PUBLICATION_AZURE_CONTAINER", "forecast-results")
    monkeypatch.setenv("QA_PUBLICATION_PREFIX", "outputs")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "exampleaccount")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "storage-key")

    storage = run_qa.build_publication_storage_config()
    envs = run_qa.publication_container_envs("/app/runtime.json")

    assert storage == {
        "type": "azure",
        "endpoint": "https://exampleaccount.blob.core.windows.net",
        "container": "forecast-results",
        "prefix": "outputs",
    }
    assert "AZURE_STORAGE_ACCOUNT_NAME=exampleaccount" in envs
    assert "AZURE_STORAGE_ACCOUNT_KEY=storage-key" in envs


def test_publication_container_envs_support_env_config_fallback(monkeypatch):
    monkeypatch.setenv("QA_PUBLICATION_STORAGE_TYPE", "s3")
    monkeypatch.setenv("QA_PUBLICATION_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("QA_PUBLICATION_PREFIX", "prefix")
    monkeypatch.setenv("QA_PUBLICATION_UPLOAD_MAX_CONCURRENT_FILES", "7")

    envs = run_qa.publication_container_envs(run_qa.PUBLICATION_ENV_CONFIG_SENTINEL)
    env_map = dict(item.split("=", 1) for item in envs)

    assert (
        env_map["PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"]
        == "/app/scripts/worker_runtime_config.json"
    )
    assert env_map["WORKER_RUNTIME_CONFIG"] == "/app/scripts/worker_runtime_config.json"
    assert env_map["PHYSICSNEMO_SERVE_OUTPUT_PUBLICATION_CONFIG_JSON"]
    assert (
        env_map["PHYSICSNEMO_SERVE_PUBLISH_ROLE_CONFIG_JSON"]
        == '{"max_concurrent_files":7}'
    )


def test_publication_runtime_config_requires_explicit_local_mount_mapping(
    monkeypatch,
    tmp_path,
):
    remote_path = tmp_path / "writable-remote-path"
    remote_path.mkdir()
    monkeypatch.delenv("QA_PUBLICATION_LOCAL_MOUNT_PATH", raising=False)
    monkeypatch.setenv("QA_PUBLICATION_STORAGE_TYPE", "s3")
    monkeypatch.setenv("QA_PUBLICATION_S3_BUCKET", "test-bucket")

    paths = run_qa.write_publication_runtime_config(
        nfs_path=str(remote_path),
        endpoint_name="test-endpoint",
    )

    assert paths == (
        Path(run_qa.PUBLICATION_ENV_CONFIG_SENTINEL),
        run_qa.PUBLICATION_ENV_CONFIG_SENTINEL,
    )
    assert not (remote_path / "qa-publication").exists()


def test_publication_runtime_config_uses_explicit_local_mount_mapping(
    monkeypatch,
    tmp_path,
):
    local_mount = tmp_path / "local-mount"
    monkeypatch.setenv("QA_PUBLICATION_LOCAL_MOUNT_PATH", str(local_mount))
    monkeypatch.setenv("QA_PUBLICATION_STORAGE_TYPE", "s3")
    monkeypatch.setenv("QA_PUBLICATION_S3_BUCKET", "test-bucket")

    host_path, container_path = run_qa.write_publication_runtime_config(
        nfs_path="/remote/nfs/test",
        endpoint_name="test-endpoint",
    )

    assert host_path == (
        local_mount / "qa-publication" / "test-endpoint" / "worker_runtime_config.json"
    )
    assert container_path == (
        "/outputs/qa-publication/test-endpoint/worker_runtime_config.json"
    )
    assert host_path.is_file()


# ---------------------------------------------------------------------------
# determine_e2s_workflows / determine_cfd_workflows routing tests
# ---------------------------------------------------------------------------


def test_determine_cfd_workflows_returns_cfd_for_full_suite(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    result = run_qa.determine_cfd_workflows(service="rust", suite="full")
    assert result == [run_qa.CFD_E2E_WORKFLOW_ID]


def test_determine_cfd_workflows_returns_cfd_for_cfd_e2e_suite(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    result = run_qa.determine_cfd_workflows(service="rust", suite="cfd_e2e")
    assert result == [run_qa.CFD_E2E_WORKFLOW_ID]


def test_determine_cfd_workflows_empty_for_other_suites(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    for suite in ("cicd", "smoke", "basic", "publication"):
        assert run_qa.determine_cfd_workflows(service="rust", suite=suite) == []
    assert run_qa.determine_cfd_workflows(service="python", suite="full") == []


def test_determine_e2s_workflows_empty_for_cfd_e2e_suite(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    result = run_qa.determine_e2s_workflows(
        service="rust", workflows_arg=None, suite="cfd_e2e"
    )
    assert result == []


def test_determine_e2s_workflows_returns_all_plugins_for_full_suite(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    result = run_qa.determine_e2s_workflows(
        service="rust", workflows_arg=None, suite="full"
    )
    assert result == list(run_qa.ALL_WORKFLOW_PLUGIN_IDS)
    assert run_qa.CFD_E2E_WORKFLOW_ID not in result


def test_determine_workflows_full_ends_with_cfd(monkeypatch):
    monkeypatch.delenv("PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID", raising=False)
    result = run_qa.determine_workflows(
        service="rust", workflows_arg=None, suite="full"
    )
    assert result[-1] == run_qa.CFD_E2E_WORKFLOW_ID
    assert result[:-1] == list(run_qa.ALL_WORKFLOW_PLUGIN_IDS)
