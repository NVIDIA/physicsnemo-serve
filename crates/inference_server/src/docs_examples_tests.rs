/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_relative(path: &str) -> String {
    fs::read_to_string(crate_root().join(path)).unwrap_or_else(|err| {
        panic!(
            "failed to read fixture '{}': {err}",
            crate_root().join(path).display()
        )
    })
}

fn assert_uses_namespaced_run_routes(path: &str, content: &str, workflow_token: &str) {
    let expected_status = format!("/v1/infer/{workflow_token}/<run_id>/status");
    let expected_results = format!("/v1/infer/{workflow_token}/<run_id>/results");
    let legacy_status = "/v1/infer/<run_id>/status";
    let legacy_results = "/v1/infer/<run_id>/results";

    assert!(
        content.contains(&expected_status),
        "{path} should document workflow-namespaced status route '{expected_status}'"
    );
    assert!(
        content.contains(&expected_results),
        "{path} should document workflow-namespaced results route '{expected_results}'"
    );
    assert!(
        !content.contains(legacy_status),
        "{path} should not document legacy flat status route '{legacy_status}'"
    );
    assert!(
        !content.contains(legacy_results),
        "{path} should not document legacy flat results route '{legacy_results}'"
    );
}

#[test]
fn docs_use_workflow_namespaced_run_routes() {
    let onboarding = read_relative("../../docs/onboarding.md");
    assert_uses_namespaced_run_routes("docs/onboarding.md", &onboarding, "<workflow_id>");

    let user_guide = read_relative("../../docs/inference-service-user-guide.md");
    assert_uses_namespaced_run_routes(
        "docs/inference-service-user-guide.md",
        &user_guide,
        "<workflow_id>",
    );
}

#[test]
fn test_api_script_uses_workflow_namespaced_run_routes() {
    let script = read_relative("scripts/test_api.sh");

    assert!(
        script.contains("$BASE_URL/v1/infer/$WORKFLOW/$RUN_ID/status"),
        "test_api.sh should poll workflow-namespaced status route"
    );
    assert!(
        script.contains("$BASE_URL/v1/infer/$WORKFLOW/$RUN_ID/results"),
        "test_api.sh should fetch workflow-namespaced results route"
    );
    assert!(
        !script.contains("$BASE_URL/v1/infer/$RUN_ID/status"),
        "test_api.sh should not use legacy flat status route"
    );
    assert!(
        !script.contains("$BASE_URL/v1/infer/$RUN_ID/results"),
        "test_api.sh should not use legacy flat results route"
    );
}
