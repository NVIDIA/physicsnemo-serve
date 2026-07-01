/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: verify the example config file parses and validates.

use worker_runtime::config::RuntimeConfig;

const EXAMPLE_CONFIG: &str = include_str!("../examples/runtime_config.json");

#[test]
fn example_config_parses_and_validates() {
    let config: RuntimeConfig =
        serde_json::from_str(EXAMPLE_CONFIG).expect("example config should parse as JSON");
    config
        .validate()
        .expect("example config should pass validation");
}

#[test]
fn example_config_has_expected_roles() {
    let config: RuntimeConfig = serde_json::from_str(EXAMPLE_CONFIG).unwrap();
    let role_names: Vec<&str> = config.roles.keys().map(|s| s.as_str()).collect();

    assert!(role_names.contains(&"prefetch"));
    assert!(role_names.contains(&"scheduler"));
    assert!(role_names.contains(&"results"));
}

#[test]
fn example_config_has_no_gpu_roles() {
    let config: RuntimeConfig = serde_json::from_str(EXAMPLE_CONFIG).unwrap();
    let gpu_roles: Vec<&str> = config
        .roles
        .keys()
        .filter(|k| k.starts_with("gpu"))
        .map(|s| s.as_str())
        .collect();
    assert!(
        gpu_roles.is_empty(),
        "GPU roles should not be in static config (discovered at runtime via gpu:registry)"
    );
}

#[test]
fn example_config_scheduler_has_empty_outputs() {
    let config: RuntimeConfig = serde_json::from_str(EXAMPLE_CONFIG).unwrap();
    let scheduler = config
        .roles
        .get("scheduler")
        .expect("scheduler role should exist");
    assert!(
        scheduler.outputs.is_empty(),
        "scheduler outputs should be empty (GPU targets discovered from gpu:registry)"
    );
}

#[test]
fn example_config_scheduler_has_gpu_registry_key_in_config() {
    let config: RuntimeConfig = serde_json::from_str(EXAMPLE_CONFIG).unwrap();
    let scheduler = config
        .roles
        .get("scheduler")
        .expect("scheduler role should exist");
    let role_config = scheduler
        .config
        .as_ref()
        .expect("scheduler should have config block");
    let key = role_config
        .get("gpu_registry_key")
        .and_then(|v| v.as_str())
        .expect("should have gpu_registry_key");
    assert_eq!(key, "gpu:registry");
}

#[test]
fn example_config_results_has_empty_outputs() {
    let config: RuntimeConfig = serde_json::from_str(EXAMPLE_CONFIG).unwrap();
    let results = config
        .roles
        .get("results")
        .expect("results role should exist");
    assert!(
        results.outputs.is_empty(),
        "results is a terminal consumer with no outputs"
    );
}

#[test]
fn example_config_resolve_env_for_each_role() {
    let config: RuntimeConfig = serde_json::from_str(EXAMPLE_CONFIG).unwrap();
    for role_name in config.roles.keys() {
        let env = config
            .resolve_env(role_name)
            .unwrap_or_else(|e| panic!("resolve_env failed for '{role_name}': {e}"));
        assert_eq!(env.role_name, *role_name);
        assert!(!env.inputs.is_empty());
    }
}
