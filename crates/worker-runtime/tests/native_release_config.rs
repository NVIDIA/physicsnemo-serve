/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use worker_runtime::config::{RuntimeConfig, SchedulerRoleConfig};

#[test]
fn native_release_uses_existing_scheduler_without_batching_or_python_roles() {
    let config: RuntimeConfig = serde_json::from_str(include_str!(
        "../../../native-inference/deploy/worker_runtime_config.json"
    ))
    .unwrap();
    config.validate().unwrap();
    assert_eq!(
        config.roles.keys().map(String::as_str).collect::<Vec<_>>(),
        ["results", "scheduler"]
    );
    assert!(config.python_runtime_envs.is_empty());
    let scheduler: SchedulerRoleConfig =
        serde_json::from_value(config.roles["scheduler"].config.clone().unwrap()).unwrap();
    assert!(!scheduler.batching_enabled);
    assert_eq!(scheduler.max_batch_size, 1);
    assert!(
        config.roles["scheduler"]
            .inputs
            .iter()
            .any(|input| input.stream == "release")
    );
}
