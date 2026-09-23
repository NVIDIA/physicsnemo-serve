/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::stage::StageContext;

#[derive(Clone, Copy)]
pub(crate) enum ScatterPayloadTarget {
    Child,
    Continuation,
}

/// Copy only stable request and runtime context across a scatter boundary.
pub(crate) fn project_scatter_payload(
    payload: &Value,
    target: ScatterPayloadTarget,
) -> Result<Value> {
    let source = payload
        .as_object()
        .context("scatter payload must be an object")?;
    let mut projected = Map::new();
    for key in [
        "run_id",
        "workflow",
        "workflow_id",
        "manifest_version",
        "operation",
        "request",
        "parameters",
        "resource_profile",
        "prefetch_plan",
        "prefetch_artifacts",
        "stage_context",
        "output_publication",
        "runtime",
        "services",
        "run_dir",
    ] {
        if let Some(value) = source.get(key) {
            projected.insert(key.to_string(), value.clone());
        }
    }
    if matches!(target, ScatterPayloadTarget::Continuation)
        && let Some(value) = source.get("batch_profile")
    {
        projected.insert("batch_profile".to_string(), value.clone());
    }
    Ok(Value::Object(projected))
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScatterChild {
    pub(crate) operation: String,
    pub(crate) parameters: Map<String, Value>,
    pub(crate) resource_profile: Option<Map<String, Value>>,
    pub(crate) batch_profile: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScatterResult {
    kind: String,
    pub(crate) children: Vec<ScatterChild>,
    pub(crate) child_stage_id: String,
    pub(crate) continuation_stage_id: String,
    pub(crate) max_in_flight: u64,
}

impl ScatterResult {
    pub(crate) fn validate(&self, context: &StageContext, parent: Option<&str>) -> Result<()> {
        ensure!(self.kind == "scatter", "scatter: invalid kind");
        ensure!(
            matches!(context.current_phase.as_str(), "prepare" | "execute"),
            "scatter is supported only from prepare or execute"
        );
        ensure!(
            parent.is_none_or(str::is_empty),
            "nested scatter is not supported"
        );
        ensure!(
            !self.children.is_empty(),
            "scatter children must be non-empty"
        );
        ensure!(
            self.max_in_flight > 0,
            "scatter max_in_flight must be positive"
        );
        for (id, phases) in [
            (&self.child_stage_id, &["execute"][..]),
            (
                &self.continuation_stage_id,
                &["prepare", "execute", "postprocess", "results"][..],
            ),
        ] {
            let matches: Vec<_> = context
                .pipeline
                .iter()
                .filter(|stage| &stage.id == id)
                .collect();
            ensure!(
                !id.trim().is_empty()
                    && matches.len() == 1
                    && phases.contains(&matches[0].phase.as_str()),
                "scatter target '{id}' must name a valid pipeline stage"
            );
        }
        for child in &self.children {
            ensure!(
                !child.operation.trim().is_empty(),
                "scatter child operation must be non-empty"
            );
            if let Some(profile) = &child.resource_profile {
                ensure!(
                    profile.get("gpus_required").and_then(Value::as_u64) == Some(1),
                    "scatter child gpus_required must be exactly 1"
                );
                if let Some(value) = profile.get("memory_mb") {
                    ensure!(
                        value.as_u64().is_some_and(|v| v >= 1),
                        "scatter child memory_mb must be an integer >= 1"
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_projection_is_allowlisted_by_target() {
        let payload = serde_json::json!({
            "run_id": "parent",
            "workflow_id": "generic",
            "manifest_version": "1",
            "parameters": {"value": 1},
            "stage_context": {"current_phase": "execute"},
            "batch_profile": {"enabled": true},
            "allocation_id": "must-not-leak",
            "future_scatter_field": "must-not-leak"
        });

        let child = project_scatter_payload(&payload, ScatterPayloadTarget::Child).unwrap();
        let continuation =
            project_scatter_payload(&payload, ScatterPayloadTarget::Continuation).unwrap();

        assert_eq!(child["manifest_version"], "1");
        assert!(child.get("batch_profile").is_none());
        assert_eq!(continuation["batch_profile"]["enabled"], true);
        for projected in [&child, &continuation] {
            assert!(projected.get("allocation_id").is_none());
            assert!(projected.get("future_scatter_field").is_none());
        }
    }

    #[test]
    fn shared_contract_cases() {
        let cases: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/scatter_contract.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            let context: StageContext =
                serde_json::from_value(case["context"]["stage_context"].clone()).unwrap();
            let actual = serde_json::from_value::<ScatterResult>(case["result"].clone())
                .map_err(anyhow::Error::from)
                .and_then(|result| {
                    result.validate(&context, case["context"]["parent_run_id"].as_str())
                });
            assert_eq!(
                actual.is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}: {actual:?}",
                case["name"]
            );
        }
    }
}
