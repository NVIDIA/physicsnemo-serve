/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// A stage descriptor within a pipeline. Each stage has an ID, a phase it
/// belongs to, the destination queue, and an optional pointer to the next stage.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct StageDescriptor {
    pub(crate) id: String,
    pub(crate) phase: String,
    pub(crate) queue: String,
    pub(crate) next: Option<String>,
}

/// Context that travels with each message through the pipeline, tracking the
/// current position and the full list of stages.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct StageContext {
    pub(crate) current_stage_id: String,
    pub(crate) current_phase: String,
    pub(crate) pipeline: Vec<StageDescriptor>,
}

impl StageContext {
    /// Advance to the next stage in the pipeline, validating that the current
    /// phase matches the pipeline definition. `role` is used solely for error
    /// context (e.g. `"prefetch"`, `"collect"`).
    pub(crate) fn next_stage(&self, role: &str) -> Result<StageDescriptor> {
        let current = self
            .pipeline
            .iter()
            .find(|stage| stage.id == self.current_stage_id)
            .ok_or_else(|| {
                anyhow!(
                    "{role}: current stage '{}' not found in pipeline",
                    self.current_stage_id
                )
            })?;

        if current.phase != self.current_phase {
            return Err(anyhow!(
                "{role}: current phase '{}' does not match pipeline phase '{}'",
                self.current_phase,
                current.phase
            ));
        }

        let next_id = current.next.as_ref().ok_or_else(|| {
            anyhow!(
                "{role}: current stage '{}' has no next stage",
                self.current_stage_id
            )
        })?;

        self.pipeline
            .iter()
            .find(|stage| &stage.id == next_id)
            .cloned()
            .ok_or_else(|| anyhow!("{role}: next stage '{next_id}' not found in pipeline"))
    }

    /// Find the first stage that matches the given phase name.
    pub(crate) fn find_phase(&self, phase: &str, role: &str) -> Result<StageDescriptor> {
        self.pipeline
            .iter()
            .find(|stage| stage.phase == phase)
            .cloned()
            .ok_or_else(|| anyhow!("{role}: pipeline is missing required '{phase}' stage"))
    }
}

/// Mutate the `stage_context` object inside a JSON payload map, advancing it
/// to `next_stage`.
pub(crate) fn update_stage_context(
    payload_map: &mut serde_json::Map<String, serde_json::Value>,
    next_stage: &StageDescriptor,
    role: &str,
) -> Result<()> {
    let stage_context = payload_map
        .get_mut("stage_context")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| anyhow!("{role}: payload missing object stage_context"))?;
    stage_context.insert(
        "current_stage_id".to_string(),
        serde_json::Value::String(next_stage.id.clone()),
    );
    stage_context.insert(
        "current_phase".to_string(),
        serde_json::Value::String(next_stage.phase.clone()),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_stage_pipeline() -> StageContext {
        StageContext {
            current_stage_id: "prefetch".to_string(),
            current_phase: "prepare".to_string(),
            pipeline: vec![
                StageDescriptor {
                    id: "prefetch".to_string(),
                    phase: "prepare".to_string(),
                    queue: "prefetch_q".to_string(),
                    next: Some("schedule".to_string()),
                },
                StageDescriptor {
                    id: "schedule".to_string(),
                    phase: "execute".to_string(),
                    queue: "schedule_q".to_string(),
                    next: None,
                },
            ],
        }
    }

    #[test]
    fn next_stage_advances_correctly() {
        let ctx = two_stage_pipeline();
        let next = ctx.next_stage("test").unwrap();
        assert_eq!(next.id, "schedule");
        assert_eq!(next.queue, "schedule_q");
    }

    #[test]
    fn next_stage_errors_when_current_stage_not_found() {
        let mut ctx = two_stage_pipeline();
        ctx.current_stage_id = "nonexistent".to_string();
        let err = ctx.next_stage("test").unwrap_err();
        assert!(
            err.to_string().contains("not found in pipeline"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn next_stage_errors_on_phase_mismatch() {
        let mut ctx = two_stage_pipeline();
        ctx.current_phase = "wrong_phase".to_string();
        let err = ctx.next_stage("test").unwrap_err();
        assert!(
            err.to_string().contains("does not match pipeline phase"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn next_stage_errors_when_no_next() {
        let mut ctx = two_stage_pipeline();
        ctx.current_stage_id = "schedule".to_string();
        ctx.current_phase = "execute".to_string();
        let err = ctx.next_stage("test").unwrap_err();
        assert!(
            err.to_string().contains("has no next stage"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn find_phase_returns_matching_stage() {
        let ctx = two_stage_pipeline();
        let stage = ctx.find_phase("execute", "test").unwrap();
        assert_eq!(stage.id, "schedule");
    }

    #[test]
    fn find_phase_errors_when_missing() {
        let ctx = two_stage_pipeline();
        let err = ctx.find_phase("nonexistent", "test").unwrap_err();
        assert!(
            err.to_string().contains("missing required"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn update_stage_context_mutates_payload() {
        let next = StageDescriptor {
            id: "schedule".to_string(),
            phase: "execute".to_string(),
            queue: "schedule_q".to_string(),
            next: None,
        };
        let mut map = serde_json::Map::new();
        let mut sc = serde_json::Map::new();
        sc.insert(
            "current_stage_id".to_string(),
            serde_json::Value::String("old".to_string()),
        );
        sc.insert(
            "current_phase".to_string(),
            serde_json::Value::String("old_phase".to_string()),
        );
        map.insert("stage_context".to_string(), serde_json::Value::Object(sc));

        update_stage_context(&mut map, &next, "test").unwrap();

        let sc = map["stage_context"].as_object().unwrap();
        assert_eq!(sc["current_stage_id"], "schedule");
        assert_eq!(sc["current_phase"], "execute");
    }

    #[test]
    fn update_stage_context_errors_when_missing() {
        let next = StageDescriptor {
            id: "x".to_string(),
            phase: "y".to_string(),
            queue: "q".to_string(),
            next: None,
        };
        let mut map = serde_json::Map::new();
        let err = update_stage_context(&mut map, &next, "test").unwrap_err();
        assert!(
            err.to_string()
                .contains("payload missing object stage_context"),
            "unexpected error: {err}"
        );
    }
}
