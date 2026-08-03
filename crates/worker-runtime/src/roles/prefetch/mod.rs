/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

pub(crate) mod download;
pub(crate) mod materializer;
pub(crate) mod plan;
pub(crate) mod prefetch_config;

#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use tracing::{info, warn};

use crate::config::{PrefetchRoleConfig, parse_role_config};
use crate::traits::{BoxFuture, MessageSink, RoleEnv, WorkerRole};

use self::materializer::HttpPlanMaterializer;
use self::prefetch_config::PrefetchConfig;

pub use self::download::{DownloadStats, MaterializationResult, MaterializedArtifact};
pub use self::materializer::PlanMaterializer;
pub use self::plan::{ByteRange, PrefetchPlanItem};

/// Materialize a plugin prefetch plan without queue or role infrastructure.
///
/// This queue-independent boundary is used by one-shot direct inference.
pub async fn materialize_prefetch_plan(
    plan: &[PrefetchPlanItem],
    cache_root: &std::path::Path,
    run_id: &str,
) -> Result<MaterializationResult> {
    HttpPlanMaterializer::new(PrefetchConfig::from_env())
        .materialize(plan, cache_root, run_id)
        .await
}

#[derive(Debug, Clone, Deserialize)]
struct PrefetchPayload {
    workflow_id: String,
    #[serde(default)]
    prefetch_plan: Vec<PrefetchPlanItem>,
    stage_context: StageContext,
}

use crate::roles::stage::{StageContext, StageDescriptor};

pub struct PrefetchRole {
    fail_on_plan_generation_error: bool,
    materializer: Arc<dyn PlanMaterializer>,
    cache_root: PathBuf,
    input_streams: Vec<String>,
}

impl PrefetchRole {
    pub fn from_env(
        env: &RoleEnv,
        materializer_override: Option<Arc<dyn PlanMaterializer>>,
    ) -> Result<Self> {
        let cfg: PrefetchRoleConfig = parse_role_config(env.role_config.as_ref())?;
        let prefetch_config = PrefetchConfig::from_env();
        let materializer: Arc<dyn PlanMaterializer> = materializer_override
            .unwrap_or_else(|| Arc::new(HttpPlanMaterializer::new(prefetch_config.clone())));

        Ok(Self {
            fail_on_plan_generation_error: cfg.fail_on_plan_generation_error,
            materializer,
            cache_root: prefetch_config.ext_cache_dir,
            input_streams: env.inputs.iter().map(|spec| spec.stream.clone()).collect(),
        })
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        _output_stream: &str,
        materializer: Arc<dyn PlanMaterializer>,
    ) -> Self {
        Self {
            fail_on_plan_generation_error: false,
            materializer,
            cache_root: std::path::PathBuf::from("/tmp/test-cache"),
            input_streams: vec!["prefetch".to_string()],
        }
    }

    async fn process_message(
        &self,
        msg: &scicomp_rq::Message,
        sink: &dyn MessageSink,
    ) -> Result<()> {
        let run_id = msg.run_id();
        let (typed, mut payload) = decode_prefetch_payload(msg.payload())?;
        let next_stage = typed.stage_context.next_stage("prefetch")?;

        info!(
            run_id = %run_id,
            workflow_id = %typed.workflow_id,
            plan_items = typed.prefetch_plan.len(),
            "starting plugin-driven prefetch stage"
        );

        let materialized = self
            .materializer
            .materialize(&typed.prefetch_plan, &self.cache_root, run_id)
            .await
            .context("prefetch: materializer failed")?;

        if materialized.stats.required_errors > 0
            && (self.fail_on_plan_generation_error
                || materialized.stats.required_verified_errors > 0)
        {
            return Err(anyhow!(
                "prefetch: required download failures: {}",
                materialized.stats.required_errors
            ));
        }

        if materialized.stats.errors > 0 {
            warn!(
                run_id = %run_id,
                required_errors = materialized.stats.required_errors,
                optional_errors = materialized.stats.optional_errors,
                "prefetch completed with degraded materialization"
            );
        }

        apply_prefetch_result(
            &mut payload,
            &typed.prefetch_plan,
            &materialized,
            &next_stage,
        )?;

        let updated_payload =
            serde_json::to_string(&payload).context("prefetch: failed to encode payload")?;
        sink.handoff(msg, &next_stage.queue, &updated_payload, &next_stage.phase)
            .await
            .with_context(|| {
                format!(
                    "prefetch: failed to hand off workflow '{}' to next stage '{}'",
                    typed.workflow_id, next_stage.phase
                )
            })?;

        Ok(())
    }
}

fn decode_prefetch_payload(raw: &str) -> Result<(PrefetchPayload, JsonValue)> {
    if raw.trim().is_empty() {
        return Err(anyhow!("prefetch: empty payload"));
    }

    let value: JsonValue =
        serde_json::from_str(raw).context("prefetch: payload must be valid JSON object")?;
    if !value.is_object() {
        return Err(anyhow!("prefetch: payload must be a JSON object"));
    }
    if let Some(prefetch_plan) = value.get("prefetch_plan")
        && !prefetch_plan.is_array()
    {
        return Err(anyhow!("prefetch: prefetch_plan must be an array"));
    }

    let typed: PrefetchPayload =
        serde_json::from_value(value.clone()).context("prefetch: invalid payload schema")?;
    if typed.workflow_id.trim().is_empty() {
        return Err(anyhow!(
            "prefetch: workflow_id is required and must be non-empty"
        ));
    }
    if typed.stage_context.current_phase != "prefetch" {
        return Err(anyhow!(
            "prefetch: payload current_phase must be 'prefetch', got '{}'",
            typed.stage_context.current_phase
        ));
    }
    Ok((typed, value))
}

fn apply_prefetch_result(
    payload: &mut JsonValue,
    plan: &[PrefetchPlanItem],
    materialized: &MaterializationResult,
    next_stage: &StageDescriptor,
) -> Result<()> {
    let map = payload
        .as_object_mut()
        .ok_or_else(|| anyhow!("prefetch: payload must remain a JSON object"))?;

    map.insert(
        "prefetch_downloaded".to_string(),
        JsonValue::Number(materialized.stats.downloaded.into()),
    );
    map.insert(
        "prefetch_cached".to_string(),
        JsonValue::Number(materialized.stats.cached.into()),
    );
    map.insert(
        "prefetch_errors".to_string(),
        JsonValue::Number(materialized.stats.errors.into()),
    );
    map.insert(
        "prefetch_required_errors".to_string(),
        JsonValue::Number(materialized.stats.required_errors.into()),
    );
    map.insert(
        "prefetch_optional_errors".to_string(),
        JsonValue::Number(materialized.stats.optional_errors.into()),
    );
    map.insert(
        "prefetch_plan_count".to_string(),
        JsonValue::Number((plan.len() as u64).into()),
    );
    map.insert("prefetch_plan".to_string(), serde_json::to_value(plan)?);
    map.insert(
        "prefetch_artifacts".to_string(),
        serde_json::to_value(&materialized.artifacts)?,
    );
    if materialized.stats.errors > 0 {
        map.insert("prefetch_degraded".to_string(), JsonValue::Bool(true));
    }

    let stage_context = map
        .get_mut("stage_context")
        .and_then(JsonValue::as_object_mut)
        .ok_or_else(|| anyhow!("prefetch: payload missing object stage_context"))?;
    stage_context.insert(
        "current_stage_id".to_string(),
        JsonValue::String(next_stage.id.clone()),
    );
    stage_context.insert(
        "current_phase".to_string(),
        JsonValue::String(next_stage.phase.clone()),
    );

    Ok(())
}

impl WorkerRole for PrefetchRole {
    fn name(&self) -> &'static str {
        "prefetch"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        stream: &'a str,
        sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if !self.input_streams.is_empty() && !self.input_streams.iter().any(|s| s == stream) {
                return Err(anyhow!(
                    "prefetch: unexpected stream '{}', expected one of {:?}",
                    stream,
                    self.input_streams
                ));
            }
            self.process_message(msg, sink).await
        })
    }
}
