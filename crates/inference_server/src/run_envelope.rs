/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::config::ResolvedOutputPublication;
use crate::plugin_registry::RegisteredPlugin;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEnvelope {
    pub run_id: String,
    pub workflow_id: String,
    pub operation: String,
    pub manifest_version: String,
    pub request: RunRequest,
    pub parameters: Value,
    pub resource_profile: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_profile: Option<Value>,
    pub prefetch_plan: Vec<Value>,
    pub stage_context: StageContext,
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_publication: Option<ResolvedOutputPublication>,
    pub runtime: RuntimeTarget,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRequest {
    pub content_type: String,
    pub raw_fields: Value,
    pub input_artifacts: Vec<ArtifactRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub field_name: String,
    pub name: String,
    pub artifact_id: String,
    pub media_type: String,
    pub size_bytes: u64,
    pub storage_path: String,
    pub original_filename: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageContext {
    pub current_stage_id: String,
    pub current_phase: String,
    pub pipeline: Vec<StageDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageDescriptor {
    pub id: String,
    pub phase: String,
    pub handler: String,
    pub queue: String,
    pub next: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTarget {
    pub kind: String,
    pub entrypoint: String,
    pub executor_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prepare_executor_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_executor_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readiness_executor_class: Option<String>,
}

impl RunEnvelope {
    pub fn for_plugin(
        plugin: &RegisteredPlugin,
        run_id: String,
        operation: String,
        request: RunRequest,
        parameters: Value,
        use_prefetch: bool,
    ) -> Result<Self> {
        Self::for_plugin_with_publication(
            plugin,
            run_id,
            operation,
            request,
            parameters,
            use_prefetch,
            None,
        )
    }

    pub fn for_plugin_with_publication(
        plugin: &RegisteredPlugin,
        run_id: String,
        operation: String,
        request: RunRequest,
        parameters: Value,
        use_prefetch: bool,
        output_publication: Option<ResolvedOutputPublication>,
    ) -> Result<Self> {
        let pipeline = build_stage_pipeline(plugin, use_prefetch, output_publication.as_ref())?;
        let first_stage = pipeline.first().ok_or_else(|| {
            anyhow!(
                "plugin '{}' must define at least one pipeline stage",
                plugin.manifest.metadata.id
            )
        })?;

        Ok(Self {
            run_id,
            workflow_id: plugin.manifest.metadata.id.clone(),
            operation,
            manifest_version: plugin.manifest.metadata.version.clone(),
            request,
            parameters,
            resource_profile: {
                let defaults = &plugin.manifest.resources.defaults;
                if defaults.memory_mb > 0 {
                    serde_json::json!({
                        "executor_class": plugin.manifest.runtime.executor_class,
                        "gpus_required": defaults.gpus_required,
                        "memory_mb": defaults.memory_mb,
                    })
                } else {
                    serde_json::Value::Null
                }
            },
            batch_profile: None,
            prefetch_plan: Vec::new(),
            stage_context: StageContext {
                current_stage_id: first_stage.id.clone(),
                current_phase: first_stage.phase.clone(),
                pipeline,
            },
            result: None,
            output_publication,
            runtime: RuntimeTarget {
                kind: plugin.manifest.runtime.kind.clone(),
                entrypoint: plugin.manifest.runtime.entrypoint.clone(),
                executor_class: plugin.manifest.runtime.executor_class.clone(),
                prepare_executor_class: plugin.manifest.runtime.prepare_executor_class.clone(),
                postprocess_executor_class: plugin
                    .manifest
                    .runtime
                    .postprocess_executor_class
                    .clone(),
                readiness_executor_class: plugin.manifest.runtime.readiness_executor_class.clone(),
            },
        })
    }
}

fn build_stage_pipeline(
    plugin: &RegisteredPlugin,
    use_prefetch: bool,
    output_publication: Option<&ResolvedOutputPublication>,
) -> Result<Vec<StageDescriptor>> {
    let mut pipeline: Vec<StageDescriptor> = plugin
        .manifest
        .pipeline
        .stages
        .iter()
        .filter(|stage| use_prefetch || stage.phase != "prefetch")
        .map(|stage| StageDescriptor {
            id: stage.id.clone(),
            phase: stage.phase.clone(),
            handler: stage.handler.clone(),
            queue: stage.queue.clone(),
            next: stage.next.clone(),
        })
        .collect();

    if pipeline.is_empty() {
        return Err(anyhow!(
            "plugin '{}' must define at least one pipeline stage",
            plugin.manifest.metadata.id
        ));
    }

    for index in 0..pipeline.len() {
        pipeline[index].next = pipeline.get(index + 1).map(|stage| stage.id.clone());
    }

    let publish_index = pipeline.iter().position(|stage| stage.phase == "publish");
    if output_publication.is_some() || publish_index.is_some() {
        let publish_stage_id = unused_stage_id(&pipeline, "publish");
        let results_stage_id = unused_stage_id(&pipeline, "results");
        let publish_stage = publish_index
            .map(|index| pipeline.remove(index))
            .unwrap_or_else(|| StageDescriptor {
                id: publish_stage_id,
                phase: "publish".to_string(),
                handler: "publish_outputs".to_string(),
                queue: "publish".to_string(),
                next: None,
            });
        let results_stage = pipeline
            .iter()
            .position(|stage| stage.phase == "results")
            .map(|index| pipeline.remove(index))
            .unwrap_or_else(|| StageDescriptor {
                id: results_stage_id,
                phase: "results".to_string(),
                handler: "persist_results".to_string(),
                queue: "results".to_string(),
                next: None,
            });
        pipeline.push(publish_stage);
        pipeline.push(results_stage);

        for index in 0..pipeline.len() {
            pipeline[index].next = pipeline.get(index + 1).map(|stage| stage.id.clone());
        }
    }

    Ok(pipeline)
}

fn unused_stage_id(pipeline: &[StageDescriptor], base: &str) -> String {
    if !pipeline.iter().any(|stage| stage.id == base) {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if !pipeline.iter().any(|stage| stage.id == candidate) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix search must find an unused stage id")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin_registry::{
        DEFAULT_PLUGIN_MANIFEST_NAME, PluginManifest, PluginStage, RegisteredPlugin,
        plugin_manifest_path,
    };
    use std::fs;
    use std::path::PathBuf;

    fn manifest_yaml() -> &'static str {
        r#"
metadata:
  id: demo-prefetch
  display_name: Demo Prefetch
  version: 1.0.0
  description: Generic prefetch plugin example
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: schemas/request.json
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.test
  prepare_executor_class: python.test
  postprocess_executor_class: python.test
  readiness_executor_class: python.test
resources:
  defaults:
    gpus_required: 1
    memory_mb: 24000
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#
    }

    fn temp_plugin_root(test_name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "physicsnemo-serve-run-envelope-{test_name}-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("plugin root should be created");
        fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), manifest_yaml())
            .expect("manifest should be written");
        root
    }

    #[test]
    fn builds_plugin_run_envelope_with_first_stage_context() {
        let root = temp_plugin_root("build");
        let manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin(
            &plugin,
            "run-123".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            true,
        )
        .expect("run envelope should be created");

        assert_eq!(envelope.workflow_id, "demo-prefetch");
        assert_eq!(envelope.stage_context.current_stage_id, "prepare");
        assert_eq!(envelope.stage_context.current_phase, "prepare");
        assert_eq!(envelope.stage_context.pipeline.len(), 3);
        assert_eq!(envelope.runtime.executor_class, "python.test");
        assert_eq!(
            envelope.runtime.prepare_executor_class.as_deref(),
            Some("python.test")
        );
        assert_eq!(
            envelope.runtime.postprocess_executor_class.as_deref(),
            Some("python.test")
        );
        assert_eq!(
            envelope.runtime.readiness_executor_class.as_deref(),
            Some("python.test")
        );
        assert_eq!(envelope.request.content_type, "application/json");
        assert_eq!(envelope.parameters, serde_json::json!({"num_steps": 20}));
        assert_eq!(
            envelope.resource_profile,
            serde_json::json!({
                "executor_class": "python.test",
                "gpus_required": 1,
                "memory_mb": 24000,
            })
        );
    }

    #[test]
    fn rejects_plugin_without_pipeline_stages() {
        let root = temp_plugin_root("empty-stages");
        let mut manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        manifest.pipeline.stages.clear();
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let result = std::panic::catch_unwind(|| {
            RunEnvelope::for_plugin(
                &plugin,
                "run-empty".to_string(),
                "run".to_string(),
                RunRequest {
                    content_type: "application/json".to_string(),
                    raw_fields: serde_json::json!({}),
                    input_artifacts: Vec::new(),
                },
                serde_json::json!({}),
                true,
            )
        });
        let envelope_result = result.expect("empty pipelines must not panic");
        assert!(
            format!(
                "{:#}",
                envelope_result.expect_err("empty pipelines must be rejected")
            )
            .contains("at least one pipeline stage"),
            "unexpected empty-stage error"
        );
    }

    #[test]
    fn removes_prefetch_stage_from_pipeline_when_prefetch_disabled() {
        let root = temp_plugin_root("prefetch-disabled");
        let manifest = PluginManifest::from_yaml_str(
            r#"
metadata:
  id: demo-prefetch
  display_name: Demo Prefetch
  version: 1.0.0
  description: Generic prefetch plugin example
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: schemas/request.json
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
      next: prefetch
    - id: prefetch
      phase: prefetch
      handler: prefetch
      queue: prefetch
      next: schedule
    - id: schedule
      phase: schedule
      handler: schedule
      queue: schedule
      next: execute
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.python.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: python
  entrypoint: plugin.py
  executor_class: python.test
resources:
  defaults:
    gpus_required: 1
    memory_mb: 24000
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
        )
        .expect("manifest should parse");
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin(
            &plugin,
            "run-123".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            false,
        )
        .expect("run envelope should be created");

        let phases: Vec<&str> = envelope
            .stage_context
            .pipeline
            .iter()
            .map(|stage| stage.phase.as_str())
            .collect();
        assert_eq!(phases, vec!["prepare", "schedule", "execute", "results"]);
        assert_eq!(envelope.stage_context.current_stage_id, "prepare");
        assert_eq!(
            envelope.stage_context.pipeline[0].next.as_deref(),
            Some("schedule")
        );
    }

    #[test]
    fn injects_publish_stage_before_results_when_output_publication_is_enabled() {
        let root = temp_plugin_root("publish-stage");
        let manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin_with_publication(
            &plugin,
            "run-publish".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            true,
            Some(crate::config::ResolvedOutputPublication {
                target: crate::config::ResolvedOutputPublicationTarget {
                    artifact: "primary".to_string(),
                    provider: crate::config::OutputPublicationProvider::S3,
                    storage: crate::config::ResolvedOutputPublicationStorage::S3 {
                        bucket: "bucket".to_string(),
                        prefix: "outputs/demo-prefetch/run-publish".to_string(),
                        region: None,
                        endpoint: None,
                    },
                },
            }),
        )
        .expect("run envelope should be created");

        let phases: Vec<&str> = envelope
            .stage_context
            .pipeline
            .iter()
            .map(|stage| stage.phase.as_str())
            .collect();
        assert_eq!(phases, vec!["prepare", "execute", "publish", "results"]);
        assert_eq!(
            envelope.stage_context.pipeline[1].next.as_deref(),
            Some("publish")
        );
        assert_eq!(
            envelope.stage_context.pipeline[2].next.as_deref(),
            Some("results")
        );
        assert!(envelope.output_publication.is_some());
    }

    #[test]
    fn injects_publish_stage_for_terminal_execute_publication_pipeline() {
        let root = temp_plugin_root("terminal-execute-publish");
        let mut manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        manifest
            .pipeline
            .stages
            .retain(|stage| stage.phase != "results");
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin_with_publication(
            &plugin,
            "run-terminal-publish".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            true,
            Some(crate::config::ResolvedOutputPublication {
                target: crate::config::ResolvedOutputPublicationTarget {
                    artifact: "primary".to_string(),
                    provider: crate::config::OutputPublicationProvider::S3,
                    storage: crate::config::ResolvedOutputPublicationStorage::S3 {
                        bucket: "bucket".to_string(),
                        prefix: "outputs/demo-prefetch/run-terminal-publish".to_string(),
                        region: None,
                        endpoint: None,
                    },
                },
            }),
        )
        .expect("run envelope should be created");

        let phases: Vec<&str> = envelope
            .stage_context
            .pipeline
            .iter()
            .map(|stage| stage.phase.as_str())
            .collect();
        assert_eq!(phases, vec!["prepare", "execute", "publish", "results"]);
        assert_eq!(
            envelope.stage_context.pipeline[1].next.as_deref(),
            Some("publish")
        );
        assert_eq!(
            envelope.stage_context.pipeline[2].next.as_deref(),
            Some("results")
        );
    }

    #[test]
    fn does_not_duplicate_manifest_declared_publish_stage() {
        let root = temp_plugin_root("manifest-publish-stage");
        let mut manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        manifest.pipeline.stages.insert(
            2,
            PluginStage {
                id: "publish".to_string(),
                phase: "publish".to_string(),
                handler: "plugin_phase".to_string(),
                queue: "publish.custom".to_string(),
                next: Some("results".to_string()),
            },
        );
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin_with_publication(
            &plugin,
            "run-publish".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            true,
            Some(crate::config::ResolvedOutputPublication {
                target: crate::config::ResolvedOutputPublicationTarget {
                    artifact: "primary".to_string(),
                    provider: crate::config::OutputPublicationProvider::S3,
                    storage: crate::config::ResolvedOutputPublicationStorage::S3 {
                        bucket: "bucket".to_string(),
                        prefix: "outputs/demo-prefetch/run-publish".to_string(),
                        region: None,
                        endpoint: None,
                    },
                },
            }),
        )
        .expect("run envelope should be created");

        let publish_stages: Vec<_> = envelope
            .stage_context
            .pipeline
            .iter()
            .filter(|stage| stage.phase == "publish")
            .collect();
        assert_eq!(publish_stages.len(), 1);
        assert_eq!(publish_stages[0].id, "publish");
        assert_eq!(publish_stages[0].queue, "publish.custom");
        assert_eq!(
            envelope.stage_context.pipeline[1].next.as_deref(),
            Some("publish")
        );
        assert_eq!(publish_stages[0].next.as_deref(), Some("results"));
    }

    #[test]
    fn normalizes_misplaced_manifest_publish_before_terminal_results() {
        let stage = |id: &str, phase: &str| PluginStage {
            id: id.to_string(),
            phase: phase.to_string(),
            handler: "plugin_phase".to_string(),
            queue: phase.to_string(),
            next: None,
        };
        let cases = [
            (
                "publish-after-results",
                vec![
                    stage("execute", "execute"),
                    stage("results", "results"),
                    stage("publish", "publish"),
                ],
                vec!["execute", "publish", "results"],
            ),
            (
                "publish-before-postprocess",
                vec![
                    stage("execute", "execute"),
                    stage("publish", "publish"),
                    stage("postprocess", "postprocess"),
                    stage("results", "results"),
                ],
                vec!["execute", "postprocess", "publish", "results"],
            ),
        ];
        let publication = crate::config::ResolvedOutputPublication {
            target: crate::config::ResolvedOutputPublicationTarget {
                artifact: "primary".to_string(),
                provider: crate::config::OutputPublicationProvider::S3,
                storage: crate::config::ResolvedOutputPublicationStorage::S3 {
                    bucket: "bucket".to_string(),
                    prefix: "outputs/demo-prefetch/run-publish".to_string(),
                    region: None,
                    endpoint: None,
                },
            },
        };

        for (test_name, stages, expected_phases) in cases {
            let root = temp_plugin_root(test_name);
            let mut manifest =
                PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
            manifest.pipeline.stages = stages;
            let plugin = RegisteredPlugin {
                root_dir: root.clone(),
                manifest_path: plugin_manifest_path(&root),
                manifest,
            };

            let pipeline = build_stage_pipeline(&plugin, true, Some(&publication))
                .expect("pipeline should be created");
            let phases: Vec<&str> = pipeline.iter().map(|stage| stage.phase.as_str()).collect();

            assert_eq!(phases, expected_phases, "case: {test_name}");
        }
    }

    #[test]
    fn adds_results_stage_after_manifest_declared_publish_without_results() {
        let root = temp_plugin_root("manifest-publish-no-results");
        let mut manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        manifest
            .pipeline
            .stages
            .retain(|stage| stage.phase != "results");
        manifest.pipeline.stages.push(PluginStage {
            id: "publish".to_string(),
            phase: "publish".to_string(),
            handler: "publish_outputs".to_string(),
            queue: "publish".to_string(),
            next: None,
        });
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin_with_publication(
            &plugin,
            "run-publish".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            true,
            Some(crate::config::ResolvedOutputPublication {
                target: crate::config::ResolvedOutputPublicationTarget {
                    artifact: "primary".to_string(),
                    provider: crate::config::OutputPublicationProvider::S3,
                    storage: crate::config::ResolvedOutputPublicationStorage::S3 {
                        bucket: "bucket".to_string(),
                        prefix: "outputs/demo-prefetch/run-publish".to_string(),
                        region: None,
                        endpoint: None,
                    },
                },
            }),
        )
        .expect("run envelope should be created");

        let phases: Vec<&str> = envelope
            .stage_context
            .pipeline
            .iter()
            .map(|stage| stage.phase.as_str())
            .collect();
        assert_eq!(phases, vec!["prepare", "execute", "publish", "results"]);
        let publish = envelope
            .stage_context
            .pipeline
            .iter()
            .find(|stage| stage.phase == "publish")
            .expect("publish stage should remain");
        assert_eq!(publish.next.as_deref(), Some("results"));
    }

    #[test]
    fn adds_results_stage_after_manifest_publish_when_publication_is_disabled() {
        let root = temp_plugin_root("manifest-publish-no-results-publication-disabled");
        let mut manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        manifest
            .pipeline
            .stages
            .retain(|stage| stage.phase != "results");
        manifest.pipeline.stages.push(PluginStage {
            id: "publish".to_string(),
            phase: "publish".to_string(),
            handler: "publish_outputs".to_string(),
            queue: "publish".to_string(),
            next: None,
        });
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin_with_publication(
            &plugin,
            "run-publish-disabled".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            true,
            None,
        )
        .expect("run envelope should be created");

        let phases: Vec<&str> = envelope
            .stage_context
            .pipeline
            .iter()
            .map(|stage| stage.phase.as_str())
            .collect();
        assert_eq!(phases, vec!["prepare", "execute", "publish", "results"]);
        let publish = envelope
            .stage_context
            .pipeline
            .iter()
            .find(|stage| stage.phase == "publish")
            .expect("publish stage should remain");
        assert_eq!(publish.next.as_deref(), Some("results"));
        assert!(envelope.output_publication.is_none());
    }

    #[test]
    fn injected_publish_stage_uses_unused_id_when_publish_id_is_taken() {
        let root = temp_plugin_root("publish-id-collision");
        let mut manifest =
            PluginManifest::from_yaml_str(manifest_yaml()).expect("manifest should parse");
        manifest.pipeline.stages.insert(
            2,
            PluginStage {
                id: "publish".to_string(),
                phase: "postprocess".to_string(),
                handler: "plugin_phase".to_string(),
                queue: "postprocess.custom".to_string(),
                next: Some("results".to_string()),
            },
        );
        let plugin = RegisteredPlugin {
            root_dir: root.clone(),
            manifest_path: plugin_manifest_path(&root),
            manifest,
        };

        let envelope = RunEnvelope::for_plugin_with_publication(
            &plugin,
            "run-publish".to_string(),
            "run".to_string(),
            RunRequest {
                content_type: "application/json".to_string(),
                raw_fields: serde_json::json!({"parameters": {"num_steps": 20}}),
                input_artifacts: Vec::new(),
            },
            serde_json::json!({"num_steps": 20}),
            true,
            Some(crate::config::ResolvedOutputPublication {
                target: crate::config::ResolvedOutputPublicationTarget {
                    artifact: "primary".to_string(),
                    provider: crate::config::OutputPublicationProvider::S3,
                    storage: crate::config::ResolvedOutputPublicationStorage::S3 {
                        bucket: "bucket".to_string(),
                        prefix: "outputs/demo-prefetch/run-publish".to_string(),
                        region: None,
                        endpoint: None,
                    },
                },
            }),
        )
        .expect("run envelope should be created");

        let publish_stages: Vec<_> = envelope
            .stage_context
            .pipeline
            .iter()
            .filter(|stage| stage.phase == "publish")
            .collect();
        assert_eq!(publish_stages.len(), 1);
        assert_eq!(publish_stages[0].id, "publish_1");
        assert_eq!(
            envelope.stage_context.pipeline[2].next.as_deref(),
            Some("publish_1")
        );
        assert_eq!(publish_stages[0].next.as_deref(), Some("results"));
    }
}
