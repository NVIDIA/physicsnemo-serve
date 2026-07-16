/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static PYTHON_ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(key);
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.previous {
                Some(prev) => std::env::set_var(self.key, prev),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn create_test_env(vars: Vec<(&str, &str)>) -> impl Fn(&str) -> Option<String> {
    let mut env_map = HashMap::new();
    for (key, value) in vars {
        env_map.insert(key.to_string(), value.to_string());
    }
    move |key: &str| env_map.get(key).cloned()
}

fn minimal_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: demo-prefetch
  display_name: Demo Prefetch Plugin
  version: 1.0.0
  description: Generic plugin manifest fixture
  tags: [demo, plugin]
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: schemas/request.json
  files: []
pipeline:
  stages:
    - id: prepare
      phase: prepare
      handler: plugin_phase
      queue: prepare
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
  hook_timeout_seconds:
    prepare: 15
    execute: 300
    postprocess: 0
resources:
  defaults:
    gpus_required: 0
    memory_mb: 1024
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#
}

fn manifest_yaml_with_id(id: &str) -> String {
    minimal_manifest_yaml().replacen("demo-prefetch", id, 1)
}

fn compact_profile_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: demo-compact
  display_name: Demo Compact
  version: 1.0.0
  description: Compact preset-driven plugin manifest
pipeline:
  profile: simple
runtime:
  profile: python-test
"#
}

fn postprocess_profile_manifest_yaml() -> &'static str {
    r#"
metadata:
  id: python-postprocess-demo
  display_name: Python Test Postprocess Demo
  version: 1.0.0
  description: Postprocess plugin using the built-in python-test runtime.
ingress:
  content_type: multipart/form-data
  request_schema: schemas/request.multipart.json
  operation:
    default: both
    allowed: [both, volume]
  files:
    - name: design_stl
      required: true
      media_types: [application/octet-stream]
      max_size_mb: 256
pipeline:
  profile: postprocess
runtime:
  profile: python-test
  phases:
    prepare: python-test
    postprocess: python-test
    readiness: python-test
outputs:
  result_schema: schemas/result.json
"#
}

fn plugin_registry_test_root() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join("plugin-registry-tests");
    fs::create_dir_all(&root).expect("plugin registry test root should be created");
    root
}

fn unique_temp_dir(test_name: &str) -> PathBuf {
    let dir = plugin_registry_test_root().join(format!(
        "physicsnemo-serve-plugin-registry-{test_name}-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&dir).expect("temp directory should be created");
    dir
}

fn write_executable_script(path: &Path, content: &str) {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("script path should end with a valid UTF-8 file name");
    let temp_path = path.with_file_name(format!(".{file_name}.{}.tmp", uuid::Uuid::new_v4()));
    let mut file = fs::File::create(&temp_path).expect("script file should be created");
    file.write_all(content.as_bytes())
        .expect("script content should be written");
    file.sync_all().expect("script should be synced to disk");
    drop(file);
    fs::set_permissions(&temp_path, fs::Permissions::from_mode(0o755))
        .expect("script should be executable");
    fs::rename(&temp_path, path).expect("script should be atomically installed");
}

fn write_fake_python_runtime(dir: &Path, name: &str, stdout_json: &str) -> PathBuf {
    let script_path = dir.join(name);
    write_executable_script(
        &script_path,
        &format!(
            "#!/bin/sh\nprintf 'PHYSICSNEMO_SERVE_JSON:%s\\n' '{}'\n",
            stdout_json.replace('\'', "'\"'\"'")
        ),
    );
    script_path
}

#[test]
fn manifest_from_yaml_parses_valid_plugin() {
    let manifest = PluginManifest::from_yaml_str(minimal_manifest_yaml())
        .expect("valid manifest should parse");

    assert_eq!(manifest.metadata.id, "demo-prefetch");
    assert_eq!(manifest.ingress.default_content_type, "application/json");
    assert_eq!(manifest.pipeline.stages.len(), 4);
    assert_eq!(manifest.runtime.executor_class, "python.test");
    assert_eq!(manifest.outputs.primary_artifact.name, "primary");
}

#[test]
fn manifest_from_yaml_expands_compact_profile_defaults() {
    let manifest = PluginManifest::from_yaml_str(compact_profile_manifest_yaml())
        .expect("compact manifest should parse");

    assert_eq!(manifest.ingress.default_content_type, "application/json");
    assert_eq!(manifest.ingress.operations.default, "run");
    assert_eq!(manifest.runtime.kind, "python");
    assert_eq!(manifest.runtime.entrypoint, "workflow.py");
    assert_eq!(manifest.runtime.executor_class, "python.test");
    assert_eq!(manifest.pipeline.stages.len(), 3);
    assert_eq!(manifest.resources.defaults.gpus_required, 0);
    assert_eq!(manifest.outputs.primary_artifact.name, "primary");
    assert_eq!(
        manifest
            .developer
            .readiness
            .recommended_check_phase
            .as_deref(),
        Some("execute")
    );
}

#[test]
fn manifest_from_yaml_accepts_default_pipeline_alias() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: demo-default
  display_name: Demo Default
  version: 1.0.0
  description: Compact preset-driven plugin manifest
pipeline:
  profile: default
runtime:
  profile: python-test
"#,
    )
    .expect("default pipeline alias should parse");

    assert_eq!(manifest.pipeline.stages.len(), 5);
    assert_eq!(manifest.pipeline.stages[0].phase, "prepare");
    assert_eq!(manifest.pipeline.stages[1].phase, "prefetch");
    assert_eq!(manifest.pipeline.stages[2].phase, "schedule");
    assert_eq!(manifest.pipeline.stages[3].phase, "execute");
    assert_eq!(manifest.pipeline.stages[4].phase, "results");
    assert_eq!(
        manifest
            .developer
            .readiness
            .recommended_check_phase
            .as_deref(),
        Some("prepare")
    );
}

#[test]
fn manifest_from_yaml_defaults_missing_executor_class() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: custom-default-executor
  display_name: Custom Default Executor
  version: 1.0.0
  description: Plugin relying on the internal default executor class
pipeline:
  profile: simple
runtime:
  kind: python
  entrypoint: workflow.py
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
"#,
    )
    .expect("manifest should parse");

    assert_eq!(manifest.runtime.executor_class, "default");
    assert_eq!(manifest.pipeline.stages[1].queue, "execute.default");
}

#[test]
fn manifest_preserves_plugin_configuration() {
    let yaml = format!(
        "{}\nconfiguration:\n  provider:\n    name: physicsnemo-cfd\n    version: 0.0.2\n  benchmark:\n    domain: surface\n",
        minimal_manifest_yaml()
    );
    let manifest = PluginManifest::from_yaml_str(&yaml).expect("manifest should parse");

    manifest.validate().expect("configuration should validate");
    assert_eq!(
        manifest.configuration["provider"]["name"],
        "physicsnemo-cfd"
    );
    assert_eq!(manifest.configuration["benchmark"]["domain"], "surface");

    let encoded = serde_json::to_value(&manifest).expect("manifest should serialize");
    assert_eq!(encoded["configuration"]["provider"]["version"], "0.0.2");
}

#[test]
fn manifest_validation_rejects_scalar_configuration() {
    let yaml = format!("{}\nconfiguration: unsafe\n", minimal_manifest_yaml());
    let manifest = PluginManifest::from_yaml_str(&yaml).expect("manifest should parse");

    let error = manifest
        .validate()
        .expect_err("configuration must be a mapping");
    assert!(
        error
            .to_string()
            .contains("configuration must be an object")
    );
}

#[test]
fn manifest_parsing_rejects_null_configuration() {
    let yaml = format!("{}\nconfiguration: null\n", minimal_manifest_yaml());

    let error = PluginManifest::from_yaml_str(&yaml)
        .expect_err("an explicit null configuration must be rejected");
    assert!(
        error
            .to_string()
            .contains("configuration must be an object")
    );
}

#[test]
fn manifest_from_yaml_expands_postprocess_profile_defaults() {
    let manifest = PluginManifest::from_yaml_str(postprocess_profile_manifest_yaml())
        .expect("postprocess profile manifest should parse");

    assert_eq!(
        manifest.ingress.form_schema.as_deref(),
        Some("schemas/request.multipart.json")
    );
    assert_eq!(manifest.runtime.executor_class, "python.test");
    assert_eq!(
        manifest.runtime.prepare_executor_class.as_deref(),
        Some("python.test")
    );
    assert_eq!(
        manifest.runtime.postprocess_executor_class.as_deref(),
        Some("python.test")
    );
    assert_eq!(manifest.pipeline.stages.len(), 5);
    assert_eq!(manifest.pipeline.stages[3].phase, "postprocess");
    assert_eq!(manifest.resources.defaults.memory_mb, 1024);
    assert!(manifest.developer.readiness.python_modules.is_empty());
    assert!(manifest.developer.readiness.env.is_empty());
}

#[test]
fn manifest_from_yaml_backfills_blank_phase_executor_fields_from_runtime_phases() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: python-phase-backfill
  display_name: Python Test Phase Backfill
  version: 1.0.0
  description: Manifest with blank phase executor fields
pipeline:
  profile: postprocess
runtime:
  profile: python-test
  prepare_executor_class: ""
  readiness_executor_class: "   "
  phases:
    prepare: python-test
    readiness: python-test
outputs:
  result_schema: schemas/result.json
"#,
    )
    .expect("manifest should parse");

    manifest
        .validate()
        .expect("blank phase executor fields should be backfilled from runtime.phases");
    assert_eq!(
        manifest.runtime.prepare_executor_class.as_deref(),
        Some("python.test")
    );
    assert_eq!(
        manifest.runtime.readiness_executor_class.as_deref(),
        Some("python.test")
    );
}

#[test]
fn manifest_validation_rejects_duplicate_stage_ids() {
    let yaml = minimal_manifest_yaml().replace(
        "- id: schedule\n      phase: schedule",
        "- id: prepare\n      phase: schedule",
    );

    let manifest = PluginManifest::from_yaml_str(&yaml).expect("yaml should parse");
    let err = manifest
        .validate()
        .expect_err("duplicate stage ids must be rejected");

    assert!(
        err.to_string().contains("duplicate"),
        "expected duplicate stage id error, got: {err}"
    );
}

#[test]
fn manifest_validation_rejects_unknown_next_stage() {
    let yaml = minimal_manifest_yaml().replace("next: execute", "next: missing_stage");
    let manifest = PluginManifest::from_yaml_str(&yaml).expect("yaml should parse");
    let err = manifest
        .validate()
        .expect_err("unknown next stage must be rejected");

    assert!(
        err.to_string().contains("missing_stage"),
        "expected missing stage reference, got: {err}"
    );
}

#[test]
fn manifest_validation_rejects_unknown_default_operation() {
    let yaml = minimal_manifest_yaml().replace("default: run", "default: predict");
    let manifest = PluginManifest::from_yaml_str(&yaml).expect("yaml should parse");
    let err = manifest
        .validate()
        .expect_err("default operation must be in the allowed list");

    assert!(
        err.to_string().contains("default operation"),
        "expected invalid default operation error, got: {err}"
    );
}

#[test]
fn manifest_validation_rejects_unknown_default_content_type() {
    let yaml = minimal_manifest_yaml().replace(
        "default_content_type: application/json",
        "default_content_type: text/plain",
    );
    let manifest = PluginManifest::from_yaml_str(&yaml).expect("yaml should parse");
    let err = manifest
        .validate()
        .expect_err("default content type must be in the supported content type list");

    assert!(
        err.to_string().contains("default content type"),
        "expected invalid default content type error, got: {err}"
    );
}

#[test]
fn manifest_validation_rejects_file_upload_fields_without_multipart_content_type() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: file-upload-no-multipart
  display_name: File Upload No Multipart
  version: 1.0.0
  description: File upload fields must declare multipart content types
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
  files:
    - name: upload
      required: true
      media_types: [application/octet-stream]
      max_size_mb: 1
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("file uploads without multipart content types must be rejected");

    assert!(
        err.to_string()
            .contains("file upload fields require 'multipart/form-data'"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_file_upload_entry_without_media_types() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: file-upload-no-media-types
  display_name: File Upload No Media Types
  version: 1.0.0
  description: File upload fields must declare media types
ingress:
  content_types: [multipart/form-data]
  default_content_type: multipart/form-data
  operations:
    default: run
    allowed: [run]
  form_schema_inline:
    type: object
  files:
    - name: upload
      required: true
      media_types: []
      max_size_mb: 1
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("file upload entries without media types must be rejected");

    assert!(
        err.to_string()
            .contains("must declare at least one media type"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_empty_pipeline_stage_list() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: empty-pipeline
  display_name: Empty Pipeline
  version: 1.0.0
  description: Pipeline stages must not be empty
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
pipeline:
  stages: []
runtime:
  kind: python
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("manifests must define at least one pipeline stage");

    assert!(
        err.to_string()
            .contains("pipeline.stages must contain at least one stage"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_blank_postprocess_executor_class() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: blank-postprocess-executor
  display_name: Blank Postprocess Executor
  version: 1.0.0
  description: Blank optional executor fields must be rejected
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
  postprocess_executor_class: "   "
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("blank postprocess executor class must be rejected");

    assert!(
        err.to_string()
            .contains("runtime.postprocess_executor_class must not be empty"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_readiness_env_without_name_or_any_of() {
    let manifest_yaml = format!(
        r#"{}
developer:
  readiness:
    env:
      - any_of: ["", "   "]
        kind: string
        required: false
"#,
        minimal_manifest_yaml()
    );
    let manifest = PluginManifest::from_yaml_str(&manifest_yaml).expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("readiness env entries must define a name or any_of values");

    assert!(
        err.to_string()
            .contains("must define 'name' or at least one non-empty 'any_of' entry"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_invalid_readiness_path_kind() {
    let manifest_yaml = format!(
        r#"{}
developer:
  readiness:
    paths:
      - path: models
        kind: socket
        required: false
"#,
        minimal_manifest_yaml()
    );
    let manifest = PluginManifest::from_yaml_str(&manifest_yaml).expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("unsupported readiness path kinds must be rejected");

    assert!(
        err.to_string()
            .contains("must be one of: file, dir, path, string"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_empty_content_types() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: empty-content-types
  display_name: Empty Content Types
  version: 1.0.0
  description: Content types must not be empty
ingress:
  content_types: []
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("empty content types must be rejected");

    assert!(
        err.to_string()
            .contains("ingress.content_types must contain at least one content type"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_empty_allowed_operations() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: empty-allowed-operations
  display_name: Empty Allowed Operations
  version: 1.0.0
  description: Allowed operations must not be empty
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: []
  json_schema_inline:
    type: object
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("empty allowed operations must be rejected");

    assert!(
        err.to_string()
            .contains("ingress.operations.allowed must contain at least one operation"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_non_python_json_content_without_schema() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: non-python-json-no-schema
  display_name: Non Python JSON No Schema
  version: 1.0.0
  description: Non-python JSON plugins must declare a request schema
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.container.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: container
  entrypoint: workflow.sh
  executor_class: container.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("non-python JSON plugins must provide a request schema");

    assert!(
        err.to_string()
            .contains("must define 'ingress.json_schema' or 'ingress.json_schema_inline'"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_rejects_non_python_multipart_content_without_schema() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: non-python-multipart-no-schema
  display_name: Non Python Multipart No Schema
  version: 1.0.0
  description: Non-python multipart plugins must declare a form schema
ingress:
  content_types: [multipart/form-data]
  default_content_type: multipart/form-data
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.container.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: container
  entrypoint: workflow.sh
  executor_class: container.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/octet-stream
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("non-python multipart plugins must provide a form schema");

    assert!(
        err.to_string()
            .contains("must define 'ingress.form_schema' or 'ingress.form_schema_inline'"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_validation_accepts_absent_resources_section() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: no-resources
  display_name: No Resources Section
  version: 1.0.0
  description: Resources section is entirely absent
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    manifest
        .validate()
        .expect("absent resources section should validate");
    assert_eq!(manifest.resources.defaults.gpus_required, 0);
    assert_eq!(manifest.resources.defaults.memory_mb, 0);
}

#[test]
fn manifest_validation_rejects_non_python_missing_result_schema() {
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: non-python-missing-result-schema-validation
  display_name: Non Python Missing Result Schema Validation
  version: 1.0.0
  description: Non-python plugins must declare a result schema
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.container.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: container
  entrypoint: workflow.sh
  executor_class: container.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should parse");
    let err = manifest
        .validate()
        .expect_err("non-python plugins must declare a result schema");

    assert!(
        err.to_string()
            .contains("must define 'outputs.result_schema' or 'outputs.result_schema_inline'"),
        "unexpected validation error: {err}"
    );
}

#[test]
fn manifest_from_yaml_rejects_non_object_runtime_phases() {
    let err = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: invalid-runtime-phases
  display_name: Invalid Runtime Phases
  version: 1.0.0
  description: runtime.phases must be an object
pipeline:
  profile: simple
runtime:
  profile: python-test
  phases: []
"#,
    )
    .expect_err("runtime.phases arrays must be rejected");

    assert!(
        err.to_string()
            .contains("plugin manifest runtime.phases must be an object"),
        "unexpected parse error: {err:#}"
    );
}

#[test]
fn validate_schema_source_rejects_dual_path_and_inline_documents() {
    let err = validate_schema_source(
        Some("schemas/request.json"),
        Some(&serde_json::json!({"type": "object"})),
        "ingress.json_schema",
        "ingress.json_schema_inline",
        false,
    )
    .expect_err("path and inline schemas must be mutually exclusive");

    assert!(
        err.to_string().contains(
            "must not define both 'ingress.json_schema' and 'ingress.json_schema_inline'"
        )
    );
}

#[test]
fn validate_schema_source_requires_source_when_marked_required() {
    let err = validate_schema_source(
        None,
        None,
        "outputs.result_schema",
        "outputs.result_schema_inline",
        true,
    )
    .expect_err("required schema source must be present");

    assert!(
        err.to_string()
            .contains("must define 'outputs.result_schema' or 'outputs.result_schema_inline'")
    );
}

#[test]
fn validate_schema_source_rejects_non_object_inline_schema() {
    let err = validate_schema_source(
        None,
        Some(&serde_json::json!(["not", "an", "object"])),
        "ingress.form_schema",
        "ingress.form_schema_inline",
        false,
    )
    .expect_err("inline schema documents must be JSON objects");

    assert!(
        err.to_string()
            .contains("ingress.form_schema_inline must be a JSON object")
    );
}

#[test]
fn normalize_ingress_aliases_maps_multipart_request_schema_inline_and_operation_alias() {
    let mut ingress = serde_json::Map::from_iter([
        (
            "content_type".to_string(),
            serde_json::Value::String("multipart/form-data".to_string()),
        ),
        (
            "request_schema_inline".to_string(),
            serde_json::json!({"type": "object", "required": ["mesh"]}),
        ),
        (
            "operation".to_string(),
            serde_json::Value::String("both".to_string()),
        ),
    ]);

    normalize_ingress_aliases(&mut ingress).expect("multipart aliases should normalize");

    assert_eq!(
        ingress.get("content_types"),
        Some(&serde_json::json!(["multipart/form-data"]))
    );
    assert_eq!(
        ingress.get("default_content_type"),
        Some(&serde_json::Value::String(
            "multipart/form-data".to_string()
        ))
    );
    assert_eq!(
        ingress.get("form_schema_inline"),
        Some(&serde_json::json!({"type": "object", "required": ["mesh"]}))
    );
    assert_eq!(
        ingress.get("operations"),
        Some(&serde_json::json!({"default": "both", "allowed": ["both"]}))
    );
}

#[test]
fn normalize_ingress_aliases_defaults_to_json_and_run_operation() {
    let mut ingress = serde_json::Map::new();

    normalize_ingress_aliases(&mut ingress).expect("default ingress aliases should normalize");

    assert_eq!(
        ingress.get("content_types"),
        Some(&serde_json::json!(["application/json"]))
    );
    assert_eq!(
        ingress.get("default_content_type"),
        Some(&serde_json::Value::String("application/json".to_string()))
    );
    assert_eq!(
        ingress.get("operations"),
        Some(&serde_json::json!({"default": "run", "allowed": ["run"]}))
    );
}

#[test]
fn build_pipeline_stages_supports_ensemble_parent_prefetch_with_postprocess() {
    let options = serde_json::Map::from_iter([
        (
            "prefetch".to_string(),
            serde_json::Value::String("parent".to_string()),
        ),
        ("postprocess".to_string(), serde_json::Value::Bool(true)),
    ]);

    let stages = build_pipeline_stages("ensemble", &options, "execute.python.test")
        .expect("ensemble stages should be generated");

    assert_eq!(
        stages
            .iter()
            .map(|stage| stage["phase"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "prepare",
            "prefetch",
            "fanout",
            "schedule",
            "execute",
            "collect",
            "postprocess",
            "results",
        ]
    );
    assert_eq!(stages[4]["queue"], "execute.python.test");
}

#[test]
fn build_pipeline_stages_rejects_invalid_ensemble_prefetch_option() {
    let options = serde_json::Map::from_iter([(
        "prefetch".to_string(),
        serde_json::Value::String("child".to_string()),
    )]);

    let err = build_pipeline_stages("ensemble", &options, "execute.python.test")
        .expect_err("invalid ensemble prefetch options must be rejected");

    assert!(
        err.to_string()
            .contains("pipeline.options.prefetch must be true, false, or 'parent'")
    );
}

#[test]
fn resolve_runtime_profile_executor_rejects_unknown_profile() {
    let err = resolve_runtime_profile_executor("unknown-profile")
        .expect_err("unknown runtime profiles must be rejected");

    assert!(
        err.to_string()
            .contains("runtime.profile 'unknown-profile' is not supported")
    );
}

#[test]
fn readiness_uses_runtime_env_specific_python_probe() {
    let root = unique_temp_dir("runtime-env-readiness");
    let runtime_python = write_fake_python_runtime(&root, "fake-python.sh", r#"{"found": true}"#);
    let manifest_yaml = r#"
metadata:
  id: runtime-env-ready
  display_name: Runtime Env Ready
  version: 1.0.0
  description: Plugin using runtime env aware readiness
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
  readiness_executor_class: python.ready.custom
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
developer:
  readiness:
    python_modules: [definitely_missing_plugin_module]
"#;
    fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), manifest_yaml)
        .expect("manifest should be written");

    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest: PluginManifest::from_yaml_str(manifest_yaml).expect("manifest should parse"),
    };
    let mut probe = PythonModuleProbe::with_runtime_envs(HashMap::from([(
        "python.ready.custom".to_string(),
        PythonRuntimeEnvConfig {
            python_executable: runtime_python.display().to_string(),
            env: HashMap::new(),
        },
    )]));

    let readiness = plugin.evaluate_readiness(&mut probe);
    assert!(
        readiness.ready,
        "runtime env python should satisfy readiness"
    );
    assert_eq!(readiness.checks.len(), 1);
    assert!(readiness.checks[0].ok);
}

#[test]
fn resolve_plugin_dirs_reads_plugin_dir() {
    let env_provider = create_test_env(vec![("PLUGIN_DIR", "/tmp/plugins_a:/tmp/plugins_b")]);

    let dirs = resolve_plugin_dirs(&env_provider);
    assert_eq!(
        dirs,
        vec![
            PathBuf::from("/tmp/plugins_a"),
            PathBuf::from("/tmp/plugins_b")
        ]
    );
}

#[test]
fn resolve_plugin_dirs_empty_when_unset() {
    let env_provider = create_test_env(vec![]);
    let dirs = resolve_plugin_dirs(&env_provider);

    assert!(dirs.is_empty());
}

#[test]
fn discover_plugins_loads_plugin_manifests_from_child_directories() {
    let root = unique_temp_dir("discover");
    let plugin_dir = root.join("demo-prefetch");
    fs::create_dir_all(&plugin_dir).expect("plugin root should be created");
    fs::write(
        plugin_dir.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        minimal_manifest_yaml(),
    )
    .expect("manifest should be written");

    let plugins =
        discover_plugins(&[root]).expect("plugin discovery should succeed for valid plugins");

    assert_eq!(plugins.len(), 1);
    assert_eq!(plugins[0].manifest.metadata.id, "demo-prefetch");
    assert_eq!(plugins[0].root_dir, plugin_dir);
}

#[test]
fn discover_plugins_rejects_duplicate_plugin_ids() {
    let root = unique_temp_dir("duplicate-ids");
    let first = root.join("plugin-a");
    let second = root.join("plugin-b");
    fs::create_dir_all(&first).expect("first plugin root should be created");
    fs::create_dir_all(&second).expect("second plugin root should be created");
    fs::write(
        first.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        minimal_manifest_yaml(),
    )
    .expect("first manifest should be written");
    fs::write(
        second.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        minimal_manifest_yaml(),
    )
    .expect("second manifest should be written");

    let err = discover_plugins(&[root]).expect_err("duplicate plugin ids must be rejected");

    assert!(
        err.to_string().contains("duplicate plugin id"),
        "expected duplicate plugin id error, got: {err}"
    );
}

#[test]
fn registered_plugin_loads_request_and_result_schemas_relative_to_root() {
    let root = unique_temp_dir("schema-load");
    let schemas_dir = root.join("schemas");
    fs::create_dir_all(&schemas_dir).expect("schemas directory should be created");
    fs::write(
        root.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        minimal_manifest_yaml(),
    )
    .expect("manifest should be written");
    fs::write(
        schemas_dir.join("request.json"),
        r#"{"type":"object","required":["start_time"]}"#,
    )
    .expect("request schema should be written");
    fs::write(
        schemas_dir.join("result.json"),
        r#"{"type":"object","properties":{"status":{"type":"string"}}}"#,
    )
    .expect("result schema should be written");

    let manifest =
        PluginManifest::from_yaml_str(minimal_manifest_yaml()).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root,
        manifest_path: plugin_manifest_path(schemas_dir.parent().expect("root should exist")),
        manifest,
    };

    let request_schemas = plugin
        .load_request_schemas()
        .expect("request schemas should load");
    let result_schema = plugin
        .load_result_schema()
        .expect("result schema should load");

    assert_eq!(
        request_schemas["application/json"],
        serde_json::json!({"type":"object","required":["start_time"]})
    );
    assert_eq!(
        result_schema,
        serde_json::json!({"type":"object","properties":{"status":{"type":"string"}}})
    );
}

#[test]
fn registered_plugin_rejects_schema_path_outside_plugin_root() {
    let parent = unique_temp_dir("schema-escape-parent");
    let root = parent.join("plugin");
    let escape_dir = parent.join("escape");
    fs::create_dir_all(&root).expect("plugin root should be created");
    fs::create_dir_all(&escape_dir).expect("escape dir should be created");
    fs::write(
        root.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        r#"
metadata:
  id: schema-escape-plugin
  display_name: Schema Escape Plugin
  version: 1.0.0
  description: Plugin attempting to load schemas outside its root
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema: ../escape/request.json
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema: ../escape/result.json
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#,
    )
    .expect("manifest should be written");
    fs::write(escape_dir.join("request.json"), r#"{"type":"object"}"#)
        .expect("escaped request schema should be written");
    fs::write(escape_dir.join("result.json"), r#"{"type":"object"}"#)
        .expect("escaped result schema should be written");

    let manifest = PluginManifest::from_yaml_str(
        &fs::read_to_string(root.join(DEFAULT_PLUGIN_MANIFEST_NAME))
            .expect("manifest should be readable"),
    )
    .expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };

    let request_err = plugin
        .load_request_schemas()
        .expect_err("schema paths outside the plugin root must be rejected");
    assert!(
        format!("{request_err:#}").contains("outside the plugin root"),
        "unexpected request error: {request_err:#}"
    );

    let result_err = plugin
        .load_result_schema()
        .expect_err("result schema paths outside the plugin root must be rejected");
    assert!(
        format!("{result_err:#}").contains("outside the plugin root"),
        "unexpected result error: {result_err:#}"
    );
}

#[test]
fn registered_plugin_loads_inline_request_and_result_schemas_from_manifest() {
    let root = unique_temp_dir("inline-schema-load");
    let manifest_yaml = r#"
metadata:
  id: inline-schema-plugin
  display_name: Inline Schema Plugin
  version: 1.0.0
  description: Plugin using inline schemas
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
    required: [value]
    properties:
      value:
        type: integer
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
    properties:
      status:
        type: string
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#;
    fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), manifest_yaml)
        .expect("manifest should be written");

    let manifest = PluginManifest::from_yaml_str(manifest_yaml).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };

    let request_schemas = plugin
        .load_request_schemas()
        .expect("request schemas should load");
    let result_schema = plugin
        .load_result_schema()
        .expect("result schema should load");

    assert_eq!(
        request_schemas["application/json"],
        serde_json::json!({"type":"object","required":["value"],"properties":{"value":{"type":"integer"}}})
    );
    assert_eq!(
        result_schema,
        serde_json::json!({"type":"object","properties":{"status":{"type":"string"}}})
    );
}

#[test]
fn registered_plugin_loads_inline_multipart_request_schema_from_manifest() {
    let root = unique_temp_dir("inline-multipart-schema-load");
    let manifest_yaml = r#"
metadata:
  id: inline-multipart-plugin
  display_name: Inline Multipart Plugin
  version: 1.0.0
  description: Plugin using an inline multipart schema
ingress:
  content_types: [multipart/form-data]
  default_content_type: multipart/form-data
  operations:
    default: run
    allowed: [run]
  form_schema_inline:
    type: object
    required: [upload]
    properties:
      upload:
        type: string
  files:
    - name: upload
      required: true
      media_types: [application/octet-stream]
      max_size_mb: 1
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema_inline:
    type: object
  primary_artifact:
    name: primary
    media_type: application/octet-stream
  retention_hours: 24
"#;
    let manifest = PluginManifest::from_yaml_str(manifest_yaml).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };

    let request_schemas = plugin
        .load_request_schemas()
        .expect("multipart request schemas should load");

    assert_eq!(request_schemas.len(), 1);
    assert_eq!(
        request_schemas["multipart/form-data"],
        serde_json::json!({"type":"object","required":["upload"],"properties":{"upload":{"type":"string"}}})
    );
}

#[test]
fn registered_plugin_load_result_schema_errors_when_non_python_manifest_has_no_schema() {
    let root = unique_temp_dir("non-python-missing-result-schema");
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: non-python-missing-result-schema
  display_name: Non Python Missing Result Schema
  version: 1.0.0
  description: Non-python plugins must provide an explicit result schema
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
  json_schema_inline:
    type: object
pipeline:
  stages:
    - id: execute
      phase: execute
      handler: plugin_phase
      queue: execute.container.test
      next: results
    - id: results
      phase: results
      handler: persist_results
      queue: results
      next: null
runtime:
  kind: container
  entrypoint: workflow.sh
  executor_class: container.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
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

    let err = plugin
        .load_result_schema()
        .expect_err("non-python manifests without explicit result schemas must error");

    assert!(
        err.to_string()
            .contains("missing outputs.result_schema or outputs.result_schema_inline"),
        "unexpected error: {err:#}"
    );
}

fn write_fake_contract_probe(dir: &Path, payload_json: &str) -> PathBuf {
    let script_path = dir.join("fake-contract-probe.sh");
    write_executable_script(
        &script_path,
        &format!("#!/bin/sh\nprintf '%s\\n' '{}'\n", payload_json),
    );
    script_path
}

#[test]
fn registered_plugin_derives_request_and_result_schemas_from_model_driven_workflow() {
    let _lock = PYTHON_ENV_LOCK.lock().unwrap();
    let root = unique_temp_dir("model-derived-schema-load");

    let probe_payload = serde_json::json!({
        "request_schema": {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
            "additionalProperties": false
        },
        "result_schema": {
            "type": "object",
            "required": ["value", "doubled"],
            "properties": {
                "value": {"type": "integer"},
                "doubled": {"type": "integer"}
            },
            "additionalProperties": false
        }
    });
    let fake_python = write_fake_contract_probe(&root, &probe_payload.to_string());
    let _env_guard = EnvGuard::set("PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE", &fake_python);

    let manifest_yaml = r#"
metadata:
  id: model-derived-plugin
  display_name: Model Derived Plugin
  version: 1.0.0
  description: Plugin deriving schemas from workflow models
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#;
    fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), manifest_yaml)
        .expect("manifest should be written");

    let manifest = PluginManifest::from_yaml_str(manifest_yaml).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };

    let request_schemas = plugin
        .load_request_schemas()
        .expect("request schemas should load");
    let result_schema = plugin
        .load_result_schema()
        .expect("result schema should load");

    assert_eq!(
        request_schemas["application/json"],
        serde_json::json!({"type":"object","required":["value"],"properties":{"value":{"type":"integer"}},"additionalProperties":false})
    );
    assert_eq!(
        result_schema,
        serde_json::json!({"type":"object","required":["value","doubled"],"properties":{"value":{"type":"integer"},"doubled":{"type":"integer"}},"additionalProperties":false})
    );
}

#[test]
fn registered_plugin_derives_multipart_form_schema_from_workflow_form_model() {
    let _lock = PYTHON_ENV_LOCK.lock().unwrap();
    let root = unique_temp_dir("multipart-form-model-derived-schema");

    let probe_payload = serde_json::json!({
        "form_schema": {
            "type": "object",
            "required": ["note"],
            "properties": {"note": {"type": "string"}},
            "additionalProperties": false
        },
        "result_schema": {
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}},
            "additionalProperties": true
        }
    });
    let fake_python = write_fake_contract_probe(&root, &probe_payload.to_string());
    let _env_guard = EnvGuard::set("PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE", &fake_python);

    let manifest_yaml = r#"
metadata:
  id: multipart-form-model-plugin
  display_name: Multipart Form Model Plugin
  version: 1.0.0
  description: Plugin deriving multipart schemas from workflow form_model
ingress:
  content_types: [multipart/form-data]
  default_content_type: multipart/form-data
  operations:
    default: run
    allowed: [run]
  files:
    - name: upload
      required: true
      media_types: [application/octet-stream]
      max_size_mb: 1
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  primary_artifact:
    name: primary
    media_type: application/json
  retention_hours: 24
"#;
    fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), manifest_yaml)
        .expect("manifest should be written");

    let manifest = PluginManifest::from_yaml_str(manifest_yaml).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };

    let request_schemas = plugin
        .load_request_schemas()
        .expect("multipart request schemas should load");

    assert_eq!(request_schemas.len(), 1);
    assert_eq!(
        request_schemas["multipart/form-data"],
        serde_json::json!({"type":"object","required":["note"],"properties":{"note":{"type":"string"}},"additionalProperties":false})
    );
}

#[test]
fn registered_plugin_load_result_schema_surfaces_workflow_probe_failure() {
    let _lock = PYTHON_ENV_LOCK.lock().expect("env lock should succeed");
    let root = unique_temp_dir("workflow-schema-probe-failure");
    let fake_python = root.join("fake-failing-python.sh");
    write_executable_script(
        &fake_python,
        "#!/bin/sh\nprintf 'workflow probe exploded\\n' >&2\nexit 9\n",
    );
    let _env_guard = EnvGuard::set("PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE", &fake_python);
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: workflow-probe-failure
  display_name: Workflow Probe Failure
  version: 1.0.0
  description: Surface stderr from workflow schema probes
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
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

    let err = plugin
        .load_result_schema()
        .expect_err("failed workflow probes must surface stderr");
    let message = format!("{err:#}");

    assert!(message.contains("workflow schema probe failed via"));
    assert!(message.contains("workflow probe exploded"));
}

#[test]
fn registered_plugin_load_result_schema_reports_missing_python_interpreter() {
    let _lock = PYTHON_ENV_LOCK.lock().expect("env lock should succeed");
    let root = unique_temp_dir("workflow-schema-no-interpreter");
    let _env_guard = EnvGuard::set(
        "PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE",
        "/definitely/missing/physicsnemo-serve-python",
    );
    let manifest = PluginManifest::from_yaml_str(
        r#"
metadata:
  id: workflow-probe-no-interpreter
  display_name: Workflow Probe No Interpreter
  version: 1.0.0
  description: Surface a missing Python interpreter for workflow schema probes
ingress:
  content_types: [application/json]
  default_content_type: application/json
  operations:
    default: run
    allowed: [run]
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
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

    let err = plugin
        .load_result_schema()
        .expect_err("missing interpreters must produce a clear error");

    assert!(
        err.to_string()
            .contains("no Python interpreter found for workflow schema probes"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn registered_plugin_loads_multipart_schema_without_deriving_json_schema() {
    let root = unique_temp_dir("multipart-schema-load");
    let schemas_dir = root.join("schemas");
    fs::create_dir_all(&schemas_dir).expect("schemas directory should be created");
    let manifest_yaml = r#"
metadata:
  id: multipart-only-plugin
  display_name: Multipart Only Plugin
  version: 1.0.0
  description: Plugin using only multipart input
ingress:
  content_types: [multipart/form-data]
  default_content_type: multipart/form-data
  operations:
    default: run
    allowed: [run]
  form_schema: schemas/request.multipart.json
  files:
    - name: upload
      required: true
      media_types: [application/octet-stream]
      max_size_mb: 1
pipeline:
  stages:
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
  entrypoint: workflow.py
  executor_class: python.test
resources:
  defaults:
    device_kind: cpu
    gpus_required: 0
    memory_mb: 1024
    cpu_cores: 1
outputs:
  result_schema: schemas/result.json
  primary_artifact:
    name: primary
    media_type: application/octet-stream
  retention_hours: 24
"#;
    fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), manifest_yaml)
        .expect("manifest should be written");
    fs::write(
        schemas_dir.join("request.multipart.json"),
        r#"{"type":"object","required":["batch_size"],"properties":{"batch_size":{"type":"integer"}}}"#,
    )
    .expect("multipart schema should be written");
    fs::write(
        schemas_dir.join("result.json"),
        r#"{"type":"object","properties":{"status":{"type":"string"}}}"#,
    )
    .expect("result schema should be written");

    let manifest = PluginManifest::from_yaml_str(manifest_yaml).expect("manifest should parse");
    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest,
    };

    let request_schemas = plugin
        .load_request_schemas()
        .expect("multipart request schema should load without probing workflow.py");

    assert_eq!(request_schemas.len(), 1);
    assert_eq!(
        request_schemas["multipart/form-data"],
        serde_json::json!({"type":"object","required":["batch_size"],"properties":{"batch_size":{"type":"integer"}}})
    );
}

#[test]
fn python_probe_candidates_prefer_explicit_env() {
    let explicit = PathBuf::from("/tmp/custom-python");
    let candidates = python_probe_candidates_from_env(|key| match key {
        "PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE" => Some(explicit.clone().into_os_string()),
        _ => None,
    });

    assert_eq!(candidates, vec![explicit.into_os_string()]);
}

#[test]
fn python_probe_candidates_deduplicate_python3_from_python_env() {
    let candidates = python_probe_candidates_from_env(|key| match key {
        "PYTHON" => Some(std::ffi::OsString::from("python3")),
        _ => None,
    });

    assert_eq!(
        candidates,
        vec![
            std::ffi::OsString::from("python3"),
            std::ffi::OsString::from("python")
        ]
    );
}

#[test]
fn unique_temp_dir_uses_workspace_backed_test_root() {
    let dir = unique_temp_dir("root-location");

    assert!(dir.starts_with(plugin_registry_test_root()));
}

#[test]
fn probe_python_module_accepts_later_candidate_that_finds_module() {
    let root = unique_temp_dir("python-probe");
    let false_probe = root.join("probe-false.sh");
    let true_probe = root.join("probe-true.sh");

    write_executable_script(
        &false_probe,
        "#!/bin/sh\nprintf 'PHYSICSNEMO_SERVE_JSON:{\"found\": false}\\n'\n",
    );
    write_executable_script(
        &true_probe,
        "#!/bin/sh\nprintf 'PHYSICSNEMO_SERVE_JSON:{\"found\": true}\\n'\n",
    );

    let found = probe_python_module_with_candidates_and_env(
        "demo_module",
        &[false_probe.into_os_string(), true_probe.into_os_string()],
        None,
    )
    .expect("probe should succeed");

    assert!(found);
}

#[test]
fn probe_python_module_tolerates_extra_stdout_before_json_marker() {
    let root = unique_temp_dir("python-probe-noisy");
    let noisy_probe = root.join("probe-noisy.sh");

    write_executable_script(
        &noisy_probe,
        "#!/bin/sh\nprintf 'Warp DeprecationWarning: something noisy\\n'\nprintf 'PHYSICSNEMO_SERVE_JSON:{\"found\": true}\\n'\n",
    );

    let found = probe_python_module_with_candidates_and_env(
        "demo_module",
        &[noisy_probe.into_os_string()],
        None,
    )
    .expect("probe should succeed");

    assert!(found);
}

#[test]
fn probe_python_module_returns_false_when_candidate_reports_missing_module() {
    let root = unique_temp_dir("python-probe-missing");
    let false_probe = root.join("probe-false-only.sh");

    write_executable_script(
        &false_probe,
        "#!/bin/sh\nprintf 'PHYSICSNEMO_SERVE_JSON:{\"found\": false}\\n'\n",
    );

    let found = probe_python_module_with_candidates_and_env(
        "definitely_missing_module",
        &[false_probe.into_os_string()],
        None,
    )
    .expect("probe should return a successful false result");

    assert!(!found);
}

#[test]
fn probe_python_module_errors_when_json_marker_missing() {
    let root = unique_temp_dir("python-probe-missing-marker");
    let invalid_probe = root.join("probe-invalid.sh");

    write_executable_script(&invalid_probe, "#!/bin/sh\nprintf '{\"found\": true}\\n'\n");

    let err = probe_python_module_with_candidates_and_env(
        "demo_module",
        &[invalid_probe.into_os_string()],
        None,
    )
    .expect_err("probe output without the marker must fail");

    assert!(err.contains("missing PHYSICSNEMO_SERVE_JSON: marker"));
}

#[test]
fn probe_python_module_errors_when_found_field_is_missing() {
    let root = unique_temp_dir("python-probe-missing-found-field");
    let invalid_probe = root.join("probe-missing-found.sh");

    write_executable_script(
        &invalid_probe,
        "#!/bin/sh\nprintf 'PHYSICSNEMO_SERVE_JSON:{}\\n'\n",
    );

    let err = probe_python_module_with_candidates_and_env(
        "demo_module",
        &[invalid_probe.into_os_string()],
        None,
    )
    .expect_err("probe output without a boolean 'found' field must fail");

    assert!(err.contains("python module probe did not return a boolean"));
}

#[test]
fn probe_python_module_reports_stderr_from_failed_candidate() {
    let root = unique_temp_dir("python-probe-stderr");
    let failing_probe = root.join("probe-fail.sh");

    write_executable_script(
        &failing_probe,
        "#!/bin/sh\nprintf 'boom on stderr\\n' >&2\nexit 7\n",
    );

    let err = probe_python_module_with_candidates_and_env(
        "demo_module",
        &[failing_probe.into_os_string()],
        None,
    )
    .expect_err("non-zero probe exit should surface stderr");

    assert!(err.contains("python module probe failed via"));
    assert!(err.contains("boom on stderr"));
}

#[test]
fn readiness_reports_required_any_of_env_missing_and_resolves_relative_dir() {
    let root = unique_temp_dir("readiness-required-env");
    let assets_dir = root.join("relative-assets");
    fs::create_dir_all(&assets_dir).expect("relative assets dir should exist");
    let env_a = format!("PHYSICSNEMO_SERVE_REQUIRED_PATH_A_{}", uuid::Uuid::new_v4());
    let env_b = format!("PHYSICSNEMO_SERVE_REQUIRED_PATH_B_{}", uuid::Uuid::new_v4());
    let manifest_yaml = format!(
        r#"{}
developer:
  readiness:
    env:
      - any_of: [{env_a}, {env_b}]
        kind: file
        required: true
    paths:
      - path: relative-assets
        kind: dir
        required: true
"#,
        minimal_manifest_yaml()
    );
    fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), &manifest_yaml)
        .expect("manifest should be written");

    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest: PluginManifest::from_yaml_str(&manifest_yaml).expect("manifest should parse"),
    };
    let mut probe = PythonModuleProbe::default();

    let readiness = plugin.evaluate_readiness(&mut probe);

    assert!(
        !readiness.ready,
        "missing required env should make readiness fail"
    );
    assert_eq!(readiness.checks.len(), 2);
    assert_eq!(readiness.checks[0].check_type, "env");
    assert!(!readiness.checks[0].ok);
    assert!(readiness.checks[0].detail.contains(&env_a));
    assert!(readiness.checks[0].detail.contains(&env_b));
    assert_eq!(readiness.checks[1].check_type, "path");
    assert!(readiness.checks[1].ok);
    assert!(readiness.checks[1].detail.contains("relative-assets"));
}

#[test]
fn readiness_optional_any_of_env_is_ready_when_unset() {
    let root = unique_temp_dir("readiness-optional-env");
    let env_a = format!("PHYSICSNEMO_SERVE_OPTIONAL_PATH_A_{}", uuid::Uuid::new_v4());
    let env_b = format!("PHYSICSNEMO_SERVE_OPTIONAL_PATH_B_{}", uuid::Uuid::new_v4());
    let manifest_yaml = format!(
        r#"{}
developer:
  readiness:
    env:
      - any_of: [{env_a}, {env_b}]
        kind: string
        required: false
"#,
        minimal_manifest_yaml()
    );
    fs::write(root.join(DEFAULT_PLUGIN_MANIFEST_NAME), &manifest_yaml)
        .expect("manifest should be written");

    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest: PluginManifest::from_yaml_str(&manifest_yaml).expect("manifest should parse"),
    };
    let mut probe = PythonModuleProbe::default();

    let readiness = plugin.evaluate_readiness(&mut probe);

    assert!(
        readiness.ready,
        "optional env checks should not block readiness"
    );
    assert_eq!(readiness.checks.len(), 1);
    assert!(readiness.checks[0].ok);
    assert!(
        readiness.checks[0]
            .detail
            .contains("optional env vars are not set")
    );
}

#[test]
fn readiness_named_optional_string_env_reports_set_value() {
    let _lock = PYTHON_ENV_LOCK.lock().expect("env lock should succeed");
    let root = unique_temp_dir("readiness-optional-string-env");
    let _env_guard = EnvGuard::set("PHYSICSNEMO_SERVE_OPTIONAL_STRING_ENV", "configured");
    let manifest_yaml = format!(
        r#"{}
developer:
  readiness:
    env:
      - name: PHYSICSNEMO_SERVE_OPTIONAL_STRING_ENV
        kind: string
        required: false
"#,
        minimal_manifest_yaml()
    );

    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest: PluginManifest::from_yaml_str(&manifest_yaml).expect("manifest should parse"),
    };
    let mut probe = PythonModuleProbe::default();

    let readiness = plugin.evaluate_readiness(&mut probe);

    assert!(readiness.ready);
    assert_eq!(readiness.checks.len(), 1);
    assert!(readiness.checks[0].ok);
    assert_eq!(
        readiness.checks[0].name,
        "PHYSICSNEMO_SERVE_OPTIONAL_STRING_ENV"
    );
    assert!(
        readiness.checks[0]
            .detail
            .contains("env var 'PHYSICSNEMO_SERVE_OPTIONAL_STRING_ENV' is set")
    );
}

#[test]
fn readiness_optional_any_of_invalid_env_reports_failed_check_without_blocking_readiness() {
    let _lock = PYTHON_ENV_LOCK.lock().expect("env lock should succeed");
    let root = unique_temp_dir("readiness-optional-invalid-any-of");
    let _env_a = EnvGuard::set(
        "PHYSICSNEMO_SERVE_OPTIONAL_INVALID_ANY_OF_A",
        root.as_os_str(),
    );
    let _env_b = EnvGuard::set("PHYSICSNEMO_SERVE_OPTIONAL_INVALID_ANY_OF_B", "");
    let manifest_yaml = format!(
        r#"{}
developer:
  readiness:
    env:
      - any_of: [PHYSICSNEMO_SERVE_OPTIONAL_INVALID_ANY_OF_A, PHYSICSNEMO_SERVE_OPTIONAL_INVALID_ANY_OF_B]
        kind: file
        required: false
"#,
        minimal_manifest_yaml()
    );

    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest: PluginManifest::from_yaml_str(&manifest_yaml).expect("manifest should parse"),
    };
    let mut probe = PythonModuleProbe::default();

    let readiness = plugin.evaluate_readiness(&mut probe);

    assert!(
        readiness.ready,
        "optional invalid env checks should not block overall readiness"
    );
    assert_eq!(readiness.checks.len(), 1);
    assert!(
        !readiness.checks[0].ok,
        "invalid optional env checks should still report the check as failed"
    );
    assert!(!readiness.checks[0].required);
    assert!(
        readiness.checks[0]
            .detail
            .contains("does not point to a file"),
        "unexpected detail: {}",
        readiness.checks[0].detail
    );
}

#[test]
fn readiness_required_absolute_path_reports_failure() {
    let root = unique_temp_dir("readiness-absolute-path");
    let missing_dir = root.join("missing-assets");
    let manifest_yaml = format!(
        r#"{}
developer:
  readiness:
    paths:
      - path: "{}"
        kind: dir
        required: true
"#,
        minimal_manifest_yaml(),
        missing_dir.display()
    );

    let plugin = RegisteredPlugin {
        root_dir: root.clone(),
        manifest_path: plugin_manifest_path(&root),
        manifest: PluginManifest::from_yaml_str(&manifest_yaml).expect("manifest should parse"),
    };
    let mut probe = PythonModuleProbe::default();

    let readiness = plugin.evaluate_readiness(&mut probe);

    assert!(!readiness.ready);
    assert_eq!(readiness.checks.len(), 1);
    assert_eq!(readiness.checks[0].check_type, "path");
    assert!(!readiness.checks[0].ok);
    assert!(
        readiness.checks[0]
            .detail
            .contains(&missing_dir.display().to_string())
    );
}

#[test]
fn resolve_plugin_dirs_deduplicates_repeated_entries_preserving_order() {
    let env_provider = create_test_env(vec![(
        "PLUGIN_DIR",
        "/tmp/plugins_a:/tmp/plugins_b:/tmp/plugins_a",
    )]);

    let dirs = resolve_plugin_dirs(&env_provider);

    assert_eq!(
        dirs,
        vec![
            PathBuf::from("/tmp/plugins_a"),
            PathBuf::from("/tmp/plugins_b")
        ]
    );
}

#[test]
fn resolve_plugin_dirs_ignores_empty_path_segments() {
    let separator = if cfg!(windows) { ';' } else { ':' };
    let env_value = format!("/tmp/plugins_a{separator}{separator}/tmp/plugins_b{separator}");
    let env_provider = create_test_env(vec![("PLUGIN_DIR", &env_value)]);

    let dirs = resolve_plugin_dirs(&env_provider);

    assert_eq!(
        dirs,
        vec![
            PathBuf::from("/tmp/plugins_a"),
            PathBuf::from("/tmp/plugins_b")
        ]
    );
}

#[test]
fn resolve_plugin_dirs_accepts_comma_separated_entries() {
    let env_provider = create_test_env(vec![("PLUGIN_DIR", "/tmp/plugins_a,/tmp/plugins_b")]);

    let dirs = resolve_plugin_dirs(&env_provider);

    assert_eq!(
        dirs,
        vec![
            PathBuf::from("/tmp/plugins_a"),
            PathBuf::from("/tmp/plugins_b")
        ]
    );
}

#[test]
fn discover_plugins_loads_direct_root_manifest_and_child_manifests_in_sorted_order() {
    let root = unique_temp_dir("discover-direct-root");
    let child_a = root.join("a-plugin");
    let child_b = root.join("b-plugin");
    fs::create_dir_all(&child_a).expect("first child plugin dir should be created");
    fs::create_dir_all(&child_b).expect("second child plugin dir should be created");
    fs::write(
        root.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        manifest_yaml_with_id("root-plugin"),
    )
    .expect("root manifest should be written");
    fs::write(
        child_a.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        manifest_yaml_with_id("child-a-plugin"),
    )
    .expect("first child manifest should be written");
    fs::write(
        child_b.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        manifest_yaml_with_id("child-b-plugin"),
    )
    .expect("second child manifest should be written");

    let plugins = discover_plugins(std::slice::from_ref(&root))
        .expect("plugin discovery should load root and children");

    assert_eq!(plugins.len(), 3);
    assert_eq!(
        plugins
            .iter()
            .map(|plugin| plugin.root_dir.clone())
            .collect::<Vec<_>>(),
        vec![root, child_a, child_b]
    );
}

#[test]
fn discover_plugins_errors_when_search_root_is_a_file() {
    let root = unique_temp_dir("discover-file-root");
    let file_root = root.join("not-a-directory");
    fs::write(&file_root, "not a directory").expect("file search root should be written");

    let err = discover_plugins(std::slice::from_ref(&file_root))
        .expect_err("non-directory plugin roots must produce a contextual error");
    let message = format!("{err:#}");

    assert!(message.contains("failed to read plugin directory"));
    assert!(message.contains(&file_root.display().to_string()));
}

#[test]
fn discover_plugins_reports_yaml_parse_error_with_manifest_context() {
    let root = unique_temp_dir("discover-invalid-yaml");
    let plugin_dir = root.join("broken-plugin");
    fs::create_dir_all(&plugin_dir).expect("plugin dir should be created");
    let manifest_path = plugin_dir.join(DEFAULT_PLUGIN_MANIFEST_NAME);
    fs::write(
        &manifest_path,
        "metadata:\n  id: broken-plugin\ningress: [\n",
    )
    .expect("invalid manifest should be written");

    let err = discover_plugins(&[root]).expect_err("invalid YAML must fail discovery");
    let message = format!("{err:#}");

    assert!(message.contains("failed to load plugin manifest"));
    assert!(message.contains(&manifest_path.display().to_string()));
}

#[test]
fn discover_plugins_skips_missing_dirs_and_deduplicates_child_seen_twice() {
    let root = unique_temp_dir("discover-dedup-child");
    let child = root.join("demo-prefetch");
    let missing = root.join("missing-parent");
    fs::create_dir_all(&child).expect("child plugin dir should be created");
    fs::write(
        child.join(DEFAULT_PLUGIN_MANIFEST_NAME),
        minimal_manifest_yaml(),
    )
    .expect("child manifest should be written");

    let plugins = discover_plugins(&[child.clone(), missing, root])
        .expect("discovery should skip missing dirs and deduplicate roots");

    assert_eq!(plugins.len(), 1);
    assert_eq!(plugins[0].root_dir, child);
}

#[test]
fn discover_plugins_reports_validation_error_with_manifest_context() {
    let root = unique_temp_dir("discover-invalid-manifest");
    let plugin_dir = root.join("invalid-plugin");
    fs::create_dir_all(&plugin_dir).expect("plugin dir should be created");
    let invalid_manifest =
        manifest_yaml_with_id("invalid-plugin").replacen("default: run", "default: predict", 1);
    let manifest_path = plugin_dir.join(DEFAULT_PLUGIN_MANIFEST_NAME);
    fs::write(&manifest_path, invalid_manifest).expect("invalid manifest should be written");

    let err = discover_plugins(&[root]).expect_err("invalid manifest should fail discovery");
    let message = format!("{err:#}");

    assert!(message.contains("failed validation"));
    assert!(message.contains(&manifest_path.display().to_string()));
}

#[test]
fn resolve_script_path_uses_deployed_layout_when_compile_time_script_is_missing() {
    let exe = std::env::current_exe().expect("current exe should be available");
    let bin_dir = exe
        .parent()
        .expect("test binary should have a parent directory");
    let scripts_dir = bin_dir.join("../scripts");
    fs::create_dir_all(&scripts_dir).expect("deployed scripts dir should be created");
    let script_name = format!("physicsnemo-serve-test-script-{}.sh", uuid::Uuid::new_v4());
    let deployed_script = scripts_dir.join(&script_name);
    write_executable_script(&deployed_script, "#!/bin/sh\nexit 0\n");

    let resolved = resolve_script_path(&script_name);

    assert_eq!(resolved, deployed_script);
    fs::remove_file(resolved).expect("temporary deployed script should be removed");
}

#[test]
fn resolve_script_path_prefers_compile_time_layout_for_known_script() {
    let expected = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts")
        .join("plugin_contract_probe.py");

    let resolved = resolve_script_path("plugin_contract_probe.py");

    assert_eq!(resolved, expected);
    assert!(resolved.is_file(), "compile-time script should exist");
}

#[test]
fn resolve_script_path_returns_compile_time_candidate_when_script_is_missing() {
    let script_name = format!(
        "physicsnemo-serve-missing-script-{}.sh",
        uuid::Uuid::new_v4()
    );
    let expected = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts")
        .join(&script_name);

    let resolved = resolve_script_path(&script_name);

    assert_eq!(resolved, expected);
    assert!(
        !resolved.is_file(),
        "random script path should remain unresolved"
    );
}
