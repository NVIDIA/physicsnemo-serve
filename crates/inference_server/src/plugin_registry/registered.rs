/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

impl RegisteredPlugin {
    pub fn load_request_schemas(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut schemas = HashMap::new();

        if let Some(document) = &self.manifest.ingress.json_schema_inline {
            schemas.insert("application/json".to_string(), document.clone());
        } else if let Some(path) = &self.manifest.ingress.json_schema {
            schemas.insert(
                "application/json".to_string(),
                self.load_json_document(path)
                    .with_context(|| format!("failed to load JSON request schema '{}'", path))?,
            );
        }

        if let Some(document) = &self.manifest.ingress.form_schema_inline {
            schemas.insert("multipart/form-data".to_string(), document.clone());
        } else if let Some(path) = &self.manifest.ingress.form_schema {
            schemas.insert(
                "multipart/form-data".to_string(),
                self.load_json_document(path).with_context(|| {
                    format!("failed to load multipart request schema '{}'", path)
                })?,
            );
        }

        if self
            .manifest
            .ingress
            .content_types
            .iter()
            .any(|content_type| content_type == "multipart/form-data")
            && !schemas.contains_key("multipart/form-data")
            && let Some(document) = self.derive_workflow_schemas()?.form_schema
        {
            schemas.insert("multipart/form-data".to_string(), document);
        }

        if self
            .manifest
            .ingress
            .content_types
            .iter()
            .any(|content_type| content_type == "application/json")
            && !schemas.contains_key("application/json")
            && let Some(document) = self.derive_workflow_schemas()?.request_schema
        {
            schemas.insert("application/json".to_string(), document);
        }

        Ok(schemas)
    }

    pub fn load_result_schema(&self) -> Result<serde_json::Value> {
        if let Some(document) = &self.manifest.outputs.result_schema_inline {
            Ok(document.clone())
        } else if let Some(path) = &self.manifest.outputs.result_schema {
            self.load_json_document(path)
                .with_context(|| format!("failed to load result schema '{}'", path))
        } else {
            self.derive_workflow_schemas()?
                .result_schema
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "plugin manifest is missing outputs.result_schema or outputs.result_schema_inline"
                    )
                })
        }
    }

    fn derive_workflow_schemas(&self) -> Result<DerivedWorkflowSchemas> {
        if self.manifest.runtime.kind != "python" {
            return Ok(DerivedWorkflowSchemas {
                request_schema: None,
                form_schema: None,
                result_schema: None,
            });
        }

        derive_workflow_schemas(&self.root_dir)
    }

    fn load_json_document(&self, relative_path: &str) -> Result<serde_json::Value> {
        let path = self.resolve_plugin_document_path(relative_path)?;
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read JSON document '{}'", path.display()))?;
        serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse JSON document '{}'", path.display()))
    }

    fn resolve_plugin_document_path(&self, relative_path: &str) -> Result<PathBuf> {
        let canonical_root = fs::canonicalize(&self.root_dir).with_context(|| {
            format!(
                "failed to canonicalize plugin root '{}'",
                self.root_dir.display()
            )
        })?;
        let candidate = self.root_dir.join(relative_path);
        let canonical_candidate = fs::canonicalize(&candidate).with_context(|| {
            format!(
                "failed to canonicalize JSON document '{}'",
                candidate.display()
            )
        })?;
        if !canonical_candidate.starts_with(&canonical_root) {
            bail!(
                "JSON document '{}' is outside the plugin root '{}'",
                canonical_candidate.display(),
                canonical_root.display()
            );
        }
        Ok(canonical_candidate)
    }
}

fn derive_workflow_schemas(plugin_root: &Path) -> Result<DerivedWorkflowSchemas> {
    let script_path = super::discovery::resolve_script_path("plugin_contract_probe.py");
    let candidates = super::readiness::python_probe_candidates();
    let mut failures = Vec::new();

    for executable in candidates {
        let executable_display = executable.to_string_lossy().to_string();
        match Command::new(&executable)
            .arg(&script_path)
            .arg("--plugin-root")
            .arg(plugin_root)
            .output()
        {
            Ok(output) if output.status.success() => {
                let payload: DerivedWorkflowSchemas = serde_json::from_slice(&output.stdout)
                    .with_context(|| {
                        format!(
                            "invalid workflow schema probe output via '{}'",
                            executable_display
                        )
                    })?;
                return Ok(payload);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                failures.push(format!(
                    "workflow schema probe failed via '{}': {}",
                    executable_display,
                    stderr.trim()
                ));
            }
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => failures.push(format!(
                "failed to run workflow schema probe via '{}': {}",
                executable_display, err
            )),
        }
    }

    if !failures.is_empty() {
        bail!(failures.join("; "));
    }

    bail!(
        "no Python interpreter found for workflow schema probes (tried PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE, PYTHON, python3, and python)"
    )
}
