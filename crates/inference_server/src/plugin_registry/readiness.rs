/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

impl RegisteredPlugin {
    pub fn evaluate_readiness(&self, probe: &mut PythonModuleProbe) -> PluginReadinessReport {
        let readiness_config = &self.manifest.developer.readiness;
        let mut checks = Vec::new();
        let readiness_executor_class = self
            .manifest
            .runtime
            .readiness_executor_class
            .as_deref()
            .or(self.manifest.runtime.prepare_executor_class.as_deref())
            .or(Some(self.manifest.runtime.executor_class.as_str()));

        for module_name in &readiness_config.python_modules {
            checks.push(probe.check_module_for_executor(module_name, readiness_executor_class));
        }

        for env_check in &readiness_config.env {
            checks.push(self.evaluate_env_check(env_check));
        }

        for path_check in &readiness_config.paths {
            checks.push(self.evaluate_path_check(path_check));
        }

        let ready = checks.iter().all(|check| !check.required || check.ok);
        PluginReadinessReport {
            ready,
            recommended_check_phase: readiness_config.recommended_check_phase.clone(),
            checks,
        }
    }

    fn evaluate_env_check(&self, spec: &PluginReadinessEnvCheck) -> PluginReadinessCheck {
        let kind = normalize_check_kind(spec.kind.as_deref());
        if let Some(name) = spec
            .name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            let raw_value = std::env::var_os(name);
            let (ok, detail) = evaluate_env_value(name, raw_value.as_deref(), kind, spec.required);
            return PluginReadinessCheck {
                check_type: "env".to_string(),
                name: name.to_string(),
                required: spec.required,
                ok,
                detail,
            };
        }

        let names: Vec<&str> = spec
            .any_of
            .iter()
            .map(String::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect();
        let mut invalid_details = Vec::new();

        for name in &names {
            let raw_value = std::env::var_os(name);
            if raw_value.as_deref().is_none_or(|value| value.is_empty()) {
                continue;
            }
            let (ok, detail) = evaluate_env_value(name, raw_value.as_deref(), kind, spec.required);
            if ok {
                return PluginReadinessCheck {
                    check_type: "env".to_string(),
                    name: names.join(" | "),
                    required: spec.required,
                    ok,
                    detail,
                };
            }
            invalid_details.push(detail);
        }

        let detail = if spec.required {
            format!(
                "none of the env vars are set or valid: {}",
                names.join(", ")
            )
        } else if invalid_details.is_empty() {
            format!("optional env vars are not set: {}", names.join(", "))
        } else {
            invalid_details.join("; ")
        };
        let ok = if spec.required {
            false
        } else {
            invalid_details.is_empty()
        };
        PluginReadinessCheck {
            check_type: "env".to_string(),
            name: names.join(" | "),
            required: spec.required,
            ok,
            detail,
        }
    }

    fn evaluate_path_check(&self, spec: &PluginReadinessPathCheck) -> PluginReadinessCheck {
        let kind = normalize_check_kind(spec.kind.as_deref());
        let raw_path = spec.path.trim();
        let mut path = PathBuf::from(raw_path);
        if !path.is_absolute() {
            path = self.root_dir.join(path);
        }

        let ok = match kind {
            "file" => path.is_file(),
            "dir" => path.is_dir(),
            _ => path.exists(),
        };
        let detail = if ok {
            format!("path check passed for '{}'", path.display())
        } else {
            format!("path check failed for '{}'", path.display())
        };

        PluginReadinessCheck {
            check_type: "path".to_string(),
            name: raw_path.to_string(),
            required: spec.required,
            ok: ok || !spec.required,
            detail,
        }
    }
}

impl PythonModuleProbe {
    pub fn with_runtime_envs(runtime_envs: HashMap<String, PythonRuntimeEnvConfig>) -> Self {
        Self {
            cache: HashMap::new(),
            runtime_envs,
        }
    }

    pub fn check_module(&mut self, module_name: &str) -> PluginReadinessCheck {
        self.check_module_for_executor(module_name, None)
    }

    pub fn check_module_for_executor(
        &mut self,
        module_name: &str,
        executor_class: Option<&str>,
    ) -> PluginReadinessCheck {
        let cache_key = format!("{}::{module_name}", executor_class.unwrap_or_default());
        let runtime_env = executor_class.and_then(|name| self.runtime_envs.get(name));
        let cached = self
            .cache
            .entry(cache_key)
            .or_insert_with(|| probe_python_module(module_name, runtime_env));

        match cached {
            Ok(true) => PluginReadinessCheck {
                check_type: "python_module".to_string(),
                name: module_name.to_string(),
                required: true,
                ok: true,
                detail: format!("module '{}' is importable", module_name),
            },
            Ok(false) => PluginReadinessCheck {
                check_type: "python_module".to_string(),
                name: module_name.to_string(),
                required: true,
                ok: false,
                detail: format!("module '{}' is not importable", module_name),
            },
            Err(detail) => PluginReadinessCheck {
                check_type: "python_module".to_string(),
                name: module_name.to_string(),
                required: true,
                ok: false,
                detail: detail.clone(),
            },
        }
    }
}

pub(super) fn validate_readiness_kind(field: &str, kind: Option<&str>) -> Result<()> {
    if let Some(kind) = kind.map(str::trim).filter(|value| !value.is_empty())
        && !matches!(kind, "file" | "dir" | "path" | "string")
    {
        bail!("{field} must be one of: file, dir, path, string");
    }
    Ok(())
}

fn normalize_check_kind(kind: Option<&str>) -> &str {
    match kind.map(str::trim).filter(|value| !value.is_empty()) {
        Some("file") => "file",
        Some("dir") => "dir",
        Some("string") => "string",
        Some("path") => "path",
        _ => "path",
    }
}

fn evaluate_env_value(
    name: &str,
    raw_value: Option<&std::ffi::OsStr>,
    kind: &str,
    required: bool,
) -> (bool, String) {
    let Some(raw_value) = raw_value.filter(|value| !value.is_empty()) else {
        return if required {
            (false, format!("required env var '{}' is not set", name))
        } else {
            (true, format!("optional env var '{}' is not set", name))
        };
    };

    if kind == "string" {
        return (true, format!("env var '{}' is set", name));
    }

    let path = PathBuf::from(raw_value);
    let ok = match kind {
        "file" => path.is_file(),
        "dir" => path.is_dir(),
        _ => path.exists(),
    };

    let detail = if ok {
        match kind {
            "file" => format!("env var '{}' points to file '{}'", name, path.display()),
            "dir" => format!(
                "env var '{}' points to directory '{}'",
                name,
                path.display()
            ),
            _ => format!("env var '{}' points to path '{}'", name, path.display()),
        }
    } else {
        match kind {
            "file" => format!(
                "env var '{}' does not point to a file: {}",
                name,
                path.display()
            ),
            "dir" => format!(
                "env var '{}' does not point to a directory: {}",
                name,
                path.display()
            ),
            _ => format!(
                "env var '{}' points to a missing path: {}",
                name,
                path.display()
            ),
        }
    };

    (ok, detail)
}

fn probe_python_module(
    module_name: &str,
    runtime_env: Option<&PythonRuntimeEnvConfig>,
) -> std::result::Result<bool, String> {
    if let Some(runtime_env) = runtime_env {
        return probe_python_module_with_candidates_and_env(
            module_name,
            &[OsString::from(&runtime_env.python_executable)],
            Some(&runtime_env.env),
        );
    }
    let candidates = python_probe_candidates();
    probe_python_module_with_candidates_and_env(module_name, &candidates, None)
}

const PYTHON_PROBE_LAUNCH_MAX_ATTEMPTS: usize = 3;
const PYTHON_PROBE_RETRY_DELAY_MILLIS: u64 = 20;

fn is_retryable_probe_launch_error(error: &std::io::Error) -> bool {
    error.raw_os_error() == Some(26)
}

fn run_probe_command_with_retry<F>(mut run_once: F) -> std::io::Result<std::process::Output>
where
    F: FnMut() -> std::io::Result<std::process::Output>,
{
    for attempt in 0..PYTHON_PROBE_LAUNCH_MAX_ATTEMPTS {
        match run_once() {
            Ok(output) => return Ok(output),
            Err(error)
                if is_retryable_probe_launch_error(&error)
                    && attempt + 1 < PYTHON_PROBE_LAUNCH_MAX_ATTEMPTS =>
            {
                std::thread::sleep(std::time::Duration::from_millis(
                    PYTHON_PROBE_RETRY_DELAY_MILLIS,
                ));
            }
            Err(error) => return Err(error),
        }
    }

    Err(std::io::Error::other(
        "python module probe retry loop exhausted without returning",
    ))
}

pub(super) fn python_probe_candidates() -> Vec<OsString> {
    python_probe_candidates_from_env(|key| std::env::var_os(key))
}

pub(super) fn python_probe_candidates_from_env<F>(env_provider: F) -> Vec<OsString>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(explicit) =
        env_provider("PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE").filter(|value| !value.is_empty())
    {
        return vec![explicit];
    }

    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    if let Some(explicit_python) = env_provider("PYTHON").filter(|value| !value.is_empty())
        && seen.insert(explicit_python.clone())
    {
        candidates.push(explicit_python);
    }

    for executable in ["python3", "python"] {
        let candidate = OsString::from(executable);
        if seen.insert(candidate.clone()) {
            candidates.push(candidate);
        }
    }

    candidates
}

pub(super) fn probe_python_module_with_candidates_and_env(
    module_name: &str,
    executables: &[OsString],
    env_overrides: Option<&HashMap<String, String>>,
) -> std::result::Result<bool, String> {
    let json_marker = "PHYSICSNEMO_SERVE_JSON:";
    let script = "import importlib.util, json, sys; print('PHYSICSNEMO_SERVE_JSON:' + json.dumps({'found': importlib.util.find_spec(sys.argv[1]) is not None}))";
    let mut probe_errors: Vec<String> = Vec::new();
    let mut found_via_interpreter = false;

    for executable in executables {
        let executable_display = executable.to_string_lossy().to_string();
        let mut command = Command::new(executable);
        command.arg("-c").arg(script).arg(module_name);
        if let Some(env_overrides) = env_overrides {
            command.envs(env_overrides);
        }
        match run_probe_command_with_retry(|| command.output()) {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    probe_errors.push(format!(
                        "python module probe failed via '{}': {}",
                        executable_display,
                        stderr.trim()
                    ));
                    continue;
                }

                let stdout = String::from_utf8_lossy(&output.stdout);
                let payload_line = stdout
                    .lines()
                    .rev()
                    .find_map(|line| line.trim().strip_prefix(json_marker))
                    .ok_or_else(|| {
                        format!(
                            "invalid python module probe output: missing {} marker",
                            json_marker
                        )
                    })?;
                let payload: serde_json::Value = serde_json::from_str(payload_line)
                    .map_err(|e| format!("invalid python module probe output: {}", e))?;
                let found = payload
                    .get("found")
                    .and_then(serde_json::Value::as_bool)
                    .ok_or_else(|| "python module probe did not return a boolean".to_string())?;
                if found {
                    return Ok(true);
                }
                found_via_interpreter = true;
            }
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                probe_errors.push(format!(
                    "failed to run python module probe via '{}': {}",
                    executable_display, err
                ));
            }
        }
    }

    if found_via_interpreter {
        return Ok(false);
    }

    if !probe_errors.is_empty() {
        return Err(probe_errors.join("; "));
    }

    Err("no Python interpreter found for readiness checks (tried PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE, PYTHON, python3, and python)".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn successful_process_output() -> std::process::Output {
        Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .output()
            .expect("test helper should create a successful process output")
    }

    #[test]
    fn retryable_probe_launch_error_detects_etxtbsy_raw_code() {
        let busy_error = std::io::Error::from_raw_os_error(26);
        let permission_error = std::io::Error::new(ErrorKind::PermissionDenied, "denied");

        assert!(is_retryable_probe_launch_error(&busy_error));
        assert!(!is_retryable_probe_launch_error(&permission_error));
    }

    #[test]
    fn run_probe_command_with_retry_retries_transient_exec_busy_errors() {
        let attempts = AtomicUsize::new(0);

        let output = run_probe_command_with_retry(|| {
            let attempt = attempts.fetch_add(1, Ordering::SeqCst);
            if attempt < 2 {
                Err(std::io::Error::from_raw_os_error(26))
            } else {
                Ok(successful_process_output())
            }
        })
        .expect("transient ETXTBSY errors should be retried and eventually succeed");

        assert!(output.status.success());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn run_probe_command_with_retry_returns_non_retryable_errors_without_retrying() {
        let attempts = AtomicUsize::new(0);

        let error = run_probe_command_with_retry(|| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(std::io::Error::new(ErrorKind::PermissionDenied, "denied"))
        })
        .expect_err("non-retryable launch errors should be returned immediately");

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
    }

    #[test]
    fn run_probe_command_with_retry_returns_last_busy_error_after_max_attempts() {
        let attempts = AtomicUsize::new(0);

        let error = run_probe_command_with_retry(|| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(std::io::Error::from_raw_os_error(26))
        })
        .expect_err("persistent ETXTBSY should surface after bounded retries");

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            PYTHON_PROBE_LAUNCH_MAX_ATTEMPTS
        );
        assert_eq!(error.raw_os_error(), Some(26));
    }
}
