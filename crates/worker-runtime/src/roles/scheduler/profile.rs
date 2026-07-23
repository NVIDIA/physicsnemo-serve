/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt;
use std::sync::{LazyLock, Mutex, MutexGuard};

use serde::Deserialize;
use serde::de::{Deserializer, Visitor};
use serde_json::Value as JsonValue;
use tracing::{debug, info, warn};

use super::SchedulePayload;

const ENV_PROFILES_JSON: &str = "SCHEDULER_PROFILES_JSON";
const ENV_PROFILES_PATH: &str = "SCHEDULER_PROFILES_PATH";

// ---------------------------------------------------------------------------
// Resource manager: JSON profiles with workflow-first match + discriminators.
// ---------------------------------------------------------------------------

pub(crate) struct ResourceManager;

impl ResourceManager {
    /// Resolve a scheduler resource profile for a schedule payload using configured
    /// JSON sources. Matching first selects profiles with the same `workflow`, then
    /// refines by request `diagnostic_model` + `prognostic_model`, then `model`, then the
    /// workflow-only default row (no model / diagnostic / prognostic on the profile).
    pub(crate) fn lookup_known_profile_resources(payload: &SchedulePayload) -> Option<Profile> {
        let wf = payload.workflow.trim();
        if wf.is_empty() {
            return None;
        }

        log_request_profile_discriminators(payload);

        if let Some(override_profiles) = load_profiles_from_env_cached() {
            debug!(
                workflow = %wf,
                source = "SCHEDULER_PROFILES_JSON",
                profile_count = override_profiles.profiles.len(),
                "lookup_known_profile_resources: using env override profile set"
            );
            return match_profile_for_payload(&override_profiles, payload);
        }

        let disk = read_profiles_from_disk_cached();
        debug!(
            workflow = %wf,
            source = "disk_profiles_json",
            profile_count = disk.profiles.len(),
            "lookup_known_profile_resources: using disk profile set"
        );
        match_profile_for_payload(&disk, payload)
    }

    /// Preload profiles (disk JSON) to avoid cold-start I/O.
    pub(crate) fn warm_known_profile_resources_cache() {
        if load_profiles_from_env_cached().is_none() {
            let _ = read_profiles_from_disk_cached();
        }
    }
}

// ---------------------------------------------------------------------------
// Profile model (mirrors profiles.json).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Profiles {
    pub(crate) profiles: Vec<Profile>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Profile {
    pub(crate) workflow: String,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default, rename = "prognostic_model")]
    pub(crate) prognostic_model: Option<String>,
    #[serde(default, rename = "diagnostic_model")]
    pub(crate) diagnostic_model: Option<String>,
    #[serde(default, rename = "type")]
    pub(crate) profile_type: Option<String>,
    #[serde(rename = "gpus.used")]
    pub(crate) gpus_used: usize,
    /// Carried for observability and future scheduling logic; reservation uses peak memory + GPU count only.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) average: ProfileAverage,
    pub(crate) peak: ProfilePeak,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ProfileAverage {
    #[serde(
        default,
        rename = "utilization.gpu",
        deserialize_with = "deserialize_optional_loose_string"
    )]
    pub(crate) utilization_gpu: Option<String>,
    #[serde(
        default,
        rename = "utilization.memory",
        deserialize_with = "deserialize_optional_loose_string"
    )]
    pub(crate) utilization_memory: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ProfilePeak {
    #[serde(
        default,
        rename = "utilization.gpu",
        deserialize_with = "deserialize_optional_loose_string"
    )]
    pub(crate) utilization_gpu: Option<String>,
    #[serde(
        default,
        rename = "utilization.memory",
        deserialize_with = "deserialize_optional_loose_string"
    )]
    pub(crate) utilization_memory: Option<String>,
    #[serde(rename = "memory.used", deserialize_with = "deserialize_memory_used")]
    pub(crate) memory_used: String,
    #[serde(
        default,
        rename = "memory.total",
        deserialize_with = "deserialize_optional_loose_string"
    )]
    pub(crate) memory_total: Option<String>,
}

impl Profile {
    /// Workflow used for the first stage of profile matching (and for load logs as `profile_lookup_key`).
    /// Discriminators are `model` / `diagnostic_model` / `prognostic_model` on the struct, not appended here.
    pub(crate) fn profile_lookup_key(&self) -> String {
        self.workflow.trim().to_string()
    }

    pub(crate) fn peak_memory_mib(&self) -> Option<u64> {
        parse_mib(self.peak.memory_used.as_str())
    }
}

/// Emits one `info!` line per loaded profile: workflow (first-stage key), discriminators, source.
fn info_log_loaded_scheduler_profile(source: &str, profile: &Profile, origin: Option<&str>) {
    let workflow_key = profile.profile_lookup_key();
    info!(
        source = %source,
        profile_lookup_key = %workflow_key,
        workflow = %profile.workflow.trim(),
        model = profile.model.as_deref().map(str::trim),
        diagnostic_model = profile.diagnostic_model.as_deref().map(str::trim),
        prognostic_model = profile.prognostic_model.as_deref().map(str::trim),
        gpus_used = profile.gpus_used,
        peak_memory_used = %profile.peak.memory_used,
        origin = %origin.unwrap_or("-"),
        "scheduler profile loaded"
    );
}

// ---------------------------------------------------------------------------
// Resource manager state.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct ResourceManagerState {
    env_cache_key: Option<String>,
    env_override_profiles: Option<Profiles>,
    disk_profiles_by_path: HashMap<Option<String>, Profiles>,
}

static RESOURCE_MANAGER_STATE: LazyLock<Mutex<ResourceManagerState>> =
    LazyLock::new(|| Mutex::new(ResourceManagerState::default()));

fn manager_lock() -> MutexGuard<'static, ResourceManagerState> {
    RESOURCE_MANAGER_STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn configured_profiles_json_override() -> Option<String> {
    let override_json = std::env::var(ENV_PROFILES_JSON).ok()?;
    let trimmed = override_json.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn configured_profiles_path_override() -> Option<String> {
    let path = std::env::var(ENV_PROFILES_PATH).ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn load_profiles_from_env_cached() -> Option<Profiles> {
    let override_json = configured_profiles_json_override()?;
    let mut cache = manager_lock();
    if cache.env_cache_key.as_ref() == Some(&override_json) {
        return cache.env_override_profiles.clone();
    }

    let parsed = match serde_json::from_str::<Profiles>(&override_json) {
        Ok(profiles) => {
            for p in &profiles.profiles {
                info_log_loaded_scheduler_profile("SCHEDULER_PROFILES_JSON", p, None);
            }
            Some(profiles)
        }
        Err(error) => {
            warn!(
                error = %error,
                "failed to parse scheduler profiles from SCHEDULER_PROFILES_JSON; ignoring override"
            );
            None
        }
    };

    cache.env_cache_key = Some(override_json);
    cache.env_override_profiles = parsed.clone();

    parsed
}

fn read_profiles_from_disk_cached() -> Profiles {
    let configured_path = configured_profiles_path_override();
    let mut cache = manager_lock();
    if let Some(profiles) = cache.disk_profiles_by_path.get(&configured_path) {
        return profiles.clone();
    }

    let loaded = read_profiles_from_disk(configured_path.as_deref());
    cache
        .disk_profiles_by_path
        .insert(configured_path, loaded.clone());
    loaded
}

fn read_profiles_from_disk(path_override: Option<&str>) -> Profiles {
    let paths = [
        path_override.map(str::to_string),
        Some("/app/config/profiles.json".to_string()),
        Some("platform/inference_rust/crates/worker-runtime/config/profiles.json".to_string()),
        Some("crates/worker-runtime/config/profiles.json".to_string()),
    ];

    for path in paths.into_iter().flatten() {
        let Ok(contents) = std::fs::read_to_string(path.as_str()) else {
            continue;
        };

        match serde_json::from_str::<Profiles>(&contents) {
            Ok(loaded) => {
                for p in &loaded.profiles {
                    info_log_loaded_scheduler_profile("profiles_json", p, Some(path.as_str()));
                }
                info!(
                    path = %path,
                    count = loaded.profiles.len(),
                    "loaded scheduler profiles JSON"
                );
                return loaded;
            }
            Err(error) => {
                warn!(
                    path = %path,
                    error = %error,
                    "failed to parse scheduler profile file; trying next path"
                );
            }
        }
    }

    Profiles {
        profiles: Vec::new(),
    }
}

fn log_request_profile_discriminators(payload: &SchedulePayload) {
    let wf = payload.workflow.trim();
    let v = &payload.raw_payload;
    let diagnostic = payload_string_field(v, "diagnostic_model")
        .or_else(|| payload_string_field(v, "diagnostic_model_type"));
    let prognostic = payload_string_field(v, "prognostic_model")
        .or_else(|| payload_string_field(v, "prognostic_model_type"));
    let model = payload_string_field(v, "model").or_else(|| payload_string_field(v, "model_type"));
    let rule = match (&diagnostic, &prognostic) {
        (Some(_), Some(_)) => "workflow-then-diagnostic-prognostic",
        _ if model.is_some() => "workflow-then-model-or-default-row",
        _ => "workflow-then-default-row",
    };
    debug!(
        workflow = %wf,
        diagnostic_model = diagnostic.as_deref(),
        prognostic_model = prognostic.as_deref(),
        model = model.as_deref(),
        match_rule = rule,
        "scheduler profile lookup: request discriminators"
    );
}

fn payload_string_field(v: &JsonValue, key: &str) -> Option<String> {
    let direct = v.get(key).and_then(JsonValue::as_str).map(str::trim);
    if let Some(s) = direct
        && !s.is_empty()
    {
        return Some(s.to_string());
    }
    v.get("parameters")
        .and_then(|p| p.get(key))
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn trim_nonempty_field(s: &str) -> Option<&str> {
    let t = s.trim();
    if t.is_empty() { None } else { Some(t) }
}

fn profile_model_field(p: &Profile) -> Option<&str> {
    p.model.as_deref().and_then(trim_nonempty_field)
}

fn profile_diagnostic_field(p: &Profile) -> Option<&str> {
    p.diagnostic_model.as_deref().and_then(trim_nonempty_field)
}

fn profile_prognostic_field(p: &Profile) -> Option<&str> {
    p.prognostic_model.as_deref().and_then(trim_nonempty_field)
}

/// No model / diagnostic / prognostic on the profile — workflow default / catch-all row.
fn is_unconstrained_profile(p: &Profile) -> bool {
    profile_model_field(p).is_none()
        && profile_diagnostic_field(p).is_none()
        && profile_prognostic_field(p).is_none()
}

fn match_profile_for_payload(profiles: &Profiles, payload: &SchedulePayload) -> Option<Profile> {
    let wf = trim_nonempty_field(payload.workflow.trim())?;
    let v = &payload.raw_payload;
    let req_diagnostic = payload_string_field(v, "diagnostic_model")
        .or_else(|| payload_string_field(v, "diagnostic_model_type"));
    let req_prognostic = payload_string_field(v, "prognostic_model")
        .or_else(|| payload_string_field(v, "prognostic_model_type"));
    let req_model =
        payload_string_field(v, "model").or_else(|| payload_string_field(v, "model_type"));

    let mut candidates: Vec<&Profile> = profiles
        .profiles
        .iter()
        .filter(|p| trim_nonempty_field(p.workflow.as_str()) == Some(wf))
        .collect();

    // Fanout fallback: if workflow is "<name>-fanout" and no direct match exists,
    // look for an ensemble profile under "<name>" (must have type: "ensemble").
    if candidates.is_empty()
        && let Some(base_wf) = wf.strip_suffix("-fanout")
    {
        candidates = profiles
            .profiles
            .iter()
            .filter(|p| {
                trim_nonempty_field(p.workflow.as_str()) == Some(base_wf)
                    && p.profile_type.as_deref() == Some("ensemble")
            })
            .collect();
        if !candidates.is_empty() {
            debug!(
                workflow = %wf,
                base_workflow = %base_wf,
                "match_profile: using ensemble profile via -fanout fallback"
            );
        }
    }

    if candidates.is_empty() {
        debug!(workflow = %wf, "match_profile: no profiles for workflow");
        return None;
    }

    debug!(
        workflow = %wf,
        candidate_count = candidates.len(),
        "match_profile: workflow filter"
    );

    let diag_prog_pair = match (&req_diagnostic, &req_prognostic) {
        (Some(dr), Some(pr)) => match (
            trim_nonempty_field(dr.as_str()),
            trim_nonempty_field(pr.as_str()),
        ) {
            (Some(d), Some(p)) => Some((d, p)),
            _ => None,
        },
        _ => None,
    };

    let matched: Option<&Profile> = if let Some((d_req, p_req)) = diag_prog_pair {
        let narrowed: Vec<&Profile> = candidates
            .iter()
            .copied()
            .filter(|p| {
                profile_diagnostic_field(p) == Some(d_req)
                    && profile_prognostic_field(p) == Some(p_req)
            })
            .collect();
        match narrowed.len() {
            0 => None,
            1 => Some(narrowed[0]),
            _ => {
                warn!(
                    workflow = %wf,
                    count = narrowed.len(),
                    "ambiguous scheduler profiles: multiple rows match diagnostic_model + prognostic_model"
                );
                None
            }
        }
    } else if let Some(ref m_req) = req_model {
        let m_req = trim_nonempty_field(m_req.as_str())?;
        let specific: Vec<&Profile> = candidates
            .iter()
            .copied()
            .filter(|p| profile_model_field(p) == Some(m_req))
            .collect();
        if specific.len() == 1 {
            Some(specific[0])
        } else if specific.len() > 1 {
            warn!(
                workflow = %wf,
                model = %m_req,
                count = specific.len(),
                "ambiguous scheduler profiles: multiple rows match model"
            );
            None
        } else {
            let defaults: Vec<&Profile> = candidates
                .iter()
                .copied()
                .filter(|p| is_unconstrained_profile(p))
                .collect();
            match defaults.len() {
                0 => None,
                1 => Some(defaults[0]),
                _ => {
                    warn!(
                        workflow = %wf,
                        model = %m_req,
                        count = defaults.len(),
                        "ambiguous scheduler profiles: multiple unconstrained default rows for workflow-model fallback"
                    );
                    None
                }
            }
        }
    } else {
        let defaults: Vec<&Profile> = candidates
            .iter()
            .copied()
            .filter(|p| is_unconstrained_profile(p))
            .collect();
        if defaults.len() == 1 {
            Some(defaults[0])
        } else if defaults.is_empty() && candidates.len() == 1 {
            Some(candidates[0])
        } else {
            if defaults.len() > 1 {
                warn!(
                    workflow = %wf,
                    count = defaults.len(),
                    "ambiguous scheduler profiles: multiple unconstrained rows for workflow-only match"
                );
            } else if candidates.len() > 1 {
                debug!(
                    workflow = %wf,
                    candidate_count = candidates.len(),
                    "no unconstrained default profile; workflow-only match needs a single candidate or one default row"
                );
            }
            None
        }
    };

    let Some(matched) = matched else {
        debug!(
            workflow = %wf,
            req_diagnostic = req_diagnostic.as_deref(),
            req_prognostic = req_prognostic.as_deref(),
            req_model = req_model.as_deref(),
            "match_profile: no profile matched after workflow + discriminator rules"
        );
        return None;
    };

    let Some(memory_mb) = matched.peak_memory_mib() else {
        warn!(
            workflow = %wf,
            profile_workflow = %matched.workflow,
            memory_raw = %matched.peak.memory_used,
            "matched scheduler profile has invalid peak.memory.used format"
        );
        return None;
    };

    info!(
        workflow = %wf,
        profile_workflow = %matched.workflow.trim(),
        profile_model = profile_model_field(matched),
        profile_diagnostic_model = profile_diagnostic_field(matched),
        profile_prognostic_model = profile_prognostic_field(matched),
        gpus_required = matched.gpus_used,
        memory_mb,
        "matched scheduler workflow profile"
    );
    Some(matched.clone())
}

fn parse_mib(raw: &str) -> Option<u64> {
    let mut parts = raw.split_whitespace();
    let value = parts.next()?.parse::<u64>().ok()?;

    match parts.next() {
        None => Some(value),
        Some(unit) if unit.eq_ignore_ascii_case("MiB") => Some(value),
        Some(_) => None,
    }
}

// ---------------------------------------------------------------------------
// serde helpers (YAML often uses bare numbers; JSON uses strings like "47.21 %").
// ---------------------------------------------------------------------------

fn deserialize_optional_loose_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor;

    impl<'de> Visitor<'de> for OptVisitor {
        type Value = Option<String>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("optional string or number")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D2>(self, deserializer: D2) -> Result<Self::Value, D2::Error>
        where
            D2: Deserializer<'de>,
        {
            LooseString::deserialize(deserializer).map(|s| Some(s.0))
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
            Ok(Some(v))
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }

        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
    }

    deserializer.deserialize_any(OptVisitor)
}

fn deserialize_memory_used<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct MemVisitor;

    impl<'de> Visitor<'de> for MemVisitor {
        type Value = String;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("MiB amount as integer or string like \"4096 MiB\"")
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(format!("{v} MiB"))
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(format!("{v} MiB"))
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.to_string())
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
            Ok(v)
        }
    }

    deserializer.deserialize_any(MemVisitor)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LooseString(String);

impl<'de> Deserialize<'de> for LooseString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;

        impl<'de> Visitor<'de> for V {
            type Value = LooseString;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("string or number")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
                Ok(LooseString(v.to_string()))
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
                Ok(LooseString(v))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                Ok(LooseString(v.to_string()))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                Ok(LooseString(v.to_string()))
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
                Ok(LooseString(v.to_string()))
            }
        }

        deserializer.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env;
    use serde_json::json;

    fn payload_with_raw(workflow: &str, raw: JsonValue) -> SchedulePayload {
        SchedulePayload {
            run_id: "r1".to_string(),
            workflow: workflow.to_string(),
            workflow_id: None,
            parent_run_id: None,
            fanout_profile: None,
            batch_profile: None,
            raw_payload: raw,
            resource_profile: None,
            gpus_required: 0,
            memory_mb: 0,
            dispatch_stage: "execute".to_string(),
        }
    }

    #[tokio::test]
    async fn lookup_requires_exact_workflow_match_when_no_model_fields() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        let prev_path = std::env::var(ENV_PROFILES_PATH).ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[{"workflow":"deterministic","gpus.used":1,"peak":{"memory.used":"4096 MiB"}}]}"#,
            ),
        );
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", None);

        let p = payload_with_raw("deterministic", json!({ "workflow": "deterministic" }));
        assert_eq!(
            ResourceManager::lookup_known_profile_resources(&p)
                .and_then(|prof| prof.peak_memory_mib().map(|m| (prof.gpus_used, m))),
            Some((1, 4_096))
        );

        let p2 = payload_with_raw("cpu-deterministic-workflow", json!({}));
        assert!(
            ResourceManager::lookup_known_profile_resources(&p2).is_none(),
            "known-profile fallback must not match partial workflow substrings"
        );

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", prev_path.as_deref());
    }

    #[tokio::test]
    async fn lookup_matches_workflow_and_model_key() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[
                {"workflow":"deterministic_workflow","model":"fcn","gpus.used":1,"peak":{"memory.used":"1000 MiB"}},
                {"workflow":"deterministic_workflow","model":"dlwp","gpus.used":1,"peak":{"memory.used":"2000 MiB"}}
            ]}"#,
            ),
        );
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", None);

        let p = payload_with_raw(
            "deterministic_workflow",
            json!({ "workflow": "deterministic_workflow", "model": "dlwp" }),
        );
        let prof = ResourceManager::lookup_known_profile_resources(&p).expect("match dlwp");
        assert_eq!(prof.peak_memory_mib(), Some(2_000));

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
    }

    #[tokio::test]
    async fn lookup_matches_diagnostic_and_prognostic_key() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[
                {"workflow":"diagnostic_workflow","diagnostic_model":"precipitation_afno","prognostic_model":"fcn","gpus.used":1,"peak":{"memory.used":"3000 MiB"}},
                {"workflow":"diagnostic_workflow","diagnostic_model":"precipitation_afno","prognostic_model":"dlwp","gpus.used":1,"peak":{"memory.used":"4000 MiB"}}
            ]}"#,
            ),
        );
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", None);

        let p = payload_with_raw(
            "diagnostic_workflow",
            json!({
                "workflow": "diagnostic_workflow",
                "diagnostic_model": "precipitation_afno",
                "prognostic_model": "dlwp"
            }),
        );
        let prof = ResourceManager::lookup_known_profile_resources(&p).expect("match triple key");
        assert_eq!(prof.peak_memory_mib(), Some(4_000));

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
    }

    #[tokio::test]
    async fn lookup_reads_model_from_parameters_object() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[
                {"workflow":"w","model":"m1","gpus.used":1,"peak":{"memory.used":"111 MiB"}},
                {"workflow":"w","model":"m2","gpus.used":1,"peak":{"memory.used":"222 MiB"}}
            ]}"#,
            ),
        );
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", None);

        let p = payload_with_raw("w", json!({ "parameters": { "model": "m2" } }));
        let prof = ResourceManager::lookup_known_profile_resources(&p).expect("parameters.model");
        assert_eq!(prof.peak_memory_mib(), Some(222));

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
    }

    #[tokio::test]
    async fn lookup_reuses_disk_profiles_without_reopening_file() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        let prev_path = std::env::var(ENV_PROFILES_PATH).ok();

        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        let profiles_path = temp_dir.path().join("profiles.json");
        std::fs::write(
            &profiles_path,
            r#"{"profiles":[{"workflow":"cached-workflow","gpus.used":2,"peak":{"memory.used":"4096 MiB"}}]}"#,
        )
        .expect("profiles fixture should be written");

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", None);
        test_env::set_env_var(
            "SCHEDULER_PROFILES_PATH",
            Some(profiles_path.to_string_lossy().as_ref()),
        );

        let p = payload_with_raw("cached-workflow", json!({ "workflow": "cached-workflow" }));
        assert_eq!(
            ResourceManager::lookup_known_profile_resources(&p)
                .and_then(|prof| prof.peak_memory_mib().map(|m| (prof.gpus_used, m))),
            Some((2, 4_096))
        );

        std::fs::remove_file(&profiles_path).expect("profiles fixture should be removable");

        assert_eq!(
            ResourceManager::lookup_known_profile_resources(&p)
                .and_then(|prof| prof.peak_memory_mib().map(|m| (prof.gpus_used, m))),
            Some((2, 4_096)),
            "lookup should be served from in-memory cache after first disk load"
        );

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", prev_path.as_deref());
    }

    #[tokio::test]
    async fn lookup_keeps_per_path_disk_cache_entries_isolated() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        let prev_path = std::env::var(ENV_PROFILES_PATH).ok();

        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        let profiles_a = temp_dir.path().join("profiles-a.json");
        let profiles_b = temp_dir.path().join("profiles-b.json");

        std::fs::write(
            &profiles_a,
            r#"{"profiles":[{"workflow":"cached-a","gpus.used":2,"peak":{"memory.used":"4096 MiB"}}]}"#,
        )
        .expect("profiles A fixture should be written");
        std::fs::write(
            &profiles_b,
            r#"{"profiles":[{"workflow":"cached-b","gpus.used":1,"peak":{"memory.used":"1024 MiB"}}]}"#,
        )
        .expect("profiles B fixture should be written");

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", None);

        test_env::set_env_var(
            "SCHEDULER_PROFILES_PATH",
            Some(profiles_a.to_string_lossy().as_ref()),
        );
        let pa = payload_with_raw("cached-a", json!({ "workflow": "cached-a" }));
        assert_eq!(
            ResourceManager::lookup_known_profile_resources(&pa)
                .and_then(|prof| prof.peak_memory_mib().map(|m| (prof.gpus_used, m))),
            Some((2, 4_096))
        );

        std::fs::remove_file(&profiles_a).expect("profiles A fixture should be removable");

        test_env::set_env_var(
            "SCHEDULER_PROFILES_PATH",
            Some(profiles_b.to_string_lossy().as_ref()),
        );
        let pb = payload_with_raw("cached-b", json!({ "workflow": "cached-b" }));
        assert_eq!(
            ResourceManager::lookup_known_profile_resources(&pb)
                .and_then(|prof| prof.peak_memory_mib().map(|m| (prof.gpus_used, m))),
            Some((1, 1_024))
        );

        test_env::set_env_var(
            "SCHEDULER_PROFILES_PATH",
            Some(profiles_a.to_string_lossy().as_ref()),
        );
        assert_eq!(
            ResourceManager::lookup_known_profile_resources(&pa)
                .and_then(|prof| prof.peak_memory_mib().map(|m| (prof.gpus_used, m))),
            Some((2, 4_096)),
            "lookup should still return cached A profiles after cache warmup even when another path is queried"
        );

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", prev_path.as_deref());
    }

    #[tokio::test]
    async fn lookup_json_defaults_and_model_overrides() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[
                {"workflow":"wf-mo","gpus.used":1,"peak":{"memory.used":"1000 MiB"}},
                {"workflow":"wf-mo","model":"dlwp","gpus.used":2,"peak":{"memory.used":"2000 MiB"}},
                {"workflow":"wf-mo","model":"fcn","gpus.used":1,"peak":{"memory.used":"3000 MiB"}}
            ]}"#,
            ),
        );
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", None);

        let base = payload_with_raw("wf-mo", json!({ "workflow": "wf-mo" }));
        let prof_base = ResourceManager::lookup_known_profile_resources(&base).unwrap();
        assert_eq!(prof_base.peak_memory_mib(), Some(1_000));
        assert_eq!(prof_base.gpus_used, 1);

        let dlwp = payload_with_raw("wf-mo", json!({ "workflow": "wf-mo", "model": "dlwp" }));
        let prof_dlwp = ResourceManager::lookup_known_profile_resources(&dlwp).unwrap();
        assert_eq!(prof_dlwp.peak_memory_mib(), Some(2_000));
        assert_eq!(prof_dlwp.gpus_used, 2);

        let fcn = payload_with_raw("wf-mo", json!({ "workflow": "wf-mo", "model": "fcn" }));
        let prof_fcn = ResourceManager::lookup_known_profile_resources(&fcn).unwrap();
        assert_eq!(prof_fcn.peak_memory_mib(), Some(3_000));
        assert_eq!(prof_fcn.gpus_used, 1);

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
    }

    #[tokio::test]
    async fn lookup_model_not_in_profiles_falls_back_to_default() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[
                {"workflow":"wf-fb","gpus.used":1,"peak":{"memory.used":"1000 MiB"}},
                {"workflow":"wf-fb","model":"dlwp","gpus.used":2,"peak":{"memory.used":"2000 MiB"}}
            ]}"#,
            ),
        );
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", None);

        let unknown = payload_with_raw("wf-fb", json!({ "workflow": "wf-fb", "model": "other" }));
        let prof = ResourceManager::lookup_known_profile_resources(&unknown).unwrap();
        assert_eq!(prof.peak_memory_mib(), Some(1_000));

        let dlwp = payload_with_raw("wf-fb", json!({ "workflow": "wf-fb", "model": "dlwp" }));
        let prof_d = ResourceManager::lookup_known_profile_resources(&dlwp).unwrap();
        assert_eq!(prof_d.peak_memory_mib(), Some(2_000));

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
    }

    #[tokio::test]
    async fn lookup_fanout_falls_back_to_ensemble_profile() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        let profiles_json = json!({
            "profiles": [
                {
                    "workflow": "fake-ensemble",
                    "model": "fcn",
                    "type": "ensemble",
                    "gpus.used": 1,
                    "peak": { "memory.used": "2799 MiB", "memory.total": "81559 MiB" }
                },
                {
                    "workflow": "fake-ensemble",
                    "model": "dlwp",
                    "gpus.used": 1,
                    "peak": { "memory.used": "1500 MiB", "memory.total": "81559 MiB" }
                }
            ]
        });
        test_env::set_env_var("SCHEDULER_PROFILES_JSON", Some(&profiles_json.to_string()));

        // Direct match still works
        let direct = payload_with_raw("fake-ensemble", json!({"model_type": "fcn"}));
        let prof = ResourceManager::lookup_known_profile_resources(&direct).unwrap();
        assert_eq!(prof.peak_memory_mib(), Some(2_799));

        // Fanout fallback: "<name>-fanout" resolves against the ensemble profile for "<name>"
        let fanout = payload_with_raw("fake-ensemble-fanout", json!({"model_type": "fcn"}));
        let prof_f = ResourceManager::lookup_known_profile_resources(&fanout).unwrap();
        assert_eq!(prof_f.peak_memory_mib(), Some(2_799));

        // Non-ensemble profile without "type" does NOT match via fanout
        let profiles_no_type = json!({
            "profiles": [
                {
                    "workflow": "fake-deterministic",
                    "model": "fcn",
                    "gpus.used": 1,
                    "peak": { "memory.used": "4777 MiB", "memory.total": "81559 MiB" }
                }
            ]
        });
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(&profiles_no_type.to_string()),
        );
        let fanout_no_match =
            payload_with_raw("fake-deterministic-fanout", json!({"model_type": "fcn"}));
        assert!(ResourceManager::lookup_known_profile_resources(&fanout_no_match).is_none());

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
    }

    /// Integration test: validates that the real profiles.json matches each
    /// e2s-* plugin's default_request.json via the actual ResourceManager lookup.
    #[tokio::test]
    async fn profiles_json_matches_all_e2s_plugins() {
        let _guard = test_env::env_lock().lock().await;
        let prev_json = std::env::var(ENV_PROFILES_JSON).ok();
        let prev_path = std::env::var(ENV_PROFILES_PATH).ok();

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let profiles_path = format!("{manifest_dir}/config/profiles.json");
        let plugins_dir = format!("{manifest_dir}/../../plugins");

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", None);
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", Some(&profiles_path));

        let plugins_to_test = [
            "e2s-stormcast-fcn3",
            "e2s-deterministic-earth2",
            "e2s-diagnostic",
            "e2s-deterministic",
            "e2s-deterministic-fcn",
            "e2s-ensemble",
        ];

        let mut failures: Vec<String> = Vec::new();

        for plugin_name in plugins_to_test {
            let plugin_dir = format!("{plugins_dir}/{plugin_name}");

            // Read plugin.yaml to get metadata.id
            let yaml_path = format!("{plugin_dir}/plugin.yaml");
            let yaml_content = std::fs::read_to_string(&yaml_path).unwrap_or_else(|e| {
                panic!("failed to read {yaml_path}: {e}");
            });
            let workflow_id = yaml_content
                .lines()
                .find_map(|line| line.strip_prefix("  id:").map(str::trim))
                .unwrap_or_else(|| panic!("no metadata.id found in {yaml_path}"));

            // Read default_request.json for request payload
            let request_path = format!("{plugin_dir}/examples/default_request.json");
            let request_content = std::fs::read_to_string(&request_path).unwrap_or_else(|e| {
                panic!("failed to read {request_path}: {e}");
            });
            let raw_payload: serde_json::Value = serde_json::from_str(&request_content)
                .unwrap_or_else(|e| {
                    panic!("invalid JSON in {request_path}: {e}");
                });

            let payload = payload_with_raw(workflow_id, raw_payload);
            let result = ResourceManager::lookup_known_profile_resources(&payload);

            match result {
                Some(profile) => {
                    assert!(
                        profile.gpus_used >= 1,
                        "plugin {plugin_name}: expected gpus_used >= 1, got {}",
                        profile.gpus_used
                    );
                    assert!(
                        profile.peak_memory_mib().is_some(),
                        "plugin {plugin_name}: peak_memory_mib() returned None"
                    );
                }
                None => {
                    failures.push(format!(
                        "plugin '{plugin_name}' (workflow_id='{workflow_id}'): no profile matched"
                    ));
                }
            }
        }

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_json.as_deref());
        test_env::set_env_var("SCHEDULER_PROFILES_PATH", prev_path.as_deref());

        if !failures.is_empty() {
            panic!(
                "profiles.json integration test failed for {} plugin(s):\n  - {}",
                failures.len(),
                failures.join("\n  - ")
            );
        }
    }
}
