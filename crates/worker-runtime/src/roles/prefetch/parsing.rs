/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Deserialize)]
pub(super) struct RawWorkflowMetadata {
    model_name: Option<String>,
    model_category: Option<String>,
    data_source: Option<String>,
    uri_prefix: Option<String>,
    variables: Option<Vec<String>>,
    lead_times: Option<Vec<i32>>,
    interp_method: Option<String>,
    time_field: Option<String>,
}

pub(super) fn parse_model_metadata(
    json: &str,
    workflow_name: &str,
) -> Result<super::plan::WorkflowMetadata> {
    let raw: RawWorkflowMetadata =
        serde_json::from_str(json).context("failed to parse model metadata JSON")?;

    let variables = raw
        .variables
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("model metadata missing variables for {workflow_name}"))?;
    let lead_times = raw
        .lead_times
        .filter(|t| !t.is_empty())
        .ok_or_else(|| anyhow!("model metadata missing lead_times for {workflow_name}"))?;

    Ok(super::plan::WorkflowMetadata {
        model_name: raw.model_name.unwrap_or_else(|| workflow_name.to_string()),
        model_category: raw
            .model_category
            .unwrap_or_else(|| "deterministic".to_string()),
        data_source: raw
            .data_source
            .unwrap_or_else(|| "gfs".to_string())
            .to_lowercase(),
        uri_prefix: raw
            .uri_prefix
            .unwrap_or_else(|| "noaa-gfs-bdp-pds".to_string()),
        variables,
        lead_times,
        interp_method: raw.interp_method,
        time_field: raw.time_field,
    })
}

pub(super) fn extract_times_from_inputs(
    inputs: &JsonValue,
    time_field: Option<&str>,
) -> Result<Vec<DateTime<Utc>>> {
    let mut fields = Vec::new();
    if let Some(field) = time_field {
        fields.push(field);
    }
    fields.push("time");
    fields.push("forecast_times");

    let times_array = fields
        .iter()
        .find_map(|field| inputs.get(*field).and_then(|v| v.as_array()))
        .context("missing or invalid time field in inputs")?;

    let mut times = Vec::new();
    for time_val in times_array {
        if let Some(ts) = time_val.as_str() {
            let ts_cleaned = ts.trim_end_matches('Z');
            let ts_cleaned = if let Some(t_pos) = ts_cleaned.find('T') {
                if let Some(plus_pos) = ts_cleaned[t_pos..].rfind('+') {
                    &ts_cleaned[..t_pos + plus_pos]
                } else if let Some(minus_pos) = ts_cleaned[t_pos..].rfind('-') {
                    if ts_cleaned[t_pos + minus_pos..].len() <= 3 {
                        ts_cleaned
                    } else {
                        &ts_cleaned[..t_pos + minus_pos]
                    }
                } else {
                    ts_cleaned
                }
            } else {
                ts_cleaned
            };

            if let Ok(naive) = NaiveDateTime::parse_from_str(ts_cleaned, "%Y-%m-%dT%H:%M:%S") {
                times.push(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc));
            }
        }
    }

    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gfs_metadata_json() -> String {
        json!({
            "model_name": "FCN",
            "data_source": "GFS",
            "uri_prefix": "noaa-gfs-bdp-pds",
            "variables": ["t2m"],
            "lead_times": [0],
            "time_field": "time"
        })
        .to_string()
    }

    // -- parse_model_metadata -------------------------------------------------

    #[test]
    fn parse_metadata_extracts_all_fields() {
        let json = gfs_metadata_json();
        let m = parse_model_metadata(&json, "test").unwrap();
        assert_eq!(m.model_name, "FCN");
        assert_eq!(m.data_source, "gfs");
        assert_eq!(m.variables, vec!["t2m"]);
        assert_eq!(m.lead_times, vec![0]);
    }

    #[test]
    fn parse_metadata_defaults_missing_optionals() {
        let json = json!({"variables": ["z500"], "lead_times": [6]}).to_string();
        let m = parse_model_metadata(&json, "wf").unwrap();
        assert_eq!(m.model_name, "wf");
        assert_eq!(m.data_source, "gfs");
        assert_eq!(m.uri_prefix, "noaa-gfs-bdp-pds");
    }

    #[test]
    fn parse_metadata_rejects_empty_variables() {
        let json = json!({"variables": [], "lead_times": [0]}).to_string();
        let err = parse_model_metadata(&json, "wf").unwrap_err();
        assert!(err.to_string().contains("variables"));
    }

    #[test]
    fn parse_metadata_rejects_missing_lead_times() {
        let json = json!({"variables": ["t2m"]}).to_string();
        let err = parse_model_metadata(&json, "wf").unwrap_err();
        assert!(err.to_string().contains("lead_times"));
    }

    #[test]
    fn parse_metadata_rejects_invalid_json() {
        let err = parse_model_metadata("not json", "wf").unwrap_err();
        assert!(err.to_string().contains("parse"));
    }

    // -- extract_times --------------------------------------------------------

    #[test]
    fn extract_times_parses_iso8601_with_z() {
        let inputs = json!({"time": ["2024-01-15T12:00:00Z"]});
        let times = extract_times_from_inputs(&inputs, None).unwrap();
        assert_eq!(times.len(), 1);
        assert_eq!(
            times[0].format("%Y-%m-%dT%H:%M:%S").to_string(),
            "2024-01-15T12:00:00"
        );
    }

    #[test]
    fn extract_times_uses_custom_time_field() {
        let inputs = json!({"forecast_times": ["2024-06-01T00:00:00Z"]});
        let times = extract_times_from_inputs(&inputs, Some("forecast_times")).unwrap();
        assert_eq!(times.len(), 1);
    }

    #[test]
    fn extract_times_errors_on_missing_field() {
        let inputs = json!({"other": "stuff"});
        assert!(extract_times_from_inputs(&inputs, None).is_err());
    }

    #[test]
    fn extract_times_handles_multiple_timestamps() {
        let inputs = json!({
            "time": [
                "2024-01-15T00:00:00Z",
                "2024-01-15T06:00:00Z",
                "2024-01-15T12:00:00Z"
            ]
        });
        let times = extract_times_from_inputs(&inputs, None).unwrap();
        assert_eq!(times.len(), 3);
    }
}
