/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

fn default_required() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PrefetchOpKind {
    HttpFetch,
    ObjectStoreFetch,
    FileCopy,
}

impl PrefetchOpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HttpFetch => "http_fetch",
            Self::ObjectStoreFetch => "object_store_fetch",
            Self::FileCopy => "file_copy",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefetchPlanItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<PrefetchOpKind>,
    pub source_uri: String,
    pub target_artifact_name: String,
    #[serde(default = "default_required")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_range: Option<ByteRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

impl PrefetchPlanItem {
    pub fn effective_kind(&self) -> PrefetchOpKind {
        if let Some(kind) = self.kind {
            return kind;
        }

        if self.source_uri.starts_with("https://") || self.source_uri.starts_with("http://") {
            return PrefetchOpKind::HttpFetch;
        }

        if self.source_uri.starts_with("s3://")
            || self.source_uri.starts_with("gs://")
            || self.source_uri.starts_with("az://")
            || self.source_uri.starts_with("azure://")
        {
            return PrefetchOpKind::ObjectStoreFetch;
        }

        PrefetchOpKind::FileCopy
    }

    pub fn effective_cache_key(&self) -> String {
        self.cache_key.clone().unwrap_or_else(|| {
            if let Some(byte_range) = &self.byte_range {
                format!(
                    "{}:{}#{}:{}",
                    self.effective_kind().as_str(),
                    self.source_uri,
                    byte_range.offset,
                    byte_range.length
                )
            } else {
                format!("{}:{}", self.effective_kind().as_str(), self.source_uri)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_required_to_true() {
        let item: PrefetchPlanItem = serde_json::from_value(json!({
            "source_uri": "s3://bucket/path/file.bin",
            "target_artifact_name": "file"
        }))
        .expect("plan item should parse");

        assert!(item.required);
        assert_eq!(item.effective_kind(), PrefetchOpKind::ObjectStoreFetch);
    }

    #[test]
    fn cache_key_uses_explicit_value_when_present() {
        let item = PrefetchPlanItem {
            kind: Some(PrefetchOpKind::HttpFetch),
            source_uri: "https://example.com/file.bin".to_string(),
            target_artifact_name: "file".to_string(),
            required: true,
            byte_range: None,
            cache_key: Some("custom-key".to_string()),
            media_type: None,
            headers: BTreeMap::new(),
        };

        assert_eq!(item.effective_cache_key(), "custom-key");
    }

    #[test]
    fn cache_key_includes_byte_range_when_present() {
        let item = PrefetchPlanItem {
            kind: Some(PrefetchOpKind::HttpFetch),
            source_uri: "https://example.com/file.bin".to_string(),
            target_artifact_name: "file".to_string(),
            required: true,
            byte_range: Some(ByteRange {
                offset: 128,
                length: 256,
            }),
            cache_key: None,
            media_type: None,
            headers: BTreeMap::new(),
        };

        assert_eq!(
            item.effective_cache_key(),
            "http_fetch:https://example.com/file.bin#128:256"
        );
    }

    #[test]
    fn infers_http_fetch_kind_from_http_source_uri() {
        let item: PrefetchPlanItem = serde_json::from_value(json!({
            "source_uri": "https://example.com/reference.txt",
            "target_artifact_name": "reference"
        }))
        .expect("plan item should parse");

        assert_eq!(item.effective_kind(), PrefetchOpKind::HttpFetch);
    }

    #[test]
    fn infers_file_copy_kind_from_local_source_uri() {
        let item: PrefetchPlanItem = serde_json::from_value(json!({
            "source_uri": "/tmp/reference.txt",
            "target_artifact_name": "reference"
        }))
        .expect("plan item should parse");

        assert_eq!(item.effective_kind(), PrefetchOpKind::FileCopy);
    }

    #[test]
    fn explicit_kind_overrides_inferred_kind() {
        let item: PrefetchPlanItem = serde_json::from_value(json!({
            "kind": "file_copy",
            "source_uri": "https://example.com/reference.txt",
            "target_artifact_name": "reference"
        }))
        .expect("plan item should parse");

        assert_eq!(item.effective_kind(), PrefetchOpKind::FileCopy);
    }
}
