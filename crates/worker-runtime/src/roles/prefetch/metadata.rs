/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result};

use crate::traits::BoxFuture;

/// Fetches workflow model metadata from a backing store.
///
/// Production: reads `workflow:{name}:model_metadata` from Redis.
/// Tests: returns canned JSON via [`StaticMetadataProvider`].
pub trait WorkflowMetadataProvider: Send + Sync + 'static {
    fn get_metadata<'a>(&'a self, workflow_name: &'a str) -> BoxFuture<'a, Result<Option<String>>>;
}

/// Redis-backed metadata provider using `scicomp_rq::QueueManager`.
pub struct RedisMetadataProvider {
    qm: scicomp_rq::QueueManager,
}

impl RedisMetadataProvider {
    pub fn new(qm: scicomp_rq::QueueManager) -> Self {
        Self { qm }
    }
}

impl WorkflowMetadataProvider for RedisMetadataProvider {
    fn get_metadata<'a>(&'a self, workflow_name: &'a str) -> BoxFuture<'a, Result<Option<String>>> {
        Box::pin(async move {
            let key = format!("workflow:{workflow_name}:model_metadata");
            let mut conn = self.qm.connection();
            let result: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .context("failed to fetch workflow model metadata")?;
            Ok(result)
        })
    }
}

/// Static metadata provider for unit/integration tests.
#[derive(Default)]
pub struct StaticMetadataProvider {
    metadata: std::collections::HashMap<String, String>,
}

impl StaticMetadataProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_entry(mut self, workflow: &str, json: &str) -> Self {
        self.metadata.insert(workflow.to_string(), json.to_string());
        self
    }
}

impl WorkflowMetadataProvider for StaticMetadataProvider {
    fn get_metadata<'a>(&'a self, workflow_name: &'a str) -> BoxFuture<'a, Result<Option<String>>> {
        let result = self.metadata.get(workflow_name).cloned();
        Box::pin(async move { Ok(result) })
    }
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

    #[tokio::test]
    async fn static_metadata_returns_configured_entry() {
        let provider = StaticMetadataProvider::new().with_entry("det", &gfs_metadata_json());
        let result = provider.get_metadata("det").await.unwrap();
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn static_metadata_returns_none_for_unknown() {
        let provider = StaticMetadataProvider::new();
        let result = provider.get_metadata("unknown").await.unwrap();
        assert!(result.is_none());
    }
}
