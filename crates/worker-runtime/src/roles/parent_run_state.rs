/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use redis::AsyncCommands;
use scicomp_rq::QueueManager;
use tokio::sync::Mutex;

use crate::traits::BoxFuture;

const DEFAULT_TERMINAL_PREFIX: &str = "parent_terminal";
const DEFAULT_TERMINAL_TTL_SECS: u64 = 24 * 60 * 60;

pub(crate) trait ParentRunStateStore: Send + Sync {
    fn is_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<bool>>;

    fn mark_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>>;
}

#[derive(Debug, Default)]
pub(crate) struct InMemoryParentRunStateStore {
    terminated: Arc<Mutex<HashSet<String>>>,
}

impl InMemoryParentRunStateStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl ParentRunStateStore for InMemoryParentRunStateStore {
    fn is_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let state = self.terminated.lock().await;
            Ok(state.contains(parent_run_id))
        })
    }

    fn mark_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut state = self.terminated.lock().await;
            state.insert(parent_run_id.to_string());
            Ok(())
        })
    }
}

pub(crate) struct RedisParentRunStateStore {
    qm: QueueManager,
    key_prefix: String,
    ttl_secs: u64,
}

impl RedisParentRunStateStore {
    pub(crate) fn new(qm: QueueManager) -> Self {
        Self {
            qm,
            key_prefix: DEFAULT_TERMINAL_PREFIX.to_string(),
            ttl_secs: DEFAULT_TERMINAL_TTL_SECS,
        }
    }

    fn key(&self, parent_run_id: &str) -> String {
        format!("{}:{}", self.key_prefix, parent_run_id)
    }
}

impl ParentRunStateStore for RedisParentRunStateStore {
    fn is_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let exists: bool = conn.exists(self.key(parent_run_id)).await?;
            Ok(exists)
        })
    }

    fn mark_terminal<'a>(&'a self, parent_run_id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut conn = self.qm.connection();
            let _: () = conn
                .set_ex(self.key(parent_run_id), "terminal", self.ttl_secs)
                .await?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_terminal_store_tracks_terminal_parent() {
        let store = InMemoryParentRunStateStore::new();
        assert!(!store.is_terminal("parent-a").await.unwrap());
        store.mark_terminal("parent-a").await.unwrap();
        assert!(store.is_terminal("parent-a").await.unwrap());
        assert!(!store.is_terminal("parent-b").await.unwrap());
    }
}
