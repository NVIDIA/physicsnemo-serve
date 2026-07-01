/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use scicomp_rq::Message;

const MAX_DLQ_PAYLOAD_BYTES: usize = 4096;
const DLQ_PAYLOAD_TRUNCATION_SUFFIX: &str = "...(truncated)";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryDlqPolicy {
    max_retries: usize,
    dlq_stream: String,
}

impl RetryDlqPolicy {
    pub fn new(max_retries: usize, dlq_stream: impl Into<String>) -> Self {
        Self {
            max_retries: max_retries.max(1),
            dlq_stream: dlq_stream.into(),
        }
    }

    pub fn max_retries(&self) -> usize {
        self.max_retries
    }

    pub fn dlq_stream(&self) -> &str {
        self.dlq_stream.as_str()
    }

    pub fn build_dlq_payload(
        &self,
        msg: &Message,
        logical_stream: &str,
        error: &str,
        attempts: usize,
    ) -> Result<String> {
        let bounded_payload = bounded_dlq_payload(msg.payload());
        serde_json::to_string(&serde_json::json!({
            "run_id": msg.run_id(),
            "status": "failed",
            "source_stream": logical_stream,
            "source_stage": msg.stage(),
            "source_message_id": msg.id(),
            "error": error,
            "attempts": attempts,
            "payload": bounded_payload,
        }))
        .context("failed to serialize DLQ payload")
    }
}

fn bounded_dlq_payload(payload: &str) -> String {
    if payload.len() <= MAX_DLQ_PAYLOAD_BYTES {
        return payload.to_string();
    }

    let mut cutoff = MAX_DLQ_PAYLOAD_BYTES;
    while cutoff > 0 && !payload.is_char_boundary(cutoff) {
        cutoff -= 1;
    }

    let mut bounded = payload[..cutoff].to_string();
    bounded.push_str(DLQ_PAYLOAD_TRUNCATION_SUFFIX);
    bounded
}

#[derive(Debug, Default, Clone)]
pub struct LocalFailureTracker {
    attempts: Arc<Mutex<HashMap<String, usize>>>,
}

impl LocalFailureTracker {
    pub fn increment(&self, msg: &Message) -> usize {
        let key = message_key(msg);
        let mut attempts = self.attempts.lock().unwrap_or_else(|e| e.into_inner());
        let entry = attempts.entry(key).or_insert(0);
        *entry = entry.saturating_add(1);
        *entry
    }

    pub fn clear(&self, msg: &Message) {
        let key = message_key(msg);
        let mut attempts = self.attempts.lock().unwrap_or_else(|e| e.into_inner());
        attempts.remove(&key);
    }
}

fn message_key(msg: &Message) -> String {
    format!("{}::{}", msg.stream(), msg.id())
}
