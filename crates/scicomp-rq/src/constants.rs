/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constants used throughout scicomp-rq.
//!
//! Centralizes magic strings and default values for easier maintenance.

/// Redis stream field names.
pub mod fields {
    /// Field name for the workflow run identifier.
    pub const RUN_ID: &str = "run_id";
    /// Field name for the JSON payload.
    pub const PAYLOAD: &str = "payload";
    /// Field name for the processing stage.
    pub const STAGE: &str = "stage";
    /// Field name for the last update timestamp.
    pub const UPDATED_AT: &str = "updated_at";
}

/// Redis key prefixes and patterns.
pub mod keys {
    /// Default prefix for stream keys.
    pub const DEFAULT_STREAM_PREFIX: &str = "stream:";
    /// Prefix for run state hashes.
    pub const RUN_HASH_PREFIX: &str = "run:";
}

/// Consumer group configuration.
pub mod groups {
    /// Suffix appended to stream names to create group names.
    pub const GROUP_SUFFIX: &str = ":grp";
}

/// Environment variable names.
pub mod env {
    /// Redis connection URL.
    pub const REDIS_URL: &str = "REDIS_URL";
    /// Path to the queue configuration file.
    pub const QUEUE_CONFIG: &str = "QUEUE_CONFIG";
    /// Override for stream prefix.
    pub const REDIS_STREAM_PREFIX: &str = "REDIS_STREAM_PREFIX";
}

/// Default values.
pub mod defaults {
    /// Default Redis URL when not specified.
    pub const REDIS_URL: &str = "redis://127.0.0.1:6379";
    /// Default block timeout for XREADGROUP in milliseconds.
    pub const BLOCK_MS: usize = 5000;
    /// Default message count for XREADGROUP.
    pub const READ_COUNT: usize = 10;
    /// Default minimum idle time for XAUTOCLAIM in milliseconds.
    pub const MIN_IDLE_MS: u64 = 60_000; // 1 minute
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_names_have_expected_values() {
        // Verify field names have the expected values (not just non-empty)
        assert_eq!(fields::RUN_ID, "run_id");
        assert_eq!(fields::PAYLOAD, "payload");
        assert_eq!(fields::STAGE, "stage");
    }

    #[test]
    fn test_key_prefixes_end_with_separator() {
        // Prefixes should end with a separator for proper concatenation
        assert!(keys::DEFAULT_STREAM_PREFIX.ends_with(':'));
        assert!(keys::RUN_HASH_PREFIX.ends_with(':'));
    }

    #[test]
    fn test_group_suffix_starts_with_separator() {
        assert!(groups::GROUP_SUFFIX.starts_with(':'));
    }

    #[test]
    fn test_default_redis_url_valid() {
        // Should be parseable by the redis crate
        assert!(defaults::REDIS_URL.starts_with("redis://"));
    }
}
