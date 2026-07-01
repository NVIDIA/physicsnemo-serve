/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hash operation helpers for Redis.
//!
//! These are low-level Redis hash primitives intentionally separated from
//! `QueueManager` queue-domain operations.

use redis::Script;
use redis::aio::ConnectionManager;
use std::collections::HashMap;

use crate::{Result, error::QueueError};

fn validate_hash_args(key: &str, field: Option<&str>, op: &str) -> Result<()> {
    if key.trim().is_empty() {
        return Err(QueueError::Config(format!("{op}: key must be non-empty")));
    }
    if let Some(field) = field
        && field.trim().is_empty()
    {
        return Err(QueueError::Config(format!("{op}: field must be non-empty")));
    }
    Ok(())
}

fn validate_hash_fields(fields: &[String], op: &str) -> Result<()> {
    for field in fields {
        validate_hash_args("hash", Some(field.as_str()), op)?;
    }
    Ok(())
}

/// Set a field in a Redis hash.
pub async fn hset(
    conn: &mut ConnectionManager,
    key: &str,
    field: &str,
    value: &str,
) -> Result<i64> {
    validate_hash_args(key, Some(field), "hset")?;
    let n: i64 = redis::cmd("HSET")
        .arg(key)
        .arg(field)
        .arg(value)
        .query_async(conn)
        .await?;
    Ok(n)
}

/// Delete a field from a Redis hash.
pub async fn hdel(conn: &mut ConnectionManager, key: &str, field: &str) -> Result<i64> {
    validate_hash_args(key, Some(field), "hdel")?;
    let n: i64 = redis::cmd("HDEL")
        .arg(key)
        .arg(field)
        .query_async(conn)
        .await?;
    Ok(n)
}

/// Get a field from a Redis hash.
pub async fn hget(conn: &mut ConnectionManager, key: &str, field: &str) -> Result<Option<String>> {
    validate_hash_args(key, Some(field), "hget")?;
    let value: Option<String> = redis::cmd("HGET")
        .arg(key)
        .arg(field)
        .query_async(conn)
        .await?;
    Ok(value)
}

/// Get all fields and values from a Redis hash.
pub async fn hgetall(conn: &mut ConnectionManager, key: &str) -> Result<HashMap<String, String>> {
    validate_hash_args(key, None, "hgetall")?;
    let result: Vec<(String, String)> = redis::cmd("HGETALL").arg(key).query_async(conn).await?;
    Ok(result.into_iter().collect())
}

/// Get multiple fields from a Redis hash.
pub async fn hmget(
    conn: &mut ConnectionManager,
    key: &str,
    fields: &[String],
) -> Result<HashMap<String, String>> {
    validate_hash_args(key, None, "hmget")?;
    validate_hash_fields(fields, "hmget")?;
    if fields.is_empty() {
        return Ok(HashMap::new());
    }

    let values: Vec<Option<String>> = redis::cmd("HMGET")
        .arg(key)
        .arg(fields)
        .query_async(conn)
        .await?;

    Ok(fields
        .iter()
        .cloned()
        .zip(values)
        .filter_map(|(field, value)| value.map(|value| (field, value)))
        .collect())
}

/// Increment a numeric field in a Redis hash.
pub async fn hincrby(
    conn: &mut ConnectionManager,
    key: &str,
    field: &str,
    delta: i64,
) -> Result<i64> {
    validate_hash_args(key, Some(field), "hincrby")?;
    let value: i64 = redis::cmd("HINCRBY")
        .arg(key)
        .arg(field)
        .arg(delta)
        .query_async(conn)
        .await?;
    Ok(value)
}

/// Set a numeric hash field to max(current, floor) + delta.
pub async fn hmaxby_incr(
    conn: &mut ConnectionManager,
    key: &str,
    field: &str,
    floor: i64,
    delta: i64,
) -> Result<i64> {
    validate_hash_args(key, Some(field), "hmaxby_incr")?;
    if floor < 0 {
        return Err(QueueError::Config(
            "hmaxby_incr: floor must be non-negative".to_string(),
        ));
    }
    if delta < 0 {
        return Err(QueueError::Config(
            "hmaxby_incr: delta must be non-negative".to_string(),
        ));
    }

    let script = Script::new(
        r#"
local current = tonumber(redis.call('HGET', KEYS[1], ARGV[1])) or 0
local floor = tonumber(ARGV[2]) or 0
local delta = tonumber(ARGV[3]) or 0
if floor < 0 then
  return redis.error_reply('floor must be non-negative')
end
if delta < 0 then
  return redis.error_reply('delta must be non-negative')
end
local base = math.max(current, floor)
local updated = base + delta
redis.call('HSET', KEYS[1], ARGV[1], updated)
return updated
"#,
    );

    let updated: i64 = script
        .key(key)
        .arg(field)
        .arg(floor)
        .arg(delta)
        .invoke_async(conn)
        .await?;
    Ok(updated)
}

/// Decrement a numeric hash field, clamping to zero and deleting empty fields.
pub async fn hdecrby_clamp_zero_and_delete(
    conn: &mut ConnectionManager,
    key: &str,
    field: &str,
    amount: i64,
) -> Result<i64> {
    validate_hash_args(key, Some(field), "hdecrby_clamp_zero_and_delete")?;
    if amount < 0 {
        return Err(QueueError::Config(
            "hdecrby_clamp_zero_and_delete: amount must be non-negative".to_string(),
        ));
    }

    let script = Script::new(
        r#"
local current = tonumber(redis.call('HGET', KEYS[1], ARGV[1])) or 0
local amount = tonumber(ARGV[2]) or 0
if amount < 0 then
  return redis.error_reply('amount must be non-negative')
end
local updated = current - amount
if updated <= 0 then
  redis.call('HDEL', KEYS[1], ARGV[1])
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], updated)
return updated
"#,
    );

    let updated: i64 = script
        .key(key)
        .arg(field)
        .arg(amount)
        .invoke_async(conn)
        .await?;
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::{
        hdecrby_clamp_zero_and_delete, hdel, hget, hgetall, hincrby, hmaxby_incr, hmget, hset,
        validate_hash_args, validate_hash_fields,
    };
    use crate::QueueManager;
    use std::collections::HashMap;
    use std::net::TcpListener;
    use std::process::{Child, Command, Stdio};
    use std::time::Duration;
    use tempfile::TempDir;

    struct TestRedisServer {
        child: Child,
        _data_dir: TempDir,
    }

    impl Drop for TestRedisServer {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    fn reserve_port(test_name: &str) -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap_or_else(|error| {
                panic!("failed to reserve ephemeral redis port for {test_name}: {error}")
            })
            .local_addr()
            .unwrap_or_else(|error| {
                panic!("listener should expose local addr for {test_name}: {error}")
            })
            .port()
    }

    async fn wait_for_tcp_listener(port: u16, test_name: &str) {
        for _ in 0..50 {
            if tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_ok()
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("redis-server for {test_name} on port {port} did not become ready in time");
    }

    async fn spawn_test_queue_manager(test_name: &str) -> (TestRedisServer, QueueManager) {
        let port = reserve_port(test_name);
        let data_dir = tempfile::tempdir().unwrap_or_else(|error| {
            panic!("redis data dir should be created for {test_name}: {error}")
        });
        let child = Command::new("redis-server")
            .arg("--save")
            .arg("")
            .arg("--appendonly")
            .arg("no")
            .arg("--bind")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--dir")
            .arg(data_dir.path())
            .arg("--loglevel")
            .arg("warning")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|error| {
                panic!("failed to spawn redis-server for {test_name}: {error}")
            });
        wait_for_tcp_listener(port, test_name).await;
        let server = TestRedisServer {
            child,
            _data_dir: data_dir,
        };
        let redis_url = format!("redis://127.0.0.1:{port}");
        let qm = QueueManager::new(redis_url.as_str())
            .await
            .unwrap_or_else(|error| {
                panic!("queue manager should connect to test redis for {test_name}: {error}")
            });
        (server, qm)
    }

    #[test]
    fn validate_hash_args_rejects_empty_key() {
        let result = validate_hash_args("", Some("field"), "hset");
        assert!(result.is_err(), "empty key must fail validation");
    }

    #[test]
    fn validate_hash_args_rejects_empty_field_for_field_ops() {
        let result = validate_hash_args("run:1", Some(""), "hset");
        assert!(result.is_err(), "empty field must fail validation");
    }

    #[test]
    fn validate_hash_args_accepts_non_empty_field_for_field_ops() {
        let result = validate_hash_args("run:1", Some("stage"), "hset");
        assert!(result.is_ok(), "non-empty key/field should pass validation");
    }

    #[test]
    fn validate_hash_args_allows_fieldless_hgetall() {
        let result = validate_hash_args("run:1", None, "hgetall");
        assert!(result.is_ok(), "hgetall only requires key");
    }

    #[test]
    fn validate_hash_fields_rejects_empty_fields() {
        let result = validate_hash_fields(&["stage".to_string(), "".to_string()], "hmget");
        assert!(result.is_err(), "empty hash field must fail validation");
    }

    #[test]
    fn validate_hash_fields_accepts_non_empty_fields() {
        let result = validate_hash_fields(&["stage".to_string(), "status".to_string()], "hmget");
        assert!(result.is_ok(), "non-empty fields should pass validation");
    }

    #[tokio::test]
    async fn hash_field_lifecycle_round_trips_through_hset_hget_and_hdel() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-lifecycle").await;
        let mut conn = qm.connection();

        assert_eq!(
            hset(&mut conn, "run:1", "stage", "queued").await.unwrap(),
            1
        );
        assert_eq!(
            hget(&mut conn, "run:1", "stage").await.unwrap(),
            Some("queued".to_string())
        );

        assert_eq!(
            hset(&mut conn, "run:1", "stage", "running").await.unwrap(),
            0
        );
        assert_eq!(
            hget(&mut conn, "run:1", "stage").await.unwrap(),
            Some("running".to_string())
        );

        assert_eq!(hdel(&mut conn, "run:1", "stage").await.unwrap(), 1);
        assert_eq!(hget(&mut conn, "run:1", "stage").await.unwrap(), None);
    }

    #[tokio::test]
    async fn hgetall_returns_all_fields_for_hash() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-hgetall").await;
        let mut conn = qm.connection();

        hset(&mut conn, "run:2", "stage", "running").await.unwrap();
        hset(&mut conn, "run:2", "status", "ok").await.unwrap();

        let values = hgetall(&mut conn, "run:2").await.unwrap();
        assert_eq!(
            values,
            HashMap::from([
                ("stage".to_string(), "running".to_string()),
                ("status".to_string(), "ok".to_string()),
            ])
        );
    }

    #[tokio::test]
    async fn hmget_returns_only_present_fields_and_handles_empty_requests() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-hmget").await;
        let mut conn = qm.connection();

        hset(&mut conn, "run:3", "stage", "running").await.unwrap();
        hset(&mut conn, "run:3", "status", "ok").await.unwrap();

        let values = hmget(
            &mut conn,
            "run:3",
            &[
                "stage".to_string(),
                "missing".to_string(),
                "status".to_string(),
            ],
        )
        .await
        .unwrap();

        assert_eq!(
            values,
            HashMap::from([
                ("stage".to_string(), "running".to_string()),
                ("status".to_string(), "ok".to_string()),
            ])
        );

        let empty = hmget(&mut conn, "run:3", &[]).await.unwrap();
        assert!(empty.is_empty(), "empty field lists should short-circuit");
    }

    #[tokio::test]
    async fn hincrby_accumulates_numeric_hash_values() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-hincrby").await;
        let mut conn = qm.connection();

        assert_eq!(hincrby(&mut conn, "run:4", "retries", 2).await.unwrap(), 2);
        assert_eq!(hincrby(&mut conn, "run:4", "retries", 3).await.unwrap(), 5);
        assert_eq!(
            hget(&mut conn, "run:4", "retries").await.unwrap(),
            Some("5".to_string())
        );
    }

    #[tokio::test]
    async fn hmaxby_incr_applies_floor_before_incrementing() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-hmaxby-incr").await;
        let mut conn = qm.connection();

        hset(&mut conn, "run:5", "reserved_mb", "10").await.unwrap();

        assert_eq!(
            hmaxby_incr(&mut conn, "run:5", "reserved_mb", 8, 5)
                .await
                .unwrap(),
            15
        );
        assert_eq!(
            hmaxby_incr(&mut conn, "run:5", "reserved_mb", 20, 4)
                .await
                .unwrap(),
            24
        );
        assert_eq!(
            hget(&mut conn, "run:5", "reserved_mb").await.unwrap(),
            Some("24".to_string())
        );
    }

    #[tokio::test]
    async fn hmaxby_incr_rejects_negative_floor_and_delta() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-hmaxby-errors").await;
        let mut conn = qm.connection();

        let floor_error = hmaxby_incr(&mut conn, "run:6", "reserved_mb", -1, 1)
            .await
            .unwrap_err();
        assert!(
            floor_error
                .to_string()
                .contains("hmaxby_incr: floor must be non-negative"),
            "unexpected error: {floor_error}"
        );

        let delta_error = hmaxby_incr(&mut conn, "run:6", "reserved_mb", 1, -1)
            .await
            .unwrap_err();
        assert!(
            delta_error
                .to_string()
                .contains("hmaxby_incr: delta must be non-negative"),
            "unexpected error: {delta_error}"
        );
    }

    #[tokio::test]
    async fn hdecrby_clamp_zero_and_delete_updates_and_removes_fields() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-hdecrby").await;
        let mut conn = qm.connection();

        hset(&mut conn, "run:7", "reserved_mb", "10").await.unwrap();

        assert_eq!(
            hdecrby_clamp_zero_and_delete(&mut conn, "run:7", "reserved_mb", 3)
                .await
                .unwrap(),
            7
        );
        assert_eq!(
            hget(&mut conn, "run:7", "reserved_mb").await.unwrap(),
            Some("7".to_string())
        );

        assert_eq!(
            hdecrby_clamp_zero_and_delete(&mut conn, "run:7", "reserved_mb", 10)
                .await
                .unwrap(),
            0
        );
        assert_eq!(hget(&mut conn, "run:7", "reserved_mb").await.unwrap(), None);
    }

    #[tokio::test]
    async fn hdecrby_clamp_zero_and_delete_rejects_negative_amount() {
        let (_server, qm) = spawn_test_queue_manager("hash-ops-hdecrby-errors").await;
        let mut conn = qm.connection();

        let error = hdecrby_clamp_zero_and_delete(&mut conn, "run:8", "reserved_mb", -1)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("hdecrby_clamp_zero_and_delete: amount must be non-negative"),
            "unexpected error: {error}"
        );
    }
}
