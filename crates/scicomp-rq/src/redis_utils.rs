/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Redis response parsing utilities.
//!
//! This module provides helpers for parsing complex Redis response structures,
//! particularly for stream operations like XREADGROUP and XAUTOCLAIM.

use crate::QueueError;
use crate::error::Result;
use crate::types::Message;
use tracing::warn;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct ParseDiagnostics {
    dropped_stream_entries: usize,
    dropped_messages: usize,
}

/// Convert a Redis value to a String, if possible.
pub fn value_to_string(v: &redis::Value) -> Option<String> {
    match v {
        redis::Value::BulkString(b) => String::from_utf8(b.clone()).ok(),
        redis::Value::SimpleString(s) => Some(s.clone()),
        redis::Value::Okay => Some("OK".to_string()),
        redis::Value::Int(i) => Some(i.to_string()),
        redis::Value::Nil => None,
        _ => None,
    }
}

fn parse_stream_entries_with_diagnostics(
    response: redis::Value,
) -> (Vec<(String, String, String, String)>, ParseDiagnostics) {
    let mut entries = Vec::new();
    let mut diagnostics = ParseDiagnostics::default();

    match response {
        redis::Value::Nil => {
            // Timeout, no messages
        }
        redis::Value::Array(streams) => {
            for stream_entry in streams {
                let (stream_entries, stream_diagnostics) =
                    extract_stream_messages_with_diagnostics(stream_entry);
                entries.extend(stream_entries);
                diagnostics.dropped_stream_entries += stream_diagnostics.dropped_stream_entries;
                diagnostics.dropped_messages += stream_diagnostics.dropped_messages;
            }
        }
        other => {
            diagnostics.dropped_stream_entries += 1;
            warn!(?other, "unexpected XREADGROUP response structure");
        }
    }

    (entries, diagnostics)
}

/// Parse the response from XREADGROUP (nested arrays of streams and messages).
///
/// Returns a list of (msg_id, run_id, payload, stage) tuples.
pub fn parse_stream_entries(response: redis::Value) -> Vec<(String, String, String, String)> {
    let (entries, diagnostics) = parse_stream_entries_with_diagnostics(response);
    if diagnostics.dropped_stream_entries > 0 || diagnostics.dropped_messages > 0 {
        warn!(
            dropped_stream_entries = diagnostics.dropped_stream_entries,
            dropped_messages = diagnostics.dropped_messages,
            "dropped malformed stream entries/messages while parsing XREADGROUP response"
        );
    }
    entries
}

/// Parse the response from XREADGROUP and return Message structs.
///
/// This is the preferred method for new code. Extracts all fields including stage,
/// and populates the stream and group fields for self-contained message operations.
///
/// # Arguments
///
/// * `response` - The raw Redis response from XREADGROUP
/// * `stream` - The full Redis stream key (e.g., "stream:prefetch")
/// * `group` - The consumer group name (e.g., "prefetch:grp")
pub fn parse_stream_messages(response: redis::Value, stream: &str, group: &str) -> Vec<Message> {
    parse_stream_entries(response)
        .into_iter()
        .map(|(id, run_id, payload, stage)| Message {
            id,
            stream: stream.to_string(),
            group: group.to_string(),
            run_id,
            payload,
            stage,
        })
        .collect()
}

/// Parse the response from XCLAIM (array of message entries).
pub fn parse_xclaim_messages(response: redis::Value, stream: &str, group: &str) -> Vec<Message> {
    let redis::Value::Array(items) = response else {
        return Vec::new();
    };

    items
        .into_iter()
        .filter_map(|entry| parse_message_entry(&entry))
        .map(|(id, run_id, payload, stage)| Message {
            id,
            stream: stream.to_string(),
            group: group.to_string(),
            run_id,
            payload,
            stage,
        })
        .collect()
}

/// Parse XPENDING range response into message IDs.
pub fn parse_xpending_ids(response: redis::Value) -> Vec<String> {
    let redis::Value::Array(items) = response else {
        return Vec::new();
    };

    let mut ids = Vec::new();
    for item in items {
        let redis::Value::Array(parts) = item else {
            continue;
        };
        if parts.is_empty() {
            continue;
        }
        if let Some(id) = value_to_string(&parts[0]) {
            ids.push(id);
        }
    }

    ids
}

/// Returns parsed stream messages with diagnostics about malformed entries.
fn extract_stream_messages_with_diagnostics(
    stream_entry: redis::Value,
) -> (Vec<(String, String, String, String)>, ParseDiagnostics) {
    let mut diagnostics = ParseDiagnostics::default();

    let redis::Value::Array(parts) = stream_entry else {
        diagnostics.dropped_stream_entries += 1;
        return (Vec::new(), diagnostics);
    };

    if parts.len() != 2 {
        diagnostics.dropped_stream_entries += 1;
        return (Vec::new(), diagnostics);
    }

    let redis::Value::Array(items) = &parts[1] else {
        diagnostics.dropped_stream_entries += 1;
        return (Vec::new(), diagnostics);
    };

    let mut messages = Vec::new();
    for item in items {
        if let Some(message) = parse_message_entry(item) {
            messages.push(message);
        } else {
            diagnostics.dropped_messages += 1;
        }
    }

    (messages, diagnostics)
}

/// Parsed message entry: (msg_id, run_id, payload, stage)
fn parse_message_entry(entry: &redis::Value) -> Option<(String, String, String, String)> {
    let redis::Value::Array(parts) = entry else {
        return None;
    };

    if parts.len() != 2 {
        return None;
    }

    let msg_id = value_to_string(&parts[0])?;
    let fields = extract_message_fields(&parts[1])?;

    Some((msg_id, fields.run_id, fields.payload, fields.stage))
}

/// Extracted message fields from Redis response.
struct MessageFields {
    run_id: String,
    payload: String,
    stage: String,
}

fn extract_message_fields(fields: &redis::Value) -> Option<MessageFields> {
    let redis::Value::Array(kvs) = fields else {
        return None;
    };
    if kvs.len() % 2 != 0 {
        // Field list must be strict key/value pairs.
        return None;
    }

    let mut run_id = None;
    let mut payload = None;
    let mut stage = None;

    let mut i = 0;
    while i < kvs.len() {
        let key = value_to_string(&kvs[i])?;
        let val = value_to_string(&kvs[i + 1])?;

        match key.as_str() {
            crate::fields::RUN_ID => run_id = Some(val),
            crate::fields::PAYLOAD => payload = Some(val),
            crate::fields::STAGE => stage = Some(val),
            _ => {}
        }

        i += 2;
    }

    let (run_id, payload) = match (run_id, payload) {
        (Some(r), Some(p)) if !r.trim().is_empty() && !p.trim().is_empty() => (r, p),
        _ => return None,
    };

    let stage = match stage {
        Some(s) if !s.trim().is_empty() => s,
        Some(_) => return None,
        None => String::new(),
    };

    Some(MessageFields {
        run_id,
        payload,
        stage,
    })
}

/// Parse the response from XAUTOCLAIM and return full Message objects.
///
/// Unlike [`parse_xautoclaim_response`] which only returns message IDs, this function
/// parses the full message data including run_id, payload, and stage fields.
///
/// # Arguments
///
/// * `val` - The raw Redis response from XAUTOCLAIM
/// * `stream` - The full Redis stream key (e.g., "stream:prefetch")
/// * `group` - The consumer group name (e.g., "prefetch:grp")
///
/// # Returns
///
/// A tuple of (next_start_id, messages) where messages is a Vec of fully populated Message objects.
pub fn parse_xautoclaim_messages(
    val: redis::Value,
    stream: &str,
    group: &str,
) -> Result<(String, Vec<Message>)> {
    let redis::Value::Array(mut outer) = val else {
        return Err(QueueError::Config(
            "malformed XAUTOCLAIM response: expected array".into(),
        ));
    };

    if outer.len() < 2 {
        return Err(QueueError::Config(
            "malformed XAUTOCLAIM response: expected at least [next_id, messages]".into(),
        ));
    }

    let next_start_id = value_to_string(&outer[0]).ok_or_else(|| {
        QueueError::Config("malformed XAUTOCLAIM response: next_start_id is not a string".into())
    })?;

    let redis::Value::Array(items) = outer.remove(1) else {
        return Err(QueueError::Config(
            "malformed XAUTOCLAIM response: messages element is not an array".into(),
        ));
    };

    let mut messages = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        // pair[0] is the message ID
        // pair[1] is the array of [field, value, field, value...]
        let redis::Value::Array(pair) = item else {
            return Err(QueueError::Config(format!(
                "malformed XAUTOCLAIM message entry at index {index}: expected array"
            )));
        };
        if pair.len() < 2 {
            return Err(QueueError::Config(format!(
                "malformed XAUTOCLAIM message entry at index {index}: missing fields"
            )));
        }
        let msg_id = pair.first().and_then(value_to_string).ok_or_else(|| {
            QueueError::Config(format!(
                "malformed XAUTOCLAIM message entry at index {index}: missing/non-string id"
            ))
        })?;
        let fields = extract_message_fields(&pair[1]).ok_or_else(|| {
            QueueError::Config(format!(
                "malformed XAUTOCLAIM message entry at index {index}: invalid field payload"
            ))
        })?;
        messages.push(Message {
            id: msg_id,
            stream: stream.to_string(),
            group: group.to_string(),
            run_id: fields.run_id,
            payload: fields.payload,
            stage: fields.stage,
        });
    }

    Ok((next_start_id, messages))
}

/// Parse the response from XAUTOCLAIM.
///
/// Returns (next_start_id, list_of_ids).
pub fn parse_xautoclaim_response(val: redis::Value) -> Result<(String, Vec<String>)> {
    let redis::Value::Array(mut outer) = val else {
        return Err(QueueError::Config(
            "malformed XAUTOCLAIM response: expected array".into(),
        ));
    };

    if outer.len() < 2 {
        return Err(QueueError::Config(
            "malformed XAUTOCLAIM response: expected at least [next_id, entries]".into(),
        ));
    }

    let next_start_id = value_to_string(&outer[0]).ok_or_else(|| {
        QueueError::Config("malformed XAUTOCLAIM response: next_start_id is not a string".into())
    })?;

    let redis::Value::Array(items) = outer.remove(1) else {
        return Err(QueueError::Config(
            "malformed XAUTOCLAIM response: ids element is not an array".into(),
        ));
    };

    let mut ids = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        let redis::Value::Array(pair) = item else {
            return Err(QueueError::Config(format!(
                "malformed XAUTOCLAIM id entry at index {index}: expected array"
            )));
        };
        let id = pair.first().and_then(value_to_string).ok_or_else(|| {
            QueueError::Config(format!(
                "malformed XAUTOCLAIM id entry at index {index}: missing/non-string id"
            ))
        })?;
        ids.push(id);
    }

    Ok((next_start_id, ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // value_to_string tests
    // =========================================================================

    #[test]
    fn test_value_to_string_bulk_string() {
        assert_eq!(
            value_to_string(&redis::Value::BulkString(b"hello".to_vec())),
            Some("hello".to_string())
        );
    }

    #[test]
    fn test_value_to_string_simple_string() {
        assert_eq!(
            value_to_string(&redis::Value::SimpleString("simple".to_string())),
            Some("simple".to_string())
        );
    }

    #[test]
    fn test_value_to_string_okay() {
        assert_eq!(value_to_string(&redis::Value::Okay), Some("OK".to_string()));
    }

    #[test]
    fn test_value_to_string_int() {
        assert_eq!(
            value_to_string(&redis::Value::Int(42)),
            Some("42".to_string())
        );
        // Negative integer
        assert_eq!(
            value_to_string(&redis::Value::Int(-123)),
            Some("-123".to_string())
        );
        // Zero
        assert_eq!(
            value_to_string(&redis::Value::Int(0)),
            Some("0".to_string())
        );
    }

    #[test]
    fn test_value_to_string_nil() {
        assert_eq!(value_to_string(&redis::Value::Nil), None);
    }

    #[test]
    fn test_value_to_string_invalid_utf8() {
        // Invalid UTF-8 sequence should return None
        let invalid_utf8 = vec![0xff, 0xfe, 0x00, 0x01];
        assert_eq!(
            value_to_string(&redis::Value::BulkString(invalid_utf8)),
            None,
            "Invalid UTF-8 should return None"
        );
    }

    #[test]
    fn test_value_to_string_empty_string() {
        assert_eq!(
            value_to_string(&redis::Value::BulkString(vec![])),
            Some(String::new())
        );
        assert_eq!(
            value_to_string(&redis::Value::SimpleString(String::new())),
            Some(String::new())
        );
    }

    #[test]
    fn test_value_to_string_array_returns_none() {
        // Array type should return None (not convertible to string)
        let array = redis::Value::Array(vec![redis::Value::Int(1)]);
        assert_eq!(value_to_string(&array), None, "Array should return None");
    }

    #[test]
    fn test_value_to_string_map_returns_none() {
        // Map type should return None
        let map = redis::Value::Map(vec![]);
        assert_eq!(value_to_string(&map), None, "Map should return None");
    }

    // =========================================================================
    // parse_stream_entries tests
    // =========================================================================

    #[test]
    fn test_parse_stream_entries_single_message() {
        let msg1_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
            redis::Value::BulkString(b"stage".to_vec()),
            redis::Value::BulkString(b"prefetch".to_vec()),
        ]);
        let msg1 =
            redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg1_fields]);

        let stream_msgs = redis::Value::Array(vec![msg1]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "1-0"); // msg_id
        assert_eq!(entries[0].1, "run_1"); // run_id
        assert_eq!(entries[0].2, "{}"); // payload
        assert_eq!(entries[0].3, "prefetch"); // stage
    }

    #[test]
    fn test_parse_stream_entries_nil() {
        let entries = parse_stream_entries(redis::Value::Nil);
        assert!(entries.is_empty(), "Nil response should return empty vec");
    }

    #[test]
    fn test_parse_stream_entries_unexpected_type() {
        // Test with unexpected response structure (triggers warn! path)
        let entries = parse_stream_entries(redis::Value::Int(42));
        assert!(
            entries.is_empty(),
            "Unexpected type should return empty vec"
        );

        let entries = parse_stream_entries(redis::Value::Okay);
        assert!(entries.is_empty(), "Okay response should return empty vec");
    }

    #[test]
    fn test_parse_stream_entries_multiple_streams() {
        // Create two streams with one message each
        let make_stream = |stream_name: &[u8], msg_id: &[u8], run_id: &[u8]| {
            let fields = redis::Value::Array(vec![
                redis::Value::BulkString(b"run_id".to_vec()),
                redis::Value::BulkString(run_id.to_vec()),
                redis::Value::BulkString(b"payload".to_vec()),
                redis::Value::BulkString(b"{}".to_vec()),
            ]);
            let msg = redis::Value::Array(vec![redis::Value::BulkString(msg_id.to_vec()), fields]);
            redis::Value::Array(vec![
                redis::Value::BulkString(stream_name.to_vec()),
                redis::Value::Array(vec![msg]),
            ])
        };

        let response = redis::Value::Array(vec![
            make_stream(b"stream1", b"1-0", b"run_1"),
            make_stream(b"stream2", b"2-0", b"run_2"),
        ]);

        let entries = parse_stream_entries(response);
        assert_eq!(entries.len(), 2, "Should have 2 messages from 2 streams");
        assert_eq!(entries[0].0, "1-0");
        assert_eq!(entries[0].1, "run_1");
        assert_eq!(entries[1].0, "2-0");
        assert_eq!(entries[1].1, "run_2");
    }

    #[test]
    fn test_parse_stream_entries_multiple_messages_per_stream() {
        let make_msg = |msg_id: &[u8], run_id: &[u8]| {
            let fields = redis::Value::Array(vec![
                redis::Value::BulkString(b"run_id".to_vec()),
                redis::Value::BulkString(run_id.to_vec()),
                redis::Value::BulkString(b"payload".to_vec()),
                redis::Value::BulkString(b"{}".to_vec()),
            ]);
            redis::Value::Array(vec![redis::Value::BulkString(msg_id.to_vec()), fields])
        };

        let stream_msgs = redis::Value::Array(vec![
            make_msg(b"1-0", b"run_1"),
            make_msg(b"1-1", b"run_2"),
            make_msg(b"1-2", b"run_3"),
        ]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert_eq!(entries.len(), 3, "Should have 3 messages from 1 stream");
        assert_eq!(entries[0].0, "1-0");
        assert_eq!(entries[1].0, "1-1");
        assert_eq!(entries[2].0, "1-2");
    }

    #[test]
    fn test_parse_stream_entries_empty_array() {
        let response = redis::Value::Array(vec![]);
        let entries = parse_stream_entries(response);
        assert!(entries.is_empty(), "Empty array should return empty vec");
    }

    #[test]
    fn test_parse_stream_entries_malformed_stream_entry_not_array() {
        // Stream entry is not an array
        let response = redis::Value::Array(vec![redis::Value::Int(42)]);
        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Non-array stream entry should be skipped"
        );
    }

    #[test]
    fn test_parse_stream_entries_malformed_stream_entry_wrong_length() {
        // Stream entry has wrong number of parts (not 2)
        let response =
            redis::Value::Array(vec![redis::Value::Array(vec![redis::Value::BulkString(
                b"stream1".to_vec(),
            )])]);
        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Stream entry with wrong length should be skipped"
        );

        // Three parts instead of two
        let response = redis::Value::Array(vec![redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            redis::Value::Array(vec![]),
            redis::Value::BulkString(b"extra".to_vec()),
        ])]);
        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Stream entry with 3 parts should be skipped"
        );
    }

    #[test]
    fn test_parse_stream_entries_messages_not_array() {
        // parts[1] (messages) is not an array
        let response = redis::Value::Array(vec![redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            redis::Value::Int(42), // Should be Array of messages
        ])]);
        let entries = parse_stream_entries(response);
        assert!(entries.is_empty(), "Non-array messages should be skipped");
    }

    #[test]
    fn test_parse_stream_entries_message_not_array() {
        // Individual message is not an array
        let stream_msgs = redis::Value::Array(vec![redis::Value::Int(42)]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(entries.is_empty(), "Non-array message should be skipped");
    }

    #[test]
    fn test_parse_stream_entries_message_wrong_parts() {
        // Message has wrong number of parts (not 2)
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec())]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Message with wrong parts count should be skipped"
        );
    }

    #[test]
    fn test_parse_stream_entries_message_id_not_string() {
        // Message ID cannot be converted to string
        let msg = redis::Value::Array(vec![
            redis::Value::Nil, // Invalid msg_id
            redis::Value::Array(vec![
                redis::Value::BulkString(b"run_id".to_vec()),
                redis::Value::BulkString(b"run_1".to_vec()),
                redis::Value::BulkString(b"payload".to_vec()),
                redis::Value::BulkString(b"{}".to_vec()),
            ]),
        ]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Message with invalid ID should be skipped"
        );
    }

    #[test]
    fn test_parse_stream_entries_fields_not_array() {
        // Message fields is not an array
        let msg = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            redis::Value::Int(42), // Should be Array of fields
        ]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Message with non-array fields should be skipped"
        );
    }

    #[test]
    fn test_parse_stream_entries_missing_run_id() {
        // Missing run_id field
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
        ]);
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg_fields]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Message without run_id should be skipped"
        );
    }

    #[test]
    fn test_parse_stream_entries_missing_payload() {
        // Missing payload field
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
        ]);
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg_fields]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Message without payload should be skipped"
        );
    }

    #[test]
    fn test_parse_stream_entries_unknown_fields_ignored() {
        // Unknown fields should be ignored, but run_id and payload still extracted
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"unknown_field".to_vec()),
            redis::Value::BulkString(b"unknown_value".to_vec()),
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
            redis::Value::BulkString(b"another_unknown".to_vec()),
            redis::Value::BulkString(b"ignored".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
        ]);
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg_fields]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert_eq!(entries.len(), 1, "Should parse message with unknown fields");
        assert_eq!(entries[0].1, "run_1");
        assert_eq!(entries[0].2, "{}");
    }

    #[test]
    fn test_parse_stream_entries_odd_number_of_kvs() {
        // Odd number of key-value pairs is malformed and must be rejected
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
            redis::Value::BulkString(b"orphan_key".to_vec()), // No value
        ]);
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg_fields]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Odd field shape must be treated as parse failure"
        );
    }

    #[test]
    fn test_parse_stream_entries_nil_values_in_fields() {
        // Nil critical values should cause parse failure (no silent defaulting)
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::Nil,
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
        ]);
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg_fields]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert!(
            entries.is_empty(),
            "Nil run_id must be treated as parse failure"
        );
    }

    #[test]
    fn test_parse_stream_entries_empty_run_id_is_rejected() {
        let msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(Vec::new()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
        ]);
        let msg = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg_fields]);
        let stream_msgs = redis::Value::Array(vec![msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);

        let entries = parse_stream_entries(redis::Value::Array(vec![stream_entry]));
        assert!(entries.is_empty(), "Empty run_id must be rejected");
    }

    #[test]
    fn test_parse_stream_entries_mixed_valid_invalid_messages() {
        // One valid, one invalid message - should get only the valid one
        let valid_msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
        ]);
        let valid_msg = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            valid_msg_fields,
        ]);

        // Invalid: missing payload
        let invalid_msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_2".to_vec()),
        ]);
        let invalid_msg = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            invalid_msg_fields,
        ]);

        let stream_msgs = redis::Value::Array(vec![valid_msg, invalid_msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let entries = parse_stream_entries(response);
        assert_eq!(entries.len(), 1, "Should only have valid message");
        assert_eq!(entries[0].0, "1-0");
    }

    #[test]
    fn test_parse_stream_entries_diagnostics_counts_dropped_invalid_messages() {
        let valid_msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
        ]);
        let valid_msg = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            valid_msg_fields,
        ]);

        let invalid_msg_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_2".to_vec()),
        ]);
        let invalid_msg = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            invalid_msg_fields,
        ]);

        let stream_msgs = redis::Value::Array(vec![valid_msg, invalid_msg]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let (_entries, diagnostics) = parse_stream_entries_with_diagnostics(response);
        assert_eq!(
            diagnostics.dropped_messages, 1,
            "invalid messages should be counted in diagnostics"
        );
    }

    #[test]
    fn test_parse_stream_entries_diagnostics_counts_malformed_stream_entries() {
        let malformed_stream_entry = redis::Value::Int(42);
        let response = redis::Value::Array(vec![malformed_stream_entry]);

        let (_entries, diagnostics) = parse_stream_entries_with_diagnostics(response);
        assert_eq!(
            diagnostics.dropped_stream_entries, 1,
            "malformed stream entries should be counted in diagnostics"
        );
    }

    // =========================================================================
    // parse_stream_messages tests
    // =========================================================================

    #[test]
    fn test_parse_stream_messages_with_stage() {
        let msg1_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(r#"{"key":"value"}"#.as_bytes().to_vec()),
            redis::Value::BulkString(b"stage".to_vec()),
            redis::Value::BulkString(b"prefetch".to_vec()),
        ]);
        let msg1 =
            redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg1_fields]);

        let stream_msgs = redis::Value::Array(vec![msg1]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let messages = parse_stream_messages(response, "stream:prefetch", "prefetch:grp");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "1-0");
        assert_eq!(messages[0].stream, "stream:prefetch");
        assert_eq!(messages[0].group, "prefetch:grp");
        assert_eq!(messages[0].run_id, "run_1");
        assert!(messages[0].payload.contains("key"));
        assert_eq!(
            messages[0].stage, "prefetch",
            "Stage should be extracted from Redis response"
        );
    }

    #[test]
    fn test_parse_xclaim_messages_single() {
        let response = redis::Value::Array(vec![redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"run_id".to_vec()),
                redis::Value::BulkString(b"run-1".to_vec()),
                redis::Value::BulkString(b"payload".to_vec()),
                redis::Value::BulkString(b"{}".to_vec()),
                redis::Value::BulkString(b"stage".to_vec()),
                redis::Value::BulkString(b"inference".to_vec()),
            ]),
        ])]);

        let messages = parse_xclaim_messages(response, "stream:test", "grp");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "1-0");
        assert_eq!(messages[0].run_id, "run-1");
        assert_eq!(messages[0].stage, "inference");
    }

    #[test]
    fn test_parse_xclaim_messages_non_array_returns_empty() {
        let messages = parse_xclaim_messages(redis::Value::Int(7), "stream:test", "grp");
        assert!(
            messages.is_empty(),
            "non-array XCLAIM response should return empty parsed messages"
        );
    }

    #[test]
    fn test_parse_xpending_ids() {
        let response = redis::Value::Array(vec![
            redis::Value::Array(vec![
                redis::Value::BulkString(b"1-0".to_vec()),
                redis::Value::BulkString(b"c1".to_vec()),
                redis::Value::Int(10),
                redis::Value::Int(1),
            ]),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"2-0".to_vec()),
                redis::Value::BulkString(b"c2".to_vec()),
                redis::Value::Int(20),
                redis::Value::Int(1),
            ]),
        ]);

        let ids = parse_xpending_ids(response);
        assert_eq!(ids, vec!["1-0".to_string(), "2-0".to_string()]);
    }

    #[test]
    fn test_parse_xpending_ids_non_array_returns_empty() {
        let ids = parse_xpending_ids(redis::Value::Int(42));
        assert!(
            ids.is_empty(),
            "non-array XPENDING response should produce empty id list"
        );
    }

    #[test]
    fn test_parse_xpending_ids_skips_invalid_entries_and_keeps_valid_ids() {
        let response = redis::Value::Array(vec![
            redis::Value::Int(11),                                  // not an array
            redis::Value::Array(vec![]),                            // empty entry
            redis::Value::Array(vec![redis::Value::Array(vec![])]), // unsupported id type
            redis::Value::Array(vec![
                redis::Value::BulkString(b"9-0".to_vec()),
                redis::Value::BulkString(b"c1".to_vec()),
                redis::Value::Int(10),
                redis::Value::Int(1),
            ]),
        ]);

        let ids = parse_xpending_ids(response);
        assert_eq!(
            ids,
            vec!["9-0".to_string()],
            "only valid id entries should be retained"
        );
    }

    #[test]
    fn test_parse_stream_messages_without_stage() {
        let msg1_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run_1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(r#"{"key":"value"}"#.as_bytes().to_vec()),
        ]);
        let msg1 =
            redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec()), msg1_fields]);

        let stream_msgs = redis::Value::Array(vec![msg1]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let messages = parse_stream_messages(response, "stream:test", "test:grp");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "1-0");
        assert_eq!(messages[0].stream, "stream:test");
        assert_eq!(messages[0].group, "test:grp");
        assert_eq!(messages[0].run_id, "run_1");
        assert_eq!(
            messages[0].stage, "",
            "Stage should default to empty string when not present"
        );
    }

    #[test]
    fn test_parse_stream_messages_empty() {
        let messages = parse_stream_messages(redis::Value::Nil, "stream:test", "test:grp");
        assert!(messages.is_empty());
    }

    #[test]
    fn test_parse_stream_messages_multiple() {
        let make_msg = |msg_id: &[u8], run_id: &[u8], stage: &[u8]| {
            let fields = redis::Value::Array(vec![
                redis::Value::BulkString(b"run_id".to_vec()),
                redis::Value::BulkString(run_id.to_vec()),
                redis::Value::BulkString(b"payload".to_vec()),
                redis::Value::BulkString(b"{}".to_vec()),
                redis::Value::BulkString(b"stage".to_vec()),
                redis::Value::BulkString(stage.to_vec()),
            ]);
            redis::Value::Array(vec![redis::Value::BulkString(msg_id.to_vec()), fields])
        };

        let stream_msgs = redis::Value::Array(vec![
            make_msg(b"1-0", b"run_1", b"prefetch"),
            make_msg(b"1-1", b"run_2", b"inference"),
        ]);
        let stream_entry = redis::Value::Array(vec![
            redis::Value::BulkString(b"stream1".to_vec()),
            stream_msgs,
        ]);
        let response = redis::Value::Array(vec![stream_entry]);

        let messages = parse_stream_messages(response, "stream:prefetch", "prefetch:grp");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].id, "1-0");
        assert_eq!(messages[0].stream, "stream:prefetch");
        assert_eq!(messages[0].group, "prefetch:grp");
        assert_eq!(messages[0].stage, "prefetch");
        assert_eq!(messages[1].id, "1-1");
        assert_eq!(messages[1].stream, "stream:prefetch");
        assert_eq!(messages[1].group, "prefetch:grp");
        assert_eq!(messages[1].stage, "inference");
    }

    // =========================================================================
    // parse_xautoclaim_response tests
    // =========================================================================

    #[test]
    fn test_parse_xautoclaim_response_single_id() {
        let id1_pair = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            redis::Value::Nil,
        ]);

        let list_of_ids = redis::Value::Array(vec![id1_pair]);

        let response =
            redis::Value::Array(vec![redis::Value::BulkString(b"2-0".to_vec()), list_of_ids]);

        let (next_id, ids) = parse_xautoclaim_response(response).unwrap();
        assert_eq!(next_id, "2-0");
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], "1-0");
    }

    #[test]
    fn test_parse_xautoclaim_response_multiple_ids() {
        let id1_pair = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            redis::Value::Nil,
        ]);
        let id2_pair = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-1".to_vec()),
            redis::Value::Nil,
        ]);
        let id3_pair = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-2".to_vec()),
            redis::Value::Nil,
        ]);

        let list_of_ids = redis::Value::Array(vec![id1_pair, id2_pair, id3_pair]);

        let response =
            redis::Value::Array(vec![redis::Value::BulkString(b"2-0".to_vec()), list_of_ids]);

        let (next_id, ids) = parse_xautoclaim_response(response).unwrap();
        assert_eq!(next_id, "2-0");
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], "1-0");
        assert_eq!(ids[1], "1-1");
        assert_eq!(ids[2], "1-2");
    }

    #[test]
    fn test_parse_xautoclaim_response_nil() {
        assert!(
            parse_xautoclaim_response(redis::Value::Nil).is_err(),
            "nil outer response is malformed for XAUTOCLAIM parsing"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_empty_array() {
        // Array with less than 2 elements
        let response = redis::Value::Array(vec![]);
        assert!(
            parse_xautoclaim_response(response).is_err(),
            "short outer response must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_single_element_array() {
        // Array with only 1 element (too short)
        let response = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec())]);
        assert!(
            parse_xautoclaim_response(response).is_err(),
            "single-element outer response must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_ids_not_array() {
        // Second element is not an array
        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            redis::Value::Int(42), // Should be Array
        ]);

        assert!(
            parse_xautoclaim_response(response).is_err(),
            "non-array id section must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_id_item_not_array() {
        // ID items are not arrays
        let list_of_ids = redis::Value::Array(vec![
            redis::Value::Int(42), // Should be Array pair
        ]);

        let response =
            redis::Value::Array(vec![redis::Value::BulkString(b"2-0".to_vec()), list_of_ids]);

        assert!(
            parse_xautoclaim_response(response).is_err(),
            "non-array id entries must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_id_item_empty_array() {
        // ID item is empty array (no first element)
        let list_of_ids = redis::Value::Array(vec![redis::Value::Array(vec![])]);

        let response =
            redis::Value::Array(vec![redis::Value::BulkString(b"2-0".to_vec()), list_of_ids]);

        assert!(
            parse_xautoclaim_response(response).is_err(),
            "empty id entries must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_id_not_string() {
        // ID cannot be converted to string
        let id_pair = redis::Value::Array(vec![
            redis::Value::Nil, // Not convertible to string
            redis::Value::Nil,
        ]);
        let list_of_ids = redis::Value::Array(vec![id_pair]);

        let response =
            redis::Value::Array(vec![redis::Value::BulkString(b"2-0".to_vec()), list_of_ids]);

        assert!(
            parse_xautoclaim_response(response).is_err(),
            "non-string id entries must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_next_id_nil() {
        // next_start_id is Nil
        let id_pair = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            redis::Value::Nil,
        ]);
        let list_of_ids = redis::Value::Array(vec![id_pair]);

        let response = redis::Value::Array(vec![redis::Value::Nil, list_of_ids]);

        assert!(
            parse_xautoclaim_response(response).is_err(),
            "non-string next_start_id must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_mixed_valid_invalid() {
        // Mix of valid and invalid ID items
        let valid_id = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            redis::Value::Nil,
        ]);
        let invalid_id_nil = redis::Value::Array(vec![
            redis::Value::Nil, // Invalid
            redis::Value::Nil,
        ]);
        let valid_id2 = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-2".to_vec()),
            redis::Value::Nil,
        ]);
        let invalid_not_array = redis::Value::Int(42);

        let list_of_ids =
            redis::Value::Array(vec![valid_id, invalid_id_nil, valid_id2, invalid_not_array]);

        let response =
            redis::Value::Array(vec![redis::Value::BulkString(b"2-0".to_vec()), list_of_ids]);

        assert!(
            parse_xautoclaim_response(response).is_err(),
            "mixed valid/invalid id entries must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_extra_elements() {
        // Response has more than 2 elements (should still work)
        let id_pair = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-0".to_vec()),
            redis::Value::Nil,
        ]);
        let list_of_ids = redis::Value::Array(vec![id_pair]);

        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            list_of_ids,
            redis::Value::BulkString(b"extra".to_vec()), // Extra element (ignored)
        ]);

        let (next_id, ids) = parse_xautoclaim_response(response).unwrap();
        assert_eq!(next_id, "2-0");
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], "1-0");
    }

    // =========================================================================
    // Additional Edge Cases for value_to_string
    // =========================================================================

    #[test]
    fn test_value_to_string_unicode() {
        let unicode_bytes = "日本語".as_bytes().to_vec();
        assert_eq!(
            value_to_string(&redis::Value::BulkString(unicode_bytes)),
            Some("日本語".to_string())
        );
    }

    #[test]
    fn test_value_to_string_large_int() {
        assert_eq!(
            value_to_string(&redis::Value::Int(i64::MAX)),
            Some(i64::MAX.to_string())
        );
    }

    #[test]
    fn test_value_to_string_simple_string_with_spaces() {
        assert_eq!(
            value_to_string(&redis::Value::SimpleString("hello world".to_string())),
            Some("hello world".to_string())
        );
    }

    #[test]
    fn test_value_to_string_double_returns_none() {
        // Double type should return None
        // Using an arbitrary value that's not close to any math constant
        let double = redis::Value::Double(1.234);
        assert_eq!(value_to_string(&double), None, "Double should return None");
    }

    #[test]
    fn test_value_to_string_boolean_returns_none() {
        // Boolean type should return None
        assert_eq!(
            value_to_string(&redis::Value::Boolean(true)),
            None,
            "Boolean should return None"
        );
        assert_eq!(
            value_to_string(&redis::Value::Boolean(false)),
            None,
            "Boolean should return None"
        );
    }

    // =========================================================================
    // parse_xautoclaim_messages tests
    // =========================================================================

    /// Helper to create a Redis message entry with full field data
    fn make_message_entry(id: &str, run_id: &str, payload: &str, stage: &str) -> redis::Value {
        redis::Value::Array(vec![
            redis::Value::BulkString(id.as_bytes().to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"run_id".to_vec()),
                redis::Value::BulkString(run_id.as_bytes().to_vec()),
                redis::Value::BulkString(b"payload".to_vec()),
                redis::Value::BulkString(payload.as_bytes().to_vec()),
                redis::Value::BulkString(b"stage".to_vec()),
                redis::Value::BulkString(stage.as_bytes().to_vec()),
            ]),
        ])
    }

    #[test]
    fn test_parse_xautoclaim_messages_single_message() {
        let msg_entry = make_message_entry("1-0", "run-123", r#"{"key":"val"}"#, "prefetch");
        let list_of_msgs = redis::Value::Array(vec![msg_entry]);

        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            list_of_msgs,
        ]);

        let (next_id, messages) =
            parse_xautoclaim_messages(response, "stream:test", "test:grp").unwrap();
        assert_eq!(next_id, "2-0");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "1-0");
        assert_eq!(messages[0].stream, "stream:test");
        assert_eq!(messages[0].group, "test:grp");
        assert_eq!(messages[0].run_id, "run-123");
        assert_eq!(messages[0].payload, r#"{"key":"val"}"#);
        assert_eq!(messages[0].stage, "prefetch");
    }

    #[test]
    fn test_parse_xautoclaim_messages_multiple_messages() {
        let msg1 = make_message_entry("1-0", "run-1", "{}", "stage1");
        let msg2 = make_message_entry("1-1", "run-2", r#"{"x":1}"#, "stage2");
        let msg3 = make_message_entry("1-2", "run-3", r#"{"y":2}"#, "stage3");
        let list_of_msgs = redis::Value::Array(vec![msg1, msg2, msg3]);

        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            list_of_msgs,
        ]);

        let (next_id, messages) =
            parse_xautoclaim_messages(response, "stream:gpu", "gpu:grp").unwrap();
        assert_eq!(next_id, "2-0");
        assert_eq!(messages.len(), 3);

        assert_eq!(messages[0].id, "1-0");
        assert_eq!(messages[0].run_id, "run-1");
        assert_eq!(messages[0].stream, "stream:gpu");

        assert_eq!(messages[1].id, "1-1");
        assert_eq!(messages[1].run_id, "run-2");

        assert_eq!(messages[2].id, "1-2");
        assert_eq!(messages[2].run_id, "run-3");
    }

    #[test]
    fn test_parse_xautoclaim_messages_empty() {
        assert!(
            parse_xautoclaim_messages(redis::Value::Nil, "stream:test", "grp").is_err(),
            "nil outer response is malformed for XAUTOCLAIM message parsing"
        );
    }

    #[test]
    fn test_parse_xautoclaim_messages_empty_array() {
        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"0-0".to_vec()),
            redis::Value::Array(vec![]),
        ]);

        let (next_id, messages) =
            parse_xautoclaim_messages(response, "stream:test", "grp").unwrap();
        assert_eq!(next_id, "0-0");
        assert!(messages.is_empty());
    }

    #[test]
    fn test_parse_xautoclaim_messages_rejects_invalid_entries() {
        // Mix of valid and invalid entries must fail closed.
        let valid_msg = make_message_entry("1-0", "run-1", "{}", "stage");
        let invalid_no_fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"1-1".to_vec()),
            // Missing fields array
        ]);
        let valid_msg2 = make_message_entry("1-2", "run-2", r#"{"a":1}"#, "stage2");

        let list_of_msgs = redis::Value::Array(vec![valid_msg, invalid_no_fields, valid_msg2]);

        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            list_of_msgs,
        ]);

        assert!(
            parse_xautoclaim_messages(response, "stream:test", "grp").is_err(),
            "invalid message entries must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_messages_with_extra_deleted_ids() {
        // XAUTOCLAIM may return a third element with deleted IDs
        let msg = make_message_entry("1-0", "run-1", "{}", "stage");
        let list_of_msgs = redis::Value::Array(vec![msg]);
        let deleted_ids = redis::Value::Array(vec![
            redis::Value::BulkString(b"0-99".to_vec()), // Deleted message
        ]);

        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            list_of_msgs,
            deleted_ids, // Extra element should be ignored
        ]);

        let (next_id, messages) =
            parse_xautoclaim_messages(response, "stream:test", "grp").unwrap();
        assert_eq!(next_id, "2-0");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "1-0");
    }

    #[test]
    fn test_parse_xautoclaim_messages_preserves_stream_and_group() {
        let msg = make_message_entry("1-0", "run-1", "{}", "stage");
        let list_of_msgs = redis::Value::Array(vec![msg]);

        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            list_of_msgs,
        ]);

        let (_, messages) =
            parse_xautoclaim_messages(response, "physicsnemo:inference", "inference:workers")
                .unwrap();

        assert_eq!(messages[0].stream, "physicsnemo:inference");
        assert_eq!(messages[0].group, "inference:workers");
    }

    #[test]
    fn test_parse_xautoclaim_messages_rejects_malformed_outer_shape() {
        let result = parse_xautoclaim_messages(redis::Value::Int(42), "stream:test", "test:grp");
        assert!(
            result.is_err(),
            "malformed outer response must fail closed instead of returning empty-success"
        );
    }

    #[test]
    fn test_parse_xautoclaim_messages_rejects_short_outer_array() {
        let response = redis::Value::Array(vec![redis::Value::BulkString(b"2-0".to_vec())]);
        let result = parse_xautoclaim_messages(response, "stream:test", "test:grp");
        assert!(
            result.is_err(),
            "outer response shorter than [next_id, messages] must fail closed"
        );
    }

    #[test]
    fn test_parse_xautoclaim_messages_rejects_non_array_messages_element() {
        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            redis::Value::Int(3),
        ]);
        let result = parse_xautoclaim_messages(response, "stream:test", "test:grp");
        assert!(
            result.is_err(),
            "messages element must be an array for XAUTOCLAIM message parsing"
        );
    }

    #[test]
    fn test_parse_xautoclaim_messages_rejects_malformed_message_entry_shape() {
        let malformed_entry = redis::Value::Array(vec![redis::Value::BulkString(b"1-0".to_vec())]);
        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            redis::Value::Array(vec![malformed_entry]),
        ]);

        let result = parse_xautoclaim_messages(response, "stream:test", "test:grp");
        assert!(
            result.is_err(),
            "malformed message entry must fail closed instead of being silently skipped"
        );
    }

    #[test]
    fn test_parse_xautoclaim_messages_rejects_non_array_message_item() {
        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            redis::Value::Array(vec![redis::Value::Int(123)]),
        ]);
        let result = parse_xautoclaim_messages(response, "stream:test", "test:grp");
        assert!(result.is_err(), "non-array message items must fail closed");
    }

    #[test]
    fn test_extract_message_fields_rejects_whitespace_run_id_or_payload() {
        let run_id_blank = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"   ".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
        ]);
        assert!(
            extract_message_fields(&run_id_blank).is_none(),
            "blank run_id should be rejected"
        );

        let payload_blank = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run-1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"   ".to_vec()),
        ]);
        assert!(
            extract_message_fields(&payload_blank).is_none(),
            "blank payload should be rejected"
        );
    }

    #[test]
    fn test_extract_message_fields_rejects_blank_explicit_stage() {
        let fields = redis::Value::Array(vec![
            redis::Value::BulkString(b"run_id".to_vec()),
            redis::Value::BulkString(b"run-1".to_vec()),
            redis::Value::BulkString(b"payload".to_vec()),
            redis::Value::BulkString(b"{}".to_vec()),
            redis::Value::BulkString(b"stage".to_vec()),
            redis::Value::BulkString(b"   ".to_vec()),
        ]);
        assert!(
            extract_message_fields(&fields).is_none(),
            "blank explicit stage should be rejected instead of defaulted"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_rejects_malformed_outer_shape() {
        let result = parse_xautoclaim_response(redis::Value::Nil);
        assert!(
            result.is_err(),
            "malformed outer response must fail closed instead of returning empty-success"
        );
    }

    #[test]
    fn test_parse_xautoclaim_response_rejects_malformed_id_entry_shape() {
        let response = redis::Value::Array(vec![
            redis::Value::BulkString(b"2-0".to_vec()),
            redis::Value::Array(vec![redis::Value::Int(7)]),
        ]);

        let result = parse_xautoclaim_response(response);
        assert!(
            result.is_err(),
            "malformed id entry must fail closed instead of being silently skipped"
        );
    }
}
