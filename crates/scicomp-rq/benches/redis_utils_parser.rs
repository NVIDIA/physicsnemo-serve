/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use redis::Value;
use scicomp_rq::redis_utils::parse_xautoclaim_messages;
use scicomp_rq::{parse_stream_entries, parse_stream_messages};
use std::hint::black_box;

fn bulk_string(value: &str) -> Value {
    Value::BulkString(value.as_bytes().to_vec())
}

fn make_message_entry(index: usize) -> Value {
    let id = format!("{}-0", index + 1);
    let run_id = format!("run-{index:06}");
    let payload = format!(r#"{{"run_idx":{index},"step":42,"model":"pangu"}}"#);

    Value::Array(vec![
        bulk_string(&id),
        Value::Array(vec![
            bulk_string("run_id"),
            bulk_string(&run_id),
            bulk_string("payload"),
            bulk_string(&payload),
            bulk_string("stage"),
            bulk_string("prefetch"),
        ]),
    ])
}

fn make_xreadgroup_response(stream_key: &str, message_count: usize) -> Value {
    let messages = (0..message_count)
        .map(make_message_entry)
        .collect::<Vec<_>>();
    Value::Array(vec![Value::Array(vec![
        bulk_string(stream_key),
        Value::Array(messages),
    ])])
}

fn make_xautoclaim_response(message_count: usize) -> Value {
    let messages = (0..message_count)
        .map(make_message_entry)
        .collect::<Vec<_>>();
    Value::Array(vec![bulk_string("999999-0"), Value::Array(messages)])
}

fn bench_parse_stream_entries(c: &mut Criterion) {
    let mut group = c.benchmark_group("redis_utils/parse_stream_entries");
    for &message_count in &[16usize, 128usize, 512usize] {
        let fixture = make_xreadgroup_response("stream:prefetch", message_count);
        group.bench_with_input(
            BenchmarkId::from_parameter(message_count),
            &message_count,
            |b, _| {
                b.iter_batched(
                    || fixture.clone(),
                    |response| {
                        let entries = parse_stream_entries(response);
                        black_box(entries.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_parse_stream_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("redis_utils/parse_stream_messages");
    for &message_count in &[16usize, 128usize, 512usize] {
        let fixture = make_xreadgroup_response("stream:prefetch", message_count);
        group.bench_with_input(
            BenchmarkId::from_parameter(message_count),
            &message_count,
            |b, _| {
                b.iter_batched(
                    || fixture.clone(),
                    |response| {
                        let messages =
                            parse_stream_messages(response, "stream:prefetch", "prefetch:grp");
                        black_box(messages.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_parse_xautoclaim_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("redis_utils/parse_xautoclaim_messages");
    for &message_count in &[16usize, 128usize, 512usize] {
        let fixture = make_xautoclaim_response(message_count);
        group.bench_with_input(
            BenchmarkId::from_parameter(message_count),
            &message_count,
            |b, _| {
                b.iter_batched(
                    || fixture.clone(),
                    |response| {
                        let parsed =
                            parse_xautoclaim_messages(response, "stream:prefetch", "prefetch:grp")
                                .expect("benchmark fixture should parse successfully");
                        black_box(parsed.1.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_parse_stream_entries,
    bench_parse_stream_messages,
    bench_parse_xautoclaim_messages
);
criterion_main!(benches);
