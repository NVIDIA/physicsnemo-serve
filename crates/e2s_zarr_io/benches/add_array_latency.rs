/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use e2s_zarr_io::{
    ArrayRegistration, CoordMap, CoordValues, SyncZarrBackendConfig, ZarrIoBackend,
    try_build_sync_zarr_backend,
};

const BENCH_ARRAY_NAMES: [&str; 26] = [
    "u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", "tp", "cape", "cin", "z", "q",
    "t", "u", "v", "w", "r", "d", "vo", "pv", "clwc", "ciwc", "cc", "o3", "ch4",
];

fn unique_dataset_root() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock moved backwards")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "e2s_add_array_bench_{}_{}.zarr",
        std::process::id(),
        nanos
    ))
}

fn sequential_i64(len: usize) -> Vec<i64> {
    (0..len)
        .map(|v| i64::try_from(v).expect("bench axis index fits i64"))
        .collect()
}

fn realistic_registration() -> ArrayRegistration {
    let mut coords = CoordMap::new();
    let _ = coords.insert("time".to_string(), CoordValues::I64(sequential_i64(1)));
    let _ = coords.insert(
        "lead_time".to_string(),
        CoordValues::I64(sequential_i64(21)),
    );
    let _ = coords.insert("lat".to_string(), CoordValues::I64(sequential_i64(721)));
    let _ = coords.insert("lon".to_string(), CoordValues::I64(sequential_i64(1440)));
    ArrayRegistration {
        coords,
        array_names: BENCH_ARRAY_NAMES.iter().map(|s| s.to_string()).collect(),
        array_dtypes: Vec::new(),
    }
}

fn bench_add_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend/add_array");
    let registration = realistic_registration();

    // Use iter_custom so we can time only add_array() — cleanup (joining the
    // background metadata thread + dir removal) happens outside the timed region.
    group.bench_function(BenchmarkId::new("localfs", "26array_4coord"), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dataset_root = unique_dataset_root();
                // SyncZarrBackendConfig is #[non_exhaustive]; use default() + field assignment.
                let mut config = SyncZarrBackendConfig::default();
                config.dataset_root = Some(dataset_root.clone());
                let backend = try_build_sync_zarr_backend(config)
                    .expect("backend construction should succeed");

                // ── timed region: only add_array() ──────────────────────────
                let started = Instant::now();
                backend
                    .add_array(black_box(registration.clone()))
                    .expect("add_array should succeed");
                total += started.elapsed();
                // ────────────────────────────────────────────────────────────

                // Cleanup outside timing: drop joins background thread + runs close.
                drop(backend);
                let _ = std::fs::remove_dir_all(&dataset_root);
            }
            total
        });
    });

    group.finish();
}

criterion_group!(benches, bench_add_array);
criterion_main!(benches);
