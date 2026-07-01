/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use e2s_zarr_io::runtime::cuda_runtime::shared_cuda_runtime_api;
use e2s_zarr_io::{BufferPool, BufferPoolConfig, MemoryBufferPool, SizeOverride};

const POOLED_BUFFER_BYTES: usize = 4 * 1024;
const REQUIRED_BYTES: usize = 2 * 1024;

#[derive(Clone, Copy, Debug)]
enum CudaBenchMode {
    Disabled,
    Auto,
}

impl CudaBenchMode {
    fn label(self) -> &'static str {
        match self {
            Self::Disabled => "cuda_disabled",
            Self::Auto => "cuda_auto",
        }
    }

    fn enable_cuda_registration(self) -> bool {
        matches!(self, Self::Auto)
    }
}

const CUDA_BENCH_MODES: [CudaBenchMode; 2] = [CudaBenchMode::Disabled, CudaBenchMode::Auto];

fn hot_only_pool_config(pin_pooled_slabs: bool, cuda_mode: CudaBenchMode) -> BufferPoolConfig {
    BufferPoolConfig {
        max_pool_buffers: 1,
        max_pool_bytes: POOLED_BUFFER_BYTES * 2,
        pool_buffer_bytes: SizeOverride::Fixed(POOLED_BUFFER_BYTES),
        hot_slab_buffers: SizeOverride::Fixed(1),
        warm_slab_buffers: SizeOverride::Auto,
        pin_pooled_slabs,
        cuda_register_pool_if_available: cuda_mode.enable_cuda_registration(),
        cuda_register_each_slab_once: cuda_mode.enable_cuda_registration(),
        ..BufferPoolConfig::default()
    }
}

fn pinning_available_for_hot_slab() -> bool {
    let pool = MemoryBufferPool::new(hot_only_pool_config(true, CudaBenchMode::Disabled));
    let initialized = pool.acquire(REQUIRED_BYTES).is_ok();
    let _ = pool.shutdown();
    initialized
}

fn bench_pool_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool/new");
    for cuda_mode in CUDA_BENCH_MODES {
        let config = hot_only_pool_config(false, cuda_mode);
        group.bench_function(BenchmarkId::from_parameter(cuda_mode.label()), |b| {
            b.iter(|| {
                let pool = MemoryBufferPool::new(config.clone());
                black_box(pool);
            });
        });
    }
    group.finish();
}

fn bench_cuda_runtime_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool/cuda_probe");
    let cuda_available = shared_cuda_runtime_api().available();
    group.bench_function("runtime_available", |b| {
        b.iter(|| {
            black_box(cuda_available);
        });
    });
    group.finish();
}

fn bench_first_write_pool_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool/first_write_initialize");
    let mut variants: Vec<(&str, bool)> = vec![("pin_disabled", false)];
    if pinning_available_for_hot_slab() {
        variants.push(("pin_enabled", true));
    }

    for cuda_mode in CUDA_BENCH_MODES {
        for (label, pin_enabled) in &variants {
            let config = hot_only_pool_config(*pin_enabled, cuda_mode);
            group.bench_function(
                BenchmarkId::new(format!("lazy_acquire/{}", cuda_mode.label()), *label),
                |b| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let pool = MemoryBufferPool::new(config.clone());
                            let started = Instant::now();
                            let lease = pool
                                .acquire(REQUIRED_BYTES)
                                .expect("first acquire should initialize and return a lease");
                            total += started.elapsed();
                            pool.release(lease);
                            pool.shutdown()
                                .expect("hot-only pool shutdown should succeed");
                        }
                        total
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_first_write_lazy_acquire_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool/first_write_lazy_acquire_release");
    let mut variants: Vec<(&str, bool)> = vec![("pin_disabled", false)];
    if pinning_available_for_hot_slab() {
        variants.push(("pin_enabled", true));
    }

    for cuda_mode in CUDA_BENCH_MODES {
        for (label, pin_enabled) in &variants {
            let config = hot_only_pool_config(*pin_enabled, cuda_mode);
            group.bench_function(
                BenchmarkId::new(format!("acquire_release/{}", cuda_mode.label()), *label),
                |b| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let pool = MemoryBufferPool::new(config.clone());
                            let started = Instant::now();
                            let lease = pool
                                .acquire(REQUIRED_BYTES)
                                .expect("first acquire should initialize and return a lease");
                            pool.release(lease);
                            total += started.elapsed();
                            pool.shutdown()
                                .expect("hot-only pool shutdown should succeed");
                        }
                        total
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_steady_state_acquire_release_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool/steady_state_acquire_release");
    let mut variants: Vec<(&str, bool)> = vec![("pin_disabled", false)];
    if pinning_available_for_hot_slab() {
        variants.push(("pin_enabled", true));
    }

    for cuda_mode in CUDA_BENCH_MODES {
        for (label, pin_enabled) in &variants {
            let config = hot_only_pool_config(*pin_enabled, cuda_mode);
            let pool = MemoryBufferPool::new(config);
            let init_lease = pool
                .acquire(REQUIRED_BYTES)
                .expect("first acquire should initialize pool");
            pool.release(init_lease);
            group.bench_function(
                BenchmarkId::new(format!("acquire_release/{}", cuda_mode.label()), *label),
                |b| {
                    b.iter(|| {
                        let lease = pool
                            .acquire(REQUIRED_BYTES)
                            .expect("steady-state acquire should succeed");
                        pool.release(black_box(lease));
                    });
                },
            );
            pool.shutdown().expect("pool shutdown should succeed");
        }
    }

    group.finish();
}

criterion_group!(
    buffer_pool_benches,
    bench_cuda_runtime_probe,
    bench_pool_creation,
    bench_first_write_pool_initialization,
    bench_first_write_lazy_acquire_release,
    bench_steady_state_acquire_release_latency
);
criterion_main!(buffer_pool_benches);
