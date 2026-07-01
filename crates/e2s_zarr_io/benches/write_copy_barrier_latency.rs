/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use e2s_zarr_io::core::errors::SyncWriteError;
use e2s_zarr_io::core::types::TupleChunkKey;
use e2s_zarr_io::runtime::coordinator::{TestWriteCoordinator, TestWriteCoordinatorComponents};
use e2s_zarr_io::runtime::copy_engine::DefaultCopyEngine;
use e2s_zarr_io::runtime::cuda_runtime::shared_cuda_runtime_api;
use e2s_zarr_io::zarr::metadata::NoopMetadataConsolidator;
use e2s_zarr_io::{
    BufferPool, BufferPoolConfig, ChunkId, ChunkKeyRegistry, ChunkPlanner, ChunkWriter, CoordMap,
    CoordValues, CopyEngine, DefaultZarrLayoutAdapter, InferenceWriteRequest, InputArray,
    InputArraySource, LocalFsChunkWriter, MemoryBufferPool, MetadataConsolidator,
    MixedRadixChunkPlanner, RayonWorkScheduler, SizeOverride, WorkScheduler, WriteExecutionConfig,
    ZarrLayoutAdapter,
};

const BENCH_WORKERS: usize = 8;
const BENCH_QUEUE_CAPACITY: usize = BENCH_WORKERS * 2;
const BENCH_CLOSE_TIMEOUT_SECONDS: f64 = 0.05;
const REALISTIC_ARRAY_COUNT: usize = 26;
const REALISTIC_BYTES_PER_ARRAY: usize = 720 * 1440 * 4;
const REALISTIC_REGISTERED_LEAD_TIMES: usize = 21;
const REALISTIC_STEP_TIME_INDEX: i64 = 0;
const REALISTIC_STEP_LEAD_TIME_INDEX: i64 = 10;
const LOCALFS_MAX_POOL_BUFFERS: usize = 64;
const LOCALFS_HOT_SLAB_BUFFERS: usize = 32;
const LOCALFS_WARM_SLAB_BUFFERS: usize = 32;

#[derive(Clone, Copy, Debug)]
struct BenchCase {
    label: &'static str,
    task_count: usize,
    bytes_per_task: usize,
}

struct RealisticBenchFixture {
    request: InferenceWriteRequest,
    array_ids: Vec<u32>,
    registered_coords: CoordMap,
}

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

const BENCH_CASES: [BenchCase; 3] = [
    BenchCase {
        label: "1task_4KiB",
        task_count: 1,
        bytes_per_task: 4 * 1024,
    },
    BenchCase {
        label: "8task_4KiB",
        task_count: 8,
        bytes_per_task: 4 * 1024,
    },
    BenchCase {
        label: "8task_1MiB",
        task_count: 8,
        bytes_per_task: 1024 * 1024,
    },
];

#[derive(Debug, Default)]
struct NoopChunkKeyRegistry;

impl ChunkKeyRegistry for NoopChunkKeyRegistry {
    fn reserve_many_ids(&self, _chunk_ids: &[ChunkId]) -> Result<(), SyncWriteError> {
        Ok(())
    }

    fn mark_committed_id(&self, _chunk_id: &ChunkId) {}

    fn release_failed_id(&self, _chunk_id: &ChunkId) {}
}

#[derive(Debug, Default)]
struct BenchNoopChunkWriter;

impl ChunkWriter for BenchNoopChunkWriter {
    fn write_chunk_by_id(
        &self,
        _array_name: &str,
        _chunk_id: &ChunkId,
        _bytes: &[u8],
    ) -> Result<(), SyncWriteError> {
        Ok(())
    }

    fn write_chunk_by_tuple_key(
        &self,
        _array_name: &str,
        _tuple_key: &TupleChunkKey,
        _bytes: &[u8],
    ) -> Result<(), SyncWriteError> {
        Ok(())
    }
}

fn sequential_i64(len: usize) -> Vec<i64> {
    (0..len)
        .map(|v| i64::try_from(v).expect("bench axis index should fit i64"))
        .collect()
}

fn build_registered_coords(case: BenchCase) -> CoordMap {
    let mut coords = CoordMap::new();
    let _ = coords.insert(
        "time".to_string(),
        CoordValues::I64(sequential_i64(case.task_count)),
    );
    coords
}

fn build_request(case: BenchCase) -> InferenceWriteRequest {
    let total_nbytes = case.task_count.saturating_mul(case.bytes_per_task);
    let mut step_coords = CoordMap::new();
    let _ = step_coords.insert(
        "time".to_string(),
        CoordValues::I64(sequential_i64(case.task_count)),
    );
    InferenceWriteRequest {
        coords: step_coords,
        array_names: vec!["temperature".to_string()],
        arrays: vec![InputArray {
            nbytes: total_nbytes,
            source: InputArraySource::HostBytes(vec![1_u8; total_nbytes].into()),
        }],
    }
}

fn build_coordinator_with_components(
    pool_buffer_bytes: usize,
    cuda_mode: CudaBenchMode,
    layout_adapter: Arc<dyn ZarrLayoutAdapter>,
    chunk_writer: Arc<dyn ChunkWriter>,
) -> TestWriteCoordinator {
    let planner: Arc<dyn ChunkPlanner> =
        Arc::new(MixedRadixChunkPlanner::new(WriteExecutionConfig::default()));
    let chunk_registry: Arc<dyn ChunkKeyRegistry> = Arc::new(NoopChunkKeyRegistry);
    let scheduler: Arc<dyn WorkScheduler> = Arc::new(RayonWorkScheduler::new());
    let pool_config = BufferPoolConfig {
        max_pool_buffers: BENCH_WORKERS,
        max_pool_bytes: pool_buffer_bytes
            .saturating_mul(BENCH_WORKERS)
            .saturating_mul(2),
        pool_buffer_bytes: SizeOverride::Fixed(pool_buffer_bytes),
        hot_slab_buffers: SizeOverride::Fixed(BENCH_WORKERS),
        warm_slab_buffers: SizeOverride::Auto,
        pin_pooled_slabs: false,
        cuda_register_pool_if_available: cuda_mode.enable_cuda_registration(),
        cuda_register_each_slab_once: cuda_mode.enable_cuda_registration(),
        ..BufferPoolConfig::default()
    };
    let buffer_pool: Arc<dyn BufferPool> = Arc::new(MemoryBufferPool::new(pool_config));
    let copy_engine: Arc<dyn CopyEngine> = Arc::new(DefaultCopyEngine::new());
    let metadata_consolidator: Arc<dyn MetadataConsolidator> =
        Arc::new(NoopMetadataConsolidator::new());

    TestWriteCoordinator::new(TestWriteCoordinatorComponents {
        planner,
        chunk_registry,
        scheduler,
        buffer_pool,
        copy_engine,
        chunk_writer,
        metadata_consolidator,
        layout_adapter,
        parallel_coord_names: vec!["time".to_string()],
        queue_capacity: BENCH_QUEUE_CAPACITY,
    })
}

fn build_localfs_pool_config(
    pool_buffer_bytes: usize,
    cuda_mode: CudaBenchMode,
) -> BufferPoolConfig {
    // Keep transient growth bounded for localfs benches so slow flush paths do not
    // balloon resident memory and OOM the benchmark runner.
    BufferPoolConfig {
        max_pool_buffers: LOCALFS_MAX_POOL_BUFFERS,
        max_pool_bytes: pool_buffer_bytes
            .saturating_mul(LOCALFS_MAX_POOL_BUFFERS)
            .saturating_mul(2),
        pool_buffer_bytes: SizeOverride::Fixed(pool_buffer_bytes),
        hot_slab_buffers: SizeOverride::Fixed(LOCALFS_HOT_SLAB_BUFFERS),
        warm_slab_buffers: SizeOverride::Fixed(LOCALFS_WARM_SLAB_BUFFERS),
        pin_pooled_slabs: false,
        cuda_register_pool_if_available: cuda_mode.enable_cuda_registration(),
        cuda_register_each_slab_once: cuda_mode.enable_cuda_registration(),
        max_inflight_transient_bytes: Some(
            pool_buffer_bytes.saturating_mul(LOCALFS_HOT_SLAB_BUFFERS + LOCALFS_WARM_SLAB_BUFFERS),
        ),
        ..BufferPoolConfig::default()
    }
}

fn build_coordinator_with_pool_buffer_bytes(
    pool_buffer_bytes: usize,
    cuda_mode: CudaBenchMode,
) -> TestWriteCoordinator {
    let layout_adapter: Arc<dyn ZarrLayoutAdapter> =
        Arc::new(DefaultZarrLayoutAdapter::v2_default());
    let chunk_writer: Arc<dyn ChunkWriter> = Arc::new(BenchNoopChunkWriter);
    build_coordinator_with_components(pool_buffer_bytes, cuda_mode, layout_adapter, chunk_writer)
}

fn unique_dataset_root(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock moved backwards")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{}_{}.zarr", std::process::id(), nanos))
}

fn remove_dataset_root(path: &PathBuf) {
    let _ = std::fs::remove_dir_all(path);
}

fn build_localfs_coordinator(
    pool_buffer_bytes: usize,
    cuda_mode: CudaBenchMode,
    dataset_root: PathBuf,
) -> TestWriteCoordinator {
    let pool_config = build_localfs_pool_config(pool_buffer_bytes, cuda_mode);
    let planner: Arc<dyn ChunkPlanner> =
        Arc::new(MixedRadixChunkPlanner::new(WriteExecutionConfig::default()));
    let chunk_registry: Arc<dyn ChunkKeyRegistry> = Arc::new(NoopChunkKeyRegistry);
    let scheduler: Arc<dyn WorkScheduler> = Arc::new(RayonWorkScheduler::new());
    let buffer_pool: Arc<dyn BufferPool> = Arc::new(MemoryBufferPool::new(pool_config));
    let copy_engine: Arc<dyn CopyEngine> = Arc::new(DefaultCopyEngine::new());
    let metadata_consolidator: Arc<dyn MetadataConsolidator> =
        Arc::new(NoopMetadataConsolidator::new());
    let layout_adapter: Arc<dyn ZarrLayoutAdapter> =
        Arc::new(DefaultZarrLayoutAdapter::v2_default());
    let chunk_writer: Arc<dyn ChunkWriter> = Arc::new(LocalFsChunkWriter::new(
        dataset_root,
        Arc::clone(&layout_adapter),
    ));

    TestWriteCoordinator::new(TestWriteCoordinatorComponents {
        planner,
        chunk_registry,
        scheduler,
        buffer_pool,
        copy_engine,
        chunk_writer,
        metadata_consolidator,
        layout_adapter,
        parallel_coord_names: vec!["time".to_string()],
        queue_capacity: BENCH_QUEUE_CAPACITY,
    })
}

fn build_coordinator(case: BenchCase, cuda_mode: CudaBenchMode) -> TestWriteCoordinator {
    build_coordinator_with_pool_buffer_bytes(case.bytes_per_task, cuda_mode)
}

/// Returns a pair of long-lived Rayon thread pools shared across all benchmark
/// iterations. This avoids measuring OS thread creation/destruction overhead
/// in per-iteration coordinator construction (critical for small-payload
/// first-write benchmarks where pool warm-up dwarfs actual work).
fn shared_bench_rayon_pools() -> (Arc<rayon::ThreadPool>, Arc<rayon::ThreadPool>) {
    static COPY_POOL: OnceLock<Arc<rayon::ThreadPool>> = OnceLock::new();
    static FLUSH_POOL: OnceLock<Arc<rayon::ThreadPool>> = OnceLock::new();
    let copy = Arc::clone(COPY_POOL.get_or_init(|| {
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(BENCH_WORKERS)
                .thread_name(|idx| format!("e2s-bench-copy-{idx}"))
                .build()
                .expect("failed to build shared benchmark copy thread pool"),
        )
    }));
    let flush = Arc::clone(FLUSH_POOL.get_or_init(|| {
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(BENCH_WORKERS)
                .thread_name(|idx| format!("e2s-bench-flush-{idx}"))
                .build()
                .expect("failed to build shared benchmark flush thread pool"),
        )
    }));
    (copy, flush)
}

fn build_coordinator_with_shared_pools(
    pool_buffer_bytes: usize,
    cuda_mode: CudaBenchMode,
) -> TestWriteCoordinator {
    let layout_adapter: Arc<dyn ZarrLayoutAdapter> =
        Arc::new(DefaultZarrLayoutAdapter::v2_default());
    let chunk_writer: Arc<dyn ChunkWriter> = Arc::new(BenchNoopChunkWriter);
    let planner: Arc<dyn ChunkPlanner> =
        Arc::new(MixedRadixChunkPlanner::new(WriteExecutionConfig::default()));
    let chunk_registry: Arc<dyn ChunkKeyRegistry> = Arc::new(NoopChunkKeyRegistry);
    let scheduler: Arc<dyn WorkScheduler> = Arc::new(RayonWorkScheduler::new());
    let pool_config = BufferPoolConfig {
        max_pool_buffers: BENCH_WORKERS,
        max_pool_bytes: pool_buffer_bytes
            .saturating_mul(BENCH_WORKERS)
            .saturating_mul(2),
        pool_buffer_bytes: SizeOverride::Fixed(pool_buffer_bytes),
        hot_slab_buffers: SizeOverride::Fixed(BENCH_WORKERS),
        warm_slab_buffers: SizeOverride::Auto,
        pin_pooled_slabs: false,
        cuda_register_pool_if_available: cuda_mode.enable_cuda_registration(),
        cuda_register_each_slab_once: cuda_mode.enable_cuda_registration(),
        ..BufferPoolConfig::default()
    };
    let buffer_pool: Arc<dyn BufferPool> = Arc::new(MemoryBufferPool::new(pool_config));
    let copy_engine: Arc<dyn CopyEngine> = Arc::new(DefaultCopyEngine::new());
    let metadata_consolidator: Arc<dyn MetadataConsolidator> =
        Arc::new(NoopMetadataConsolidator::new());

    let (copy_pool, flush_pool) = shared_bench_rayon_pools();
    TestWriteCoordinator::new_with_shared_pools(
        TestWriteCoordinatorComponents {
            planner,
            chunk_registry,
            scheduler,
            buffer_pool,
            copy_engine,
            chunk_writer,
            metadata_consolidator,
            layout_adapter,
            parallel_coord_names: vec!["time".to_string()],
            queue_capacity: BENCH_QUEUE_CAPACITY,
        },
        copy_pool,
        flush_pool,
    )
}

fn close_bench_coordinator(coordinator: &TestWriteCoordinator) {
    // Bench teardown should never dominate benchmark wall-clock execution.
    // We intentionally ignore close timeout errors because submit_write timing
    // is the measured metric for this suite.
    let _ = coordinator.close(BENCH_CLOSE_TIMEOUT_SECONDS, None);
}

fn measure_realistic_steady_state_submit_write_once(
    coordinator: &TestWriteCoordinator,
    fixture: &RealisticBenchFixture,
    warmup_context: &'static str,
    steady_state_context: &'static str,
) -> Duration {
    let warmup_ack = coordinator
        .submit_write(
            &fixture.request,
            &fixture.array_ids,
            &fixture.registered_coords,
        )
        .expect(warmup_context);
    black_box(warmup_ack.copied_tasks);

    let start = Instant::now();
    let steady_ack = coordinator
        .submit_write(
            &fixture.request,
            &fixture.array_ids,
            &fixture.registered_coords,
        )
        .expect(steady_state_context);
    black_box(steady_ack.copied_tasks);
    start.elapsed()
}

fn build_realistic_fcn_step_fixture() -> RealisticBenchFixture {
    let mut registered_coords = CoordMap::new();
    let _ = registered_coords.insert(
        "time".to_string(),
        CoordValues::I64(vec![REALISTIC_STEP_TIME_INDEX]),
    );
    let _ = registered_coords.insert(
        "lead_time".to_string(),
        CoordValues::I64(sequential_i64(REALISTIC_REGISTERED_LEAD_TIMES)),
    );

    let mut step_coords = CoordMap::new();
    let _ = step_coords.insert(
        "time".to_string(),
        CoordValues::I64(vec![REALISTIC_STEP_TIME_INDEX]),
    );
    let _ = step_coords.insert(
        "lead_time".to_string(),
        CoordValues::I64(vec![REALISTIC_STEP_LEAD_TIME_INDEX]),
    );

    let mut array_names = Vec::with_capacity(REALISTIC_ARRAY_COUNT);
    let mut arrays = Vec::with_capacity(REALISTIC_ARRAY_COUNT);
    let mut array_ids = Vec::with_capacity(REALISTIC_ARRAY_COUNT);
    for idx in 0..REALISTIC_ARRAY_COUNT {
        let fill = u8::try_from(idx % 251).expect("fill byte should fit in u8");
        let bytes = vec![fill; REALISTIC_BYTES_PER_ARRAY];
        array_names.push(format!("var_{idx:02}"));
        arrays.push(InputArray {
            nbytes: REALISTIC_BYTES_PER_ARRAY,
            source: InputArraySource::HostBytes(bytes.into()),
        });
        array_ids.push(u32::try_from(idx).expect("array index should fit in u32"));
    }

    let request = InferenceWriteRequest {
        coords: step_coords,
        array_names,
        arrays,
    };
    RealisticBenchFixture {
        request,
        array_ids,
        registered_coords,
    }
}

fn bench_submit_write_copy_barrier_first_write(c: &mut Criterion) {
    // Pre-warm the shared Rayon pools once so the first iteration doesn't pay
    // thread creation cost either.
    let _ = shared_bench_rayon_pools();

    let mut group = c.benchmark_group("coordinator/submit_write_copy_barrier/first_write");
    for cuda_mode in CUDA_BENCH_MODES {
        for case in BENCH_CASES {
            group.bench_function(
                BenchmarkId::new(format!("submit_write/{}", cuda_mode.label()), case.label),
                |b| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let coordinator =
                                build_coordinator_with_shared_pools(case.bytes_per_task, cuda_mode);
                            let request = build_request(case);
                            let array_ids = vec![0_u32];
                            let registered_coords = build_registered_coords(case);

                            let start = Instant::now();
                            let ack = coordinator
                                .submit_write(&request, &array_ids, &registered_coords)
                                .expect("first submit_write should succeed");
                            total += start.elapsed();
                            black_box(ack.copied_tasks);
                            close_bench_coordinator(&coordinator);
                        }
                        total
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_submit_write_copy_barrier_steady_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator/submit_write_copy_barrier/steady_state");
    for cuda_mode in CUDA_BENCH_MODES {
        for case in BENCH_CASES {
            let coordinator = build_coordinator(case, cuda_mode);
            let request = build_request(case);
            let array_ids = vec![0_u32];
            let registered_coords = build_registered_coords(case);

            coordinator
                .submit_write(&request, &array_ids, &registered_coords)
                .expect("warmup submit_write should succeed");

            group.bench_function(
                BenchmarkId::new(format!("submit_write/{}", cuda_mode.label()), case.label),
                |b| {
                    b.iter(|| {
                        let ack = coordinator
                            .submit_write(&request, &array_ids, &registered_coords)
                            .expect("steady-state submit_write should succeed");
                        black_box(ack.copied_tasks);
                    });
                },
            );

            close_bench_coordinator(&coordinator);
        }
    }
    group.finish();
}

fn bench_cuda_runtime_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator/submit_write_copy_barrier/cuda_probe");
    let cuda_available = shared_cuda_runtime_api().available();
    group.bench_function("runtime_available", |b| {
        b.iter(|| {
            black_box(cuda_available);
        });
    });
    group.finish();
}

fn bench_submit_write_copy_barrier_realistic_fcn_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator/submit_write_copy_barrier/realistic_fcn_step");
    let fixture = build_realistic_fcn_step_fixture();

    for cuda_mode in CUDA_BENCH_MODES {
        group.bench_function(
            format!(
                "first_write/submit_write/26array_4MiB_host_bytes/{}",
                cuda_mode.label()
            ),
            |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let coordinator = build_coordinator_with_pool_buffer_bytes(
                            REALISTIC_BYTES_PER_ARRAY,
                            cuda_mode,
                        );
                        let start = Instant::now();
                        let ack = coordinator
                            .submit_write(
                                &fixture.request,
                                &fixture.array_ids,
                                &fixture.registered_coords,
                            )
                            .expect("realistic first submit_write should succeed");
                        total += start.elapsed();
                        black_box(ack.copied_tasks);
                        close_bench_coordinator(&coordinator);
                    }
                    total
                });
            },
        );
        group.bench_function(
            format!(
                "steady_state/submit_write/26array_4MiB_host_bytes/{}",
                cuda_mode.label()
            ),
            |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let coordinator = build_coordinator_with_pool_buffer_bytes(
                            REALISTIC_BYTES_PER_ARRAY,
                            cuda_mode,
                        );
                        total += measure_realistic_steady_state_submit_write_once(
                            &coordinator,
                            &fixture,
                            "realistic warmup submit_write should succeed",
                            "realistic steady-state submit_write should succeed",
                        );
                        close_bench_coordinator(&coordinator);
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

fn bench_submit_write_copy_barrier_realistic_fcn_step_localfs(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("coordinator/submit_write_copy_barrier/realistic_fcn_step_localfs");
    // This variant performs real chunk writes to local filesystem, so keep sample
    // collection modest to avoid excessive disk pressure during routine bench runs.
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));
    let fixture = build_realistic_fcn_step_fixture();

    for cuda_mode in CUDA_BENCH_MODES {
        group.bench_function(
            format!(
                "first_write/submit_write/26array_4MiB_host_bytes/{}",
                cuda_mode.label()
            ),
            |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let dataset_root = unique_dataset_root("e2s_real_writer_first");
                        let coordinator = build_localfs_coordinator(
                            REALISTIC_BYTES_PER_ARRAY,
                            cuda_mode,
                            dataset_root.clone(),
                        );
                        let start = Instant::now();
                        let ack = coordinator
                            .submit_write(
                                &fixture.request,
                                &fixture.array_ids,
                                &fixture.registered_coords,
                            )
                            .expect("real-writer first submit_write should succeed");
                        total += start.elapsed();
                        black_box(ack.copied_tasks);
                        close_bench_coordinator(&coordinator);
                        remove_dataset_root(&dataset_root);
                    }
                    total
                });
            },
        );

        group.bench_function(
            format!(
                "steady_state/submit_write/26array_4MiB_host_bytes/{}",
                cuda_mode.label()
            ),
            |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let dataset_root = unique_dataset_root("e2s_real_writer_steady");
                        let coordinator = build_localfs_coordinator(
                            REALISTIC_BYTES_PER_ARRAY,
                            cuda_mode,
                            dataset_root.clone(),
                        );
                        total += measure_realistic_steady_state_submit_write_once(
                            &coordinator,
                            &fixture,
                            "real-writer warmup submit_write should succeed",
                            "real-writer steady-state submit_write should succeed",
                        );
                        close_bench_coordinator(&coordinator);
                        remove_dataset_root(&dataset_root);
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    write_copy_barrier_benches,
    bench_cuda_runtime_probe,
    bench_submit_write_copy_barrier_first_write,
    bench_submit_write_copy_barrier_steady_state,
    bench_submit_write_copy_barrier_realistic_fcn_step,
    bench_submit_write_copy_barrier_realistic_fcn_step_localfs
);
criterion_main!(write_copy_barrier_benches);
