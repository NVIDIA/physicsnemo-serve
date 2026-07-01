# e2s_zarr_io Benchmark Report — Three-Way IO Backend Comparison

**Hardware:** NVIDIA H100 80GB PCIe, Local SSD
**Software:** earth2studio 0.12.1, CUDA 12.x, Rust e2s_zarr_io
**Date:** 2026-03-17

## Backends Under Test

| Backend | Implementation | Description |
|---------|---------------|-------------|
| **rust** | `e2s_zarr_io.E2sZarrIoBackend` | Optimized Rust extension with copy-barrier semantics, pre-allocated pinned slab pool, parallel rayon flush workers, FsyncPolicy::Never |
| **py_async** | `earth2studio.io.AsyncZarrBackend` | Python async backend running in `blocking=True` mode (synchronous per-call, thread-pool flush) |
| **zarr_sync** | `earth2studio.io.ZarrBackend` | Default synchronous Python Zarr backend (pure Python, no async, no thread pool) |

---

## Cross-Model Summary

### Total IO Time

| Model | Vars | Grid | MB/step | Steps | zarr_sync | py_async | rust | rust vs zarr_sync | rust vs py_async | py_async vs zarr_sync |
|-------|------|------|---------|-------|-----------|----------|------|-------------------|------------------|-----------------------|
| DLWP | 7 | 721x1440 | 29MB | 20 | 6.96s | 1.65s | 274ms | 25.4x | 6.0x | 4.2x |
| SFNO | 73 | 721x1440 | 289MB | 10 | 45.88s | 8.44s | 659ms | 69.6x | 12.8x | 5.4x |
| Pangu3 | 69 | 721x1440 | 273MB | 10 | 34.65s | 7.65s | 612ms | 56.6x | 12.5x | 4.5x |
| StormCast | 99 | 512x640 | 124MB | 10 | 33.92s | 7.30s | 468ms | 72.5x | 15.6x | 4.6x |
| FCN3 | 72 | 721x1440 | 285MB | 10 | 34.83s | 9.06s | 597ms | 58.4x | 15.2x | 3.8x |

### Steady-State Per-Step io_write

| Model | zarr_sync avg | py_async avg | rust avg | rust vs zarr_sync | rust vs py_async |
|-------|--------------|-------------|---------|-------------------|------------------|
| DLWP | 324.3ms | 70.7ms | 3.2ms | 102x | 22x |
| SFNO | 4116.6ms | 697.0ms | 24.9ms | 165x | 28x |
| Pangu3 | 3076.9ms | 616.5ms | 25.9ms | 119x | 24x |
| StormCast | 3017.0ms | 602.7ms | 21.6ms | 139x | 28x |
| FCN3 | 3107.2ms | 774.9ms | 35.0ms | 89x | 22x |

### End-to-End Wall Time

| Model | zarr_sync | py_async | rust | rust vs zarr_sync | rust vs py_async |
|-------|-----------|----------|------|-------------------|------------------|
| DLWP | 9.95s | 4.49s | 3.02s | 3.3x | 1.5x |
| SFNO | 60.13s | 24.45s | 16.78s | 3.6x | 1.5x |
| Pangu3 | 82.22s | 61.37s | 53.24s | 1.5x | 1.2x |
| StormCast | 107.34s | 78.35s | 73.67s | 1.5x | 1.1x |
| FCN3 | 1502.41s | 1477.18s | 1467.24s | 1.0x | 1.0x |

---

## DLWP — 7 vars, 721x1440, ~29 MB/step, 20 steps

**Rust pool config:** hot_slab_buffers=52, warm_slab_buffers=26, max_pool_buffers=128

### Phase Breakdown

| Phase | zarr_sync | py_async | rust | zs/rust | pa/rust |
|-------|-----------|----------|------|---------|---------|
| total_wall | 9.95s | 4.49s | 3.02s | 3.3x | 1.5x |
| data_fetch | 1.59s | 1.37s | 1.37s | 1.2x | 1.0x |
| model_load | 1.21s | 1.29s | 1.20s | 1.0x | 1.1x |
| model_to_device | 13.3ms | 12.3ms | 12.9ms | 1.0x | 1.0x |
| io_setup | 139.9ms | 19.7ms | 61.7ms | 2.3x | 0.3x |
| io_write | 6.82s | 1.63s | 206.3ms | 33.1x | 7.9x |
| io_close | 0.0ms | 0.0ms | 6.2ms | 0.0x | 0.0x |
| step_compute | 173.1ms | 158.1ms | 156.5ms | 1.1x | 1.0x |
| **total_io** | **6.96s** | **1.65s** | **274.3ms** | **25.4x** | **6.0x** |

**Rust close() internals:** async_drain=4.0ms, consolidate=6.0ms, teardown=0.0ms, total=6.2ms

### Per-Step io_write

| Step | zarr_sync | py_async | rust caller | rust worker_copy | zs/rust | pa/rust |
|------|-----------|----------|-------------|-----------------|---------|---------|
| 0 (init) | 333.9ms | 219.1ms | 143.0ms | 9.4ms | 2x | 2x |
| 1 | 312.5ms | 70.7ms | 3.2ms | 9.0ms | 98x | 22x |
| 2 | 330.9ms | 89.9ms | 3.3ms | 9.2ms | 100x | 27x |
| 3 | 320.3ms | 78.4ms | 3.4ms | 9.5ms | 96x | 23x |
| 4 | 325.2ms | 78.3ms | 3.3ms | 9.0ms | 100x | 24x |
| 5 | 320.3ms | 69.6ms | 3.0ms | 8.8ms | 108x | 24x |
| 6 | 328.1ms | 68.9ms | 3.6ms | 10.3ms | 91x | 19x |
| 7 | 331.9ms | 70.1ms | 3.1ms | 9.3ms | 107x | 23x |
| 8 | 327.1ms | 85.8ms | 2.9ms | 9.2ms | 112x | 29x |
| 9 | 335.8ms | 63.7ms | 3.2ms | 9.6ms | 107x | 20x |
| 10 | 332.7ms | 63.1ms | 3.0ms | 9.4ms | 110x | 21x |
| 11 | 311.6ms | 64.5ms | 3.3ms | 9.9ms | 95x | 20x |
| 12 | 326.9ms | 72.1ms | 3.1ms | 9.4ms | 107x | 24x |
| 13 | 328.2ms | 69.3ms | 3.3ms | 10.0ms | 100x | 21x |
| 14 | 320.0ms | 66.1ms | 3.1ms | 9.6ms | 103x | 21x |
| 15 | 337.0ms | 65.2ms | 3.2ms | 9.6ms | 104x | 20x |
| 16 | 314.4ms | 71.0ms | 3.2ms | 9.6ms | 99x | 22x |
| 17 | 323.0ms | 65.8ms | 3.2ms | 9.7ms | 102x | 21x |
| 18 | 316.9ms | 66.9ms | 3.0ms | 9.3ms | 104x | 22x |
| 19 | 313.2ms | 68.0ms | 3.1ms | 9.5ms | 100x | 22x |
| 20 | 329.8ms | 67.1ms | 3.0ms | 9.2ms | 110x | 22x |

**Steady state (steps 1-20):**
- zarr_sync: avg=324.3ms, min=311.6ms, max=337.0ms
- py_async: avg=70.7ms, min=63.1ms, max=89.9ms
- rust: avg=3.2ms, min=2.9ms, max=3.6ms
- **Rust vs zarr_sync: 102x faster per step**
- **Rust vs py_async: 22x faster per step**

---

## SFNO — 73 vars, 721x1440, ~289 MB/step, 10 steps

**Rust pool config:** hot_slab_buffers=150, warm_slab_buffers=75, max_pool_buffers=300, max_pool_bytes=2147483648

### Phase Breakdown

| Phase | zarr_sync | py_async | rust | zs/rust | pa/rust |
|-------|-----------|----------|------|---------|---------|
| total_wall | 60.13s | 24.45s | 16.78s | 3.6x | 1.5x |
| data_fetch | 2.78s | 4.20s | 4.20s | 0.7x | 1.0x |
| model_load | 9.67s | 10.06s | 10.11s | 1.0x | 1.0x |
| model_to_device | 456.4ms | 414.0ms | 416.5ms | 1.1x | 1.0x |
| io_setup | 593.9ms | 19.6ms | 3.0ms | 198.1x | 6.6x |
| io_write | 45.29s | 8.42s | 634.3ms | 71.4x | 13.3x |
| io_close | 0.0ms | 0.0ms | 21.7ms | 0.0x | 0.0x |
| step_compute | 1.33s | 1.34s | 1.39s | 1.0x | 1.0x |
| **total_io** | **45.88s** | **8.44s** | **659.0ms** | **69.6x** | **12.8x** |

**Rust close() internals:** async_drain=21.7ms, consolidate=20.8ms, teardown=0.0ms, total=21.7ms

### Per-Step io_write

| Step | zarr_sync | py_async | rust caller | rust worker_copy | zs/rust | pa/rust |
|------|-----------|----------|-------------|-----------------|---------|---------|
| 0 (init) | 4122.7ms | 1445.9ms | 384.9ms | 179.8ms | 11x | 4x |
| 1 | 3925.5ms | 682.5ms | 25.0ms | 176.4ms | 157x | 27x |
| 2 | 4006.4ms | 741.5ms | 24.1ms | 169.8ms | 166x | 31x |
| 3 | 4225.0ms | 680.2ms | 24.7ms | 174.1ms | 171x | 28x |
| 4 | 4221.3ms | 660.0ms | 25.2ms | 176.0ms | 168x | 26x |
| 5 | 4088.8ms | 658.2ms | 24.8ms | 172.7ms | 165x | 27x |
| 6 | 3988.4ms | 719.6ms | 25.0ms | 174.2ms | 160x | 29x |
| 7 | 4367.5ms | 617.0ms | 24.8ms | 173.8ms | 176x | 25x |
| 8 | 4261.5ms | 739.1ms | 24.7ms | 173.4ms | 173x | 30x |
| 9 | 3990.6ms | 690.2ms | 25.4ms | 176.1ms | 157x | 27x |
| 10 | 4091.1ms | 781.9ms | 25.8ms | 180.9ms | 159x | 30x |

**Steady state (steps 1-10):**
- zarr_sync: avg=4116.6ms, min=3925.5ms, max=4367.5ms
- py_async: avg=697.0ms, min=617.0ms, max=781.9ms
- rust: avg=24.9ms, min=24.1ms, max=25.8ms
- **Rust vs zarr_sync: 165x faster per step**
- **Rust vs py_async: 28x faster per step**

---

## Pangu3 — 69 vars, 721x1440, ~273 MB/step, 10 steps

**Rust pool config:** hot_slab_buffers=140, warm_slab_buffers=70, max_pool_buffers=280, max_pool_bytes=2147483648

### Phase Breakdown

| Phase | zarr_sync | py_async | rust | zs/rust | pa/rust |
|-------|-----------|----------|------|---------|---------|
| total_wall | 82.22s | 61.37s | 53.24s | 1.5x | 1.2x |
| data_fetch | 2.82s | 8.22s | 8.22s | 0.3x | 1.0x |
| model_load | 9.66s | 9.96s | 9.70s | 1.0x | 1.0x |
| model_to_device | 10.11s | 10.22s | 10.21s | 1.0x | 1.0x |
| io_setup | 538.0ms | 31.1ms | 2.2ms | 245.9x | 14.2x |
| io_write | 34.11s | 7.62s | 587.3ms | 58.1x | 13.0x |
| io_close | 0.0ms | 0.0ms | 22.1ms | 0.0x | 0.0x |
| step_compute | 24.98s | 25.32s | 24.50s | 1.0x | 1.0x |
| **total_io** | **34.65s** | **7.65s** | **611.6ms** | **56.6x** | **12.5x** |

**Rust close() internals:** async_drain=22.0ms, consolidate=20.3ms, teardown=0.0ms, total=22.1ms

### Per-Step io_write

| Step | zarr_sync | py_async | rust caller | rust worker_copy | zs/rust | pa/rust |
|------|-----------|----------|-------------|-----------------|---------|---------|
| 0 (init) | 3340.2ms | 1456.3ms | 328.6ms | 181.6ms | 10x | 4x |
| 1 | 3091.2ms | 731.4ms | 25.9ms | 181.1ms | 119x | 28x |
| 2 | 2904.1ms | 607.4ms | 25.8ms | 179.9ms | 113x | 24x |
| 3 | 3127.3ms | 597.1ms | 26.1ms | 182.8ms | 120x | 23x |
| 4 | 2984.5ms | 590.3ms | 26.0ms | 183.6ms | 115x | 23x |
| 5 | 3161.6ms | 600.7ms | 25.2ms | 176.7ms | 125x | 24x |
| 6 | 3139.3ms | 608.6ms | 27.0ms | 188.9ms | 116x | 23x |
| 7 | 3188.8ms | 618.7ms | 26.1ms | 182.5ms | 122x | 24x |
| 8 | 3148.9ms | 625.6ms | 25.4ms | 178.0ms | 124x | 25x |
| 9 | 3211.7ms | 587.6ms | 25.6ms | 180.3ms | 125x | 23x |
| 10 | 2811.9ms | 597.8ms | 25.7ms | 180.1ms | 109x | 23x |

**Steady state (steps 1-10):**
- zarr_sync: avg=3076.9ms, min=2811.9ms, max=3211.7ms
- py_async: avg=616.5ms, min=587.6ms, max=731.4ms
- rust: avg=25.9ms, min=25.2ms, max=27.0ms
- **Rust vs zarr_sync: 119x faster per step**
- **Rust vs py_async: 24x faster per step**

---

## StormCast — 99 vars, 512x640, ~124 MB/step, 10 steps

**Rust pool config:** hot_slab_buffers=100, warm_slab_buffers=50, max_pool_buffers=200, max_pool_bytes=1073741824

### Phase Breakdown

| Phase | zarr_sync | py_async | rust | zs/rust | pa/rust |
|-------|-----------|----------|------|---------|---------|
| total_wall | 107.34s | 78.35s | 73.67s | 1.5x | 1.1x |
| data_fetch | 5.06s | 4.95s | 4.95s | 1.0x | 1.0x |
| model_load | 8.98s | 8.68s | 8.93s | 1.0x | 1.0x |
| model_to_device | 169.9ms | 157.7ms | 180.1ms | 0.9x | 0.9x |
| io_setup | 758.1ms | 16.7ms | 2.4ms | 317.5x | 7.0x |
| io_write | 33.17s | 7.29s | 432.8ms | 76.6x | 16.8x |
| io_close | 0.0ms | 0.0ms | 32.4ms | 0.0x | 0.0x |
| step_compute | 59.21s | 57.25s | 59.14s | 1.0x | 1.0x |
| **total_io** | **33.92s** | **7.30s** | **467.7ms** | **72.5x** | **15.6x** |

**Rust close() internals:** async_drain=28.6ms, consolidate=31.9ms, teardown=0.0ms, total=32.3ms

### Per-Step io_write

| Step | zarr_sync | py_async | rust caller | rust worker_copy | zs/rust | pa/rust |
|------|-----------|----------|-------------|-----------------|---------|---------|
| 0 (init) | 2996.1ms | 1261.2ms | 216.5ms | 112.2ms | 14x | 6x |
| 1 | 2959.5ms | 601.8ms | 19.3ms | 117.2ms | 153x | 31x |
| 2 | 3294.3ms | 597.3ms | 18.1ms | 111.6ms | 182x | 33x |
| 3 | 2990.8ms | 886.9ms | 19.0ms | 115.7ms | 158x | 47x |
| 4 | 2994.1ms | 571.4ms | 18.2ms | 112.0ms | 164x | 31x |
| 5 | 2974.7ms | 517.3ms | 47.9ms | 336.2ms | 62x | 11x |
| 6 | 2984.3ms | 575.3ms | 19.6ms | 123.5ms | 152x | 29x |
| 7 | 2999.6ms | 583.3ms | 19.3ms | 114.9ms | 156x | 30x |
| 8 | 2984.3ms | 542.2ms | 18.0ms | 110.6ms | 166x | 30x |
| 9 | 2989.1ms | 581.4ms | 18.7ms | 113.1ms | 159x | 31x |
| 10 | 2999.7ms | 569.7ms | 18.2ms | 111.3ms | 165x | 31x |

**Steady state (steps 1-10):**
- zarr_sync: avg=3017.0ms, min=2959.5ms, max=3294.3ms
- py_async: avg=602.7ms, min=517.3ms, max=886.9ms
- rust: avg=21.6ms, min=18.0ms, max=47.9ms
- **Rust vs zarr_sync: 139x faster per step**
- **Rust vs py_async: 28x faster per step**

---

## FCN3 — 72 vars, 721x1440, ~285 MB/step, 10 steps

**Rust pool config:** hot_slab_buffers=73, warm_slab_buffers=36, max_pool_buffers=150, max_pool_bytes=1073741824

### Phase Breakdown

| Phase | zarr_sync | py_async | rust | zs/rust | pa/rust |
|-------|-----------|----------|------|---------|---------|
| total_wall | 1502.41s | 1477.18s | 1467.24s | 1.0x | 1.0x |
| data_fetch | 2.56s | 2.49s | 2.49s | 1.0x | 1.0x |
| model_load | 15.03s | 15.46s | 15.48s | 1.0x | 1.0x |
| model_to_device | 1.03s | 1.34s | 1.19s | 0.9x | 1.1x |
| io_setup | 570.0ms | 19.6ms | 2.1ms | 276.9x | 9.5x |
| io_write | 34.26s | 9.04s | 571.3ms | 60.0x | 15.8x |
| io_close | 0.0ms | 0.0ms | 23.4ms | 0.0x | 0.0x |
| step_compute | 1448.95s | 1448.84s | 1447.48s | 1.0x | 1.0x |
| **total_io** | **34.83s** | **9.06s** | **596.8ms** | **58.4x** | **15.2x** |

**Rust close() internals:** async_drain=23.4ms, consolidate=22.7ms, teardown=0.0ms, total=23.4ms

### Per-Step io_write

| Step | zarr_sync | py_async | rust caller | rust worker_copy | zs/rust | pa/rust |
|------|-----------|----------|-------------|-----------------|---------|---------|
| 0 (init) | 3188.2ms | 1290.2ms | 221.0ms | 194.5ms | 14x | 6x |
| 1 | 3126.3ms | 794.5ms | 99.5ms | 739.8ms | 31x | 8x |
| 2 | 3052.9ms | 882.8ms | 28.4ms | 200.3ms | 107x | 31x |
| 3 | 3060.8ms | 766.1ms | 28.3ms | 196.4ms | 108x | 27x |
| 4 | 3101.1ms | 788.3ms | 27.8ms | 193.8ms | 112x | 28x |
| 5 | 3061.0ms | 769.0ms | 28.0ms | 195.1ms | 109x | 28x |
| 6 | 3059.7ms | 724.3ms | 28.7ms | 197.6ms | 107x | 25x |
| 7 | 3038.6ms | 815.5ms | 28.2ms | 193.2ms | 108x | 29x |
| 8 | 3064.3ms | 723.4ms | 26.9ms | 188.2ms | 114x | 27x |
| 9 | 3364.9ms | 748.2ms | 27.5ms | 191.9ms | 122x | 27x |
| 10 | 3142.8ms | 737.3ms | 27.1ms | 189.9ms | 116x | 27x |

**Steady state (steps 1-10):**
- zarr_sync: avg=3107.2ms, min=3038.6ms, max=3364.9ms
- py_async: avg=774.9ms, min=723.4ms, max=882.8ms
- rust: avg=35.0ms, min=26.9ms, max=99.5ms
- **Rust vs zarr_sync: 89x faster per step**
- **Rust vs py_async: 22x faster per step**

---

## Architecture Notes

### How the Rust backend achieves these results

1. **Copy-barrier semantics:** `write()` copies data from GPU to pinned host buffer, then returns immediately. The actual disk flush happens asynchronously on background threads. The caller only waits for the memcpy, not the fsync.
2. **Pre-allocated pinned slab pool:** Hot slab buffers are pre-allocated and page-faulted at `add_array()` time. Buffer reservation is lock-free in steady state.
3. **Parallel rayon flush workers:** Each chunk is flushed to disk on a rayon thread pool. Multiple chunks from the same step flush concurrently.
4. **Close overlap:** Metadata consolidation runs concurrently with the final async drain, not sequentially.
5. **FsyncPolicy::Never:** Skips per-chunk and per-metadata fsync calls. Safe for re-computable inference outputs. Default remains `FsyncPolicy::Always`.

### Why zarr_sync is slowest

- Pure synchronous Python: each array is written sequentially with `np.isin()` index lookups (O(n*m) per dimension)
- No chunked parallel writes: all data goes through a single Python thread
- `add_array()` eagerly allocates full arrays upfront (higher setup cost for many variables)
- No thread pool or async machinery to overlap IO with compute

### Why py_async is faster than zarr_sync but slower than Rust

- Uses a thread pool (default 8 workers) for parallel chunk writes
- Runs in `blocking=True` mode for this benchmark, so each `write()` call waits for completion
- Still pure Python: numpy array slicing, zarr codec overhead, GIL contention in the thread pool
- No pinned memory, no copy-barrier: the caller is blocked until the full write + flush completes

### Pool sizing guidelines

| Model type | Recommended hot_slab | Rationale |
|------------|---------------------|-----------|
| Few vars (DLWP) | 2-4x variable count | Ample headroom, small memory |
| Many vars (SFNO, Pangu3, FCN3) | vars + 1-5 buffers | Tight sizing to avoid GPU memory pressure |
| Diffusion (StormCast) | vars + 1 buffer | Minimize pinned memory, GPU needs the headroom |

Over-provisioning pinned memory can cause GPU memory pressure and slow down inference (observed: StormCast step_compute 83s with 200 hot -> 59s with 100 hot).

### Timing columns explained

- **zarr_sync:** Wall time for `ZarrBackend.write()` (synchronous Python, no async)
- **py_async:** Wall time for `AsyncZarrBackend.write()` with `blocking=True`
- **rust caller:** Wall time for `SyncZarrBackend.write()` as seen by the Python caller (includes memcpy to pool buffer, excludes async flush)
- **rust worker_copy:** Total time spent by background rayon workers copying data and flushing to disk (runs off the caller's critical path)
