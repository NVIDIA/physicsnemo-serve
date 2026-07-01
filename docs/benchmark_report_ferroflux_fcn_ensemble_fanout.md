# PhysicsNeMo Serve FCN Ensemble Fanout Benchmark Report — Python vs PhysicsNeMo Serve

**Hardware:** NVIDIA H100 GPUs  
**Python baseline:** 1x H100 GPU  
**PhysicsNeMo Serve ensemble fanout:** 8x H100 GPUs  
**Date:** 2026-05-07

## Test Configuration

| Field | Value |
|-------|-------|
| Model | `fcn` |
| Device kind | `gpu` |
| Ensemble size | 512 |
| Steps | 10 |
| Max in flight | 16 |
| Perturbation | `gaussian` |
| Noise amplitude | 0.15 |
| Varying parameter | `batch_size` |

This benchmark compares the Python baseline against PhysicsNeMo Serve ensemble fanout for the
same FCN workload while sweeping `batch_size`. Speedup values below are wall-clock
speedups: `Python wall time / PhysicsNeMo Serve E2E wall time`. They are not GPU-normalized.

---

## Total Wall-Time Summary

| Batch size | Python wall time | PhysicsNeMo Serve E2E wall time | PhysicsNeMo Serve children | Run ID | Wall-clock speedup |
|------------|------------------|-------------------------|--------------------|--------|--------------------|
| 16 | 6687s (111.5 min) | 269.7s (4.5 min) | 32/32 | `3ed26db7` | 24.8x |
| 32 | 6959s (116.0 min) | 226.3s (3.8 min) | 16/16 | `93482caa` | 30.8x |
| 64 | 6688s (111.5 min) | 239.6s (4.0 min) | 8/8 | `89b5bf1f` | 27.9x |

The best measured wall-clock result was `batch_size=32`, with 226.3s E2E runtime and
30.8x speedup over the Python baseline.


---

## PhysicsNeMo Serve Ensemble Fanout Phase Breakdown

| Batch size | prepare | fanout + execute + collect | postprocess + results | Total E2E |
|------------|---------|-----------------------------|-----------------------|-----------|
| 16 | 165.7s | 84.0s | 20.0s | 269.7s |
| 32 | 142.3s | 73.0s | 11.0s | 226.3s |
| 64 | 161.6s | 72.0s | 6.0s | 239.6s |

### Phase Share of E2E Runtime

| Batch size | prepare | fanout + execute + collect | postprocess + results |
|------------|---------|-----------------------------|-----------------------|
| 16 | 61.4% | 31.1% | 7.4% |
| 32 | 62.9% | 32.3% | 4.9% |
| 64 | 67.4% | 30.1% | 2.5% |

Across the measured batch sizes, `prepare` is the dominant fixed-cost phase, accounting
for roughly 61-67% of total E2E runtime.

---

## Batch-Size Detail Tables

### Batch Size 16

| Metric | Value |
|--------|-------|
| Run ID | `3ed26db7` |
| Ensemble size | 512 |
| Batch size | 16 |
| Steps | 10 |
| Children completed | 32/32 |
| Python wall time | 6687s |
| PhysicsNeMo Serve E2E wall time | 269.7s |
| Wall-clock speedup | 24.8x |

| Phase | Duration | Share of E2E |
|-------|----------|--------------|
| prepare | 165.7s | 61.4% |
| fanout + execute + collect | 84.0s | 31.1% |
| postprocess + results | 20.0s | 7.4% |
| **Total E2E** | **269.7s** | **100.0%** |

### Batch Size 32

| Metric | Value |
|--------|-------|
| Run ID | `93482caa` |
| Ensemble size | 512 |
| Batch size | 32 |
| Steps | 10 |
| Children completed | 16/16 |
| Python wall time | 6959s |
| PhysicsNeMo Serve E2E wall time | 226.3s |
| Wall-clock speedup | 30.8x |

| Phase | Duration | Share of E2E |
|-------|----------|--------------|
| prepare | 142.3s | 62.9% |
| fanout + execute + collect | 73.0s | 32.3% |
| postprocess + results | 11.0s | 4.9% |
| **Total E2E** | **226.3s** | **100.0%** |

### Batch Size 64

| Metric | Value |
|--------|-------|
| Run ID | `89b5bf1f` |
| Ensemble size | 512 |
| Batch size | 64 |
| Steps | 10 |
| Children completed | 8/8 |
| Python wall time | 6688s |
| PhysicsNeMo Serve E2E wall time | 239.6s |
| Wall-clock speedup | 27.9x |

| Phase | Duration | Share of E2E |
|-------|----------|--------------|
| prepare | 161.6s | 67.4% |
| fanout + execute + collect | 72.0s | 30.1% |
| postprocess + results | 6.0s | 2.5% |
| **Total E2E** | **239.6s** | **100.0%** |

---

## Observations

- PhysicsNeMo Serve ensemble fanout delivered 24.8x-30.8x wall-clock speedup over the Python
  baseline across the measured batch sizes.
- `batch_size=32` produced the fastest measured PhysicsNeMo Serve E2E runtime at 226.3s.
- Increasing batch size reduced child count from 32 to 16 to 8, and reduced
  `postprocess + results` time from 20.0s to 11.0s to 6.0s.
- `prepare` remained the largest runtime component in all PhysicsNeMo Serve runs, suggesting it
  is the next primary optimization target.

## Generated Figure

The report-style speedup plot is available at:

`outputs/python_vs_physicsnemo_serve_speedup_h100_report.png`
