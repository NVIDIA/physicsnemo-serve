# FourCastNet (FCN) Ensemble multi-GPU Benchmark

**Hardware:** NVIDIA H100 GPUs  
**Python baseline:** 1x H100 GPU  
**PhysicsNeMo Serve multi-GPU:** 8x H100 GPUs in parallel with optimized I/O backend  
**Date:** 2026-05-07

## Test Setup

This benchmark compares the Python baseline against the PhysicsNeMo Serve multi-GPU ensemble version for the
same FCN workload while sweeping `batch_size`. Speedup values are wall-clock speedups:
`Python baseline wall time / PhysicsNeMo Serve E2E wall time`. They are not GPU-normalized.

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

## Plots

The high-resolution PNG generated from this trimmed report is available at:

`outputs/benchmark_report_physicsnemo_serve_fcn_ensemble_fanout_plots_only_1200dpi.png`

It includes:

- End-to-end runtime speedup by batch size
- Enhanced-version E2E phase breakdown by batch size
