# Ensemble Performance Analysis Plan

## Objective

Demonstrate that the PhysicsNeMo Serve Rust-based multi-GPU inference service significantly
outperforms the legacy single-GPU Python inference service for large ensemble
forecasting workloads (512 members). Produce metrics and visualizations suitable
for benchmarking and documentation.

---

## Pre-Work: Establish a Meaningful Single-Ensemble Runtime

Before running scaled experiments, settle on a `default_request.json` input
configuration that makes a single ensemble member run for at least 3-4 minutes.
This ensures the execute stage dominates total runtime and makes scaling
results visually clear.

**Knobs to adjust (without affecting `batch_size`):**

- **`nsteps`** -- The primary lever. Controls how many forecast timesteps the
  model iterates. Each step is one forward pass. Going from `nsteps=10` to
  `nsteps=100` should roughly 10x the execute time. This is the cleanest knob
  since it does not affect memory footprint per member.
- **Model choice (FCN vs DLWP)** -- FCN (FourCastNet) is a larger model than
  DLWP, so each forward pass takes longer. FCN gives a longer per-step time
  without changing `nsteps`. Try both and measure per-step latency.
- **Both combined** -- FCN with high `nsteps` would give the longest runtime per
  ensemble if DLWP is too fast even at high step counts.

**Suggested approach:**

1. Run a single ensemble member (`nensemble=1`, `batch_size=1`) on one GPU with
   DLWP at `nsteps` = 10, 50, 100, 200. Record wall-clock execute time.
2. Repeat with FCN at the same `nsteps` values.
3. Pick the model + `nsteps` combination that lands in the 3-4 minute range per
   single member.
4. Lock that configuration for all subsequent experiments.

---

## Experiments

**Note on `max_in_flight`:**

- **Experiments 1 and 3:** `max_in_flight = 16`. This ensures the scheduler
  always has work pre-queued on each GPU's execute stream, eliminating idle gaps
  between children.
- **Experiment 2:** `max_in_flight = num_gpus`. Since this experiment varies the
  GPU count, we match `max_in_flight` to the number of available GPUs.

### Experiment 1: Wall-Clock Speedup (Baseline Comparison)

The headline experiment. Run the same 512-ensemble workload on both systems and
compare total wall-clock time. Sweep across `batch_size` ∈ {16, 32, 64}.

| Config | Python Service | PhysicsNeMo Serve (Rust) |
|--------|---------------|------------------|
| GPUs | 1 | 8 |
| Ensemble members | 512 | 512 |
| Batch size | 16, 32, 64 | 16, 32, 64 |
| `max_in_flight` | N/A | 16 |

**Measure:** Total wall-clock time from first request submission to final
aggregated result, for each batch size.

### Experiment 2: GPU Scaling Efficiency

Run a 64-ensemble workload on PhysicsNeMo Serve with different GPU counts.
`max_in_flight` always equals the GPU count.

| GPU count | `max_in_flight` | `nensemble` | `batch_size` |
|-----------|-----------------|-------------|--------------|
| 1 | 1 | 64 | 64 |
| 2 | 2 | 64 | 64 |
| 4 | 4 | 64 | 64 |
| 8 | 8 | 64 | 64 |

- Plot wall-clock time vs GPU count
- Plot speedup vs GPU count (normalized to 1 GPU)
- Plot efficiency = speedup / GPU count

This reveals how close the system is to linear scaling and where overhead lives
(scheduling, data movement, aggregation).

### Experiment 3: Batch Size Sensitivity

Fix GPUs at 8, `max_in_flight` at 16. Vary `batch_size` across
{16, 32, 64} while keeping `nensemble=512`.

**Measure:**

- Wall-clock time
- GPU memory utilization (does larger batch use more VRAM?)
- GPU compute utilization (does larger batch increase SM occupancy?)

This reveals the sweet spot where batch size maximizes throughput without
running out of GPU memory.

---

