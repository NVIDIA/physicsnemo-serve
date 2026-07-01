# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Threaded Python-boundary checks for Rust backend GIL release behavior."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest


def _has_overlap(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> bool:
    """Return True if any interval in `first` overlaps any interval in `second`."""
    for first_start, first_end in first:
        for second_start, second_end in second:
            if first_start < second_end and second_start < first_end:
                return True
    return False


def test_rust_backend_multithreaded_writes_are_not_serialized(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    e2s_zarr_io = pytest.importorskip("e2s_zarr_io")
    backend_cls = getattr(e2s_zarr_io, "E2sZarrIoBackend", None)
    if backend_cls is None:
        pytest.skip("E2sZarrIoBackend is not exposed by e2s_zarr_io")

    dataset_path = tmp_path / "threaded_write_overlap.zarr"
    # Use payloads large enough to exceed typical scheduler/GIL timeslice windows.
    # This makes true call overlap observable and avoids false negatives from
    # microsecond-scale writes that can complete before OS preemption.
    grid_size = 2048
    lat_values = np.linspace(-90.0, 90.0, grid_size, dtype=np.float32).tolist()
    lon_values = np.linspace(0.0, 359.75, grid_size, dtype=np.float32).tolist()
    time_values = list(range(8))
    array_names = [f"var_{idx}" for idx in range(8)]

    backend = backend_cls(
        file_name=str(dataset_path),
        queue_capacity=128,
        max_pool_buffers=16,
        hot_slab_buffers=4,
        warm_slab_buffers=4,
        pin_pooled_slabs=False,
        cuda_register_pool_if_available=False,
    )
    try:
        backend.add_array(
            {
                "time": time_values,
                "lat": lat_values,
                "lon": lon_values,
            },
            array_names,
        )

        # One reusable payload set per worker to avoid allocation noise.
        payloads_worker0 = [
            np.full((grid_size, grid_size), fill_value=float(idx), dtype=np.float32)
            for idx in range(len(array_names))
        ]
        payloads_worker1 = [
            np.full(
                (grid_size, grid_size), fill_value=float(100 + idx), dtype=np.float32
            )
            for idx in range(len(array_names))
        ]
        worker0_times: list[tuple[float, float]] = []
        worker1_times: list[tuple[float, float]] = []
        errors: list[str] = []
        lock = threading.Lock()
        start_barrier = threading.Barrier(3)

        def worker(
            worker_id: int,
            worker_times: list[tuple[float, float]],
            write_times: list[int],
            payloads: list[object],
        ) -> None:
            try:
                start_barrier.wait(timeout=10.0)
                for t in write_times:
                    start = time.perf_counter()
                    backend.write(
                        payloads,
                        {
                            "time": [t],
                            "lat": lat_values,
                            "lon": lon_values,
                        },
                        array_names,
                    )
                    end = time.perf_counter()
                    with lock:
                        worker_times.append((start, end))
            except Exception as exc:  # pragma: no cover - failure path assertion below
                with lock:
                    errors.append(f"worker {worker_id}: {exc!r}")

        thread0 = threading.Thread(
            target=worker,
            args=(0, worker0_times, [0, 2, 4], payloads_worker0),
            name="e2s-write-worker-0",
        )
        thread1 = threading.Thread(
            target=worker,
            args=(1, worker1_times, [1, 3, 5], payloads_worker1),
            name="e2s-write-worker-1",
        )

        thread0.start()
        thread1.start()
        # Release both workers into write() at the same time.
        start_barrier.wait(timeout=10.0)
        thread0.join(timeout=180.0)
        thread1.join(timeout=180.0)

        assert not thread0.is_alive(), "worker 0 did not finish (possible deadlock)"
        assert not thread1.is_alive(), "worker 1 did not finish (possible deadlock)"
        assert not errors, f"threaded write workers failed: {errors}"
        assert len(worker0_times) == 3
        assert len(worker1_times) == 3
        assert _has_overlap(worker0_times, worker1_times), (
            "threaded write() calls did not overlap; observed serialized execution"
        )
    finally:
        if not backend.is_closed():
            backend.close()
