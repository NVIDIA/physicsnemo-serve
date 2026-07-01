# e2s_zarr_io

Rust-backed Zarr IO backend for Earth2Studio.

This crate provides the Rust write pipeline and Python bindings used by Earth2Studio for
high-throughput inference-time writes to local Zarr datasets.

## Quickstart

### Build and install Python bindings locally

From this crate directory:

```bash
python -m pip install maturin
python -m maturin develop --features python-bindings
```

### Minimal Python usage

```python
import numpy as np
import e2s_zarr_io

backend = e2s_zarr_io.E2sZarrIoBackend(
    file_name="example.zarr",
    parallel_coords={"time": [0], "lead_time": [0]},
    zarr_format="v3",
)

backend.add_array(
    {
        "time": [0],
        "lead_time": [0],
        "lat": [-90.0, 90.0],
        "lon": [0.0, 180.0],
    },
    ["t2m"],
)

payload = np.zeros((1, 1, 2, 2), dtype=np.float32)
backend.write(
    [payload],
    {
        "time": [0],
        "lead_time": [0],
        "lat": [-90.0, 90.0],
        "lon": [0.0, 180.0],
    },
    ["t2m"],
)
backend.close()
```

## Configuration notes

- `parallel_coords` controls chunked parallel dimensions (`time`, `lead_time`, `ensemble` by default).
- `zarr_format` supports `v2` and `v3`.
- `chunk_key_encoding` and `chunk_key_separator` are validated per selected Zarr format.
- `close_lease_timeout_seconds` sets the default close timeout; Python `close(timeout_seconds=None)` uses this configured value.

## Supported dtypes

`E2sZarrIoBackend` preserves the dtype contract used by `earth2studio.io.ZarrBackend`
for uncompressed local Zarr stores. Data array dtypes are captured at `add_array(...)`
when initial data is provided. If an array is registered without initial data, the
registered dtype defaults to `float32`, matching Earth2Studio's Python backend; later
`write(...)` payloads with a different dtype are cast to that registered dtype before
chunk bytes are copied.

| Category | Supported dtypes | Notes |
| --- | --- | --- |
| Data arrays | `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64`, `datetime64[ns]`, `timedelta64[ns]` | Dtype is parsed from NumPy/Torch dtype metadata or array-interface `typestr`. Big-endian and structured dtypes are not supported. |
| Coordinate arrays | `int32`, `int64`, `uint32`, `uint64`, `float32`, `float64`, `datetime64[ns]`, `timedelta64[ns]`, fixed-width strings | Coordinate dtype metadata and values are materialized in the Zarr store. |

## Known limitations

- Native Rust read runtime is deferred; the current Python API includes a parity-oriented read shim.
- Compression codecs are deferred in v1 (raw chunk writes only).
- Sub-byte and experimental dtypes such as FP4/FP8 are not represented as native Zarr dtypes.
- CUDA parity lanes are scaffolded but not yet mandatory in CI.

## Compatibility

- Minimum supported Rust version (MSRV): `1.85`.
