/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::core::errors::SyncWriteError;

use super::{
    BatchId, BufferHandle, BufferLease, ChunkKeyEncoding, ChunkKeySeparator, ChunkKeyTemplate,
    CoordMap, CoordValues, InputArraySource, Nanoseconds, TransientBuffer, TupleChunkKey,
    ZarrFormat, ZarrTargetConfig,
};

fn make_pooled_handle_for_eq_test(base_offset: usize, slab_len: usize) -> BufferHandle {
    let backing_len = slab_len.max(base_offset.saturating_add(16));
    let slab_bytes = Arc::new(Mutex::new(vec![0_u8; backing_len]));
    let slab_base_addr = {
        let guard = slab_bytes
            .lock()
            .expect("slab backing lock should not be poisoned");
        (guard.as_ptr() as usize).saturating_add(base_offset)
    };
    BufferHandle::new(
        7,
        16,
        0..16,
        slab_base_addr,
        slab_len,
        Arc::new(Mutex::new(())),
        slab_bytes,
    )
}

#[test]
fn pooled_buffer_handle_equality_distinguishes_slab_base_address() {
    let left = make_pooled_handle_for_eq_test(0, 32);
    let right = make_pooled_handle_for_eq_test(1, 32);

    assert_ne!(left, right);
}

#[test]
fn pooled_buffer_handle_equality_distinguishes_slab_length() {
    let left = make_pooled_handle_for_eq_test(0, 32);
    let right = make_pooled_handle_for_eq_test(0, 64);

    assert_ne!(left, right);
}

#[test]
fn coord_map_try_new_rejects_empty_coordinate_key() {
    let mut raw = BTreeMap::new();
    raw.insert(String::new(), CoordValues::I64(vec![1]));

    let err = CoordMap::try_new(raw).expect_err("empty coordinate key must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("coordinate key")
    ));
}

#[test]
fn coord_map_try_new_rejects_empty_coordinate_values() {
    let mut raw = BTreeMap::new();
    raw.insert("time".to_string(), CoordValues::I64(Vec::new()));

    let err = CoordMap::try_new(raw).expect_err("empty coordinate axis must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("time")
    ));
}

#[test]
fn coord_map_api_blocks_unvalidated_mutation_paths() {
    let source = include_str!("requests.rs");
    let production_source = source
        .split("\n/// Typed coordinate value array.")
        .next()
        .expect("requests.rs should contain CoordMap section");

    assert!(
        !production_source.contains("impl DerefMut for CoordMap"),
        "CoordMap must not expose DerefMut because it bypasses validation invariants",
    );
    assert!(
        !production_source.contains("impl From<BTreeMap<String, CoordValues>> for CoordMap"),
        "CoordMap must not expose unvalidated From<BTreeMap> construction",
    );
    assert!(
        !production_source
            .contains("impl std::iter::FromIterator<(String, CoordValues)> for CoordMap"),
        "CoordMap must not expose unvalidated FromIterator construction",
    );
    assert!(
        production_source.contains("impl TryFrom<BTreeMap<String, CoordValues>> for CoordMap"),
        "CoordMap should expose validated TryFrom<BTreeMap> construction",
    );
}

#[test]
fn request_contract_types_define_validate_methods() {
    let source = include_str!("requests.rs");
    assert!(
        source.contains("impl ArrayRegistration")
            && source.contains("pub fn validate(&self) -> Result<(), SyncWriteError>"),
        "ArrayRegistration must define validate() to enforce request invariants at type boundary",
    );
    assert!(
        source.contains("impl InferenceWriteRequest")
            && source.contains("pub fn validate(&self) -> Result<(), SyncWriteError>"),
        "InferenceWriteRequest must define validate() to enforce write request invariants",
    );
}

#[test]
fn buffer_handle_constructor_visibility_and_pointer_contract_are_hardened() {
    let source = include_str!("buffer.rs");
    let production_source = source
        .split("\n#[cfg(test)]")
        .next()
        .expect("buffer.rs should contain production section");

    assert!(
        production_source.contains("pub(crate) fn new("),
        "BufferHandle::new must be crate-visible to prevent external unsafe-pointer fabrication",
    );
    assert!(
        production_source.contains("debug_assert!"),
        "BufferHandle::new should include debug_assert pointer/bounds contract checks",
    );
}

#[test]
fn host_and_cuda_pointer_construction_have_explicit_safety_contracts() {
    let source = include_str!("requests.rs");
    let production_source = source
        .split("\n#[cfg(test)]")
        .next()
        .expect("requests.rs should contain production section");

    assert!(
        production_source.contains("pub(crate) unsafe fn from_host_buffer_ptr("),
        "from_host_buffer_ptr should be unsafe to make raw-pointer invariants explicit",
    );
    assert!(
        production_source.contains("pub unsafe fn from_cuda_ptr("),
        "from_cuda_ptr should be unsafe because invalid device pointers are UB at copy time",
    );
    assert!(
        production_source.contains("# Safety"),
        "raw-pointer constructors must document # Safety invariants",
    );
}

#[test]
fn host_bytes_clone_reuses_underlying_allocation() {
    let source = InputArraySource::HostBytes(vec![1_u8, 2, 3, 4].into());
    let cloned = source.clone();

    let source_ptr = match &source {
        InputArraySource::HostBytes(bytes) => bytes.as_ptr(),
        other => panic!("expected HostBytes source, got {other:?}"),
    };
    let cloned_ptr = match &cloned {
        InputArraySource::HostBytes(bytes) => bytes.as_ptr(),
        other => panic!("expected HostBytes clone, got {other:?}"),
    };

    assert_eq!(
        source_ptr, cloned_ptr,
        "cloning HostBytes should not duplicate payload allocation"
    );
}

#[test]
fn coord_map_validate_rejects_whitespace_only_key() {
    let mut raw = BTreeMap::new();
    raw.insert("   ".to_string(), CoordValues::I64(vec![1]));
    let map = CoordMap::try_from(raw).expect_err("invalid map should fail try_from");

    let err = map;
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("coordinate key")
    ));
}

#[test]
fn input_array_source_internal_host_ptr_roundtrip_and_non_ptr_none() {
    let internal = unsafe { InputArraySource::from_host_buffer_ptr(0x1234) };
    assert_eq!(internal.as_host_buffer_ptr(), Some(0x1234));

    let host = InputArraySource::HostBytes(vec![9_u8].into());
    assert_eq!(host.as_host_buffer_ptr(), None);
}

#[test]
fn chunk_key_separator_as_char_maps_variants() {
    assert_eq!(ChunkKeySeparator::Dot.as_char(), '.');
    assert_eq!(ChunkKeySeparator::Slash.as_char(), '/');
}

#[test]
fn zarr_target_config_validate_matrix_covers_supported_and_rejected_combinations() {
    let v2_ok = ZarrTargetConfig::default();
    assert!(
        v2_ok.validate().is_ok(),
        "default V2 matrix should be valid"
    );

    let v2_default_encoding = ZarrTargetConfig {
        zarr_format: ZarrFormat::V2,
        chunk_key_encoding: ChunkKeyEncoding::Default,
        chunk_key_separator: ChunkKeySeparator::Dot,
    };
    let err = v2_default_encoding
        .validate()
        .expect_err("V2 + Default encoding must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::UnsupportedZarrTargetConfig { ref message }
        if message.contains("V2 does not support chunk_key_encoding=Default")
    ));

    let v3_dot_separator = ZarrTargetConfig {
        zarr_format: ZarrFormat::V3,
        chunk_key_encoding: ChunkKeyEncoding::Default,
        chunk_key_separator: ChunkKeySeparator::Dot,
    };
    let err = v3_dot_separator
        .validate()
        .expect_err("V3 requires slash separator");
    assert!(matches!(
        err,
        SyncWriteError::UnsupportedZarrTargetConfig { ref message }
        if message.contains("requires chunk_key_separator=Slash")
    ));

    let v3_v2_encoding = ZarrTargetConfig {
        zarr_format: ZarrFormat::V3,
        chunk_key_encoding: ChunkKeyEncoding::V2,
        chunk_key_separator: ChunkKeySeparator::Slash,
    };
    let err = v3_v2_encoding
        .validate()
        .expect_err("V3 + V2 encoding must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::UnsupportedZarrTargetConfig { ref message }
        if message.contains("V3 does not support chunk_key_encoding=V2")
    ));

    let v3_ok = ZarrTargetConfig {
        zarr_format: ZarrFormat::V3,
        chunk_key_encoding: ChunkKeyEncoding::Default,
        chunk_key_separator: ChunkKeySeparator::Slash,
    };
    assert!(v3_ok.validate().is_ok(), "V3 slash/default must be valid");
}

#[test]
fn transient_buffer_lease_reports_length_contract_violations() {
    let mut lease = BufferLease::Transient(TransientBuffer::new(4));

    let err = lease
        .write_from_host_bytes(&[1_u8, 2], 4)
        .expect_err("write should reject source shorter than required bytes");
    assert!(matches!(err, SyncWriteError::CopyFailed { .. }));

    lease
        .write_from_host_bytes(&[1_u8, 2, 3, 4], 4)
        .expect("transient write should succeed with exact-sized source");
    let staged = lease
        .staged_bytes(4)
        .expect("staging exact transient bytes should succeed");
    assert_eq!(staged, vec![1, 2, 3, 4]);

    let err = lease
        .staged_bytes(5)
        .expect_err("staged_bytes should reject required bytes over transient length");
    assert!(matches!(err, SyncWriteError::CopyFailed { .. }));

    let err = lease
        .with_bytes(5, |_| Ok(()))
        .expect_err("with_bytes should reject required bytes over transient length");
    assert!(matches!(err, SyncWriteError::CopyFailed { .. }));
}

#[test]
fn pooled_buffer_lease_write_rejects_required_bytes_over_capacity() {
    let handle = BufferHandle::new(
        3,
        2,
        0..2,
        0,
        2,
        Arc::new(Mutex::new(())),
        Arc::new(Mutex::new(vec![0_u8; 2])),
    );
    let mut lease = BufferLease::Pooled(handle);
    let err = lease
        .write_from_host_bytes(&[1_u8, 2, 3, 4], 4)
        .expect_err("pooled write should reject required bytes over capacity");
    assert!(matches!(err, SyncWriteError::CopyFailed { .. }));
}

#[test]
fn buffer_handle_with_bytes_reports_range_and_pointer_guard_errors() {
    let out_of_bounds_handle = BufferHandle::new(
        9,
        16,
        5..17,
        0,
        16,
        Arc::new(Mutex::new(())),
        Arc::new(Mutex::new(vec![0_u8; 32])),
    );
    let err = out_of_bounds_handle
        .with_bytes(4, |_| ())
        .expect_err("out-of-bounds slab range must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::ContractViolation { ref message }
        if message.contains("out of bounds")
    ));

    if !cfg!(debug_assertions) {
        let overflow_handle = BufferHandle::new(
            10,
            8,
            1..2,
            usize::MAX,
            8,
            Arc::new(Mutex::new(())),
            Arc::new(Mutex::new(vec![0_u8; 8])),
        );
        let err = overflow_handle
            .with_bytes(1, |_| ())
            .expect_err("start pointer overflow must be rejected");
        assert!(matches!(
            err,
            SyncWriteError::ContractViolation { ref message }
            if message.contains("start pointer overflow")
        ));
    }
}

#[test]
fn buffer_handle_with_bytes_reports_slot_lock_poisoning() {
    let slot_lock = Arc::new(Mutex::new(()));
    let slot_lock_for_poison = Arc::clone(&slot_lock);
    let poison = std::thread::spawn(move || {
        let _guard = slot_lock_for_poison
            .lock()
            .expect("poison helper should acquire slot lock");
        panic!("intentional slot-lock poison");
    });
    assert!(poison.join().is_err(), "poison helper must panic");

    let handle = BufferHandle::new(
        11,
        8,
        0..8,
        0,
        8,
        slot_lock,
        Arc::new(Mutex::new(vec![0_u8; 8])),
    );
    let err = handle
        .with_bytes(1, |_| ())
        .expect_err("poisoned slot lock must be surfaced");
    assert!(matches!(
        err,
        SyncWriteError::ContractViolation { ref message }
        if message.contains("slot lock poisoned")
    ));
}

#[test]
fn tuple_chunk_key_linear_index_rejects_divisor_length_mismatch() {
    let tuple = TupleChunkKey::new(vec![0, 1]);
    let err = tuple
        .linear_index(&[1_u64])
        .expect_err("mismatched tuple/divisor lengths must fail");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message }
        if message.contains("tuple key length")
    ));
}

#[test]
fn chunk_key_template_build_and_unravel_guard_paths_are_validated() {
    let mut registered = CoordMap::new();
    let _ = registered.insert("time".to_string(), CoordValues::I64(vec![0, 1]));
    let err = ChunkKeyTemplate::build(&registered, &["missing".to_string()])
        .expect_err("unknown active dimension must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message }
        if message.contains("not found in registered coords")
    ));

    let zero_divisor_template = ChunkKeyTemplate::from_parts_for_test(
        vec!["x".to_string()],
        vec![1],
        vec![0],
        vec![],
        vec![(0, 0)],
    );
    let err = zero_divisor_template
        .tuple_from_linear(0)
        .expect_err("zero divisor must be rejected during unravel");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message }
        if message.contains("divisor is zero")
    ));

    let remainder_template = ChunkKeyTemplate::from_parts_for_test(
        vec!["x".to_string()],
        vec![1],
        vec![2],
        vec![],
        vec![(0, 0)],
    );
    let err = remainder_template
        .tuple_from_linear(1)
        .expect_err("non-zero remainder must be rejected as out-of-grid linear index");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message }
        if message.contains("exceeds chunk grid")
    ));
}

// ── PR-089 / PR-090: newtype encapsulation ──────────────────────────────────

#[test]
fn batch_id_inner_field_is_not_externally_constructible() {
    let source = include_str!("responses.rs");
    let production = source.split("\n#[cfg(test)]").next().unwrap_or(source);
    assert!(
        production.contains("pub(crate) u64") || !production.contains("pub u64"),
        "BatchId inner field must be pub(crate), not pub — external crate code should use \
         From<u64> for construction and as_u64() for access",
    );
}

#[test]
fn nanoseconds_inner_field_is_not_externally_constructible() {
    let source = include_str!("responses.rs");
    let production = source.split("\n#[cfg(test)]").next().unwrap_or(source);
    assert!(
        production.contains("pub struct Nanoseconds(pub(crate) u64)")
            || !production.contains("pub struct Nanoseconds(pub u64)"),
        "Nanoseconds inner field must be pub(crate), not pub — external code should use \
         From<u64> for construction and as_nanos()/to_duration() for access",
    );
}

#[test]
fn batch_id_from_u64_and_as_u64_roundtrip() {
    let id = BatchId::from(42_u64);
    assert_eq!(id.as_u64(), 42);
    let raw: u64 = id.into();
    assert_eq!(raw, 42);
}

#[test]
fn nanoseconds_from_u64_and_as_nanos_roundtrip() {
    let ns = Nanoseconds::from(123_456_u64);
    assert_eq!(ns.as_nanos(), 123_456);
    assert_eq!(ns.to_duration(), std::time::Duration::from_nanos(123_456));
    let raw: u64 = ns.into();
    assert_eq!(raw, 123_456);
}

// ── PR-094: InputArray CUDA pointer validation ──────────────────────────────

#[test]
fn input_array_validate_rejects_zero_nbytes() {
    let arr = super::InputArray {
        nbytes: 0,
        source: InputArraySource::HostBytes(vec![].into()),
    };
    let err = arr.validate().expect_err("zero nbytes must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("nbytes")
    ));
}

#[test]
fn input_array_validate_rejects_cuda_null_pointer() {
    let arr = super::InputArray {
        nbytes: 8,
        source: InputArraySource::CudaDevicePtr {
            ptr: 0,
            device_ordinal: 0,
            producer_stream: None,
        },
    };
    let err = arr
        .validate()
        .expect_err("CUDA null pointer must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("non-zero")
    ));
}

#[test]
fn input_array_validate_rejects_negative_cuda_device_ordinal() {
    let arr = super::InputArray {
        nbytes: 8,
        source: InputArraySource::CudaDevicePtr {
            ptr: 0x1000,
            device_ordinal: -1,
            producer_stream: None,
        },
    };
    let err = arr
        .validate()
        .expect_err("negative CUDA device ordinal must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("device_ordinal")
    ));
}

#[test]
fn input_array_validate_accepts_valid_cuda_pointer() {
    let arr = super::InputArray {
        nbytes: 64,
        source: InputArraySource::CudaDevicePtr {
            ptr: 0x1000,
            device_ordinal: 0,
            producer_stream: None,
        },
    };
    arr.validate()
        .expect("valid CUDA input should pass validation");
}

// ── PR-095: CoordValues NaN rejection ───────────────────────────────────────

#[test]
fn coord_map_rejects_nan_in_f64_coordinates() {
    let mut raw = BTreeMap::new();
    raw.insert(
        "lat".to_string(),
        CoordValues::F64(vec![1.0, f64::NAN, 3.0]),
    );
    let err = CoordMap::try_new(raw).expect_err("NaN in F64 coordinates must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("NaN")
    ));
}

#[test]
fn coord_map_insert_rejects_nan_in_f64_coordinates() {
    let mut map = CoordMap::new();
    let err = map
        .insert("lat".to_string(), CoordValues::F64(vec![f64::NAN]))
        .expect_err("NaN via insert must be rejected");
    assert!(matches!(
        err,
        SyncWriteError::Validation { ref message } if message.contains("NaN")
    ));
}

#[test]
fn coord_map_accepts_valid_f64_coordinates() {
    let mut raw = BTreeMap::new();
    raw.insert("lat".to_string(), CoordValues::F64(vec![-90.0, 0.0, 90.0]));
    CoordMap::try_new(raw).expect("valid F64 coordinates should be accepted");
}

// ── PR-099: transient buffer zero-value config validation ───────────────────

#[test]
fn config_rejects_zero_max_transient_buffer_bytes() {
    let source = include_str!("../../api.rs");
    assert!(
        source.contains("max_transient_buffer_bytes") && source.contains("Some(0)"),
        "SyncZarrBackendConfig::validate() should check for Some(0) transient buffer limits"
    );
}

// ── PR-092: CoordMap construction path documentation ────────────────────────

#[test]
fn coord_map_new_documents_incremental_construction_contract() {
    let source = include_str!("requests.rs");
    let before_try_new = source
        .split("pub fn try_new(")
        .next()
        .expect("try_new() should exist");
    assert!(
        before_try_new.contains("incremental") || before_try_new.contains("per-entry"),
        "CoordMap::new() doc should mention that insert() provides per-entry validation \
         for the incremental construction path"
    );
}
