/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Chunk writer implementations for local filesystem output.

#[cfg(test)]
use std::cell::Cell;
#[cfg(any(test, feature = "test-utils"))]
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(any(test, feature = "test-utils"))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::chunk_id::ChunkId;
use crate::core::contracts::{ChunkWriter, ZarrLayoutAdapter};
use crate::core::errors::SyncWriteError;
use crate::core::types::{FsyncPolicy, TupleChunkKey};
#[cfg(any(test, feature = "test-utils"))]
use crate::zarr::zarr_layout::DefaultZarrLayoutAdapter;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
std::thread_local! {
    static TEMP_FILE_SYNC_CALLS: Cell<usize> = const { Cell::new(0) };
    static PARENT_DIR_SYNC_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Compile-safe writer scaffold.
///
/// Provides deterministic `ChunkId` to path/key rendering helpers while deferring actual
/// filesystem mutation to later implementation stages.
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub struct NoopChunkWriter {
    dataset_root: PathBuf,
    layout_adapter: Arc<dyn ZarrLayoutAdapter>,
    rendered_cache: Mutex<HashMap<ChunkId, String>>,
}

#[cfg(any(test, feature = "test-utils"))]
impl Default for NoopChunkWriter {
    fn default() -> Self {
        Self {
            dataset_root: PathBuf::new(),
            layout_adapter: Arc::new(DefaultZarrLayoutAdapter::v2_default()),
            rendered_cache: Mutex::new(HashMap::new()),
        }
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl NoopChunkWriter {
    /// Create a new chunk writer with the given root and layout adapter.
    #[must_use]
    pub fn new(dataset_root: PathBuf, layout_adapter: Arc<dyn ZarrLayoutAdapter>) -> Self {
        Self {
            dataset_root,
            layout_adapter,
            ..Self::default()
        }
    }

    /// Render the chunk key string for the given `ChunkId`.
    pub fn render_chunk_key(
        &self,
        array_name: &str,
        chunk_id: &ChunkId,
    ) -> Result<String, SyncWriteError> {
        let spec = self.layout_adapter.chunk_path_for(array_name, chunk_id)?;
        Ok(spec.relative_path)
    }

    /// Render the full filesystem path for the given `ChunkId`.
    pub fn render_chunk_path(
        &self,
        array_name: &str,
        chunk_id: &ChunkId,
    ) -> Result<PathBuf, SyncWriteError> {
        Ok(self
            .dataset_root
            .join(self.render_chunk_key(array_name, chunk_id)?))
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl ChunkWriter for NoopChunkWriter {
    fn write_chunk_by_id(
        &self,
        array_name: &str,
        chunk_id: &ChunkId,
        _bytes: &[u8],
    ) -> Result<(), SyncWriteError> {
        let rendered_path = self.render_chunk_path(array_name, chunk_id)?;
        let mut cache =
            self.rendered_cache
                .lock()
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "writer cache lock poisoned".to_string(),
                })?;
        cache.insert(*chunk_id, rendered_path.display().to_string());
        Ok(())
    }

    fn write_chunk_by_tuple_key(
        &self,
        array_name: &str,
        tuple_key: &TupleChunkKey,
        _bytes: &[u8],
    ) -> Result<(), SyncWriteError> {
        let spec = self
            .layout_adapter
            .chunk_path_for_tuple_key(array_name, tuple_key)?;
        // NoopChunkWriter doesn't actually write; just cache the rendered path
        // under a synthetic ChunkId for diagnostic inspection.
        let _ = spec.relative_path;
        Ok(())
    }
}

/// Local-filesystem chunk writer used by the Earth2Studio-facing backend path.
///
/// Writes are performed via temp-file + atomic rename in the destination directory.
pub struct LocalFsChunkWriter {
    dataset_root: PathBuf,
    layout_adapter: Arc<dyn ZarrLayoutAdapter>,
    fsync_policy: FsyncPolicy,
}

impl LocalFsChunkWriter {
    /// Create a new local-filesystem writer rooted at `dataset_root`.
    #[must_use]
    pub fn new(dataset_root: PathBuf, layout_adapter: Arc<dyn ZarrLayoutAdapter>) -> Self {
        Self {
            dataset_root,
            layout_adapter,
            fsync_policy: FsyncPolicy::Always,
        }
    }

    /// Create a writer with an explicit fsync policy.
    #[must_use]
    pub fn with_fsync_policy(
        dataset_root: PathBuf,
        layout_adapter: Arc<dyn ZarrLayoutAdapter>,
        fsync_policy: FsyncPolicy,
    ) -> Self {
        Self {
            dataset_root,
            layout_adapter,
            fsync_policy,
        }
    }

    fn render_chunk_path(
        &self,
        array_name: &str,
        chunk_id: &ChunkId,
    ) -> Result<PathBuf, SyncWriteError> {
        if self.dataset_root.as_os_str().is_empty() {
            return Err(SyncWriteError::io_failed(
                "dataset_root cannot be empty for local chunk writer",
            ));
        }
        let spec = self.layout_adapter.chunk_path_for(array_name, chunk_id)?;
        Ok(self.dataset_root.join(spec.relative_path))
    }

    fn temp_path_for(final_path: &std::path::Path) -> Result<PathBuf, SyncWriteError> {
        let parent = final_path.parent().ok_or_else(|| {
            SyncWriteError::io_failed(format!(
                "chunk path has no parent: {}",
                final_path.display()
            ))
        })?;
        let file_name = final_path
            .file_name()
            .ok_or_else(|| {
                SyncWriteError::io_failed(format!(
                    "chunk path has no file name: {}",
                    final_path.display()
                ))
            })?
            .to_string_lossy();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| {
                SyncWriteError::io_failed_with_cause(
                    format!("failed to derive temp-file nonce: {e}"),
                    e,
                )
            })?
            .as_nanos();
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(parent.join(format!(".{file_name}.tmp-{nonce}-{counter}")))
    }
}

impl ChunkWriter for LocalFsChunkWriter {
    fn write_chunk_by_id(
        &self,
        array_name: &str,
        chunk_id: &ChunkId,
        bytes: &[u8],
    ) -> Result<(), SyncWriteError> {
        let final_path = self.render_chunk_path(array_name, chunk_id)?;
        Self::atomic_write_to_path(&final_path, bytes, self.fsync_policy)
    }

    fn write_chunk_by_tuple_key(
        &self,
        array_name: &str,
        tuple_key: &TupleChunkKey,
        bytes: &[u8],
    ) -> Result<(), SyncWriteError> {
        if self.dataset_root.as_os_str().is_empty() {
            return Err(SyncWriteError::io_failed(
                "dataset_root cannot be empty for local chunk writer",
            ));
        }
        let spec = self
            .layout_adapter
            .chunk_path_for_tuple_key(array_name, tuple_key)?;
        let final_path = self.dataset_root.join(spec.relative_path);
        Self::atomic_write_to_path(&final_path, bytes, self.fsync_policy)
    }
}

impl LocalFsChunkWriter {
    fn remove_temp_file_best_effort(tmp_path: &std::path::Path) {
        let _ = std::fs::remove_file(tmp_path);
    }

    fn sync_temp_file(file: &std::fs::File, path: &std::path::Path) -> Result<(), SyncWriteError> {
        #[cfg(test)]
        TEMP_FILE_SYNC_CALLS.with(|calls| {
            calls.set(calls.get().saturating_add(1));
        });
        file.sync_all().map_err(|e| {
            SyncWriteError::io_failed_with_cause(
                format!("failed syncing temp chunk file '{}': {e}", path.display()),
                e,
            )
        })
    }

    fn sync_parent_dir_best_effort(parent: &std::path::Path) {
        #[cfg(test)]
        PARENT_DIR_SYNC_CALLS.with(|calls| {
            calls.set(calls.get().saturating_add(1));
        });
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }

    #[cfg(test)]
    fn debug_sync_counters() -> (usize, usize) {
        (
            TEMP_FILE_SYNC_CALLS.with(|calls| calls.get()),
            PARENT_DIR_SYNC_CALLS.with(|calls| calls.get()),
        )
    }

    #[cfg(test)]
    fn debug_reset_sync_counters() {
        TEMP_FILE_SYNC_CALLS.with(|calls| calls.set(0));
        PARENT_DIR_SYNC_CALLS.with(|calls| calls.set(0));
    }

    /// Atomically write `bytes` to `final_path` via temp-file + rename.
    fn atomic_write_to_path(
        final_path: &std::path::Path,
        bytes: &[u8],
        fsync_policy: FsyncPolicy,
    ) -> Result<(), SyncWriteError> {
        let parent = final_path.parent().ok_or_else(|| {
            SyncWriteError::io_failed(format!(
                "chunk path has no parent: {}",
                final_path.display()
            ))
        })?;
        std::fs::create_dir_all(parent).map_err(|e| {
            SyncWriteError::io_failed_with_cause(
                format!(
                    "failed creating chunk directory '{}': {e}",
                    parent.display()
                ),
                e,
            )
        })?;

        let tmp_path = Self::temp_path_for(final_path)?;
        {
            let mut file = std::fs::File::create(&tmp_path).map_err(|e| {
                SyncWriteError::io_failed_with_cause(
                    format!(
                        "failed creating temp chunk file '{}': {e}",
                        tmp_path.display()
                    ),
                    e,
                )
            })?;
            if let Err(e) = file.write_all(bytes) {
                drop(file);
                Self::remove_temp_file_best_effort(&tmp_path);
                return Err(SyncWriteError::io_failed_with_cause(
                    format!(
                        "failed writing temp chunk file '{}': {e}",
                        tmp_path.display()
                    ),
                    e,
                ));
            }
            if fsync_policy == FsyncPolicy::Always {
                if let Err(err) = Self::sync_temp_file(&file, &tmp_path) {
                    drop(file);
                    Self::remove_temp_file_best_effort(&tmp_path);
                    return Err(err);
                }
            }
            drop(file);
        }

        if let Err(e) = std::fs::rename(&tmp_path, final_path) {
            Self::remove_temp_file_best_effort(&tmp_path);
            return Err(SyncWriteError::io_failed_with_cause(
                format!(
                    "failed renaming chunk temp file '{}' -> '{}': {e}",
                    tmp_path.display(),
                    final_path.display()
                ),
                e,
            ));
        }
        if fsync_policy == FsyncPolicy::Always {
            Self::sync_parent_dir_best_effort(parent);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    #[cfg(unix)]
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::chunk_id::ChunkId;
    use crate::core::contracts::ChunkWriter;
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{
        ChunkKeyEncoding, ChunkKeySeparator, TupleChunkKey, ZarrFormat, ZarrTargetConfig,
    };
    use crate::zarr::zarr_layout::DefaultZarrLayoutAdapter;

    use super::LocalFsChunkWriter;
    use crate::core::types::FsyncPolicy;

    fn unique_dataset_root(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock moved backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{nanos}.zarr"))
    }

    #[test]
    fn local_fs_writer_rejects_empty_dataset_root_for_chunk_id_writes() {
        let writer = LocalFsChunkWriter::new(
            PathBuf::new(),
            Arc::new(DefaultZarrLayoutAdapter::v2_default()),
        );
        let err = writer
            .write_chunk_by_id(
                "temperature",
                &ChunkId::new(1, 0),
                &[1_u8, 2_u8, 3_u8, 4_u8],
            )
            .expect_err("empty dataset_root should be rejected");
        assert!(matches!(
            err,
            SyncWriteError::IoFailed { ref message, .. }
            if message.contains("dataset_root cannot be empty")
        ));
    }

    #[test]
    fn local_fs_writer_rejects_empty_dataset_root_for_tuple_key_writes() {
        let writer = LocalFsChunkWriter::new(
            PathBuf::new(),
            Arc::new(DefaultZarrLayoutAdapter::v2_default()),
        );
        let err = writer
            .write_chunk_by_tuple_key(
                "temperature",
                &TupleChunkKey::new(vec![0, 0]),
                &[1_u8, 2_u8, 3_u8, 4_u8],
            )
            .expect_err("empty dataset_root should be rejected");
        assert!(matches!(
            err,
            SyncWriteError::IoFailed { ref message, .. }
            if message.contains("dataset_root cannot be empty")
        ));
    }

    #[test]
    fn local_fs_writer_temp_path_for_rejects_path_without_file_name() {
        let err = LocalFsChunkWriter::temp_path_for(Path::new("/"))
            .expect_err("path without terminal file name should be rejected");
        assert!(matches!(
            err,
            SyncWriteError::IoFailed { ref message, .. }
            if message.contains("no parent")
        ));
    }

    #[test]
    fn local_fs_writer_atomic_write_rejects_chunk_path_without_parent() {
        let err = LocalFsChunkWriter::atomic_write_to_path(
            Path::new(""),
            &[1_u8, 2_u8],
            FsyncPolicy::Always,
        )
        .expect_err("path without parent should be rejected");
        assert!(matches!(
            err,
            SyncWriteError::IoFailed { ref message, .. }
            if message.contains("no parent")
        ));
    }

    #[test]
    fn local_fs_writer_sync_parent_dir_best_effort_tolerates_open_failure() {
        LocalFsChunkWriter::debug_reset_sync_counters();
        let (_, before) = LocalFsChunkWriter::debug_sync_counters();
        LocalFsChunkWriter::sync_parent_dir_best_effort(Path::new(
            "/definitely/nonexistent/e2s_writer_parent_sync",
        ));
        let (_, after) = LocalFsChunkWriter::debug_sync_counters();
        assert_eq!(after, before + 1, "sync attempt counter should increment");
    }

    #[test]
    fn local_fs_writer_sync_counters_are_thread_local() {
        LocalFsChunkWriter::debug_reset_sync_counters();
        let (_, main_before) = LocalFsChunkWriter::debug_sync_counters();

        let worker_after = std::thread::spawn(|| {
            LocalFsChunkWriter::debug_reset_sync_counters();
            LocalFsChunkWriter::sync_parent_dir_best_effort(Path::new(
                "/definitely/nonexistent/e2s_writer_parent_sync_thread_local",
            ));
            let (_, worker_after) = LocalFsChunkWriter::debug_sync_counters();
            worker_after
        })
        .join()
        .expect("worker thread should finish");

        let (_, main_after) = LocalFsChunkWriter::debug_sync_counters();
        assert_eq!(
            worker_after, 1,
            "worker thread should observe exactly one parent-dir sync attempt"
        );
        assert_eq!(
            main_after, main_before,
            "worker thread must not mutate main-thread sync counters"
        );
    }

    #[test]
    fn local_fs_chunk_writer_persists_bytes_to_rendered_chunk_path() {
        let dataset_root = unique_dataset_root("e2s_writer_test");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let chunk_id = ChunkId::new(3, 11);
        let payload = vec![1_u8, 2, 3, 4, 5];

        writer
            .write_chunk_by_id("temperature", &chunk_id, &payload)
            .expect("chunk write should succeed");

        let expected_path = dataset_root.join("temperature").join("11");
        let actual = std::fs::read(&expected_path).expect("chunk file should exist");
        assert_eq!(actual, payload);
    }

    #[test]
    fn local_fs_chunk_writer_respects_v3_slash_layout() {
        let dataset_root = unique_dataset_root("e2s_writer_v3_test");
        let adapter = Arc::new(
            DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
                zarr_format: ZarrFormat::V3,
                chunk_key_encoding: ChunkKeyEncoding::Default,
                chunk_key_separator: ChunkKeySeparator::Slash,
            })
            .expect("valid v3 target"),
        );
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let chunk_id = ChunkId::new(7, 21);

        writer
            .write_chunk_by_id("tcwv", &chunk_id, &[9_u8, 8_u8])
            .expect("chunk write should succeed");

        let expected_path = dataset_root.join("tcwv").join("c").join("21");
        let actual = std::fs::read(&expected_path).expect("chunk file should exist");
        assert_eq!(actual, vec![9_u8, 8_u8]);
    }

    // ── ck8: tuple-key writer tests ──────────────────────────────────────

    #[test]
    fn local_fs_writer_writes_chunk_to_v2_dot_tuple_key_path() {
        let dataset_root = unique_dataset_root("e2s_writer_tuple_v2_dot");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![0, 4, 0, 0]);
        let payload = vec![10_u8, 20, 30, 40];

        writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &payload)
            .expect("tuple-key chunk write should succeed");

        // V2 dot: temperature/0.4.0.0
        let expected_path = dataset_root.join("temperature").join("0.4.0.0");
        let actual = std::fs::read(&expected_path)
            .unwrap_or_else(|_| panic!("chunk file should exist at {}", expected_path.display()));
        assert_eq!(actual, payload, "chunk bytes should match");
    }

    #[test]
    fn local_fs_writer_writes_chunk_to_v3_slash_tuple_key_path() {
        let dataset_root = unique_dataset_root("e2s_writer_tuple_v3");
        let adapter = Arc::new(
            DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
                zarr_format: ZarrFormat::V3,
                chunk_key_encoding: ChunkKeyEncoding::Default,
                chunk_key_separator: ChunkKeySeparator::Slash,
            })
            .expect("valid v3 target"),
        );
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![0, 4, 0, 0]);
        let payload = vec![11_u8, 22, 33, 44];

        writer
            .write_chunk_by_tuple_key("tcwv", &tuple_key, &payload)
            .expect("tuple-key chunk write should succeed");

        // V3: tcwv/c/0/4/0/0
        let expected_path = dataset_root
            .join("tcwv")
            .join("c")
            .join("0")
            .join("4")
            .join("0")
            .join("0");
        let actual = std::fs::read(&expected_path)
            .unwrap_or_else(|_| panic!("chunk file should exist at {}", expected_path.display()));
        assert_eq!(actual, payload, "chunk bytes should match");
    }

    #[test]
    fn local_fs_writer_single_dim_tuple_key() {
        let dataset_root = unique_dataset_root("e2s_writer_tuple_1d");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![5]);
        let payload = vec![99_u8];

        writer
            .write_chunk_by_tuple_key("humidity", &tuple_key, &payload)
            .expect("single-dim tuple-key write should succeed");

        let expected_path = dataset_root.join("humidity").join("5");
        let actual = std::fs::read(&expected_path).expect("chunk file should exist");
        assert_eq!(actual, payload);
    }

    #[test]
    fn local_fs_writer_v2_slash_tuple_key_creates_nested_dirs() {
        let dataset_root = unique_dataset_root("e2s_writer_tuple_v2_slash");
        let adapter = Arc::new(
            DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
                zarr_format: ZarrFormat::V2,
                chunk_key_encoding: ChunkKeyEncoding::V2,
                chunk_key_separator: ChunkKeySeparator::Slash,
            })
            .expect("valid v2 slash target"),
        );
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![1, 3, 0, 0]);
        let payload = vec![55_u8, 66, 77, 88];

        writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &payload)
            .expect("v2 slash tuple-key write should succeed");

        // V2 slash: temperature/1/3/0/0  (nested directories)
        let expected_path = dataset_root
            .join("temperature")
            .join("1")
            .join("3")
            .join("0")
            .join("0");
        let actual = std::fs::read(&expected_path)
            .unwrap_or_else(|_| panic!("chunk file should exist at {}", expected_path.display()));
        assert_eq!(actual, payload, "chunk bytes should match");
    }

    #[test]
    fn local_fs_writer_v3_single_dim_tuple_key() {
        let dataset_root = unique_dataset_root("e2s_writer_tuple_v3_1d");
        let adapter = Arc::new(
            DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
                zarr_format: ZarrFormat::V3,
                chunk_key_encoding: ChunkKeyEncoding::Default,
                chunk_key_separator: ChunkKeySeparator::Slash,
            })
            .expect("valid v3 target"),
        );
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![7]);
        let payload = vec![42_u8];

        writer
            .write_chunk_by_tuple_key("tcwv", &tuple_key, &payload)
            .expect("v3 single-dim tuple-key write should succeed");

        // V3: tcwv/c/7
        let expected_path = dataset_root.join("tcwv").join("c").join("7");
        let actual = std::fs::read(&expected_path)
            .unwrap_or_else(|_| panic!("chunk file should exist at {}", expected_path.display()));
        assert_eq!(actual, payload);
    }

    // ── Multi-chunk end-to-end: write several chunks, verify all on disk ──

    #[test]
    fn local_fs_writer_multi_chunk_v2_dot() {
        let dataset_root = unique_dataset_root("e2s_writer_multi_v2_dot");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);

        // Simulate 4 chunks for array "t2m" with dimensions [time(2), lead_time(2)].
        let chunks = vec![
            (vec![0_usize, 0], vec![10_u8, 20]),
            (vec![0, 1], vec![30, 40]),
            (vec![1, 0], vec![50, 60]),
            (vec![1, 1], vec![70, 80]),
        ];

        for (indices, payload) in &chunks {
            let tuple_key = TupleChunkKey::new(indices.clone());
            writer
                .write_chunk_by_tuple_key("t2m", &tuple_key, payload)
                .expect("chunk write should succeed");
        }

        // Verify all 4 chunk files exist with correct bytes.
        let expected_files = ["0.0", "0.1", "1.0", "1.1"];
        for (i, file_name) in expected_files.iter().enumerate() {
            let path = dataset_root.join("t2m").join(file_name);
            let actual = std::fs::read(&path)
                .unwrap_or_else(|_| panic!("chunk {file_name} should exist at {}", path.display()));
            assert_eq!(actual, chunks[i].1, "chunk {file_name} bytes mismatch");
        }
    }

    #[test]
    fn local_fs_writer_multi_chunk_v3_slash() {
        let dataset_root = unique_dataset_root("e2s_writer_multi_v3");
        let adapter = Arc::new(
            DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
                zarr_format: ZarrFormat::V3,
                chunk_key_encoding: ChunkKeyEncoding::Default,
                chunk_key_separator: ChunkKeySeparator::Slash,
            })
            .expect("valid v3 target"),
        );
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);

        // Simulate 4 chunks for array "t2m" with dimensions [time(2), lead_time(2)].
        let chunks: Vec<(Vec<usize>, Vec<u8>)> = vec![
            (vec![0, 0], vec![10, 20]),
            (vec![0, 1], vec![30, 40]),
            (vec![1, 0], vec![50, 60]),
            (vec![1, 1], vec![70, 80]),
        ];

        for (indices, payload) in &chunks {
            let tuple_key = TupleChunkKey::new(indices.clone());
            writer
                .write_chunk_by_tuple_key("t2m", &tuple_key, payload)
                .expect("chunk write should succeed");
        }

        // V3 chunks live under t2m/c/<idx0>/<idx1>.
        let expected_paths: Vec<std::path::PathBuf> = vec![
            dataset_root.join("t2m/c/0/0"),
            dataset_root.join("t2m/c/0/1"),
            dataset_root.join("t2m/c/1/0"),
            dataset_root.join("t2m/c/1/1"),
        ];
        for (i, path) in expected_paths.iter().enumerate() {
            let actual = std::fs::read(path)
                .unwrap_or_else(|_| panic!("v3 chunk should exist at {}", path.display()));
            assert_eq!(
                actual,
                chunks[i].1,
                "v3 chunk {} bytes mismatch",
                path.display()
            );
        }
    }

    #[test]
    fn local_fs_writer_multi_chunk_v2_slash() {
        let dataset_root = unique_dataset_root("e2s_writer_multi_v2_slash");
        let adapter = Arc::new(
            DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
                zarr_format: ZarrFormat::V2,
                chunk_key_encoding: ChunkKeyEncoding::V2,
                chunk_key_separator: ChunkKeySeparator::Slash,
            })
            .expect("valid v2 slash target"),
        );
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);

        // Simulate 4 chunks for "temperature" with 3 dims: [time(2), lt(2), 1].
        let chunks: Vec<(Vec<usize>, Vec<u8>)> = vec![
            (vec![0, 0, 0], vec![1, 2, 3, 4]),
            (vec![0, 1, 0], vec![5, 6, 7, 8]),
            (vec![1, 0, 0], vec![9, 10, 11, 12]),
            (vec![1, 1, 0], vec![13, 14, 15, 16]),
        ];

        for (indices, payload) in &chunks {
            let tuple_key = TupleChunkKey::new(indices.clone());
            writer
                .write_chunk_by_tuple_key("temperature", &tuple_key, payload)
                .expect("v2 slash chunk write should succeed");
        }

        // V2 slash: temperature/<idx0>/<idx1>/<idx2> (nested directories).
        let expected_paths: Vec<std::path::PathBuf> = vec![
            dataset_root.join("temperature/0/0/0"),
            dataset_root.join("temperature/0/1/0"),
            dataset_root.join("temperature/1/0/0"),
            dataset_root.join("temperature/1/1/0"),
        ];
        for (i, path) in expected_paths.iter().enumerate() {
            let actual = std::fs::read(path)
                .unwrap_or_else(|_| panic!("v2 slash chunk should exist at {}", path.display()));
            assert_eq!(
                actual,
                chunks[i].1,
                "v2 slash chunk {} bytes mismatch",
                path.display()
            );
        }
    }

    #[test]
    fn local_fs_writer_atomic_write_leaves_no_temp_files_on_success() {
        let dataset_root = unique_dataset_root("e2s_writer_atomic_tmp_cleanup");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![2, 1]);

        writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &[1_u8, 2, 3, 4])
            .expect("atomic tuple-key write should succeed");

        let chunk_dir = dataset_root.join("temperature");
        let entries: Vec<String> = std::fs::read_dir(&chunk_dir)
            .expect("array chunk directory should exist")
            .map(|entry| {
                entry
                    .expect("directory entry should be readable")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        assert!(
            entries.iter().any(|name| name == "2.1"),
            "expected final chunk file, found entries: {entries:?}"
        );
        assert!(
            entries.iter().all(|name| !name.contains(".tmp-")),
            "successful atomic write should not leave temp files behind: {entries:?}"
        );
    }

    #[test]
    fn local_fs_writer_atomic_write_invokes_durability_sync_steps() {
        let dataset_root = unique_dataset_root("e2s_writer_atomic_sync_steps");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root, adapter);
        let tuple_key = TupleChunkKey::new(vec![1, 0]);

        LocalFsChunkWriter::debug_reset_sync_counters();
        let (temp_sync_before, dir_sync_before) = LocalFsChunkWriter::debug_sync_counters();
        writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &[5_u8, 6, 7, 8])
            .expect("atomic tuple-key write should succeed");
        let (temp_sync_after, dir_sync_after) = LocalFsChunkWriter::debug_sync_counters();

        assert!(
            temp_sync_after > temp_sync_before,
            "durability path should fsync temp file before rename"
        );
        assert!(
            dir_sync_after > dir_sync_before,
            "durability path should fsync parent directory after rename"
        );
    }

    #[test]
    fn local_fs_writer_failed_rename_keeps_existing_destination_unchanged() {
        let dataset_root = unique_dataset_root("e2s_writer_atomic_rename_failure");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![0, 0]);
        let destination = dataset_root.join("temperature").join("0.0");

        std::fs::create_dir_all(&destination).expect("seed existing destination directory");

        let err = writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &[9_u8, 8, 7])
            .expect_err("rename into existing directory should fail");
        let message = err.to_string();
        assert!(
            message.contains("failed renaming chunk temp file"),
            "expected rename failure, got: {message}"
        );

        let metadata = std::fs::metadata(&destination).expect("destination should still exist");
        assert!(
            metadata.is_dir(),
            "failed rename must not clobber existing destination path"
        );
    }

    #[test]
    fn local_fs_writer_failed_rename_cleans_up_temp_files() {
        let dataset_root = unique_dataset_root("e2s_writer_atomic_rename_cleanup");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![0, 0]);
        let destination = dataset_root.join("temperature").join("0.0");

        std::fs::create_dir_all(&destination).expect("seed existing destination directory");

        writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &[9_u8, 8, 7])
            .expect_err("rename into existing directory should fail");

        let chunk_dir = dataset_root.join("temperature");
        let entries: Vec<String> = std::fs::read_dir(&chunk_dir)
            .expect("chunk directory should exist")
            .map(|entry| {
                entry
                    .expect("directory entry should be readable")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(
            entries.iter().all(|name| !name.contains(".tmp-")),
            "failed write should clean up temp files best-effort, found: {entries:?}"
        );
    }

    #[test]
    fn local_fs_writer_failed_rename_surfaces_io_source_error() {
        let dataset_root = unique_dataset_root("e2s_writer_atomic_rename_source");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![0, 0]);
        let destination = dataset_root.join("temperature").join("0.0");

        std::fs::create_dir_all(&destination).expect("seed existing destination directory");

        let err = writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &[9_u8, 8, 7])
            .expect_err("rename into existing directory should fail");

        let source = err.source();
        assert!(
            source.is_some(),
            "expected IoFailed to preserve underlying source chain"
        );
        assert!(
            !source
                .map(std::string::ToString::to_string)
                .unwrap_or_default()
                .is_empty()
        );
    }

    #[test]
    fn local_fs_writer_temp_file_name_has_secondary_uniquifier_component() {
        let final_path = std::env::temp_dir()
            .join("e2s_writer_tmp_name_shape")
            .join("0.0");
        let temp_path = LocalFsChunkWriter::temp_path_for(&final_path)
            .expect("temp path generation should succeed");
        let temp_name = temp_path
            .file_name()
            .expect("temp path should have a file name")
            .to_string_lossy();

        let suffix = temp_name
            .split_once(".tmp-")
            .map(|(_, tail)| tail)
            .expect("temp name should contain .tmp- delimiter");
        assert!(
            suffix.contains('-'),
            "temp suffix should include secondary uniqueness component: {temp_name}"
        );
    }

    #[test]
    fn local_fs_writer_temp_path_generation_is_collision_free_under_concurrency() {
        use std::collections::HashSet;

        let final_path = std::env::temp_dir()
            .join("e2s_writer_tmp_name_collision_check")
            .join("0.0");
        let thread_count = 12_usize;
        let iterations_per_thread = 500_usize;

        let mut handles = Vec::with_capacity(thread_count);
        for _ in 0..thread_count {
            let final_path = final_path.clone();
            handles.push(std::thread::spawn(move || {
                let mut out = Vec::with_capacity(iterations_per_thread);
                for _ in 0..iterations_per_thread {
                    let temp_path = LocalFsChunkWriter::temp_path_for(&final_path)
                        .expect("temp path generation should succeed");
                    out.push(temp_path);
                }
                out
            }));
        }

        let mut all = HashSet::new();
        for handle in handles {
            for temp_path in handle.join().expect("worker thread should not panic") {
                let inserted = all.insert(temp_path);
                assert!(inserted, "duplicate temp path observed under concurrency");
            }
        }

        assert_eq!(all.len(), thread_count * iterations_per_thread);
    }

    #[cfg(unix)]
    #[test]
    fn local_fs_writer_atomic_replace_never_exposes_partial_payloads() {
        let dataset_root = unique_dataset_root("e2s_writer_atomic_replace_visibility");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::new(dataset_root.clone(), adapter);
        let tuple_key = TupleChunkKey::new(vec![0, 0]);
        let chunk_path = dataset_root.join("temperature").join("0.0");

        let payload_len = 256 * 1024;
        let payload_a = vec![0xAA_u8; payload_len];
        let payload_b = vec![0x55_u8; payload_len];

        writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &payload_a)
            .expect("initial seed write should succeed");

        let keep_reading = Arc::new(AtomicBool::new(true));
        let reader_flag = Arc::clone(&keep_reading);
        let reader_path = chunk_path.clone();
        let reader = std::thread::spawn(move || {
            let mut reads = 0_usize;
            while reader_flag.load(Ordering::Acquire) {
                let bytes = std::fs::read(&reader_path).expect("chunk should remain readable");
                reads += 1;
                let all_a = bytes.iter().all(|b| *b == 0xAA);
                let all_b = bytes.iter().all(|b| *b == 0x55);
                assert_eq!(
                    bytes.len(),
                    payload_len,
                    "reader observed a truncated or oversized chunk payload"
                );
                assert!(
                    all_a || all_b,
                    "reader observed mixed bytes, expected only full old/new payload"
                );
                std::thread::yield_now();
            }
            reads
        });

        for i in 0..150 {
            let payload = if i % 2 == 0 { &payload_b } else { &payload_a };
            writer
                .write_chunk_by_tuple_key("temperature", &tuple_key, payload)
                .expect("overwrite write should succeed");
        }

        keep_reading.store(false, Ordering::Release);
        let observed_reads = reader.join().expect("reader thread should not panic");
        assert!(
            observed_reads > 0,
            "reader should observe at least one payload snapshot"
        );

        let final_bytes = std::fs::read(&chunk_path).expect("final chunk should exist");
        let final_all_a = final_bytes.iter().all(|b| *b == 0xAA);
        let final_all_b = final_bytes.iter().all(|b| *b == 0x55);
        assert_eq!(final_bytes.len(), payload_len);
        assert!(
            final_all_a || final_all_b,
            "final payload should be one fully written version"
        );
    }

    #[test]
    fn local_fs_writer_fsync_never_writes_correct_bytes_via_chunk_id() {
        let dataset_root = unique_dataset_root("e2s_writer_fsync_never_id");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::with_fsync_policy(
            dataset_root.clone(),
            adapter,
            FsyncPolicy::Never,
        );
        let payload = vec![10_u8, 20, 30, 40];

        writer
            .write_chunk_by_id("temperature", &ChunkId::new(0, 0), &payload)
            .expect("chunk write with FsyncPolicy::Never should succeed");

        let written = std::fs::read(dataset_root.join("temperature").join("0"))
            .expect("chunk file should exist on disk");
        assert_eq!(written, payload, "written bytes must match input payload");
    }

    #[test]
    fn local_fs_writer_fsync_never_writes_correct_bytes_via_tuple_key() {
        let dataset_root = unique_dataset_root("e2s_writer_fsync_never_tuple");
        let adapter = Arc::new(DefaultZarrLayoutAdapter::v2_default());
        let writer = LocalFsChunkWriter::with_fsync_policy(
            dataset_root.clone(),
            adapter,
            FsyncPolicy::Never,
        );
        let payload = vec![5_u8, 6, 7, 8];
        let tuple_key = TupleChunkKey::new(vec![1, 0]);

        writer
            .write_chunk_by_tuple_key("temperature", &tuple_key, &payload)
            .expect("tuple-key write with FsyncPolicy::Never should succeed");

        let written = std::fs::read(dataset_root.join("temperature").join("1.0"))
            .expect("chunk file should exist on disk");
        assert_eq!(written, payload, "written bytes must match input payload");
    }
}
