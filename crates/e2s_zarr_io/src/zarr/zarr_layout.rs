/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Zarr v2/v3 layout adapter for chunk path and metadata path rendering.

use crate::core::chunk_id::ChunkId;
use crate::core::contracts::ZarrLayoutAdapter;
use crate::core::errors::SyncWriteError;
use crate::core::types::{
    ChunkPathSpec, MetadataPathSpec, TupleChunkKey, ZarrFormat, ZarrTargetConfig,
};

/// Compile-safe layout adapter for format-specific metadata and chunk path rendering.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DefaultZarrLayoutAdapter {
    target: ZarrTargetConfig,
}

impl DefaultZarrLayoutAdapter {
    /// Construct a new adapter, validating the target configuration.
    pub fn new(target: ZarrTargetConfig) -> Result<Self, SyncWriteError> {
        target.validate()?;
        Ok(Self { target })
    }

    /// Construct the default V2 adapter.
    #[must_use]
    pub fn v2_default() -> Self {
        Self::default()
    }

    /// Returns the zarr target configuration.
    #[must_use]
    pub fn target(&self) -> &ZarrTargetConfig {
        &self.target
    }

    /// Render chunk path using the linear index (legacy path, used by `write_chunk_by_id`).
    fn render_chunk_path(&self, array_name: &str, chunk_id: &ChunkId) -> String {
        match self.target.zarr_format {
            ZarrFormat::V2 => format!("{array_name}/{}", chunk_id.linear_index()),
            ZarrFormat::V3 => format!("{array_name}/c/{}", chunk_id.linear_index()),
        }
    }
}

impl ZarrLayoutAdapter for DefaultZarrLayoutAdapter {
    fn zarr_format(&self) -> ZarrFormat {
        self.target.zarr_format
    }

    fn chunk_path_for(
        &self,
        array_name: &str,
        chunk_id: &ChunkId,
    ) -> Result<ChunkPathSpec, SyncWriteError> {
        Ok(ChunkPathSpec {
            relative_path: self.render_chunk_path(array_name, chunk_id),
        })
    }

    fn chunk_path_for_tuple_key(
        &self,
        array_name: &str,
        tuple_key: &TupleChunkKey,
    ) -> Result<ChunkPathSpec, SyncWriteError> {
        let rendered = match self.target.zarr_format {
            ZarrFormat::V2 => {
                let key = tuple_key.render(self.target.chunk_key_separator.as_char());
                format!("{array_name}/{key}")
            }
            ZarrFormat::V3 => {
                // V3 always uses '/' separator under the `c/` prefix.
                let key = tuple_key.render('/');
                format!("{array_name}/c/{key}")
            }
        };
        Ok(ChunkPathSpec {
            relative_path: rendered,
        })
    }

    fn metadata_paths(&self) -> Result<MetadataPathSpec, SyncWriteError> {
        match self.target.zarr_format {
            ZarrFormat::V2 => Ok(MetadataPathSpec {
                root_paths: vec![
                    ".zgroup".to_string(),
                    ".zattrs".to_string(),
                    ".zmetadata".to_string(),
                ],
                per_array_paths: vec![".zarray".to_string(), ".zattrs".to_string()],
            }),
            ZarrFormat::V3 => Ok(MetadataPathSpec {
                root_paths: vec!["zarr.json".to_string()],
                per_array_paths: vec!["zarr.json".to_string()],
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::chunk_id::ChunkId;
    use crate::core::contracts::ZarrLayoutAdapter;
    use crate::core::types::{
        ChunkKeyEncoding, ChunkKeySeparator, TupleChunkKey, ZarrFormat, ZarrTargetConfig,
    };

    use super::DefaultZarrLayoutAdapter;

    #[test]
    fn v2_layout_renders_chunk_paths_under_array_name_namespace() {
        let layout = DefaultZarrLayoutAdapter::v2_default();
        let path = layout
            .chunk_path_for("temperature", &ChunkId::new(0, 7))
            .expect("path render should succeed");
        assert_eq!(path.relative_path, "temperature/7");
    }

    #[test]
    fn v3_layout_renders_chunk_paths_under_array_name_namespace() {
        let layout = DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
            zarr_format: ZarrFormat::V3,
            chunk_key_encoding: ChunkKeyEncoding::Default,
            chunk_key_separator: ChunkKeySeparator::Slash,
        })
        .expect("valid v3 target");
        let path = layout
            .chunk_path_for("tcwv", &ChunkId::new(1, 9))
            .expect("path render should succeed");
        assert_eq!(path.relative_path, "tcwv/c/9");
    }

    // ── ck8: tuple-key path rendering ────────────────────────────────────

    #[test]
    fn v2_dot_layout_renders_tuple_key_path() {
        let layout = DefaultZarrLayoutAdapter::v2_default();
        let key = TupleChunkKey::new(vec![0, 4, 0, 0]);
        let path = layout
            .chunk_path_for_tuple_key("temperature", &key)
            .expect("tuple-key path render should succeed");
        assert_eq!(
            path.relative_path, "temperature/0.4.0.0",
            "V2 dot separator should produce dot-joined tuple key"
        );
    }

    #[test]
    fn v2_dot_layout_renders_single_dim_tuple_key() {
        let layout = DefaultZarrLayoutAdapter::v2_default();
        let key = TupleChunkKey::new(vec![7]);
        let path = layout
            .chunk_path_for_tuple_key("tcwv", &key)
            .expect("single-dim tuple-key should succeed");
        assert_eq!(path.relative_path, "tcwv/7");
    }

    #[test]
    fn v2_slash_layout_renders_tuple_key_path() {
        let layout = DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
            zarr_format: ZarrFormat::V2,
            chunk_key_encoding: ChunkKeyEncoding::V2,
            chunk_key_separator: ChunkKeySeparator::Slash,
        })
        .expect("valid v2 slash target");
        let key = TupleChunkKey::new(vec![0, 4, 0, 0]);
        let path = layout
            .chunk_path_for_tuple_key("temperature", &key)
            .expect("tuple-key path render should succeed");
        assert_eq!(
            path.relative_path, "temperature/0/4/0/0",
            "V2 slash separator should produce slash-joined tuple key"
        );
    }

    #[test]
    fn v3_layout_renders_tuple_key_path_with_c_prefix() {
        let layout = DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
            zarr_format: ZarrFormat::V3,
            chunk_key_encoding: ChunkKeyEncoding::Default,
            chunk_key_separator: ChunkKeySeparator::Slash,
        })
        .expect("valid v3 target");
        let key = TupleChunkKey::new(vec![0, 4, 0, 0]);
        let path = layout
            .chunk_path_for_tuple_key("temperature", &key)
            .expect("tuple-key path render should succeed");
        assert_eq!(
            path.relative_path, "temperature/c/0/4/0/0",
            "V3 should produce slash-separated tuple key under c/ prefix"
        );
    }

    #[test]
    fn v3_layout_renders_single_dim_tuple_key() {
        let layout = DefaultZarrLayoutAdapter::new(ZarrTargetConfig {
            zarr_format: ZarrFormat::V3,
            chunk_key_encoding: ChunkKeyEncoding::Default,
            chunk_key_separator: ChunkKeySeparator::Slash,
        })
        .expect("valid v3 target");
        let key = TupleChunkKey::new(vec![7]);
        let path = layout
            .chunk_path_for_tuple_key("tcwv", &key)
            .expect("single-dim tuple-key should succeed");
        assert_eq!(path.relative_path, "tcwv/c/7");
    }
}
