/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dtype descriptors that must survive the Python/Rust boundary.

/// Logical array dtype used to emit Zarr metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataType {
    /// Boolean values.
    Bool,
    /// Signed integer values.
    Int8,
    /// Signed integer values.
    Int16,
    /// Signed integer values.
    Int32,
    /// Signed integer values.
    Int64,
    /// Unsigned integer values.
    UInt8,
    /// Unsigned integer values.
    UInt16,
    /// Unsigned integer values.
    UInt32,
    /// Unsigned integer values.
    UInt64,
    /// Floating-point values.
    Float16,
    /// Floating-point values.
    #[default]
    Float32,
    /// Floating-point values.
    Float64,
    /// NumPy-compatible nanosecond datetime values.
    DatetimeNs,
    /// NumPy-compatible nanosecond timedelta values.
    TimedeltaNs,
}

impl DataType {
    /// Return the Zarr v2 dtype string.
    #[must_use]
    pub const fn v2_dtype(self) -> &'static str {
        match self {
            Self::Bool => "|b1",
            Self::Int8 => "|i1",
            Self::Int16 => "<i2",
            Self::Int32 => "<i4",
            Self::Int64 => "<i8",
            Self::UInt8 => "|u1",
            Self::UInt16 => "<u2",
            Self::UInt32 => "<u4",
            Self::UInt64 => "<u8",
            Self::Float16 => "<f2",
            Self::Float32 => "<f4",
            Self::Float64 => "<f8",
            Self::DatetimeNs => "<M8[ns]",
            Self::TimedeltaNs => "<m8[ns]",
        }
    }

    /// Return the Zarr v3 `data_type` JSON payload.
    #[must_use]
    pub const fn v3_data_type_json(self) -> &'static str {
        match self {
            Self::Bool => "\"bool\"",
            Self::Int8 => "\"int8\"",
            Self::Int16 => "\"int16\"",
            Self::Int32 => "\"int32\"",
            Self::Int64 => "\"int64\"",
            Self::UInt8 => "\"uint8\"",
            Self::UInt16 => "\"uint16\"",
            Self::UInt32 => "\"uint32\"",
            Self::UInt64 => "\"uint64\"",
            Self::Float16 => "\"float16\"",
            Self::Float32 => "\"float32\"",
            Self::Float64 => "\"float64\"",
            Self::DatetimeNs => {
                "{\"name\":\"numpy.datetime64\",\"configuration\":{\"unit\":\"ns\",\"scale_factor\":1}}"
            }
            Self::TimedeltaNs => {
                "{\"name\":\"numpy.timedelta64\",\"configuration\":{\"unit\":\"ns\",\"scale_factor\":1}}"
            }
        }
    }

    /// Return the primitive Zarr v3 data type name or object JSON for extensions.
    #[must_use]
    pub const fn v3_data_type(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::DatetimeNs => {
                "{\"name\":\"numpy.datetime64\",\"configuration\":{\"unit\":\"ns\",\"scale_factor\":1}}"
            }
            Self::TimedeltaNs => {
                "{\"name\":\"numpy.timedelta64\",\"configuration\":{\"unit\":\"ns\",\"scale_factor\":1}}"
            }
        }
    }

    /// Return the dtype width in bytes.
    #[must_use]
    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 | Self::Float16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 | Self::DatetimeNs | Self::TimedeltaNs => 8,
        }
    }

    /// Return the Zarr v3 fill value JSON payload used by Python Zarr defaults.
    #[must_use]
    pub const fn v3_fill_value_json(self) -> &'static str {
        match self {
            Self::Float16 | Self::Float32 | Self::Float64 => "0.0",
            Self::Bool => "false",
            Self::DatetimeNs | Self::TimedeltaNs => "-9223372036854775808",
            _ => "0",
        }
    }
}

/// Typed coordinate value array. `Eq` is not derived because floats contain NaN.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CoordValues {
    /// Signed 64-bit integer coordinates.
    I64(Vec<i64>),
    /// NumPy-compatible nanosecond datetime coordinates.
    DatetimeNs(Vec<i64>),
    /// NumPy-compatible nanosecond timedelta coordinates.
    TimedeltaNs(Vec<i64>),
    /// Unsigned 64-bit integer coordinates.
    U64(Vec<u64>),
    /// Signed 32-bit integer coordinates.
    I32(Vec<i32>),
    /// Unsigned 32-bit integer coordinates.
    U32(Vec<u32>),
    /// 32-bit float coordinates.
    F32(Vec<f32>),
    /// 64-bit float coordinates.
    F64(Vec<f64>),
    /// String-typed coordinates.
    Utf8(Vec<String>),
}

impl CoordValues {
    /// Number of coordinate values along this axis.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::I64(v) | Self::DatetimeNs(v) | Self::TimedeltaNs(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Utf8(v) => v.len(),
        }
    }

    /// Returns `true` if the coordinate axis has no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
