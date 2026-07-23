/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use tar::{Archive, Builder, HeaderMode};
use tempfile::{Builder as TempBuilder, NamedTempFile};

const FOOTER_MAGIC: &[u8; 16] = b"PHYSNEMOSERVEV1\0";
const FOOTER_SIZE: u64 = 16 + 8 + 8 + 32;
const COMPLETE_MARKER: &str = ".physicsnemo-serve-runtime-complete";

struct BundleFooter {
    payload_offset: u64,
    payload_length: u64,
    digest: [u8; 32],
}

pub fn package_executable(base_executable: &Path, runtime_dir: &Path, output: &Path) -> Result<()> {
    if !base_executable.is_file() {
        return Err(anyhow!(
            "base executable does not exist: {}",
            base_executable.display()
        ));
    }
    validate_runtime_layout(runtime_dir)?;
    if base_executable == output {
        return Err(anyhow!(
            "packaged output must differ from the base executable"
        ));
    }

    let output_parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent).with_context(|| {
        format!(
            "failed to create packaged output directory: {}",
            output_parent.display()
        )
    })?;
    let archive = build_runtime_archive(runtime_dir, output_parent)?;

    fs::copy(base_executable, output).with_context(|| {
        format!(
            "failed to copy base executable '{}' to '{}'",
            base_executable.display(),
            output.display()
        )
    })?;
    let payload_offset = fs::metadata(output)?.len();
    let mut output_file = OpenOptions::new()
        .append(true)
        .open(output)
        .with_context(|| format!("failed to open packaged output: {}", output.display()))?;
    let (payload_length, digest) = copy_payload_with_digest(archive.path(), &mut output_file)?;
    write_footer(
        &mut output_file,
        &BundleFooter {
            payload_offset,
            payload_length,
            digest,
        },
    )?;
    output_file.sync_all()?;
    fs::set_permissions(output, fs::metadata(base_executable)?.permissions())?;
    Ok(())
}

pub fn extract_runtime(executable: &Path, cache_root: &Path) -> Result<PathBuf> {
    let mut executable_file = File::open(executable)
        .with_context(|| format!("failed to open executable: {}", executable.display()))?;
    let footer = read_footer(&mut executable_file)?;
    verify_payload_checksum(&mut executable_file, &footer)?;
    let cache_key = digest_hex(&footer.digest);
    let runtime_root = cache_root.join(&cache_key);
    if runtime_is_ready(&runtime_root) {
        return Ok(runtime_root);
    }

    fs::create_dir_all(cache_root).with_context(|| {
        format!(
            "failed to create runtime cache directory: {}",
            cache_root.display()
        )
    })?;
    if runtime_root.exists() {
        fs::remove_dir_all(&runtime_root).with_context(|| {
            format!(
                "failed to remove incomplete runtime cache: {}",
                runtime_root.display()
            )
        })?;
    }

    let temporary = TempBuilder::new()
        .prefix(".physicsnemo-serve-runtime-")
        .tempdir_in(cache_root)?;
    executable_file.seek(SeekFrom::Start(footer.payload_offset))?;
    let payload = executable_file.take(footer.payload_length);
    let decoder = zstd::Decoder::new(payload).context("failed to open runtime payload")?;
    let mut archive = Archive::new(decoder);
    archive
        .unpack(temporary.path())
        .context("failed to extract bundled runtime")?;
    validate_runtime_layout(temporary.path())?;
    fs::write(
        temporary.path().join(COMPLETE_MARKER),
        format!("{cache_key}\n"),
    )?;

    let temporary_path = temporary.keep();
    match fs::rename(&temporary_path, &runtime_root) {
        Ok(()) => {}
        Err(error) if runtime_is_ready(&runtime_root) => {
            let _ = fs::remove_dir_all(&temporary_path);
            let _ = error;
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary_path);
            return Err(error).with_context(|| {
                format!(
                    "failed to install runtime cache: {}",
                    runtime_root.display()
                )
            });
        }
    }
    Ok(runtime_root)
}

fn build_runtime_archive(runtime_dir: &Path, output_parent: &Path) -> Result<NamedTempFile> {
    let archive_file = NamedTempFile::new_in(output_parent)?;
    let writer = archive_file.reopen()?;
    let encoder = zstd::Encoder::new(writer, 9).context("failed to initialize zstd encoder")?;
    let mut archive = Builder::new(encoder);
    archive.mode(HeaderMode::Deterministic);
    append_tree_sorted(&mut archive, runtime_dir, Path::new(""))?;
    let encoder = archive
        .into_inner()
        .context("failed to finish runtime tar archive")?;
    encoder
        .finish()
        .context("failed to finish compressed runtime archive")?;
    Ok(archive_file)
}

fn append_tree_sorted<W: Write>(
    archive: &mut Builder<W>,
    root: &Path,
    relative: &Path,
) -> Result<()> {
    let directory = root.join(relative);
    let mut entries = fs::read_dir(&directory)
        .with_context(|| format!("failed to read runtime directory: {}", directory.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let relative_path = relative.join(entry.file_name());
        let source_path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            archive
                .append_dir(&relative_path, &source_path)
                .with_context(|| {
                    format!("failed to add runtime directory: {}", source_path.display())
                })?;
            append_tree_sorted(archive, root, &relative_path)?;
        } else {
            archive
                .append_path_with_name(&source_path, &relative_path)
                .with_context(|| {
                    format!("failed to add runtime file: {}", source_path.display())
                })?;
        }
    }
    Ok(())
}

fn copy_payload_with_digest(payload_path: &Path, output: &mut File) -> Result<(u64, [u8; 32])> {
    let mut payload = File::open(payload_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut length = 0u64;
    loop {
        let read = payload.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        output.write_all(&buffer[..read])?;
        length += read as u64;
    }
    Ok((length, hasher.finalize().into()))
}

fn write_footer(output: &mut File, footer: &BundleFooter) -> Result<()> {
    output.write_all(FOOTER_MAGIC)?;
    output.write_all(&footer.payload_offset.to_le_bytes())?;
    output.write_all(&footer.payload_length.to_le_bytes())?;
    output.write_all(&footer.digest)?;
    Ok(())
}

fn read_footer(executable: &mut File) -> Result<BundleFooter> {
    let file_length = executable.metadata()?.len();
    if file_length < FOOTER_SIZE {
        return Err(anyhow!("executable does not contain a bundled runtime"));
    }
    executable.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
    let mut footer = [0u8; FOOTER_SIZE as usize];
    executable.read_exact(&mut footer)?;
    if &footer[..16] != FOOTER_MAGIC {
        return Err(anyhow!("executable does not contain a bundled runtime"));
    }

    let payload_offset = u64::from_le_bytes(footer[16..24].try_into()?);
    let payload_length = u64::from_le_bytes(footer[24..32].try_into()?);
    let digest = footer[32..64].try_into()?;
    let footer_offset = file_length - FOOTER_SIZE;
    if payload_offset
        .checked_add(payload_length)
        .filter(|end| *end == footer_offset)
        .is_none()
    {
        return Err(anyhow!("bundled runtime footer has invalid payload bounds"));
    }
    Ok(BundleFooter {
        payload_offset,
        payload_length,
        digest,
    })
}

fn verify_payload_checksum(executable: &mut File, footer: &BundleFooter) -> Result<()> {
    executable.seek(SeekFrom::Start(footer.payload_offset))?;
    let mut payload = executable.take(footer.payload_length);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = payload.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual: [u8; 32] = hasher.finalize().into();
    if actual != footer.digest {
        return Err(anyhow!("bundled runtime checksum mismatch"));
    }
    Ok(())
}

fn validate_runtime_layout(runtime_dir: &Path) -> Result<()> {
    for relative_path in ["bin/python", "scripts/plugin_direct_runner.py"] {
        let path = runtime_dir.join(relative_path);
        if !path.is_file() {
            return Err(anyhow!(
                "runtime is missing required file '{}': {}",
                relative_path,
                path.display()
            ));
        }
    }
    Ok(())
}

fn runtime_is_ready(runtime_root: &Path) -> bool {
    runtime_root.join(COMPLETE_MARKER).is_file() && validate_runtime_layout(runtime_root).is_ok()
}

fn digest_hex(digest: &[u8; 32]) -> String {
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}
