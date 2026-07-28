/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs::File;
use std::io::{Read, Result, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

pub fn copy_and_sha256(mut reader: impl Read, mut writer: impl Write) -> Result<(u64, [u8; 32])> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    let mut length = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        writer.write_all(&buffer[..read])?;
        length += read as u64;
    }
    Ok((length, hasher.finalize().into()))
}

pub fn sha256_reader(mut reader: impl Read) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

pub fn sha256_file_hex(path: &Path) -> Result<String> {
    let digest = sha256_reader(File::open(path)?)?;
    Ok(hex(&digest))
}

pub fn hex(digest: &[u8]) -> String {
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_and_copies_the_same_bytes() {
        let input = b"physicsnemo";
        let mut output = Vec::new();
        let (length, copied_digest) = copy_and_sha256(input.as_slice(), &mut output).unwrap();
        let read_digest = sha256_reader(input.as_slice()).unwrap();
        assert_eq!(length, input.len() as u64);
        assert_eq!(output, input);
        assert_eq!(copied_digest, read_digest);
    }
}
