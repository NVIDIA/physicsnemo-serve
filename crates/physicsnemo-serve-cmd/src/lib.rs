/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

use anyhow::Result;

pub mod bundle;
pub mod digest;
pub mod installer;
pub mod prefetch;

/// Hidden JSON-over-stdin command reserved for internal prefetch subprocess use.
pub const PREFETCH_COMMAND: &str = "__prefetch";
pub const USAGE: &str = "\
physicsnemo-serve — run manifest-driven inference without service processes

USAGE:
  physicsnemo-serve infer --plugin PATH --request FILE --output-dir DIR [--runtime-dir DIR] [--run-id ID] [--device DEVICE]
  physicsnemo-serve package --runtime-dir DIR --output FILE [--compression-level LEVEL]
  physicsnemo-serve --help
  physicsnemo-serve --version

The internal __prefetch command is reserved for CLI subprocess communication.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliCommand {
    Help,
    Version,
    Infer(InferArgs),
    Package(PackageArgs),
    Prefetch(PrefetchArgs),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferArgs {
    pub plugin_root: PathBuf,
    pub request: PathBuf,
    pub output_dir: PathBuf,
    pub runtime_dir: Option<PathBuf>,
    pub run_id: Option<String>,
    pub device: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageArgs {
    pub runtime_dir: PathBuf,
    pub output: PathBuf,
    pub compression_level: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchArgs {
    pub cache_dir: PathBuf,
    pub run_id: String,
}

pub fn parse_args<I, S>(args: I) -> Result<CliCommand>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let command = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing command: expected infer or package"))?;
    let remaining: Vec<String> = args.collect();
    match command.as_str() {
        "--help" | "-h" | "help" => Ok(CliCommand::Help),
        "--version" | "-V" => Ok(CliCommand::Version),
        "infer" if is_help_request(&remaining) => Ok(CliCommand::Help),
        "package" if is_help_request(&remaining) => Ok(CliCommand::Help),
        "infer" => parse_infer_args(&remaining).map(CliCommand::Infer),
        "package" => parse_package_args(&remaining).map(CliCommand::Package),
        PREFETCH_COMMAND => parse_prefetch_args(&remaining).map(CliCommand::Prefetch),
        unsupported => Err(anyhow::anyhow!(
            "unsupported command '{unsupported}': expected infer or package"
        )),
    }
}

fn parse_infer_args(args: &[String]) -> Result<InferArgs> {
    let options = parse_options(
        args,
        &[
            "--plugin",
            "--request",
            "--output-dir",
            "--runtime-dir",
            "--run-id",
            "--device",
        ],
    )?;
    let run_id = options.get("--run-id").cloned();
    if let Some(run_id) = run_id.as_deref() {
        validate_run_id(run_id)?;
    }
    let device = options.get("--device").cloned();
    if let Some(device) = device.as_deref() {
        validate_device(device)?;
    }
    Ok(InferArgs {
        plugin_root: required_path(&options, "--plugin")?,
        request: required_path(&options, "--request")?,
        output_dir: required_path(&options, "--output-dir")?,
        runtime_dir: options.get("--runtime-dir").map(PathBuf::from),
        run_id,
        device,
    })
}

fn parse_package_args(args: &[String]) -> Result<PackageArgs> {
    let options = parse_options(args, &["--runtime-dir", "--output", "--compression-level"])?;
    let compression_level = options
        .get("--compression-level")
        .map(|value| {
            value
                .parse::<i32>()
                .map_err(|_| anyhow::anyhow!("compression level must be a signed integer"))
        })
        .transpose()?
        .unwrap_or(bundle::DEFAULT_COMPRESSION_LEVEL);
    Ok(PackageArgs {
        runtime_dir: required_path(&options, "--runtime-dir")?,
        output: required_path(&options, "--output")?,
        compression_level,
    })
}

fn parse_prefetch_args(args: &[String]) -> Result<PrefetchArgs> {
    let options = parse_options(args, &["--cache-dir", "--run-id"])?;
    let run_id = required_value(&options, "--run-id")?;
    validate_run_id(run_id)?;
    Ok(PrefetchArgs {
        cache_dir: required_path(&options, "--cache-dir")?,
        run_id: run_id.to_string(),
    })
}

fn parse_options(
    args: &[String],
    allowed: &[&str],
) -> Result<std::collections::BTreeMap<String, String>> {
    let mut options = std::collections::BTreeMap::new();
    let mut index = 0;
    while index < args.len() {
        let option = &args[index];
        if !option.starts_with("--") {
            return Err(anyhow::anyhow!("unexpected positional argument '{option}'"));
        }
        if !allowed.contains(&option.as_str()) {
            return Err(anyhow::anyhow!("unknown option '{option}'"));
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| anyhow::anyhow!("missing value for {option}"))?;
        if value.starts_with("--") {
            return Err(anyhow::anyhow!("missing value for {option}"));
        }
        if options.insert(option.clone(), value.clone()).is_some() {
            return Err(anyhow::anyhow!("duplicate option {option}"));
        }
        index += 2;
    }
    Ok(options)
}

fn is_help_request(args: &[String]) -> bool {
    matches!(args, [arg] if arg == "--help" || arg == "-h")
}

pub fn validate_run_id(run_id: &str) -> Result<()> {
    if run_id.is_empty() || run_id.len() > 128 {
        return Err(anyhow::anyhow!(
            "run ID must contain between 1 and 128 characters"
        ));
    }
    if run_id.starts_with('.') {
        return Err(anyhow::anyhow!("run ID must not start with '.'"));
    }
    if !run_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(anyhow::anyhow!(
            "run ID may contain only ASCII letters, digits, '.', '-', and '_'"
        ));
    }
    Ok(())
}

fn validate_device(device: &str) -> Result<()> {
    if device.is_empty() || device.len() > 256 {
        return Err(anyhow::anyhow!(
            "device must contain between 1 and 256 characters"
        ));
    }
    if device.starts_with(',') || device.ends_with(',') || device.contains(",,") {
        return Err(anyhow::anyhow!(
            "device contains an empty CUDA device entry"
        ));
    }
    if !device.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b',' | b'-' | b'_' | b'.' | b'/')
    }) {
        return Err(anyhow::anyhow!(
            "device contains characters not valid in CUDA_VISIBLE_DEVICES"
        ));
    }
    Ok(())
}

fn required_path(
    options: &std::collections::BTreeMap<String, String>,
    name: &str,
) -> Result<PathBuf> {
    required_value(options, name).map(PathBuf::from)
}

fn required_value<'a>(
    options: &'a std::collections::BTreeMap<String, String>,
    name: &str,
) -> Result<&'a str> {
    options
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing required option {name}"))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::fs::symlink;
    use std::path::Path;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use tempfile::tempdir;

    use super::bundle::{extract_runtime, package_executable, package_executable_with_compression};
    use super::*;

    #[test]
    fn parses_infer_command() {
        let command = parse_args([
            "infer",
            "--plugin",
            "/plugins/demo",
            "--request",
            "request.json",
            "--output-dir",
            "outputs",
            "--run-id",
            "run-1",
            "--device",
            "2",
        ])
        .expect("infer arguments should parse");

        assert_eq!(
            command,
            CliCommand::Infer(InferArgs {
                plugin_root: PathBuf::from("/plugins/demo"),
                request: PathBuf::from("request.json"),
                output_dir: PathBuf::from("outputs"),
                runtime_dir: None,
                run_id: Some("run-1".to_string()),
                device: Some("2".to_string()),
            })
        );
    }

    #[test]
    fn parses_external_runtime_directory() {
        let command = parse_args([
            "infer",
            "--plugin",
            "/plugins/demo",
            "--request",
            "request.json",
            "--output-dir",
            "outputs",
            "--runtime-dir",
            "/opt/physicsnemo-runtime",
        ])
        .unwrap();

        let CliCommand::Infer(args) = command else {
            panic!("expected infer command");
        };
        assert_eq!(
            args.runtime_dir,
            Some(PathBuf::from("/opt/physicsnemo-runtime"))
        );
    }

    #[test]
    fn rejects_removed_run_command() {
        let error = parse_args([
            "run",
            "--plugin",
            "/plugins/demo",
            "--request",
            "request.json",
            "--output-dir",
            "outputs",
        ])
        .expect_err("the removed run command must be rejected");

        assert!(error.to_string().contains("unsupported command 'run'"));
    }

    #[test]
    fn rejects_missing_required_infer_argument() {
        let error = parse_args(["infer", "--plugin", "/plugins/demo"])
            .expect_err("missing request and output arguments should fail");

        assert!(error.to_string().contains("--request"));
    }

    #[test]
    fn parses_help_and_version_commands() {
        assert_eq!(parse_args(["--help"]).unwrap(), CliCommand::Help);
        assert_eq!(parse_args(["infer", "--help"]).unwrap(), CliCommand::Help);
        assert_eq!(parse_args(["--version"]).unwrap(), CliCommand::Version);
    }

    #[test]
    fn rejects_unsafe_run_ids_and_device_values() {
        for run_id in ["../escape", "..", ".hidden", "nested/run", "has space"] {
            let error = parse_args([
                "infer",
                "--plugin",
                "/plugins/demo",
                "--request",
                "request.json",
                "--output-dir",
                "outputs",
                "--run-id",
                run_id,
            ])
            .unwrap_err();
            assert!(error.to_string().contains("run ID"));
        }

        let error = parse_args([
            "infer",
            "--plugin",
            "/plugins/demo",
            "--request",
            "request.json",
            "--output-dir",
            "outputs",
            "--device",
            "0;echo injected",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("CUDA_VISIBLE_DEVICES"));
    }

    #[test]
    fn rejects_unknown_options() {
        let error = parse_args([
            "infer",
            "--plugin",
            "/plugins/demo",
            "--request",
            "request.json",
            "--output-dir",
            "outputs",
            "--unknown",
            "value",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("unknown option"));
    }

    #[test]
    fn parses_package_compression_level() {
        assert_eq!(
            parse_args([
                "package",
                "--runtime-dir",
                "runtime",
                "--output",
                "physicsnemo-serve",
            ])
            .unwrap(),
            CliCommand::Package(PackageArgs {
                runtime_dir: PathBuf::from("runtime"),
                output: PathBuf::from("physicsnemo-serve"),
                compression_level: bundle::DEFAULT_COMPRESSION_LEVEL,
            })
        );
        assert_eq!(
            parse_args([
                "package",
                "--runtime-dir",
                "runtime",
                "--output",
                "physicsnemo-serve",
                "--compression-level",
                "12",
            ])
            .unwrap(),
            CliCommand::Package(PackageArgs {
                runtime_dir: PathBuf::from("runtime"),
                output: PathBuf::from("physicsnemo-serve"),
                compression_level: 12,
            })
        );
    }

    #[test]
    fn rejects_non_integer_package_compression_level() {
        let error = parse_args([
            "package",
            "--runtime-dir",
            "runtime",
            "--output",
            "physicsnemo-serve",
            "--compression-level",
            "fast",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("signed integer"));
    }

    #[test]
    fn rejects_out_of_range_package_compression_level() {
        let error = package_executable_with_compression(
            Path::new("base"),
            Path::new("runtime"),
            Path::new("output"),
            23,
        )
        .unwrap_err();
        assert!(error.to_string().contains("between -7 and 22"));
    }

    #[test]
    fn packages_and_extracts_runtime_with_executable_permissions() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let packaged = temp.path().join("physicsnemo-serve-packaged");

        package_executable(&base, &runtime, &packaged).expect("runtime should package");
        let cache_root = temp.path().join("cache");
        let extracted =
            extract_runtime(&packaged, &cache_root).expect("runtime should extract once");
        let extracted_again =
            extract_runtime(&packaged, &cache_root).expect("cached runtime should be reused");

        assert_eq!(extracted, extracted_again);
        assert_eq!(
            fs::read_to_string(extracted.join("scripts/plugin_direct_runner.py"))
                .expect("runner should be extracted"),
            "print('runner')"
        );
        let mode = fs::metadata(extracted.join("bin/python"))
            .expect("python should be extracted")
            .permissions()
            .mode();
        assert_ne!(mode & 0o111, 0);
    }

    #[test]
    fn cached_runtime_is_reused_without_reading_the_payload() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let packaged = temp.path().join("physicsnemo-serve-packaged");
        package_executable(&base, &runtime, &packaged).expect("runtime should package");
        let cache_root = temp.path().join("cache");
        let extracted = extract_runtime(&packaged, &cache_root).expect("runtime should extract");

        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&packaged)
            .expect("packaged executable should open");
        file.seek(SeekFrom::Start(8))
            .expect("payload byte should be seekable");
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte)
            .expect("payload byte should be readable");
        file.seek(SeekFrom::Start(8))
            .expect("payload byte should be seekable");
        file.write_all(&[byte[0] ^ 0xff])
            .expect("payload byte should be changed");

        let reused = extract_runtime(&packaged, &cache_root)
            .expect("a completed digest cache should bypass payload hashing");
        assert_eq!(reused, extracted);
    }

    #[test]
    fn cached_runtime_repairs_missing_or_non_executable_essential_files() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let packaged = temp.path().join("physicsnemo-serve-packaged");
        package_executable(&base, &runtime, &packaged).expect("runtime should package");
        let cache_root = temp.path().join("cache");
        let extracted = extract_runtime(&packaged, &cache_root).expect("runtime should extract");

        fs::remove_file(extracted.join("scripts/plugin_direct_runner.py"))
            .expect("cached runner should be removed");
        let repaired = extract_runtime(&packaged, &cache_root)
            .expect("missing cached runner should be repaired");
        assert!(repaired.join("scripts/plugin_direct_runner.py").is_file());

        let python = repaired.join("bin/python");
        let mut permissions = fs::metadata(&python)
            .expect("cached python should exist")
            .permissions();
        permissions.set_mode(0o644);
        fs::set_permissions(&python, permissions)
            .expect("cached python permissions should be changed");
        let repaired = extract_runtime(&packaged, &cache_root)
            .expect("non-executable cached python should be repaired");
        let repaired_mode = fs::metadata(repaired.join("bin/python"))
            .expect("repaired python should exist")
            .permissions()
            .mode();
        assert_ne!(repaired_mode & 0o111, 0);
    }

    #[test]
    fn concurrent_extraction_serializes_incomplete_cache_repair() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let packaged = temp.path().join("physicsnemo-serve-packaged");
        package_executable(&base, &runtime, &packaged).expect("runtime should package");
        let cache_root = temp.path().join("cache");
        let extracted = extract_runtime(&packaged, &cache_root).expect("runtime should extract");
        fs::remove_file(extracted.join(".physicsnemo-serve-runtime-complete"))
            .expect("cache should be marked incomplete");

        let workers = 8;
        let barrier = Arc::new(Barrier::new(workers));
        let handles = (0..workers)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let packaged = packaged.clone();
                let cache_root = cache_root.clone();
                thread::spawn(move || {
                    barrier.wait();
                    extract_runtime(&packaged, &cache_root)
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            let repaired = handle
                .join()
                .expect("cache repair thread should not panic")
                .expect("concurrent cache repair should succeed");
            assert!(repaired.join("bin/python").is_file());
            assert!(
                repaired
                    .join(".physicsnemo-serve-runtime-complete")
                    .is_file()
            );
        }
    }

    #[test]
    fn rejects_tampered_runtime_payload() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let packaged = temp.path().join("physicsnemo-serve-packaged");
        package_executable(&base, &runtime, &packaged).expect("runtime should package");

        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&packaged)
            .expect("packaged executable should open");
        file.seek(SeekFrom::Start(8))
            .expect("payload byte should be seekable");
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte)
            .expect("payload byte should be readable");
        file.seek(SeekFrom::Start(8))
            .expect("payload byte should be seekable");
        file.write_all(&[byte[0] ^ 0xff])
            .expect("payload byte should be changed");

        let error = extract_runtime(&packaged, &temp.path().join("cache"))
            .expect_err("tampered payload should fail");
        assert!(error.to_string().contains("checksum"));
    }

    #[test]
    fn rejects_runtime_without_python() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = temp.path().join("runtime");
        fs::create_dir_all(runtime.join("scripts")).expect("scripts directory should be created");
        fs::write(
            runtime.join("scripts/plugin_direct_runner.py"),
            "print('runner')",
        )
        .expect("runner should be written");

        let error = package_executable(&base, &runtime, &temp.path().join("packaged"))
            .expect_err("missing Python should fail");
        assert!(error.to_string().contains("bin/python"));
    }

    #[test]
    fn package_rejects_existing_output() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let output = temp.path().join("packaged");
        fs::write(&output, b"keep-me").expect("existing output should be written");

        let error = package_executable(&base, &runtime, &output)
            .expect_err("existing output must not be overwritten");
        assert!(error.to_string().contains("already exists"));
        assert_eq!(fs::read(&output).unwrap(), b"keep-me");
    }

    #[test]
    fn package_rejects_output_inside_runtime() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let output = runtime.join("nested/dist/physicsnemo-serve");

        let error = package_executable(&base, &runtime, &output)
            .expect_err("output inside the runtime must be rejected");

        assert!(error.to_string().contains("inside the runtime directory"));
        assert!(!runtime.join("nested").exists());
    }

    #[test]
    fn package_rejects_output_inside_runtime_through_symlink() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let runtime_alias = temp.path().join("runtime-alias");
        symlink(&runtime, &runtime_alias).expect("runtime alias should be created");
        let output = runtime_alias.join("nested/../physicsnemo-serve-packaged");

        let error = package_executable(&base, &runtime, &output)
            .expect_err("symlinked output inside the runtime must be rejected");

        assert!(error.to_string().contains("inside the runtime directory"));
        assert!(!runtime.join("physicsnemo-serve-packaged").exists());
    }

    #[test]
    fn package_preserves_runtime_symlinks() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        fs::write(runtime.join("bin/python3.12"), "#!/bin/sh\nexit 0\n")
            .expect("symlink target should be written");
        symlink("python3.12", runtime.join("bin/python3"))
            .expect("runtime symlink should be created");
        let packaged = temp.path().join("packaged");
        package_executable(&base, &runtime, &packaged).expect("runtime should package");

        let extracted =
            extract_runtime(&packaged, &temp.path().join("cache")).expect("runtime should extract");
        assert_eq!(
            fs::read_link(extracted.join("bin/python3")).unwrap(),
            PathBuf::from("python3.12")
        );
    }

    #[test]
    fn package_rejects_absolute_runtime_symlinks() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let outside = temp.path().join("outside");
        fs::write(&outside, b"outside").expect("outside file should be written");
        symlink(&outside, runtime.join("absolute-link"))
            .expect("absolute runtime symlink should be created");

        let error = package_executable(&base, &runtime, &temp.path().join("packaged"))
            .expect_err("absolute runtime symlink must be rejected");

        assert!(
            error
                .to_string()
                .contains("non-relocatable absolute symlink")
        );
    }

    #[test]
    fn package_rejects_runtime_symlinks_that_escape() {
        let temp = tempdir().expect("temp directory should be created");
        let base = temp.path().join("physicsnemo-serve");
        fs::write(&base, b"fake-elf").expect("base executable should be written");
        let runtime = create_runtime(temp.path());
        let outside = temp.path().join("outside");
        fs::write(&outside, b"outside").expect("outside file should be written");
        symlink("../../outside", runtime.join("bin/escaping-link"))
            .expect("escaping runtime symlink should be created");

        let error = package_executable(&base, &runtime, &temp.path().join("packaged"))
            .expect_err("runtime-escaping symlink must be rejected");

        assert!(error.to_string().contains("escapes the runtime"));
    }

    fn create_runtime(root: &Path) -> PathBuf {
        let runtime = root.join("runtime");
        fs::create_dir_all(runtime.join("bin")).expect("bin directory should be created");
        fs::create_dir_all(runtime.join("scripts")).expect("scripts directory should be created");
        let python = runtime.join("bin/python");
        fs::write(&python, "#!/bin/sh\nexit 0\n").expect("python stub should be written");
        let mut permissions = fs::metadata(&python)
            .expect("python metadata should exist")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&python, permissions).expect("python should be executable");
        fs::write(
            runtime.join("scripts/plugin_direct_runner.py"),
            "print('runner')",
        )
        .expect("runner should be written");
        runtime
    }
}
