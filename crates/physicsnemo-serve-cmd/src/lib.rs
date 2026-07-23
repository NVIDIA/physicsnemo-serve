/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

use anyhow::Result;

pub mod bundle;
pub mod prefetch;

pub const PREFETCH_COMMAND: &str = "__prefetch";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliCommand {
    Infer(InferArgs),
    Package(PackageArgs),
    Prefetch(PrefetchArgs),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferArgs {
    pub plugin_root: PathBuf,
    pub request: PathBuf,
    pub output_dir: PathBuf,
    pub run_id: Option<String>,
    pub device: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageArgs {
    pub runtime_dir: PathBuf,
    pub output: PathBuf,
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
        "infer" | "run" => parse_infer_args(&remaining).map(CliCommand::Infer),
        "package" => parse_package_args(&remaining).map(CliCommand::Package),
        PREFETCH_COMMAND => parse_prefetch_args(&remaining).map(CliCommand::Prefetch),
        unsupported => Err(anyhow::anyhow!(
            "unsupported command '{unsupported}': expected infer or package"
        )),
    }
}

fn parse_infer_args(args: &[String]) -> Result<InferArgs> {
    let options = parse_options(args)?;
    Ok(InferArgs {
        plugin_root: required_path(&options, "--plugin")?,
        request: required_path(&options, "--request")?,
        output_dir: required_path(&options, "--output-dir")?,
        run_id: options.get("--run-id").cloned(),
        device: options.get("--device").cloned(),
    })
}

fn parse_package_args(args: &[String]) -> Result<PackageArgs> {
    let options = parse_options(args)?;
    Ok(PackageArgs {
        runtime_dir: required_path(&options, "--runtime-dir")?,
        output: required_path(&options, "--output")?,
    })
}

fn parse_prefetch_args(args: &[String]) -> Result<PrefetchArgs> {
    let options = parse_options(args)?;
    Ok(PrefetchArgs {
        cache_dir: required_path(&options, "--cache-dir")?,
        run_id: required_value(&options, "--run-id")?.to_string(),
    })
}

fn parse_options(args: &[String]) -> Result<std::collections::BTreeMap<String, String>> {
    let mut options = std::collections::BTreeMap::new();
    let mut index = 0;
    while index < args.len() {
        let option = &args[index];
        if !option.starts_with("--") {
            return Err(anyhow::anyhow!("unexpected positional argument '{option}'"));
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
    use std::path::Path;

    use tempfile::tempdir;

    use super::bundle::{extract_runtime, package_executable};
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
                run_id: Some("run-1".to_string()),
                device: Some("2".to_string()),
            })
        );
    }

    #[test]
    fn rejects_missing_required_infer_argument() {
        let error = parse_args(["infer", "--plugin", "/plugins/demo"])
            .expect_err("missing request and output arguments should fail");

        assert!(error.to_string().contains("--request"));
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
