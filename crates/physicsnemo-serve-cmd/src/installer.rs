/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::{Cursor, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::GzDecoder;
use serde::Serialize;
use serde_yaml::Value;
use sha2::{Digest, Sha256};
use tar::Archive;
use tempfile::Builder;

const UV_VERSION: &str = "0.11.16";
const UV_RELEASE_ROOT: &str = "https://releases.astral.sh/github/uv/releases/download/0.11.16";
const MAX_UV_ARCHIVE_BYTES: u64 = 128 * 1024 * 1024;
const UV_LINUX_X86_64_SHA256: &str =
    "74947fe2c03315cf07e82ab3acc703eddef01aba4d5232a98e4c6825ec116131";
const UV_LINUX_AARCH64_SHA256: &str =
    "8c9d0f0e00add7cead46d2c3cf8778dd907a0d136bd1611f8580246bcb15c22a";
const UV_MACOS_X86_64_SHA256: &str =
    "6b91ae3dd32c9d86bcb87e94259c2f2cd55edc1b1e0a39f81ee25d6bdf517f2e";
const UV_MACOS_AARCH64_SHA256: &str =
    "2b25be1a32945fb4239762afee1fb38a9bc923e7f23c26e847ebd37d7ff388fb";

pub const INSTALLER_USAGE: &str = "\
physicsnemo-serve-install — create a plugin-specific external runtime

USAGE:
  physicsnemo-serve-install --plugin PATH [--plugin PATH ...] --runtime-dir DIR [OPTIONS]

OPTIONS:
  --plugin PATH           Plugin whose readiness imports must pass; may be repeated
  --requirements FILE     Additional requirements file; may be repeated
  --python VERSION        Python version or interpreter for uv (default: 3.12)
  --torch-backend VALUE   uv PyTorch backend (default: auto)
  --uv PATH               Use this uv executable instead of auto-detection
  --skip-import-checks    Do not verify developer.readiness.python_modules
  --help                  Show this help

If no --requirements option is supplied, <plugin>/requirements.txt is used
when present. If uv is not on PATH, a pinned copy is installed in the user's
local PhysicsNeMo Serve data directory. The destination must not already exist.";

const BASE_REQUIREMENTS: &str =
    include_str!("../../../packaging/physicsnemo-serve-cmd/runtime-base.in");

const SUPPORT_FILES: &[(&str, &[u8])] = &[
    (
        "scripts/plugin_direct_runner.py",
        include_bytes!("../../../scripts/plugin_direct_runner.py"),
    ),
    (
        "scripts/plugin_runtime.py",
        include_bytes!("../../../scripts/plugin_runtime.py"),
    ),
    (
        "scripts/plugin_sdk.py",
        include_bytes!("../../../scripts/plugin_sdk.py"),
    ),
    (
        "python/e2s_workflow.py",
        include_bytes!("../../../python/e2s_workflow.py"),
    ),
    (
        "python/e2s_tools/__init__.py",
        include_bytes!("../../../python/e2s_tools/__init__.py"),
    ),
    (
        "python/e2s_tools/model_metadata.py",
        include_bytes!("../../../python/e2s_tools/model_metadata.py"),
    ),
    (
        "python/e2s_tools/python_bridge.py",
        include_bytes!("../../../python/e2s_tools/python_bridge.py"),
    ),
    (
        "python/physicsnemo_cfd_runtime/__init__.py",
        include_bytes!("../../../python/physicsnemo_cfd_runtime/__init__.py"),
    ),
    (
        "python/physicsnemo_cfd_runtime/artifacts.py",
        include_bytes!("../../../python/physicsnemo_cfd_runtime/artifacts.py"),
    ),
    (
        "python/physicsnemo_cfd_runtime/safe_files.py",
        include_bytes!("../../../python/physicsnemo_cfd_runtime/safe_files.py"),
    ),
    (
        "python/physicsnemo_cfd_runtime/supervisor.py",
        include_bytes!("../../../python/physicsnemo_cfd_runtime/supervisor.py"),
    ),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallerArgs {
    pub plugin_roots: Vec<PathBuf>,
    pub runtime_dir: PathBuf,
    pub requirements: Vec<PathBuf>,
    pub python: String,
    pub torch_backend: String,
    pub uv: Option<PathBuf>,
    pub skip_import_checks: bool,
    pub help: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledRuntime {
    pub plugin_ids: Vec<String>,
    pub runtime_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct RuntimeManifest<'a> {
    schema_version: u8,
    plugin_ids: &'a [String],
    python: &'a str,
    requirements: &'a [String],
    checked_modules: &'a [String],
}

struct PluginManifest {
    id: String,
    root: PathBuf,
    manifest: Value,
}

pub fn parse_installer_args<I, S>(args: I) -> Result<InstallerArgs>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let args = args
        .into_iter()
        .map(|value| value.into())
        .collect::<Vec<_>>();
    if matches!(args.as_slice(), [arg] if arg == "--help" || arg == "-h") {
        return Ok(InstallerArgs {
            plugin_roots: Vec::new(),
            runtime_dir: PathBuf::new(),
            requirements: Vec::new(),
            python: "3.12".to_string(),
            torch_backend: "auto".to_string(),
            uv: None,
            skip_import_checks: false,
            help: true,
        });
    }

    let mut values = BTreeMap::<String, OsString>::new();
    let mut plugin_roots = Vec::new();
    let mut requirements = Vec::new();
    let mut skip_import_checks = false;
    let mut index = 0;
    while index < args.len() {
        let option = args[index]
            .to_str()
            .ok_or_else(|| anyhow!("installer options must be valid UTF-8"))?;
        if option == "--skip-import-checks" {
            if skip_import_checks {
                bail!("duplicate option --skip-import-checks");
            }
            skip_import_checks = true;
            index += 1;
            continue;
        }
        if !matches!(
            option,
            "--plugin"
                | "--runtime-dir"
                | "--requirements"
                | "--python"
                | "--torch-backend"
                | "--uv"
        ) {
            bail!("unknown option '{option}'");
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| anyhow!("missing value for {option}"))?;
        if option == "--plugin" {
            plugin_roots.push(PathBuf::from(value));
        } else if option == "--requirements" {
            requirements.push(PathBuf::from(value));
        } else if values.insert(option.to_string(), value.clone()).is_some() {
            bail!("duplicate option {option}");
        }
        index += 2;
    }

    if plugin_roots.is_empty() {
        bail!("missing required option --plugin");
    }
    Ok(InstallerArgs {
        plugin_roots,
        runtime_dir: required_path(&values, "--runtime-dir")?,
        requirements,
        python: optional_string(&values, "--python", "3.12")?,
        torch_backend: optional_string(&values, "--torch-backend", "auto")?,
        uv: values.get("--uv").map(PathBuf::from),
        skip_import_checks,
        help: false,
    })
}

pub fn install_runtime(args: &InstallerArgs) -> Result<InstalledRuntime> {
    let plugins = args
        .plugin_roots
        .iter()
        .map(|root| load_plugin(root))
        .collect::<Result<Vec<_>>>()?;
    let plugin_ids = plugins
        .iter()
        .map(|plugin| plugin.id.clone())
        .collect::<Vec<_>>();
    let mut checked_modules = Vec::new();
    for plugin in &plugins {
        for module in readiness_modules(&plugin.manifest)? {
            if !checked_modules.contains(&module) {
                checked_modules.push(module);
            }
        }
    }
    let plugin_roots = plugins
        .iter()
        .map(|plugin| plugin.root.as_path())
        .collect::<Vec<_>>();
    let requirements = resolve_requirements(args, &plugin_roots)?;
    let manifest_requirements = requirements
        .iter()
        .map(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .ok_or_else(|| anyhow!("requirements path has no file name: {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    if args.runtime_dir.exists() {
        bail!(
            "runtime directory already exists: {}",
            args.runtime_dir.display()
        );
    }
    let parent = args
        .runtime_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create runtime parent: {}", parent.display()))?;
    let staging = Builder::new()
        .prefix(".physicsnemo-runtime-")
        .tempdir_in(parent)
        .context("failed to create temporary runtime directory")?;
    let uv = resolve_uv(args.uv.as_deref())?;

    run(
        Command::new(&uv)
            .arg("venv")
            .arg("--python")
            .arg(&args.python)
            .arg(staging.path())
            .arg("--relocatable")
            .arg("--quiet"),
        "create Python environment",
    )?;

    let base_requirements = staging.path().join(".physicsnemo-base-requirements.txt");
    fs::write(&base_requirements, BASE_REQUIREMENTS)
        .context("failed to write embedded base requirements")?;
    let python = staging.path().join("bin/python");
    let mut install = Command::new(&uv);
    install
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&python)
        .arg("--torch-backend")
        .arg(&args.torch_backend)
        .arg("--requirements")
        .arg(&base_requirements);
    for requirement in &requirements {
        install.arg("--requirements").arg(requirement);
    }
    install.arg("--quiet");
    run(&mut install, "install runtime dependencies")?;
    fs::remove_file(&base_requirements).context("failed to remove temporary requirements file")?;

    for (relative, contents) in SUPPORT_FILES {
        let destination = staging.path().join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&destination, contents)
            .with_context(|| format!("failed to write {}", destination.display()))?;
    }

    if !args.skip_import_checks {
        verify_imports(&python, staging.path(), &checked_modules)?;
    }
    let runtime_manifest = RuntimeManifest {
        schema_version: 1,
        plugin_ids: &plugin_ids,
        python: &args.python,
        requirements: &manifest_requirements,
        checked_modules: &checked_modules,
    };
    fs::write(
        staging.path().join("runtime-manifest.json"),
        serde_json::to_vec_pretty(&runtime_manifest)?,
    )?;

    let staging_path = staging.keep();
    if let Err(error) = fs::rename(&staging_path, &args.runtime_dir) {
        if let Err(cleanup_error) = fs::remove_dir_all(&staging_path) {
            return Err(error).with_context(|| {
                format!(
                    "failed to publish runtime at {}; cleanup of staging directory {} also failed: {cleanup_error}",
                    args.runtime_dir.display(),
                    staging_path.display()
                )
            });
        }
        return Err(error).with_context(|| {
            format!(
                "failed to publish runtime at {}",
                args.runtime_dir.display()
            )
        });
    }
    Ok(InstalledRuntime {
        plugin_ids,
        runtime_dir: args.runtime_dir.clone(),
    })
}

fn resolve_uv(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        verify_uv_version(path, UV_VERSION).with_context(|| {
            format!(
                "the uv executable supplied with --uv must be version {UV_VERSION}: {}",
                path.display()
            )
        })?;
        return Ok(path.to_path_buf());
    }
    if verify_uv_version(Path::new("uv"), UV_VERSION).is_ok() {
        return Ok(PathBuf::from("uv"));
    }
    bootstrap_uv()
}

fn verify_uv_version(path: &Path, expected_version: &str) -> Result<()> {
    let output = Command::new(path)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to start {}", path.display()))?;
    if !output.status.success() {
        bail!("{} --version failed with {}", path.display(), output.status);
    }
    let stdout = std::str::from_utf8(&output.stdout).context("uv --version is not valid UTF-8")?;
    let mut fields = stdout.split_whitespace();
    let program = fields.next();
    let actual_version = fields.next();
    if program != Some("uv") || actual_version != Some(expected_version) {
        bail!(
            "{} has unexpected version output {:?}; expected uv {expected_version}",
            path.display(),
            stdout.trim()
        );
    }
    Ok(())
}

fn bootstrap_uv() -> Result<PathBuf> {
    let tools_root = dirs::data_local_dir()
        .ok_or_else(|| anyhow!("could not determine the user-local data directory"))?
        .join("physicsnemo-serve/tools/uv");
    let install_dir = tools_root.join(UV_VERSION);
    let installed_uv = install_dir.join("uv");
    if verify_uv_version(&installed_uv, UV_VERSION).is_ok() {
        return Ok(installed_uv);
    }

    fs::create_dir_all(&tools_root).with_context(|| {
        format!(
            "failed to create uv tools directory: {}",
            tools_root.display()
        )
    })?;
    quarantine_invalid_uv_install(&install_dir)?;
    if verify_uv_version(&installed_uv, UV_VERSION).is_ok() {
        return Ok(installed_uv);
    }
    let staging = Builder::new()
        .prefix(".uv-install-")
        .tempdir_in(&tools_root)
        .context("failed to create temporary uv installation directory")?;
    eprintln!("uv was not found; installing pinned uv {UV_VERSION}");
    let (asset, expected_sha256) = uv_asset()?;
    let archive_url = format!("{UV_RELEASE_ROOT}/{asset}");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .user_agent(format!(
            "physicsnemo-serve-install/{}",
            env!("CARGO_PKG_VERSION")
        ))
        .build()
        .context("failed to initialize the uv installer HTTP client")?;
    let archive = download_capped(&client, &archive_url, MAX_UV_ARCHIVE_BYTES)
        .context("failed to download the uv archive")?;
    verify_download_checksum(&archive, expected_sha256)?;

    let staged_uv = staging.path().join("uv");
    extract_uv(&archive, &staged_uv)?;
    verify_uv_version(&staged_uv, UV_VERSION)
        .context("the bootstrapped uv executable has an unexpected version")?;

    let staging_path = staging.keep();
    if let Err(error) = fs::rename(&staging_path, &install_dir) {
        let _ = fs::remove_dir_all(&staging_path);
        if verify_uv_version(&installed_uv, UV_VERSION).is_ok() {
            return Ok(installed_uv);
        }
        return Err(error)
            .with_context(|| format!("failed to publish uv at {}", install_dir.display()));
    }
    Ok(installed_uv)
}

fn quarantine_invalid_uv_install(install_dir: &Path) -> Result<()> {
    let installed_uv = install_dir.join("uv");
    if !install_dir.exists() || verify_uv_version(&installed_uv, UV_VERSION).is_ok() {
        return Ok(());
    }
    let parent = install_dir
        .parent()
        .ok_or_else(|| anyhow!("uv installation directory has no parent"))?;
    let quarantine_root = Builder::new()
        .prefix(".uv-invalid-")
        .tempdir_in(parent)
        .context("failed to create uv quarantine directory")?;
    let quarantined = quarantine_root.path().join("install");
    match fs::rename(install_dir, &quarantined) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to quarantine invalid uv installation at {}",
                    install_dir.display()
                )
            });
        }
    }
    let metadata = fs::symlink_metadata(&quarantined)?;
    if metadata.is_dir() {
        fs::remove_dir_all(&quarantined)
    } else {
        fs::remove_file(&quarantined)
    }
    .with_context(|| {
        format!(
            "failed to remove invalid uv installation from {}",
            quarantined.display()
        )
    })?;
    Ok(())
}

fn uv_asset() -> Result<(&'static str, &'static str)> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok((
            "uv-x86_64-unknown-linux-gnu.tar.gz",
            UV_LINUX_X86_64_SHA256,
        )),
        ("linux", "aarch64") => Ok((
            "uv-aarch64-unknown-linux-gnu.tar.gz",
            UV_LINUX_AARCH64_SHA256,
        )),
        ("macos", "x86_64") => Ok((
            "uv-x86_64-apple-darwin.tar.gz",
            UV_MACOS_X86_64_SHA256,
        )),
        ("macos", "aarch64") => Ok((
            "uv-aarch64-apple-darwin.tar.gz",
            UV_MACOS_AARCH64_SHA256,
        )),
        (os, arch) => bail!("automatic uv installation is unsupported on {os}/{arch}; use --uv"),
    }
}

fn download_capped(client: &reqwest::blocking::Client, url: &str, limit: u64) -> Result<Vec<u8>> {
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("failed to download {url}"))?
        .error_for_status()
        .with_context(|| format!("download returned an error: {url}"))?;
    let mut bytes = Vec::new();
    response
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {url}"))?;
    if bytes.len() as u64 > limit {
        bail!("download exceeded the {limit}-byte limit: {url}");
    }
    Ok(bytes)
}

fn verify_download_checksum(archive: &[u8], expected: &str) -> Result<()> {
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("pinned uv checksum is not a SHA-256 digest");
    }
    let digest = Sha256::digest(archive);
    let actual = crate::digest::hex(digest.as_ref());
    if actual != expected.to_ascii_lowercase() {
        bail!("downloaded uv archive failed SHA-256 verification");
    }
    Ok(())
}

fn extract_uv(archive: &[u8], destination: &Path) -> Result<()> {
    let decoder = GzDecoder::new(Cursor::new(archive));
    let mut archive = Archive::new(decoder);
    for entry in archive.entries().context("failed to read the uv archive")? {
        let mut entry = entry.context("failed to read an entry from the uv archive")?;
        let path = entry
            .path()
            .context("uv archive contains an invalid path")?;
        if path.file_name().and_then(|name| name.to_str()) == Some("uv")
            && entry.header().entry_type().is_file()
        {
            entry
                .unpack(destination)
                .with_context(|| format!("failed to extract {}", destination.display()))?;
            return Ok(());
        }
    }
    bail!("uv archive does not contain the uv executable")
}

fn load_plugin(plugin_root: &Path) -> Result<PluginManifest> {
    let root = plugin_root
        .canonicalize()
        .with_context(|| format!("plugin directory does not exist: {}", plugin_root.display()))?;
    if !root.is_dir() {
        bail!("plugin path is not a directory: {}", root.display());
    }
    let manifest_path = root.join("plugin.yaml");
    let manifest_source = fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "plugin manifest does not exist: {}",
            manifest_path.display()
        )
    })?;
    let manifest: Value = serde_yaml::from_str(&manifest_source)
        .with_context(|| format!("invalid plugin manifest: {}", manifest_path.display()))?;
    let id = manifest
        .get("metadata")
        .and_then(|value| value.get("id"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("plugin manifest is missing metadata.id"))?
        .to_string();
    Ok(PluginManifest { id, root, manifest })
}

fn resolve_requirements(args: &InstallerArgs, plugin_roots: &[&Path]) -> Result<Vec<PathBuf>> {
    let candidates = if args.requirements.is_empty() {
        plugin_roots
            .iter()
            .map(|root| root.join("requirements.txt"))
            .filter(|path| path.is_file())
            .collect()
    } else {
        args.requirements.clone()
    };
    candidates
        .into_iter()
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("requirements file does not exist: {}", path.display()))
                .and_then(|path| {
                    if path.is_file() {
                        Ok(path)
                    } else {
                        bail!("requirements path is not a file: {}", path.display())
                    }
                })
        })
        .collect()
}

fn readiness_modules(manifest: &Value) -> Result<Vec<String>> {
    let mut result = vec!["jsonschema".to_string(), "yaml".to_string()];
    let Some(modules) = manifest
        .get("developer")
        .and_then(|value| value.get("readiness"))
        .and_then(|value| value.get("python_modules"))
    else {
        return Ok(result);
    };
    let plugin_modules = modules
        .as_sequence()
        .ok_or_else(|| anyhow!("developer.readiness.python_modules must be an array"))?
        .iter()
        .map(|module| {
            module
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    anyhow!("developer.readiness.python_modules entries must be strings")
                })
        })
        .collect::<Result<Vec<_>>>()?;
    for module in plugin_modules {
        if !result.contains(&module) {
            result.push(module);
        }
    }
    Ok(result)
}

fn verify_imports(python: &Path, runtime: &Path, modules: &[String]) -> Result<()> {
    let encoded = serde_json::to_string(modules)?;
    let script =
        "import importlib,json,sys; [importlib.import_module(m) for m in json.loads(sys.argv[1])]";
    let python_path = runtime_python_path(runtime)?;
    run(
        Command::new(python)
            .arg("-c")
            .arg(script)
            .arg(encoded)
            .env("PYTHONNOUSERSITE", "1")
            .env("PYTHONPATH", python_path),
        "verify plugin Python imports",
    )
}

fn runtime_python_path(runtime: &Path) -> Result<OsString> {
    std::env::join_paths([runtime.join("scripts"), runtime.join("python")])
        .context("failed to construct runtime PYTHONPATH")
}

fn run(command: &mut Command, description: &str) -> Result<()> {
    let status = command
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("failed to {description}"))?;
    if !status.success() {
        bail!("{description} failed with {status}");
    }
    Ok(())
}

fn required_path(values: &BTreeMap<String, OsString>, option: &str) -> Result<PathBuf> {
    values
        .get(option)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("missing required option {option}"))
}

fn optional_string(
    values: &BTreeMap<String, OsString>,
    option: &str,
    default: &str,
) -> Result<String> {
    values
        .get(option)
        .map(|value| {
            value
                .to_str()
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| anyhow!("{option} must be non-empty UTF-8"))
        })
        .transpose()
        .map(|value| value.unwrap_or_else(|| default.to_string()))
}

#[cfg(test)]
mod tests {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::os::unix::fs::PermissionsExt;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn parses_plugin_installer_options() {
        let args = parse_installer_args([
            "--plugin",
            "plugins/demo",
            "--plugin",
            "plugins/other",
            "--runtime-dir",
            "/tmp/demo-runtime",
            "--requirements",
            "base.txt",
            "--requirements",
            "plugin.txt",
            "--python",
            "3.13",
            "--torch-backend",
            "cu128",
            "--skip-import-checks",
        ])
        .unwrap();

        assert_eq!(
            args.plugin_roots,
            [
                PathBuf::from("plugins/demo"),
                PathBuf::from("plugins/other")
            ]
        );
        assert_eq!(args.runtime_dir, Path::new("/tmp/demo-runtime"));
        assert_eq!(
            args.requirements,
            [PathBuf::from("base.txt"), PathBuf::from("plugin.txt")]
        );
        assert_eq!(args.python, "3.13");
        assert_eq!(args.torch_backend, "cu128");
        assert!(args.skip_import_checks);
    }

    #[test]
    fn requires_plugin_and_runtime_paths() {
        let error = parse_installer_args(["--plugin", "plugins/demo"]).unwrap_err();
        assert!(error.to_string().contains("--runtime-dir"));
    }

    #[test]
    fn reads_readiness_modules() {
        let manifest: Value = serde_yaml::from_str(
            "developer:\n  readiness:\n    python_modules:\n      - torch\n      - demo.runtime\n",
        )
        .unwrap();
        assert_eq!(
            readiness_modules(&manifest).unwrap(),
            ["jsonschema", "yaml", "torch", "demo.runtime"]
        );
    }

    #[test]
    fn runtime_import_path_includes_scripts_and_python_modules() {
        let root = Path::new("/runtime");
        let python_path = runtime_python_path(root).unwrap();

        assert_eq!(
            std::env::split_paths(&python_path).collect::<Vec<_>>(),
            [root.join("scripts"), root.join("python")]
        );
    }

    #[test]
    fn verifies_and_extracts_uv_archive() {
        let payload = b"#!/bin/sh\nexit 0\n";
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(payload.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        archive
            .append_data(&mut header, "uv-test-target/uv", payload.as_slice())
            .unwrap();
        let archive = archive.into_inner().unwrap().finish().unwrap();
        let digest = crate::digest::hex(Sha256::digest(&archive).as_ref());
        verify_download_checksum(&archive, &digest).unwrap();

        let root = tempdir().unwrap();
        let destination = root.path().join("uv");
        extract_uv(&archive, &destination).unwrap();

        assert_eq!(fs::read(&destination).unwrap(), payload);
        assert_ne!(
            fs::metadata(destination).unwrap().permissions().mode() & 0o111,
            0
        );
        assert!(verify_download_checksum(&archive, "not-a-checksum").is_err());
    }

    #[test]
    fn verifies_exact_bootstrapped_uv_version() {
        let root = tempdir().unwrap();
        let uv = root.path().join("uv");
        fs::write(&uv, format!("#!/bin/sh\necho 'uv {UV_VERSION}'\n")).unwrap();
        fs::set_permissions(&uv, fs::Permissions::from_mode(0o755)).unwrap();

        verify_uv_version(&uv, UV_VERSION).unwrap();
        let error = verify_uv_version(&uv, "0.0.0").unwrap_err();
        assert!(error.to_string().contains("unexpected version"));
    }

    #[test]
    fn rejects_malformed_uv_version_output() {
        let root = tempdir().unwrap();
        let uv = root.path().join("uv");
        fs::write(&uv, "#!/bin/sh\necho 'not-uv 0.11.16'\n").unwrap();
        fs::set_permissions(&uv, fs::Permissions::from_mode(0o755)).unwrap();

        let error = verify_uv_version(&uv, UV_VERSION).unwrap_err();
        assert!(error.to_string().contains("unexpected version"));
    }

    #[test]
    fn rejects_explicit_uv_with_an_incompatible_version() {
        let root = tempdir().unwrap();
        let uv = root.path().join("uv");
        fs::write(&uv, "#!/bin/sh\necho 'uv 0.10.0'\n").unwrap();
        fs::set_permissions(&uv, fs::Permissions::from_mode(0o755)).unwrap();

        let error = resolve_uv(Some(&uv)).unwrap_err();

        assert!(error.to_string().contains("must be version"));
    }

    #[test]
    fn removes_corrupt_cached_uv_installation() {
        let root = tempdir().unwrap();
        let install_dir = root.path().join(UV_VERSION);
        fs::create_dir(&install_dir).unwrap();
        fs::write(install_dir.join("uv"), b"corrupt").unwrap();

        quarantine_invalid_uv_install(&install_dir).unwrap();

        assert!(!install_dir.exists());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn preserves_valid_cached_uv_installation() {
        let root = tempdir().unwrap();
        let install_dir = root.path().join(UV_VERSION);
        fs::create_dir(&install_dir).unwrap();
        let uv = install_dir.join("uv");
        fs::write(&uv, format!("#!/bin/sh\necho 'uv {UV_VERSION}'\n")).unwrap();
        fs::set_permissions(&uv, fs::Permissions::from_mode(0o755)).unwrap();

        quarantine_invalid_uv_install(&install_dir).unwrap();

        assert!(install_dir.exists());
        verify_uv_version(&uv, UV_VERSION).unwrap();
    }

    #[test]
    fn installs_runtime_with_embedded_support_files() {
        let root = tempdir().unwrap();
        let plugin = root.path().join("plugin");
        fs::create_dir(&plugin).unwrap();
        fs::write(
            plugin.join("plugin.yaml"),
            "metadata:\n  id: demo\npipeline:\n  profile: simple\n",
        )
        .unwrap();
        let second_plugin = root.path().join("second-plugin");
        fs::create_dir(&second_plugin).unwrap();
        fs::write(
            second_plugin.join("plugin.yaml"),
            "metadata:\n  id: second\ndeveloper:\n  readiness:\n    python_modules:\n      - second_runtime\n",
        )
        .unwrap();
        let requirements = root.path().join("plugin-requirements.txt");
        fs::write(&requirements, "demo-package==1.0\n").unwrap();
        let fake_uv = root.path().join("uv");
        fs::write(
            &fake_uv,
            "#!/bin/sh\n\
             if [ \"$1\" = --version ]; then\n\
               echo 'uv 0.11.16'\n\
               exit 0\n\
             fi\n\
             if [ \"$1\" = venv ]; then\n\
               case \" $* \" in\n\
                 *\" --relocatable \"*) ;;\n\
                 *) echo 'missing --relocatable' >&2; exit 9 ;;\n\
               esac\n\
               runtime=\"$4\"\n\
               mkdir -p \"$runtime/bin\"\n\
               printf '#!/bin/sh\\nexit 0\\n' > \"$runtime/bin/python\"\n\
               chmod +x \"$runtime/bin/python\"\n\
             fi\n",
        )
        .unwrap();
        fs::set_permissions(&fake_uv, fs::Permissions::from_mode(0o755)).unwrap();
        let runtime = root.path().join("runtime");
        let args = InstallerArgs {
            plugin_roots: vec![plugin, second_plugin],
            runtime_dir: runtime.clone(),
            requirements: vec![requirements],
            python: "3.12".to_string(),
            torch_backend: "auto".to_string(),
            uv: Some(fake_uv),
            skip_import_checks: true,
            help: false,
        };

        let installed = install_runtime(&args).unwrap();

        assert_eq!(installed.plugin_ids, ["demo", "second"]);
        assert!(runtime.join("bin/python").is_file());
        assert!(runtime.join("scripts/plugin_direct_runner.py").is_file());
        assert!(
            runtime
                .join("python/physicsnemo_cfd_runtime/__init__.py")
                .is_file()
        );
        assert!(runtime.join("runtime-manifest.json").is_file());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(runtime.join("runtime-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest["requirements"],
            serde_json::json!(["plugin-requirements.txt"])
        );
        assert_eq!(
            manifest["plugin_ids"],
            serde_json::json!(["demo", "second"])
        );
        assert!(
            manifest["checked_modules"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("second_runtime"))
        );
    }
}
