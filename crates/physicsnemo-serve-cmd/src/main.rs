/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use physicsnemo_serve_cmd::bundle::{extract_runtime, package_executable_with_compression};
use physicsnemo_serve_cmd::prefetch::{materialize_direct_plan, read_prefetch_plan};
use physicsnemo_serve_cmd::{CliCommand, InferArgs, PackageArgs, PrefetchArgs, USAGE, parse_args};
use serde_json::Value;
use tokio::process::Command;
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};

const RUNTIME_OVERRIDE_ENV: &str = "PHYSICSNEMO_SERVE_RUNTIME_DIR";
const CACHE_OVERRIDE_ENV: &str = "PHYSICSNEMO_SERVE_CLI_CACHE_DIR";

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<u8> {
    let command = parse_args(env::args().skip(1)).map_err(|error| anyhow!("{error}\n\n{USAGE}"))?;
    match command {
        CliCommand::Help => {
            println!("{USAGE}");
            Ok(0)
        }
        CliCommand::Version => {
            println!("physicsnemo-serve {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        CliCommand::Infer(args) => run_inference(args).await,
        CliCommand::Package(args) => {
            package_runtime(args)?;
            Ok(0)
        }
        CliCommand::Prefetch(args) => {
            run_prefetch(args).await?;
            Ok(0)
        }
    }
}

async fn run_inference(args: InferArgs) -> Result<u8> {
    let executable = env::current_exe().context("failed to resolve the CLI executable")?;
    let runtime_root = resolve_runtime(&executable, args.runtime_dir.as_deref())?;
    let python = runtime_root.join("bin/python");
    let runner = runtime_root.join("scripts/plugin_direct_runner.py");
    fs::create_dir_all(&args.output_dir).with_context(|| {
        format!(
            "failed to create output directory: {}",
            args.output_dir.display()
        )
    })?;
    let run_id = args.run_id.unwrap_or_else(default_run_id);

    let mut command = Command::new(&python);
    command
        .arg(&runner)
        .arg("--plugin-root")
        .arg(&args.plugin_root)
        .arg("--request")
        .arg(&args.request)
        .arg("--output-dir")
        .arg(&args.output_dir)
        .arg("--run-id")
        .arg(&run_id)
        .env("PHYSICSNEMO_SERVE_PREFETCH_HELPER", &executable)
        .env("PYTHONNOUSERSITE", "1")
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME");
    if let Some(device) = args.device {
        command.env("CUDA_VISIBLE_DEVICES", device);
    }

    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(command.as_std_mut(), 0);
    #[cfg(unix)]
    let mut runner_signals = RunnerSignals::new()?;
    command.kill_on_drop(true);

    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start bundled Python: {}", python.display()))?;
    #[cfg(unix)]
    let status = wait_for_runner(&mut child, &mut runner_signals).await?;
    #[cfg(not(unix))]
    let status = child.wait().await?;
    Ok(status.code().unwrap_or(1).clamp(0, u8::MAX as i32) as u8)
}

#[cfg(unix)]
struct RunnerSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl RunnerSignals {
    fn new() -> Result<Self> {
        Ok(Self {
            interrupt: signal(SignalKind::interrupt())?,
            terminate: signal(SignalKind::terminate())?,
        })
    }
}

#[cfg(unix)]
async fn wait_for_runner(
    child: &mut tokio::process::Child,
    signals: &mut RunnerSignals,
) -> Result<std::process::ExitStatus> {
    let process_group = child
        .id()
        .filter(|pid| *pid <= i32::MAX as u32)
        .ok_or_else(|| anyhow!("failed to resolve the bundled Python process group"))?
        as i32;

    tokio::select! {
        status = child.wait() => Ok(status?),
        _ = signals.interrupt.recv() => terminate_process_group(child, process_group, libc::SIGINT).await,
        _ = signals.terminate.recv() => terminate_process_group(child, process_group, libc::SIGTERM).await,
    }
}

#[cfg(unix)]
async fn terminate_process_group(
    child: &mut tokio::process::Child,
    process_group: i32,
    signal: i32,
) -> Result<std::process::ExitStatus> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    send_process_group_signal(process_group, signal)?;
    match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(status) => {
            let status = status?;
            if !wait_for_process_group_exit(process_group, deadline).await? {
                send_process_group_signal(process_group, libc::SIGKILL)?;
            }
            Ok(status)
        }
        Err(_) => {
            send_process_group_signal(process_group, libc::SIGKILL)?;
            Ok(child.wait().await?)
        }
    }
}

#[cfg(unix)]
async fn wait_for_process_group_exit(
    process_group: i32,
    deadline: tokio::time::Instant,
) -> Result<bool> {
    loop {
        if !process_group_exists(process_group)? {
            return Ok(true);
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        tokio::time::sleep_until(std::cmp::min(
            deadline,
            now + std::time::Duration::from_millis(50),
        ))
        .await;
    }
}

#[cfg(unix)]
fn process_group_exists(process_group: i32) -> Result<bool> {
    // SAFETY: signal 0 performs an existence/permission check without
    // delivering a signal. The negative id targets the runner's process group.
    let result = unsafe { libc::kill(-process_group, 0) };
    if result == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(error).context("failed to inspect the bundled Python process group"),
    }
}

#[cfg(unix)]
fn send_process_group_signal(process_group: i32, signal: i32) -> Result<()> {
    // SAFETY: process_group comes from Child::id, is bounded to i32, and its
    // negative value deliberately targets the process group created above.
    let result = unsafe { libc::kill(-process_group, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error).context("failed to signal the bundled Python process group")
    }
}

fn package_runtime(args: PackageArgs) -> Result<()> {
    let executable = env::current_exe().context("failed to resolve the CLI executable")?;
    package_executable_with_compression(
        &executable,
        &args.runtime_dir,
        &args.output,
        args.compression_level,
    )?;
    Ok(())
}

async fn run_prefetch(args: PrefetchArgs) -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::ERROR)
        .with_ansi(false)
        .with_writer(io::stderr)
        .try_init();
    let plan: Value = read_prefetch_plan(io::stdin().lock())?;
    let result = materialize_direct_plan(plan, &args.cache_dir, &args.run_id).await?;
    serde_json::to_writer(io::stdout().lock(), &result)?;
    println!();
    Ok(())
}

fn resolve_runtime(executable: &Path, runtime_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(runtime_dir) = runtime_dir {
        validate_runtime(runtime_dir).context("runtime supplied with --runtime-dir is invalid")?;
        return Ok(runtime_dir.to_path_buf());
    }
    if let Some(runtime_dir) = env::var_os(RUNTIME_OVERRIDE_ENV) {
        let runtime_dir = PathBuf::from(runtime_dir);
        validate_runtime(&runtime_dir)
            .with_context(|| format!("runtime supplied by {RUNTIME_OVERRIDE_ENV} is invalid"))?;
        return Ok(runtime_dir);
    }
    let cache_root = if let Some(cache_dir) = env::var_os(CACHE_OVERRIDE_ENV) {
        PathBuf::from(cache_dir)
    } else {
        dirs::cache_dir()
            .ok_or_else(|| anyhow!("could not determine the runtime cache directory"))?
            .join("physicsnemo-serve/inference-cli")
    };
    let runtime_dir = extract_runtime(executable, &cache_root)?;
    validate_runtime(&runtime_dir).context("bundled runtime is invalid")?;
    Ok(runtime_dir)
}

fn validate_runtime(runtime_root: &Path) -> Result<()> {
    for relative_path in ["bin/python", "scripts/plugin_direct_runner.py"] {
        let path = runtime_root.join(relative_path);
        if !path.is_file() {
            return Err(anyhow!(
                "runtime is missing '{}': {}",
                relative_path,
                path.display()
            ));
        }
    }
    Ok(())
}

fn default_run_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("direct-{timestamp}-{}", std::process::id())
}
