/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::env;
use std::process::ExitCode;

use physicsnemo_serve_cmd::installer::{INSTALLER_USAGE, install_runtime, parse_installer_args};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let args = parse_installer_args(env::args().skip(1))?;
    if args.help {
        println!("{INSTALLER_USAGE}");
        return Ok(());
    }
    let installed = install_runtime(&args)?;
    println!(
        "Runtime for plugin(s) '{}' installed at {}",
        installed.plugin_ids.join(", "),
        installed.runtime_dir.display()
    );
    Ok(())
}
