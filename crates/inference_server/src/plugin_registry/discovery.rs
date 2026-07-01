/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

pub fn resolve_plugin_dirs<F>(env_provider: &F) -> Vec<PathBuf>
where
    F: Fn(&str) -> Option<String>,
{
    let Some(env_value) = env_provider("PLUGIN_DIR") else {
        return Vec::new();
    };

    let path_separator = if cfg!(windows) { ';' } else { ':' };
    let normalized = env_value.replace(',', &path_separator.to_string());
    let mut seen = HashSet::new();
    std::env::split_paths(&OsString::from(normalized))
        .filter(|path| !path.as_os_str().is_empty())
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

pub fn discover_plugins(plugin_dirs: &[PathBuf]) -> Result<Vec<RegisteredPlugin>> {
    discover_plugins_with_enabled_id(plugin_dirs, None)
}

pub fn discover_plugins_with_enabled_id(
    plugin_dirs: &[PathBuf],
    enabled_plugin_id: Option<&str>,
) -> Result<Vec<RegisteredPlugin>> {
    let enabled_plugin_id = enabled_plugin_id
        .map(str::trim)
        .filter(|plugin_id| !plugin_id.is_empty());
    let mut candidate_roots = Vec::new();
    let mut seen_roots = HashSet::new();

    let mut sorted_plugin_dirs = plugin_dirs.to_vec();
    sorted_plugin_dirs.sort();

    for plugin_dir in sorted_plugin_dirs {
        if !plugin_dir.exists() {
            continue;
        }

        if plugin_manifest_path(&plugin_dir).is_file() && seen_roots.insert(plugin_dir.clone()) {
            candidate_roots.push(plugin_dir.clone());
        }

        let read_dir = fs::read_dir(&plugin_dir).with_context(|| {
            format!("failed to read plugin directory '{}'", plugin_dir.display())
        })?;
        let mut child_dirs = Vec::new();
        for entry in read_dir {
            let entry = entry.with_context(|| {
                format!(
                    "failed to read an entry from plugin directory '{}'",
                    plugin_dir.display()
                )
            })?;
            let path = entry.path();
            if path.is_dir() && plugin_manifest_path(&path).is_file() {
                child_dirs.push(path);
            }
        }
        child_dirs.sort();
        for child_dir in child_dirs {
            if seen_roots.insert(child_dir.clone()) {
                candidate_roots.push(child_dir);
            }
        }
    }

    let mut plugins = Vec::new();
    let mut manifest_ids = HashMap::new();
    for root_dir in candidate_roots {
        let manifest_path = plugin_manifest_path(&root_dir);
        let manifest_yaml = fs::read_to_string(&manifest_path).with_context(|| {
            format!(
                "failed to read plugin manifest '{}'",
                manifest_path.display()
            )
        })?;
        let manifest = PluginManifest::from_yaml_str(&manifest_yaml).with_context(|| {
            format!(
                "failed to load plugin manifest '{}'",
                manifest_path.display()
            )
        })?;
        if enabled_plugin_id.is_some_and(|plugin_id| plugin_id != manifest.metadata.id) {
            continue;
        }
        manifest.validate().with_context(|| {
            format!(
                "plugin manifest '{}' failed validation",
                manifest_path.display()
            )
        })?;

        if let Some(existing_root) =
            manifest_ids.insert(manifest.metadata.id.clone(), root_dir.clone())
        {
            bail!(
                "duplicate plugin id '{}' discovered in '{}' and '{}'",
                manifest.metadata.id,
                existing_root.display(),
                root_dir.display()
            );
        }

        plugins.push(RegisteredPlugin {
            root_dir,
            manifest_path,
            manifest,
        });
    }

    if let Some(plugin_id) = enabled_plugin_id
        && plugins.is_empty()
    {
        bail!(
            "enabled plugin id '{}' was not discovered in configured plugin directories",
            plugin_id
        );
    }

    Ok(plugins)
}

pub fn plugin_manifest_path(root: &Path) -> PathBuf {
    root.join(DEFAULT_PLUGIN_MANIFEST_NAME)
}

/// Resolve a script path by name, trying the compile-time repo layout first,
/// then falling back to the deployed layout (`../scripts/` relative to the
/// running binary, i.e. `/app/scripts/` when the binary is `/app/bin/*`).
pub fn resolve_script_path(script_name: &str) -> PathBuf {
    let compile_time = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts")
        .join(script_name);
    if compile_time.is_file() {
        return compile_time;
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(bin_dir) = exe.parent()
    {
        let deployed = bin_dir.join("../scripts").join(script_name);
        if deployed.is_file() {
            return deployed;
        }
    }
    compile_time
}
