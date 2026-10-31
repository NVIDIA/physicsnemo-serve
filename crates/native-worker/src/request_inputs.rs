//! Admit immutable release inputs and run-owned multipart files before C++ sees paths.
use crate::deployment::{InputArtifact, WorkflowDeploymentConfig};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
};

pub struct AdmittedInputs {
    pub paths: Map<String, Value>,
    pub metadata: Map<String, Value>,
    owned: Option<OwnedInputs>,
}
impl AdmittedInputs {
    pub(crate) fn take_owned(&mut self) -> Option<OwnedInputs> {
        self.owned.take()
    }
}

/// Ownership moves from admission into the queued native command. Dropping a
/// request future cannot remove inputs while C++ is still using them.
pub(crate) struct OwnedInputs {
    root: std::path::PathBuf,
}
impl Drop for OwnedInputs {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.root) {
            tracing::warn!(path=%self.root.display(),%error,"failed to remove admitted attempt inputs");
        }
    }
}

#[derive(Deserialize)]
struct UploadedArtifact {
    field_name: String,
    artifact_id: String,
    media_type: String,
    size_bytes: u64,
    storage_path: std::path::PathBuf,
    sha256: String,
}

async fn no_symlinks(path: &Path) -> Result<()> {
    ensure!(path.is_absolute(), "admitted input path must be absolute");
    let mut current = std::path::PathBuf::new();
    for part in path.components() {
        ensure!(
            !matches!(part, std::path::Component::ParentDir),
            "input path contains traversal"
        );
        current.push(part);
        ensure!(
            !fs::symlink_metadata(&current)
                .await?
                .file_type()
                .is_symlink(),
            "input path contains a symlink"
        );
    }
    Ok(())
}

/// Attempt copies remain owned during admission, queued submission and C++ use.
/// Their owner removes them once the actual native call finishes or is discarded.
pub async fn admit_inputs(
    config: &WorkflowDeploymentConfig,
    payload: &Value,
    run_id: &str,
    output_dir: &Path,
) -> Result<AdmittedInputs> {
    if let Some(version) = &config.manifest_version {
        ensure!(
            payload["manifest_version"].as_str() == Some(version),
            "request manifest_version differs from deployed workflow"
        );
    }
    let mut admitted = AdmittedInputs {
        paths: Map::new(),
        metadata: Map::new(),
        owned: None,
    };
    for (name, input) in &config.input_artifacts {
        let (path, size, digest) = match input {
            InputArtifact::Local(asset) => {
                crate::inputs::verify_file(&asset.path, asset.size_bytes, &asset.sha256).await?;
                (asset.path.clone(), asset.size_bytes, asset.sha256.clone())
            }
            InputArtifact::Remote(object) => (
                crate::inputs::materialize(object, &config.input_cache).await?,
                object.size_bytes,
                object.sha256.clone(),
            ),
        };
        admitted.paths.insert(name.clone(), json!(path));
        admitted.metadata.insert(
            name.clone(),
            json!({"artifact_id":name,"size_bytes":size,
            "sha256":digest,"media_type":"application/octet-stream","source":"release"}),
        );
    }
    let supplied = payload
        .pointer("/request/input_artifacts")
        .cloned()
        .unwrap_or(json!([]));
    let artifacts: Vec<UploadedArtifact> = serde_json::from_value(supplied)
        .context("invalid native request artifact descriptors (digest required)")?;
    ensure!(
        artifacts.len() <= config.input_files.len(),
        "undeclared or duplicate request files"
    );
    if !artifacts.is_empty() {
        ensure!(
            payload
                .pointer("/request/content_type")
                .and_then(Value::as_str)
                == Some("multipart/form-data"),
            "native request files require multipart ingress"
        );
        let run_key = format!("{:x}", Sha256::digest(run_id.as_bytes()));
        // ArtifactStore excludes .cache: ordinary output retention cannot unlink
        // bytes that a cancelled native call may still be reading.
        let root = config
            .artifact_root
            .join(".cache")
            .join("native-inputs")
            .join(run_key)
            .join(output_dir.file_name().context("attempt input directory")?);
        fs::create_dir_all(root.parent().context("attempt input parent")?).await?;
        fs::create_dir(&root).await?;
        admitted.owned = Some(OwnedInputs { root });
    }
    let mut names = BTreeSet::new();
    for artifact in artifacts {
        let name = &artifact.field_name;
        let rule = config
            .input_files
            .get(name)
            .context("undeclared native request file")?;
        ensure!(
            names.insert(name.clone()) && !admitted.paths.contains_key(name),
            "duplicate or conflicting native input name"
        );
        ensure!(
            rule.media_types.contains(&artifact.media_type),
            "request file media type is not admitted"
        );
        ensure!(
            artifact.size_bytes <= rule.max_size_bytes,
            "request file exceeds declared size limit"
        );
        crate::request::validate_digest(&artifact.sha256)?;
        uuid::Uuid::parse_str(&artifact.artifact_id).context("invalid request artifact ID")?;
        ensure!(
            !run_id.is_empty()
                && run_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_')),
            "invalid admitted run identity"
        );
        let root = config
            .ingress_root
            .as_ref()
            .context("missing trusted ingress root")?;
        no_symlinks(root).await?;
        no_symlinks(&artifact.storage_path).await?;
        let expected = fs::canonicalize(root.join(run_id)).await?;
        let source = fs::canonicalize(&artifact.storage_path).await?;
        ensure!(
            expected.starts_with(fs::canonicalize(root).await?)
                && source.parent() == Some(expected.as_path())
                && source
                    .file_name()
                    .and_then(|v| v.to_str())
                    .is_some_and(|v| v.starts_with(&(artifact.artifact_id.clone() + "-"))),
            "request artifact does not belong to this run"
        );
        ensure!(
            fs::symlink_metadata(&source).await?.is_file(),
            "request artifact must be a regular file"
        );
        ensure!(
            fs::metadata(&source).await?.len() == artifact.size_bytes,
            "request artifact size differs"
        );
        let owned = admitted
            .owned
            .as_ref()
            .expect("upload input root")
            .root
            .join(name);
        let mut input = fs::File::open(&source).await?;
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&owned)
            .await?;
        let mut digest = Sha256::new();
        let mut size = 0_u64;
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let count = input.read(&mut buffer).await?;
            if count == 0 {
                break;
            }
            size = size
                .checked_add(count as u64)
                .context("request file size overflow")?;
            ensure!(
                size <= artifact.size_bytes,
                "request artifact grew during admission"
            );
            output.write_all(&buffer[..count]).await?;
            digest.update(&buffer[..count]);
        }
        output.flush().await?;
        ensure!(
            size == artifact.size_bytes && format!("{:x}", digest.finalize()) == artifact.sha256,
            "request artifact SHA256/size mismatch"
        );
        drop(output);
        let mut permissions = fs::metadata(&owned).await?.permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&owned, permissions).await?;
        admitted.paths.insert(name.clone(), json!(owned));
        admitted.metadata.insert(
            name.clone(),
            json!({"artifact_id":artifact.artifact_id,"size_bytes":size,
            "sha256":artifact.sha256,"media_type":artifact.media_type,"source":"request"}),
        );
    }
    for (name, rule) in &config.input_files {
        ensure!(
            !rule.required || names.contains(name),
            "missing required native request file: {name}"
        );
    }
    validate_raw_inputs(config, &payload["parameters"], &admitted.paths).await?;
    Ok(admitted)
}

async fn validate_raw_inputs(
    config: &WorkflowDeploymentConfig,
    parameters: &Value,
    inputs: &Map<String, Value>,
) -> Result<()> {
    if config.workflow["adapter"] != "tensor.v1" {
        return Ok(());
    }
    let mapping = &config.workflow["tensor_adapter"];
    let Some(tensors) = mapping["inputs"].as_object() else {
        return Ok(());
    };
    for (name, entry) in tensors {
        if entry["codec"] != "raw-fp32-le" {
            continue;
        }
        let alias = mapping["model_alias"]
            .as_str()
            .context("raw tensor model alias")?;
        let package = config.workflow["models"][alias]["package_path"]
            .as_str()
            .context("raw tensor package")?;
        let manifest: Value =
            serde_json::from_slice(&fs::read(Path::new(package).join("model.json")).await?)?;
        let tensor = manifest["inputs"]
            .as_array()
            .context("model input signature")?
            .iter()
            .find(|spec| spec["name"].as_str() == Some(name))
            .context("unknown raw tensor name")?;
        let shape = tensor["shape"]
            .as_array()
            .context("static model input shape")?;
        let mut bytes = 4_u64;
        for dimension in shape {
            bytes = bytes
                .checked_mul(
                    dimension
                        .as_u64()
                        .filter(|n| *n > 0)
                        .context("positive static tensor dimension required")?,
                )
                .context("raw tensor size overflow")?;
        }
        let limit = config.workflow["limits"]["max_tensor_bytes"]
            .as_u64()
            .unwrap_or(64 * 1024 * 1024);
        ensure!(
            tensor["dtype"] == "float32" && bytes <= limit.min(256 * 1024 * 1024),
            "unsupported or oversized raw tensor"
        );
        let field = entry["request_field"]
            .as_str()
            .context("raw tensor request field")?;
        let descriptor = &parameters[field];
        ensure!(
            descriptor.as_object().is_some_and(|v| v.len() == 6)
                && descriptor["dtype"] == "float32"
                && descriptor["shape"] == tensor["shape"]
                && descriptor["byte_order"] == "little"
                && descriptor["layout"] == "contiguous"
                && descriptor["size_bytes"].as_u64() == Some(bytes),
            "raw tensor descriptor differs from compiled signature"
        );
        let admitted = descriptor["artifact"]
            .as_str()
            .context("raw tensor artifact name")?;
        let path = inputs
            .get(admitted)
            .and_then(Value::as_str)
            .context("raw tensor artifact is not admitted")?;
        ensure!(
            fs::metadata(path).await?.len() == bytes,
            "raw tensor file differs from compiled byte count"
        );
        let data = fs::read(path).await?;
        ensure!(
            data.chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().expect("four bytes")).is_finite()),
            "raw tensor contains non-finite float32"
        );
    }
    Ok(())
}
