use crate::deployment::{DeviceSnapshot, RuntimeDeploymentConfig};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

#[derive(Debug, PartialEq, Eq)]
enum DeviceSelector {
    Index(u32),
    Uuid(String),
}
fn device_selector(index: u32, uuid: Option<&str>) -> Result<DeviceSelector> {
    if let Some(value) = uuid {
        let raw = value
            .strip_prefix("GPU-")
            .context("PNMS_DEVICE_UUID must be an NVML GPU UUID")?;
        ensure!(
            raw.len() == 36 && uuid::Uuid::parse_str(raw).is_ok(),
            "PNMS_DEVICE_UUID must be a full NVML GPU UUID; MIG targets are unsupported"
        );
        return Ok(DeviceSelector::Uuid(value.into()));
    }
    Ok(DeviceSelector::Index(index))
}

fn mig_mode(status: u32, current: u32, pending: u32, user_build: bool) -> Result<bool> {
    // NVML_ERROR_NOT_SUPPORTED (3): this GPU product cannot enable MIG. Keep
    // the older qualified H100 path's successful-query requirement unchanged.
    if status == 3 && user_build {
        return Ok(false);
    }
    ensure!(
        status == 0,
        "unable to verify MIG mode: NVML status {status}"
    );
    Ok(current != 0 || pending != 0)
}

#[derive(Clone, Serialize)]
pub struct AdmittedDevice {
    /// Logical scheduler resource class. Explicit CPU fixtures emulate a GPU
    /// slot, while physical_device_kind/qualification retain the actual scope.
    pub device_kind: String,
    pub physical_device_kind: String,
    pub qualification: String,
    pub device_name: String,
    pub device_uuid: String,
    pub memory_mb: u64,
    pub available_memory_mb: u64,
    pub driver_version: String,
    pub compute_capability: [i32; 2],
    pub mig_enabled: bool,
}
fn platform() -> String {
    let arch = match std::env::consts::ARCH {
        "x86_64" => "amd64",
        other => other,
    };
    format!("{}/{arch}", std::env::consts::OS)
}
pub fn admit(config: &RuntimeDeploymentConfig) -> Result<AdmittedDevice> {
    if let RuntimeDeploymentConfig::Workflow(c) = config
        && c.target.is_cpu_fixture()
    {
        ensure!(
            std::env::var("NATIVE_ALLOW_CPU_TEST_ONLY").as_deref() == Ok("1"),
            "CPU fixture execution requires NATIVE_ALLOW_CPU_TEST_ONLY=1 and is not production qualification"
        );
        let memory = std::fs::read_to_string("/proc/meminfo")?;
        let available = memory
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .context("CPU available memory")?
            .parse::<u64>()?
            / 1024;
        c.target.admit(&DeviceSnapshot {
            platform: platform(),
            product: "cpu-test-only".into(),
            compute_capability: [0, 0],
            mig_enabled: false,
            available_memory_mb: available,
            driver_version: "0".into(),
        })?;
        return Ok(AdmittedDevice {
            device_kind: "gpu".into(),
            physical_device_kind: "cpu".into(),
            qualification: "cpu-test-only".into(),
            device_name: "cpu-test-only".into(),
            device_uuid: format!("cpu-test-only:{}", config.device_index()),
            memory_mb: available,
            available_memory_mb: available,
            driver_version: "0".into(),
            compute_capability: [0, 0],
            mig_enabled: false,
        });
    }
    let allocated_uuid = std::env::var_os("PNMS_DEVICE_UUID")
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("PNMS_DEVICE_UUID must be Unicode"))
        })
        .transpose()?;
    let selector = device_selector(config.device_index(), allocated_uuid.as_deref())?;
    let nvml = nvml_wrapper::Nvml::init().or_else(|_| {
        nvml_wrapper::Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init()
    })?;
    let device = match selector {
        DeviceSelector::Uuid(uuid) => nvml.device_by_uuid(uuid)?,
        DeviceSelector::Index(index) => nvml.device_by_index(index)?,
    };
    let capability = device.cuda_compute_capability()?;
    let product = device.name()?;
    let memory = device.memory_info()?;
    let driver = nvml.sys_driver_version()?;
    // nvml-wrapper 0.10 does not yet expose this NVML entry point. Query the
    // same already-initialized library with the wrapper's live device handle.
    let mig_enabled = unsafe {
        let library = libloading::Library::new("libnvidia-ml.so.1")?;
        type MigMode = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32, *mut u32) -> u32;
        let query = library.get::<MigMode>(b"nvmlDeviceGetMigMode\0")?;
        let (mut current, mut pending) = (0, 0);
        let status = query(device.handle().cast(), &mut current, &mut pending);
        mig_mode(
            status,
            current,
            pending,
            matches!(config,
            RuntimeDeploymentConfig::Workflow(c) if c.target.profile == "build-product-v1"),
        )?
    };
    let snapshot = DeviceSnapshot {
        platform: platform(),
        product: product.clone(),
        compute_capability: [capability.major, capability.minor],
        mig_enabled,
        available_memory_mb: memory.free / (1024 * 1024),
        driver_version: driver.clone(),
    };
    match config {
        RuntimeDeploymentConfig::Workflow(c) => c.target.admit(&snapshot)?,
        RuntimeDeploymentConfig::Transolver(c) => {
            if let Some(target) = &c.target {
                target.admit(&snapshot)?;
            }
            ensure!(
                snapshot.platform == "linux/amd64",
                "native target requires Linux amd64"
            );
            ensure!(
                snapshot.product == "NVIDIA H100 80GB HBM3"
                    && snapshot.compute_capability == c.compute_capability,
                "legacy release is qualified only for the exact H100 product/sm90"
            );
            ensure!(
                !snapshot.mig_enabled,
                "legacy release is not qualified for MIG"
            );
        }
    }
    Ok(AdmittedDevice {
        device_kind: "gpu".into(),
        physical_device_kind: "gpu".into(),
        qualification: "hardware-policy-admitted".into(),
        device_name: product,
        device_uuid: device.uuid()?,
        memory_mb: memory.total / (1024 * 1024),
        available_memory_mb: snapshot.available_memory_mb,
        driver_version: driver,
        compute_capability: snapshot.compute_capability,
        mig_enabled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocated_device_uuid_overrides_physical_index_for_nvml() {
        let uuid = "GPU-12345678-1234-5678-9abc-123456789abc";
        assert_eq!(
            device_selector(0, Some(uuid)).unwrap(),
            DeviceSelector::Uuid(uuid.into()),
            "CUDA logical device zero must retain the allocated physical GPU identity"
        );
        assert_eq!(device_selector(2, None).unwrap(), DeviceSelector::Index(2));
        for invalid in [
            "",
            "0",
            "GPU-not-a-uuid",
            "MIG-12345678-1234-5678-9abc-123456789abc",
        ] {
            assert!(
                device_selector(0, Some(invalid)).is_err(),
                "invalid UUID: {invalid}"
            );
        }
    }

    #[test]
    fn gpu_without_mig_capability_is_allowed_only_by_user_build_policy() {
        assert_eq!(
            mig_mode(3, 0, 0, true).ok(),
            Some(false),
            "NVML NOT_SUPPORTED means this product cannot enable MIG"
        );
        assert!(mig_mode(3, 0, 0, false).is_err());
        assert!(mig_mode(999, 0, 0, true).is_err());
        assert!(mig_mode(0, 0, 1, true).unwrap());
    }
}
