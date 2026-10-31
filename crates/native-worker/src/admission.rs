use crate::deployment::{DeviceSnapshot, RuntimeDeploymentConfig};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

#[repr(C)]
struct CudaUuid {
    bytes: [u8; 16],
}
type CudaInit = unsafe extern "system" fn(u32) -> i32;
type CudaDeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
type CudaDeviceGetUuid = unsafe extern "system" fn(*mut CudaUuid, i32) -> i32;

fn cuda_device_uuid_with(
    index: u32,
    init: CudaInit,
    get_device: CudaDeviceGet,
    get_uuid: CudaDeviceGetUuid,
) -> Result<[u8; 16]> {
    let ordinal = i32::try_from(index).context("CUDA logical device index exceeds i32")?;
    // Driver API queries create no inference context. CUDA applies its own
    // visibility/order rules; NVML physical indices need not follow that order.
    let status = unsafe { init(0) };
    ensure!(status == 0, "cuInit failed with CUDA status {status}");
    let mut device = 0;
    let status = unsafe { get_device(&mut device, ordinal) };
    ensure!(
        status == 0,
        "CUDA logical device {index} is unavailable (status {status})"
    );
    let mut uuid = CudaUuid { bytes: [0; 16] };
    let status = unsafe { get_uuid(&mut uuid, device) };
    ensure!(
        status == 0,
        "CUDA logical device {index} UUID query failed (status {status})"
    );
    Ok(uuid.bytes)
}

fn cuda_device_uuid(index: u32) -> Result<[u8; 16]> {
    #[cfg(target_os = "windows")]
    const LIBRARY: &str = "nvcuda.dll";
    #[cfg(not(target_os = "windows"))]
    const LIBRARY: &str = "libcuda.so.1";
    // These symbols use the CUDA driver ABI; the library remains loaded while
    // they are called, and every output buffer has the corresponding C layout.
    unsafe {
        let library =
            libloading::Library::new(LIBRARY).context("load CUDA driver for device admission")?;
        cuda_device_uuid_with(
            index,
            *library.get::<CudaInit>(b"cuInit\0")?,
            *library.get::<CudaDeviceGet>(b"cuDeviceGet\0")?,
            // v2 preserves MIG instance identity instead of substituting a parent.
            *library.get::<CudaDeviceGetUuid>(b"cuDeviceGetUuid_v2\0")?,
        )
    }
}

fn verified_device_uuid(cuda_uuid: &str, asserted_uuid: Option<&str>) -> Result<String> {
    let parse = |value: &str| -> Result<uuid::Uuid> {
        let raw = value
            .strip_prefix("GPU-")
            .context("device UUID must be a full NVML GPU UUID; MIG targets are unsupported")?;
        ensure!(raw.len() == 36, "device UUID must be a full NVML GPU UUID");
        let uuid = uuid::Uuid::parse_str(raw).context("invalid NVML GPU UUID")?;
        ensure!(!uuid.is_nil(), "CUDA device UUID cannot be nil");
        Ok(uuid)
    };
    let actual = parse(cuda_uuid).context("invalid CUDA device UUID")?;
    if let Some(value) = asserted_uuid {
        ensure!(
            parse(value).context("invalid PNMS_DEVICE_UUID")? == actual,
            "PNMS_DEVICE_UUID differs from CUDA selected device {cuda_uuid}"
        );
    }
    Ok(format!("GPU-{actual}"))
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
    let cuda_uuid = uuid::Uuid::from_bytes(cuda_device_uuid(config.device_index())?);
    let selected_uuid =
        verified_device_uuid(&format!("GPU-{cuda_uuid}"), allocated_uuid.as_deref())?;
    let nvml = nvml_wrapper::Nvml::init().or_else(|_| {
        nvml_wrapper::Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init()
    })?;
    let device = nvml.device_by_uuid(selected_uuid.as_str())?;
    let physical_uuid = device.uuid()?;
    ensure!(
        physical_uuid == selected_uuid,
        "CUDA selected UUID must identify the same physical NVML GPU; MIG targets are unsupported"
    );
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
        device_uuid: physical_uuid,
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
    fn cuda_logical_ordinal_uses_the_resolved_uuid_for_nvml() {
        let physical_one = "GPU-12345678-1234-5678-9abc-123456789abc";
        assert_eq!(
            verified_device_uuid(physical_one, None).unwrap(),
            physical_one,
            "CUDA logical zero may refer to physical GPU one"
        );
    }

    #[test]
    fn explicit_uuid_must_match_the_cuda_selected_device() {
        let actual = "GPU-12345678-1234-5678-9abc-123456789abc";
        let other = "GPU-87654321-4321-8765-abcd-cba987654321";
        assert!(
            verified_device_uuid(actual, Some(other)).is_err(),
            "an override admitted a different GPU"
        );
        assert_eq!(verified_device_uuid(actual, Some(actual)).unwrap(), actual);
        for invalid in [
            "",
            "0",
            "GPU-not-a-uuid",
            "MIG-12345678-1234-5678-9abc-123456789abc",
        ] {
            assert!(verified_device_uuid(actual, Some(invalid)).is_err());
        }
    }

    unsafe extern "system" fn init_ok(flags: u32) -> i32 {
        assert_eq!(flags, 0);
        0
    }
    unsafe extern "system" fn device_ok(device: *mut i32, ordinal: i32) -> i32 {
        assert_eq!(ordinal, 2);
        unsafe { *device = 77 };
        0
    }
    unsafe extern "system" fn uuid_ok(uuid: *mut CudaUuid, device: i32) -> i32 {
        assert_eq!(device, 77);
        unsafe { (*uuid).bytes = [42; 16] };
        0
    }
    unsafe extern "system" fn init_failed(_: u32) -> i32 {
        100
    }
    unsafe extern "system" fn device_failed(_: *mut i32, _: i32) -> i32 {
        101
    }
    unsafe extern "system" fn uuid_failed(_: *mut CudaUuid, _: i32) -> i32 {
        999
    }

    #[test]
    fn cuda_driver_resolves_the_configured_logical_ordinal_and_returned_handle() {
        assert_eq!(
            cuda_device_uuid_with(2, init_ok, device_ok, uuid_ok).unwrap(),
            [42; 16]
        );
    }

    #[test]
    fn cuda_driver_errors_never_fall_back_to_a_physical_index() {
        for (index, init, device, uuid) in [
            (
                2,
                init_failed as CudaInit,
                device_ok as CudaDeviceGet,
                uuid_ok as CudaDeviceGetUuid,
            ),
            (2, init_ok, device_failed, uuid_ok),
            (2, init_ok, device_ok, uuid_failed),
            (u32::MAX, init_ok, device_ok, uuid_ok),
        ] {
            assert!(cuda_device_uuid_with(index, init, device, uuid).is_err());
        }
        assert!(verified_device_uuid("GPU-00000000-0000-0000-0000-000000000000", None).is_err());
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
