fn main() {
    println!("cargo:rustc-check-cfg=cfg(native_workflow_only)");
    #[cfg(feature = "cpp")]
    {
        let workflow_only = std::env::var("NATIVE_WORKFLOW_ONLY").as_deref() == Ok("1");
        let runner_include = std::env::var("NATIVE_RUNNER_INCLUDE").ok();
        let runner_lib = std::env::var("NATIVE_RUNNER_LIB_DIR").ok();
        let workflow_include = std::env::var("NATIVE_WORKFLOW_INCLUDE")
            .ok()
            .or_else(|| runner_include.clone())
            .expect("cpp requires NATIVE_WORKFLOW_INCLUDE (installed workflow headers)");
        let workflow_lib = std::env::var("NATIVE_WORKFLOW_LIB_DIR")
            .ok()
            .or_else(|| runner_lib.clone())
            .expect("cpp requires NATIVE_WORKFLOW_LIB_DIR (installed workflow library)");
        let mut bridge = cxx_build::bridge("src/bridge.rs");
        bridge
            .file("src/bridge.cpp")
            .include("include")
            .include(workflow_include)
            .std("c++20");
        if workflow_only {
            bridge.define("PNMS_GENERIC_ONLY", None);
            println!("cargo:rustc-cfg=native_workflow_only");
        } else {
            bridge.include(
                runner_include
                    .expect("cpp requires NATIVE_RUNNER_INCLUDE (installed runner headers)"),
            );
            let lib =
                runner_lib.expect("cpp requires NATIVE_RUNNER_LIB_DIR (installed runner library)");
            println!("cargo:rustc-link-search=native={lib}");
            println!("cargo:rustc-link-lib=dylib=transolver_workflow");
            if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
            }
        }
        bridge.compile("native-worker-bridge");
        println!("cargo:rustc-link-search=native={workflow_lib}");
        println!("cargo:rustc-link-lib=dylib=physicsnemo_workflow");
        if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{workflow_lib}");
        }
        println!("cargo:rerun-if-env-changed=NATIVE_WORKFLOW_ONLY");
        println!("cargo:rerun-if-env-changed=NATIVE_WORKFLOW_INCLUDE");
        println!("cargo:rerun-if-env-changed=NATIVE_WORKFLOW_LIB_DIR");
        println!("cargo:rerun-if-env-changed=NATIVE_RUNNER_INCLUDE");
        println!("cargo:rerun-if-env-changed=NATIVE_RUNNER_LIB_DIR");
        for file in [
            "src/bridge.rs",
            "src/bridge.cpp",
            "include/native_worker/bridge.hpp",
        ] {
            println!("cargo:rerun-if-changed={file}");
        }
    }
}
