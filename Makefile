# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read a value from deploy/config.yaml (flat key: "value" format).
_cfg = $(shell grep '^$(1):' deploy/config.yaml 2>/dev/null | head -1 | sed 's/^[^:]*:[[:space:]]*//' | sed 's/"//g' | sed "s/'//g" | sed 's/[[:space:]]*\#.*//')

DOCKER_REPO ?= $(call _cfg,docker_registry)
IMAGE_NAME ?= $(DOCKER_REPO)/$(call _cfg,image_name)
IMAGE_TAG = v0.1.20260714.0
RUNTIME_BASE_IMAGE_NAME ?= $(DOCKER_REPO)/$(call _cfg,runtime_base_image)
RUNTIME_BASE_IMAGE_TAG = pytorch-26.01-py3-th0.8.0
RUNTIME_BASE_IMAGE = $(RUNTIME_BASE_IMAGE_NAME):$(RUNTIME_BASE_IMAGE_TAG)
SERVE_CMD_DIST_DIR ?= dist
SERVE_CMD_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve
SERVE_CMD_LINUX_AMD64_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve-linux-amd64
SERVE_INSTALLER_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve-install
SERVE_INSTALLER_LINUX_AMD64_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve-install-linux-amd64
SERVE_CMD_LINUX_AMD64_TARGET ?= x86_64-unknown-linux-gnu.2.17
SERVE_CMD_LINUX_AMD64_TARGET_DIR ?= x86_64-unknown-linux-gnu
SERVE_CMD_RUST_TOOLCHAIN ?= 1.94.1

.PHONY: image runtime-base-image build build-serve-cmd build-serve-cmd-linux-amd64 build-serve-installer build-serve-installer-linux-amd64 clean clean-all experiments observe stress

image: runtime-base-image
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build --build-arg PHYSICSNEMO_SERVE_RUNTIME_BASE_IMAGE=$(RUNTIME_BASE_IMAGE) -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile.Earth2Studio.scicomp-rust-slim .

runtime-base-image:
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build --build-arg PYTORCH_BASE_IMAGE=$(DOCKER_REPO)/pytorch:26.01-py3 -t $(RUNTIME_BASE_IMAGE) -f Dockerfile.Earth2Studio.runtime-base .

build:
	cargo build --release -p inference_server -p worker-runtime

build-serve-cmd:
	cargo build --locked --release --package physicsnemo-serve-cmd --bin physicsnemo-serve
	mkdir -p $(SERVE_CMD_DIST_DIR)
	cp target/release/physicsnemo-serve $(SERVE_CMD_OUTPUT)
	chmod +x $(SERVE_CMD_OUTPUT)
	@echo "Thin CLI built: $(SERVE_CMD_OUTPUT)"
	@echo "Run it with: $(SERVE_CMD_OUTPUT) infer --runtime-dir <DIR> ..."

build-serve-cmd-linux-amd64:
	@command -v zig >/dev/null 2>&1 || (echo "zig is required; install it with: brew install zig" && exit 1)
	@command -v cmake >/dev/null 2>&1 || (echo "cmake is required by aws-lc-sys; install it with: brew install cmake" && exit 1)
	@command -v cargo-zigbuild >/dev/null 2>&1 || (echo "cargo-zigbuild is required; install it with: cargo install cargo-zigbuild --locked" && exit 1)
	rustup toolchain install $(SERVE_CMD_RUST_TOOLCHAIN) --profile minimal
	rustup target add --toolchain $(SERVE_CMD_RUST_TOOLCHAIN) $(SERVE_CMD_LINUX_AMD64_TARGET_DIR)
	cargo +$(SERVE_CMD_RUST_TOOLCHAIN) zigbuild --locked --release --target $(SERVE_CMD_LINUX_AMD64_TARGET) --package physicsnemo-serve-cmd --bin physicsnemo-serve
	mkdir -p $(SERVE_CMD_DIST_DIR)
	cp target/$(SERVE_CMD_LINUX_AMD64_TARGET_DIR)/release/physicsnemo-serve $(SERVE_CMD_LINUX_AMD64_OUTPUT)
	chmod +x $(SERVE_CMD_LINUX_AMD64_OUTPUT)
	@echo "Thin Linux x86_64 CLI built: $(SERVE_CMD_LINUX_AMD64_OUTPUT)"
	@echo "Run it with: $(SERVE_CMD_LINUX_AMD64_OUTPUT) infer --runtime-dir <DIR> ..."

build-serve-installer:
	cargo build --locked --release --package physicsnemo-serve-cmd --bin physicsnemo-serve-install
	mkdir -p $(SERVE_CMD_DIST_DIR)
	cp target/release/physicsnemo-serve-install $(SERVE_INSTALLER_OUTPUT)
	chmod +x $(SERVE_INSTALLER_OUTPUT)
	@echo "Runtime installer built: $(SERVE_INSTALLER_OUTPUT)"

build-serve-installer-linux-amd64:
	@command -v zig >/dev/null 2>&1 || (echo "zig is required; install it with: brew install zig" && exit 1)
	@command -v cmake >/dev/null 2>&1 || (echo "cmake is required by aws-lc-sys; install it with: brew install cmake" && exit 1)
	@command -v cargo-zigbuild >/dev/null 2>&1 || (echo "cargo-zigbuild is required; install it with: cargo install cargo-zigbuild --locked" && exit 1)
	rustup toolchain install $(SERVE_CMD_RUST_TOOLCHAIN) --profile minimal
	rustup target add --toolchain $(SERVE_CMD_RUST_TOOLCHAIN) $(SERVE_CMD_LINUX_AMD64_TARGET_DIR)
	cargo +$(SERVE_CMD_RUST_TOOLCHAIN) zigbuild --locked --release --target $(SERVE_CMD_LINUX_AMD64_TARGET) --package physicsnemo-serve-cmd --bin physicsnemo-serve-install
	mkdir -p $(SERVE_CMD_DIST_DIR)
	cp target/$(SERVE_CMD_LINUX_AMD64_TARGET_DIR)/release/physicsnemo-serve-install $(SERVE_INSTALLER_LINUX_AMD64_OUTPUT)
	chmod +x $(SERVE_INSTALLER_LINUX_AMD64_OUTPUT)
	@echo "Linux x86_64 runtime installer built: $(SERVE_INSTALLER_LINUX_AMD64_OUTPUT)"

test-rust:
	cargo test -p physicsnemo-serve-cmd --lib
	cargo test -p worker-runtime --lib
	cargo test -p e2s_zarr_io --lib
	cargo test -p inference_server --lib
	cargo test -p scicomp-rq --lib

clean:
	cargo clean

clean-all:
	cargo clean
	rm -rf artifacts/ outputs/

experiments:
	cd tests/e2e && go build -o ../../bin/run_experiments ./run_experiments.go
	@echo "Binary built: bin/run_experiments"
	@echo "Usage: bin/run_experiments --service_url_rust <RUST_URL> --service_url_python <PYTHON_URL> --ep_token <TOKEN> [--expt 1|2|3|all]"

# Run this command and browse on localhost:3000
observe:
	@test -n "$(SERVICE_URL)" || (echo "SERVICE_URL is not set!" && exit 1)
	@test -n "$(EP_TOKEN)" || (echo "EP_TOKEN is not set!" && exit 1)
	SERVICE_URL=$(SERVICE_URL) EP_TOKEN=$(EP_TOKEN) docker compose -f observability/docker-compose.yml up

stress:
	cd tests/e2e && go build -o ../../bin/stress_service ./stress_service.go
	@echo "Binary built: bin/stress_service"
	@echo "Usage: bin/stress_service --server_url <URL> --ep_token <TOKEN> [--test_time_min N] [--cadence_sec N]"
