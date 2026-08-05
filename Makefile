# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read a value from deploy/config.yaml (flat key: "value" format).
_cfg = $(shell grep '^$(1):' deploy/config.yaml 2>/dev/null | head -1 | sed 's/^[^:]*:[[:space:]]*//' | sed 's/"//g' | sed "s/'//g" | sed 's/[[:space:]]*\#.*//')

DOCKER_REPO ?= $(call _cfg,docker_registry)
IMAGE_NAME ?= $(DOCKER_REPO)/$(call _cfg,image_name)
IMAGE_TAG = v0.1.20260805.0
RUNTIME_BASE_IMAGE_NAME ?= $(DOCKER_REPO)/$(call _cfg,runtime_base_image)
RUNTIME_BASE_IMAGE_TAG = pytorch-26.01-py3-th0.8.0
RUNTIME_BASE_IMAGE = $(RUNTIME_BASE_IMAGE_NAME):$(RUNTIME_BASE_IMAGE_TAG)
SERVE_CMD_DIST_DIR ?= dist
SERVE_CMD_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve
SERVE_CMD_SELF_CONTAINED_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve-self-contained
SERVE_CMD_LINUX_AMD64_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve-linux-amd64
SERVE_INSTALLER_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve-install
SERVE_INSTALLER_LINUX_AMD64_OUTPUT ?= $(SERVE_CMD_DIST_DIR)/physicsnemo-serve-install-linux-amd64
SERVE_CMD_LINUX_AMD64_TARGET ?= x86_64-unknown-linux-gnu.2.17
SERVE_CMD_LINUX_AMD64_TARGET_DIR ?= x86_64-unknown-linux-gnu
SERVE_CMD_RUST_TOOLCHAIN ?= 1.94.1
SERVE_CMD_PYTHON_VERSION ?= 3.12
SERVE_CMD_RUNTIME_REQUIREMENTS ?= packaging/physicsnemo-serve-cmd/runtime-base.lock
SERVE_CMD_COMPRESSION_LEVEL ?= 5

.PHONY: image runtime-base-image build install-serve-cmd-builders build-serve-cmd build-serve-cmd-self-contained build-serve-cmd-linux-amd64 build-serve-installer build-serve-installer-linux-amd64 clean clean-all experiments observe stress

# Build the main PhysicsNeMo Serve container image on top of the runtime base image.
image: runtime-base-image
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build --build-arg PHYSICSNEMO_SERVE_RUNTIME_BASE_IMAGE=$(RUNTIME_BASE_IMAGE) -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile.physicsnemo-serve.scicomp-rust-slim .

# Build the shared runtime base container image used by the service image.
runtime-base-image:
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build --build-arg PYTORCH_BASE_IMAGE=$(DOCKER_REPO)/pytorch:26.01-py3 -t $(RUNTIME_BASE_IMAGE) -f Dockerfile.physicsnemo-serve.runtime-base .

# Compile the inference server and worker runtime in release mode.
build:
	cargo build --release -p inference_server -p worker-runtime

# Install the Linux system packages and Rust toolchain required to build the CLI binaries.
install-serve-cmd-builders:
	@set -eu; \
	if [ "$$(uname -s)" != "Linux" ]; then \
		echo "install-serve-cmd-builders currently supports Linux only" >&2; \
		exit 1; \
	fi; \
	if ! command -v apt-get >/dev/null 2>&1; then \
		echo "apt-get is required; install build-essential, ca-certificates, cmake, curl, and pkg-config with your distribution package manager" >&2; \
		exit 1; \
	fi; \
	if [ "$$(id -u)" -eq 0 ]; then \
		sudo_cmd=""; \
	elif command -v sudo >/dev/null 2>&1; then \
		sudo_cmd="sudo"; \
	else \
		echo "run this target as root or install sudo" >&2; \
		exit 1; \
	fi; \
	$$sudo_cmd apt-get update; \
	$$sudo_cmd apt-get install --yes build-essential ca-certificates cmake curl pkg-config; \
	if command -v rustup >/dev/null 2>&1; then \
		rustup_bin="$$(command -v rustup)"; \
	else \
		rustup_init="$$(mktemp)"; \
		trap 'rm -f "$$rustup_init"' EXIT HUP INT TERM; \
		curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
			--output "$$rustup_init" https://sh.rustup.rs; \
		sh "$$rustup_init" -y --profile minimal --default-toolchain none; \
		rm -f "$$rustup_init"; \
		trap - EXIT HUP INT TERM; \
		rustup_bin="$$HOME/.cargo/bin/rustup"; \
	fi; \
	"$$rustup_bin" toolchain install $(SERVE_CMD_RUST_TOOLCHAIN) --profile minimal; \
	"$$rustup_bin" default $(SERVE_CMD_RUST_TOOLCHAIN); \
	"$$rustup_bin" run $(SERVE_CMD_RUST_TOOLCHAIN) rustc --version; \
	"$$rustup_bin" run $(SERVE_CMD_RUST_TOOLCHAIN) cargo --version; \
	echo "Builder installation complete. Run: source \"$$HOME/.cargo/env\""

# Build the thin native CLI, which expects a separate Python runtime directory.
build-serve-cmd:
	cargo build --locked --release --package physicsnemo-serve-cmd --bin physicsnemo-serve
	mkdir -p $(SERVE_CMD_DIST_DIR)
	cp target/release/physicsnemo-serve $(SERVE_CMD_OUTPUT)
	chmod +x $(SERVE_CMD_OUTPUT)
	@echo "Thin CLI built: $(SERVE_CMD_OUTPUT)"
	@echo "Run it with: $(SERVE_CMD_OUTPUT) infer --runtime-dir <DIR> ..."

# Build a self-contained CLI with the Python runtime and dependencies embedded in one executable.
build-serve-cmd-self-contained:
	@command -v uv >/dev/null 2>&1 || (echo "uv is required; install it from https://docs.astral.sh/uv/" >&2 && exit 1)
	@test -f "$(SERVE_CMD_RUNTIME_REQUIREMENTS)" || (echo "requirements lock not found: $(SERVE_CMD_RUNTIME_REQUIREMENTS)" >&2 && exit 1)
	uv python install $(SERVE_CMD_PYTHON_VERSION)
	cargo build --locked --release --package physicsnemo-serve-cmd --bin physicsnemo-serve
	@set -eu; \
	python_bin="$$(uv python find $(SERVE_CMD_PYTHON_VERSION))"; \
	python_prefix="$$(dirname "$$(dirname "$$python_bin")")"; \
	runtime_workspace="$$(mktemp -d "$${TMPDIR:-/tmp}/physicsnemo-serve-runtime.XXXXXX")"; \
	staged_output="$(SERVE_CMD_SELF_CONTAINED_OUTPUT).tmp.$$$$"; \
	trap 'rm -rf "$$runtime_workspace"; rm -f "$$staged_output"' EXIT HUP INT TERM; \
	uv run python packaging/physicsnemo-serve-cmd/assemble_runtime.py \
		--python-prefix "$$python_prefix" \
		--requirements "$(SERVE_CMD_RUNTIME_REQUIREMENTS)" \
		--output "$$runtime_workspace/runtime"; \
	mkdir -p "$(dir $(SERVE_CMD_SELF_CONTAINED_OUTPUT))"; \
	target/release/physicsnemo-serve package \
		--runtime-dir "$$runtime_workspace/runtime" \
		--output "$$staged_output" \
		--compression-level "$(SERVE_CMD_COMPRESSION_LEVEL)"; \
	mv -f "$$staged_output" "$(SERVE_CMD_SELF_CONTAINED_OUTPUT)"; \
	chmod +x "$(SERVE_CMD_SELF_CONTAINED_OUTPUT)"; \
	trap - EXIT HUP INT TERM; \
	rm -rf "$$runtime_workspace"
	@echo "Self-contained CLI built: $(SERVE_CMD_SELF_CONTAINED_OUTPUT)"
	@echo "Run it with: $(SERVE_CMD_SELF_CONTAINED_OUTPUT) infer ..."

# Cross-compile the thin CLI for Linux x86_64 using Zig if building on a different platform.
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

# Build the native installer that extracts an embedded runtime from a self-contained CLI.
build-serve-installer:
	cargo build --locked --release --package physicsnemo-serve-cmd --bin physicsnemo-serve-install
	mkdir -p $(SERVE_CMD_DIST_DIR)
	cp target/release/physicsnemo-serve-install $(SERVE_INSTALLER_OUTPUT)
	chmod +x $(SERVE_INSTALLER_OUTPUT)
	@echo "Runtime installer built: $(SERVE_INSTALLER_OUTPUT)"

# Cross-compile the runtime installer for Linux x86_64 using Zig if building on a different platform.
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

# Run the Rust unit tests for the CLI and service workspace packages.
test-rust:
	cargo test -p physicsnemo-serve-cmd --lib
	cargo test -p worker-runtime --lib
	cargo test -p e2s_zarr_io --lib
	cargo test -p inference_server --lib
	cargo test -p scicomp-rq --lib

# Remove all Cargo build artifacts.
clean:
	cargo clean

# Remove Cargo build artifacts plus generated experiment artifacts and outputs.
clean-all:
	cargo clean
	rm -rf artifacts/ outputs/

# Build the end-to-end experiment runner binary.
experiments:
	cd tests/e2e && go build -o ../../bin/run_experiments ./run_experiments.go
	@echo "Binary built: bin/run_experiments"
	@echo "Usage: bin/run_experiments --service_url_rust <RUST_URL> --service_url_python <PYTHON_URL> --ep_token <TOKEN> [--expt 1|2|3|all]"

# Start the local observability stack; browse it at http://localhost:3000.
observe:
	@test -n "$(SERVICE_URL)" || (echo "SERVICE_URL is not set!" && exit 1)
	@test -n "$(EP_TOKEN)" || (echo "EP_TOKEN is not set!" && exit 1)
	SERVICE_URL=$(SERVICE_URL) EP_TOKEN=$(EP_TOKEN) docker compose -f observability/docker-compose.yml up

# Build the end-to-end service stress-test binary.
stress:
	cd tests/e2e && go build -o ../../bin/stress_service ./stress_service.go
	@echo "Binary built: bin/stress_service"
	@echo "Usage: bin/stress_service --server_url <URL> --ep_token <TOKEN> [--test_time_min N] [--cadence_sec N]"
