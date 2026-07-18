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

.PHONY: image runtime-base-image build clean clean-all experiments observe stress

image: runtime-base-image
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build --build-arg PHYSICSNEMO_SERVE_RUNTIME_BASE_IMAGE=$(RUNTIME_BASE_IMAGE) -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile.Earth2Studio.scicomp-rust-slim .

runtime-base-image:
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build --build-arg PYTORCH_BASE_IMAGE=$(DOCKER_REPO)/pytorch:26.01-py3 -t $(RUNTIME_BASE_IMAGE) -f Dockerfile.Earth2Studio.runtime-base .

build:
	cargo build --release -p inference_server -p worker-runtime

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

