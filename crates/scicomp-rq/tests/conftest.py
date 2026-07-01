# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration and fixtures for scicomp_rq tests.
"""

import os

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Redis)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip integration tests by default unless explicitly requested.

    Run integration tests with: pytest -m integration
    Run all tests with: pytest --run-integration
    """
    if config.getoption("--run-integration", default=False):
        # Don't skip integration tests
        return

    skip_integration = pytest.mark.skip(
        reason="Integration tests require Redis. Use --run-integration to run."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require Redis",
    )


@pytest.fixture
def redis_url():
    """
    Get Redis URL from environment or use default.

    Set REDIS_URL environment variable to override.
    """
    return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
