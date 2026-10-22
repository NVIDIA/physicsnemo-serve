# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI autofix agent: triage CI failures on a PR and repair the ones it caused."""

AGENT_COMMIT_PREFIX = "ci-fix(agent):"
MAX_AGENT_ATTEMPTS = 2
RUN_TRAILER = "Autofix-Run:"
AUTOFIX_LABEL = "ci-autofix"
TEST_CHANGED_LABEL = "ci-autofix:test-changed"
COMMENT_MARKER = "<!-- ci-autofix -->"
