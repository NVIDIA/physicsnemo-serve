/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

pub mod memory;
pub mod redis;

pub(crate) fn consumer_group_name(stream: &str) -> String {
    format!("{stream}:grp")
}

#[cfg(test)]
mod tests {
    use super::consumer_group_name;

    #[test]
    fn consumer_group_name_appends_grp_suffix() {
        assert_eq!(consumer_group_name("schedule"), "schedule:grp");
    }
}
