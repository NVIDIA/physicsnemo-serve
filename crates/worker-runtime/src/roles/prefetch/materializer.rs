/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::Path;

use anyhow::Result;

use super::download::{HttpDownloader, MaterializationResult};
use super::plan::PrefetchPlanItem;
use super::prefetch_config::PrefetchConfig;
use crate::traits::BoxFuture;

pub trait PlanMaterializer: Send + Sync + 'static {
    fn materialize<'a>(
        &'a self,
        plan: &'a [PrefetchPlanItem],
        cache_root: &'a Path,
        run_id: &'a str,
    ) -> BoxFuture<'a, Result<MaterializationResult>>;
}

pub struct HttpPlanMaterializer {
    downloader: HttpDownloader,
}

impl HttpPlanMaterializer {
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            downloader: HttpDownloader::with_config(config),
        }
    }
}

impl PlanMaterializer for HttpPlanMaterializer {
    fn materialize<'a>(
        &'a self,
        plan: &'a [PrefetchPlanItem],
        cache_root: &'a Path,
        run_id: &'a str,
    ) -> BoxFuture<'a, Result<MaterializationResult>> {
        Box::pin(async move {
            self.downloader
                .materialize_plan(plan, cache_root, run_id)
                .await
        })
    }
}
