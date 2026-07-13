/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

pub mod batch;
pub mod collect;
pub mod fanout;
pub mod parent_run_state;
pub mod postprocess;
pub mod prefetch;
pub mod prepare;
pub mod publish;
pub mod results;
pub mod scheduler;
pub(crate) mod stage;

use std::sync::Arc;

use anyhow::{Result, anyhow};
use scicomp_rq::QueueManager;

use crate::metrics::WorkerMetrics;
use crate::retry_dlq::RetryDlqPolicy;
use crate::roles::prefetch::PlanMaterializer;
use crate::traits::{BackgroundTask, RoleEnv, WorkerRole};

/// Return type for [`build_role`]: the role and its background tasks.
pub type RoleBuildResult = (Box<dyn WorkerRole>, Vec<Box<dyn BackgroundTask>>);

/// Optional providers injected into roles that need external data sources.
pub struct RoleProviders {
    pub materializer: Option<Arc<dyn PlanMaterializer>>,
    pub results_persistence: Option<Arc<dyn results::ResultsPersistence>>,
    pub queue_manager: Option<QueueManager>,
    pub retry_dlq_policy: Option<RetryDlqPolicy>,
    pub metrics: Option<WorkerMetrics>,
}

impl RoleProviders {
    pub fn empty() -> Self {
        Self {
            materializer: None,
            results_persistence: None,
            queue_manager: None,
            retry_dlq_policy: None,
            metrics: None,
        }
    }

    pub fn with_materializer(mut self, mat: Arc<dyn PlanMaterializer>) -> Self {
        self.materializer = Some(mat);
        self
    }

    pub fn with_results_persistence(mut self, rp: Arc<dyn results::ResultsPersistence>) -> Self {
        self.results_persistence = Some(rp);
        self
    }

    pub fn with_queue_manager(mut self, qm: QueueManager) -> Self {
        self.queue_manager = Some(qm);
        self
    }

    pub fn with_retry_dlq_policy(mut self, policy: RetryDlqPolicy) -> Self {
        self.retry_dlq_policy = Some(policy);
        self
    }

    pub fn with_metrics(mut self, metrics: WorkerMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

/// Build a role and its background tasks from the resolved environment.
///
/// `providers.results_persistence` is required for the results role.
pub fn build_role(env: &RoleEnv, providers: RoleProviders) -> Result<RoleBuildResult> {
    match env.role_name.as_str() {
        "collect" => {
            let (role, tasks) = match providers.queue_manager {
                Some(qm) => collect::CollectRole::from_env_with_queue_manager(env, qm)?,
                None => collect::CollectRole::from_env(env)?,
            };
            Ok((Box::new(role), tasks))
        }
        "batch" => {
            let (role, tasks) = match providers.queue_manager {
                Some(qm) => batch::BatchRole::from_env_with_queue_manager(env, qm)?,
                None => batch::BatchRole::from_env(env)?,
            };
            Ok((Box::new(role), tasks))
        }
        "fanout" => {
            let (role, tasks) = match providers.queue_manager {
                Some(qm) => fanout::FanoutRole::from_env_with_queue_manager(env, qm)?,
                None => fanout::FanoutRole::from_env(env)?,
            };
            Ok((Box::new(role), tasks))
        }
        "prepare" => {
            let role = prepare::PrepareRole::from_env(env)?;
            Ok((Box::new(role), vec![]))
        }
        "postprocess" => {
            let role = postprocess::PostprocessRole::from_env(env)?;
            Ok((Box::new(role), vec![]))
        }
        "publish" => {
            let role = match providers.queue_manager {
                Some(qm) => publish::PublishRole::from_env_with_queue_manager(env, qm)?,
                None => publish::PublishRole::from_env(env)?,
            };
            Ok((Box::new(role), vec![]))
        }
        "prefetch" => {
            let role = prefetch::PrefetchRole::from_env(env, providers.materializer)?;
            Ok((Box::new(role), vec![]))
        }
        "scheduler" => {
            let retry_dlq_policy = providers
                .retry_dlq_policy
                .unwrap_or_else(|| RetryDlqPolicy::new(5, "dlq"));
            let qm = providers
                .queue_manager
                .ok_or_else(|| anyhow!("scheduler role requires a QueueManager provider"))?;
            let (role, tasks) =
                scheduler::SchedulerRole::from_env(env, qm, retry_dlq_policy, providers.metrics)?;
            Ok((Box::new(role), tasks))
        }
        "results" => {
            let persistence = providers
                .results_persistence
                .ok_or_else(|| anyhow!("results role requires a ResultsPersistence provider"))?;
            let role = results::ResultsRole::from_env(env, persistence)?;
            Ok((Box::new(role), vec![]))
        }
        unknown => Err(anyhow!(
            "unknown role '{}': expected batch, collect, fanout, prepare, postprocess, prefetch, publish, scheduler, or results",
            unknown
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod test_support {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/mod.rs"));
    }

    use crate::config::InputStreamSpec;
    use crate::roles::results::NoopResultsPersistence;
    use crate::test_env;
    use crate::traits::RoleEnv;
    use test_support::spawn_test_queue_manager;

    fn env_for(role: &str, outputs: &[&str]) -> RoleEnv {
        let inputs = if role == "scheduler" {
            vec![
                InputStreamSpec {
                    stream: "schedule".to_string(),
                    max_dequeue_items: 1,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
                InputStreamSpec {
                    stream: "release".to_string(),
                    max_dequeue_items: 1,
                    poll_interval_ms: 10,
                    block_ms: 50,
                    reclaim_idle_ms: 60_000,
                },
            ]
        } else {
            vec![InputStreamSpec {
                stream: "input".to_string(),
                max_dequeue_items: 1,
                poll_interval_ms: 10,
                block_ms: 50,
                reclaim_idle_ms: 60_000,
            }]
        };
        RoleEnv {
            role_name: role.to_string(),
            stream_prefix: "test:".to_string(),
            inputs,
            resolved_outputs: outputs.iter().map(|s| s.to_string()).collect(),
            role_config: None,
            python_runtime_envs: Default::default(),
        }
    }

    #[test]
    fn build_role_creates_prefetch() {
        let env = env_for("prefetch", &["schedule"]);
        let (role, tasks) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "prefetch");
        assert!(tasks.is_empty());
    }

    #[test]
    fn build_role_creates_prepare() {
        let env = env_for("prepare", &["schedule"]);
        let (role, tasks) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "prepare");
        assert!(tasks.is_empty());
    }

    #[test]
    fn build_role_creates_batch() {
        let env = env_for("batch", &["schedule"]);
        let (role, tasks) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "batch");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].name(), "batch_flush");
    }

    #[test]
    fn build_role_creates_collect() {
        let env = env_for("collect", &["results"]);
        let (role, tasks) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "collect");
        assert!(tasks.is_empty());
    }

    #[test]
    fn build_role_creates_fanout() {
        let env = env_for("fanout", &["schedule"]);
        let (role, tasks) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "fanout");
        assert!(tasks.is_empty());
    }

    #[test]
    fn build_role_creates_postprocess() {
        let env = env_for("postprocess", &["results"]);
        let (role, tasks) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "postprocess");
        assert!(tasks.is_empty());
    }

    #[test]
    fn build_role_creates_publish() {
        let env = env_for("publish", &["results"]);
        let (role, tasks) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "publish");
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn build_role_creates_scheduler() {
        let _guard = test_env::env_lock().lock().await;
        let prev_discovery = std::env::var("SCHEDULER_DISCOVERY_JSON").ok();
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        test_env::set_env_var("SCHEDULER_DISCOVERY_JSON", None);
        test_env::set_env_var("SCHEDULER_PROFILES_JSON", None);

        let env = env_for("scheduler", &[]);
        let (_redis_server, qm) = spawn_test_queue_manager("build-role-scheduler").await;
        let (role, tasks) =
            build_role(&env, RoleProviders::empty().with_queue_manager(qm)).unwrap();
        assert_eq!(role.name(), "scheduler");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].name(), "resource_discovery");
        assert_eq!(tasks[1].name(), "scheduler_task");

        test_env::set_env_var("SCHEDULER_DISCOVERY_JSON", prev_discovery.as_deref());
        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[test]
    fn build_role_creates_results() {
        let env = env_for("results", &[]);
        let providers = RoleProviders {
            results_persistence: Some(Arc::new(NoopResultsPersistence::new())),
            ..RoleProviders::empty()
        };
        let (role, tasks) = build_role(&env, providers).unwrap();
        assert_eq!(role.name(), "results");
        assert!(tasks.is_empty());
    }

    #[test]
    fn build_role_results_fails_without_results_persistence() {
        let env = env_for("results", &[]);
        let err = build_role(&env, RoleProviders::empty())
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.to_string().contains("ResultsPersistence"),
            "expected ResultsPersistence error, got: {err}"
        );
    }

    #[test]
    fn build_role_rejects_gpu_prefixed_roles() {
        let env = env_for("gpu_0", &["results"]);
        let result = build_role(&env, RoleProviders::empty());
        assert!(
            result.is_err(),
            "gpu_* roles should not be supported by worker-runtime"
        );
    }

    #[test]
    fn build_role_rejects_unknown() {
        let env = env_for("admin", &[]);
        let result = build_role(&env, RoleProviders::empty());
        assert!(result.is_err());
    }

    // --- PR-049: builder pattern for RoleProviders ---

    #[test]
    fn role_providers_builder_creates_prefetch() {
        let env = env_for("prefetch", &["schedule"]);
        let (role, _) = build_role(&env, RoleProviders::empty()).unwrap();
        assert_eq!(role.name(), "prefetch");
    }

    #[test]
    fn role_providers_builder_creates_results() {
        let env = env_for("results", &[]);
        let providers = RoleProviders::empty()
            .with_results_persistence(Arc::new(NoopResultsPersistence::new()));
        let (role, _) = build_role(&env, providers).unwrap();
        assert_eq!(role.name(), "results");
    }
}
