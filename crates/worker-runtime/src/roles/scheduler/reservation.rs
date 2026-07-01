/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use scicomp_rq::QueueManager;
use serde::Deserialize;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::discovery::{
    DiscoveryUpdate, ResourceInfo, warmup_status_allows_scheduling, warmup_status_is_loading,
};
use super::profile::ResourceManager;
#[cfg(test)]
use super::reserved_memory::InMemoryReservedMemoryStore;
use super::reserved_memory::{RedisReservedMemoryStore, ReservedMemoryStore};
use super::{SchedulePayload, ScheduleResourceProfile};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservedResource {
    pub stream_name: String,
    pub resource_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReservationBlockedError;

impl fmt::Display for ReservationBlockedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("insufficient resources available with sufficient memory")
    }
}

impl std::error::Error for ReservationBlockedError {}

pub(crate) fn is_reservation_blocked_error(error: &anyhow::Error) -> bool {
    error.downcast_ref::<ReservationBlockedError>().is_some()
}

/// GPU scheduling strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SchedulingStrategy {
    BestFit,
    #[default]
    RoundRobin,
}

impl<'de> Deserialize<'de> for SchedulingStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        SchedulingStrategy::parse(&raw).map_err(serde::de::Error::custom)
    }
}

impl SchedulingStrategy {
    /// Parse from a config string, rejecting unknown values.
    ///
    /// Valid values (case-insensitive): `best_fit`, `round_robin`, `roundrobin`.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "best_fit" | "bestfit" => Ok(Self::BestFit),
            "round_robin" | "roundrobin" => Ok(Self::RoundRobin),
            _ => Err(anyhow!(
                "unknown scheduling strategy '{}'; valid options: best_fit, round_robin",
                s
            )),
        }
    }
}

/// In-memory view of discovered resource inventory.
#[derive(Clone)]
pub struct ResourceReservationTable {
    /// Current discovered worker inventory keyed by resource id.
    resource_map: Arc<Mutex<HashMap<u32, ResourceInfo>>>,
    /// Durable per-resource reserved memory accounting store.
    reserved_memory: Arc<dyn ReservedMemoryStore>,
    /// Percent of each GPU's total memory that the scheduler may allocate.
    memory_utilization_percent: u64,
    /// Placement strategy used when choosing among matching workers.
    strategy: SchedulingStrategy,
    /// Round-robin cursor used to rotate the ordered candidate list between requests.
    rr_cursor: Arc<std::sync::Mutex<usize>>,
    /// Serializes the full reserve/release accounting flow against the durable
    /// counters so concurrent requests cannot admit against the same stale snapshot.
    reservation_flow_lock: Arc<Mutex<()>>,
}

impl ResourceReservationTable {
    pub fn new(
        qm: QueueManager,
        memory_utilization_percent: u64,
        strategy: SchedulingStrategy,
    ) -> Self {
        Self::with_store(
            memory_utilization_percent,
            strategy,
            Arc::new(RedisReservedMemoryStore::new(qm)),
        )
    }

    fn with_store(
        memory_utilization_percent: u64,
        strategy: SchedulingStrategy,
        reserved_memory: Arc<dyn ReservedMemoryStore>,
    ) -> Self {
        Self {
            resource_map: Arc::new(Mutex::new(HashMap::new())),
            reserved_memory,
            memory_utilization_percent,
            strategy,
            rr_cursor: Arc::new(std::sync::Mutex::new(0)),
            reservation_flow_lock: Arc::new(Mutex::new(())),
        }
    }

    #[cfg(test)]
    fn new_for_tests(memory_utilization_percent: u64, strategy: SchedulingStrategy) -> Self {
        Self::with_store(
            memory_utilization_percent,
            strategy,
            Arc::new(InMemoryReservedMemoryStore::default()),
        )
    }

    #[cfg(test)]
    pub async fn used_memory_mb(&self, resource_id: u32) -> Option<u64> {
        let resource = {
            let map = self.resource_map.lock().await;
            map.get(&resource_id).cloned()
        }?;
        let reserved_memory = self
            .reserved_memory
            .get_many(&[resource_id])
            .await
            .ok()
            .and_then(|values| values.get(&resource_id).copied())
            .unwrap_or(0);
        Some(effective_used_memory_mb(&resource, reserved_memory))
    }

    /// Refresh reservation table from a discovery update.
    pub async fn sync_from_discovery_update(&self, update: DiscoveryUpdate) -> Result<()> {
        match update {
            DiscoveryUpdate::Authoritative(discovered) => {
                debug!(
                    gpu_worker_count = discovered.len(),
                    "applying discovery update to reservation table"
                );

                let mut discovered_by_resource_id: HashMap<u32, ResourceInfo> =
                    HashMap::with_capacity(discovered.len());
                for discovered_gpu in discovered {
                    discovered_by_resource_id.insert(discovered_gpu.resource_id, discovered_gpu);
                }

                let mut map = self.resource_map.lock().await;
                let discovered_ids: HashSet<u32> =
                    discovered_by_resource_id.keys().copied().collect();
                let existing_ids: Vec<u32> = map.keys().copied().collect();

                // Log removals by diffing the previous map against the new authoritative
                // snapshot before we replace the map wholesale below.
                for resource_id in existing_ids {
                    if !discovered_ids.contains(&resource_id)
                        && let Some(removed) = map.get(&resource_id)
                    {
                        warn!(
                            resource_id,
                            stream = %removed.stream_name,
                            "removed discovered GPU worker entry from reservation table"
                        );
                    }
                }

                // Log additions by diffing the new authoritative snapshot against the
                // previous map before we replace the map wholesale below.
                for (resource_id, discovered_gpu) in &discovered_by_resource_id {
                    if !map.contains_key(resource_id) {
                        info!(
                            resource_id,
                            stream = %discovered_gpu.stream_name,
                            available_mb = discovered_gpu
                                .available_memory_mb(self.memory_utilization_percent),
                            used_memory_mb = discovered_gpu.used_memory_mb,
                            "added discovered worker to reservation table"
                        );
                    }
                }

                // DiscoveryUpdate::Authoritative replaces the entire tracked inventory.
                *map = discovered_by_resource_id;

                debug!(
                    tracked_worker_count = map.len(),
                    "reservation table synchronized from discovery update"
                );
            }
            DiscoveryUpdate::Stale => {
                debug!("skipping reservation table update because discovery was stale");
            }
        }

        Ok(())
    }

    pub(super) async fn tracked_worker_count(&self) -> usize {
        self.resource_map.lock().await.len()
    }

    /// Reserve GPUs for a schedule payload using the configured strategy.
    ///
    /// Returns selected GPU ids when enough resources are available.
    pub async fn reserve(&self, payload: &mut SchedulePayload) -> Result<Vec<ReservedResource>> {
        // Serialize the read-select-increment flow so two concurrent reservations
        // do not choose from the same pre-increment reserved-memory snapshot.
        let _reservation_flow_guard = self.reservation_flow_lock.lock().await;

        let (gpus_required, memory_mb, config_source) = resolve_resource_config(payload)?;
        let resources = {
            let map = self.resource_map.lock().await;
            map.values().cloned().collect::<Vec<_>>()
        };
        // Known-profile fallback is only safe when all matching workers share one capability group.
        if matches!(config_source, ResourceConfigSource::KnownProfileFallback) {
            ensure_uniform_worker_capabilities(payload.workflow.as_str(), resources.iter())?;
        }

        payload.gpus_required = gpus_required;
        payload.memory_mb = memory_mb;

        let needed_gpus = payload.gpus_required;
        let needed_memory_mb = payload.memory_mb;
        let limit = self.memory_utilization_percent;
        debug!(
            run_id = %payload.run_id,
            workflow = %payload.workflow,
            config_source = ?config_source,
            strategy = ?self.strategy,
            requested_gpus = payload.gpus_required,
            needed_workers = needed_gpus,
            requested_memory_mb = needed_memory_mb,
            tracked_worker_count = resources.len(),
            "attempting worker reservation"
        );

        let candidates = select_candidate_resources(&resources, payload, limit, needed_gpus)?;

        let candidates: Vec<ResourceInfo> = match self.strategy {
            SchedulingStrategy::BestFit => {
                let resource_ids: Vec<u32> = candidates
                    .iter()
                    .map(|resource| resource.resource_id)
                    .collect();
                let reserved_memory_by_resource =
                    self.reserved_memory.get_many(&resource_ids).await?;

                let matching_candidate_count = candidates.len();
                // Admission uses a conservative view of pressure per resource: take the
                // higher of live discovery usage and the durable scheduler-accounted
                // usage, then subtract that from the usable-memory budget.
                let mut candidates: Vec<(ResourceInfo, u64)> = candidates
                    .into_iter()
                    .map(|resource| {
                        let reserved_memory_mb = reserved_memory_by_resource
                            .get(&resource.resource_id)
                            .copied()
                            .unwrap_or(0);
                        let available_memory_mb =
                            available_memory_mb(&resource, reserved_memory_mb, limit);
                        (resource, available_memory_mb)
                    })
                    .collect();
                candidates
                    .retain(|(_, available_memory_mb)| *available_memory_mb >= needed_memory_mb);
                debug!(
                    run_id = %payload.run_id,
                    workflow = %payload.workflow,
                    matching_candidate_count,
                    memory_candidate_count = candidates.len(),
                    "reservation candidate evaluation complete"
                );
                if candidates.len() < needed_gpus {
                    info!(
                        run_id = %payload.run_id,
                        workflow = %payload.workflow,
                        matching_candidate_count,
                        memory_candidate_count = candidates.len(),
                        needed_workers = needed_gpus,
                        requested_memory_mb = needed_memory_mb,
                        "reservation blocked because matching workers did not have enough available memory"
                    );
                    return Err(ReservationBlockedError.into());
                }

                candidates.sort_by_key(|(resource, available_memory_mb)| {
                    (*available_memory_mb, resource.resource_id)
                });
                candidates
                    .into_iter()
                    .map(|(resource, _available_memory_mb)| resource)
                    .collect()
            }
            SchedulingStrategy::RoundRobin => {
                let mut candidates = candidates;
                debug!(
                    run_id = %payload.run_id,
                    workflow = %payload.workflow,
                    matching_candidate_count = candidates.len(),
                    "round-robin candidate evaluation complete without available-memory filtering"
                );
                candidates.sort_by_key(|candidate| candidate.resource_id);
                let mut cursor = self
                    .rr_cursor
                    .lock()
                    .map_err(|_| anyhow!("rr_cursor poisoned"))?;
                let offset = *cursor % candidates.len();
                candidates.rotate_left(offset);
                *cursor = cursor.wrapping_add(needed_gpus);
                candidates
            }
        };

        let mut selected_gpu_ids = Vec::with_capacity(needed_gpus);
        let mut reserved_resource_ids: Vec<u32> = Vec::with_capacity(needed_gpus);
        for candidate in &candidates[..needed_gpus] {
            let resource_id = candidate.resource_id;
            // Commit the reservation against a floor of the currently observed
            // used memory so newly admitted work is added on top of existing
            // live pressure instead of being hidden by it.
            let updated_accounted_memory_mb = match self
                .reserved_memory
                .reserve(resource_id, candidate.used_memory_mb, needed_memory_mb)
                .await
            {
                Ok(updated) => updated,
                Err(error) => {
                    for reserved_resource_id in reserved_resource_ids {
                        let _ = self
                            .reserved_memory
                            .decrement(reserved_resource_id, needed_memory_mb)
                            .await;
                    }
                    return Err(error);
                }
            };
            reserved_resource_ids.push(resource_id);
            selected_gpu_ids.push(ReservedResource {
                stream_name: candidate.stream_name.clone(),
                resource_id,
            });
            info!(
                run_id = %payload.run_id,
                workflow = %payload.workflow,
                stream = %candidate.stream_name,
                resource_id = candidate.resource_id,
                reserved_mb = needed_memory_mb,
                used_mb = updated_accounted_memory_mb,
                available_mb = available_memory_mb(
                    candidate,
                    updated_accounted_memory_mb,
                    limit,
                ),
                strategy = ?self.strategy,
                "reserved worker capacity"
            );
        }
        debug!(
            run_id = %payload.run_id,
            workflow = %payload.workflow,
            selected_target_count = selected_gpu_ids.len(),
            selected_targets = ?selected_gpu_ids,
            "reservation completed"
        );

        Ok(selected_gpu_ids)
    }

    /// Release reserved memory from a resource id.
    pub(super) async fn release(&self, resource_id: u32, memory_mb: u64) -> Result<()> {
        // Serialize the decrement with the same flow lock used by reserve() so
        // releases and new admissions observe a consistent reserved-memory view.
        let _reservation_flow_guard = self.reservation_flow_lock.lock().await;
        let resource = {
            let map = self.resource_map.lock().await;
            map.get(&resource_id).cloned()
        };
        let Some(reserved_after) = self
            .reserved_memory
            .decrement(resource_id, memory_mb)
            .await?
        else {
            warn!(
                resource_id,
                stream = ?resource.as_ref().map(|resource| resource.stream_name.as_str()),
                requested_memory_mb = memory_mb,
                "release requested for resource with no reserved accounting"
            );
            return Err(anyhow!(
                "scheduler: no reserved allocation found for resource '{resource_id}'"
            ));
        };
        let effective_used_after_mb = resource
            .as_ref()
            .map(|resource| effective_used_memory_mb(resource, reserved_after))
            .unwrap_or(reserved_after);
        info!(
            resource_id,
            stream = ?resource.as_ref().map(|resource| resource.stream_name.as_str()),
            requested_memory_mb = memory_mb,
            reserved_after_mb = reserved_after,
            effective_used_after_mb,
            "released GPU reservation"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceConfigSource {
    ExplicitProfile,
    KnownProfileFallback,
}

fn resolve_resource_config(
    payload: &mut SchedulePayload,
) -> Result<(usize, u64, ResourceConfigSource)> {
    if let Some(profile) = payload.resource_profile.as_ref() {
        let gpus_required = profile
            .gpus_required
            .ok_or_else(|| anyhow!("scheduler: resource_profile.gpus_required is required"))?;
        let memory_mb = profile
            .memory_mb
            .ok_or_else(|| anyhow!("scheduler: resource_profile.memory_mb is required"))?;
        if gpus_required == 0 {
            return Err(anyhow!(
                "scheduler: resource_profile.gpus_required must be at least 1"
            ));
        }
        if memory_mb == 0 {
            return Err(anyhow!(
                "scheduler: resource_profile.memory_mb must be greater than 0"
            ));
        }
        info!(
            workflow = %payload.workflow,
            run_id = %payload.run_id,
            source = "ExplicitProfile",
            gpus_required,
            memory_mb,
            "resolve_resource_config: using explicit resource_profile from payload"
        );
        return Ok((
            gpus_required,
            memory_mb,
            ResourceConfigSource::ExplicitProfile,
        ));
    }

    if let Some(profile) = ResourceManager::lookup_known_profile_resources(payload) {
        let gpus_required = profile.gpus_used;
        let memory_mb = profile.peak_memory_mib().ok_or_else(|| {
            anyhow!(
                "scheduler: matched profile for workflow '{}' has invalid peak.memory.used",
                payload.workflow
            )
        })?;
        if gpus_required == 0 {
            return Err(anyhow!(
                "scheduler: matched profile for workflow '{}' must request at least one GPU because known-profile fallback is GPU-only",
                payload.workflow
            ));
        }
        if memory_mb == 0 {
            return Err(anyhow!(
                "scheduler: matched profile for workflow '{}' has invalid memory_mb=0",
                payload.workflow
            ));
        }
        info!(
            workflow = %payload.workflow,
            run_id = %payload.run_id,
            source = "KnownProfileFallback",
            gpus_required,
            memory_mb,
            "resolve_resource_config: using known-profile fallback"
        );
        payload.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(gpus_required),
            memory_mb: Some(memory_mb),
            executor_class: None,
            tags: None,
        });
        return Ok((
            gpus_required,
            memory_mb,
            ResourceConfigSource::KnownProfileFallback,
        ));
    }

    Err(anyhow!(
        "scheduler: resource_profile is required unless workflow '{}' matches a known scheduler profile",
        payload.workflow
    ))
}

fn ensure_uniform_worker_capabilities<'a>(
    workflow: &str,
    workers: impl Iterator<Item = &'a ResourceInfo>,
) -> Result<()> {
    let mut capability_groups: HashSet<(Option<String>, Vec<String>)> = HashSet::new();
    for worker in workers {
        let mut normalized_tags = worker.tags.clone();
        normalized_tags.sort_unstable();
        capability_groups.insert((worker.executor_class.clone(), normalized_tags));
        if capability_groups.len() > 1 {
            return Err(anyhow!(
                "scheduler: resource_profile is required for workflow '{}' because known-profile fallback matches multiple GPU worker capability groups",
                workflow,
            ));
        }
    }

    Ok(())
}

// Scheduler uses whichever source reports higher usage: live discovery or the
// scheduler's durable reserved-memory counter. This avoids undercounting across
// restarts and between discovery refreshes.
fn effective_used_memory_mb(resource: &ResourceInfo, reserved_memory_mb: u64) -> u64 {
    resource.used_memory_mb.max(reserved_memory_mb)
}

fn available_memory_mb(
    resource: &ResourceInfo,
    reserved_memory_mb: u64,
    utilization_limit: u64,
) -> u64 {
    resource
        .usable_memory_mb(utilization_limit)
        .saturating_sub(effective_used_memory_mb(resource, reserved_memory_mb))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReservationBlockReason {
    Loading,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceEligibility {
    Available,
    TemporarilyUnavailable(ReservationBlockReason),
    Ineligible,
}

fn select_candidate_resources(
    resources: &[ResourceInfo],
    payload: &SchedulePayload,
    utilization_limit: u64,
    needed_gpus: usize,
) -> Result<Vec<ResourceInfo>> {
    let mut candidates = Vec::new();
    let mut loading_candidate_count = 0usize;

    for resource in resources {
        match resource_eligibility_for_request(resource, payload, utilization_limit) {
            ResourceEligibility::Available => candidates.push(resource.clone()),
            ResourceEligibility::TemporarilyUnavailable(ReservationBlockReason::Loading) => {
                loading_candidate_count += 1;
            }
            ResourceEligibility::Ineligible => {}
        }
    }

    if candidates.len() >= needed_gpus {
        return Ok(candidates);
    }

    if candidates.len() + loading_candidate_count >= needed_gpus {
        info!(
            run_id = %payload.run_id,
            workflow = %payload.workflow,
            matching_candidate_count = candidates.len(),
            loading_candidate_count,
            needed_workers = needed_gpus,
            tracked_worker_count = resources.len(),
            "reservation blocked because matching workers are temporarily unavailable"
        );
        return Err(ReservationBlockedError.into());
    }

    info!(
        run_id = %payload.run_id,
        workflow = %payload.workflow,
        matching_candidate_count = candidates.len(),
        loading_candidate_count,
        needed_workers = needed_gpus,
        "reservation failed because too few workers can satisfy the request"
    );
    Err(anyhow!(
        "not enough workers can satisfy the requested requirements"
    ))
}

fn resource_matches_static_requirements(
    resource: &ResourceInfo,
    payload: &SchedulePayload,
    utilization_limit: u64,
) -> bool {
    if resource.usable_memory_mb(utilization_limit) < payload.memory_mb {
        return false;
    }

    let Some(profile) = payload.resource_profile.as_ref() else {
        return true;
    };

    if let Some(executor_class) = profile.executor_class.as_deref()
        && resource.executor_class.as_deref() != Some(executor_class)
    {
        return false;
    }

    if let Some(required_tags) = profile.tags.as_ref()
        && !required_tags.is_empty()
        && !required_tags
            .iter()
            .all(|tag| resource.tags.iter().any(|candidate| candidate == tag))
    {
        return false;
    }

    true
}

fn resource_eligibility_for_request(
    resource: &ResourceInfo,
    payload: &SchedulePayload,
    utilization_limit: u64,
) -> ResourceEligibility {
    if !resource_matches_static_requirements(resource, payload, utilization_limit) {
        return ResourceEligibility::Ineligible;
    }

    if resource_has_conflicting_workflow(resource, payload) {
        return ResourceEligibility::Ineligible;
    }

    if warmup_status_is_loading(resource.warmup_status.as_deref()) {
        return ResourceEligibility::TemporarilyUnavailable(ReservationBlockReason::Loading);
    }

    if !warmup_status_allows_scheduling(resource.warmup_status.as_deref()) {
        return ResourceEligibility::Ineligible;
    }

    ResourceEligibility::Available
}

fn resource_has_conflicting_workflow(resource: &ResourceInfo, payload: &SchedulePayload) -> bool {
    let Some(requested_workflow_id) = payload
        .workflow_id
        .as_deref()
        .map(str::trim)
        .filter(|workflow_id| !workflow_id.is_empty())
    else {
        return false;
    };

    let resource_workflow_ids = resource
        .model_cache_workflow_ids
        .iter()
        .map(String::as_str)
        .chain(resource.warmup_workflow_id.as_deref());

    let mut has_resource_workflow = false;
    for resource_workflow_id in resource_workflow_ids {
        has_resource_workflow = true;
        if resource_workflow_id.eq_ignore_ascii_case(requested_workflow_id) {
            return false;
        }
    }

    has_resource_workflow
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::scheduler::DEFAULT_MEMORY_UTILIZATION_PERCENT;
    use crate::roles::scheduler::ScheduleResourceProfile;
    use crate::roles::scheduler::discovery::WARMUP_STATUS_SUCCEEDED;
    use crate::test_env;
    use serde_json::json;

    fn table_with_gpus(gpus: Vec<ResourceInfo>) -> ResourceReservationTable {
        table_with_gpus_strategy(gpus, SchedulingStrategy::BestFit)
    }

    fn table_with_gpus_strategy(
        gpus: Vec<ResourceInfo>,
        strategy: SchedulingStrategy,
    ) -> ResourceReservationTable {
        let mut table =
            ResourceReservationTable::new_for_tests(DEFAULT_MEMORY_UTILIZATION_PERCENT, strategy);
        let resource_map = gpus
            .iter()
            .cloned()
            .map(|gpu| (gpu.resource_id, gpu))
            .collect();
        table.resource_map = Arc::new(Mutex::new(resource_map));
        table
    }

    fn table_with_gpus_and_limit(
        gpus: Vec<ResourceInfo>,
        percent: u64,
    ) -> ResourceReservationTable {
        let mut table =
            ResourceReservationTable::new_for_tests(percent, SchedulingStrategy::BestFit);
        let resource_map = gpus
            .iter()
            .cloned()
            .map(|gpu| (gpu.resource_id, gpu))
            .collect();
        table.resource_map = Arc::new(Mutex::new(resource_map));
        table
    }

    fn payload(run_id: &str, workflow: &str) -> SchedulePayload {
        SchedulePayload {
            run_id: run_id.to_string(),
            workflow: workflow.to_string(),
            workflow_id: None,
            parent_run_id: None,
            fanout_profile: None,
            raw_payload: json!({ "workflow": workflow }),
            resource_profile: Some(ScheduleResourceProfile {
                gpus_required: Some(1),
                memory_mb: Some(20_000),
                executor_class: Some("python.gpu.demo".to_string()),
                tags: Some(vec!["demo".to_string()]),
            }),
            gpus_required: 0,
            memory_mb: 0,
            dispatch_stage: "execute".to_string(),
        }
    }

    fn payload_with_resource_profile(
        run_id: &str,
        workflow: &str,
        gpus_required: usize,
        memory_mb: u64,
    ) -> SchedulePayload {
        let mut payload = payload(run_id, workflow);
        payload.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(gpus_required),
            memory_mb: Some(memory_mb),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string()]),
        });
        payload
    }

    fn gpu(
        gpu_id: u32,
        stream_name: &str,
        total_memory_mb: u64,
        used_memory_mb: u64,
    ) -> ResourceInfo {
        ResourceInfo {
            resource_id: gpu_id,
            stream_name: stream_name.to_string(),
            total_memory_mb,
            used_memory_mb,
            executor_class: Some("python.gpu.demo".to_string()),
            tags: vec!["demo".to_string()],
            model_cache_workflow_ids: Vec::new(),
            warmup_workflow_id: None,
            warmup_status: None,
        }
    }

    fn worker(
        gpu_id: u32,
        stream_name: &str,
        total_memory_mb: u64,
        used_memory_mb: u64,
        executor_class: &str,
        tags: &[&str],
    ) -> ResourceInfo {
        ResourceInfo {
            resource_id: gpu_id,
            stream_name: stream_name.to_string(),
            total_memory_mb,
            used_memory_mb,
            executor_class: Some(executor_class.to_string()),
            tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
            model_cache_workflow_ids: Vec::new(),
            warmup_workflow_id: None,
            warmup_status: None,
        }
    }

    fn worker_cached_for(mut resource: ResourceInfo, workflow_id: &str) -> ResourceInfo {
        resource.model_cache_workflow_ids = vec![workflow_id.to_string()];
        resource.warmup_status = Some(WARMUP_STATUS_SUCCEEDED.to_string());
        resource
    }

    #[tokio::test]
    async fn reserve_best_fit_selects_smallest_sufficient_gpu() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 30_000, 0), gpu(1, "gpu:1", 50_000, 0)]);

        let mut msg = payload_with_resource_profile("run-1", "__unknown_workflow__", 1, 20_000);
        let selected = table.reserve(&mut msg).await.unwrap();

        assert_eq!(
            selected,
            vec![ReservedResource {
                stream_name: "gpu:0".to_string(),
                resource_id: 0,
            }],
            "best-fit should pick tighter GPU first"
        );
        assert_eq!(msg.gpus_required, 1);
        assert_eq!(msg.memory_mb, 20_000);
        assert_eq!(table.used_memory_mb(0).await, Some(20_000));
        assert_eq!(table.used_memory_mb(1).await, Some(0));
    }

    #[tokio::test]
    async fn reserve_best_fit_returns_error_when_all_gpus_too_small() {
        let table = table_with_gpus(vec![
            gpu(0, "gpu:0", 20_000, 0),
            gpu(1, "gpu:1", 23_000, 5_000),
        ]);

        let mut msg = payload_with_resource_profile("run-2", "__unknown_workflow__", 1, 20_000);
        let result = table.reserve(&mut msg).await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "should fail when no GPU can ever satisfy required memory"
        );
        assert!(
            !is_reservation_blocked_error(&error),
            "insufficient total capacity should be a hard error"
        );
        assert_eq!(table.used_memory_mb(0).await, Some(0));
        assert_eq!(table.used_memory_mb(1).await, Some(5_000));
    }

    #[tokio::test]
    async fn reserve_adds_new_reservations_on_top_of_existing_observed_usage() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 40_000, 20_000)]);

        let mut first = payload_with_resource_profile("run-1", "__unknown_workflow__", 1, 10_000);
        table.reserve(&mut first).await.unwrap();

        assert_eq!(
            table.used_memory_mb(0).await,
            Some(30_000),
            "new reservations should add on top of already-observed GPU usage"
        );

        let mut second = payload_with_resource_profile("run-2", "__unknown_workflow__", 1, 15_000);
        let error = table.reserve(&mut second).await.unwrap_err();
        assert!(
            is_reservation_blocked_error(&error),
            "admission should block once observed usage plus reserved memory exceeds availability"
        );
    }

    #[tokio::test]
    async fn reserve_keeps_admission_blocked_when_discovery_drops_below_durable_accounted_usage() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 40_000, 20_000)]);

        let mut first = payload_with_resource_profile("run-1", "__unknown_workflow__", 1, 10_000);
        table.reserve(&mut first).await.unwrap();
        assert_eq!(table.used_memory_mb(0).await, Some(30_000));

        table
            .sync_from_discovery_update(DiscoveryUpdate::Authoritative(vec![gpu(
                0, "gpu:0", 40_000, 10_000,
            )]))
            .await
            .expect("discovery refresh should succeed");

        assert_eq!(
            table.used_memory_mb(0).await,
            Some(30_000),
            "lower live discovery should not erase durable accounted pressure"
        );

        let mut second = payload_with_resource_profile("run-2", "__unknown_workflow__", 1, 5_000);
        let error = table.reserve(&mut second).await.unwrap_err();
        assert!(
            is_reservation_blocked_error(&error),
            "admission should remain blocked while durable accounted usage still exceeds availability"
        );
    }

    #[tokio::test]
    async fn release_drops_observed_floor_after_last_reservation() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 40_000, 472)]);

        let mut first = payload_with_resource_profile("run-1", "__unknown_workflow__", 1, 4_777);
        table.reserve(&mut first).await.unwrap();
        assert_eq!(table.used_memory_mb(0).await, Some(5_249));

        table.release(0, 4_777).await.unwrap();
        assert_eq!(
            table.used_memory_mb(0).await,
            Some(472),
            "release should leave only live discovered usage, not a durable reservation floor"
        );

        table
            .sync_from_discovery_update(DiscoveryUpdate::Authoritative(vec![gpu(
                0, "gpu:0", 40_000, 0,
            )]))
            .await
            .expect("discovery refresh should succeed");

        assert_eq!(
            table.used_memory_mb(0).await,
            Some(0),
            "stale observed floor must not persist after discovery reports idle"
        );
    }

    #[tokio::test]
    async fn reserve_returns_non_blocking_error_when_not_enough_workers_can_satisfy_request() {
        let table = table_with_gpus(vec![worker(
            0,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        )]);

        let mut msg = payload("run-no-match", "demo-plugin");
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.physicsnemo".to_string()),
            tags: Some(vec!["physicsnemo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "mismatched capabilities should surface a hard error"
        );
        assert!(
            !is_reservation_blocked_error(&error),
            "capability mismatches should go through retry/DLQ instead of blocking"
        );
    }

    #[tokio::test]
    async fn reserve_keeps_loading_worker_capability_mismatch_as_hard_error() {
        let mut loading_worker = worker(
            0,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        );
        loading_worker.warmup_status = Some("warming".to_string());
        let table = table_with_gpus(vec![loading_worker]);

        let mut msg = payload("run-loading-no-match", "demo-plugin");
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.physicsnemo".to_string()),
            tags: Some(vec!["physicsnemo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "loading workers with mismatched capabilities should remain ineligible"
        );
        assert!(
            !is_reservation_blocked_error(&error),
            "temporary loading only blocks after static requirements match"
        );
    }

    #[tokio::test]
    async fn reserve_keeps_matching_worker_memory_shortage_as_blocked_error() {
        let table = table_with_gpus(vec![worker(
            0,
            "execute.python.gpu.demo",
            16_000,
            15_000,
            "python.gpu.demo",
            &["demo", "gpu"],
        )]);

        let mut msg = payload("run-memory-blocked", "demo-plugin");
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            is_reservation_blocked_error(&error),
            "memory pressure on otherwise compatible workers should remain blocked"
        );
    }

    #[tokio::test]
    async fn reserve_selects_worker_with_matching_cached_workflow_id() {
        let table = table_with_gpus(vec![
            worker_cached_for(
                worker(
                    0,
                    "execute.python.gpu.demo-a",
                    32_000,
                    0,
                    "python.gpu.demo",
                    &["demo", "gpu"],
                ),
                "workflow-a",
            ),
            worker_cached_for(
                worker(
                    1,
                    "execute.python.gpu.demo-b",
                    32_000,
                    0,
                    "python.gpu.demo",
                    &["demo", "gpu"],
                ),
                "Workflow-B",
            ),
        ]);

        let mut msg = payload("run-cache-hit", "workflow-b");
        msg.workflow_id = Some("workflow-b".to_string());
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string(), "gpu".to_string()]),
        });

        let selected = table.reserve(&mut msg).await.unwrap();

        assert_eq!(
            selected,
            vec![ReservedResource {
                stream_name: "execute.python.gpu.demo-b".to_string(),
                resource_id: 1,
            }]
        );
    }

    #[tokio::test]
    async fn reserve_rejects_worker_with_mismatched_cached_workflow_id() {
        let table = table_with_gpus(vec![worker_cached_for(
            worker(
                0,
                "execute.python.gpu.demo",
                32_000,
                0,
                "python.gpu.demo",
                &["demo", "gpu"],
            ),
            "workflow-a",
        )]);

        let mut msg = payload("run-cache-miss", "workflow-b");
        msg.workflow_id = Some("workflow-b".to_string());
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "cached workflow mismatches should remove the worker from the candidate set"
        );
        assert!(
            !is_reservation_blocked_error(&error),
            "cache-affinity mismatches should remain hard ineligible cases"
        );
    }

    #[tokio::test]
    async fn reserve_allows_worker_without_workflow_affinity() {
        let mut gpu = worker(
            0,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        );
        gpu.warmup_status = Some("skipped".to_string());
        let table = table_with_gpus(vec![gpu]);

        let mut msg = payload("run-without-affinity", "workflow-b");
        msg.workflow_id = Some("workflow-b".to_string());
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string(), "gpu".to_string()]),
        });

        let selected = table.reserve(&mut msg).await.unwrap();
        assert_eq!(
            selected,
            vec![ReservedResource {
                stream_name: "execute.python.gpu.demo".to_string(),
                resource_id: 0,
            }]
        );
    }

    #[tokio::test]
    async fn reserve_blocks_loading_worker_that_matches_static_requirements() {
        let mut gpu = worker(
            0,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        );
        gpu.warmup_status = Some("warming".to_string());
        let table = table_with_gpus(vec![gpu]);

        let mut msg = payload("run-warming", "demo-plugin");
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            is_reservation_blocked_error(&error),
            "temporary loading should block when the worker otherwise matches"
        );
    }

    #[tokio::test]
    async fn reserve_rejects_loading_worker_with_mismatched_warmup_workflow() {
        let mut gpu = worker(
            0,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        );
        gpu.warmup_status = Some("warming".to_string());
        gpu.warmup_workflow_id = Some("workflow-a".to_string());
        let table = table_with_gpus(vec![gpu]);

        let mut msg = payload("run-warming-mismatch", "workflow-b");
        msg.workflow_id = Some("workflow-b".to_string());
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "mismatched pending warmup workflow should be hard ineligible"
        );
        assert!(
            !is_reservation_blocked_error(&error),
            "mismatched pending warmup workflow should not block as loading"
        );
    }

    #[tokio::test]
    async fn reserve_rejects_failed_warmup_worker_as_hard_error() {
        let mut gpu = worker(
            0,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        );
        gpu.warmup_status = Some("failed".to_string());
        let table = table_with_gpus(vec![gpu]);

        let mut msg = payload("run-failed-warmup", "demo-plugin");
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(1),
            memory_mb: Some(2_048),
            executor_class: Some("python.gpu.demo".to_string()),
            tags: Some(vec!["demo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "failed warmup should not block indefinitely"
        );
        assert!(
            !is_reservation_blocked_error(&error),
            "non-loading warmup failures should remain hard ineligible cases"
        );
    }

    #[tokio::test]
    async fn reserve_rejects_zero_gpu_explicit_profile() {
        let table = table_with_gpus(vec![worker(
            0,
            "execute.python.gpu.physicsnemo",
            32_000,
            0,
            "python.gpu.physicsnemo",
            &["physicsnemo", "gpu"],
        )]);

        let mut msg = payload("run-gpu", "demo-plugin");
        msg.resource_profile = Some(ScheduleResourceProfile {
            gpus_required: Some(0),
            memory_mb: Some(4_096),
            executor_class: Some("python.gpu.physicsnemo".to_string()),
            tags: Some(vec!["physicsnemo".to_string(), "gpu".to_string()]),
        });

        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("resource_profile.gpus_required must be at least 1"),
            "zero-GPU explicit profiles should be rejected"
        );
    }

    #[tokio::test]
    async fn sync_from_snapshot_uses_discovered_used_memory_when_it_is_higher() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 16_000, 2_000)]);

        table
            .sync_from_discovery_update(DiscoveryUpdate::Authoritative(vec![gpu(
                0, "gpu:0", 16_000, 9_000,
            )]))
            .await
            .expect("snapshot sync should succeed");

        assert_eq!(table.used_memory_mb(0).await, Some(9_000));
    }

    #[tokio::test]
    async fn sync_from_snapshot_replaces_discovered_used_memory_with_latest_snapshot() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 16_000, 9_000)]);

        table
            .sync_from_discovery_update(DiscoveryUpdate::Authoritative(vec![gpu(
                0, "gpu:0", 16_000, 2_000,
            )]))
            .await
            .expect("snapshot sync should succeed");

        assert_eq!(table.used_memory_mb(0).await, Some(2_000));
    }

    #[tokio::test]
    async fn sync_from_discovery_update_preserves_existing_gpu_entries_when_gpu_update_is_stale() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 16_000, 9_000)]);

        table
            .sync_from_discovery_update(DiscoveryUpdate::Stale)
            .await
            .expect("stale GPU updates should preserve existing GPU workers");

        assert_eq!(table.used_memory_mb(0).await, Some(9_000));
    }

    #[tokio::test]
    async fn sync_from_discovery_update_clears_entries_when_snapshot_is_authoritatively_empty() {
        let table = table_with_gpus(vec![gpu(0, "gpu:0", 16_000, 9_000)]);

        table
            .sync_from_discovery_update(DiscoveryUpdate::Authoritative(Vec::new()))
            .await
            .expect("empty authoritative snapshots should clear tracked GPU workers");

        assert_eq!(table.used_memory_mb(0).await, None);
    }

    #[tokio::test]
    async fn reserve_uses_known_profile_when_resource_profile_is_missing() {
        let _guard = test_env::env_lock().lock().await;
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[{"workflow":"legacy-demo","gpus.used":1,"peak":{"memory.used":"2048 MiB"}}]}"#,
            ),
        );

        let table = table_with_gpus(vec![worker(
            1,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        )]);

        let mut msg = payload("run-legacy", "legacy-demo");
        msg.resource_profile = None;

        let selected = table.reserve(&mut msg).await.unwrap();
        assert_eq!(
            selected,
            vec![ReservedResource {
                stream_name: "execute.python.gpu.demo".to_string(),
                resource_id: 1,
            }]
        );
        assert_eq!(msg.gpus_required, 1);
        assert_eq!(msg.memory_mb, 2_048);
        assert_eq!(
            msg.resource_profile
                .as_ref()
                .and_then(|profile| profile.gpus_required),
            Some(1),
            "known-profile fallback should synthesize resource_profile.gpus_required"
        );
        assert_eq!(
            msg.resource_profile
                .as_ref()
                .and_then(|profile| profile.memory_mb),
            Some(2_048),
            "known-profile fallback should synthesize resource_profile.memory_mb"
        );
        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn reserve_rejects_known_profile_fallback_when_gpu_capabilities_are_ambiguous() {
        let _guard = test_env::env_lock().lock().await;
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[{"workflow":"legacy-demo","gpus.used":1,"peak":{"memory.used":"2048 MiB"}}]}"#,
            ),
        );

        let table = table_with_gpus(vec![
            worker(
                0,
                "execute.python.gpu.demo-a",
                32_000,
                0,
                "python.gpu.demo_a",
                &["demo-a", "gpu"],
            ),
            worker(
                1,
                "execute.python.gpu.demo-b",
                32_000,
                0,
                "python.gpu.demo_b",
                &["demo-b", "gpu"],
            ),
        ]);

        let mut msg = payload("run-legacy", "legacy-demo");
        msg.resource_profile = None;

        let result = table.reserve(&mut msg).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("resource_profile"),
            "known-profile fallback should require explicit resource_profile when worker capabilities differ"
        );

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn reserve_rejects_missing_resource_profile_for_unknown_workflow() {
        let _guard = test_env::env_lock().lock().await;
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        test_env::set_env_var("SCHEDULER_PROFILES_JSON", None);

        let table = table_with_gpus(vec![gpu(0, "gpu:0", 40_000, 0)]);
        let mut msg = payload("run-unknown", "workflow-without-profile");
        msg.resource_profile = None;

        let result = table.reserve(&mut msg).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("resource_profile"));

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn reserve_rejects_known_profile_with_zero_gpu_requirement() {
        let _guard = test_env::env_lock().lock().await;
        let prev_profiles = std::env::var("SCHEDULER_PROFILES_JSON").ok();
        test_env::set_env_var(
            "SCHEDULER_PROFILES_JSON",
            Some(
                r#"{"profiles":[{"workflow":"legacy-cpu","gpus.used":0,"peak":{"memory.used":"1024 MiB"}}]}"#,
            ),
        );

        let table = table_with_gpus(vec![worker(
            1,
            "execute.python.gpu.demo",
            32_000,
            0,
            "python.gpu.demo",
            &["demo", "gpu"],
        )]);
        let mut msg = payload("run-legacy", "legacy-cpu");
        msg.resource_profile = None;

        let result = table.reserve(&mut msg).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must request at least one GPU"),
            "known-profile fallback should reject zero-GPU profiles"
        );

        test_env::set_env_var("SCHEDULER_PROFILES_JSON", prev_profiles.as_deref());
    }

    #[tokio::test]
    async fn release_saturates_to_zero_and_errors_when_resource_has_no_reservation() {
        let table = table_with_gpus(vec![gpu(7, "gpu:7", 40_000, 0)]);
        let mut msg =
            payload_with_resource_profile("run-release", "__unknown_workflow__", 1, 8_000);

        table.reserve(&mut msg).await.unwrap();
        table.release(7, 20_000).await.unwrap();
        assert_eq!(table.used_memory_mb(7).await, Some(0));

        let unknown_release = table.release(7, 1_000).await;
        assert!(
            unknown_release.is_err(),
            "release should fail when the resource has no durable reserved allocation"
        );
    }

    #[tokio::test]
    async fn sync_from_discovery_update_deduplicates_by_resource_id() {
        let table = ResourceReservationTable::new_for_tests(
            DEFAULT_MEMORY_UTILIZATION_PERCENT,
            SchedulingStrategy::BestFit,
        );
        table
            .sync_from_discovery_update(DiscoveryUpdate::Authoritative(vec![
                gpu(0, "gpu:pod-a:0", 32_000, 0),
                gpu(0, "gpu:pod-b:0", 32_000, 0),
            ]))
            .await
            .expect("duplicate resource ids in discovery should still sync");

        let mut msg = payload_with_resource_profile("run-dup", "__unknown_workflow__", 2, 2_048);
        let error = table.reserve(&mut msg).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "duplicate stream aliases for one resource id must not create extra schedulable capacity"
        );
    }

    #[tokio::test]
    async fn reserve_best_fit_respects_custom_memory_limit() {
        // GPU has 10000 total. With limit=40% => usable=4000, needs 5000 => should fail.
        let table = table_with_gpus_and_limit(vec![gpu(0, "gpu:0", 10_000, 0)], 40);
        let mut p1 = payload_with_resource_profile("run-1", "wf", 1, 5_000);
        let result = table.reserve(&mut p1).await;
        assert!(
            result.is_err(),
            "40% limit should reject 5000 MiB on 10000 GPU"
        );

        // Same GPU with limit=80% => usable=8000 >= 5000 => should succeed.
        let table = table_with_gpus_and_limit(vec![gpu(0, "gpu:0", 10_000, 0)], 80);
        let mut p2 = payload_with_resource_profile("run-2", "wf", 1, 5_000);
        let result = table.reserve(&mut p2).await;
        assert!(
            result.is_ok(),
            "80% limit should accept 5000 MiB on 10000 GPU"
        );
    }

    // --- PR-054: unknown strategy strings are rejected ---

    #[test]
    fn from_str_config_rejects_unknown_strategy() {
        let result = SchedulingStrategy::parse("best_fti");
        assert!(result.is_err(), "typo 'best_fti' should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("best_fit") && err_msg.contains("round_robin"),
            "error should list valid options, got: {err_msg}"
        );
    }

    #[test]
    fn from_str_config_accepts_known_strategies() {
        assert_eq!(
            SchedulingStrategy::parse("best_fit").unwrap(),
            SchedulingStrategy::BestFit
        );
        assert_eq!(
            SchedulingStrategy::parse("round_robin").unwrap(),
            SchedulingStrategy::RoundRobin
        );
        assert_eq!(
            SchedulingStrategy::parse("RoundRobin").unwrap(),
            SchedulingStrategy::RoundRobin
        );
    }

    // --- PR-004: round-robin strategy ---

    #[tokio::test]
    async fn round_robin_cycles_through_gpus() {
        let gpus = vec![
            gpu(0, "gpu:0", 40_000, 0),
            gpu(1, "gpu:1", 40_000, 0),
            gpu(2, "gpu:2", 40_000, 0),
        ];
        let table = table_with_gpus_strategy(gpus, SchedulingStrategy::RoundRobin);

        let mut p1 = payload_with_resource_profile("r1", "wf", 1, 1_000);
        let ids1 = table.reserve(&mut p1).await.unwrap();

        let mut p2 = payload_with_resource_profile("r2", "wf", 1, 1_000);
        let ids2 = table.reserve(&mut p2).await.unwrap();

        let mut p3 = payload_with_resource_profile("r3", "wf", 1, 1_000);
        let ids3 = table.reserve(&mut p3).await.unwrap();

        assert_ne!(ids1, ids2, "consecutive reservations should differ");
        assert_ne!(ids2, ids3, "consecutive reservations should differ");
    }

    #[tokio::test]
    async fn round_robin_ignores_gpu_memory_pressure() {
        let table = table_with_gpus_strategy(
            vec![gpu(0, "gpu:0", 40_000, 39_000)],
            SchedulingStrategy::RoundRobin,
        );

        let mut payload = payload_with_resource_profile("r1", "wf", 1, 20_000);
        let selected = table.reserve(&mut payload).await.unwrap();

        assert_eq!(
            selected,
            vec![ReservedResource {
                stream_name: "gpu:0".to_string(),
                resource_id: 0,
            }]
        );
    }

    #[tokio::test]
    async fn round_robin_rejects_gpus_below_total_memory_requirement() {
        let table = table_with_gpus_strategy(
            vec![gpu(0, "gpu:0", 10_000, 0)],
            SchedulingStrategy::RoundRobin,
        );

        let mut payload = payload_with_resource_profile("r1", "wf", 1, 20_000);
        let error = table.reserve(&mut payload).await.unwrap_err();

        assert!(
            error
                .to_string()
                .contains("not enough workers can satisfy the requested requirements"),
            "round-robin should still reject workers whose total usable memory is too small"
        );
    }
}
