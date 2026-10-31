//! Registry changes are fenced by the exact value owned by this process.
//! Warming is visible before any model is loaded; no execute polling starts
//! until the model-executing self-test and the available transition succeed.
//! A retired process releases its live lease while retaining a warming logical
//! slot, so the existing scheduler keeps queued work blocked through restarts.
use anyhow::{Context, Result, ensure};
use scicomp_rq::QueueManager;
use serde_json::{Value, json};
use std::time::Duration;
use tokio::sync::Mutex;

const REGISTRY: &str = "gpu:registry";
#[cfg(test)]
use crate::test_support as support;
const UPDATE: &str = r#"
local kind = redis.call('TYPE', KEYS[1]).ok
if kind ~= 'none' and kind ~= 'hash' then
  return redis.error_reply('native registry must be a hash')
end
local lease_kind = redis.call('TYPE', KEYS[2]).ok
if lease_kind ~= 'none' and lease_kind ~= 'string' then
  return redis.error_reply('native worker lease must be a string')
end
local ttl = tonumber(ARGV[6])
if not ttl or ttl < 1 or ARGV[2] == '' or ARGV[5] == '' then
  return redis.error_reply('invalid native registry arguments')
end
if ARGV[1] ~= 'create' and ARGV[1] ~= 'replace' and ARGV[1] ~= 'release' then
  return redis.error_reply('invalid native registry operation')
end
local current = redis.call('HGET', KEYS[1], ARGV[2])
local owner = redis.call('GET', KEYS[2])
if ARGV[1] == 'create' then
  if owner and (owner ~= ARGV[5] or current ~= ARGV[4]) then return 0 end
  redis.call('SET', KEYS[2], ARGV[5], 'PX', ttl)
  redis.call('HSET', KEYS[1], ARGV[2], ARGV[4])
  return 1
end
if ARGV[1] == 'release' and not current and not owner then return 1 end
if owner ~= ARGV[5] and not (ARGV[1] == 'release' and not owner) then return 0 end
if ARGV[1] == 'replace' then
  if current ~= ARGV[3] and current ~= ARGV[4] then return 0 end
  redis.call('HSET', KEYS[1], ARGV[2], ARGV[4])
  redis.call('PEXPIRE', KEYS[2], ttl)
elseif ARGV[1] == 'release' then
  local ok, metadata = pcall(cjson.decode, current or '')
  if not ok or type(metadata) ~= 'table' or metadata.worker_instance_id ~= ARGV[5] then return 0 end
  redis.call('HSET', KEYS[1], ARGV[2], ARGV[4])
  redis.call('DEL', KEYS[2])
end
return 1
"#;

pub(crate) fn lease_key(field: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("native:worker-lease:{:x}", Sha256::digest(field.as_bytes()))
}

fn restart_pending(metadata: &mut Value) {
    if metadata["model_cache"]["warmup"]["reason"] != "restart_pending" {
        metadata["updated_at"] = json!(chrono::Utc::now().to_rfc3339());
    }
    metadata["status"] = json!("warming");
    metadata["model_cache"] = json!({
        "schema_version":1,"scope":"process","entries":[],"total_entries":0,
        "warmup":{"workflow_id":metadata["workflow_id"],"status":"warming","reason":"restart_pending"}
    });
}

pub struct WorkerRegistration {
    qm: QueueManager,
    field: String,
    value: Mutex<String>,
    owner: String,
    lease_key: String,
    lease_millis: u64,
}
#[derive(Debug)]
struct SlotBusy;
impl std::fmt::Display for SlotBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native worker slot is still owned by another process")
    }
}
impl std::error::Error for SlotBusy {}
impl WorkerRegistration {
    pub async fn acquire(
        qm: QueueManager,
        field: String,
        metadata: Value,
        deadline: tokio::time::Instant,
    ) -> Result<Self> {
        loop {
            ensure!(
                tokio::time::Instant::now() < deadline,
                "native worker slot acquisition deadline exceeded"
            );
            match tokio::time::timeout_at(
                deadline,
                Self::warming(qm.clone(), field.clone(), metadata.clone()),
            )
            .await
            .context("native worker slot acquisition deadline exceeded")?
            {
                Ok(registration) => return Ok(registration),
                Err(error) if error.downcast_ref::<SlotBusy>().is_some() => {
                    ensure!(
                        tokio::time::Instant::now() < deadline,
                        "native worker slot acquisition deadline exceeded"
                    );
                    tokio::time::sleep_until(
                        (tokio::time::Instant::now() + Duration::from_millis(200)).min(deadline),
                    )
                    .await;
                }
                Err(error) => return Err(error),
            }
        }
    }
    pub async fn warming(qm: QueueManager, field: String, metadata: Value) -> Result<Self> {
        Self::warming_with_lease(qm, field, metadata, 15_000).await
    }
    async fn warming_with_lease(
        qm: QueueManager,
        field: String,
        mut metadata: Value,
        lease_millis: u64,
    ) -> Result<Self> {
        ensure!(!field.is_empty(), "worker registry identity is empty");
        ensure!(metadata.is_object(), "worker metadata must be an object");
        metadata["status"] = json!("warming");
        metadata["registered_at"] = json!(chrono::Utc::now().to_rfc3339());
        let owner = uuid::Uuid::new_v4().to_string();
        metadata["worker_instance_id"] = json!(owner);
        metadata["heartbeat_at"] = json!(chrono::Utc::now().to_rfc3339());
        metadata["updated_at"] = metadata["heartbeat_at"].clone();
        metadata["model_cache"] = json!({"schema_version":1,"scope":"process","entries":[],"total_entries":0,
            "warmup":{"workflow_id":metadata["workflow_id"],"status":"warming"}});
        let lease_key = lease_key(&field);
        let registration = Self {
            qm,
            field,
            value: Mutex::new(metadata.to_string()),
            owner,
            lease_key,
            lease_millis,
        };
        let value = registration.value.lock().await;
        if !registration.update("create", "", &value).await? {
            return Err(SlotBusy.into());
        }
        drop(value);
        Ok(registration)
    }
    async fn update(&self, operation: &str, previous: &str, next: &str) -> Result<bool> {
        let mut connection = self.qm.connection();
        let script = redis::Script::new(UPDATE);
        let mut call = script.prepare_invoke();
        call.key(REGISTRY)
            .key(&self.lease_key)
            .arg(operation)
            .arg(&self.field)
            .arg(previous)
            .arg(next)
            .arg(&self.owner)
            .arg(self.lease_millis);
        let accepted: i32 =
            tokio::time::timeout(Duration::from_secs(3), call.invoke_async(&mut connection))
                .await
                .context("native registry update deadline")??;
        Ok(accepted == 1)
    }
    pub async fn available(&self, report: &Value) -> Result<()> {
        ensure!(
            report["status"] == "passed",
            "native model self-test did not pass"
        );
        let mut previous = self.value.lock().await;
        let mut metadata: Value = serde_json::from_str(&previous)?;
        ensure!(
            metadata["status"] == "warming"
                && metadata["model_cache"]["warmup"]["reason"] != "restart_pending",
            "native worker no longer warming for startup"
        );
        metadata["status"] = json!("available");
        metadata["model_cache"]["entries"] =
            json!([{"workflow_id":metadata["workflow_id"],"status":"loaded"}]);
        metadata["model_cache"]["total_entries"] = json!(1);
        metadata["model_cache"]["warmup"]["status"] = json!("succeeded");
        metadata["model_cache"]["warmup"]["self_test"] = report.clone();
        let next = metadata.to_string();
        ensure!(
            self.update("replace", &previous, &next).await?,
            "native registry ownership was lost before readiness"
        );
        *previous = next;
        Ok(())
    }
    pub async fn check_ownership(&self) -> Result<()> {
        let expected = self.value.lock().await;
        let mut connection = self.qm.connection();
        let current: Option<String> = tokio::time::timeout(
            Duration::from_secs(3),
            redis::cmd("HGET")
                .arg(REGISTRY)
                .arg(&self.field)
                .query_async(&mut connection),
        )
        .await
        .context("native registry ownership deadline")??;
        ensure!(
            current.as_deref() == Some(expected.as_str()),
            "native registry ownership was lost"
        );
        let owner: Option<String> = tokio::time::timeout(
            Duration::from_secs(3),
            redis::cmd("GET")
                .arg(&self.lease_key)
                .query_async(&mut connection),
        )
        .await
        .context("native slot ownership deadline")??;
        ensure!(
            owner.as_deref() == Some(self.owner.as_str()),
            "native worker slot lease was lost"
        );
        Ok(())
    }
    pub async fn heartbeat(&self) -> Result<()> {
        let mut previous = self.value.lock().await;
        let mut metadata: Value = serde_json::from_str(&previous)?;
        metadata["heartbeat_at"] = json!(chrono::Utc::now().to_rfc3339());
        metadata["updated_at"] = metadata["heartbeat_at"].clone();
        let next = metadata.to_string();
        ensure!(
            self.update("replace", &previous, &next).await?,
            "native worker slot lease/registry ownership was lost"
        );
        *previous = next;
        Ok(())
    }
    pub async fn restarting(&self) -> Result<()> {
        let mut previous = self.value.lock().await;
        let mut metadata: Value = serde_json::from_str(&previous)?;
        restart_pending(&mut metadata);
        let next = metadata.to_string();
        ensure!(
            self.update("replace", &previous, &next).await?,
            "native registry ownership lost while restarting"
        );
        *previous = next;
        Ok(())
    }
    pub async fn release(&self) -> Result<()> {
        let mut previous = self.value.lock().await;
        let mut metadata: Value = serde_json::from_str(&previous)?;
        restart_pending(&mut metadata);
        let next = metadata.to_string();
        ensure!(
            self.update("release", &previous, &next).await?,
            "native registry ownership was lost during cleanup"
        );
        *previous = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    async fn value(qm: &QueueManager) -> Option<Value> {
        let mut conn = qm.connection();
        let value: Option<String> = redis::cmd("HGET")
            .arg(REGISTRY)
            .arg("worker")
            .query_async(&mut conn)
            .await
            .unwrap();
        value.map(|v| serde_json::from_str(&v).unwrap())
    }
    #[tokio::test]
    async fn warming_self_test_and_owned_release_are_visible_in_redis() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-startup-registry").await;
        let registration = WorkerRegistration::warming(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test","stream":"test:execute","status":"available"}),
        )
        .await
        .unwrap();
        assert_eq!(value(&qm).await.unwrap()["status"], "warming");
        assert!(
            registration
                .available(&json!({"status":"failed"}))
                .await
                .is_err()
        );
        assert_eq!(
            value(&qm).await.unwrap()["model_cache"]["warmup"]["status"],
            "warming"
        );
        registration
            .available(&json!({"status":"passed","models":[{}]}))
            .await
            .unwrap();
        let available = value(&qm).await.unwrap();
        assert_eq!(available["status"], "available");
        assert_eq!(available["model_cache"]["warmup"]["status"], "succeeded");
        registration.check_ownership().await.unwrap();
        registration.release().await.unwrap();
        assert_eq!(value(&qm).await.unwrap()["status"], "warming");
    }
    #[tokio::test]
    async fn retiring_generation_keeps_a_loading_slot_and_cannot_become_available_again() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-retiring-slot").await;
        let registration = WorkerRegistration::warming(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test","stream":"stable:execute","executor_class":"native.test"}),
        )
        .await
        .unwrap();
        registration
            .available(&json!({"status":"passed"}))
            .await
            .unwrap();
        registration.restarting().await.unwrap();
        let retiring = value(&qm).await.unwrap();
        assert_eq!(retiring["status"], "warming");
        assert_eq!(retiring["stream"], "stable:execute");
        assert_eq!(retiring["executor_class"], "native.test");
        assert_eq!(retiring["model_cache"]["entries"], json!([]));
        assert_eq!(retiring["model_cache"]["total_entries"], 0);
        assert_eq!(retiring["model_cache"]["warmup"]["workflow_id"], "test");
        assert_eq!(retiring["model_cache"]["warmup"]["status"], "warming");
        assert_eq!(
            retiring["model_cache"]["warmup"]["reason"],
            "restart_pending"
        );
        assert!(
            registration
                .available(&json!({"status":"passed"}))
                .await
                .is_err(),
            "a late startup self-test cannot reopen a retired generation"
        );
        assert_eq!(value(&qm).await.unwrap(), retiring);
    }
    #[tokio::test]
    async fn retired_slot_survives_lease_cleanup_and_replacement_is_owner_fenced() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-retired-slot").await;
        let metadata =
            json!({"workflow_id":"test","stream":"stable:execute","executor_class":"native.test"});
        let original = WorkerRegistration::warming(qm.clone(), "worker".into(), metadata.clone())
            .await
            .unwrap();
        original
            .available(&json!({"status":"passed"}))
            .await
            .unwrap();
        original.restarting().await.unwrap();
        original.release().await.unwrap();
        let retired = value(&qm)
            .await
            .expect("the logical slot must survive its process lease");
        assert_eq!(retired["status"], "warming");
        assert_eq!(retired["stream"], "stable:execute");
        assert_eq!(
            retired["model_cache"]["warmup"]["reason"],
            "restart_pending"
        );
        let mut conn = qm.connection();
        let owner: Option<String> = redis::cmd("GET")
            .arg(&original.lease_key)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert!(owner.is_none(), "retained slot must not imply a live owner");
        assert!(original.heartbeat().await.is_err());
        assert!(original.check_ownership().await.is_err());
        original.release().await.unwrap();
        let replacement = WorkerRegistration::warming(qm.clone(), "worker".into(), metadata)
            .await
            .unwrap();
        let warming = value(&qm).await.unwrap();
        assert_eq!(warming["status"], "warming");
        assert_eq!(warming["stream"], retired["stream"]);
        assert_ne!(warming["worker_instance_id"], retired["worker_instance_id"]);
        assert!(original.release().await.is_err());
        assert_eq!(value(&qm).await.unwrap(), warming);
        replacement
            .available(&json!({"status":"passed"}))
            .await
            .unwrap();
        assert_eq!(value(&qm).await.unwrap()["status"], "available");
        replacement.release().await.unwrap();
    }
    #[tokio::test]
    async fn failed_startup_retains_a_waiting_slot_without_a_live_owner() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-failed-startup-slot").await;
        let registration = WorkerRegistration::warming(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test","stream":"stable:execute"}),
        )
        .await
        .unwrap();
        assert!(
            registration
                .available(&json!({"status":"failed"}))
                .await
                .is_err()
        );
        registration.release().await.unwrap();
        let waiting = value(&qm)
            .await
            .expect("failed native generation must not remove its logical slot");
        assert_eq!(waiting["status"], "warming");
        assert_eq!(waiting["model_cache"]["entries"], json!([]));
        assert_eq!(
            waiting["model_cache"]["warmup"]["reason"],
            "restart_pending"
        );
        assert!(registration.check_ownership().await.is_err());
    }
    #[tokio::test]
    async fn replaced_registry_identity_cannot_be_overwritten_or_deleted() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-registry-replacement").await;
        let registration =
            WorkerRegistration::warming(qm.clone(), "worker".into(), json!({"workflow_id":"test"}))
                .await
                .unwrap();
        let replacement = json!({"status":"warming","owner":"other"});
        let mut conn = qm.connection();
        let _: usize = redis::cmd("HSET")
            .arg(REGISTRY)
            .arg("worker")
            .arg(replacement.to_string())
            .query_async(&mut conn)
            .await
            .unwrap();
        assert!(
            registration
                .available(&json!({"status":"passed"}))
                .await
                .is_err()
        );
        assert!(registration.check_ownership().await.is_err());
        assert!(registration.release().await.is_err());
        assert_eq!(value(&qm).await.unwrap(), replacement);
    }
    #[tokio::test]
    async fn wrong_registry_key_type_is_rejected_before_mutation() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-registry-wrong-type").await;
        let mut conn = qm.connection();
        let _: () = redis::cmd("SET")
            .arg(REGISTRY)
            .arg("existing")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert!(
            WorkerRegistration::warming(qm, "worker".into(), json!({"workflow_id":"test"}))
                .await
                .is_err()
        );
        let after: String = redis::cmd("GET")
            .arg(REGISTRY)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(after, "existing");
    }
    #[tokio::test]
    async fn wrong_lease_type_preserves_the_existing_registry_record() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-lease-wrong-type").await;
        let registration =
            WorkerRegistration::warming(qm.clone(), "worker".into(), json!({"workflow_id":"test"}))
                .await
                .unwrap();
        let before = value(&qm).await.unwrap();
        let mut conn = qm.connection();
        let _: () = redis::cmd("DEL")
            .arg(&registration.lease_key)
            .query_async(&mut conn)
            .await
            .unwrap();
        let _: usize = redis::cmd("LPUSH")
            .arg(&registration.lease_key)
            .arg("sentinel")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert!(registration.heartbeat().await.is_err());
        assert!(registration.release().await.is_err());
        assert!(
            WorkerRegistration::warming(qm.clone(), "worker".into(), json!({"workflow_id":"test"}))
                .await
                .is_err()
        );
        assert_eq!(value(&qm).await.unwrap(), before);
        let contents: Vec<String> = redis::cmd("LRANGE")
            .arg(&registration.lease_key)
            .arg(0)
            .arg(-1)
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(contents, ["sentinel"]);
    }
    #[tokio::test]
    async fn lost_heartbeat_reply_is_idempotent_and_owned_cleanup_still_succeeds() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-heartbeat-lost-reply").await;
        let registration =
            WorkerRegistration::warming(qm.clone(), "worker".into(), json!({"workflow_id":"test"}))
                .await
                .unwrap();
        let previous = registration.value.lock().await.clone();
        let mut next: Value = serde_json::from_str(&previous).unwrap();
        next["heartbeat_at"] = json!("reply-lost-after-commit");
        let next = next.to_string();
        // Redis commits, then the caller loses the reply before updating its cache.
        assert!(
            registration
                .update("replace", &previous, &next)
                .await
                .unwrap()
        );
        assert!(
            registration
                .update("replace", &previous, &next)
                .await
                .unwrap()
        );
        assert_ne!(registration.value.lock().await.as_str(), next);
        registration.release().await.unwrap();
        assert_eq!(value(&qm).await.unwrap()["status"], "warming");
    }
    #[tokio::test]
    async fn lost_release_reply_keeps_the_slot_waiting_and_cleanup_is_idempotent() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-release-lost-reply").await;
        let registration = WorkerRegistration::warming(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test","stream":"stable:execute"}),
        )
        .await
        .unwrap();
        registration
            .available(&json!({"status":"passed"}))
            .await
            .unwrap();
        let previous = registration.value.lock().await.clone();
        let mut next: Value = serde_json::from_str(&previous).unwrap();
        restart_pending(&mut next);
        let next = next.to_string();
        // Redis commits the lease release, but the process still caches available.
        assert!(
            registration
                .update("release", &previous, &next)
                .await
                .unwrap()
        );
        let waiting = value(&qm).await.unwrap();
        assert_eq!(waiting["status"], "warming");
        assert!(registration.heartbeat().await.is_err());
        assert!(registration.check_ownership().await.is_err());
        assert!(
            registration
                .available(&json!({"status":"passed"}))
                .await
                .is_err()
        );
        registration.release().await.unwrap();
        let waiting = value(&qm).await.unwrap();
        registration.release().await.unwrap();
        assert_eq!(value(&qm).await.unwrap(), waiting);
    }
    #[tokio::test]
    async fn dead_owner_lease_expires_and_restart_takes_the_same_slot() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-slot-takeover").await;
        let original = WorkerRegistration::warming_with_lease(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test","stream":"stable:execute"}),
            150,
        )
        .await
        .unwrap();
        original
            .available(&json!({"status":"passed"}))
            .await
            .unwrap();
        assert!(
            WorkerRegistration::warming(qm.clone(), "worker".into(), json!({"workflow_id":"test"}))
                .await
                .is_err()
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
        let replacement = WorkerRegistration::warming(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test","stream":"stable:execute"}),
        )
        .await
        .unwrap();
        let next = value(&qm).await.unwrap();
        assert_eq!(next["status"], "warming");
        assert_eq!(next["stream"], "stable:execute");
        assert_ne!(next["worker_instance_id"], original.owner);
        assert!(original.heartbeat().await.is_err());
        assert!(
            original
                .available(&json!({"status":"passed"}))
                .await
                .is_err()
        );
        assert!(original.release().await.is_err());
        assert_eq!(value(&qm).await.unwrap(), next);
        replacement
            .available(&json!({"status":"passed"}))
            .await
            .unwrap();
        replacement.release().await.unwrap();
    }
    #[tokio::test]
    async fn heartbeat_extends_lease_without_changing_warming_or_identity() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-slot-heartbeat").await;
        let registration = WorkerRegistration::warming_with_lease(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test"}),
            1000,
        )
        .await
        .unwrap();
        let before = value(&qm).await.unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
        registration.heartbeat().await.unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
        registration.check_ownership().await.unwrap();
        let after = value(&qm).await.unwrap();
        assert_eq!(after["status"], "warming");
        assert_eq!(after["registered_at"], before["registered_at"]);
        assert_eq!(after["worker_instance_id"], before["worker_instance_id"]);
        assert_ne!(after["heartbeat_at"], before["heartbeat_at"]);
        registration.release().await.unwrap();
    }
    #[tokio::test]
    async fn supervised_restart_waits_for_dead_owner_without_spending_supervisor_retries() {
        let (_redis, qm) = support::spawn_test_queue_manager("native-slot-acquisition").await;
        let old = WorkerRegistration::warming_with_lease(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test"}),
            150,
        )
        .await
        .unwrap();
        let started = tokio::time::Instant::now();
        let replacement = WorkerRegistration::acquire(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test"}),
            started + Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert!(started.elapsed() >= Duration::from_millis(150));
        assert!(old.release().await.is_err());
        let deadline = tokio::time::Instant::now() + Duration::from_millis(40);
        let blocked = WorkerRegistration::acquire(
            qm.clone(),
            "worker".into(),
            json!({"workflow_id":"test"}),
            deadline,
        )
        .await;
        assert!(blocked.err().unwrap().to_string().contains("deadline"));
        replacement.release().await.unwrap();
    }
}
