/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fenced child outcomes and atomic retry/continuation dispatch. No GPU release
//! happens here: accepting an outcome is not proof that a dispatch released resources.
use anyhow::{Context, Result, ensure};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{RedisRoundStore, RoundPlan, identity};
use crate::roles::parent_run_state::STATE_TOMBSTONE_TTL_SECS;
use crate::roles::scatter::{ScatterPayloadTarget, project_scatter_payload};
use crate::roles::stage::{StageContext, update_stage_context};

#[cfg(test)]
mod tests;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Status {
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChildState {
    attempt: u32,
    token: String,
    result: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct RoundState {
    status: Status,
    children: Vec<ChildState>,
}

impl RoundState {
    pub(super) fn new(plan: &RoundPlan) -> Self {
        Self {
            status: Status::Waiting,
            children: plan
                .children
                .iter()
                .map(|child| ChildState {
                    attempt: 1,
                    token: child["round_context"]["attempt_token"]
                        .as_str()
                        .unwrap_or_default()
                        .into(),
                    result: None,
                })
                .collect(),
        }
    }
}

struct Dispatch {
    stream: String,
    stage: String,
    run_id: String,
    payload: Value,
}

// CAS compares the complete serialized state, avoiding a second revision counter.
// Rust owns JSON interpretation so scientific integers never pass through Lua doubles.
const COMMIT: &str = r#"
local function type_is(key, expected)
  local t = redis.call('TYPE', key)['ok']
  return t == 'none' or t == expected
end
if not type_is(KEYS[1], 'hash') or not type_is(KEYS[2], 'string')
   or not type_is(KEYS[3], 'string') or not type_is(KEYS[4], 'stream') then
  return redis.error_reply('ROUND_KEY_WRONG_TYPE')
end
if redis.call('HGET', KEYS[1], 'state') ~= ARGV[1] then return 0 end
if redis.call('EXISTS', KEYS[3]) ~= tonumber(ARGV[5]) then return 0 end
if redis.call('GET', KEYS[2]) ~= ARGV[4] then
  return redis.error_reply('ROUND_NOT_ACTIVE')
end
local output = nil
if ARGV[6] ~= '' then
  local r = redis.pcall('XADD', KEYS[4], '*', 'run_id', ARGV[6],
      'payload', ARGV[7], 'stage', ARGV[8])
  if type(r) == 'table' and r.err then return redis.error_reply(r.err) end
  output = r
end
local r = redis.pcall('HSET', KEYS[1], 'state', ARGV[2], 'status', ARGV[3])
if type(r) == 'table' and r.err then
  if output then redis.call('XDEL', KEYS[4], output) end
  return redis.error_reply(r.err)
end
if (ARGV[3] == 'failed' or ARGV[3] == 'cancelled') and ARGV[5] == '0' then
  redis.call('SET', KEYS[3], 'terminal', 'EX', ARGV[9])
end
if ARGV[3] == 'failed' or ARGV[3] == 'cancelled' then
  redis.call('EXPIRE', KEYS[3], ARGV[9])
end
if ARGV[3] ~= 'waiting' then
  redis.call('DEL', KEYS[2])
  redis.call('EXPIRE', KEYS[1], ARGV[9])
end
return 1
"#;

fn continuation(plan: &RoundPlan, state: &RoundState) -> Result<Dispatch> {
    let mut payload =
        project_scatter_payload(&plan.parent_payload, ScatterPayloadTarget::Continuation)?;
    let context: StageContext = serde_json::from_value(payload["stage_context"].clone())?;
    let target = context
        .pipeline
        .iter()
        .find(|stage| stage.id == plan.continuation_stage_id)
        .context("round continuation stage missing")?;
    let map = payload
        .as_object_mut()
        .context("round parent must be an object")?;
    let children: Vec<_> = state.children.iter().enumerate().map(|(index, child)| json!({
        "child_run_id": plan.children[index]["run_id"], "item_index": index, "result": child.result
    })).collect();
    map.insert("child_results".into(), json!(children));
    map.insert(
        "result".into(),
        json!({"status": "succeeded", "child_results": children}),
    );
    update_stage_context(map, target, "round gather")?;
    let (stream, stage) = if target.phase == "execute" {
        map.insert("dispatch_stage_id".into(), json!(target.id));
        (plan.schedule_stream.clone(), "schedule".into())
    } else {
        (target.queue.clone(), target.phase.clone())
    };
    if target.phase == "results" {
        let workflow = plan.parent_payload["workflow_id"]
            .as_str()
            .context("round workflow missing")?;
        let (execution, result_payload) = crate::roles::collect::build_execution_and_payload(
            &plan.parent_run_id,
            workflow,
            "succeeded",
            &payload["result"],
        )?;
        payload = json!({"run_id": plan.parent_run_id, "workflow": workflow, "status": "succeeded",
            "request": crate::roles::collect::build_request_envelope(&plan.parent_payload),
            "execution": execution, "payload": result_payload});
    }
    Ok(Dispatch {
        stream,
        stage,
        run_id: plan.parent_run_id.clone(),
        payload,
    })
}

fn terminal_result(plan: &RoundPlan, status: &str, error: &str) -> Result<Dispatch> {
    let workflow = plan.parent_payload["workflow_id"]
        .as_str()
        .context("round workflow missing")?;
    let context: StageContext =
        serde_json::from_value(plan.parent_payload["stage_context"].clone())?;
    let results_stages: Vec<_> = context
        .pipeline
        .iter()
        .filter(|stage| stage.phase == "results")
        .collect();
    let [results_stage] = results_stages.as_slice() else {
        anyhow::bail!("round pipeline must contain exactly one results stage");
    };
    let result = json!({"status": status, "error": error, "artifacts": []});
    let (execution, payload) = crate::roles::collect::build_execution_and_payload(
        &plan.parent_run_id,
        workflow,
        status,
        &result,
    )?;
    Ok(Dispatch {
        stream: results_stage.queue.clone(),
        stage: results_stage.phase.clone(),
        run_id: plan.parent_run_id.clone(),
        payload: json!({"run_id": plan.parent_run_id, "workflow": workflow, "status": status,
            "request": crate::roles::collect::build_request_envelope(&plan.parent_payload),
            "execution": execution, "payload": payload}),
    })
}

impl RedisRoundStore {
    pub(crate) async fn is_current_child(&self, run_id: &str, payload: &Value) -> Result<bool> {
        let reference = &payload["round_context"];
        let round_id = reference["round_id"].as_str().context("round_id missing")?;
        let index = usize::try_from(
            reference["child_index"]
                .as_u64()
                .context("child_index missing")?,
        )?;
        let mut conn = self.qm.connection();
        let values: (String, String) = redis::cmd("HMGET")
            .arg(format!("{}round:{round_id}", self.prefix))
            .arg("plan")
            .arg("state")
            .query_async(&mut conn)
            .await?;
        let plan: RoundPlan = serde_json::from_str(&values.0)?;
        let state: RoundState = serde_json::from_str(&values.1)?;
        let original = plan
            .children
            .get(index)
            .context("child index out of range")?;
        ensure!(
            original["run_id"].as_str() == Some(run_id),
            "child run_id mismatch"
        );
        let member = state.children.get(index).context("child state missing")?;
        Ok(state.status == Status::Waiting
            && member.result.is_none()
            && reference["attempt"].as_u64() == Some(u64::from(member.attempt))
            && reference["attempt_token"].as_str() == Some(&member.token))
    }

    /// Called by the existing collect role. Commit precedes source ACK; replay is
    /// harmless even if that ACK or the worker process was lost.
    pub(crate) async fn gather(&self, run_id: &str, payload: &Value) -> Result<()> {
        let reference = &payload["round_context"];
        let round_id = reference["round_id"].as_str().context("round_id missing")?;
        let index = reference["child_index"]
            .as_u64()
            .context("child_index missing")?;
        let index = usize::try_from(index)?;
        let attempt = reference["attempt"].as_u64().context("attempt missing")?;
        let token = reference["attempt_token"]
            .as_str()
            .context("attempt_token missing")?;
        let result = payload
            .get("result")
            .filter(|v| v.is_object())
            .context("child result missing")?;
        let outcome = result["status"].as_str().context("child status missing")?;
        ensure!(
            ["succeeded", "success", "completed", "failed", "cancelled"].contains(&outcome),
            "invalid child status"
        );
        let key = format!("{}round:{round_id}", self.prefix);
        let mut conn = self.qm.connection();
        let encoded: Option<String> = conn.hget(&key, "plan").await?;
        let plan: RoundPlan = serde_json::from_str(&encoded.context("round plan missing")?)?;
        ensure!(plan.round_id == round_id, "round identity mismatch");
        let child = plan
            .children
            .get(index)
            .context("child index out of range")?;
        ensure!(
            child["run_id"].as_str() == Some(run_id),
            "child run_id mismatch"
        );
        ensure!(
            payload["parent_run_id"].as_str() == Some(&plan.parent_run_id),
            "parent run_id mismatch"
        );
        loop {
            let before: String = conn.hget(&key, "state").await?;
            let mut state: RoundState = serde_json::from_str(&before)?;
            if state.status != Status::Waiting {
                return Ok(());
            }
            let member = state
                .children
                .get_mut(index)
                .context("child state missing")?;
            if u64::from(member.attempt) != attempt
                || member.token != token
                || member.result.is_some()
            {
                return Ok(());
            }
            let terminal: bool = conn
                .exists(format!("parent_terminal:{}", plan.parent_run_id))
                .await?;
            let mut dispatch = None;
            if terminal {
                state.status = Status::Cancelled;
            } else if outcome == "cancelled" {
                state.status = Status::Cancelled;
                dispatch = Some(terminal_result(
                    &plan,
                    "cancelled",
                    result["error"].as_str().unwrap_or("child cancelled"),
                )?);
            } else if ["succeeded", "success", "completed"].contains(&outcome) {
                member.result = Some(result.clone());
                if state.children.iter().all(|child| child.result.is_some()) {
                    state.status = Status::Succeeded;
                    dispatch = Some(continuation(&plan, &state)?);
                }
            } else if result["retryable"].as_bool() == Some(true)
                && member.attempt < plan.max_attempts
            {
                member.attempt += 1;
                member.token =
                    identity(&[round_id, &index.to_string(), &member.attempt.to_string()]);
                let mut retry = child.clone();
                retry["round_context"]["attempt"] = json!(member.attempt);
                retry["round_context"]["attempt_token"] = json!(member.token);
                let now_ms =
                    u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())?;
                retry["retry_not_before_ms"] =
                    json!(now_ms + u64::from(member.attempt - 1).min(2) * 1000);
                dispatch = Some(Dispatch {
                    stream: plan.schedule_stream.clone(),
                    stage: "schedule".into(),
                    run_id: run_id.into(),
                    payload: retry,
                });
            } else {
                member.result = Some(result.clone());
                state.status = Status::Failed;
                dispatch = Some(terminal_result(
                    &plan,
                    "failed",
                    result["error"]
                        .as_str()
                        .unwrap_or("child attempts exhausted"),
                )?);
            }
            let status = serde_json::to_value(&state.status)?;
            let destination = dispatch
                .as_ref()
                .map(|d| d.stream.as_str())
                .unwrap_or(&plan.schedule_stream);
            let committed: bool = redis::Script::new(COMMIT)
                .key(&key)
                .key(format!(
                    "{}active_round:{}",
                    self.prefix, plan.parent_run_id
                ))
                .key(format!("parent_terminal:{}", plan.parent_run_id))
                .key(format!("{}{destination}", self.prefix))
                .arg(&before)
                .arg(serde_json::to_string(&state)?)
                .arg(status.as_str().context("invalid round status")?)
                .arg(round_id)
                .arg(u8::from(terminal))
                .arg(dispatch.as_ref().map(|d| d.run_id.as_str()).unwrap_or(""))
                .arg(
                    dispatch
                        .as_ref()
                        .map(|d| serde_json::to_string(&d.payload))
                        .transpose()?
                        .unwrap_or_default(),
                )
                .arg(dispatch.as_ref().map(|d| d.stage.as_str()).unwrap_or(""))
                .arg(STATE_TOMBSTONE_TTL_SECS)
                .invoke_async(&mut conn)
                .await?;
            if committed {
                return Ok(());
            }
            tokio::task::yield_now().await;
        }
    }
}
