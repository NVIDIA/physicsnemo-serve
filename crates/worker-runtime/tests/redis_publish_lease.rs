mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use scicomp_rq::StreamKey;
use tokio::sync::Barrier;
use worker_runtime::config::RuntimeConfig;
use worker_runtime::engine::EngineBuilder;
use worker_runtime::traits::{BoxFuture, MessageSink, WorkerRole};
use worker_runtime::transport::redis::RedisTransport;

use support::spawn_test_queue_manager;

fn publish_lease_config() -> RuntimeConfig {
    publish_lease_config_with_max_dequeue_items(1)
}

fn publish_lease_config_with_max_dequeue_items(max_dequeue_items: usize) -> RuntimeConfig {
    serde_json::from_value(serde_json::json!({
        "stream_prefix": "test:",
        "max_retries": 2,
        "shared_dlq_stream": "dlq",
        "streams": ["publish", "dlq"],
        "roles": {
            "publish": {
                "inputs": [{
                    "stream": "publish",
                    "max_dequeue_items": max_dequeue_items,
                    "poll_interval_ms": 10,
                    "block_ms": 10,
                    "reclaim_idle_ms": 50
                }],
                "outputs": []
            }
        }
    }))
    .expect("publish lease config should parse")
}

struct SlowFirstRole {
    first_started: Arc<Barrier>,
    first_completed: Arc<AtomicUsize>,
    total_completed: Arc<AtomicUsize>,
    sleep_for: Duration,
}

impl WorkerRole for SlowFirstRole {
    fn name(&self) -> &'static str {
        "slow-first-publish-role"
    }

    fn handle<'a>(
        &'a self,
        msg: &'a scicomp_rq::Message,
        _stream: &'a str,
        _sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if msg.run_id() == "run-first" {
                self.first_started.wait().await;
                tokio::time::sleep(self.sleep_for).await;
                self.first_completed.fetch_add(1, Ordering::SeqCst);
            }
            self.total_completed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

struct SlowRole {
    started: Arc<Barrier>,
    completed: Arc<AtomicUsize>,
    sleep_for: Duration,
}

impl WorkerRole for SlowRole {
    fn name(&self) -> &'static str {
        "slow-publish-role"
    }

    fn handle<'a>(
        &'a self,
        _msg: &'a scicomp_rq::Message,
        _stream: &'a str,
        _sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.started.wait().await;
            tokio::time::sleep(self.sleep_for).await;
            self.completed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

struct CountingRole {
    handled: Arc<AtomicUsize>,
}

impl WorkerRole for CountingRole {
    fn name(&self) -> &'static str {
        "counting-publish-role"
    }

    fn handle<'a>(
        &'a self,
        _msg: &'a scicomp_rq::Message,
        _stream: &'a str,
        _sink: &'a dyn MessageSink,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.handled.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

#[tokio::test]
async fn redis_transport_renews_active_message_lease() {
    let (_server, qm) = spawn_test_queue_manager("redis-publish-lease").await;
    let config = publish_lease_config();

    let started = Arc::new(Barrier::new(2));
    let completed = Arc::new(AtomicUsize::new(0));
    let engine_a = EngineBuilder::new(&config, "publish")
        .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
        .role(Box::new(SlowRole {
            started: Arc::clone(&started),
            completed: Arc::clone(&completed),
            sleep_for: Duration::from_millis(220),
        }))
        .consumer("consumer-a")
        .build()
        .expect("worker A engine should build");
    engine_a
        .ensure_consumer_groups()
        .await
        .expect("consumer group should exist");

    qm.enqueue_to_stream(
        &StreamKey::new("test:publish"),
        "run-large-publish",
        r#"{"ok":true}"#,
        "publish",
    )
    .await
    .expect("publish message should enqueue");

    let worker_a = tokio::spawn(async move { engine_a.run_once().await });
    started.wait().await;
    tokio::time::sleep(Duration::from_millis(110)).await;

    let reclaimed_by_b = Arc::new(AtomicUsize::new(0));
    let engine_b = EngineBuilder::new(&config, "publish")
        .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
        .role(Box::new(CountingRole {
            handled: Arc::clone(&reclaimed_by_b),
        }))
        .consumer("consumer-b")
        .build()
        .expect("worker B engine should build");

    let stats_b = engine_b
        .run_once()
        .await
        .expect("worker B run should succeed");
    assert_eq!(
        stats_b.polled, 0,
        "worker B must not reclaim a message while worker A heartbeat is active"
    );
    assert_eq!(
        reclaimed_by_b.load(Ordering::SeqCst),
        0,
        "worker B handler must not run for worker A's active message"
    );

    let stats_a = worker_a
        .await
        .expect("worker A task should join")
        .expect("worker A run should succeed");
    assert_eq!(stats_a.succeeded, 1);
    assert_eq!(completed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn redis_transport_renews_all_polled_message_leases() {
    let (_server, qm) = spawn_test_queue_manager("redis-publish-batch-lease").await;
    let config = publish_lease_config_with_max_dequeue_items(2);

    let first_started = Arc::new(Barrier::new(2));
    let first_completed = Arc::new(AtomicUsize::new(0));
    let total_completed = Arc::new(AtomicUsize::new(0));
    let engine_a = EngineBuilder::new(&config, "publish")
        .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
        .role(Box::new(SlowFirstRole {
            first_started: Arc::clone(&first_started),
            first_completed: Arc::clone(&first_completed),
            total_completed: Arc::clone(&total_completed),
            sleep_for: Duration::from_millis(220),
        }))
        .consumer("consumer-a")
        .build()
        .expect("worker A engine should build");
    engine_a
        .ensure_consumer_groups()
        .await
        .expect("consumer group should exist");

    qm.enqueue_to_stream(
        &StreamKey::new("test:publish"),
        "run-first",
        r#"{"ok":true}"#,
        "publish",
    )
    .await
    .expect("first publish message should enqueue");
    qm.enqueue_to_stream(
        &StreamKey::new("test:publish"),
        "run-second",
        r#"{"ok":true}"#,
        "publish",
    )
    .await
    .expect("second publish message should enqueue");

    let worker_a = tokio::spawn(async move { engine_a.run_once().await });
    first_started.wait().await;
    tokio::time::sleep(Duration::from_millis(110)).await;

    let reclaimed_by_b = Arc::new(AtomicUsize::new(0));
    let engine_b = EngineBuilder::new(&config, "publish")
        .transport(Arc::new(RedisTransport::new(qm.clone(), "test:")))
        .role(Box::new(CountingRole {
            handled: Arc::clone(&reclaimed_by_b),
        }))
        .consumer("consumer-b")
        .build()
        .expect("worker B engine should build");

    let stats_b = engine_b
        .run_once()
        .await
        .expect("worker B run should succeed");
    assert_eq!(
        stats_b.polled, 0,
        "worker B must not reclaim later messages already polled by worker A"
    );
    assert_eq!(
        reclaimed_by_b.load(Ordering::SeqCst),
        0,
        "worker B handler must not run for worker A's waiting batch message"
    );

    let stats_a = worker_a
        .await
        .expect("worker A task should join")
        .expect("worker A run should succeed");
    assert_eq!(stats_a.succeeded, 2);
    assert_eq!(first_completed.load(Ordering::SeqCst), 1);
    assert_eq!(total_completed.load(Ordering::SeqCst), 2);
}
