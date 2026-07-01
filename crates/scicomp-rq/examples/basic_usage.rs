/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use scicomp_rq::{LogicalStreamName, QueueManager, StreamKey};

#[tokio::main]
async fn main() -> scicomp_rq::Result<()> {
    let qm = QueueManager::from_redis_url("redis://127.0.0.1:6379").await?;

    let stream_key = StreamKey::new("prefetch");
    let _created = qm
        .create_consumer_group(&stream_key, "prefetch:grp", "$", true)
        .await?;

    let stream_name = LogicalStreamName::new("prefetch");
    let msg_id = qm
        .enqueue(
            &stream_name,
            "run-example-001",
            r#"{"model":"pangu"}"#,
            "prefetch",
        )
        .await?;

    let messages = qm
        .read_messages(&stream_key, "prefetch:grp", "example-consumer", 1, 0)
        .await?;

    if let Some(message) = messages.first() {
        let _acked = qm.ack_message(message).await?;
        println!("acknowledged message {}", message.id());
    }

    println!("enqueued message {}", msg_id);
    Ok(())
}
