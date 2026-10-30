use anyhow::{Result, bail};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};

#[derive(Clone, Default)]
pub struct Cancellation {
    requested: Arc<AtomicBool>,
    retired: Option<Arc<AtomicBool>>,
}
impl Cancellation {
    pub fn cancel(&self) {
        self.requested.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.requested.load(Ordering::Acquire)
            || self
                .retired
                .as_ref()
                .is_some_and(|retired| retired.load(Ordering::Acquire))
    }
}

/// Implementations are created, used and destroyed on the native OS thread.
/// They need not be Send: SDK executors and GPU buffers never leave that thread.
pub trait NativeHost: 'static {
    fn run(&mut self, request_json: &str, cancellation: &Cancellation) -> Result<String>;
    fn self_test(&mut self, _cancellation: &Cancellation) -> Result<String> {
        bail!("native host does not implement startup self-test")
    }
}

struct Command {
    request: String,
    cancel: Cancellation,
    permit: OwnedSemaphorePermit,
    reply: oneshot::Sender<Completion>,
}

/// Retains the execution gate until the caller has committed the outcome.
pub struct Completion {
    pub result: Result<String>,
    _permit: OwnedSemaphorePermit,
    retired: Arc<AtomicBool>,
    committed: bool,
}
impl Completion {
    pub fn committed(mut self) {
        self.committed = true;
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if !self.committed {
            self.retired.store(true, Ordering::Release);
        }
    }
}

#[derive(Clone)]
pub struct HostClient {
    sender: mpsc::Sender<Option<Command>>,
    gate: Arc<Semaphore>,
    retired: Arc<AtomicBool>,
    exited: watch::Receiver<bool>,
}

pub struct Execution {
    reply: oneshot::Receiver<Completion>,
    cancel: Cancellation,
    retired: Arc<AtomicBool>,
    finished: bool,
}

impl Drop for Execution {
    fn drop(&mut self) {
        if !self.finished {
            self.cancel.cancel();
            self.retired.store(true, Ordering::Release);
        }
    }
}

impl Execution {
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
    pub async fn wait(&mut self) -> Result<Completion> {
        let completion = (&mut self.reply).await?;
        self.finished = true;
        Ok(completion)
    }
}

impl HostClient {
    pub async fn spawn_with_self_test<F, H>(
        factory: F,
        timeout: std::time::Duration,
        mut shutdown: watch::Receiver<bool>,
    ) -> Result<(Self, String)>
    where
        F: FnOnce() -> Result<H> + Send + 'static,
        H: NativeHost,
    {
        let (client, ready) = Self::start(factory, true)?;
        struct StartupGuard {
            retired: Arc<AtomicBool>,
            completed: bool,
        }
        impl Drop for StartupGuard {
            fn drop(&mut self) {
                if !self.completed {
                    self.retired.store(true, Ordering::Release);
                }
            }
        }
        let mut guard = StartupGuard {
            retired: client.retired.clone(),
            completed: false,
        };
        let stop = async {
            loop {
                if *shutdown.borrow() {
                    return;
                }
                if shutdown.changed().await.is_err() {
                    return;
                }
            }
        };
        let initialized = tokio::select! {
            biased;
            _ = stop => Err(anyhow::anyhow!("native startup cancelled")),
            result = tokio::time::timeout(timeout, ready) => match result {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err(anyhow::anyhow!("native thread exited during startup")),
                Err(_) => Err(anyhow::anyhow!("native startup deadline exceeded")),
            },
        };
        match initialized {
            Ok(Some(report)) => {
                guard.completed = true;
                Ok((client, report))
            }
            Ok(None) => {
                client.retire();
                bail!("missing native startup self-test report")
            }
            Err(error) => {
                client.retire();
                Err(error)
            }
        }
    }
    pub async fn spawn<F, H>(factory: F) -> Result<Self>
    where
        F: FnOnce() -> Result<H> + Send + 'static,
        H: NativeHost,
    {
        let (client, ready) = Self::start(factory, false)?;
        ready.await??;
        Ok(client)
    }

    fn start<F, H>(
        factory: F,
        self_test: bool,
    ) -> Result<(Self, oneshot::Receiver<Result<Option<String>>>)>
    where
        F: FnOnce() -> Result<H> + Send + 'static,
        H: NativeHost,
    {
        let (sender, mut receiver) = mpsc::channel::<Option<Command>>(1);
        let retired = Arc::new(AtomicBool::new(false));
        let retired_thread = retired.clone();
        let (ready_tx, ready_rx) = oneshot::channel();
        let (exit_tx, exited) = watch::channel(false);
        std::thread::Builder::new()
            .name("native-workflow".into())
            // CUDA driver initialization overflowed Rust's default 2 MiB stack
            // on the qualified H100 stack. Keep its tested size independent of
            // RUST_MIN_STACK and the Tokio runtime's thread configuration.
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                struct Exit {
                    exited: watch::Sender<bool>,
                    retired: Arc<AtomicBool>,
                }
                impl Drop for Exit {
                    fn drop(&mut self) {
                        self.retired.store(true, Ordering::Release);
                        self.exited.send_replace(true);
                    }
                }
                let _exit = Exit {
                    exited: exit_tx,
                    retired: retired_thread.clone(),
                };
                let mut host = match factory() {
                    Ok(host) => host,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                let startup = if self_test {
                    host.self_test(&Cancellation {
                        requested: Arc::default(),
                        retired: Some(retired_thread.clone()),
                    })
                    .map(Some)
                } else {
                    Ok(None)
                };
                let failed = startup.is_err();
                if ready_tx.send(startup).is_err()
                    || failed
                    || retired_thread.load(Ordering::Acquire)
                {
                    return;
                }
                while let Some(Some(command)) = receiver.blocking_recv() {
                    let result = if retired_thread.load(Ordering::Acquire) {
                        Err(anyhow::anyhow!("native host retired before execution"))
                    } else {
                        host.run(&command.request, &command.cancel)
                    };
                    let _ = command.reply.send(Completion {
                        result,
                        _permit: command.permit,
                        retired: retired_thread.clone(),
                        committed: false,
                    });
                    if retired_thread.load(Ordering::Acquire) {
                        break;
                    }
                }
                // The host and all of its allocations are destroyed on this thread.
                drop(host);
            })?;
        Ok((
            Self {
                sender,
                retired,
                gate: Arc::new(Semaphore::new(1)),
                exited,
            },
            ready_rx,
        ))
    }

    pub fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }
    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }
    pub async fn drain(&self, timeout: std::time::Duration) -> bool {
        tokio::time::timeout(timeout, self.gate.clone().acquire_owned())
            .await
            .is_ok()
    }
    pub async fn shutdown(&self, timeout: std::time::Duration) -> bool {
        self.retire();
        tokio::time::timeout(timeout, async {
            let _ = self.sender.send(None).await;
            let mut exited = self.exited.clone();
            while !*exited.borrow() {
                if exited.changed().await.is_err() {
                    return false;
                }
            }
            true
        })
        .await
        .unwrap_or(false)
    }

    pub async fn submit(&self, request: String) -> Result<Execution> {
        let permit = self.gate.clone().acquire_owned().await?;
        if self.is_retired() {
            bail!("native host is retired");
        }
        let cancel = Cancellation {
            requested: Arc::new(AtomicBool::new(false)),
            retired: Some(self.retired.clone()),
        };
        let (reply, response) = oneshot::channel();
        self.sender
            .send(Some(Command {
                request,
                cancel: cancel.clone(),
                permit,
                reply,
            }))
            .await?;
        Ok(Execution {
            reply: response,
            cancel,
            retired: self.retired.clone(),
            finished: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, atomic::AtomicUsize};
    use std::time::Duration;

    struct StartupFailure;
    impl NativeHost for StartupFailure {
        fn run(&mut self, _: &str, _: &Cancellation) -> Result<String> {
            Ok("{}".into())
        }
        fn self_test(&mut self, _: &Cancellation) -> Result<String> {
            bail!("model produced nonfinite output")
        }
    }
    #[tokio::test]
    async fn failed_model_self_test_prevents_admission() {
        let (_tx, rx) = watch::channel(false);
        let result =
            HostClient::spawn_with_self_test(|| Ok(StartupFailure), Duration::from_secs(1), rx)
                .await;
        assert!(
            result.is_err(),
            "failed model self-test advertised a usable host"
        );
        assert!(result.err().unwrap().to_string().contains("nonfinite"));
    }
    struct StartupBlock {
        cancelled: Arc<AtomicBool>,
    }
    struct StartupSuccess {
        events: Arc<Mutex<Vec<std::thread::ThreadId>>>,
    }
    impl NativeHost for StartupSuccess {
        fn self_test(&mut self, _: &Cancellation) -> Result<String> {
            self.events
                .lock()
                .unwrap()
                .push(std::thread::current().id());
            Ok("{\"status\":\"passed\"}".into())
        }
        fn run(&mut self, _: &str, _: &Cancellation) -> Result<String> {
            self.events
                .lock()
                .unwrap()
                .push(std::thread::current().id());
            Ok("{}".into())
        }
    }
    impl Drop for StartupSuccess {
        fn drop(&mut self) {
            self.events
                .lock()
                .unwrap()
                .push(std::thread::current().id());
        }
    }
    #[tokio::test]
    async fn initialization_self_test_requests_and_destruction_share_one_native_thread() {
        let events = Arc::new(Mutex::new(vec![]));
        let thread_events = events.clone();
        let (_shutdown, rx) = watch::channel(false);
        let (client, report) = HostClient::spawn_with_self_test(
            move || {
                thread_events
                    .lock()
                    .unwrap()
                    .push(std::thread::current().id());
                Ok(StartupSuccess {
                    events: thread_events,
                })
            },
            Duration::from_secs(1),
            rx,
        )
        .await
        .unwrap();
        assert!(report.contains("passed"));
        let mut execution = client.submit("{}".into()).await.unwrap();
        execution.wait().await.unwrap().committed();
        assert!(client.shutdown(Duration::from_secs(1)).await);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4);
        assert!(events.iter().all(|id| id == &events[0]));
        assert_ne!(events[0], std::thread::current().id());
    }
    #[tokio::test]
    async fn startup_deadline_and_shutdown_cancel_the_model_self_test() {
        for shutdown in [false, true] {
            let cancelled = Arc::new(AtomicBool::new(false));
            let thread_cancelled = cancelled.clone();
            let (tx, rx) = watch::channel(false);
            let task = tokio::spawn(HostClient::spawn_with_self_test(
                move || {
                    Ok(StartupBlock {
                        cancelled: thread_cancelled,
                    })
                },
                Duration::from_millis(30),
                rx,
            ));
            if shutdown {
                tokio::time::sleep(Duration::from_millis(10)).await;
                tx.send(true).unwrap();
            }
            let error = task.await.unwrap().err().unwrap();
            assert!(
                error
                    .to_string()
                    .contains(if shutdown { "cancelled" } else { "deadline" })
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert!(cancelled.load(Ordering::Acquire));
        }
    }
    impl NativeHost for StartupBlock {
        fn run(&mut self, _: &str, _: &Cancellation) -> Result<String> {
            unreachable!()
        }
        fn self_test(&mut self, cancellation: &Cancellation) -> Result<String> {
            while !cancellation.is_cancelled() {
                std::thread::sleep(Duration::from_millis(1));
            }
            self.cancelled.store(true, Ordering::Release);
            bail!("startup cancelled")
        }
    }
    #[tokio::test]
    async fn abandoned_startup_cancels_native_self_test_without_admitting_work() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let thread_cancelled = cancelled.clone();
        let (_tx, rx) = watch::channel(false);
        let startup = HostClient::spawn_with_self_test(
            move || {
                Ok(StartupBlock {
                    cancelled: thread_cancelled,
                })
            },
            Duration::from_secs(30),
            rx,
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(30), startup)
                .await
                .is_err()
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            cancelled.load(Ordering::Acquire),
            "dropping startup left native self-test uncancelled"
        );
    }

    struct Fake {
        events: Arc<Mutex<Vec<std::thread::ThreadId>>>,
        calls: Arc<AtomicUsize>,
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            self.events
                .lock()
                .unwrap()
                .push(std::thread::current().id());
        }
    }
    impl NativeHost for Fake {
        fn run(&mut self, request: &str, cancel: &Cancellation) -> Result<String> {
            self.events
                .lock()
                .unwrap()
                .push(std::thread::current().id());
            self.calls.fetch_add(1, Ordering::SeqCst);
            if request == "block" {
                while !cancel.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                bail!("cancelled")
            }
            Ok(request.into())
        }
    }
    async fn setup() -> (
        HostClient,
        Arc<Mutex<Vec<std::thread::ThreadId>>>,
        Arc<AtomicUsize>,
    ) {
        let events = Arc::new(Mutex::new(vec![]));
        let calls = Arc::new(AtomicUsize::new(0));
        let (e, c) = (events.clone(), calls.clone());
        let client = HostClient::spawn(move || {
            e.lock().unwrap().push(std::thread::current().id());
            Ok(Fake {
                events: e,
                calls: c,
            })
        })
        .await
        .unwrap();
        (client, events, calls)
    }
    #[tokio::test]
    async fn gate_is_held_until_completion_committed_and_host_is_reused() {
        let (client, events, calls) = setup().await;
        let first = client
            .submit("first".into())
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(first.result.as_ref().unwrap(), "first");
        assert!(
            tokio::time::timeout(Duration::from_millis(30), client.submit("second".into()))
                .await
                .is_err()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        first.committed();
        let second = client
            .submit("second".into())
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(second.result.as_ref().unwrap(), "second");
        second.committed();
        drop(client);
        tokio::time::sleep(Duration::from_millis(20)).await;
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4); // construct, two runs, destroy
        assert!(events.iter().all(|id| *id == events[0]));
        assert_ne!(events[0], std::thread::current().id());
    }
    #[tokio::test]
    async fn dropped_handler_cancels_drains_and_retires_host() {
        let (client, events, calls) = setup().await;
        let execution = client.submit("block".into()).await.unwrap();
        while calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        drop(execution);
        assert!(client.is_retired());
        assert!(
            tokio::time::timeout(Duration::from_secs(1), client.submit("late".into()))
                .await
                .unwrap()
                .is_err()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(events.lock().unwrap().len(), 3);
    }
    #[tokio::test]
    async fn initialization_failure_prevents_admission() {
        let result = HostClient::spawn(|| -> Result<Fake> { bail!("incompatible model") }).await;
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("incompatible model")
        );
    }
    // Keep all frames live across recursion and touch every byte. This models
    // native startup's stack demand without loading CUDA in the CPU suite.
    #[cfg(unix)]
    #[inline(never)]
    fn touch_native_startup_stack(frames: usize) -> usize {
        let mut frame = [0_u8; 64 * 1024];
        for (index, byte) in frame.iter_mut().enumerate() {
            *byte = (index ^ frames) as u8;
        }
        std::hint::black_box(&mut frame);
        let nested = if frames > 1 {
            touch_native_startup_stack(frames - 1)
        } else {
            0
        };
        std::hint::black_box(&frame);
        nested + frame.len()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn native_factory_has_stack_space_for_cuda_initialization() {
        const CHILD: &str = "NATIVE_HOST_STACK_PROBE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let client = HostClient::spawn(|| {
                assert_eq!(touch_native_startup_stack(48), 3 * 1024 * 1024);
                Ok(Fake {
                    events: Arc::default(),
                    calls: Arc::default(),
                })
            })
            .await
            .unwrap();
            assert!(client.shutdown(Duration::from_secs(1)).await);
            return;
        }

        // Stack overflow aborts rather than unwinds. Isolate it so a regression
        // reports a failed assertion instead of terminating the whole suite.
        let output = std::process::Command::new("sh")
            .args(["-c", "ulimit -c 0; exec \"$@\"", "native-stack-probe"])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "host::tests::native_factory_has_stack_space_for_cuda_initialization",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("RUST_MIN_STACK", (2 * 1024 * 1024).to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "native startup stack probe failed ({}):\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    #[tokio::test]
    async fn an_uncommitted_completion_retires_the_host() {
        let (client, _, _) = setup().await;
        let completion = client
            .submit("first".into())
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
        drop(completion);
        assert!(
            client.is_retired(),
            "an abandoned result must not reopen admission"
        );
    }
    #[tokio::test]
    async fn non_preemptible_work_retains_gate_after_handler_drop() {
        struct Blocking {
            receiver: std::sync::mpsc::Receiver<()>,
            entered: Arc<AtomicBool>,
        }
        impl NativeHost for Blocking {
            fn run(&mut self, _: &str, _: &Cancellation) -> Result<String> {
                self.entered.store(true, Ordering::Release);
                self.receiver.recv()?;
                Ok("drained".into())
            }
        }
        let (release, blocked) = std::sync::mpsc::channel();
        let entered = Arc::new(AtomicBool::new(false));
        let start = entered.clone();
        let client = HostClient::spawn(move || {
            Ok(Blocking {
                receiver: blocked,
                entered: start,
            })
        })
        .await
        .unwrap();
        let execution = client.submit("first".into()).await.unwrap();
        while !entered.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
        drop(execution);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), client.submit("unsafe".into()))
                .await
                .is_err()
        );
        release.send(()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), client.submit("late".into()))
                .await
                .unwrap()
                .is_err()
        );
        assert!(client.is_retired());
    }
    #[tokio::test]
    async fn retirement_cancels_active_native_work() {
        let (client, _, calls) = setup().await;
        let mut active = client.submit("block".into()).await.unwrap();
        while calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        client.retire();
        let done = tokio::time::timeout(Duration::from_secs(1), active.wait()).await;
        assert!(
            done.is_ok(),
            "retirement must reach the native cancellation token"
        );
        assert!(done.unwrap().unwrap().result.is_err());
    }
    #[tokio::test]
    async fn shutdown_confirms_native_host_destruction_on_its_own_thread() {
        let (client, events, _) = setup().await;
        assert!(client.shutdown(Duration::from_secs(1)).await);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], events[1]);
        assert_ne!(events[0], std::thread::current().id());
    }
}
