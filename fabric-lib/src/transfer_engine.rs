use std::{
    ffi::c_void,
    mem::MaybeUninit,
    num::NonZeroU8,
    num::NonZeroU32,
    ptr::NonNull,
    sync::Arc,
    sync::atomic::{AtomicI64, AtomicU64, Ordering::SeqCst},
    thread::JoinHandle,
};

use dashmap::DashMap;
use parking_lot::Mutex;
use parking_lot::RwLock;
use thread_lib::pin_cpu;
use tracing::{error, warn};

use crate::{
    BouncingErrorCallback, BouncingRecvCallback, ErrorCallback, FabricLibError,
    RdmaEngine, RecvCallback, SendBuffer, SendCallback, SendRecvEngine,
    api::{
        DomainAddress, GdrCounter, ImmCounter, MemoryRegionDescriptor,
        MemoryRegionHandle, PeerGroupHandle, REMOTE_CONFIRMED_OPERATION_BIT, SmallVec,
        TransferCompletionEntry, TransferCounter, TransferId, TransferRequest,
        UvmWatcherId,
    },
    cuda_compat::{Device, gdr::GdrFlag},
    error::Result,
    fabric_engine::FabricEngine,
    imm_count::ImmCount,
    worker::{IdlePoller, PollingMode, Worker},
};
#[cfg(feature = "tokio")]
use {crate::AsyncTransferEngine, tokio::sync::oneshot};

pub type CallbackResult = std::result::Result<(), String>;

pub struct TransferCallback {
    pub on_done: Box<dyn FnOnce() -> CallbackResult + Send + Sync>,
    pub on_error: ErrorCallback,
}

pub type TransferResultCallback =
    Box<dyn FnOnce(Result<()>) -> CallbackResult + Send + Sync>;

enum TransferCallbackEntry {
    Split(TransferCallback),
    Result(TransferResultCallback),
}

struct RecvContext {
    worker_index: usize,
    mr: MemoryRegionHandle,
    ptr: NonNull<c_void>,
    len: usize,
    on_recv: RecvCallback,
    on_error: ErrorCallback,
}
unsafe impl Send for RecvContext {}
unsafe impl Sync for RecvContext {}

pub type ImmCallbackFn = Box<dyn Fn(u32) -> CallbackResult + Send + Sync>;
pub type UvmWatcherCallback =
    Box<dyn Fn(u64, u64) -> std::result::Result<bool, String> + Send + Sync>;

pub type ImmCountCallback =
    Box<dyn Fn() -> std::result::Result<bool, String> + Send + Sync>;

enum ImmCountFn {
    /// The callback will be called once when the expected count is reached.
    #[allow(dead_code)]
    Once(Box<dyn FnOnce() + Send + Sync>),
    /// The callback will be called every time the expected count is reached.
    Repeated(Box<dyn Fn() -> std::result::Result<bool, String> + Send + Sync>),
}
struct Callbacks {
    imm: RwLock<Vec<ImmCallbackFn>>,
    recv_ops: DashMap<TransferId, RecvContext>,
    send_ops: DashMap<TransferId, SendCallback>,
    transfer_ops: DashMap<TransferId, TransferCallbackEntry>,
    imm_count: DashMap<u32, ImmCountFn>,
    watchers: DashMap<UvmWatcherId, UvmWatcherCallback>,
}

pub struct TransferEngine {
    next_transfer_id: AtomicU64,
    engine: Arc<FabricEngine>,
    callbacks: Arc<Callbacks>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

enum CompletionCpuPolicy {
    Strided(Option<u16>),
    Exact(Vec<Option<u16>>),
}

impl TransferEngine {
    pub fn new(workers: Vec<(u8, Worker)>) -> Result<Self> {
        Self::new_with_completion_cpu(workers, None)
    }

    pub fn new_with_completion_cpu(
        workers: Vec<(u8, Worker)>,
        completion_cpu: Option<u16>,
    ) -> Result<Self> {
        Self::new_inner(workers, CompletionCpuPolicy::Strided(completion_cpu), false)
    }

    /// Builds one host-memory engine from independent worker/context replicas.
    ///
    /// Host MRs are registered with every worker and data transfers are striped per operation;
    /// control SEND/RECV operations remain on the first worker.
    pub fn new_host_striped_with_completion_cpu(
        workers: Vec<(u8, Worker)>,
        completion_cpu: Option<u16>,
    ) -> Result<Self> {
        Self::new_inner(workers, CompletionCpuPolicy::Strided(completion_cpu), true)
    }

    /// Builds a striped host engine whose completion pollers use exact CPU assignments.
    ///
    /// Unlike [`Self::new_host_striped_with_completion_cpu`], this does not assume that CPU IDs
    /// are contiguous. `completion_cpus` must contain one entry for every worker/context.
    pub fn new_host_striped_with_completion_cpus(
        workers: Vec<(u8, Worker)>,
        completion_cpus: Vec<Option<u16>>,
    ) -> Result<Self> {
        if completion_cpus.len() != workers.len() {
            return Err(FabricLibError::Custom(
                "completion CPU count must match worker count",
            ));
        }
        Self::new_inner(workers, CompletionCpuPolicy::Exact(completion_cpus), true)
    }

    fn new_inner(
        workers: Vec<(u8, Worker)>,
        completion_cpu_policy: CompletionCpuPolicy,
        host_striped: bool,
    ) -> Result<Self> {
        let completion_polling =
            workers.iter().map(|(_, worker)| worker.polling_mode).collect::<Vec<_>>();
        let engine = Arc::new(if host_striped {
            FabricEngine::new_host_striped(workers)?
        } else {
            FabricEngine::new(workers)?
        });

        let callbacks = Arc::new(Callbacks {
            imm: RwLock::new(Vec::new()),
            recv_ops: DashMap::new(),
            send_ops: DashMap::new(),
            transfer_ops: DashMap::new(),
            imm_count: DashMap::new(),
            watchers: DashMap::new(),
        });

        let callback_workers = if host_striped { engine.worker_count() } else { 1 };
        let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(callback_workers);
        for worker_index in 0..callback_workers {
            let thread_engine = engine.clone();
            let thread_calbacks = callbacks.clone();
            let polling_mode = completion_polling
                .get(worker_index)
                .copied()
                .unwrap_or(PollingMode::Busy);
            let pin_offset =
                u16::try_from(worker_index.saturating_mul(3)).map_err(|_| {
                    FabricLibError::Custom("completion CPU offset overflows")
                })?;
            let thread_cpu = match &completion_cpu_policy {
                CompletionCpuPolicy::Strided(completion_cpu) => match completion_cpu {
                    Some(cpu) => Some(cpu.checked_add(pin_offset).ok_or(
                        FabricLibError::Custom("completion CPU index overflows"),
                    )?),
                    None => None,
                },
                CompletionCpuPolicy::Exact(completion_cpus) => {
                    completion_cpus[worker_index]
                }
            };
            let thread = std::thread::Builder::new()
                .name(format!("tx_engine_callback_{worker_index}"))
                .spawn(move || {
                    if let Some(cpu) = thread_cpu
                        && let Err(error) = pin_cpu(cpu as usize)
                    {
                        warn!(cpu, %error, "failed to pin transfer completion thread");
                    }
                    callback_worker_thread(
                        thread_engine,
                        thread_calbacks,
                        host_striped.then_some(worker_index),
                        polling_mode,
                    )
                });
            let thread = match thread {
                Ok(thread) => thread,
                Err(_) => {
                    engine.stop();
                    for thread in threads {
                        let _ = thread.join();
                    }
                    return Err(FabricLibError::Custom(
                        "failed to launch callback worker thread",
                    ));
                }
            };
            threads.push(thread);
        }

        Ok(TransferEngine {
            next_transfer_id: AtomicU64::new(0),
            engine,
            callbacks,
            threads: Mutex::new(threads),
        })
    }

    pub fn num_domains(&self) -> usize {
        self.engine.num_domains()
    }

    pub fn num_groups(&self) -> usize {
        self.engine.num_groups()
    }

    pub fn aggregated_link_speed(&self) -> u64 {
        self.engine.aggregated_link_speed()
    }

    pub fn control_addresses(&self) -> Vec<DomainAddress> {
        self.engine.main_addresses()
    }

    pub fn control_lane_count(&self) -> usize {
        self.engine.worker_count()
    }

    pub fn add_imm_callback(&self, callback: ImmCallbackFn) {
        self.callbacks.imm.write().push(callback);
    }

    pub fn set_imm_count_expected(
        &self,
        imm: u32,
        expected_count: NonZeroU32,
        callback: ImmCountCallback,
    ) -> Option<ImmCount> {
        self.callbacks.imm_count.insert(imm, ImmCountFn::Repeated(callback));
        self.engine.set_imm_count_expected(imm, expected_count)
    }

    pub fn remove_imm_count(&self, imm: u32) -> Option<ImmCount> {
        self.callbacks.imm_count.remove(&imm);
        self.engine.remove_imm_count(imm)
    }

    pub fn get_imm_counter(&self, imm: u32) -> ImmCounter {
        self.engine.get_imm_counter(imm)
    }

    pub fn get_gdr_counter(&self, imm: u32, flag: Arc<GdrFlag>) -> GdrCounter {
        self.engine.get_gdr_counter(imm, flag)
    }

    pub fn add_peer_group(
        &self,
        addrs: Vec<SmallVec<DomainAddress>>,
        device: Device,
    ) -> Result<PeerGroupHandle> {
        self.engine.add_peer_group(addrs, device)
    }

    pub fn alloc_scalar_watcher(
        &self,
        callback: UvmWatcherCallback,
    ) -> Result<UvmWatcherId> {
        let watcher_id = self.engine.acquire_uvm_watcher()?;
        self.callbacks.watchers.insert(watcher_id, callback);
        Ok(watcher_id)
    }

    pub fn submit_transfer(
        &self,
        request: TransferRequest,
        callback: TransferCallback,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.callbacks
            .transfer_ops
            .insert(transfer_id, TransferCallbackEntry::Split(callback));
        if let Err(error) = self.engine.submit_transfer(transfer_id, request, None) {
            self.callbacks.transfer_ops.remove(&transfer_id);
            return Err(error);
        }
        Ok(())
    }

    /// Submit a transfer with one result callback.
    ///
    /// This avoids the shared state needed to join separate success and error closures and is the
    /// preferred path for futures and owned-buffer lifetimes.
    pub fn submit_transfer_result(
        &self,
        request: TransferRequest,
        callback: TransferResultCallback,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.callbacks
            .transfer_ops
            .insert(transfer_id, TransferCallbackEntry::Result(callback));
        if let Err(error) = self.engine.submit_transfer(transfer_id, request, None) {
            self.callbacks.transfer_ops.remove(&transfer_id);
            return Err(error);
        }
        Ok(())
    }

    pub fn submit_transfer_atomic(
        &self,
        request: TransferRequest,
        tx_counter: Arc<AtomicI64>,
        err_counter: Arc<AtomicI64>,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.engine.submit_transfer(
            transfer_id,
            request,
            Some(TransferCounter::new(tx_counter, err_counter)),
        )
    }

    /// Submits a counter-completed transfer while retaining an owned resource until completion.
    pub fn submit_transfer_atomic_guarded(
        &self,
        request: TransferRequest,
        tx_counter: Arc<AtomicI64>,
        err_counter: Arc<AtomicI64>,
        guard: Arc<dyn Send + Sync>,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.engine.submit_transfer(
            transfer_id,
            request,
            Some(TransferCounter::with_guard(tx_counter, err_counter, guard)),
        )
    }

    pub fn stop(&self) {
        self.engine.stop();
        let threads = std::mem::take(&mut *self.threads.lock());
        for thread in threads {
            if let Err(error) = thread.join() {
                error!(?error, "Failed to join a Transfer Engine callback thread.");
            }
        }
    }

    fn assign_transfer_id(&self) -> TransferId {
        let transfer_id = self.next_transfer_id.fetch_add(1, SeqCst);
        TransferId(transfer_id)
    }

    /// Submits a SEND whose source remains valid until the remote application confirms receipt.
    ///
    /// No per-operation callback is allocated or inserted into the callback table. The caller must
    /// keep `buffer` alive until an application-level reply/ack proves that the peer consumed the
    /// message. Asynchronous posting errors are therefore observed as a missing reply and handled
    /// by the caller's timeout.
    pub fn submit_send_remote_confirmed(
        &self,
        addr: DomainAddress,
        buffer: SendBuffer,
    ) -> Result<()> {
        self.submit_send_remote_confirmed_on(0, addr, buffer)
    }

    pub fn submit_send_remote_confirmed_on(
        &self,
        worker_index: usize,
        addr: DomainAddress,
        buffer: SendBuffer,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        debug_assert_eq!(transfer_id.0 & REMOTE_CONFIRMED_OPERATION_BIT, 0);
        self.engine.submit_send_on(
            worker_index,
            TransferId(transfer_id.0 | REMOTE_CONFIRMED_OPERATION_BIT),
            addr,
            buffer.mr_handle,
            buffer.ptr,
            buffer.len,
            buffer.coalescible,
        )
    }

    /// Submits a transfer whose source lifetime is retained by a stronger remote response/ACK.
    ///
    /// The transport still tracks and retires the WR internally, but it does not publish a local
    /// completion into the callback table. This is the WRITE equivalent of
    /// `submit_send_remote_confirmed_on` and avoids per-operation callback allocation when the
    /// application protocol already proves remote consumption.
    pub fn submit_transfer_remote_confirmed(
        &self,
        request: TransferRequest,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        debug_assert_eq!(transfer_id.0 & REMOTE_CONFIRMED_OPERATION_BIT, 0);
        self.engine.submit_transfer(
            TransferId(transfer_id.0 | REMOTE_CONFIRMED_OPERATION_BIT),
            request,
            None,
        )
    }

    pub fn submit_send_on(
        &self,
        worker_index: usize,
        addr: DomainAddress,
        buffer: SendBuffer,
        callback: SendCallback,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.callbacks.send_ops.insert(transfer_id, callback);
        if let Err(error) = self.engine.submit_send_on(
            worker_index,
            transfer_id,
            addr,
            buffer.mr_handle,
            buffer.ptr,
            buffer.len,
            buffer.coalescible,
        ) {
            self.callbacks.send_ops.remove(&transfer_id);
            return Err(error);
        }
        Ok(())
    }

    fn submit_recv_on(
        &self,
        worker_index: usize,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        on_recv: RecvCallback,
        on_error: ErrorCallback,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.callbacks.recv_ops.insert(
            transfer_id,
            RecvContext { worker_index, mr, ptr, len, on_recv, on_error },
        );
        if let Err(error) =
            self.engine.submit_recv_on(worker_index, transfer_id, mr, ptr, len)
        {
            self.callbacks.recv_ops.remove(&transfer_id);
            return Err(error);
        }
        Ok(())
    }
}

impl RdmaEngine for TransferEngine {
    fn main_address(&self) -> DomainAddress {
        self.engine.main_address()
    }

    fn nets_per_gpu(&self) -> NonZeroU8 {
        self.engine.nets_per_gpu()
    }

    fn register_memory_local(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<MemoryRegionHandle> {
        self.engine.register_memory_local(ptr, len, device)
    }

    fn register_memory_allow_remote(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)> {
        self.engine.register_memory_allow_remote(ptr, len, device)
    }

    fn unregister_memory(&self, ptr: NonNull<c_void>) -> Result<()> {
        self.engine.unregister_memory(ptr)
    }
}

impl SendRecvEngine for TransferEngine {
    fn submit_send(
        &self,
        addr: DomainAddress,
        buffer: SendBuffer,
        callback: SendCallback,
    ) -> Result<()> {
        self.submit_send_on(0, addr, buffer, callback)
    }

    fn submit_recv(
        &self,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        on_recv: RecvCallback,
        on_error: ErrorCallback,
    ) -> Result<()> {
        self.submit_recv_on(0, mr, ptr, len, on_recv, on_error)
    }

    fn submit_bouncing_recvs(
        &self,
        len: usize,
        count: usize,
        on_recv: BouncingRecvCallback,
        on_error: BouncingErrorCallback,
    ) -> Result<()> {
        // Allocate buffers for the recv ops
        let lane_count = self.control_lane_count();
        let storage: Arc<Vec<MaybeUninit<u8>>> =
            Arc::new(Vec::with_capacity(len * count * lane_count));
        let buf_base =
            unsafe { NonNull::new_unchecked(storage.as_ref().as_ptr() as *mut c_void) };

        // Register memory regions
        let mr_handle = self.engine.register_memory_local(
            buf_base,
            len * count * lane_count,
            Device::Host,
        )?;

        // Submit RECV ops.
        for i in 0..count * lane_count {
            let worker_index = i / count;
            let on_recv_ref = on_recv.clone();
            let on_error_ref = on_error.clone();

            let storage_ref: Arc<Vec<MaybeUninit<u8>>> = storage.clone();

            let on_recv_wrapper = Box::new(move |data_len: usize| {
                let ptr = unsafe {
                    NonNull::new_unchecked(
                        storage_ref.as_ref().as_ptr().byte_add(i * len) as *mut c_void,
                    )
                };
                let data = unsafe {
                    std::slice::from_raw_parts(ptr.as_ptr() as *const u8, data_len)
                };

                on_recv_ref(data)?;
                Ok(())
            });

            let on_error_wrapper =
                { Box::new(move |e: FabricLibError| on_error_ref(e)) };

            self.submit_recv_on(
                worker_index,
                mr_handle,
                unsafe {
                    NonNull::new_unchecked(storage.as_ref().as_ptr() as *mut c_void)
                        .byte_add(i * len)
                },
                len,
                on_recv_wrapper,
                on_error_wrapper,
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "tokio")]
impl AsyncTransferEngine for TransferEngine {
    async fn wait_for_imm_count(
        &self,
        imm: u32,
        expected_count: NonZeroU32,
    ) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        let callback = Box::new(move || {
            tx.send(Ok(())).expect("Failed to send through oneshot channel");
        });

        tokio::task::block_in_place(move || {
            self.callbacks.imm_count.insert(imm, ImmCountFn::Once(callback));
            self.engine.set_imm_count_expected(imm, expected_count);
        });

        rx.await.map_err(|e| {
            FabricLibError::CompletionError(format!(
                "Failed to receive result from oneshot channel: {}",
                e
            ))
        })?
    }

    async fn submit_send_async(
        &self,
        addr: DomainAddress,
        buffer: SendBuffer,
    ) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        let callback = Box::new(move |result: Result<()>| {
            if tx.send(result).is_err() {
                Err("Failed to send result through oneshot channel".to_string())
            } else {
                Ok(())
            }
        });

        tokio::task::block_in_place(move || self.submit_send(addr, buffer, callback))?;

        rx.await.map_err(|e| {
            FabricLibError::CompletionError(format!(
                "Failed to receive result from oneshot channel: {}",
                e
            ))
        })?
    }

    async fn submit_transfer_async(&self, request: TransferRequest) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        let callback: TransferResultCallback = Box::new(move |result| {
            tx.send(result).map_err(|_| {
                "Failed to send result through oneshot channel".to_string()
            })
        });

        // Enqueuing a transfer is non-blocking at the normal in-flight depths. Entering Tokio's
        // blocking region for every WR forces an unnecessary scheduler hand-off and leaves gaps in
        // a saturated RDMA pipeline, especially when a caller immediately submits its next write
        // from the completion wake-up.
        self.submit_transfer_result(request, callback)?;

        rx.await.map_err(|_| {
            FabricLibError::CompletionError(
                "Failed to receive result from oneshot channel".to_string(),
            )
        })?
    }
}

fn callback_worker_thread(
    engine: Arc<FabricEngine>,
    states: Arc<Callbacks>,
    worker_index: Option<usize>,
    polling_mode: PollingMode,
) {
    let mut idle_poller = IdlePoller::new(polling_mode);
    while !engine.is_stopped() {
        let completion = match worker_index {
            Some(worker_index) => engine.poll_worker_completion(worker_index),
            None => engine.poll_transfer_completion(),
        };
        let Some(comp) = completion else {
            idle_poller.wait(false);
            continue;
        };
        idle_poller.wait(true);

        if let Err(e) = handle_transfer_completion(&engine, &states, comp) {
            error!(?e, "Transfer Engine callback thread error. Exiting.");
            engine.stop();
            break;
        }
    }
}

fn handle_transfer_completion(
    engine: &FabricEngine,
    states: &Callbacks,
    comp: TransferCompletionEntry,
) -> CallbackResult {
    match comp {
        TransferCompletionEntry::Transfer(transfer_id) => {
            if transfer_id.0 & REMOTE_CONFIRMED_OPERATION_BIT != 0 {
                return Ok(());
            }
            let Some((_, handler)) = states.transfer_ops.remove(&transfer_id) else {
                warn!(?transfer_id, "Transfer callback not found");
                return Ok(());
            };
            match handler {
                TransferCallbackEntry::Split(handler) => (handler.on_done)(),
                TransferCallbackEntry::Result(handler) => handler(Ok(())),
            }
        }
        TransferCompletionEntry::Recv { transfer_id, data_len } => {
            let Some(handler) = states.recv_ops.get(&transfer_id) else {
                warn!(?transfer_id, data_len, "Recv callback not found");
                return Ok(());
            };
            // Run the callback handler.
            (handler.on_recv)(data_len)?;
            // Re-register the operation.
            engine
                .submit_recv_on(
                    handler.worker_index,
                    transfer_id,
                    handler.mr,
                    handler.ptr,
                    handler.len,
                )
                .map_err(|e| format!("Failed to re-register recv operation: {}", e))
        }
        TransferCompletionEntry::Send(transfer_id) => {
            if transfer_id.0 & REMOTE_CONFIRMED_OPERATION_BIT != 0 {
                return Ok(());
            }
            let Some((_, handler)) = states.send_ops.remove(&transfer_id) else {
                warn!(?transfer_id, "Send callback not found");
                return Ok(());
            };
            (handler)(Ok(()))
        }
        TransferCompletionEntry::ImmData(imm_data) => {
            for callback in states.imm.read().iter() {
                callback(imm_data)?
            }
            Ok(())
        }
        TransferCompletionEntry::ImmCountReached(imm) => {
            let Some(handler) = states.imm_count.get(&imm) else {
                warn!(imm, "Imm count context not found");
                return Ok(());
            };
            match &*handler {
                ImmCountFn::Once(_) => {
                    // Remove the counter from the engine.
                    drop(handler);
                    let (_, once_handler) = states.imm_count.remove(&imm).unwrap();
                    let ImmCountFn::Once(handler_fn) = once_handler else {
                        unreachable!("Expected ImmCountFn::Once");
                    };
                    handler_fn();
                    Ok(())
                }
                ImmCountFn::Repeated(callback) => {
                    let res = callback();
                    drop(handler);
                    match res {
                        Ok(true) => {
                            // The counter has already been reset as soon as it reached the expected value.
                            // The callback will be called again when reaching the expected value again.
                            Ok(())
                        }
                        Ok(false) | Err(_) => {
                            // Remove ImmCount callback
                            states.imm_count.remove(&imm);

                            // Remove ImmCount from the engine.
                            // Call ImmData callback if there are overflow counts.
                            if let Some(imm_count) = engine.remove_imm_count(imm) {
                                let (count, _expected) = imm_count.consume();
                                if count > 0 {
                                    for _ in 0..count {
                                        for callback in states.imm.read().iter() {
                                            callback(imm)?
                                        }
                                    }
                                }
                            }
                            Ok(())
                        }
                    }
                }
            }
        }
        TransferCompletionEntry::UvmWatch { id, old, new } => {
            let Some(callback) = states.watchers.get(&id) else {
                warn!(?id, "UvmWatcher not found");
                return Ok(());
            };
            match callback(old, new) {
                Ok(true) => Ok(()),
                Ok(false) | Err(_) => {
                    // Stop the watcher if the callback returns false or an error occurs
                    if let Err(e) = engine.release_uvm_watcher(id) {
                        error!("Failed to release UvmWatcher: {}", e);
                    };
                    states.watchers.remove(&id);
                    Ok(())
                }
            }
        }
        TransferCompletionEntry::Error(transfer_id, fabric_lib_error) => {
            if transfer_id.0 & REMOTE_CONFIRMED_OPERATION_BIT != 0 {
                warn!(
                    ?transfer_id,
                    ?fabric_lib_error,
                    "remote-confirmed operation failed"
                );
                return Ok(());
            }
            let callback_result = {
                if let Some((_, op)) = states.transfer_ops.remove(&transfer_id) {
                    match op {
                        TransferCallbackEntry::Split(op) => {
                            (op.on_error)(fabric_lib_error)
                        }
                        TransferCallbackEntry::Result(op) => op(Err(fabric_lib_error)),
                    }
                } else if let Some((_, op)) = states.send_ops.remove(&transfer_id) {
                    op(Err(fabric_lib_error))
                } else if let Some((_, op)) = states.recv_ops.remove(&transfer_id) {
                    (op.on_error)(fabric_lib_error)
                } else {
                    error!(?transfer_id, ?fabric_lib_error, "Unhandled transfer error");
                    return Ok(());
                }
            };
            if let Err(e) = callback_result {
                error!("Failed to call error callback: {}", e);
            };
            Ok(())
        }
    }
}
