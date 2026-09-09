use std::{
    collections::HashSet,
    ffi::c_void,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    time::{Duration, Instant},
};

use crossbeam_channel::TryRecvError;
use thread_lib::pin_cpu;
use tracing::{debug, warn};

#[cfg(feature = "efa")]
use crate::efa::EfaDomain;
use crate::{
    api::{
        DomainAddress, MemoryRegionDescriptor, MemoryRegionHandle, PeerGroupHandle,
        REMOTE_CONFIRMED_OPERATION_BIT, SmallVec, TransferCompletionEntry,
        TransferCounter, TransferId, TransferRequest, UvmWatcherId,
    },
    cuda_compat::CudaHostMemory,
    domain_group::DomainGroup,
    error::{FabricLibError, Result},
    imm_count::ImmCountMap,
    mr::MemoryRegion,
    provider::{DomainCompletionEntry, RdmaDomain, RdmaDomainInfo},
    provider_dispatch::DomainInfo,
    verbs::VerbsDomain,
};

#[allow(clippy::enum_variant_names, clippy::large_enum_variant)]
pub enum WorkerCommand {
    SubmitWriteBatches(
        Vec<(TransferId, crate::api::WriteBatchRequest, Arc<dyn Send + Sync>)>,
    ),
    SubmitTransfer {
        transfer_id: TransferId,
        request: TransferRequest,
        tx_counter: Option<TransferCounter>,
        admission: Option<Arc<dyn Send + Sync>>,
    },
    SubmitSend {
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        addr: DomainAddress,
        coalescible: bool,
    },
    SubmitRecv {
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    },
}

unsafe impl Send for WorkerCommand {}
unsafe impl Sync for WorkerCommand {}

pub enum WorkerCall {
    RegisterMRLocal {
        region: MemoryRegion,
        ret: oneshot::Sender<Result<MemoryRegionHandle>>,
    },
    RegisterMRAllowRemote {
        region: MemoryRegion,
        ret: oneshot::Sender<Result<(MemoryRegionHandle, MemoryRegionDescriptor)>>,
    },
    UnregisterMR {
        ptr: NonNull<c_void>,
        ret: oneshot::Sender<()>,
    },
    AddPeerGroup {
        addrs: Vec<SmallVec<DomainAddress>>,
        ret: oneshot::Sender<Result<PeerGroupHandle>>,
    },
}

unsafe impl Send for WorkerCall {}

pub enum UvmWatcherCall {
    AcquireUvmWatcher { ret: oneshot::Sender<Option<UvmWatcherId>> },
    ReleaseUvmWatcher { watcher: UvmWatcherId, ret: oneshot::Sender<()> },
}

pub struct Worker {
    pub domain_list: Vec<DomainInfo>,
    pub pin_worker_cpu: Option<u16>,
    pub pin_uvm_cpu: Option<u16>,
    pub polling_mode: PollingMode,
}

unsafe impl Send for Worker {}

/// Controls how a fabric progress thread behaves after it stops observing work.
///
/// Busy polling remains the default for latency-sensitive primary data paths. Adaptive polling is
/// intended for warm standby paths: it spins through short gaps, then sleeps until the next poll so
/// an idle engine does not consume an entire CPU indefinitely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PollingMode {
    #[default]
    Busy,
    Adaptive {
        spin_duration: Duration,
        idle_sleep: Duration,
    },
}

pub(crate) struct IdlePoller {
    mode: PollingMode,
    idle_since: Option<Instant>,
    idle_polls: u32,
    sleeping: bool,
}

impl IdlePoller {
    const CLOCK_CHECK_INTERVAL: u32 = 64;

    pub(crate) fn new(mode: PollingMode) -> Self {
        Self { mode, idle_since: None, idle_polls: 0, sleeping: false }
    }

    #[inline]
    pub(crate) fn wait(&mut self, active: bool) {
        if active {
            self.idle_since = None;
            self.idle_polls = 0;
            self.sleeping = false;
            return;
        }

        let PollingMode::Adaptive { spin_duration, idle_sleep } = self.mode else {
            std::hint::spin_loop();
            return;
        };

        if self.sleeping {
            Self::sleep(idle_sleep);
            return;
        }

        let idle_since = self.idle_since.get_or_insert_with(Instant::now);
        self.idle_polls = self.idle_polls.wrapping_add(1);
        if self.idle_polls < Self::CLOCK_CHECK_INTERVAL {
            std::hint::spin_loop();
            return;
        }
        self.idle_polls = 0;

        if idle_since.elapsed() >= spin_duration {
            self.sleeping = true;
            Self::sleep(idle_sleep);
        } else {
            std::hint::spin_loop();
        }
    }

    #[inline]
    fn sleep(duration: Duration) {
        if duration.is_zero() {
            std::thread::yield_now();
        } else {
            std::thread::sleep(duration);
        }
    }
}

pub struct InitializingWorker {
    worker_handle: std::thread::JoinHandle<()>,
    init_worker_rx: oneshot::Receiver<Result<InitializedWorker>>,
    uvm_handle: std::thread::JoinHandle<()>,
    init_uvm_rx: oneshot::Receiver<Result<InitializedUvmWatcher>>,
    cq_rx: crossbeam_channel::Receiver<TransferCompletionEntry>,
}

struct InitializedWorker {
    connection_admission: Arc<AtomicBool>,
    stop_signal: Arc<AtomicBool>,
    aggregated_link_speed: u64,
    address_list: Vec<DomainAddress>,
    call_tx: crossbeam_channel::Sender<WorkerCall>,
    cmd_tx: crossbeam_channel::Sender<WorkerCommand>,
    write_batches: bool,
}

struct InitializedUvmWatcher {
    stop_signal: Arc<AtomicBool>,
    call_tx: crossbeam_channel::Sender<UvmWatcherCall>,
}

pub struct WorkerHandle {
    pub(crate) connection_admission: Arc<AtomicBool>,
    pub write_batches: bool,
    pub aggregated_link_speed: u64,
    pub address_list: Vec<DomainAddress>,
    pub worker_call_tx: crossbeam_channel::Sender<WorkerCall>,
    pub uvm_call_tx: crossbeam_channel::Sender<UvmWatcherCall>,
    pub cmd_tx: crossbeam_channel::Sender<WorkerCommand>,
    pub cq_rx: crossbeam_channel::Receiver<TransferCompletionEntry>,
    worker_stop_signal: Arc<AtomicBool>,
    worker_handle: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
    uvm_stop_signal: Arc<AtomicBool>,
    uvm_handle: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl WorkerHandle {
    pub fn stop(&self) {
        self.worker_stop_signal.store(true, SeqCst);
        self.uvm_stop_signal.store(true, SeqCst);
    }

    pub fn join(self) {
        self.join_stopped();
    }

    pub(crate) fn join_stopped(&self) {
        // Keep the lock through join: a concurrent stop must not return while DMA is active.
        for slot in [&self.worker_handle, &self.uvm_handle] {
            let mut slot = slot.lock();
            if let Some(handle) = slot.take()
                && let Err(error) = handle.join()
            {
                warn!(?error, "fabric worker exited with a panic");
            }
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.stop();
        self.join_stopped();
    }
}

impl Worker {
    pub fn spawn(self, imm_count_map: Arc<ImmCountMap>) -> Result<InitializingWorker> {
        let (init_worker_tx, init_worker_rx) = oneshot::channel();
        let (init_uvm_tx, init_uvm_rx) = oneshot::channel();
        let polling_mode = self.polling_mode;

        // Dynamic dispatch to EFA or Verbs
        let total_domains = self.domain_list.len();
        #[cfg(feature = "efa")]
        let mut efa_domain_list = Vec::new();
        let mut verbs_domain_list = Vec::new();
        for info in self.domain_list.into_iter() {
            match info {
                #[cfg(feature = "efa")]
                DomainInfo::Efa(info) => efa_domain_list.push(info),
                DomainInfo::Verbs(info) => verbs_domain_list.push(info),
            }
        }

        // Completion callbacks may synchronously enqueue response or acknowledgement commands
        // back to this worker. A bounded completion queue can therefore deadlock with the bounded
        // command queue: the RDMA progress thread waits to publish a completion while the callback
        // thread waits for that same progress thread to consume a command. Keep command admission
        // bounded, but never block transport progress while publishing completions.
        let (cq_tx, cq_rx) = crossbeam_channel::unbounded();

        // Spawn thread
        let worker_thread_builder =
            std::thread::Builder::new().name("tx_engine_domain_worker".to_string());
        let worker_cq_tx = cq_tx.clone();
        #[cfg(feature = "efa")]
        let all_efa = efa_domain_list.len() == total_domains;
        #[cfg(not(feature = "efa"))]
        let all_efa = false;
        let worker_handle = if all_efa {
            #[cfg(feature = "efa")]
            {
                match efa_domain_list.len() {
                    1 => worker_thread_builder.spawn(move || {
                        rdma_worker_thread::<EfaDomain, 1>(
                            efa_domain_list,
                            self.pin_worker_cpu,
                            polling_mode,
                            imm_count_map,
                            init_worker_tx,
                            worker_cq_tx,
                        )
                    }),
                    2 => worker_thread_builder.spawn(move || {
                        rdma_worker_thread::<EfaDomain, 2>(
                            efa_domain_list,
                            self.pin_worker_cpu,
                            polling_mode,
                            imm_count_map,
                            init_worker_tx,
                            worker_cq_tx,
                        )
                    }),
                    4 => worker_thread_builder.spawn(move || {
                        rdma_worker_thread::<EfaDomain, 4>(
                            efa_domain_list,
                            self.pin_worker_cpu,
                            polling_mode,
                            imm_count_map,
                            init_worker_tx,
                            worker_cq_tx,
                        )
                    }),
                    _ => {
                        return Err(FabricLibError::Custom(
                            "Only support 1, 2, or 4 domains per GPU for EFA",
                        ));
                    }
                }
            }
            #[cfg(not(feature = "efa"))]
            unreachable!("EFA support is disabled")
        } else if verbs_domain_list.len() == total_domains {
            match verbs_domain_list.len() {
                1 => worker_thread_builder.spawn(move || {
                    rdma_worker_thread::<VerbsDomain, 1>(
                        verbs_domain_list,
                        self.pin_worker_cpu,
                        polling_mode,
                        imm_count_map,
                        init_worker_tx,
                        worker_cq_tx,
                    )
                }),
                2 => worker_thread_builder.spawn(move || {
                    rdma_worker_thread::<VerbsDomain, 2>(
                        verbs_domain_list,
                        self.pin_worker_cpu,
                        polling_mode,
                        imm_count_map,
                        init_worker_tx,
                        worker_cq_tx,
                    )
                }),
                _ => {
                    return Err(FabricLibError::Custom(
                        "Only support 1 or 2 domains per GPU for Verbs",
                    ));
                }
            }
        } else {
            return Err(FabricLibError::Custom("Cannot mix EFA and Verbs domains"));
        };
        let worker_handle = worker_handle
            .map_err(|_| FabricLibError::Custom("Failed to spawn worker thread"))?;

        let uvm_thread_builder =
            std::thread::Builder::new().name("tx_engine_uvm_worker".to_string());
        let uvm_handle = uvm_thread_builder
            .spawn(move || {
                uvm_worker_thread(self.pin_uvm_cpu, polling_mode, init_uvm_tx, cq_tx)
            })
            .map_err(|_| FabricLibError::Custom("Failed to spawn UVM worker thread"))?;

        Ok(InitializingWorker {
            worker_handle,
            init_worker_rx,
            uvm_handle,
            init_uvm_rx,
            cq_rx,
        })
    }
}

impl InitializingWorker {
    pub fn wait_init(self) -> Result<WorkerHandle> {
        let init_worker = self
            .init_worker_rx
            .recv()
            .map_err(|_| FabricLibError::Custom("Failed to receive worker init"))?;
        let init_uvm = self.init_uvm_rx.recv().map_err(|_| {
            FabricLibError::Custom("Failed to receive UVM watcher init")
        })?;

        let init_worker_args = match init_worker {
            Ok(init) => init,
            Err(e) => {
                self.worker_handle.join().expect("Failed to join worker thread");
                return Err(e);
            }
        };
        let init_uvm_args = match init_uvm {
            Ok(init) => init,
            Err(e) => {
                self.worker_handle.join().expect("Failed to join worker thread");
                return Err(e);
            }
        };

        Ok(WorkerHandle {
            connection_admission: init_worker_args.connection_admission,
            write_batches: init_worker_args.write_batches,
            worker_stop_signal: init_worker_args.stop_signal,
            uvm_stop_signal: init_uvm_args.stop_signal,
            aggregated_link_speed: init_worker_args.aggregated_link_speed,
            address_list: init_worker_args.address_list,
            worker_call_tx: init_worker_args.call_tx,
            uvm_call_tx: init_uvm_args.call_tx,
            cmd_tx: init_worker_args.cmd_tx,
            cq_rx: self.cq_rx,
            worker_handle: parking_lot::Mutex::new(Some(self.worker_handle)),
            uvm_handle: parking_lot::Mutex::new(Some(self.uvm_handle)),
        })
    }
}

struct UvmWatcherContext {
    uvm_memory: CudaHostMemory,
    last_values: Vec<u64>,
    free_slots: Vec<usize>,
    used_slots: HashSet<usize>,
}

impl UvmWatcherContext {
    fn new() -> Self {
        const UVM_WATCHER_BYTES: usize = 4096;
        let uvm_memory = CudaHostMemory::alloc(UVM_WATCHER_BYTES)
            .expect("Failed to allocate UVM memory");
        let num_slots = uvm_memory.size / size_of::<u64>();
        Self {
            uvm_memory,
            last_values: vec![0; num_slots],
            free_slots: (0..num_slots).rev().collect(),
            used_slots: HashSet::new(),
        }
    }

    fn slot_to_id(&self, slot: usize) -> UvmWatcherId {
        let v = self.uvm_memory.get_ref(slot);
        UvmWatcherId(unsafe { NonNull::new_unchecked(v as *const u64 as *mut c_void) })
    }

    fn acquire(&mut self) -> Option<UvmWatcherId> {
        if let Some(slot) = self.free_slots.pop() {
            self.used_slots.insert(slot);
            Some(self.slot_to_id(slot))
        } else {
            None
        }
    }

    fn release(&mut self, id: UvmWatcherId) {
        let ptr = id.as_non_null();
        if ptr < self.uvm_memory.ptr
            || ptr >= unsafe { self.uvm_memory.ptr.byte_add(self.uvm_memory.size) }
        {
            panic!(
                "UvmWatcherId out of bounds. ptr: {:?}, size: {}. Got: {:?}",
                self.uvm_memory.ptr, self.uvm_memory.size, ptr
            );
        }

        let slot = (ptr.as_ptr() as usize - self.uvm_memory.ptr.as_ptr() as usize)
            / size_of::<u64>();
        if self.used_slots.remove(&slot) {
            self.free_slots.push(slot);
            // Reset the value to 0
            *self.uvm_memory.get_mut(slot) = 0;
            self.last_values[slot] = 0;
        } else {
            warn!(?id, "Ignoring release of an unused UvmWatcher");
        }
    }
}

fn rdma_worker_thread<D: RdmaDomain, const N: usize>(
    domain_list: Vec<D::Info>,
    maybe_pin_cpu: Option<u16>,
    polling_mode: PollingMode,
    imm_count_map: Arc<ImmCountMap>,
    init_tx: oneshot::Sender<Result<InitializedWorker>>,
    cq_tx: crossbeam_channel::Sender<TransferCompletionEntry>,
) {
    // Pin CPU if specified
    if let Some(cpu) = maybe_pin_cpu {
        let names: Vec<_> = domain_list.iter().map(|info| info.name()).collect();
        debug!("Pin Domain Worker CPU {} for {:?}", cpu, names);
        if let Err(e) = pin_cpu(cpu as usize) {
            // Ignore send error
            let _ = init_tx.send(Err(FabricLibError::Errno(e)));
            return;
        }
    }

    // Create domains
    //
    // NOTE(lequn): We'd like to create the domain after pinning the CPU so that
    // the allocated resources are on the correct NUMA node.
    let connection_admission = Arc::new(AtomicBool::new(true));
    let mut domains = Vec::with_capacity(N);
    for info in domain_list.into_iter() {
        match D::open(info, imm_count_map.clone()) {
            Ok(mut domain) => {
                domain.set_connection_admission(connection_admission.clone());
                domains.push(domain);
            }
            Err(e) => {
                let _ = init_tx.send(Err(e)); // Ignore send error
                return;
            }
        }
    }
    let address_list = domains.iter().map(|d| d.addr().clone()).collect();

    // Create domain group
    let domains: [D; N] = domains.try_into().unwrap_or_else(|v: Vec<D>| {
        panic!(
            "The number of domains mismatch the const generic N: {} != {}",
            v.len(),
            N
        )
    });
    let mut group = DomainGroup::new(domains);

    // Create channels
    let (call_tx, call_rx) = crossbeam_channel::bounded(128);
    let (cmd_tx, cmd_rx) = crossbeam_channel::bounded(128);

    // Initialization complete
    let stop_signal = Arc::new(AtomicBool::new(false));
    let init = InitializedWorker {
        connection_admission,
        write_batches: group.supports_write_batch(),
        stop_signal: stop_signal.clone(),
        aggregated_link_speed: group.aggregate_link_speed(),
        address_list,
        call_tx,
        cmd_tx,
    };
    if init_tx.send(Ok(init)).is_err() {
        // Failed to send init message. Caller has discarded the thread.
        // Let's just exit.
        return;
    }

    // Main loop
    let mut idle_poller = IdlePoller::new(polling_mode);
    while !stop_signal.load(SeqCst) {
        let active =
            worker_step(&mut group, &call_rx, &cmd_rx, &cq_tx).unwrap_or_else(|_| {
                panic!("fabric-lib internal error: Worker step failed")
            });
        idle_poller.wait(active);
    }
}

fn worker_step<D: RdmaDomain, const N: usize>(
    group: &mut DomainGroup<D, N>,
    call_rx: &crossbeam_channel::Receiver<WorkerCall>,
    cmd_rx: &crossbeam_channel::Receiver<WorkerCommand>,
    cq_tx: &crossbeam_channel::Sender<TransferCompletionEntry>,
) -> std::result::Result<bool, ()> {
    let mut active = false;

    // Process function call
    match call_rx.try_recv() {
        Ok(call) => {
            active = true;
            match call {
                WorkerCall::RegisterMRLocal { region, ret } => {
                    let result = group.register_mr_local(&region);
                    ret.send(result).map_err(|_| ())?;
                }
                WorkerCall::RegisterMRAllowRemote { region, ret } => {
                    let result = group.register_mr_allow_remote(&region);
                    ret.send(result).map_err(|_| ())?;
                }
                WorkerCall::UnregisterMR { ptr, ret } => {
                    group.unregister_mr(ptr);
                    ret.send(()).map_err(|_| ())?;
                }
                WorkerCall::AddPeerGroup { addrs, ret } => {
                    let result = group.add_peer_group(addrs);
                    ret.send(result).map_err(|_| ())?;
                }
            }
        }
        Err(TryRecvError::Disconnected) => {
            // Channel disconnected, exit the thread
            return Err(());
        }
        Err(TryRecvError::Empty) => {
            // No function call, continue
        }
    }

    // Drain a bounded batch before polling the CQ. Concurrent callers commonly publish a whole
    // window together; processing only one command per poll inserts an empty CQ lookup between
    // every post and stretches both the initial fill and steady-state refill of the send queue.
    // Keep a bound so receive completions cannot be starved by a continuously full command queue.
    for _ in 0..64 {
        let cmd = match cmd_rx.try_recv() {
            Ok(cmd) => cmd,
            Err(TryRecvError::Disconnected) => return Err(()),
            Err(TryRecvError::Empty) => break,
        };
        active = true;
        match cmd {
            WorkerCommand::SubmitWriteBatches(batches) => {
                for (transfer_id, request, admission) in batches {
                    if let Err(error) = group.submit_transfer_request_guarded(
                        transfer_id,
                        TransferRequest::WriteBatch(request),
                        None,
                        Some(admission),
                    ) {
                        cq_tx
                            .send(TransferCompletionEntry::Error(transfer_id, error))
                            .map_err(|_| ())?;
                    }
                }
            }
            WorkerCommand::SubmitTransfer {
                transfer_id,
                request,
                tx_counter,
                admission,
            } => {
                let result = group.submit_transfer_request_guarded(
                    transfer_id,
                    request,
                    tx_counter,
                    admission,
                );
                if let Err(e) = result {
                    let comp = TransferCompletionEntry::Error(transfer_id, e);
                    cq_tx.send(comp).map_err(|_| ())?;
                }
            }
            WorkerCommand::SubmitSend {
                transfer_id,
                mr,
                ptr,
                len,
                addr,
                coalescible,
            } => {
                let result =
                    group.submit_send(transfer_id, mr, ptr, len, addr, coalescible);
                if let Err(e) = result {
                    let comp = TransferCompletionEntry::Error(transfer_id, e);
                    cq_tx.send(comp).map_err(|_| ())?;
                }
            }
            WorkerCommand::SubmitRecv { transfer_id, mr, ptr, len } => {
                let result = group.submit_recv(transfer_id, mr, ptr, len);
                if let Err(e) = result {
                    let comp = TransferCompletionEntry::Error(transfer_id, e);
                    cq_tx.send(comp).map_err(|_| ())?;
                }
            }
        }
    }

    // Make progress
    group.poll_progress();

    // Send completions
    while let Some(comp) = group.get_completion() {
        active = true;
        let tx_comp = match comp {
            DomainCompletionEntry::Recv { transfer_id, data_len } => {
                TransferCompletionEntry::Recv { transfer_id, data_len }
            }
            DomainCompletionEntry::Send(transfer_id)
                if transfer_id.0 & REMOTE_CONFIRMED_OPERATION_BIT != 0 =>
            {
                continue;
            }
            DomainCompletionEntry::Send(transfer_id) => {
                TransferCompletionEntry::Send(transfer_id)
            }
            DomainCompletionEntry::Transfer(transfer_id)
                if transfer_id.0 & REMOTE_CONFIRMED_OPERATION_BIT != 0 =>
            {
                continue;
            }
            DomainCompletionEntry::Transfer(transfer_id) => {
                TransferCompletionEntry::Transfer(transfer_id)
            }
            DomainCompletionEntry::ImmData(imm) => {
                TransferCompletionEntry::ImmData(imm)
            }
            DomainCompletionEntry::Immediate(event) => {
                TransferCompletionEntry::Immediate(event)
            }
            DomainCompletionEntry::ImmCountReached(imm) => {
                TransferCompletionEntry::ImmCountReached(imm)
            }
            DomainCompletionEntry::Error(transfer_id, err) => {
                TransferCompletionEntry::Error(transfer_id, err)
            }
        };
        cq_tx.send(tx_comp).map_err(|_| ())?;
    }
    Ok(active)
}

fn uvm_worker_thread(
    maybe_pin_cpu: Option<u16>,
    polling_mode: PollingMode,
    init_tx: oneshot::Sender<Result<InitializedUvmWatcher>>,
    cq_tx: crossbeam_channel::Sender<TransferCompletionEntry>,
) {
    // Pin CPU if specified
    if let Some(cpu) = maybe_pin_cpu {
        debug!("Pin UVM Worker CPU {}", cpu);
        if let Err(e) = pin_cpu(cpu as usize) {
            // Ignore send error
            let _ = init_tx.send(Err(FabricLibError::Errno(e)));
            return;
        }
    }

    let (call_tx, call_rx) = crossbeam_channel::bounded(128);

    // Lazy init for UvmWatcher
    let mut uvm_ctx: Option<UvmWatcherContext> = None;

    // Initialization complete
    let stop_signal = Arc::new(AtomicBool::new(false));
    let init = InitializedUvmWatcher { stop_signal: stop_signal.clone(), call_tx };
    if init_tx.send(Ok(init)).is_err() {
        // Failed to send init message. Caller has discarded the thread.
        // Let's just exit.
        return;
    }

    // Main loop
    let mut idle_poller = IdlePoller::new(polling_mode);
    while !stop_signal.load(SeqCst) {
        let active =
            uwm_worker_step(&mut uvm_ctx, &call_rx, &cq_tx).unwrap_or_else(|_| {
                panic!("fabric-lib internal error: Worker step failed")
            });
        idle_poller.wait(active);
    }
}

fn uwm_worker_step(
    uvm_ctx: &mut Option<UvmWatcherContext>,
    call_rx: &crossbeam_channel::Receiver<UvmWatcherCall>,
    cq_tx: &crossbeam_channel::Sender<TransferCompletionEntry>,
) -> std::result::Result<bool, ()> {
    let mut active = false;
    match call_rx.try_recv() {
        Ok(call) => {
            active = true;
            match call {
                UvmWatcherCall::AcquireUvmWatcher { ret } => {
                    let uvm_ctx = uvm_ctx.get_or_insert_with(UvmWatcherContext::new);
                    let result = uvm_ctx.acquire();
                    ret.send(result).map_err(|_| ())?;
                }
                UvmWatcherCall::ReleaseUvmWatcher { watcher, ret } => {
                    if let Some(uvm_ctx) = uvm_ctx {
                        uvm_ctx.release(watcher);
                    }
                    ret.send(()).map_err(|_| ())?;
                }
            }
        }
        Err(TryRecvError::Disconnected) => {
            // Channel disconnected, exit the thread
            return Err(());
        }
        Err(TryRecvError::Empty) => {
            // No function call, continue
        }
    }

    // UvmWatcher
    if let Some(uvm_ctx) = uvm_ctx {
        for slot in uvm_ctx.used_slots.iter() {
            let value = *uvm_ctx.uvm_memory.get_mut(*slot);
            let old = uvm_ctx.last_values[*slot];
            if value != old {
                active = true;
                uvm_ctx.last_values[*slot] = value;
                let id = uvm_ctx.slot_to_id(*slot);
                let comp = TransferCompletionEntry::UvmWatch { id, old, new: value };
                cq_tx.send(comp).map_err(|_| ())?;
            }
        }
    }

    Ok(active)
}

#[cfg(test)]
mod tests {
    use super::{IdlePoller, PollingMode};
    use std::time::Duration;

    #[test]
    fn adaptive_poller_parks_only_after_idle_and_rearms_on_activity() {
        let mut poller = IdlePoller::new(PollingMode::Adaptive {
            spin_duration: Duration::ZERO,
            idle_sleep: Duration::ZERO,
        });
        for _ in 0..IdlePoller::CLOCK_CHECK_INTERVAL {
            poller.wait(false);
        }
        assert!(poller.sleeping);

        poller.wait(true);
        assert!(!poller.sleeping);
        assert!(poller.idle_since.is_none());
        assert_eq!(poller.idle_polls, 0);
    }

    #[test]
    fn busy_poller_never_enters_the_sleeping_state() {
        let mut poller = IdlePoller::new(PollingMode::Busy);
        for _ in 0..IdlePoller::CLOCK_CHECK_INTERVAL * 2 {
            poller.wait(false);
        }
        assert!(!poller.sleeping);
    }
}
