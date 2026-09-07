use std::{
    collections::BTreeMap,
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{
            AtomicBool, AtomicUsize,
            Ordering::{Relaxed, SeqCst},
        },
    },
};

use dashmap::DashMap;

use crate::{
    api::{
        DomainAddress, GdrCounter, ImmCounter, MemoryRegionDescriptor,
        MemoryRegionHandle, PeerGroupHandle, SmallVec, TransferCompletionEntry,
        TransferCounter, TransferId, TransferRequest, UvmWatcherId,
    },
    cuda_compat::{CudaDeviceId, Device, gdr::GdrFlag},
    error::{FabricLibError, Result},
    imm_count::{ImmCount, ImmCountMap},
    mr::MemoryRegion,
    worker::{UvmWatcherCall, Worker, WorkerCall, WorkerCommand, WorkerHandle},
};

pub struct FabricEngine {
    workers: BTreeMap<u8, WorkerContext>,
    main_address: DomainAddress,
    num_groups: usize,
    num_domains: usize,
    aggregated_link_speed: u64,
    nets_per_gpu: NonZeroU8,
    stop_signal: AtomicBool,
    host_striped: bool,
    host_submit_rr: AtomicUsize,
    completion_rr: AtomicUsize,
    mr_device_map: DashMap<MemoryRegionHandle, Device>,
    imm_count_map: Arc<ImmCountMap>,
}

struct WorkerContext {
    worker: WorkerHandle,
    admission: parking_lot::Mutex<()>,
    write_budget: Vec<Arc<AtomicUsize>>,
}

// A conservative domain-wide SQ window (all peers and both QPs combined). Legacy RPC writes
// charge the same counters; their pre-existing blocking admission is unchanged. A newly
// reserved batch is queued before later legacy commands while holding the admission lock.
const DOMAIN_WRITE_BUDGET: usize = 1024;
struct WriteBudgetGuard(Vec<(Arc<AtomicUsize>, usize)>);
impl Drop for WriteBudgetGuard {
    fn drop(&mut self) {
        for (used, count) in &self.0 {
            used.fetch_sub(*count, SeqCst);
        }
    }
}

fn check_write_budget(used: usize, required: usize) -> Result<()> {
    if required > DOMAIN_WRITE_BUDGET {
        return Err(FabricLibError::Custom("batch exceeds domain SQ budget"));
    }
    if used.saturating_add(required) > DOMAIN_WRITE_BUDGET {
        return Err(FabricLibError::Full);
    }
    Ok(())
}

impl FabricEngine {
    pub fn new(workers: Vec<(u8, Worker)>) -> Result<Self> {
        Self::new_inner(workers, false)
    }

    pub fn new_host_striped(workers: Vec<(u8, Worker)>) -> Result<Self> {
        Self::new_inner(workers, true)
    }

    fn new_inner(workers: Vec<(u8, Worker)>, host_striped: bool) -> Result<Self> {
        if workers.is_empty() {
            return Err(FabricLibError::Custom(
                "FabricEngine requires at least one worker",
            ));
        }
        let imm_count_map = Arc::new(ImmCountMap::default());

        let spawned_workers = workers
            .into_iter()
            .map(|(device, w)| {
                let worker = w.spawn(imm_count_map.clone())?;
                Ok((device, worker))
            })
            .collect::<Result<Vec<_>>>()?;

        let initialized_workers = spawned_workers
            .into_iter()
            .map(|(device, w)| {
                let worker = w.wait_init()?;
                Ok((device, worker))
            })
            .collect::<Result<Vec<_>>>()?;

        let main_worker = &initialized_workers.first().unwrap().1;
        if host_striped
            && initialized_workers.iter().any(|(_, worker)| {
                worker.address_list.len() != main_worker.address_list.len()
            })
        {
            return Err(FabricLibError::Custom(
                "striped host workers must have the same domain topology",
            ));
        }
        let main_address = main_worker.address_list[0].clone();
        let nets_per_gpu =
            unsafe { NonZeroU8::new_unchecked(main_worker.address_list.len() as u8) };

        let mut contexts = BTreeMap::new();
        let mut num_groups = 0;
        let mut num_domains = 0;
        let mut aggregated_link_speed = 0;
        for (device, w) in initialized_workers.into_iter() {
            num_groups += 1;
            num_domains += w.address_list.len();
            aggregated_link_speed += w.aggregated_link_speed;
            let write_budget = (0..w.address_list.len())
                .map(|_| Arc::new(AtomicUsize::new(0)))
                .collect();
            contexts.insert(
                device,
                WorkerContext {
                    worker: w,
                    admission: parking_lot::Mutex::new(()),
                    write_budget,
                },
            );
        }

        Ok(Self {
            workers: contexts,
            main_address,
            num_groups,
            num_domains,
            aggregated_link_speed,
            nets_per_gpu,
            stop_signal: AtomicBool::new(false),
            host_striped,
            host_submit_rr: AtomicUsize::new(0),
            completion_rr: AtomicUsize::new(0),
            mr_device_map: DashMap::new(),
            imm_count_map,
        })
    }

    pub fn main_address(&self) -> DomainAddress {
        self.main_address.clone()
    }

    pub fn main_addresses(&self) -> Vec<DomainAddress> {
        self.workers
            .values()
            .map(|context| context.worker.address_list[0].clone())
            .collect()
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn num_domains(&self) -> usize {
        self.num_domains
    }

    pub fn aggregated_link_speed(&self) -> u64 {
        self.aggregated_link_speed
    }

    pub fn nets_per_gpu(&self) -> NonZeroU8 {
        self.nets_per_gpu
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn host_write_lane_count(&self) -> usize {
        (if self.host_striped {
            self.num_domains
        } else {
            self.nets_per_gpu.get() as usize
        }) * crate::api::WRITE_QP_LANES
    }

    pub fn supports_write_batches(&self) -> bool {
        self.workers.values().all(|worker| worker.worker.write_batches)
    }

    /// Reserves a command slot on every participating worker before publishing any group.
    /// Returned per-transfer errors occur after admission and must be completed, never replayed.
    pub fn try_submit_write_batches(
        &self,
        batches: Vec<(TransferId, crate::api::WriteBatchRequest)>,
    ) -> Result<Vec<(TransferId, FabricLibError)>> {
        if self.is_stopped() {
            return Err(FabricLibError::Custom("engine is stopped"));
        }
        let mut routed = BTreeMap::<usize, Vec<_>>::new();
        for (id, batch) in batches {
            let (index, batch) = self.prepare_write_batch(batch)?;
            routed.entry(index).or_default().push((id, batch));
        }
        let mut guards = Vec::with_capacity(routed.len());
        for &index in routed.keys() {
            let worker = self.get_worker_by_index(index)?;
            guards.push(worker.admission.try_lock().ok_or(FabricLibError::Full)?);
            if worker.worker.cmd_tx.is_full() {
                return Err(FabricLibError::Full);
            }
            let mut required = vec![0usize; worker.write_budget.len()];
            for (_, batch) in &routed[&index] {
                required[batch.lane / crate::api::WRITE_QP_LANES] += batch.writes.len();
            }
            for (used, count) in worker.write_budget.iter().zip(required) {
                check_write_budget(used.load(SeqCst), count)?;
            }
        }
        let mut failures = Vec::new();
        for (index, batches) in routed {
            let worker = self.get_worker_by_index(index)?;
            let batches = batches
                .into_iter()
                .map(|(id, batch)| {
                    let used = worker.write_budget
                        [batch.lane / crate::api::WRITE_QP_LANES]
                        .clone();
                    let count = batch.writes.len();
                    used.fetch_add(count, SeqCst);
                    let guard: Arc<dyn Send + Sync> =
                        Arc::new(WriteBudgetGuard(vec![(used, count)]));
                    (id, batch, guard)
                })
                .collect();
            // All writers take admission; the consumer can only create more space. Disconnection
            // is the only possible failure after reservation and is a terminal admitted result.
            if let Err(error) = worker
                .worker
                .cmd_tx
                .try_send(WorkerCommand::SubmitWriteBatches(batches))
            {
                let WorkerCommand::SubmitWriteBatches(batches) = error.into_inner()
                else {
                    unreachable!()
                };
                failures.extend(batches.into_iter().map(|(id, _, _)| {
                    (id, FabricLibError::Custom("worker stopped after batch admission"))
                }));
            }
        }
        Ok(failures)
    }

    fn prepare_write_batch(
        &self,
        mut batch: crate::api::WriteBatchRequest,
    ) -> Result<(usize, crate::api::WriteBatchRequest)> {
        if batch.writes.is_empty()
            || batch.writes.len() > crate::api::MAX_WRITE_BATCH_WR
            || batch.lane >= self.host_write_lane_count()
            || !self.supports_write_batches()
        {
            return Err(FabricLibError::Custom("invalid or unsupported write batch"));
        }
        let domains = self.nets_per_gpu.get() as usize;
        let (index, lane) = if self.host_striped {
            (
                batch.lane / (domains * crate::api::WRITE_QP_LANES),
                batch.lane % (domains * crate::api::WRITE_QP_LANES),
            )
        } else {
            (0, batch.lane)
        };
        let mut destination = None;
        for write in &mut batch.writes {
            if (write.segments.is_empty() && write.imm_data.is_none())
                || write.segments.len() > crate::api::MAX_GATHER_SEGMENTS
            {
                return Err(FabricLibError::Custom("invalid write batch SGE count"));
            }
            let mut length = 0u64;
            for source in &write.segments {
                if self.device_for_mr(source.src_mr)? != Device::Host
                    || source.length == 0
                {
                    return Err(FabricLibError::Custom(
                        "write batches require nonempty host-memory segments",
                    ));
                }
                length = length
                    .checked_add(source.length)
                    .filter(|&len| len <= u32::MAX as u64)
                    .ok_or(FabricLibError::Custom(
                        "write batch SGE length exceeds u32",
                    ))?;
                (source.src_mr.ptr.as_ptr() as u64)
                    .checked_add(source.src_offset)
                    .and_then(|start| start.checked_add(source.length))
                    .ok_or(FabricLibError::Custom("write batch source overflow"))?;
            }
            if self.host_striped {
                narrow_host_descriptor(
                    &mut write.dst_mr,
                    index,
                    self.workers.len(),
                    domains,
                )?;
            }
            if write.dst_mr.addr_rkey_list.len() != domains {
                return Err(FabricLibError::Custom(
                    "write batch domain topology mismatch",
                ));
            }
            let address =
                &write.dst_mr.addr_rkey_list[lane / crate::api::WRITE_QP_LANES].0;
            if destination.as_ref().is_some_and(|old| old != address) {
                return Err(FabricLibError::Custom(
                    "one write batch cannot target different peers",
                ));
            }
            destination = Some(address.clone());
            write
                .dst_mr
                .ptr
                .checked_add(write.dst_offset)
                .and_then(|start| start.checked_add(length))
                .ok_or(FabricLibError::Custom("write batch destination overflow"))?;
        }
        batch.lane = lane;
        Ok((index, batch))
    }

    pub fn register_memory_local(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<MemoryRegionHandle> {
        let handle = if self.host_striped && device == Device::Host {
            let mut handle = None;
            for worker in self.workers.values() {
                let registered = worker
                    .register_memory_local(MemoryRegion::new(ptr, len, device)?)?;
                if handle.is_some_and(|current| current != registered) {
                    return Err(FabricLibError::Custom(
                        "striped host workers returned inconsistent memory handles",
                    ));
                }
                handle = Some(registered);
            }
            handle.expect("a FabricEngine always has a worker")
        } else {
            self.get_worker(&device)?
                .register_memory_local(MemoryRegion::new(ptr, len, device)?)?
        };
        self.mr_device_map.insert(handle, device);
        Ok(handle)
    }

    pub fn register_memory_allow_remote(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)> {
        let (handle, desc) = if self.host_striped && device == Device::Host {
            let mut handle = None;
            let mut address_keys = SmallVec::new();
            for worker in self.workers.values() {
                let (registered, descriptor) = worker.register_memory_allow_remote(
                    MemoryRegion::new(ptr, len, device)?,
                )?;
                if handle.is_some_and(|current| current != registered) {
                    return Err(FabricLibError::Custom(
                        "striped host workers returned inconsistent memory handles",
                    ));
                }
                if descriptor.ptr != ptr.as_ptr() as u64
                    || descriptor.addr_rkey_list.len()
                        != worker.worker.address_list.len()
                {
                    return Err(FabricLibError::Custom(
                        "striped host worker returned an inconsistent domain descriptor",
                    ));
                }
                handle = Some(registered);
                address_keys.extend(descriptor.addr_rkey_list);
            }
            (
                handle.expect("a FabricEngine always has a worker"),
                MemoryRegionDescriptor {
                    ptr: ptr.as_ptr() as u64,
                    addr_rkey_list: address_keys,
                },
            )
        } else {
            self.get_worker(&device)?
                .register_memory_allow_remote(MemoryRegion::new(ptr, len, device)?)?
        };
        self.mr_device_map.insert(handle, device);
        Ok((handle, desc))
    }

    pub fn unregister_memory(&self, ptr: NonNull<c_void>) -> Result<()> {
        let handle = MemoryRegionHandle::new(ptr);
        let device = self
            .mr_device_map
            .get(&handle)
            .map(|entry| *entry)
            .ok_or(FabricLibError::Custom("Invalid memory region"))?;
        if self.host_striped && device == Device::Host {
            for worker in self.workers.values() {
                worker.unregister_memory(ptr)?;
            }
        } else {
            self.get_worker(&device)?.unregister_memory(ptr)?;
        }
        self.mr_device_map.remove(&handle);
        Ok(())
    }

    pub fn add_peer_group(
        &self,
        addrs: Vec<SmallVec<DomainAddress>>,
        device: Device,
    ) -> Result<PeerGroupHandle> {
        let worker = self.get_worker(&device)?;
        let (tx, rx) = oneshot::channel();
        let cmd = WorkerCall::AddPeerGroup { addrs, ret: tx };
        worker
            .worker
            .worker_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        let handle =
            rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))??;
        Ok(handle)
    }

    pub fn acquire_uvm_watcher(&self) -> Result<UvmWatcherId> {
        let worker = self.get_main_worker()?;
        let (tx, rx) = oneshot::channel();
        let cmd = UvmWatcherCall::AcquireUvmWatcher { ret: tx };
        worker
            .worker
            .uvm_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        let maybe = rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))?;
        maybe.ok_or(FabricLibError::Custom("Failed to acquire UVM watcher"))
    }

    pub fn release_uvm_watcher(&self, watcher: UvmWatcherId) -> Result<()> {
        let worker = self.get_main_worker()?;
        let (tx, rx) = oneshot::channel();
        let cmd = UvmWatcherCall::ReleaseUvmWatcher { watcher, ret: tx };
        worker
            .worker
            .uvm_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))?;
        Ok(())
    }

    pub fn set_imm_count_expected(
        &self,
        imm: u32,
        expected_count: NonZeroU32,
    ) -> Option<ImmCount> {
        self.imm_count_map.set_expected(imm, expected_count)
    }

    pub fn remove_imm_count(&self, imm: u32) -> Option<ImmCount> {
        self.imm_count_map.remove(imm)
    }

    pub fn get_imm_counter(&self, imm: u32) -> ImmCounter {
        self.imm_count_map.get_imm_counter(imm)
    }

    pub fn get_gdr_counter(&self, imm: u32, flag: Arc<GdrFlag>) -> GdrCounter {
        self.imm_count_map.get_gdr_counter(imm, flag)
    }

    pub fn submit_send(
        &self,
        transfer_id: TransferId,
        addr: DomainAddress,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        coalescible: bool,
    ) -> Result<()> {
        self.submit_send_on(0, transfer_id, addr, mr, ptr, len, coalescible)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit_send_on(
        &self,
        worker_index: usize,
        transfer_id: TransferId,
        addr: DomainAddress,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        coalescible: bool,
    ) -> Result<()> {
        let worker = self.get_worker_by_index(worker_index)?;
        let cmd =
            WorkerCommand::SubmitSend { transfer_id, addr, mr, ptr, len, coalescible };
        worker.send_command(cmd)?;
        Ok(())
    }

    pub fn submit_recv(
        &self,
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<()> {
        self.submit_recv_on(0, transfer_id, mr, ptr, len)
    }

    pub fn submit_recv_on(
        &self,
        worker_index: usize,
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<()> {
        let worker = self.get_worker_by_index(worker_index)?;
        worker.send_command(WorkerCommand::SubmitRecv { transfer_id, mr, ptr, len })?;
        Ok(())
    }

    pub fn submit_transfer(
        &self,
        transfer_id: TransferId,
        mut request: TransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        if let TransferRequest::WriteBatch(mut batch) = request {
            if batch.writes.is_empty()
                || batch.writes.len() > crate::api::MAX_WRITE_BATCH_WR
            {
                return Err(FabricLibError::Custom("invalid write batch length"));
            }
            for write in &batch.writes {
                for source in &write.segments {
                    if self.device_for_mr(source.src_mr)? != Device::Host {
                        return Err(FabricLibError::Custom(
                            "write batches currently require host memory",
                        ));
                    }
                }
            }
            let worker = if self.host_striped {
                let domains = self.nets_per_gpu.get() as usize;
                let (index, domain) = host_write_lane(
                    batch.lane / crate::api::WRITE_QP_LANES,
                    self.workers.len(),
                    domains,
                )?;
                let worker = self.workers.values().nth(index).ok_or(
                    FabricLibError::Custom("write batch lane is out of bounds"),
                )?;
                for write in &mut batch.writes {
                    narrow_host_descriptor(
                        &mut write.dst_mr,
                        index,
                        self.workers.len(),
                        domains,
                    )?;
                }
                batch.lane = domain * crate::api::WRITE_QP_LANES
                    + batch.lane % crate::api::WRITE_QP_LANES;
                worker
            } else {
                self.get_worker(&Device::Host)?
            };
            return worker.send_command(WorkerCommand::SubmitTransfer {
                transfer_id,
                request: TransferRequest::WriteBatch(batch),
                tx_counter,
                admission: None,
            });
        }
        let source_device = match &request {
            TransferRequest::WriteBatch(_) => {
                unreachable!("write batches use explicit lanes")
            }
            TransferRequest::Imm(_) | TransferRequest::Barrier(_) => None,
            TransferRequest::Single(req) => Some(self.device_for_mr(req.src_mr)?),
            TransferRequest::Gather(req) => {
                let first = req.segments.first().ok_or(FabricLibError::Custom(
                    "GatherTransferRequest must contain at least one segment",
                ))?;
                let device = self.device_for_mr(first.src_mr)?;
                for segment in req.segments.iter().skip(1) {
                    if self.device_for_mr(segment.src_mr)? != device {
                        return Err(FabricLibError::Custom(
                            "GatherTransferRequest segments must belong to one device",
                        ));
                    }
                }
                Some(device)
            }
            TransferRequest::Paged(req) => Some(self.device_for_mr(req.src_mr)?),
            TransferRequest::Scatter(req) => Some(self.device_for_mr(req.src_mr)?),
        };
        let worker = if self.host_striped
            && source_device.is_none_or(|device| device == Device::Host)
        {
            let index = if source_device.is_some() {
                self.host_submit_rr.fetch_add(1, Relaxed) % self.workers.len()
            } else {
                0
            };
            narrow_host_transfer_request(
                &mut request,
                index,
                self.workers.len(),
                self.nets_per_gpu.get() as usize,
            )?;
            self.workers
                .values()
                .nth(index)
                .expect("validated striped host worker index")
        } else {
            match source_device {
                Some(device) => self.get_worker(&device)?,
                None => self.get_main_worker()?,
            }
        };
        worker.send_command(WorkerCommand::SubmitTransfer {
            transfer_id,
            request,
            tx_counter,
            admission: None,
        })
    }

    pub fn poll_transfer_completion(&self) -> Option<TransferCompletionEntry> {
        let start = self.completion_rr.fetch_add(1, Relaxed) % self.workers.len();
        for ctx in self.workers.values().cycle().skip(start).take(self.workers.len()) {
            if let Ok(completion) = ctx.worker.cq_rx.try_recv() {
                return Some(completion);
            }
        }

        None
    }

    pub fn poll_worker_completion(
        &self,
        worker_index: usize,
    ) -> Option<TransferCompletionEntry> {
        self.workers.values().nth(worker_index)?.worker.cq_rx.try_recv().ok()
    }

    pub fn stop(&self) {
        self.stop_signal.store(true, SeqCst);
        for ctx in self.workers.values() {
            ctx.worker.stop();
        }
        // Joining destroys every domain/QP before callers may release DMA source owners.
        for ctx in self.workers.values() {
            ctx.worker.join_stopped();
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stop_signal.load(SeqCst)
    }

    fn device_for_mr(&self, mr: MemoryRegionHandle) -> Result<Device> {
        self.mr_device_map
            .get(&mr)
            .map(|device| *device)
            .ok_or(FabricLibError::Custom("Invalid memory region"))
    }

    fn get_worker(&self, device: &Device) -> Result<&WorkerContext> {
        match device {
            Device::Host => self.get_main_worker(),
            Device::Cuda(CudaDeviceId(device_id)) => self
                .workers
                .get(device_id)
                .ok_or(FabricLibError::Custom("Worker not found")),
        }
    }

    fn get_main_worker(&self) -> Result<&WorkerContext> {
        Ok(self.workers.first_key_value().unwrap().1)
    }

    fn get_worker_by_index(&self, worker_index: usize) -> Result<&WorkerContext> {
        self.workers
            .values()
            .nth(worker_index)
            .ok_or(FabricLibError::Custom("Worker index is out of range"))
    }
}

fn narrow_host_descriptor(
    descriptor: &mut MemoryRegionDescriptor,
    worker_index: usize,
    worker_count: usize,
    domains_per_worker: usize,
) -> Result<()> {
    if worker_index >= worker_count
        || domains_per_worker == 0
        || Some(descriptor.addr_rkey_list.len())
            != worker_count.checked_mul(domains_per_worker)
    {
        return Err(FabricLibError::Custom(
            "Remote memory descriptor does not match striped host worker/domain topology",
        ));
    }
    let start = worker_index * domains_per_worker;
    descriptor.addr_rkey_list = descriptor.addr_rkey_list
        [start..start + domains_per_worker]
        .iter()
        .cloned()
        .collect();
    Ok(())
}

fn host_write_lane(
    lane: usize,
    workers: usize,
    domains: usize,
) -> Result<(usize, usize)> {
    if domains == 0 || workers.checked_mul(domains).is_none_or(|count| lane >= count) {
        return Err(FabricLibError::Custom("write batch lane is out of bounds"));
    }
    Ok((lane / domains, lane % domains))
}

fn narrow_host_transfer_request(
    request: &mut TransferRequest,
    worker_index: usize,
    worker_count: usize,
    domains_per_worker: usize,
) -> Result<()> {
    match request {
        TransferRequest::WriteBatch(_) => {
            unreachable!("write batches use explicit lanes")
        }
        TransferRequest::Imm(request) => narrow_host_descriptor(
            &mut request.dst_mr,
            worker_index,
            worker_count,
            domains_per_worker,
        ),
        TransferRequest::Barrier(request) => {
            for descriptor in &mut request.dst_mrs {
                narrow_host_descriptor(
                    descriptor,
                    worker_index,
                    worker_count,
                    domains_per_worker,
                )?;
            }
            Ok(())
        }
        TransferRequest::Single(request) => narrow_host_descriptor(
            &mut request.dst_mr,
            worker_index,
            worker_count,
            domains_per_worker,
        ),
        TransferRequest::Gather(request) => narrow_host_descriptor(
            &mut request.dst_mr,
            worker_index,
            worker_count,
            domains_per_worker,
        ),
        TransferRequest::Paged(request) => narrow_host_descriptor(
            &mut request.dst_mr,
            worker_index,
            worker_count,
            domains_per_worker,
        ),
        TransferRequest::Scatter(request) => {
            for target in Arc::make_mut(&mut request.dsts) {
                narrow_host_descriptor(
                    &mut target.dst_mr,
                    worker_index,
                    worker_count,
                    domains_per_worker,
                )?;
            }
            Ok(())
        }
    }
}

impl Drop for FabricEngine {
    fn drop(&mut self) {
        self.stop();
        while let Some((_, ctx)) = self.workers.pop_first() {
            ctx.worker.stop();
        }
    }
}

impl WorkerContext {
    fn register_memory_local(
        &self,
        region: MemoryRegion,
    ) -> Result<MemoryRegionHandle> {
        let (tx, rx) = oneshot::channel();
        self.worker
            .worker_call_tx
            .send(WorkerCall::RegisterMRLocal { region, ret: tx })
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))?
    }

    fn register_memory_allow_remote(
        &self,
        region: MemoryRegion,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)> {
        let (tx, rx) = oneshot::channel();
        self.worker
            .worker_call_tx
            .send(WorkerCall::RegisterMRAllowRemote { region, ret: tx })
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))?
    }

    fn unregister_memory(&self, ptr: NonNull<c_void>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.worker
            .worker_call_tx
            .send(WorkerCall::UnregisterMR { ptr, ret: tx })
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))
    }

    pub fn send_command(&self, mut cmd: WorkerCommand) -> Result<()> {
        let _admission = self.admission.lock();
        if let WorkerCommand::SubmitTransfer { request, admission, .. } = &mut cmd {
            use crate::api::DomainGroupRouting;
            let (route, count) = match request {
                TransferRequest::Single(request) => (Some(request.domain), 1),
                TransferRequest::Gather(request) => (Some(request.domain), 1),
                TransferRequest::Imm(request) => (Some(request.domain), 1),
                TransferRequest::Barrier(request) => {
                    (Some(request.domain), request.dst_mrs.len())
                }
                TransferRequest::WriteBatch(batch) => (
                    Some(DomainGroupRouting::Pinned {
                        domain_idx: (batch.lane / crate::api::WRITE_QP_LANES) as u8,
                    }),
                    batch.writes.len(),
                ),
                // Paged/scatter use conservative domain-wide upper bounds as well.
                TransferRequest::Paged(request) => {
                    (None, request.src_page_indices.len().saturating_add(1))
                }
                // Scatter posts one WRITE (optionally WITH_IMM) per target/domain.
                TransferRequest::Scatter(request) => (None, request.dsts.len()),
            };
            let slots = self
                .write_budget
                .iter()
                .enumerate()
                .filter(|(index, _)| match route {
                    Some(DomainGroupRouting::Pinned { domain_idx }) => {
                        *index == domain_idx as usize
                    }
                    _ => true,
                })
                .map(|(_, used)| {
                    used.fetch_add(count, SeqCst);
                    (used.clone(), count)
                })
                .collect();
            *admission = Some(Arc::new(WriteBudgetGuard(slots)));
        }
        self.worker
            .cmd_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        Ok(())
    }
}

#[cfg(test)]
mod host_lane_tests {
    use super::*;
    use crate::api::MemoryRegionRemoteKey;

    #[test]
    fn domain_budget_distinguishes_invalid_from_temporary_full_and_reclaims_once() {
        assert!(check_write_budget(0, DOMAIN_WRITE_BUDGET).is_ok());
        assert!(matches!(
            check_write_budget(1, DOMAIN_WRITE_BUDGET),
            Err(FabricLibError::Full)
        ));
        assert!(matches!(check_write_budget(usize::MAX, 1), Err(FabricLibError::Full)));
        assert!(matches!(
            check_write_budget(0, DOMAIN_WRITE_BUDGET + 1),
            Err(FabricLibError::Custom(_))
        ));
        let used = Arc::new(AtomicUsize::new(7));
        let guard = Arc::new(WriteBudgetGuard(vec![(used.clone(), 7)]));
        let in_flight = guard.clone();
        drop(guard);
        assert_eq!(used.load(SeqCst), 7);
        drop(in_flight);
        assert_eq!(used.load(SeqCst), 0);
    }

    #[test]
    fn every_worker_and_hca_is_addressable() {
        let descriptor = MemoryRegionDescriptor {
            ptr: 4096,
            addr_rkey_list: (0..6)
                .map(|i| {
                    (DomainAddress(vec![i].into()), MemoryRegionRemoteKey(i as u64))
                })
                .collect(),
        };
        for lane in 0..6 {
            let (worker, domain) = host_write_lane(lane, 3, 2).unwrap();
            let mut narrowed = descriptor.clone();
            narrow_host_descriptor(&mut narrowed, worker, 3, 2).unwrap();
            assert_eq!(narrowed.addr_rkey_list.len(), 2);
            assert_eq!(
                narrowed.addr_rkey_list[domain],
                descriptor.addr_rkey_list[lane]
            );
        }
        assert!(host_write_lane(6, 3, 2).is_err());
        assert!(host_write_lane(0, 3, 0).is_err());
        assert!(narrow_host_descriptor(&mut descriptor.clone(), 0, 3, 1).is_err());
        assert!(narrow_host_descriptor(&mut descriptor.clone(), 3, 3, 2).is_err());
    }

    #[test]
    fn single_hca_layout_is_unchanged() {
        for lane in 0..4 {
            assert_eq!(host_write_lane(lane, 4, 1).unwrap(), (lane, 0));
        }
    }
}
