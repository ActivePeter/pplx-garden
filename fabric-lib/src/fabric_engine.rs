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
            contexts.insert(device, WorkerContext { worker: w });
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
                    || descriptor.addr_rkey_list.len() != 1
                {
                    return Err(FabricLibError::Custom(
                        "striped host workers require one domain per worker",
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
        let source_device = match &request {
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
            narrow_host_transfer_request(&mut request, index, self.workers.len())?;
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
        for (_, ctx) in self.workers.iter() {
            ctx.worker.stop();
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
) -> Result<()> {
    if descriptor.addr_rkey_list.len() != worker_count {
        return Err(FabricLibError::Custom(
            "Remote memory descriptor does not match striped host worker count",
        ));
    }
    let selected = descriptor.addr_rkey_list[worker_index].clone();
    descriptor.addr_rkey_list.clear();
    descriptor.addr_rkey_list.push(selected);
    Ok(())
}

fn narrow_host_transfer_request(
    request: &mut TransferRequest,
    worker_index: usize,
    worker_count: usize,
) -> Result<()> {
    match request {
        TransferRequest::Imm(request) => {
            narrow_host_descriptor(&mut request.dst_mr, worker_index, worker_count)
        }
        TransferRequest::Barrier(request) => {
            for descriptor in &mut request.dst_mrs {
                narrow_host_descriptor(descriptor, worker_index, worker_count)?;
            }
            Ok(())
        }
        TransferRequest::Single(request) => {
            narrow_host_descriptor(&mut request.dst_mr, worker_index, worker_count)
        }
        TransferRequest::Gather(request) => {
            narrow_host_descriptor(&mut request.dst_mr, worker_index, worker_count)
        }
        TransferRequest::Paged(request) => {
            narrow_host_descriptor(&mut request.dst_mr, worker_index, worker_count)
        }
        TransferRequest::Scatter(request) => {
            for target in Arc::make_mut(&mut request.dsts) {
                narrow_host_descriptor(&mut target.dst_mr, worker_index, worker_count)?;
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

    pub fn send_command(&self, cmd: WorkerCommand) -> Result<()> {
        self.worker
            .cmd_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        Ok(())
    }
}
