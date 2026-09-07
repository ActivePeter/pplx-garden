use std::{cmp::min, collections::HashMap, ffi::c_void, ptr::NonNull, sync::Arc};

use crate::{
    api::{
        DomainAddress, DomainGroupRouting, GatherTransferRequest, GroupTransferRouting,
        MAX_GATHER_SEGMENTS, MemoryRegionDescriptor, MemoryRegionHandle,
        PagedTransferRequest, PeerGroupHandle, ScatterTransferRequest,
        SingleTransferRequest, SmallVec, TransferCounter, TransferId, TransferRequest,
    },
    error::{FabricLibError, Result},
    mr::MemoryRegion,
    provider::{DomainCompletionEntry, RdmaDomain},
    rdma_op::{
        GatherSource, GatherWriteOp, GroupWriteOp, ImmWriteOp, PagedWriteOp, RecvOp,
        ScatterGroupWriteOp, SendOp, SingleWriteOp, WriteOp,
    },
};

pub struct DomainGroup<D: RdmaDomain, const N: usize> {
    domains: [D; N],
    write_ops: HashMap<TransferId, WriteOpContext>,
    rr_next: usize,
    next_peer_group_handle: u32,
    peer_groups: HashMap<PeerGroupHandle, PeerGroup>,
}

struct WriteOpContext {
    num_used_domains: usize,
    cnt_domain_completion: usize,
    tx_counter: Option<TransferCounter>,
    error: Option<FabricLibError>,
    admission: Option<Arc<dyn Send + Sync>>,
}

impl WriteOpContext {
    fn retire_domain(&mut self, error: Option<FabricLibError>) -> bool {
        self.cnt_domain_completion += 1;
        if self.error.is_none() {
            self.error = error;
        }
        self.cnt_domain_completion == self.num_used_domains
    }
}

struct PeerGroup {
    addrs: Vec<SmallVec<DomainAddress>>,
}

impl<D: RdmaDomain, const N: usize> DomainGroup<D, N> {
    pub fn new(domains: [D; N]) -> Self {
        Self {
            domains,
            write_ops: HashMap::new(),
            rr_next: 0,
            next_peer_group_handle: 0,
            peer_groups: HashMap::new(),
        }
    }

    pub fn aggregate_link_speed(&self) -> u64 {
        self.domains.iter().map(|d| d.link_speed()).sum()
    }

    pub fn supports_write_batch(&self) -> bool {
        self.domains.iter().all(RdmaDomain::supports_write_batch)
    }

    pub fn register_mr_allow_remote(
        &mut self,
        region: &MemoryRegion,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)> {
        let mut addr_rkey_list = SmallVec::new();
        for domain in self.domains.iter_mut() {
            let rkey = domain.register_mr_allow_remote(region)?;
            addr_rkey_list.push((domain.addr(), rkey));
        }
        let ptr = region.ptr();
        let local = MemoryRegionHandle::new(ptr);
        let remote =
            MemoryRegionDescriptor { ptr: ptr.as_ptr() as u64, addr_rkey_list };
        Ok((local, remote))
    }

    pub fn register_mr_local(
        &mut self,
        region: &MemoryRegion,
    ) -> Result<MemoryRegionHandle> {
        for domain in self.domains.iter_mut() {
            domain.register_mr_local(region)?;
        }
        Ok(MemoryRegionHandle::new(region.ptr()))
    }

    pub fn unregister_mr(&mut self, ptr: NonNull<c_void>) {
        for domain in self.domains.iter_mut() {
            domain.unregister_mr(ptr);
        }
    }

    pub fn add_peer_group(
        &mut self,
        addrs: Vec<SmallVec<DomainAddress>>,
    ) -> Result<PeerGroupHandle> {
        for (handle, group) in self.peer_groups.iter() {
            if group.addrs == addrs {
                return Ok(*handle);
            }
        }
        let handle = PeerGroupHandle(self.next_peer_group_handle);
        self.next_peer_group_handle += 1;
        self.peer_groups.insert(handle, PeerGroup { addrs: addrs.clone() });

        for (i, domain) in self.domains.iter_mut().enumerate() {
            let domain_addrs = addrs.iter().map(|addr| addr[i].clone()).collect();
            domain.add_peer_group(handle, domain_addrs)?;
        }

        Ok(handle)
    }

    pub fn submit_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: TransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        match request {
            TransferRequest::Imm(request) => self.submit_imm_transfer_request(
                transfer_id,
                request.imm_data,
                &[request.dst_mr],
                request.domain,
                tx_counter,
            ),
            TransferRequest::Barrier(request) => self.submit_imm_transfer_request(
                transfer_id,
                request.imm_data,
                &request.dst_mrs,
                request.domain,
                tx_counter,
            ),
            TransferRequest::Single(request) => {
                self.submit_single_transfer_request(transfer_id, request, tx_counter)
            }
            TransferRequest::Gather(request) => {
                self.submit_gather_transfer_request(transfer_id, request, tx_counter)
            }
            TransferRequest::WriteBatch(request) => {
                self.submit_write_batch(transfer_id, request, tx_counter)
            }
            TransferRequest::Paged(request) => {
                self.submit_paged_transfer_request(transfer_id, request, tx_counter)
            }
            TransferRequest::Scatter(request) => {
                self.submit_scatter_transfer_request(transfer_id, request, tx_counter)
            }
        }
    }

    pub fn submit_transfer_request_guarded(
        &mut self,
        id: TransferId,
        request: TransferRequest,
        counter: Option<TransferCounter>,
        admission: Option<Arc<dyn Send + Sync>>,
    ) -> Result<()> {
        self.submit_transfer_request(id, request, counter)?;
        if let Some(context) = self.write_ops.get_mut(&id) {
            context.admission = admission;
        }
        Ok(())
    }

    fn submit_write_batch(
        &mut self,
        transfer_id: TransferId,
        request: crate::api::WriteBatchRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        use crate::api::MAX_WRITE_BATCH_WR;
        if request.writes.is_empty() || request.writes.len() > MAX_WRITE_BATCH_WR {
            return Err(FabricLibError::Custom("invalid write batch length"));
        }
        let domain_index = request.lane / crate::api::WRITE_QP_LANES;
        let domain = self
            .domains
            .get_mut(domain_index)
            .ok_or(FabricLibError::Custom("write batch lane is out of bounds"))?;
        if !domain.supports_write_batch() {
            return Err(FabricLibError::Custom(
                "provider does not support ordered write batches",
            ));
        }
        let mut destination = None;
        let mut ops = Vec::with_capacity(request.writes.len());
        // Validate the complete list before submitting any write to the domain.
        for write in request.writes {
            if write.dst_mr.addr_rkey_list.len() != N
                || (write.segments.is_empty() && write.imm_data.is_none())
                || write.segments.len() > MAX_GATHER_SEGMENTS
            {
                return Err(FabricLibError::Custom(
                    "invalid write batch registration or segments",
                ));
            }
            let (address, rkey) = &write.dst_mr.addr_rkey_list[domain_index];
            if destination.as_ref().is_some_and(|old| old != address) {
                return Err(FabricLibError::Custom(
                    "write batch destinations use different peers",
                ));
            }
            destination = Some(address.clone());
            let mut sources = SmallVec::new();
            let mut length = 0_u64;
            for segment in write.segments {
                length = length
                    .checked_add(segment.length)
                    .ok_or(FabricLibError::Custom("write batch length overflow"))?;
                if segment.length == 0 || length > u32::MAX as u64 {
                    return Err(FabricLibError::Custom(
                        "write batch exceeds SGE length limit",
                    ));
                }
                (segment.src_mr.ptr.as_ptr() as u64)
                    .checked_add(segment.src_offset)
                    .and_then(|start| start.checked_add(segment.length))
                    .ok_or(FabricLibError::Custom("write batch source overflow"))?;
                sources.push(GatherSource {
                    src_ptr: segment.src_mr.ptr,
                    src_desc: domain.get_mem_desc(segment.src_mr.ptr)?,
                    src_offset: segment.src_offset,
                    length: segment.length,
                });
            }
            write
                .dst_mr
                .ptr
                .checked_add(write.dst_offset)
                .and_then(|start| start.checked_add(length))
                .ok_or(FabricLibError::Custom("write batch destination overflow"))?;
            ops.push(GatherWriteOp {
                sources,
                length,
                imm_data: write.imm_data,
                dst_ptr: write.dst_mr.ptr,
                dst_rkey: *rkey,
                dst_offset: write.dst_offset,
            });
        }
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: 1,
                cnt_domain_completion: 0,
                tx_counter,
                error: None,
                admission: None,
            },
        );
        domain.submit_write(
            transfer_id,
            destination.unwrap(),
            WriteOp::Batch {
                qp_lane: request.lane % crate::api::WRITE_QP_LANES,
                writes: ops,
            },
        );
        Ok(())
    }

    pub fn submit_imm_transfer_request(
        &mut self,
        transfer_id: TransferId,
        imm_data: u32,
        dst_mrs: &[MemoryRegionDescriptor],
        domain: DomainGroupRouting,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Sanity check the input descriptors.
        for dst_mr in dst_mrs {
            if dst_mr.addr_rkey_list.len() != self.domains.len() {
                return Err(FabricLibError::Custom(
                    "Number of target addresses must match the number of domains",
                ));
            }
        }

        // Bookkeeping.
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: dst_mrs.len(),
                cnt_domain_completion: 0,
                tx_counter,
                error: None,
                admission: None,
            },
        );

        // Determine the number of imms to send via each domain.
        let num_domains = self.domains.len();
        let (first_domain, imm_per_domain) = match domain {
            DomainGroupRouting::RoundRobinSharded { num_shards } => {
                if num_shards.get() != 1 {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::RoundRobinSharded should have num_shards = 1 for BarrierTransferRequest",
                    ));
                }
                let first_domain = self.rr_next;
                self.rr_next =
                    (self.rr_next + num_domains.min(dst_mrs.len())) % num_domains;
                (first_domain, dst_mrs.len().div_ceil(num_domains))
            }
            DomainGroupRouting::Pinned { domain_idx } => {
                if domain_idx as usize >= self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::Pinned.domain_idx is out of bounds",
                    ));
                }
                (domain_idx as usize, dst_mrs.len())
            }
        };

        // Chunk by domain and submit the writes.
        for (i, dst_mrs) in dst_mrs.chunks(imm_per_domain).enumerate() {
            let domain = &mut self.domains[(first_domain + i) % num_domains];
            for dst_mr in dst_mrs {
                let (dst_addr, dst_rkey) = &dst_mr.addr_rkey_list[i];

                // Construct rdma op
                let rdma_op = WriteOp::Imm(ImmWriteOp {
                    imm_data,
                    dst_ptr: dst_mr.ptr,
                    dst_rkey: *dst_rkey,
                });

                // Submit the transfer request to the domain
                domain.submit_write(transfer_id, dst_addr.clone(), rdma_op);
            }
        }
        Ok(())
    }

    pub fn submit_single_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: SingleTransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Validate
        if request.dst_mr.addr_rkey_list.len() != self.domains.len() {
            return Err(FabricLibError::Custom(
                "Number of target addresses must match the number of domains",
            ));
        }

        // Statically shard the bytes across domains
        let num_shards = match request.domain {
            DomainGroupRouting::RoundRobinSharded { num_shards } => {
                if num_shards.get() as usize > self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::RoundRobinSharded.num_shards is greater than the number of domains",
                    ));
                }
                num_shards.get() as usize
            }
            DomainGroupRouting::Pinned { domain_idx } => {
                if domain_idx as usize >= self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::Pinned.domain_idx is out of bounds",
                    ));
                }
                1usize
            }
        };
        let ranges = shard_single_transfer(request.length as usize, num_shards);

        // Construct rdma ops
        let mut rdma_ops = SmallVec::with_capacity(ranges.len());
        for (offset, len) in ranges {
            // Sharding
            let i = match request.domain {
                DomainGroupRouting::RoundRobinSharded { .. } => {
                    let ret = self.rr_next;
                    self.rr_next = (self.rr_next + 1) % self.domains.len();
                    ret
                }
                DomainGroupRouting::Pinned { domain_idx } => domain_idx as usize,
            };

            // Address lookup
            let domain = &mut self.domains[i];
            let (dst_addr, dst_rkey) = &request.dst_mr.addr_rkey_list[i];

            // Get the source memory region descriptor
            let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;

            // Build the rdma op iter
            let op = WriteOp::Single(SingleWriteOp {
                src_ptr: request.src_mr.ptr,
                src_desc,
                src_offset: request.src_offset + offset as u64,
                length: len as u64,
                imm_data: request.imm_data,
                dst_ptr: request.dst_mr.ptr,
                dst_rkey: *dst_rkey,
                dst_offset: request.dst_offset + offset as u64,
            });
            rdma_ops.push((i, op, dst_addr.clone()));
        }

        // Bookkeeping
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: rdma_ops.len(),
                cnt_domain_completion: 0,
                tx_counter,
                error: None,
                admission: None,
            },
        );

        // Submit the transfer request to each domain
        for (domain_idx, rdma_op, dst_addr) in rdma_ops {
            self.domains[domain_idx].submit_write(transfer_id, dst_addr, rdma_op);
        }
        Ok(())
    }

    pub fn submit_gather_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: GatherTransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        if request.segments.is_empty()
            || request.segments.iter().any(|part| part.length == 0)
        {
            return Err(FabricLibError::Custom(
                "GatherTransferRequest requires non-empty segments",
            ));
        }
        if request.segments.len() > MAX_GATHER_SEGMENTS {
            return Err(FabricLibError::Custom(
                "GatherTransferRequest has too many segments",
            ));
        }
        if request.segments.iter().any(|part| part.length > u32::MAX as u64) {
            return Err(FabricLibError::Custom(
                "GatherTransferRequest segment exceeds the verbs SGE length limit",
            ));
        }
        if request.dst_mr.addr_rkey_list.len() != self.domains.len() {
            return Err(FabricLibError::Custom(
                "Number of target addresses must match the number of domains",
            ));
        }

        let num_shards = match request.domain {
            DomainGroupRouting::RoundRobinSharded { num_shards } => {
                if num_shards.get() as usize > self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::RoundRobinSharded.num_shards is greater than the number of domains",
                    ));
                }
                num_shards.get() as usize
            }
            DomainGroupRouting::Pinned { domain_idx } => {
                if domain_idx as usize >= self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::Pinned.domain_idx is out of bounds",
                    ));
                }
                1
            }
        };

        let length = request.segments.iter().try_fold(0_u64, |length, segment| {
            length.checked_add(segment.length).ok_or(FabricLibError::Custom(
                "GatherTransferRequest destination range overflows",
            ))
        })?;
        let length_usize = usize::try_from(length).map_err(|_| {
            FabricLibError::Custom("GatherTransferRequest length does not fit usize")
        })?;
        let ranges = divide_evenly(length_usize, num_shards);
        let first_domain = match request.domain {
            DomainGroupRouting::RoundRobinSharded { .. } => self.rr_next,
            DomainGroupRouting::Pinned { domain_idx } => domain_idx as usize,
        };
        if matches!(request.domain, DomainGroupRouting::RoundRobinSharded { .. }) {
            self.rr_next = (self.rr_next + ranges.len()) % self.domains.len();
        }

        let mut rdma_ops = SmallVec::with_capacity(ranges.len());
        for (shard_index, (begin, end)) in ranges.into_iter().enumerate() {
            let domain_idx = (first_domain + shard_index) % self.domains.len();
            let domain = &mut self.domains[domain_idx];
            let (dst_addr, dst_rkey) = &request.dst_mr.addr_rkey_list[domain_idx];
            let mut sources = SmallVec::with_capacity(request.segments.len());
            let mut logical_offset = 0_usize;
            for segment in request.segments.iter() {
                let segment_len = segment.length as usize;
                let segment_end = logical_offset + segment_len;
                let overlap_begin = begin.max(logical_offset);
                let overlap_end = end.min(segment_end);
                if overlap_begin < overlap_end {
                    let src_desc = domain.get_mem_desc(segment.src_mr.ptr)?;
                    sources.push(GatherSource {
                        src_ptr: segment.src_mr.ptr,
                        src_desc,
                        src_offset: segment.src_offset
                            + (overlap_begin - logical_offset) as u64,
                        length: (overlap_end - overlap_begin) as u64,
                    });
                }
                logical_offset = segment_end;
                if logical_offset >= end {
                    break;
                }
            }
            if sources.is_empty() {
                return Err(FabricLibError::Custom(
                    "GatherTransferRequest produced an empty shard",
                ));
            }
            rdma_ops.push((
                domain_idx,
                dst_addr.clone(),
                WriteOp::Gather(GatherWriteOp {
                    sources,
                    imm_data: request.imm_data,
                    dst_ptr: request.dst_mr.ptr,
                    dst_rkey: *dst_rkey,
                    dst_offset: request.dst_offset + begin as u64,
                    length: (end - begin) as u64,
                }),
            ));
        }

        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: rdma_ops.len(),
                cnt_domain_completion: 0,
                tx_counter,
                error: None,
                admission: None,
            },
        );
        for (domain_idx, dst_addr, rdma_op) in rdma_ops {
            self.domains[domain_idx].submit_write(transfer_id, dst_addr, rdma_op);
        }
        Ok(())
    }

    pub fn submit_paged_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: PagedTransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Validate
        if request.dst_mr.addr_rkey_list.len() != self.domains.len() {
            return Err(FabricLibError::Custom(
                "Number of target addresses must match the number of domains",
            ));
        }
        if request.src_page_indices.len() != request.dst_page_indices.len() {
            return Err(FabricLibError::Custom(
                "Length of source and destination page indices must match",
            ));
        }

        // Statically shard the page indices across domains
        let page_range =
            divide_evenly(request.src_page_indices.len(), self.domains.len());

        // Bookkeeping
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: page_range.len(),
                cnt_domain_completion: 0,
                tx_counter,
                error: None,
                admission: None,
            },
        );

        // Construct rdma op iter
        for (beg, end) in page_range {
            // Round-robin
            let i = self.rr_next;
            self.rr_next = (self.rr_next + 1) % self.domains.len();

            // Address lookup
            let domain = &mut self.domains[i];
            let (dst_addr, dst_rkey) = &request.dst_mr.addr_rkey_list[i];

            // Get the source memory region descriptor
            let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;

            // Build the rdma op
            let op = WriteOp::Paged(PagedWriteOp {
                src_page_indices: Arc::clone(&request.src_page_indices),
                dst_page_indices: Arc::clone(&request.dst_page_indices),
                page_indices_beg: beg,
                page_indices_end: end,
                length: request.length,
                src_ptr: request.src_mr.ptr,
                src_desc,
                src_stride: request.src_stride,
                src_offset: request.src_offset,
                dst_ptr: request.dst_mr.ptr,
                dst_rkey: *dst_rkey,
                dst_stride: request.dst_stride,
                dst_offset: request.dst_offset,
                imm_data: request.imm_data,
            });
            domain.submit_write(transfer_id, dst_addr.clone(), op);
        }
        Ok(())
    }

    pub fn submit_scatter_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: ScatterTransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Validate
        if request.dsts.is_empty() {
            return Err(FabricLibError::Custom("Empty scatter targets"));
        }
        if let Some(dst_handle) = &request.dst_handle {
            let group = self
                .peer_groups
                .get(dst_handle)
                .ok_or(FabricLibError::Custom("PeerGroupHandle not found"))?;
            if request.dsts.len() != group.addrs.len() {
                return Err(FabricLibError::Custom(
                    "Number of scatter targets must match the number of peer group addresses",
                ));
            }
        }

        // Statically shard the transfer across domains.
        let mut domain_indices = SmallVec::new();
        let mut rdma_ops = SmallVec::new();
        match request.domain {
            GroupTransferRouting::AllDomainsShardPeers => {
                let dst_range = divide_evenly(request.dsts.len(), self.domains.len());
                for ((domain_idx, domain), (beg, end)) in
                    self.domains.iter_mut().enumerate().zip(dst_range)
                {
                    let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;
                    let op = GroupWriteOp::Scatter(ScatterGroupWriteOp {
                        domain_idx,
                        src_ptr: request.src_mr.ptr,
                        src_desc,
                        imm_data: request.imm_data,
                        dsts: Arc::clone(&request.dsts),
                        dst_beg: beg,
                        dst_end: end,
                        byte_shards: 1,
                        byte_shard_idx: 0,
                    });
                    rdma_ops.push(op);
                    domain_indices.push(domain_idx);
                }
            }
            GroupTransferRouting::AllDomainsShardBytes => {
                for (domain_idx, domain) in self.domains.iter_mut().enumerate() {
                    let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;
                    let op = GroupWriteOp::Scatter(ScatterGroupWriteOp {
                        domain_idx,
                        src_ptr: request.src_mr.ptr,
                        src_desc,
                        imm_data: request.imm_data,
                        dsts: Arc::clone(&request.dsts),
                        dst_beg: 0,
                        dst_end: request.dsts.len(),
                        byte_shards: N as u32,
                        byte_shard_idx: domain_idx as u32,
                    });
                    rdma_ops.push(op);
                    domain_indices.push(domain_idx);
                }
            }
            GroupTransferRouting::Single { domain_idx } => {
                let domain = &mut self.domains[domain_idx as usize];
                let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;
                let op = GroupWriteOp::Scatter(ScatterGroupWriteOp {
                    domain_idx: domain_idx as usize,
                    src_ptr: request.src_mr.ptr,
                    src_desc,
                    imm_data: request.imm_data,
                    dsts: Arc::clone(&request.dsts),
                    dst_beg: 0,
                    dst_end: request.dsts.len(),
                    byte_shards: 1,
                    byte_shard_idx: 0,
                });
                rdma_ops.push(op);
                domain_indices.push(domain_idx as usize);
            }
        }

        // Bookkeeping
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: rdma_ops.len(),
                cnt_domain_completion: 0,
                tx_counter,
                error: None,
                admission: None,
            },
        );

        // Submit the transfer request to each domain
        for (domain_idx, rdma_op) in domain_indices.into_iter().zip(rdma_ops) {
            let domain = &mut self.domains[domain_idx];
            domain.submit_group_write(transfer_id, request.dst_handle, rdma_op);
        }

        Ok(())
    }

    pub fn submit_send(
        &mut self,
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        addr: DomainAddress,
        coalescible: bool,
    ) -> Result<()> {
        let domain = &mut self.domains[0];
        let desc = domain.get_mem_desc(mr.ptr)?;
        domain.submit_send(transfer_id, addr, SendOp { ptr, len, desc, coalescible });
        Ok(())
    }

    pub fn submit_recv(
        &mut self,
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<()> {
        let domain = &mut self.domains[0];
        let desc = domain.get_mem_desc(mr.ptr)?;
        domain.submit_recv(transfer_id, RecvOp { ptr, len, desc });
        Ok(())
    }

    /// Poll RDMA completion queue and send pending RDMA operations.
    pub fn poll_progress(&mut self) {
        for domain in self.domains.iter_mut() {
            domain.poll_progress();
        }
    }

    /// Return Error after every submitted domain has retired, if any domain failed.
    /// Return Transfer if all domains have completed the transfer.
    /// Return Recv, Send, ImmData if any domain returns so.
    /// Return None otherwise.
    pub fn get_completion(&mut self) -> Option<DomainCompletionEntry> {
        for index in 0..N {
            if let Some(c) = self.domains[index].get_completion() {
                match c {
                    DomainCompletionEntry::Error(transfer_id, err) => {
                        if self.write_ops.contains_key(&transfer_id) {
                            if let Some(completion) =
                                self.retire_write(transfer_id, Some(err))
                            {
                                return Some(completion);
                            }
                        } else {
                            return Some(DomainCompletionEntry::Error(
                                transfer_id,
                                err,
                            ));
                        }
                    }
                    DomainCompletionEntry::Transfer(transfer_id) => {
                        if let Some(completion) = self.retire_write(transfer_id, None) {
                            return Some(completion);
                        }
                    }
                    DomainCompletionEntry::ImmData(imm_data) => {
                        return Some(DomainCompletionEntry::ImmData(imm_data));
                    }
                    DomainCompletionEntry::Immediate(event) => {
                        return Some(DomainCompletionEntry::Immediate(event));
                    }
                    DomainCompletionEntry::Recv { transfer_id, data_len } => {
                        return Some(DomainCompletionEntry::Recv {
                            transfer_id,
                            data_len,
                        });
                    }
                    DomainCompletionEntry::Send(transfer_id) => {
                        return Some(DomainCompletionEntry::Send(transfer_id));
                    }
                    DomainCompletionEntry::ImmCountReached(imm) => {
                        return Some(DomainCompletionEntry::ImmCountReached(imm));
                    }
                }
            }
        }
        None
    }

    fn retire_write(
        &mut self,
        id: TransferId,
        error: Option<FabricLibError>,
    ) -> Option<DomainCompletionEntry> {
        let context = self.write_ops.get_mut(&id)?;
        if !context.retire_domain(error) {
            return None;
        }
        let context = self.write_ops.remove(&id).unwrap();
        if let Some(counter) = context.tx_counter {
            if context.error.is_some() {
                counter.error();
            } else {
                counter.done();
            }
            None
        } else if let Some(error) = context.error {
            Some(DomainCompletionEntry::Error(id, error))
        } else {
            Some(DomainCompletionEntry::Transfer(id))
        }
    }
}

fn divide_evenly(n: usize, k: usize) -> SmallVec<(usize, usize)> {
    let mut result = SmallVec::new();
    let step = n.div_ceil(k);
    let mut remainder = n;
    let mut beg = 0;
    for _ in 0..k {
        let chunk = remainder.min(step);
        if chunk == 0 {
            break;
        }
        result.push((beg, beg + chunk));
        beg += chunk;
        remainder -= chunk;
    }
    debug_assert!(remainder == 0);
    result
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn shard_single_transfer(
    total_size: usize,
    num_shards: usize,
) -> SmallVec<(usize, usize)> {
    const MIN_SIZE: usize = 8192;
    let mut result = SmallVec::new();
    let base = round_up(total_size.div_ceil(num_shards), MIN_SIZE);
    let mut offset = 0;
    for _ in 0..num_shards {
        let len = min(base, total_size - offset);
        if len > 0 {
            result.push((offset, len));
            offset += len;
        } else {
            // For 0-length WRITE, it's possible that the offset is at the end of the MR.
            // To avoid out of bound access, we always use 0 offset for a 0-length WRITE.
            result.push((0, 0));
        }
    }
    result
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    #[test]
    fn first_domain_error_keeps_other_domain_resources_alive() {
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let mut context = WriteOpContext {
            num_used_domains: 3,
            cnt_domain_completion: 0,
            tx_counter: None,
            error: None,
            admission: Some(resource),
        };
        assert!(
            !context.retire_domain(Some(FabricLibError::Custom("first QP failed")))
        );
        assert!(weak.upgrade().is_some());
        assert!(
            !context.retire_domain(Some(FabricLibError::Custom("another QP failed")))
        );
        assert!(weak.upgrade().is_some());
        assert!(context.retire_domain(None));
        assert!(matches!(
            context.error,
            Some(FabricLibError::Custom("first QP failed"))
        ));
        drop(context);
        assert!(weak.upgrade().is_none());
    }
}
