//! A stable, bounded WR chain on one RC QP. Only the tail requests a success CQE.
use std::{
    ffi::c_void,
    ptr::{NonNull, null_mut},
};

use libibverbs_sys::{
    IBV_SEND_SIGNALED, IBV_WR_RDMA_WRITE, IBV_WR_RDMA_WRITE_WITH_IMM, ibv_qp,
    ibv_send_wr, ibv_sge,
};

use crate::{
    api::{MAX_GATHER_SEGMENTS, MAX_WRITE_BATCH_WR},
    rdma_op::GatherWriteOp,
};

#[repr(C)]
struct CompletionToken {
    parent: *mut c_void,
    index: usize,
}

struct Slot {
    token: CompletionToken,
    wr: ibv_send_wr,
    sges: [ibv_sge; MAX_GATHER_SEGMENTS],
}

pub struct BatchWriteOpIter {
    qp: NonNull<ibv_qp>,
    slots: Box<[Slot]>,
    posted: usize,
    end: usize,
    aborting: bool,
}

impl BatchWriteOpIter {
    pub fn new(
        ops: Vec<GatherWriteOp>,
        qp: NonNull<ibv_qp>,
        parent: *mut c_void,
    ) -> Self {
        assert!(!ops.is_empty() && ops.len() <= MAX_WRITE_BATCH_WR);
        let mut slots = (0..ops.len())
            .map(|index| Slot {
                token: CompletionToken { parent, index },
                wr: ibv_send_wr::default(),
                sges: [ibv_sge::default(); MAX_GATHER_SEGMENTS],
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let length = slots.len();
        let base = slots.as_mut_ptr();
        for (index, (slot, op)) in slots.iter_mut().zip(ops).enumerate() {
            assert!(op.sources.len() <= MAX_GATHER_SEGMENTS);
            for (sge, source) in slot.sges.iter_mut().zip(&op.sources) {
                *sge = ibv_sge {
                    addr: (source.src_ptr.as_ptr() as u64)
                        .checked_add(source.src_offset)
                        .unwrap(),
                    length: u32::try_from(source.length).expect("validated SGE length"),
                    lkey: source.src_desc.0 as u32,
                };
            }
            let tail = index + 1 == length;
            slot.wr = ibv_send_wr {
                // Tagged pointers distinguish per-WR retirement indices from legacy contexts.
                wr_id: (&raw mut slot.token as u64) | 1,
                next: if tail {
                    null_mut()
                } else {
                    unsafe { &raw mut (*base.add(index + 1)).wr }
                },
                sg_list: slot.sges.as_mut_ptr(),
                num_sge: op.sources.len() as i32,
                opcode: if op.imm_data.is_some() {
                    IBV_WR_RDMA_WRITE_WITH_IMM
                } else {
                    IBV_WR_RDMA_WRITE
                },
                send_flags: if tail { IBV_SEND_SIGNALED } else { 0 },
                ..Default::default()
            };
            slot.wr.__bindgen_anon_1.imm_data = op.imm_data.unwrap_or(0);
            slot.wr.wr.rdma.remote_addr =
                op.dst_ptr.checked_add(op.dst_offset).unwrap();
            slot.wr.wr.rdma.rkey = op.dst_rkey.0 as u32;
        }
        Self { qp, slots, posted: 0, end: length, aborting: false }
    }

    pub fn total_ops(&self) -> usize {
        self.end
    }

    pub fn peek(&self) -> (*mut ibv_qp, *mut ibv_send_wr, usize) {
        if self.posted == self.end {
            (self.qp.as_ptr(), null_mut(), 0)
        } else {
            (
                self.qp.as_ptr(),
                (&raw const self.slots[self.posted].wr).cast_mut(),
                self.end - self.posted,
            )
        }
    }

    pub fn advance(&mut self, count: usize) {
        assert!(count <= self.end - self.posted);
        self.posted += count;
    }

    /// Replace the rejected suffix with one zero-byte, signaled drain marker. Its CQE retires
    /// any already completed unsignaled prefix as well as still-pending WRs. No notification is
    /// delivered, and the caller always reports the original submission error.
    pub fn abort_tail(&mut self) -> bool {
        if self.aborting || self.posted == 0 || self.posted == self.end {
            return false;
        }
        let wr = &mut self.slots[self.posted].wr;
        wr.next = null_mut();
        wr.num_sge = 0;
        wr.sg_list = null_mut();
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        self.end = self.posted + 1;
        self.aborting = true;
        true
    }
}

/// Only call for a CQE on an RMA QP owned by this domain, before retiring its WR context.
pub unsafe fn write_completion_context(wr_id: u64) -> (*mut c_void, Option<usize>) {
    if wr_id & 1 == 0 {
        (wr_id as *mut c_void, None)
    } else {
        let token = unsafe { &*((wr_id & !1) as *const CompletionToken) };
        (token.parent, Some(token.index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::MemoryRegionRemoteKey, mr::MemoryRegionLocalDescriptor,
        rdma_op::GatherSource,
    };

    fn ops(count: usize) -> Vec<GatherWriteOp> {
        (0..count)
            .map(|index| GatherWriteOp {
                sources: [GatherSource {
                    src_ptr: NonNull::dangling(),
                    src_desc: MemoryRegionLocalDescriptor(1),
                    src_offset: 0,
                    length: 8,
                }]
                .into_iter()
                .collect(),
                imm_data: (index + 1 == count).then_some(7),
                dst_ptr: 4096,
                dst_rkey: MemoryRegionRemoteKey(2),
                dst_offset: index as u64 * 8,
                length: 8,
            })
            .collect()
    }

    #[test]
    fn one_qp_one_tail_completion_and_stable_partial_post() {
        let parent = NonNull::<u64>::dangling().as_ptr().cast();
        let mut batch =
            BatchWriteOpIter::new(ops(MAX_WRITE_BATCH_WR), NonNull::dangling(), parent);
        let (_, mut wr, count) = batch.peek();
        assert_eq!(count, MAX_WRITE_BATCH_WR);
        for index in 0..count {
            let current = unsafe { &*wr };
            assert_eq!(current.send_flags & IBV_SEND_SIGNALED != 0, index + 1 == count);
            assert_eq!(
                unsafe { write_completion_context(current.wr_id) },
                (parent, Some(index))
            );
            wr = current.next;
        }
        assert!(wr.is_null());
        batch.advance(17);
        let (_, wr, left) = batch.peek();
        assert_eq!(left, count - 17);
        assert_eq!(
            unsafe { write_completion_context((*wr).wr_id) },
            (parent, Some(17))
        );
        batch.advance(left);
        assert_eq!(batch.peek().2, 0);
    }

    #[test]
    fn rejected_suffix_is_replaced_by_a_single_drain_marker() {
        let mut batch = BatchWriteOpIter::new(
            ops(8),
            NonNull::dangling(),
            NonNull::<u64>::dangling().as_ptr().cast(),
        );
        batch.advance(3);
        assert!(batch.abort_tail());
        let (_, wr, count) = batch.peek();
        assert_eq!(count, 1);
        assert_eq!(batch.total_ops(), 4);
        let wr = unsafe { &*wr };
        assert_eq!(wr.num_sge, 0);
        assert_eq!(wr.opcode, IBV_WR_RDMA_WRITE);
        assert_eq!(wr.send_flags, IBV_SEND_SIGNALED);
        assert!(wr.next.is_null());
        assert_eq!(unsafe { write_completion_context(wr.wr_id) }.1, Some(3));
        assert!(!batch.abort_tail());
    }
}
