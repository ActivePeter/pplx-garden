pub mod api;
mod cuda_compat;
mod domain_group;
mod efa;
mod error;
mod fabric_engine;
mod host_buffer;
mod imm_count;
mod interface;
mod mr;
mod provider;
mod provider_dispatch;
mod rdma_op;
#[cfg(feature = "cuda")]
mod topo;
mod transfer_engine;
#[cfg(feature = "cuda")]
mod transfer_engine_builder;
mod utils;
mod verbs;
mod worker;

pub use cuda_compat::{CudaDeviceId, CudaHostMemory, Device, GdrFlag};
pub use domain_group::DomainGroup;
pub use efa::{EfaDomainInfo, get_efa_domains};
pub use error::*;
pub use fabric_engine::FabricEngine;
pub use host_buffer::{HostBuffer, HostBufferAllocator};
pub use interface::{
    AsyncTransferEngine, BouncingErrorCallback, BouncingRecvCallback, ErrorCallback,
    RdmaEngine, RecvCallback, SendBuffer, SendCallback, SendRecvEngine,
};
pub use provider::{RdmaDomain, RdmaDomainInfo};
pub use provider_dispatch::DomainInfo;
#[cfg(feature = "cuda")]
pub use topo::{TopologyGroup, detect_topology};
pub use transfer_engine::{
    ImmCountCallback, TransferCallback, TransferEngine, UvmWatcherCallback,
};
#[cfg(feature = "cuda")]
pub use transfer_engine_builder::TransferEngineBuilder;
pub use verbs::{VerbsDeviceInfo, VerbsDeviceList};
pub use worker::{InitializingWorker, Worker, WorkerHandle};

pub use interface::MockTestTransferEngine;
