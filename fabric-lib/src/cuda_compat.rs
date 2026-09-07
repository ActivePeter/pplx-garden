//! Small compatibility surface that keeps host-memory RDMA independent from CUDA.

#[cfg(feature = "cuda")]
pub use cuda_lib::*;

#[cfg(not(feature = "cuda"))]
mod host_only {
    use std::{
        ffi::c_void,
        ptr::NonNull,
        sync::atomic::{AtomicBool, Ordering},
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CudaDeviceId(pub u8);

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Device {
        Host,
        Cuda(CudaDeviceId),
    }

    #[derive(Debug, Default)]
    pub struct GdrFlag {
        value: AtomicBool,
    }

    impl GdrFlag {
        pub fn set(&self, value: bool) {
            self.value.store(value, Ordering::Relaxed);
        }
    }

    #[derive(Clone, Debug, thiserror::Error)]
    #[error("CUDA driver is unavailable in a host-only fabric-lib build: {detail}")]
    pub struct CudaDriverError {
        detail: &'static str,
    }

    #[derive(Clone, Debug, thiserror::Error)]
    #[error("CUDA runtime is unavailable in a host-only fabric-lib build: {detail}")]
    pub struct CudartError {
        detail: &'static str,
    }

    pub fn cu_get_dma_buf_fd(
        _ptr: NonNull<c_void>,
        _len: usize,
    ) -> Result<i32, CudaDriverError> {
        Err(CudaDriverError { detail: "DMA-BUF export requires the 'cuda' feature" })
    }

    #[allow(non_upper_case_globals)]
    pub const cudaMemoryTypeDevice: i32 = 0;

    #[derive(Debug, Clone, Copy)]
    pub struct CudaPointerAttributes {
        pub type_: i32,
    }

    #[allow(non_snake_case)]
    pub fn cudaPointerGetAttributes(
        _ptr: NonNull<c_void>,
    ) -> Result<CudaPointerAttributes, CudartError> {
        Err(CudartError {
            detail: "CUDA pointer inspection requires the 'cuda' feature",
        })
    }

    pub struct CudaHostMemory {
        storage: Box<[u64]>,
        pub ptr: NonNull<c_void>,
        pub size: usize,
    }

    impl CudaHostMemory {
        pub fn alloc(size: usize) -> Result<Self, CudartError> {
            let words = size.div_ceil(std::mem::size_of::<u64>()).max(1);
            let mut storage = vec![0_u64; words].into_boxed_slice();
            let ptr = NonNull::new(storage.as_mut_ptr().cast::<c_void>()).ok_or(
                CudartError { detail: "host allocation returned a null pointer" },
            )?;
            Ok(Self { storage, ptr, size })
        }

        pub fn get_ref(&self, index: usize) -> &u64 {
            &self.storage[index]
        }

        pub fn get_mut(&mut self, index: usize) -> &mut u64 {
            &mut self.storage[index]
        }
    }

    unsafe impl Send for CudaHostMemory {}
    unsafe impl Sync for CudaHostMemory {}

    pub mod driver {
        pub use super::{CudaDriverError, cu_get_dma_buf_fd};
    }

    pub mod gdr {
        pub use super::GdrFlag;
    }

    pub mod rt {
        pub use super::{CudartError, cudaMemoryTypeDevice, cudaPointerGetAttributes};
    }
}

#[cfg(not(feature = "cuda"))]
pub use host_only::*;

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::CudaHostMemory;

    #[test]
    fn host_memory_is_available_without_cuda() {
        let mut memory = CudaHostMemory::alloc(16).unwrap();
        assert_eq!(*memory.get_ref(0), 0);
        *memory.get_mut(1) = 42;
        assert_eq!(*memory.get_ref(1), 42);
    }
}
