//! This module provides conversions between `Arc<Mutex<Vec<T>>>` and
//! `Arc<RwLock<Vec<T>>>` to `DLPackTensor`, allowing you to share data between
//! Rust and DLPack while ensuring that the data is not modified by Rust while
//! it's used by DLPack and reciprocally. The locks will be held until the
//! `DLPackTensor` is dropped, ensuring safe access to the data.
//!
//! The following conversions to DLPack types are supported:
//!
//! - `Arc<Mutex<Vec<T>>> -> DLPackTensor`
//! - `ReadOnly<Arc<RwLock<Vec<T>>>> -> DLPackTensor`, creating a read-only
//!   DLPack tensor and locking the RwLock for reading
//! - `ReadWrite<Arc<RwLock<Vec<T>>>> -> DLPackTensor`, creating a read-write
//!   DLPack tensor and locking the RwLock for writing

use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockWriteGuard, RwLockReadGuard};

use ouroboros::self_referencing;

use crate::sys::{self, DLPACK_FLAG_BITMASK_READ_ONLY};
use crate::{DLPackTensor, GetDLPackDataType};

use crate::vec::DLPackVecError;

use crate::{ReadOnly, ReadWrite};

#[self_referencing]
struct MutexCtx<T> where T: 'static {
    array: Arc<Mutex<Vec<T>>>,
    #[borrows(array)]
    #[covariant]
    lock: MutexGuard<'this, Vec<T>>,
    // Use Box<i64> so that pointers derived via with_*_mut target heap
    // memory rather than inline struct fields. This avoids Stacked Borrows
    // violations when multiple with_*_mut calls each create exclusive
    // reborrows of the ouroboros struct.
    shape: Box<i64>,
    stride: Box<i64>,
}

unsafe extern "C" fn mutex_deleter_fn<T>(tensor: *mut sys::DLManagedTensorVersioned) where T: 'static {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*tensor).manager_ctx.cast::<MutexCtx<T>>();
    let _ = Box::from_raw(ctx);

    // also drop the tensor itself
    let _ = Box::from_raw(tensor);
}

impl<T> TryFrom<Arc<Mutex<Vec<T>>>> for DLPackTensor where T: GetDLPackDataType + 'static {
    type Error = DLPackVecError;

    fn try_from(array: Arc<Mutex<Vec<T>>>) -> Result<DLPackTensor, Self::Error> {
        let ctx = MutexCtxBuilder {
            array: array,
            lock_builder: |array| { array.try_lock().expect("could not lock the mutex") },
            shape: Box::new(0),
            stride: Box::new(1),
        };
        let mut ctx = Box::new(ctx.build());

        // set the shape after acquiring the lock to avoid deadlocks
        let shape = ctx.with_lock(|lock| lock.len() as i64);
        ctx.with_shape_mut(|v| **v = shape);

        // extract pointers out of the boxed context to use in the DLPack tensor
        let mut shape_ptr = std::ptr::null_mut();
        ctx.with_shape_mut(|shape| {
            shape_ptr = shape.as_mut();
        });

        let mut stride_ptr = std::ptr::null_mut();
        ctx.with_stride_mut(|stride| {
            stride_ptr = stride.as_mut();
        });

        let mut data = std::ptr::null_mut();
        ctx.with_lock_mut(|lock| {
            // We can give out a mutable pointer to the data because the lock
            // will be held until the `DLPackTensor` is dropped, so the data
            // won't be modified by Rust while it's used by DLPack.
            data = lock.as_mut_ptr().cast()
        });

        let dl_tensor = sys::DLTensor {
            data: data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(mutex_deleter_fn::<T>),
            flags: 0,
            dl_tensor,
        });

        unsafe {
            Ok(DLPackTensor::from_ptr(Box::into_raw(managed_tensor)))
        }
    }
}

#[self_referencing]
struct RwLockCtxRead<T> where T: 'static {
    array: Arc<RwLock<Vec<T>>>,
    #[borrows(array)]
    #[covariant]
    lock: RwLockReadGuard<'this, Vec<T>>,
    shape: Box<i64>,
    stride: Box<i64>,
}

#[self_referencing]
struct RwLockCtxWrite<T> where T: 'static {
    array: Arc<RwLock<Vec<T>>>,
    #[borrows(array)]
    #[covariant]
    lock: RwLockWriteGuard<'this, Vec<T>>,
    shape: Box<i64>,
    stride: Box<i64>,
}

unsafe extern "C" fn rwlock_read_deleter_fn<T>(tensor: *mut sys::DLManagedTensorVersioned) where T: 'static {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*tensor).manager_ctx.cast::<RwLockCtxRead<T>>();
    let _ = Box::from_raw(ctx);

    // also drop the tensor itself
    let _ = Box::from_raw(tensor);
}

unsafe extern "C" fn rwlock_write_deleter_fn<T>(tensor: *mut sys::DLManagedTensorVersioned) where T: 'static {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*tensor).manager_ctx.cast::<RwLockCtxWrite<T>>();
    let _ = Box::from_raw(ctx);

    // also drop the tensor itself
    let _ = Box::from_raw(tensor);
}

impl<T> TryFrom<ReadWrite<Arc<RwLock<Vec<T>>>>> for DLPackTensor where T: GetDLPackDataType + 'static {
    type Error = DLPackVecError;

    fn try_from(ReadWrite(array): ReadWrite<Arc<RwLock<Vec<T>>>>) -> Result<DLPackTensor, Self::Error> {
        let ctx = RwLockCtxWriteBuilder {
            array: array,
            lock_builder: move |array| { array.try_write().expect("could not lock the rwlock") },
            shape: Box::new(0),
            stride: Box::new(1),
        };
        let mut ctx = Box::new(ctx.build());

        // set the shape after acquiring the lock to avoid deadlocks
        let shape = ctx.with_lock(|lock| lock.len() as i64);
        ctx.with_shape_mut(|v| **v = shape);

        // extract pointers out of the boxed context to use in the DLPack tensor
        let mut shape_ptr = std::ptr::null_mut();
        ctx.with_shape_mut(|shape| {
            shape_ptr = shape.as_mut();
        });

        let mut stride_ptr = std::ptr::null_mut();
        ctx.with_stride_mut(|stride| {
            stride_ptr = stride.as_mut();
        });

        let mut data = std::ptr::null_mut();
        ctx.with_lock_mut(|lock| {
            // We can give out a mutable pointer to the data because the lock
            // will be held until the `DLPackTensor` is dropped, so the data
            // won't be modified by Rust while it's used by DLPack.
            data = lock.as_mut_ptr().cast()
        });

        let dl_tensor = sys::DLTensor {
            data: data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(rwlock_write_deleter_fn::<T>),
            flags: 0,
            dl_tensor,
        });

        unsafe {
            Ok(DLPackTensor::from_ptr(Box::into_raw(managed_tensor)))
        }
    }
}

impl<T> TryFrom<ReadOnly<Arc<RwLock<Vec<T>>>>> for DLPackTensor where T: GetDLPackDataType + 'static {
    type Error = DLPackVecError;

    fn try_from(ReadOnly(array): ReadOnly<Arc<RwLock<Vec<T>>>>) -> Result<DLPackTensor, Self::Error> {
        let ctx = RwLockCtxReadBuilder {
            array: array,
            lock_builder: move |array| { array.try_read().expect("could not lock the rwlock") },
            shape: Box::new(0),
            stride: Box::new(1),
        };
        let mut ctx = Box::new(ctx.build());

        // set the shape after acquiring the lock to avoid deadlocks
        let shape = ctx.with_lock(|lock| lock.len() as i64);
        ctx.with_shape_mut(|v| **v = shape);

        // extract pointers out of the boxed context to use in the DLPack tensor
        let mut shape_ptr = std::ptr::null_mut();
        ctx.with_shape_mut(|shape| {
            shape_ptr = shape.as_mut();
        });

        let mut stride_ptr = std::ptr::null_mut();
        ctx.with_stride_mut(|stride| {
            stride_ptr = stride.as_mut();
        });

        let mut data = std::ptr::null_mut();
        ctx.with_lock_mut(|lock| {
            // Cast mut is fine, we set the read-only flag in the DLPack tensor
            data = lock.as_ptr().cast_mut().cast()
        });

        let dl_tensor = sys::DLTensor {
            data: data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(rwlock_read_deleter_fn::<T>),
            flags: DLPACK_FLAG_BITMASK_READ_ONLY,
            dl_tensor,
        });

        unsafe {
            Ok(DLPackTensor::from_ptr(Box::into_raw(managed_tensor)))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutex() {
        let data = Arc::new(Mutex::new(vec![1, 2, 3]));

        {
            let mut tensor: DLPackTensor = Arc::clone(&data).try_into().unwrap();
            assert!(data.try_lock().is_err(), "the mutex should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let slice: &mut[i32] = tensor_mut_ref.try_into().unwrap();

            slice[1] = 42;
        }

        let lock = data.try_lock().unwrap();
        assert_eq!(&*lock, &[1, 42, 3]);
    }

    #[test]
    fn test_rwlock_write() {
        let data = Arc::new(RwLock::new(vec![1, 2, 3]));

        {
            let mut tensor: DLPackTensor = ReadWrite(Arc::clone(&data)).try_into().unwrap();
            assert!(data.try_read().is_err(), "the rwlock should be locked while the DLPackTensor exists");
            assert!(data.try_write().is_err(), "the rwlock should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let slice: &mut[i32] = tensor_mut_ref.try_into().unwrap();

            slice[1] = 42;
        }

        let lock = data.try_read().unwrap();
        assert_eq!(&*lock, &[1, 42, 3]);
    }

    #[test]
    fn test_rwlock_read() {
        let data = Arc::new(RwLock::new(vec![1, 2, 3]));

        {
            let tensor: DLPackTensor = ReadOnly(Arc::clone(&data)).try_into().unwrap();
            assert!(data.try_read().is_ok(), "the rwlock can be read while the DLPackTensor exists");
            assert!(data.try_write().is_err(), "the rwlock should be locked while the DLPackTensor exists");

            let tensor_ref = tensor.as_ref();
            let slice: &[i32] = tensor_ref.try_into().unwrap();
            assert_eq!(slice, &[1, 2, 3]);
        }

        let lock = data.try_read().unwrap();
        assert_eq!(&*lock, &[1, 2, 3]);
    }

    // Last-ref tests: the tensor holds the only Arc reference, so dropping
    // it actually deallocates the ManagerContext via the deleter function.

    #[test]
    fn test_mutex_drop() {
        let data = Arc::new(Mutex::new(vec![1i32, 2, 3]));

        let mut tensor: DLPackTensor = data.try_into().unwrap();
        let tensor_mut_ref = tensor.as_mut();
        let slice: &mut [i32] = tensor_mut_ref.try_into().unwrap();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_rwlock_write_drop() {
        let data = Arc::new(RwLock::new(vec![1i32, 2, 3]));

        let mut tensor: DLPackTensor = ReadWrite(Arc::clone(&data)).try_into().unwrap();
        let tensor_mut_ref = tensor.as_mut();
        let slice: &mut [i32] = tensor_mut_ref.try_into().unwrap();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_rwlock_read_drop() {
        let data = Arc::new(RwLock::new(vec![1i32, 2, 3]));

        let tensor: DLPackTensor = ReadOnly(Arc::clone(&data)).try_into().unwrap();
        let tensor_ref = tensor.as_ref();
        let slice: &[i32] = tensor_ref.try_into().unwrap();
        assert_eq!(slice, &[1, 2, 3]);
    }
}
