//! This module provides conversions between `Arc<Mutex<Array<T, D>>>` and
//! `Arc<RwLock<Array<T, D>>>` to `DLPackTensor`, allowing you to share data
//! between Rust and DLPack while ensuring that the data is not modified by Rust
//! while it's used by DLPack and reciprocally. The locks will be held until the
//! `DLPackTensor` is dropped, ensuring safe access to the data.
//!
//! The following conversions to DLPack types are supported:
//!
//! - `Arc<Mutex<Array<T, D>>> -> DLPackTensor`
//! - `ReadOnly<Arc<RwLock<Array<T, D>>>> -> DLPackTensor`, creating a read-only
//!   DLPack tensor and locking the `RwLock` for reading.
//! - `ReadWrite<Arc<RwLock<Array<T, D>>>> -> DLPackTensor`, creating a
//!   read-write DLPack tensor and locking the `RwLock` for writing.

use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockWriteGuard, RwLockReadGuard};
use ndarray::{Array, Dimension};

use ouroboros::self_referencing;

use crate::sys;
use crate::{DLPackTensor, GetDLPackDataType};

use crate::{ReadOnly, ReadWrite};

use super::DLPackNDarrayError;


#[self_referencing]
struct RwLockCtxWrite<Array> where Array: 'static {
    array: Arc<RwLock<Array>>,
    #[borrows(array)]
    #[covariant]
    lock: RwLockWriteGuard<'this, Array>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

#[self_referencing]
struct RwLockCtxRead<Array> where Array: 'static {
    array: Arc<RwLock<Array>>,
    #[borrows(array)]
    #[covariant]
    lock: RwLockReadGuard<'this, Array>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn rwlock_write_deleter_fn<T>(tensor: *mut sys::DLManagedTensorVersioned) where T: 'static {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*tensor).manager_ctx.cast::<RwLockCtxWrite<T>>();
    let _ = Box::from_raw(ctx);

    // also drop the tensor itself
    let _ = Box::from_raw(tensor);
}

unsafe extern "C" fn rwlock_read_deleter_fn<T>(tensor: *mut sys::DLManagedTensorVersioned) where T: 'static {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*tensor).manager_ctx.cast::<RwLockCtxRead<T>>();
    let _ = Box::from_raw(ctx);

    // also drop the tensor itself
    let _ = Box::from_raw(tensor);
}

impl<T, D> TryFrom<ReadWrite<Arc<RwLock<Array<T, D>>>>> for DLPackTensor
where
    D: Dimension + 'static,
    T: GetDLPackDataType + 'static + Clone,
{
    type Error = DLPackNDarrayError;

    fn try_from(ReadWrite(array): ReadWrite<Arc<RwLock<Array<T, D>>>>) -> Result<Self, Self::Error> {
        let empty = array.read().expect("could not aquire lock").is_empty();

        let ctx = RwLockCtxWriteBuilder {
            array: array,
            lock_builder: move |array| { array.try_write().expect("could not lock the rwlock") },
            shape: vec![],
            strides: vec![],
        };
        let mut ctx = Box::new(ctx.build());

        // set the shape after acquiring the lock to avoid deadlocks
        let (shape, strides) = ctx.with_lock(|lock| {
            let shape: Vec<_> = lock.shape().iter().map(|&s| s as i64).collect();
            let strides: Vec<_> = lock.strides().iter().map(|&s| s as i64).collect();
            (shape, strides)
        });
        let ndim = shape.len() as i32;

        ctx.with_shape_mut(|v| *v = shape);
        ctx.with_strides_mut(|v| *v = strides);

        // extract pointers out of the boxed context to use in the DLPack tensor
        let mut shape_ptr = std::ptr::null_mut();
        ctx.with_shape_mut(|shape| {
            shape_ptr = shape.as_mut_ptr();
        });

        let mut stride_ptr = std::ptr::null_mut();
        ctx.with_strides_mut(|strides| {
            stride_ptr = strides.as_mut_ptr();
        });

        let mut data = std::ptr::null_mut();
        if !empty {
            ctx.with_lock_mut(|lock| {
                // We can give out a mutable pointer to the data because the lock
                // will be held until the `DLPackTensor` is dropped, so the data
                // won't be modified by Rust while it's used by DLPack.
                data = lock.as_mut_ptr().cast()
            });
        }

        let dl_tensor = sys::DLTensor {
            data: data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: ndim,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(rwlock_write_deleter_fn::<Array<T, D>>),
            flags: 0,
            dl_tensor,
        });

        unsafe {
            Ok(DLPackTensor::from_ptr(Box::into_raw(managed_tensor)))
        }
    }
}

impl<T, D> TryFrom<ReadOnly<Arc<RwLock<Array<T, D>>>>> for DLPackTensor
where
    D: Dimension + 'static,
    T: GetDLPackDataType + 'static + Clone,
{
    type Error = DLPackNDarrayError;

    fn try_from(ReadOnly(array): ReadOnly<Arc<RwLock<Array<T, D>>>>) -> Result<Self, Self::Error> {
        let empty = array.read().expect("could not aquire lock").is_empty();

        let ctx = RwLockCtxReadBuilder {
            array: array,
            lock_builder: move |array| { array.try_read().expect("could not lock the rwlock") },
            shape: vec![],
            strides: vec![],
        };
        let mut ctx = Box::new(ctx.build());

        // set the shape after acquiring the lock to avoid deadlocks
        let (shape, strides) = ctx.with_lock(|lock| {
            let shape: Vec<_> = lock.shape().iter().map(|&s| s as i64).collect();
            let strides: Vec<_> = lock.strides().iter().map(|&s| s as i64).collect();
            (shape, strides)
        });
        let ndim = shape.len() as i32;

        ctx.with_shape_mut(|v| *v = shape);
        ctx.with_strides_mut(|v| *v = strides);

        // extract pointers out of the boxed context to use in the DLPack tensor
        let mut shape_ptr = std::ptr::null_mut();
        ctx.with_shape_mut(|shape| {
            shape_ptr = shape.as_mut_ptr();
        });

        let mut stride_ptr = std::ptr::null_mut();
        ctx.with_strides_mut(|strides| {
            stride_ptr = strides.as_mut_ptr();
        });

        let mut data = std::ptr::null_mut();
        if !empty {
            ctx.with_lock_mut(|lock| {
                // Cast mut is fine, we set the read-only flag in the DLPack tensor
                data = lock.as_ptr().cast_mut().cast()
            });
        }

        let dl_tensor = sys::DLTensor {
            data: data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: ndim,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(rwlock_read_deleter_fn::<Array<T, D>>),
            flags: 0,
            dl_tensor,
        });

        unsafe {
            Ok(DLPackTensor::from_ptr(Box::into_raw(managed_tensor)))
        }
    }
}


#[self_referencing]
struct MutexCtx<Array> where Array: 'static {
    array: Arc<Mutex<Array>>,
    #[borrows(array)]
    #[covariant]
    lock: MutexGuard<'this, Array>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn mutex_deleter_fn<T>(tensor: *mut sys::DLManagedTensorVersioned) where T: 'static {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*tensor).manager_ctx.cast::<MutexCtx<T>>();
    let _ = Box::from_raw(ctx);

    // also drop the tensor itself
    let _ = Box::from_raw(tensor);
}

impl<T, D> TryFrom<Arc<Mutex<Array<T, D>>>> for DLPackTensor
where
    D: Dimension + 'static,
    T: GetDLPackDataType + 'static + Clone,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: Arc<Mutex<Array<T, D>>>) -> Result<Self, Self::Error> {
        let empty = array.lock().expect("could not aquire lock").is_empty();

        let ctx = MutexCtxBuilder {
            array: array,
            lock_builder: move |array| { array.try_lock().expect("could not lock the mutex") },
            shape: vec![],
            strides: vec![],
        };
        let mut ctx = Box::new(ctx.build());

        // set the shape after acquiring the lock to avoid deadlocks
        let (shape, strides) = ctx.with_lock(|lock| {
            let shape: Vec<_> = lock.shape().iter().map(|&s| s as i64).collect();
            let strides: Vec<_> = lock.strides().iter().map(|&s| s as i64).collect();
            (shape, strides)
        });
        let ndim = shape.len() as i32;

        ctx.with_shape_mut(|v| *v = shape);
        ctx.with_strides_mut(|v| *v = strides);

        // extract pointers out of the boxed context to use in the DLPack tensor
        let mut shape_ptr = std::ptr::null_mut();
        ctx.with_shape_mut(|shape| {
            shape_ptr = shape.as_mut_ptr();
        });

        let mut stride_ptr = std::ptr::null_mut();
        ctx.with_strides_mut(|strides| {
            stride_ptr = strides.as_mut_ptr();
        });

        let mut data = std::ptr::null_mut();
        if !empty {
            ctx.with_lock_mut(|lock| {
                // We can give out a mutable pointer to the data because the lock
                // will be held until the `DLPackTensor` is dropped, so the data
                // won't be modified by Rust while it's used by DLPack.
                data = lock.as_mut_ptr().cast()
            });
        }

        let dl_tensor = sys::DLTensor {
            data: data,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: ndim,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: stride_ptr,
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(mutex_deleter_fn::<Array<T, D>>),
            flags: 0,
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
    use ndarray::{ArrayViewD, ArrayViewMutD, arr2};

    #[test]
    fn test_mutex() {
        let array = Arc::new(Mutex::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));

        {
            let mut tensor: DLPackTensor = Arc::clone(&array).try_into().unwrap();
            assert!(array.try_lock().is_err(), "the mutex should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let mut view: ArrayViewMutD<f64> = tensor_mut_ref.try_into().unwrap();

            view[[1, 1]] = 42.0;
        }

        let array = array.try_lock().unwrap();
        assert_eq!(*array, arr2(&[[1.0, 2.0, 3.0], [4.0, 42.0, 6.0]]));
    }

    #[test]
    fn test_rwlock_write() {
        let array = Arc::new(RwLock::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));

        {
            let mut tensor: DLPackTensor = ReadWrite(Arc::clone(&array)).try_into().unwrap();
            assert!(array.try_read().is_err(), "the rwlock should be locked while the DLPackTensor exists");
            assert!(array.try_write().is_err(), "the rwlock should be locked while the DLPackTensor exists");

            let tensor_mut_ref = tensor.as_mut();
            let mut view: ArrayViewMutD<f64> = tensor_mut_ref.try_into().unwrap();

            view[[1, 1]] = 42.0;
        }

        let array = array.try_read().unwrap();
        assert_eq!(*array, arr2(&[[1.0, 2.0, 3.0], [4.0, 42.0, 6.0]]));
    }

    // These tests exercise the deleter with the last Arc reference, so the
    // ManagerContext is actually deallocated (not just the Arc refcount
    // decremented). Without the correct type parameter on the deleter function,
    // miri catches an incorrect layout on deallocation.

    #[test]
    fn test_mutex_drop() {
        let array = Arc::new(Mutex::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));

        let mut tensor: DLPackTensor = array.try_into().unwrap();
        let tensor_mut_ref = tensor.as_mut();
        let view: ArrayViewMutD<f64> = tensor_mut_ref.try_into().unwrap();
        assert_eq!(view[[0, 2]], 3.0);
    }

    #[test]
    fn test_rwlock_write_drop() {
        let array = Arc::new(RwLock::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));

        let mut tensor: DLPackTensor = ReadWrite(Arc::clone(&array)).try_into().unwrap();
        let tensor_mut_ref = tensor.as_mut();
        let view: ArrayViewMutD<f64> = tensor_mut_ref.try_into().unwrap();
        assert_eq!(view[[0, 2]], 3.0);
    }

    #[test]
    fn test_rwlock_read_drop() {
        let array = Arc::new(RwLock::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));

        let tensor: DLPackTensor = ReadOnly(Arc::clone(&array)).try_into().unwrap();
        let tensor_ref = tensor.as_ref();
        let view: ArrayViewD<f64> = tensor_ref.try_into().unwrap();
        assert_eq!(view[[0, 2]], 3.0);
    }

    #[test]
    fn empty_mutex_to_dlpack() {
        let array = Arc::new(Mutex::new(ndarray::Array3::<f64>::from_shape_vec([0, 0, 0], vec![]).unwrap()));
        let tensor: DLPackTensor = Arc::clone(&array).try_into().unwrap();
        assert_eq!(tensor.shape(), &[0, 0, 0]);
        assert!(tensor.as_dltensor().data.is_null());
    }

    #[test]
    fn empty_rwlock_write_to_dlpack() {
        let array = Arc::new(RwLock::new(ndarray::Array3::<f64>::from_shape_vec([0, 0, 0], vec![]).unwrap()));
        let tensor: DLPackTensor = ReadWrite(Arc::clone(&array)).try_into().unwrap();
        assert_eq!(tensor.shape(), &[0, 0, 0]);
        assert!(tensor.as_dltensor().data.is_null());
    }

    #[test]
    fn empty_rwlock_read_to_dlpack() {
        let array = Arc::new(RwLock::new(ndarray::Array3::<f64>::from_shape_vec([0, 0, 0], vec![]).unwrap()));
        let tensor: DLPackTensor = ReadOnly(Arc::clone(&array)).try_into().unwrap();
        assert_eq!(tensor.shape(), &[0, 0, 0]);
        assert!(tensor.as_dltensor().data.is_null());
    }
}
