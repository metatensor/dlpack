//! Conversion of standard `Vec` and slices to and from dlpack.
//!
//! The following conversions from DLPack types are supported:
//!
//! - DLPackTensor -> Vec<T>, making a copy of the data
//! - DLPackTensor -> Box<[T]>, making a copy of the data
//! - DLPackTensorRef -> &[T]
//! - DLPackTensorRefMut -> &mut [T]
//!
//! The following conversions to DLPack types are supported:
//!
//! - Vec<T> -> DLPackTensor, creating a DLPack tensor which owns its data.
//! - Box<[T]> -> DLPackTensor, creating a DLPack tensor which owns its data.

use crate::sys;
use crate::{DLPackTensor, DLPackTensorRef, DLPackTensorRefMut};
use crate::{CastError, DLPackPointerCast, GetDLPackDataType};

/// Possible error causes when converting between Vec/slice and DLPack
#[derive(Debug)]
pub enum DLPackVecError {
    /// We only support data which lives on CPU or host-accessible memory
    DeviceShouldBeCpu(sys::DLDevice),
    /// The DLPack type can not be converted to a supported Rust type
    InvalidType(CastError),
    /// The shape/stride of the data does not match expectations
    ShapeError(Vec<i64>),
}

impl std::fmt::Display for DLPackVecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DLPackVecError::DeviceShouldBeCpu(device) => {
                write!(f, "can not convert from device {} (only CPU, CUDA host, and ROCm host are supported)", device)
            }
            DLPackVecError::InvalidType(error) => {
                write!(f, "type conversion error: {}", error)
            }
            DLPackVecError::ShapeError(shape) => {
                write!(f, "shape error, expected a 1D array, got shape: {:?}", shape)
            }
        }
    }
}

impl std::error::Error for DLPackVecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DLPackVecError::DeviceShouldBeCpu(_) => None,
            DLPackVecError::InvalidType(e) => Some(e),
            DLPackVecError::ShapeError(_) => None,
        }
    }
}

impl From<CastError> for DLPackVecError {
    fn from(value: CastError) -> Self {
        DLPackVecError::InvalidType(value)
    }
}

/// Check whether a device type has host-accessible memory, i.e. the data
/// pointer can be dereferenced from the CPU without an explicit copy.
fn is_host_accessible(device_type: sys::DLDeviceType) -> bool {
    matches!(
        device_type,
        sys::DLDeviceType::kDLCPU
        | sys::DLDeviceType::kDLCUDAHost
        | sys::DLDeviceType::kDLROCMHost
    )
}

/// Check that the DLPackTensor is on CPU, and return the size of the vector.
fn check_tensor(tensor: &DLPackTensorRef<'_>) -> Result<usize, DLPackVecError> {
    if !is_host_accessible(tensor.device().device_type) {
        return Err(DLPackVecError::DeviceShouldBeCpu(tensor.device()));
    }

    let shape = tensor.shape();

    if shape.len() != 1 {
        return Err(DLPackVecError::ShapeError(tensor.shape().to_vec()));
    }

    Ok(shape[0] as usize)
}

impl<T> TryFrom<DLPackTensor> for Vec<T>  where T: DLPackPointerCast + Clone {
    type Error = DLPackVecError;

    fn try_from(value: DLPackTensor) -> Result<Self, Self::Error> {
        let size = check_tensor(&value.as_ref())?;
        let ptr = value.data_ptr::<T>()?;

        let slice = if size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(ptr, size) }
        };

        return Ok(slice.to_vec());
    }
}

impl<T> TryFrom<DLPackTensor> for Box<[T]>  where T: DLPackPointerCast + Clone {
    type Error = DLPackVecError;

    fn try_from(value: DLPackTensor) -> Result<Self, Self::Error> {
        let size = check_tensor(&value.as_ref())?;
        let ptr = value.data_ptr::<T>()?;

        let slice = if size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(ptr, size) }
        };

        return Ok(slice.into());
    }
}

impl<'a, T> TryFrom<DLPackTensorRef<'a>> for &'a [T] where T: DLPackPointerCast {
    type Error = DLPackVecError;

    fn try_from(value: DLPackTensorRef<'a>) -> Result<Self, Self::Error> {
        let size = check_tensor(&value)?;
        let ptr = value.data_ptr::<T>()?;

        if size == 0 {
            return Ok(&[]);
        } else {
            let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
            return Ok(slice);
        }
    }
}


impl<'a, T> TryFrom<DLPackTensorRefMut<'a>> for &'a mut [T] where T: DLPackPointerCast {
    type Error = DLPackVecError;

    fn try_from(mut value: DLPackTensorRefMut<'a>) -> Result<Self, Self::Error> {
        let size = check_tensor(&value.as_ref())?;
        let ptr = value.data_ptr_mut::<T>()?;

        if size == 0 {
            return Ok(&mut []);
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
            return Ok(slice);
        }
    }
}

struct ManagerContext<T> {
    array: T,
    shape: Box<i64>,
    stride: Box<i64>,
}

unsafe extern "C" fn deleter_fn<T>(tensor: *mut sys::DLManagedTensorVersioned) {
    unsafe {
        // Reconstruct the box and drop it, freeing the memory.
        let ctx = (*tensor).manager_ctx.cast::<ManagerContext<T>>();
        let _ = Box::from_raw(ctx);

        // also drop the tensor itself
        let _ = Box::from_raw(tensor);
    }
}

macro_rules! impl_try_from {
    ($Type: ty) => {
        impl<T> TryFrom<$Type> for DLPackTensor where T: GetDLPackDataType {
            type Error = DLPackVecError;

            fn try_from(value: $Type) -> Result<DLPackTensor, Self::Error> {
                let len = value.len();
                let mut ctx = Box::new(ManagerContext {
                    array: value,
                    shape: Box::new(len as i64),
                    stride: Box::new(1),
                });

                let shape_ptr = ctx.shape.as_mut();
                let stride_ptr = ctx.stride.as_mut();

                let data = if ctx.array.is_empty() {
                    std::ptr::null_mut()
                } else {
                    ctx.array.as_ptr()
                };

                let dl_tensor = sys::DLTensor {
                    data: data.cast_mut().cast(),
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
                    deleter: Some(deleter_fn::<$Type>),
                    flags: sys::DLPACK_FLAG_BITMASK_IS_COPIED,
                    dl_tensor,
                });

                unsafe {
                    Ok(DLPackTensor::from_ptr(Box::into_raw(managed_tensor)))
                }
            }
        }
    };
}

impl_try_from!(Vec<T>);
impl_try_from!(Box<[T]>);

#[cfg(test)]
#[cfg(feature = "ndarray")]
mod tests {
    use super::*;

    use ndarray::Array1;

    #[test]
    fn vec_to_ndarray() {
        let data = vec![1, 2, 3, 4, 5];

        let tensor: DLPackTensor = data.try_into().unwrap();
        let array: Array1<i32> = tensor.try_into().unwrap();

        assert_eq!(array, ndarray::arr1(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn boxed_slice_to_ndarray() {
        let data: Box<[u32]> = Box::new([1, 2, 3, 4, 5]);

        let tensor: DLPackTensor = data.try_into().unwrap();
        let array: Array1<u32> = tensor.try_into().unwrap();

        assert_eq!(array, ndarray::arr1(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn ndarray_to_vec() {
        let data = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);

        let tensor: DLPackTensor = data.try_into().unwrap();
        let vec: Vec<f64> = tensor.try_into().unwrap();
        assert_eq!(vec, [1.0, 2.0, 3.0, 4.0]);

        let data = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
        let mut tensor: DLPackTensor = data.try_into().unwrap();

        {
            let tensor_ref = tensor.as_ref();
            let slice: &[f64] = tensor_ref.try_into().unwrap();
            assert_eq!(slice, [1.0, 2.0, 3.0, 4.0]);
        }

        let tensor_mut = tensor.as_mut();
        let slice: &mut [f64] = tensor_mut.try_into().unwrap();
        assert_eq!(slice, [1.0, 2.0, 3.0, 4.0]);

        let data = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
        let data = data.into_shape_with_order((2, 2)).unwrap();
        let tensor: DLPackTensor = data.try_into().unwrap();

        let err = TryInto::<Vec<f64>>::try_into(tensor).unwrap_err();
        match err {
             DLPackVecError::ShapeError(shape) => {
                assert_eq!(shape, [2, 2]);
            }
            _ => panic!("unexpected error: {}", err),
        }
    }

    #[test]
    fn empty_vec_to_dlpack() {
        let data: Vec<i32> = Vec::new();
        let tensor: DLPackTensor = data.try_into().unwrap();
        assert_eq!(tensor.shape(), &[0]);
        assert!(tensor.as_dltensor().data.is_null());
    }

    #[test]
    fn empty_dlpack_to_vec_ref() {
        let mut shape = vec![0i64];
        let mut strides = vec![1i64];

        let dl_tensor = crate::sys::DLTensor {
            data: std::ptr::null_mut(),
            device: crate::sys::DLDevice {
                device_type: crate::sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: i32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        unsafe {
            let tensor_ref = DLPackTensorRef::from_raw(dl_tensor);
            let slice: &[i32] = tensor_ref.try_into().unwrap();
            assert!(slice.is_empty());
        }
    }

    unsafe extern "C" fn box_deleter(tensor: *mut sys::DLManagedTensorVersioned) {
        unsafe {
            let _ = Box::from_raw(tensor);
        }
    }

    #[test]
    fn empty_dlpack_to_vec_owned() {
        let mut shape = vec![0i64];
        let mut strides = vec![1i64];

        let dl_tensor = crate::sys::DLTensor {
            data: std::ptr::null_mut(),
            device: crate::sys::DLDevice {
                device_type: crate::sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: i32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let managed = Box::new(crate::sys::DLManagedTensorVersioned {
            version: crate::sys::DLPackVersion::current(),
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(box_deleter),
            flags: 0,
            dl_tensor,
        });

        let tensor = unsafe { DLPackTensor::from_ptr(Box::into_raw(managed)) };
        let vec: Vec<i32> = tensor.try_into().unwrap();
        assert!(vec.is_empty());
    }

    #[test]
    fn scalar_vec_to_dlpack() {
        let data = vec![42i32];
        let tensor: DLPackTensor = data.try_into().unwrap();
        assert_eq!(tensor.shape(), &[1]);
        assert!(!tensor.as_dltensor().data.is_null());
    }

    #[test]
    fn scalar_dlpack_to_vec() {
        let mut shape = vec![1i64];
        let mut strides = vec![1i64];
        let mut value = 42f64;

        let dl_tensor = crate::sys::DLTensor {
            data: (&mut value as *mut f64).cast(),
            device: crate::sys::DLDevice {
                device_type: crate::sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: f64::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let managed = Box::new(crate::sys::DLManagedTensorVersioned {
            version: crate::sys::DLPackVersion::current(),
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(box_deleter),
            flags: 0,
            dl_tensor,
        });

        let tensor = unsafe { DLPackTensor::from_ptr(Box::into_raw(managed)) };
        let vec: Vec<f64> = tensor.try_into().unwrap();
        assert_eq!(vec, vec![42.0]);
    }
}
