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
    /// We only support data which lives on CPU
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
                write!(f, "can not convert from device {} (only cpu is supported)", device)
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

/// Check that the DLPackTensor is on CPU, and return the size of the vector.
fn check_tensor(tensor: &DLPackTensorRef<'_>) -> Result<usize, DLPackVecError> {
    if tensor.device().device_type != sys::DLDeviceType::kDLCPU {
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

        let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
        return Ok(slice.to_vec());
    }
}

impl<T> TryFrom<DLPackTensor> for Box<[T]>  where T: DLPackPointerCast + Clone {
    type Error = DLPackVecError;

    fn try_from(value: DLPackTensor) -> Result<Self, Self::Error> {
        let size = check_tensor(&value.as_ref())?;
        let ptr = value.data_ptr::<T>()?;

        let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
        return Ok(slice.into());
    }
}

impl<'a, T> TryFrom<DLPackTensorRef<'a>> for &'a [T] where T: DLPackPointerCast {
    type Error = DLPackVecError;

    fn try_from(value: DLPackTensorRef<'a>) -> Result<Self, Self::Error> {
        let size = check_tensor(&value)?;
        let ptr = value.data_ptr::<T>()?;

        let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
        return Ok(slice);
    }
}


impl<'a, T> TryFrom<DLPackTensorRefMut<'a>> for &'a mut [T] where T: DLPackPointerCast {
    type Error = DLPackVecError;

    fn try_from(mut value: DLPackTensorRefMut<'a>) -> Result<Self, Self::Error> {
        let size = check_tensor(&value.as_ref())?;
        let ptr = value.data_ptr_mut::<T>()?;

        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        return Ok(slice);
    }
}

struct ManagerContext<T> {
    array: T,
    shape: i64,
    stride: i64,
}

unsafe extern "C" fn deleter_fn<T>(manager: *mut sys::DLManagedTensorVersioned) {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*manager).manager_ctx.cast::<ManagerContext<T>>();
    let _ = Box::from_raw(ctx);
}

macro_rules! impl_try_from {
    ($Type: ty) => {
        impl<T> TryFrom<$Type> for DLPackTensor where T: GetDLPackDataType {
            type Error = DLPackVecError;

            fn try_from(value: $Type) -> Result<DLPackTensor, Self::Error> {
                let len = value.len();
                let mut ctx = Box::new(ManagerContext {
                    array: value,
                    shape: len as i64,
                    stride: 1,
                });

                let shape_ptr = &mut ctx.shape;
                let stride_ptr = &mut ctx.stride;

                let dl_tensor = sys::DLTensor {
                    data: ctx.array.as_ptr().cast_mut().cast(),
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

                let managed_tensor = sys::DLManagedTensorVersioned {
                    version: sys::DLPackVersion::current(),
                    manager_ctx: Box::into_raw(ctx).cast(),
                    deleter: Some(deleter_fn::<$Type>),
                    flags: sys::DLPACK_FLAG_BITMASK_IS_COPIED,
                    dl_tensor,
                };

                unsafe {
                    Ok(DLPackTensor::from_raw(managed_tensor))
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
}
