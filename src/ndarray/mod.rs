//! Conversion between DLPack and ndarray, this module requires the `ndarray`
//! feature to be enabled.
//!
//! The following conversions are supported:
//!
//! - `DLPackTensor` => `ndarray::Array` (makes a copy of the data)
//! - `DLPackTensor` => `ndarray::ArcArray` (makes a copy of the data)
//! - `DLPackTensorRef` => `ndarray::ArrayView`
//! - `DLPackTensorRefMut` => `ndarray::ArrayViewMut`
//! - `ndarray::Array` => `DLPackTensor`
//! - `&ndarray::Array` => `DLPackTensorRef`
//! - `&mut ndarray::Array` => `DLPackTensorRefMut`
//! - `ndarray::ArrayView` => `DLPackTensorRef`
//! - `ndarray::ArrayViewMut` => `DLPackTensorRefMut`
//! - `ndarray::ArcArray` => `DLPackTensor` (share data, but creates a read-only DLPackTensor)
//! - `&ndarray::ArcArray` => `DLPackTensorRef`
//!
//! # Examples
//!
//! ```no_run
//! use dlpk::{DLPackTensor, DLPackTensorRef};
//! # fn get_tensor_from_somewhere() -> DLPackTensor { unimplemented!() }
//!
//! let tensor: DLPackTensor = get_tensor_from_somewhere();
//!
//! // makes a copy of the data
//! let array: ndarray::ArrayD<f32> = tensor.try_into().unwrap();
//!
//! // no copy, share data with the original tensor
//! let tensor: DLPackTensor = get_tensor_from_somewhere();
//! let tensor_ref: DLPackTensorRef = tensor.as_ref();
//! let reference: ndarray::ArrayView2<f32> = tensor_ref.try_into().unwrap();
//!
//! // convert an ndarray array into a DLPack tensor
//! let array = ndarray::Array::from_elem((2, 3), 1.0f32);
//! let tensor: DLPackTensor = array.clone().try_into().unwrap();
//!
//! let tensor_ref: DLPackTensorRef = (&array).try_into().unwrap();
//! ```

#[cfg(feature = "sync")]
pub mod sync;

use ndarray::{Array, ArcArray, Dimension, ShapeBuilder};

use crate::sys;
use crate::{DLPackTensor, DLPackTensorRef, DLPackTensorRefMut};
use crate::{CastError, DLPackPointerCast, GetDLPackDataType};

#[cfg(feature = "pyo3")]
use pyo3::PyErr;

/// Possible error causes when converting between ndarray and DLPack
#[derive(Debug)]
pub enum DLPackNDarrayError {
    /// ndarray only support data which lives on CPU or host-accessible memory
    DeviceShouldBeCpu(sys::DLDevice),
    /// The DLPack type can not be converted to a supported Rust type
    InvalidType(CastError),
    /// The shape/stride of the data does not match expectations
    ShapeError(ndarray::ShapeError),
}

impl From<CastError> for DLPackNDarrayError {
    fn from(err: CastError) -> Self {
        DLPackNDarrayError::InvalidType(err)
    }
}

impl From<ndarray::ShapeError> for DLPackNDarrayError {
    fn from(err: ndarray::ShapeError) -> Self {
        DLPackNDarrayError::ShapeError(err)
    }
}

#[cfg(feature = "pyo3")]
impl From<DLPackNDarrayError> for PyErr {
    fn from(err: DLPackNDarrayError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}


impl std::fmt::Display for DLPackNDarrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DLPackNDarrayError::DeviceShouldBeCpu(device) => {
                write!(f, "can not convert from device {} (only CPU, CUDA host, and ROCm host are supported)", device)
            }
            DLPackNDarrayError::InvalidType(error) => {
                write!(f, "type conversion error: {}", error)
            }
            DLPackNDarrayError::ShapeError(error) => {
                write!(f, "shape error: {}", error)
            }
        }
    }
}

impl std::error::Error for DLPackNDarrayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DLPackNDarrayError::DeviceShouldBeCpu(_) => None,
            DLPackNDarrayError::InvalidType(err) => Some(err),
            DLPackNDarrayError::ShapeError(err) => Some(err),
        }
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

/*****************************************************************************/
/*                            DLPack => ndarray                              */
/*****************************************************************************/

impl<'a, T, D> TryFrom<DLPackTensorRef<'a>> for ndarray::ArrayView<'a, T, D> where
    T: DLPackPointerCast + 'static,
    D: DimFromVec + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(tensor: DLPackTensorRef<'a>) -> Result<Self, Self::Error> {
        if !is_host_accessible(tensor.device().device_type) {
            return Err(DLPackNDarrayError::DeviceShouldBeCpu(tensor.device()))
        }

        let ptr = tensor.data_ptr::<T>()?;
        let shape = tensor.shape().iter().map(|&s| s as usize).collect::<Vec<_>>();
        let shape = <D as DimFromVec>::dim_from_vec(shape)?;

        if shape.size() == 0 {
            // handle empty arrays
            return Ok(ndarray::ArrayView::from_shape(shape, &[])?);
        }

        let array = match tensor.strides() {
            Some(strides) =>{
                let s_vec = strides.iter().map(|&s| s as usize).collect::<Vec<_>>();
                let dim_strides = <D as DimFromVec>::dim_from_vec(s_vec)?;
                let shape = shape.strides(dim_strides);
                unsafe { ndarray::ArrayView::from_shape_ptr(shape, ptr) }
            }
            None => unsafe { ndarray::ArrayView::from_shape_ptr(shape, ptr) }
        };

        return Ok(array);
    }
}

impl<'a, T, D> TryFrom<DLPackTensorRefMut<'a>> for ndarray::ArrayViewMut<'a, T, D> where
    T: DLPackPointerCast + 'static,
    D: DimFromVec + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(mut tensor: DLPackTensorRefMut<'a>) -> Result<Self, Self::Error> {
        if !is_host_accessible(tensor.device().device_type) {
            return Err(DLPackNDarrayError::DeviceShouldBeCpu(tensor.device()))
        }

        let ptr = tensor.data_ptr_mut::<T>()?;
        let shape = tensor.shape().iter().map(|&s| s as usize).collect::<Vec<_>>();
        let shape = <D as DimFromVec>::dim_from_vec(shape)?;

        if shape.size() == 0 {
            // handle empty arrays
            return Ok(ndarray::ArrayViewMut::from_shape(shape, &mut [])?);
        }

        let array;
        if let Some(strides) = tensor.strides() {
            let strides = strides.iter().map(|&s| s as usize).collect::<Vec<_>>();
            let strides = <D as DimFromVec>::dim_from_vec(strides)?;
            let shape = shape.strides(strides);
            array = unsafe {
                ndarray::ArrayViewMut::<T, _>::from_shape_ptr(shape, ptr)
            };
        } else {
            array = unsafe {
                ndarray::ArrayViewMut::<T, _>::from_shape_ptr(shape, ptr)
            };
        }

        return Ok(array);
    }
}

/// This implementation provides a conversion from a DLPack `DLPackTensor` to an
/// `ndarray::Array`.
///
/// **Note:** This conversion makes a copy of the underlying tensor data. The
/// original DLPack tensor memory is released after the copy is complete.
impl<T, D> TryFrom<DLPackTensor> for Array<T, D>
where
    D: Dimension + DimFromVec + 'static,
    T: DLPackPointerCast + Clone + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(tensor: DLPackTensor) -> Result<Self, Self::Error> {
        let tensor_view = tensor.as_ref();
        let array_view: ndarray::ArrayView<T, D> = tensor_view.try_into()?;
        Ok(array_view.to_owned())
    }
}

/// This implementation provides a conversion from a DLPack `DLPackTensor` to an
/// `ndarray::ArcArray`.
///
/// **Note:** This conversion makes a copy of the underlying tensor data.
impl<T, D> TryFrom<DLPackTensor> for ArcArray<T, D>
where
    D: Dimension + DimFromVec + 'static,
    T: DLPackPointerCast + Clone + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(tensor: DLPackTensor) -> Result<Self, Self::Error> {
        let array: Array<T, D> = tensor.try_into()?;
        Ok(array.into())
    }
}

/*****************************************************************************/
/*                            ndarray => DLPack                              */
/*****************************************************************************/

fn array_to_tensor_view<'a, S, D, T>(array: &'a ndarray::ArrayBase<S, D>) -> Result<sys::DLTensor, DLPackNDarrayError> where
    D: ndarray::Dimension,
    S: ndarray::RawData<Elem = T>,
    T: GetDLPackDataType,
{
    // SAFETY: we make sure that shape and strides are valid for the lifetime of
    // the array
    let shape: &'a [_] = array.shape();
    let strides: &'a[_] = ndarray::ArrayBase::strides(array);

    // we need a `*const i64` for DLTensor, but we have usize and isize.
    // on 64-bit targets, isize will be the same as i64, so that's fine.
    if std::mem::size_of::<isize>() != std::mem::size_of::<i64>() {
        unimplemented!("DLPack conversion is only supported on 64-bit targets")
    }
    let strides = strides.as_ptr().cast_mut().cast();

    // usize will have the same binary representation as i64 for striclty
    // positive values, which is the most important case here.
    if std::mem::size_of::<isize>() != std::mem::size_of::<i64>() {
        unimplemented!("DLPack conversion is only supported on 64-bit targets")
    }
    let ndim = shape.len() as i32;
    let shape = shape.as_ptr().cast_mut().cast::<i64>();

    let device = sys::DLDevice {
        device_type: sys::DLDeviceType::kDLCPU,
        device_id: 0,
    };

    return Ok(sys::DLTensor {
        data: array.as_ptr().cast_mut().cast(),
        device: device,
        ndim: ndim,
        dtype: T::get_dlpack_data_type(),
        shape: shape,
        strides: strides,
        byte_offset: 0,
    });
}

impl<'a, T, D> TryFrom<&'a ndarray::ArrayView<'a, T, D>> for DLPackTensorRef<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a ndarray::ArrayView<'a, T, D>) -> Result<Self, Self::Error> {
        let tensor = array_to_tensor_view(array)?;

        return Ok(unsafe {
            // SAFETY: we are constraining the lifetime of the return value
            DLPackTensorRef::from_raw(tensor)
        });
    }
}

impl<'a, T, D> TryFrom<&'a ndarray::ArrayViewMut<'a, T, D>> for DLPackTensorRefMut<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a ndarray::ArrayViewMut<'a, T, D>) -> Result<Self, Self::Error> {
        let tensor = array_to_tensor_view(array)?;

        return Ok(unsafe {
            // SAFETY: we are constraining the lifetime of the return value, and
            // returning a mut ref from a mut ref
            DLPackTensorRefMut::from_raw(tensor)
        });
    }
}

impl<'a, T, D> TryFrom<&'a ndarray::Array<T, D>> for DLPackTensorRef<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        let tensor = array_to_tensor_view(array)?;

        return Ok(unsafe {
            // SAFETY: we are constraining the lifetime of the return value, and
            // returning a mut ref from a mut ref
            DLPackTensorRef::from_raw(tensor)
        });
    }
}

impl<'a, T, D> TryFrom<&'a ArcArray<T, D>> for DLPackTensorRef<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a ArcArray<T, D>) -> Result<Self, Self::Error> {
        let tensor = array_to_tensor_view(array)?;

        return Ok(unsafe {
            // SAFETY: we are constraining the lifetime of the return value
            DLPackTensorRef::from_raw(tensor)
        });
    }
}

impl<'a, T, D> TryFrom<&'a mut ndarray::Array<T, D>> for DLPackTensorRefMut<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a mut ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        let tensor = array_to_tensor_view(array)?;

        return Ok(unsafe {
            // SAFETY: we are constraining the lifetime of the return value, and
            // returning a mut ref from a mut ref
            DLPackTensorRefMut::from_raw(tensor)
        });
    }
}

/// Internal trait that will convert a `Vec<usize>` into one of ndarray's Dim
/// type.
pub trait DimFromVec where Self: ndarray::Dimension {
    fn dim_from_vec(vec: Vec<usize>) -> Result<Self, ndarray::ShapeError>;
}

macro_rules! impl_dim_for_vec_array {
    ($N: expr) => {
        impl DimFromVec for ndarray::Dim<[ndarray::Ix; $N]> {
            fn dim_from_vec(vec: Vec<usize>) -> Result<Self, ndarray::ShapeError> {
                let shape: [ndarray::Ix; $N] = match vec.try_into() {
                    Ok(shape) => shape,
                    Err(_) => {
                        return Err(ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape));
                    },
                };

                return Ok(ndarray::Dim(shape));
            }
        }
    };
}

impl_dim_for_vec_array!(0);
impl_dim_for_vec_array!(1);
impl_dim_for_vec_array!(2);
impl_dim_for_vec_array!(3);
impl_dim_for_vec_array!(4);
impl_dim_for_vec_array!(5);
impl_dim_for_vec_array!(6);

impl DimFromVec for ndarray::IxDyn {
    fn dim_from_vec(shape: Vec<usize>) -> Result<Self, ndarray::ShapeError> {
        return Ok(ndarray::Dim(shape));
    }
}

// Private struct to manage the lifetime of the array and its shape/strides
struct ManagerContext<T> {
    array: T,
    shape: Vec<i64>,
    strides: Vec<i64>,
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

impl<T, D> TryFrom<Array<T, D>> for DLPackTensor
where
    D: Dimension,
    T: GetDLPackDataType + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: Array<T, D>) -> Result<Self, Self::Error> {
        let shape: Vec<i64> = array.shape().iter().map(|&s| s as i64).collect();
        let strides: Vec<i64> = array.strides().iter().map(|&s| s as i64).collect();

        let mut ctx = Box::new(ManagerContext {
            array,
            shape,
            strides,
        });

        let data = if ctx.array.is_empty() {
            std::ptr::null_mut()
        } else {
            ctx.array.as_ptr()
        };

        let ndim = ctx.shape.len() as i32;
        let dl_tensor = sys::DLTensor {
            // Casting to a mut pointer is not necessarily safe, but is required
            // by DLPack. The data can be mutated through this pointer, we
            // should try to find a way to make this work in Rust type system in
            // the future.
            data: data.cast_mut().cast(),
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: ndim,
            dtype: T::get_dlpack_data_type(),
            shape: if ndim == 0 { std::ptr::null_mut() } else { ctx.shape.as_mut_ptr() },
            strides: if ndim == 0 { std::ptr::null_mut() } else { ctx.strides.as_mut_ptr() },
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(deleter_fn::<Array<T, D>>),
            flags: sys::DLPACK_FLAG_BITMASK_IS_COPIED,
            dl_tensor,
        });

        unsafe {
            Ok(DLPackTensor::from_ptr(Box::into_raw(managed_tensor)))
        }
    }
}

/// Convert a shared `ArcArray` into a `DLPackTensor`.
/// This is ZERO-COPY: it increments the reference count of the data.
impl<T, D> TryFrom<ArcArray<T, D>> for DLPackTensor
where
    D: Dimension,
    T: GetDLPackDataType + 'static + Clone,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: ArcArray<T, D>) -> Result<Self, Self::Error> {
        let shape: Vec<i64> = array.shape().iter().map(|&s| s as i64).collect();
        let strides: Vec<i64> = array.strides().iter().map(|&s| s as i64).collect();
        let ndim = shape.len() as i32;

        let mut ctx = Box::new(ManagerContext {
            array,
            shape,
            strides,
        });

        let data = if ctx.array.is_empty() {
            std::ptr::null_mut()
        } else {
            ctx.array.as_ptr()
        };

        let dl_tensor = sys::DLTensor {
            // Same as above, casting to a mut pointer is not necessarily safe.
            data: data.cast_mut().cast(),
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim,
            dtype: T::get_dlpack_data_type(),
            shape: if ndim == 0 { std::ptr::null_mut() } else { ctx.shape.as_mut_ptr() },
            strides: if ndim == 0 { std::ptr::null_mut() } else { ctx.strides.as_mut_ptr() },
            byte_offset: 0,
        };

        let managed_tensor = Box::new(sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(deleter_fn::<ArcArray<T, D>>),
            flags: sys::DLPACK_FLAG_BITMASK_READ_ONLY,
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
    use crate::sys::{DLDevice, DLDeviceType, DLTensor};
    use ndarray::prelude::*;
    use ndarray::ArcArray2;

    #[test]
    fn test_dlpack_to_ndarray() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut shape = vec![2i64, 3];
        let mut strides = vec![3i64, 1];

        let dl_tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: f32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let dlpack_ref = unsafe { DLPackTensorRef::from_raw(dl_tensor) };
        let array_view = ArrayView2::<f32>::try_from(dlpack_ref).unwrap();

        let expected = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(array_view, expected);
    }

    #[test]
    fn test_dlpack_to_ndarray_f_contiguous() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut shape = vec![2i64, 3];
        // Fortran-contiguous strides
        let mut strides = vec![1i64, 2];

        let dl_tensor = DLTensor {
            data: data.as_mut_ptr().cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: f32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let dlpack_ref = unsafe { DLPackTensorRef::from_raw(dl_tensor) };
        let array_view = ArrayView2::<f32>::try_from(dlpack_ref).unwrap();

        assert!(!array_view.is_standard_layout());
        let expected = arr2(&[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]);
        assert_eq!(array_view, expected);
    }

    #[test]
    fn test_dlpack_to_ndarray_wrong_device() {
        let mut data = vec![1.0f32];
        let mut shape = vec![1i64];

        let dl_tensor = DLTensor {
            data: data.as_mut_ptr().cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCUDA,
                device_id: 0,
            },
            ndim: 1,
            dtype: f32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };

        let dlpack_ref = unsafe { DLPackTensorRef::from_raw(dl_tensor) };
        let result = ArrayView1::<f32>::try_from(dlpack_ref);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("only CPU"), "got: {}", err);
    }

    #[test]
    fn test_ndarray_to_dlpack() {
        let array = arr2(&[[1i64, 2, 3], [4, 5, 6]]);
        let view = array.view();
        let dlpack_ref = DLPackTensorRef::try_from(&view).unwrap();
        let raw = dlpack_ref.raw;

        assert_eq!(raw.ndim, 2);
        assert_eq!(raw.device.device_type, DLDeviceType::kDLCPU);
        assert_eq!(raw.dtype, i64::get_dlpack_data_type());
        assert_eq!(raw.data as *const i64, array.as_ptr());

        let shape = unsafe { std::slice::from_raw_parts(raw.shape, 2) };
        assert_eq!(shape, &[2, 3]);

        let strides = unsafe { std::slice::from_raw_parts(raw.strides, 2) };
        assert_eq!(strides, &[3, 1]);
    }

    #[test]
    fn test_dlpack_to_ndarray_mut() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut shape = vec![2i64, 3];
        let mut strides = vec![3i64, 1];

        let dl_tensor = DLTensor {
            data: data.as_mut_ptr().cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: f32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let dlpack_ref_mut = unsafe { DLPackTensorRefMut::from_raw(dl_tensor) };
        let mut array_view_mut = ArrayViewMut2::<f32>::try_from(dlpack_ref_mut).unwrap();

        array_view_mut[[0, 0]] = 100.0;
        assert_eq!(data[0], 100.0);

        let expected = arr2(&[[100.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(array_view_mut, expected);
    }

    #[test]
    fn test_ndarray_to_managed_tensor() {
        let array = arr2(&[[1i64, 2, 3], [4, 5, 6]]);
        // The original array is moved into the manager.
        let tensor: DLPackTensor = array.try_into().unwrap();

        let raw = unsafe {
            &tensor.raw.as_ref().dl_tensor
        };
        assert_eq!(raw.ndim, 2);
        assert_eq!(raw.device.device_type, DLDeviceType::kDLCPU);
        assert_eq!(raw.dtype, i64::get_dlpack_data_type());

        let shape = unsafe { std::slice::from_raw_parts(raw.shape, 2) };
        assert_eq!(shape, &[2, 3]);

        let strides = unsafe { std::slice::from_raw_parts(raw.strides, 2) };
        assert_eq!(strides, &[3, 1]);

        // To check correctness, we can create a view from the managed tensor's data.
        let view = unsafe {
            let tensor_ref = DLPackTensorRef::from_raw(*raw);
            ndarray::ArrayView2::<i64>::try_from(tensor_ref).unwrap()
        };
        assert_eq!(view, arr2(&[[1, 2, 3], [4, 5, 6]]));
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original_array = arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let tensor: DLPackTensor = original_array.clone().try_into().unwrap();
        let final_array: Array<f32, _> = tensor.try_into().unwrap();

        assert_eq!(original_array, final_array);
    }

    #[test]
    fn test_arc_array_to_dlpack_share() {
        let array = ArcArray2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let ptr = array.as_ptr();

        // Conversion to DLPackTensor should share data
        let tensor: DLPackTensor = array.clone().try_into().unwrap();
        let raw = unsafe { &tensor.raw.as_ref().dl_tensor };

        assert_eq!(raw.data as *const f32, ptr);

        // Convert back to Array (Copy)
        let array_copy: Array<f32, _> = tensor.try_into().unwrap();
        assert_eq!(array, array_copy);
        // Pointers should differ due to copy
        assert_ne!(array_copy.as_ptr(), ptr);
    }

    #[test]
    fn test_dlpack_to_arc_array() {
        let array = arr2(&[[10.0f32, 11.0], [12.0, 13.0]]);
        let tensor: DLPackTensor = array.clone().try_into().unwrap();

        let arc_array: ArcArray<f32, _> = tensor.try_into().unwrap();
        assert_eq!(arc_array, array);
    }

    #[test]
    fn test_arc_array_to_dlpack_ref() {
        let array = ArcArray2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let tensor_ref: DLPackTensorRef = (&array).try_into().unwrap();

        assert_eq!(tensor_ref.n_dims(), 2);
        let shape = tensor_ref.shape();
        assert_eq!(shape, &[2, 2]);
    }

    #[test]
    fn test_array_conversion_permits_mutation() {
        let array = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let mut tensor: DLPackTensor = array.try_into().unwrap();

        // This should not panic because flags include IS_COPIED
        // and do not include READ_ONLY.
        let mut tensor_mut = tensor.as_mut();
        let ptr = tensor_mut.data_ptr_mut::<f32>().unwrap();

        unsafe {
            *ptr = 42.0;
        }

        let val = tensor.as_ref().data_ptr::<f32>().unwrap();
        assert_eq!(unsafe { *val }, 42.0);
    }

    #[test]
    fn test_arc_array_conversion_allows_readonly_access() {
        let array = ArcArray2::from_elem((2, 2), 1.0f32);
        let tensor: DLPackTensor = array.clone().try_into().unwrap();

        // Standard immutable access should remain functional.
        let tensor_ref = tensor.as_ref();
        assert_eq!(tensor_ref.dtype(), f32::get_dlpack_data_type());
    }


    #[test]
    fn empty_ndarray_to_dlpack() {
        let array = Array1::<f64>::from_shape_vec([0], vec![]).unwrap();
        let tensor: DLPackTensor = array.try_into().unwrap();
        assert_eq!(tensor.shape(), &[0]);
        assert!(tensor.as_dltensor().data.is_null());

        let array = Array3::<f64>::from_shape_vec([0, 0, 0], vec![]).unwrap();
        let tensor: DLPackTensor = array.try_into().unwrap();
        assert_eq!(tensor.shape(), &[0, 0, 0]);
        assert!(tensor.as_dltensor().data.is_null());
    }

    #[test]
    fn empty_arc_array_to_dlpack() {
        let array = ndarray::ArcArray3::<f64>::from_shape_vec([0, 0, 0], vec![]).unwrap();
        let tensor: DLPackTensor = array.try_into().unwrap();
        assert_eq!(tensor.shape(), &[0, 0, 0]);
        assert!(tensor.as_dltensor().data.is_null());
    }

    #[test]
    fn empty_dlpack_to_ndarray_view() {
        let mut shape = vec![0i64, 0, 0];
        let mut strides = vec![0i64, 0, 0];

        let dl_tensor = DLTensor {
            data: std::ptr::null_mut(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 3,
            dtype: f32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let dlpack_ref = unsafe { DLPackTensorRef::from_raw(dl_tensor) };
        let array_view = ArrayView3::<f32>::try_from(dlpack_ref).unwrap();
        assert_eq!(array_view.shape(), &[0, 0, 0]);
    }

    unsafe extern "C" fn box_deleter(tensor: *mut sys::DLManagedTensorVersioned) {
        unsafe {
            let _ = Box::from_raw(tensor);
        }
    }

    #[test]
    fn empty_dlpack_to_ndarray_owned() {
        let mut shape = vec![0i64, 0, 0];
        let mut strides = vec![0i64, 0, 0];

        let dl_tensor = DLTensor {
            data: std::ptr::null_mut(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 3,
            dtype: f32::get_dlpack_data_type(),
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
        let array: Array3<f32> = tensor.try_into().unwrap();
        assert_eq!(array.shape(), &[0, 0, 0]);
    }

    #[test]
    fn scalar_ndarray_to_dlpack() {
        let array = arr0(42.0f64);
        let tensor: DLPackTensor = array.try_into().unwrap();
        assert_eq!(tensor.n_dims(), 0);
        assert!(tensor.as_dltensor().shape.is_null());
        assert!(tensor.shape().is_empty());
        assert!(tensor.as_dltensor().strides.is_null());
        assert!(tensor.strides().is_none());
    }

    #[test]
    fn scalar_arc_array_to_dlpack() {
        let array = ndarray::ArcArray::<f64, ndarray::Ix0>::from_elem((), 42.0f64);
        let tensor: DLPackTensor = array.try_into().unwrap();
        assert_eq!(tensor.n_dims(), 0);
        assert!(tensor.as_dltensor().shape.is_null());
        assert!(tensor.shape().is_empty());
        assert!(tensor.as_dltensor().strides.is_null());
        assert!(tensor.strides().is_none());
    }

    #[test]
    fn scalar_dlpack_to_ndarray_view() {
        let mut value = 3.41f32;

        let dl_tensor = DLTensor {
            data: (&mut value as *mut f32).cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 0,
            dtype: f32::get_dlpack_data_type(),
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };

        let dlpack_ref = unsafe { DLPackTensorRef::from_raw(dl_tensor) };
        let array_view = ArrayView0::<f32>::try_from(dlpack_ref).unwrap();
        assert!(array_view.shape().is_empty());
        assert_eq!(array_view[()], 3.41);
    }

    #[test]
    fn scalar_dlpack_to_ndarray_owned() {
        let mut value = 2.72f64;

        let dl_tensor = DLTensor {
            data: (&mut value as *mut f64).cast(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 0,
            dtype: f64::get_dlpack_data_type(),
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
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
        let array: Array0<f64> = tensor.try_into().unwrap();
        assert_eq!(array[()], 2.72);
    }
}
