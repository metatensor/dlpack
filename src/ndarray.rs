//! Conversion between DLPack and ndarray, this module requires the `ndarray`
//! feature to be enabled.
//!
//! The following conversions are supported:
//!
//! - `DLPackTensor` => `ndarray::Array` (makes a copy of the data)
//! - `DLPackTensorRef` => `ndarray::ArrayView`
//! - `DLPackTensorRefMut` => `ndarray::ArrayViewMut`
//! - `ndarray::Array` => `DLPackTensor`
//! - `&ndarray::Array` => `DLPackTensorRef`
//! - `&mut ndarray::Array` => `DLPackTensorRefMut`
//! - `ndarray::ArrayView` => `DLPackTensorRef`
//! - `ndarray::ArrayViewMut` => `DLPackTensorRefMut`
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

use ndarray::{Array, Dimension, ShapeBuilder};

use crate::data_types::{CastError, DLPackPointerCast, GetDLPackDataType};
use crate::sys;
use crate::{DLPackTensor, DLPackTensorRef, DLPackTensorRefMut};

#[cfg(feature = "pyo3")]
use pyo3::PyErr;

/// Possible error causes when converting between ndarray and DLPack
#[derive(Debug)]
pub enum DLPackNDarrayError {
    /// ndarray only support data which lives on CPU
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
            DLPackNDarrayError::DeviceShouldBeCpu(device) => write!(f, "can not convert from device {} (only cpu is supported)", device),
            DLPackNDarrayError::InvalidType(error) => write!(f, "type conversion error: {}", error),
            DLPackNDarrayError::ShapeError(error) => write!(f, "shape error: {}", error),
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

/*****************************************************************************/
/*                            DLPack => ndarray                              */
/*****************************************************************************/

impl<'a, T, D> TryFrom<DLPackTensorRef<'a>> for ndarray::ArrayView<'a, T, D> where
    T: DLPackPointerCast + 'static,
    D: DimFromVec + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(tensor: DLPackTensorRef<'a>) -> Result<Self, Self::Error> {
        if tensor.device().device_type != sys::DLDeviceType::kDLCPU {
            return Err(DLPackNDarrayError::DeviceShouldBeCpu(tensor.device()))
        }

        let ptr = tensor.data_ptr::<T>()?;
        let shape = tensor.shape().iter().map(|&s| s as usize).collect::<Vec<_>>();
        let shape = <D as DimFromVec>::dim_from_vec(shape)?;

        let array;
        let strides_opt = DLPackTensorRef::strides(&tensor);
        // If the version is None, we assume it is a legacy version (< 1.2).
        // This allows NULL strides for unversioned tensors.
        let is_v1_2_or_newer = tensor.version().map_or(false, |v| {
            v.major > 1 || (v.major == 1 && v.minor >= 2)
        });

        // v1.2+ onwards: strides cannot be NULL if ndim != 0
        if is_v1_2_or_newer && tensor.n_dims() > 0 && strides_opt.is_none() {
            return Err(ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleLayout).into());
        }
        array = match strides_opt{
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
        if tensor.device().device_type != sys::DLDeviceType::kDLCPU {
            return Err(DLPackNDarrayError::DeviceShouldBeCpu(tensor.device()))
        }

        let ptr = tensor.data_ptr_mut::<T>()?;
        let shape = tensor.shape().iter().map(|&s| s as usize).collect::<Vec<_>>();
        let shape = <D as DimFromVec>::dim_from_vec(shape)?;

        let array;
        if let Some(strides) = DLPackTensorRefMut::strides(&tensor) {
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
///
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
    _array: T,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn deleter_fn<T>(manager: *mut sys::DLManagedTensorVersioned) {
    // Reconstruct the box and drop it, freeing the memory.
    let ctx = (*manager).manager_ctx as *mut ManagerContext<T>;
    let _ = Box::from_raw(ctx);
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
            _array: array,
            shape,
            strides,
        });

        let dl_tensor = sys::DLTensor {
            data: ctx._array.as_ptr() as *mut _,
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: ctx.shape.len() as i32,
            dtype: T::get_dlpack_data_type(),
            shape: ctx.shape.as_mut_ptr(),
            strides: ctx.strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let managed_tensor = sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion::current(),
            manager_ctx: Box::into_raw(ctx) as *mut _,
            deleter: Some(deleter_fn::<Array<T, D>>),
            flags: 0,
            dl_tensor,
        };

        unsafe {
            Ok(DLPackTensor::from_raw(managed_tensor))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::{DLDevice, DLDeviceType, DLTensor};
    use ndarray::prelude::*;

    #[test]
    fn test_dlpack_to_ndarray() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut shape = vec![2i64, 3];
        let mut strides = vec![3i64, 1];

        let dl_tensor = DLTensor {
            data: data.as_mut_ptr() as *mut _,
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
            data: data.as_mut_ptr() as *mut _,
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
            data: data.as_mut_ptr() as *mut _,
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
        assert!(matches!(result, Err(DLPackNDarrayError::DeviceShouldBeCpu(_))));
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
            data: data.as_mut_ptr() as *mut _,
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
            let tensor_ref = DLPackTensorRef::from_raw(raw.clone());
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
    fn test_v1_2_null_strides_error() {
        let mut data = vec![1.0f32, 2.0];
        let mut shape = vec![2i64];
        
        let dl_tensor = DLTensor {
           data: data.as_mut_ptr() as *mut _,
            device: DLDevice::cpu(),
            ndim: 1,
            dtype: f32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(), // NULL strides
            byte_offset: 0,
        };

        let dlpack_ref = unsafe { 
            DLPackTensorRef::from_raw_with_version(dl_tensor, Some(sys::DLPackVersion::current())) 
        };
        let result = ArrayView1::<f32>::try_from(dlpack_ref);
        assert!(result.is_err());
    }
}
