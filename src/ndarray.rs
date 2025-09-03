use ndarray::ShapeBuilder;

use crate::sys;
use crate::{DLPackTensorRefMut, DLPackTensorRef};
use crate::data_types::{CastError, DLPackPointerCast, GetDLPackDataType};

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
    ShapeError(ndarray::ShapeError)
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

// Also implement conversion to a PyErr for pyo3 integration
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
        if let Some(strides) = DLPackTensorRef::strides(&tensor) {
            let strides = strides.iter().map(|&s| s as usize).collect::<Vec<_>>();
            let strides = <D as DimFromVec>::dim_from_vec(strides)?;
            let shape = shape.strides(strides);
            array = unsafe {
                ndarray::ArrayView::<T, _>::from_shape_ptr(shape, ptr)
            };
        } else {
            array = unsafe {
                ndarray::ArrayView::<T, _>::from_shape_ptr(shape, ptr)
            };
        }

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


/*****************************************************************************/
/*                            ndarray => DLPack                              */
/*****************************************************************************/

fn array_to_tensor_view<T, D>(array: ndarray::ArrayView<'_, T, D>) -> Result<sys::DLTensor, DLPackNDarrayError> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    let shape = array.shape();
    let strides = ndarray::ArrayView::strides(&array);

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

impl<'a, T, D> TryFrom<ndarray::ArrayView<'a, T, D>> for DLPackTensorRef<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: ndarray::ArrayView<'a, T, D>) -> Result<Self, Self::Error> {
        let tensor = array_to_tensor_view(array)?;

        return Ok(unsafe {
            // SAFETY: we are constraining the lifetime of the return value
            DLPackTensorRef::from_raw(tensor)
        });
    }
}

impl<'a, T, D> TryFrom<ndarray::ArrayViewMut<'a, T, D>> for DLPackTensorRefMut<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: ndarray::ArrayViewMut<'a, T, D>) -> Result<Self, Self::Error> {
        let tensor = array_to_tensor_view(array.view())?;

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
        Self::try_from(array.view())
    }
}

impl<'a, T, D> TryFrom<&'a mut ndarray::Array<T, D>> for DLPackTensorRefMut<'a> where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a mut ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        Self::try_from(array.view_mut())
    }
}

/// Internal trait that will convert a Vec<usize> into one of ndarray's Dim
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
        let dlpack_ref = DLPackTensorRef::try_from(array.view()).unwrap();
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
}
