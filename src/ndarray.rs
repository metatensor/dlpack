use ndarray::ShapeBuilder;

use crate::data_types::{CastError, DLPackPointerCast, GetDLPackDataType};
use crate::sys;
use crate::{DLPackTensor, DLPackTensorRef, DLPackTensorRefMut};

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

impl std::fmt::Display for DLPackNDarrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DLPackNDarrayError::DeviceShouldBeCpu(device) => write!(
                f,
                "can not convert from device {} (only cpu is supported)",
                device
            ),
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

impl<'a, T, D> TryFrom<DLPackTensorRef<'a>> for ndarray::ArrayView<'a, T, D>
where
    T: DLPackPointerCast + 'static,
    D: DimFromVec + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(tensor: DLPackTensorRef<'a>) -> Result<Self, Self::Error> {
        if tensor.device().device_type != sys::DLDeviceType::kDLCPU {
            return Err(DLPackNDarrayError::DeviceShouldBeCpu(tensor.device()));
        }

        let ptr = tensor.data_ptr::<T>()?;
        let shape = tensor
            .shape()
            .iter()
            .map(|&s| s as usize)
            .collect::<Vec<_>>();
        let shape = <D as DimFromVec>::dim_from_vec(shape)?;

        let array;
        if let Some(strides) = DLPackTensorRef::strides(&tensor) {
            let strides = strides.iter().map(|&s| s as usize).collect::<Vec<_>>();
            let strides = <D as DimFromVec>::dim_from_vec(strides)?;
            let shape = shape.strides(strides);
            array = unsafe { ndarray::ArrayView::<T, _>::from_shape_ptr(shape, ptr) };
        } else {
            array = unsafe { ndarray::ArrayView::<T, _>::from_shape_ptr(shape, ptr) };
        }

        return Ok(array);
    }
}

impl<'a, T, D> TryFrom<DLPackTensorRefMut<'a>> for ndarray::ArrayViewMut<'a, T, D>
where
    T: DLPackPointerCast + 'static,
    D: DimFromVec + 'static,
{
    type Error = DLPackNDarrayError;

    fn try_from(mut tensor: DLPackTensorRefMut<'a>) -> Result<Self, Self::Error> {
        if tensor.device().device_type != sys::DLDeviceType::kDLCPU {
            return Err(DLPackNDarrayError::DeviceShouldBeCpu(tensor.device()));
        }

        let ptr = tensor.data_ptr_mut::<T>()?;
        let shape = tensor
            .shape()
            .iter()
            .map(|&s| s as usize)
            .collect::<Vec<_>>();
        let shape = <D as DimFromVec>::dim_from_vec(shape)?;

        let array;
        if let Some(strides) = DLPackTensorRefMut::strides(&tensor) {
            let strides = strides.iter().map(|&s| s as usize).collect::<Vec<_>>();
            let strides = <D as DimFromVec>::dim_from_vec(strides)?;
            let shape = shape.strides(strides);
            array = unsafe { ndarray::ArrayViewMut::<T, _>::from_shape_ptr(shape, ptr) };
        } else {
            array = unsafe { ndarray::ArrayViewMut::<T, _>::from_shape_ptr(shape, ptr) };
        }

        return Ok(array);
    }
}

/*****************************************************************************/
/*                            ndarray => DLPack                              */
/*****************************************************************************/

fn array_to_tensor_view<T, D>(
    array: ndarray::ArrayView<'_, T, D>,
) -> Result<sys::DLTensor, DLPackNDarrayError>
where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    let shape = array.shape();
    let strides = ndarray::ArrayView::strides(&array);

    // we need a `*const i64` for DLTensor, but we have usize and isize.
    // on 64-bit targets, isize will be the same as i64, so that's fine.
    if std::mem::size_of::<isize>() != std::mem::size_of::<i64>() {
        unimplemented!()
    }
    let strides = strides.as_ptr().cast_mut().cast();

    // usize will have the same binary representation as i64 for striclty
    // positive values, which is the most important case here.
    if std::mem::size_of::<isize>() != std::mem::size_of::<i64>() {
        unimplemented!()
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

impl<'a, T, D> TryFrom<ndarray::ArrayView<'a, T, D>> for DLPackTensorRef<'a>
where
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

impl<'a, T, D> TryFrom<ndarray::ArrayViewMut<'a, T, D>> for DLPackTensorRefMut<'a>
where
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

impl<'a, T, D> TryFrom<&'a ndarray::Array<T, D>> for DLPackTensorRef<'a>
where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        Self::try_from(array.view())
    }
}

impl<'a, T, D> TryFrom<&'a mut ndarray::Array<T, D>> for DLPackTensorRefMut<'a>
where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: &'a mut ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        Self::try_from(array.view_mut())
    }
}

// Array + other data we need to keep alive inside a DLPackTensor
struct ShapeStrideI64<T> {
    array: T,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn shape_strides_i64_box_deleter<T>(tensor: *mut sys::DLManagedTensorVersioned) {
    let data = (*tensor).manager_ctx.cast::<ShapeStrideI64<T>>();
    let boxed = Box::from_raw(data);
    std::mem::drop(boxed);
}

impl<T, D> TryFrom<ndarray::Array<T, D>> for DLPackTensor
where
    D: ndarray::Dimension,
    T: GetDLPackDataType,
{
    type Error = DLPackNDarrayError;

    fn try_from(array: ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().iter().map(|&v| v as i64).collect();
        let strides = ndarray::Array::strides(&array)
            .iter()
            .map(|&v| v as i64)
            .collect();

        let mut data = Box::new(ShapeStrideI64 {
            array,
            shape,
            strides,
        });

        let device = sys::DLDevice {
            device_type: sys::DLDeviceType::kDLCPU,
            device_id: 0,
        };

        let dl_tensor = sys::DLTensor {
            data: data.array.as_mut_ptr().cast(),
            device: device,
            ndim: data.shape.len() as i32,
            dtype: T::get_dlpack_data_type(),
            shape: data.shape.as_mut_ptr(),
            strides: data.strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let tensor = sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion {
                major: sys::DLPACK_MAJOR_VERSION,
                minor: sys::DLPACK_MINOR_VERSION,
            },
            manager_ctx: Box::into_raw(data).cast(),
            deleter: Some(shape_strides_i64_box_deleter::<T>),
            flags: sys::DLPACK_FLAG_BITMASK_IS_COPIED,
            dl_tensor: dl_tensor,
        };

        return Ok(unsafe {
            // SAFETY: we made sure the deleter is correct
            DLPackTensor::from_raw(tensor)
        });
    }
}

/*****************************************************************************/

/// Internal trait that will convert a Vec<usize> into one of ndarray's Dim
/// type.
pub trait DimFromVec
where
    Self: ndarray::Dimension,
{
    fn dim_from_vec(vec: Vec<usize>) -> Result<Self, ndarray::ShapeError>;
}

macro_rules! impl_dim_for_vec_array {
    ($N: expr) => {
        impl DimFromVec for ndarray::Dim<[ndarray::Ix; $N]> {
            fn dim_from_vec(vec: Vec<usize>) -> Result<Self, ndarray::ShapeError> {
                let shape: [ndarray::Ix; $N] = match vec.try_into() {
                    Ok(shape) => shape,
                    Err(_) => {
                        return Err(ndarray::ShapeError::from_kind(
                            ndarray::ErrorKind::IncompatibleShape,
                        ));
                    }
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
