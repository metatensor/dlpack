use ndarray::ShapeBuilder;

use crate::sys;
use crate::{DLPackTensorRefMut, DLPackTensorRef};
use crate::data_types::{CastError, DLPackPointerCast, GetDLPackDataType};

use pyo3::prelude::*;
use pyo3::ffi;
use std::ffi::CStr;

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

// Array + other data needed alive inside the DLPackTensor
struct ShapeStrideI64<T> {
    // This field holds the actual array data.
    // When an instance of ShapeStrideI64 is dropped,
    // this array is dropped too.
    array: T,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

// This is the "deleter" for the DLManagedTensor.
unsafe extern "C" fn shape_strides_i64_box_deleter<T>(tensor: *mut sys::DLManagedTensorVersioned) {
    // The manager_ctx is a raw pointer to the Box<ShapeStrideI64<T>>.
    // The reconstructed box goes out of scope at the end of this function,
    // so ShapeStrideI64 is deallocated by the memory manager's drop, ergo removing
    // the ndarray::Array it contains.
    let data = (*tensor).manager_ctx.cast::<ShapeStrideI64<T>>();
    let boxed = Box::from_raw(data);
    std::mem::drop(boxed);
}

/*****************************************************************************/
/*                      ndarray => Python (via PyO3)                         */
/*****************************************************************************/

// The name for the PyCapsule, as per the DLPack standard.
const DLTENSOR_VERSIONED_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"dltensor_versioned\0") };

unsafe extern "C" fn pycapsule_deleter(capsule: *mut ffi::PyObject) {
    if ffi::PyCapsule_IsValid(capsule, DLTENSOR_VERSIONED_NAME.as_ptr()) == 0 {
        return;
    }

    let managed_tensor_ptr = ffi::PyCapsule_GetPointer(capsule, DLTENSOR_VERSIONED_NAME.as_ptr())
        as *mut sys::DLManagedTensorVersioned;

    if managed_tensor_ptr.is_null() {
        return;
    }

    // Reclaim ownership.
    let mut boxed_managed_tensor = Box::from_raw(managed_tensor_ptr);

    // Call the deleter for the underlying data (the ndarray).
    if let Some(deleter) = boxed_managed_tensor.deleter {
        deleter(boxed_managed_tensor.as_mut());
    }

    // `boxed_managed_tensor` goes out of scope so the memory for the
    // DLManagedTensorVersioned struct itself is freed.
}

pub fn ndarray_to_py_capsule<'py, T, D>(py: Python<'py>, array: ndarray::Array<T, D>) -> PyResult<PyObject>
where
    D: ndarray::Dimension,
// 'static bound is needed for the deleter XXX(rg): Not pretty..
    T: GetDLPackDataType + 'static,
{
    let shape = array.shape().iter().map(|&v| v as i64).collect();
    let strides = array.strides().iter().map(|v| *v as i64).collect();
    let mut data_ctx = Box::new(ShapeStrideI64 { array, shape, strides });

    let dl_tensor = sys::DLTensor {
        data: data_ctx.array.as_mut_ptr().cast(),
        device: sys::DLDevice {
            device_type: sys::DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: data_ctx.shape.len() as i32,
        dtype: T::get_dlpack_data_type(),
        shape: data_ctx.shape.as_mut_ptr(),
        strides: data_ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed_tensor = sys::DLManagedTensorVersioned {
        version: sys::DLPackVersion {
            major: sys::DLPACK_MAJOR_VERSION,
            minor: sys::DLPACK_MINOR_VERSION,
        },
        manager_ctx: Box::into_raw(data_ctx).cast(),
        deleter: Some(shape_strides_i64_box_deleter::<ndarray::Array<T, D>>),
        flags: sys::DLPACK_FLAG_BITMASK_IS_COPIED,
        dl_tensor: dl_tensor,
    };

    let managed_tensor_ptr = Box::into_raw(Box::new(managed_tensor));

    let capsule = unsafe {
        ffi::PyCapsule_New(
            managed_tensor_ptr as *mut _,
            DLTENSOR_VERSIONED_NAME.as_ptr(),
            Some(pycapsule_deleter),
        )
    };
    Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
}

/*****************************************************************************/

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
    use pyo3::ffi::c_str;
    use pyo3::types::{PyCapsule, PyDict};
    use pyo3::PyResult;

    #[test]
    fn test_numpy_to_ndarray_via_dlpack() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals.set_item("np", py.import("numpy")?)?;
            locals.set_item("dlpack", py.import("dlpack")?)?;

            let code = c_str!(r"
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
dl_obj = dlpack.asdlpack(arr)
result_capsule = dl_obj.__dlpack__()
");
            py.run(code, None, Some(&locals))?;

            let result = locals.get_item("result_capsule")?.unwrap();
            let capsule: Bound<PyCapsule> = result.extract()?;

            let dl_tensor_ptr =  capsule.pointer() as *const sys::DLTensor ;
            let dlpack_ref = unsafe { crate::DLPackTensorRef::from_raw((*dl_tensor_ptr).clone()) };
            let array = ndarray::ArrayView2::<f32>::try_from(dlpack_ref).unwrap();

            assert_eq!(array.shape(), [2, 3]);
            assert_eq!(array, ndarray::arr2(&[[1.0, 2.0, 3.0],
                                              [4.0, 5.0, 6.0]]));
            Ok(())
        })
    }

#[test]
fn test_ndarray_to_numpy_via_dlpack() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let rust_array = ndarray::arr2(&[
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0]]);
        let py_capsule = ndarray_to_py_capsule(py, rust_array)?;
        let locals = PyDict::new(py);
        locals.set_item("np", py.import("numpy")?)?;
        locals.set_item("rust_capsule", py_capsule)?;

        let setup_code = c_str!("
class DLPackWrapper:
    def __init__(self, capsule):
        self.capsule = capsule
    def __dlpack__(self, stream=None):
        return self.capsule

rust_dlpack_obj = DLPackWrapper(rust_capsule)
");
        py.run(setup_code, None, Some(&locals))?;

        let test_code = c_str!(r"
arr = np.from_dlpack(rust_dlpack_obj)
expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert arr.shape == (2, 3)
assert np.allclose(arr, expected)
");
        py.run(test_code, None, Some(&locals))?;
        Ok(())
    })
}
}
