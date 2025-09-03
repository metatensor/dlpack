pub use ndarray::{Array, ShapeBuilder};

use crate::data_types::{CastError, GetDLPackDataType};
use crate::sys;

use pyo3::ffi;
use pyo3::prelude::*;

/// Possible error causes when converting between ndarray and DLPack
#[derive(Debug)]
pub enum DLPackNDarrayError {
    DeviceShouldBeCpu(sys::DLDevice),
    InvalidType(CastError),
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
impl From<DLPackNDarrayError> for PyErr {
    fn from(err: DLPackNDarrayError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
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
/*                       ndarray => Python                                   */
/*****************************************************************************/

use std::ffi::CStr;

const CAPSULE_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"dltensor_versioned\0") };

#[pyclass]
struct DLPackPyTensor {
    array: Option<Array<f32, ndarray::IxDyn>>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

// Deleter for the underlying ndarray data
unsafe extern "C" fn array_box_deleter(tensor: *mut sys::DLManagedTensorVersioned) {
    let _ = Box::from_raw((*tensor).manager_ctx as *mut Array<f32, ndarray::IxDyn>);
}

// Deleter for the PyCapsule itself.
unsafe extern "C" fn capsule_deleter(capsule: *mut ffi::PyObject) {
    if ffi::PyCapsule_IsValid(capsule, CAPSULE_NAME.as_ptr()) == 0 {
        return;
    }

    let managed = ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr())
        as *mut sys::DLManagedTensorVersioned;
    if managed.is_null() {
        return;
    }

    let boxed_managed = Box::from_raw(managed);
    if let Some(deleter) = boxed_managed.deleter {
        deleter(Box::into_raw(boxed_managed));
    }
}

impl DLPackPyTensor {
    fn new(array: Array<f32, ndarray::IxDyn>) -> Self {
        let shape = array.shape().iter().map(|&v| v as i64).collect();
        let strides = array.strides().iter().map(|v| *v as i64).collect();
        Self {
            array: Some(array),
            shape,
            strides,
        }
    }
}

#[pymethods]
impl DLPackPyTensor {
    fn __dlpack__(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let array = self.array.take().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("DLPack tensor has already been consumed")
        })?;

        let manager_ctx = Box::into_raw(Box::new(array)).cast();
        let data_ptr = unsafe { (*(manager_ctx as *mut Array<f32, ndarray::IxDyn>)).as_mut_ptr() };

        let dl_tensor = sys::DLTensor {
            data: data_ptr.cast(),
            device: sys::DLDevice {
                device_type: sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: self.shape.len() as i32,
            dtype: f32::get_dlpack_data_type(),
            shape: self.shape.as_mut_ptr(),
            strides: self.strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let managed_tensor = sys::DLManagedTensorVersioned {
            version: sys::DLPackVersion {
                major: sys::DLPACK_MAJOR_VERSION,
                minor: sys::DLPACK_MINOR_VERSION,
            },
            manager_ctx,
            deleter: Some(array_box_deleter),
            flags: 0,
            dl_tensor,
        };

        let managed_tensor_ptr = Box::into_raw(Box::new(managed_tensor));

        unsafe {
            let capsule_ptr = ffi::PyCapsule_New(
                managed_tensor_ptr.cast(),
                CAPSULE_NAME.as_ptr(),
                Some(capsule_deleter),
            );

            Ok(PyObject::from_owned_ptr(py, capsule_ptr))
        }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (sys::DLDeviceType::kDLCPU as i32, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DLPackTensorRef;
    use pyo3::ffi::c_str;
    use pyo3::types::PyCapsule;
    use pyo3::types::PyDict;

    #[test]
    fn test_numpy_to_ndarray_via_dlpack() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals.set_item("np", py.import("numpy")?)?;
            locals.set_item("dlpack", py.import("dlpack")?)?;

            let code = c_str!(
                "
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
dl_obj = dlpack.asdlpack(arr)
result_capsule = dl_obj.__dlpack__()
"
            );
            py.run(code, None, Some(&locals))?;

            let result = locals.get_item("result_capsule")?.unwrap();
            let capsule: Bound<PyCapsule> = result.extract()?;

            let dl_tensor_ptr = capsule.pointer() as *const sys::DLTensor;
            let dlpack_ref = unsafe { DLPackTensorRef::from_raw((*dl_tensor_ptr).clone()) };
            let array = ndarray::ArrayView2::<f32>::try_from(dlpack_ref).unwrap();

            assert_eq!(array.shape(), [2, 3]);
            assert_eq!(array, ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
            Ok(())
        })
    }

    #[test]
    fn test_ndarray_to_numpy_via_dlpack() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rust_array = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            let rust_dlpack_obj = DLPackPyTensor::new(rust_array.into_dyn());

            let locals = PyDict::new(py);
            locals.set_item("np", py.import("numpy")?)?;
            locals.set_item("rust_dlpack_obj", Py::new(py, rust_dlpack_obj)?)?;

            let code = c_str!(
                "
arr = np.from_dlpack(rust_dlpack_obj)
expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert arr.shape == (2, 3)
assert np.allclose(arr, expected)
"
            );
            py.run(code, None, Some(&locals))?;
            Ok(())
        })
    }
}
