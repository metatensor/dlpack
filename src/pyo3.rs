pub use ndarray::{Array, ShapeBuilder};

use crate::data_types::{CastError, GetDLPackDataType};
use crate::sys;

use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::CStr;

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
            DLPackNDarrayError::InvalidType(error) => {
                write!(f, "type conversion error: {}", error)
            }
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

const CAPSULE_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"dltensor_versioned\0") };

/// Deleter for the PyCapsule. Its only job is to call the tensor's own deleter.
unsafe extern "C" fn capsule_deleter(capsule: *mut ffi::PyObject) {
    if ffi::PyCapsule_IsValid(capsule, CAPSULE_NAME.as_ptr()) == 0 {
        return;
    }

    let managed = ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr())
        as *mut sys::DLManagedTensorVersioned;

    if !managed.is_null() {
        if let Some(deleter) = (*managed).deleter {
            // This call is responsible for all cleanup.
            deleter(managed);
        }
    }
}

/// Macro to generate a unique deleter function for each data type.
macro_rules! define_deleter {
    ($deleter_name:ident, $rust_type:ty) => {
        /// This deleter is responsible for cleaning up two things:
        /// 1. The `manager_ctx`, which is the Boxed `ndarray::Array`.
        /// 2. The `DLManagedTensorVersioned` struct itself.
        unsafe extern "C" fn $deleter_name(tensor: *mut sys::DLManagedTensorVersioned) {
            if tensor.is_null() {
                return;
            }
            // Reconstruct the box for the manager context and let it drop.
            let _ = Box::from_raw((*tensor).manager_ctx as *mut Array<$rust_type, ndarray::IxDyn>);

            // Reconstruct the box for the DLManagedTensorVersioned itself and let it drop.
            let _ = Box::from_raw(tensor);
        }
    };
}

// Generate the concrete deleter functions.
define_deleter!(array_box_deleter_f32, f32);
define_deleter!(array_box_deleter_f64, f64);
define_deleter!(array_box_deleter_i32, i32);
define_deleter!(array_box_deleter_i64, i64);

/// Macro to generate a concrete DLPackPyTensor struct and its Python bindings
/// for a specific Rust type.
macro_rules! define_dlpack_py_tensor {
    ($struct_name:ident, $py_name:literal, $rust_type:ty, $deleter_name:ident) => {
        #[pyclass(name = $py_name)]
        struct $struct_name {
            array: Option<Array<$rust_type, ndarray::IxDyn>>,
            shape: Vec<i64>,
            strides: Vec<i64>,
        }

        // Rust-only implementation block
        impl $struct_name {
            #[allow(dead_code)]
            fn new(array: Array<$rust_type, ndarray::IxDyn>) -> Self {
                let shape = array.shape().iter().map(|&v| v as i64).collect();
                let strides = array.strides().iter().map(|v| *v as i64).collect();
                Self {
                    array: Some(array),
                    shape,
                    strides,
                }
            }
        }

        // Python-exposed methods
        #[pymethods]
        impl $struct_name {
            fn __dlpack__(&mut self, py: Python<'_>) -> PyResult<PyObject> {
                let array = self.array.take().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "DLPack tensor has already been consumed",
                    )
                })?;

                let manager_ctx = Box::into_raw(Box::new(array)).cast();
                let data_ptr = unsafe {
                    (*(manager_ctx as *mut Array<$rust_type, ndarray::IxDyn>)).as_mut_ptr()
                };

                let dl_tensor = sys::DLTensor {
                    data: data_ptr.cast(),
                    device: sys::DLDevice {
                        device_type: sys::DLDeviceType::kDLCPU,
                        device_id: 0,
                    },
                    ndim: self.shape.len() as i32,
                    dtype: <$rust_type>::get_dlpack_data_type(),
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
                    deleter: Some($deleter_name),
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
    };
}

define_dlpack_py_tensor!(
    DLPackPyTensorF32,
    "DLPackPyTensorF32",
    f32,
    array_box_deleter_f32
);
define_dlpack_py_tensor!(
    DLPackPyTensorF64,
    "DLPackPyTensorF64",
    f64,
    array_box_deleter_f64
);
define_dlpack_py_tensor!(
    DLPackPyTensorI32,
    "DLPackPyTensorI32",
    i32,
    array_box_deleter_i32
);
define_dlpack_py_tensor!(
    DLPackPyTensorI64,
    "DLPackPyTensorI64",
    i64,
    array_box_deleter_i64
);

#[pymodule]
fn dlpack_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<DLPackPyTensorF32>()?;
    m.add_class::<DLPackPyTensorF64>()?;
    m.add_class::<DLPackPyTensorI32>()?;
    m.add_class::<DLPackPyTensorI64>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DLPackTensorRef;
    use ndarray::arr2;
    use pyo3::ffi::c_str;
    use pyo3::types::{PyCapsule, PyDict};

    macro_rules! test_numpy_to_ndarray_dtype {
        ($test_name:ident, $rust_type:ty, $np_dtype:expr) => {
            #[test]
            fn $test_name() -> PyResult<()> {
                pyo3::prepare_freethreaded_python();
                Python::with_gil(|py| {
                    let locals = PyDict::new(py);
                    locals.set_item("np", py.import("numpy")?)?;
                    locals.set_item("dlpack", py.import("dlpack")?)?;
                    locals.set_item("dtype", $np_dtype)?;

                    let code = c_str!(
                        "
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
dl_obj = dlpack.asdlpack(arr)
result_capsule = dl_obj.__dlpack__()
"
                    );
                    py.run(code, None, Some(&locals))?;

                    let result = locals.get_item("result_capsule")?.unwrap();
                    let capsule: Bound<PyCapsule> = result.extract()?;

                    let dl_tensor_ptr = capsule.pointer() as *const sys::DLTensor;
                    let dlpack_ref = unsafe { DLPackTensorRef::from_raw((*dl_tensor_ptr).clone()) };
                    let array = ndarray::ArrayView2::<$rust_type>::try_from(dlpack_ref).unwrap();

                    let expected = arr2(&[
                        [1 as $rust_type, 2 as $rust_type, 3 as $rust_type],
                        [4 as $rust_type, 5 as $rust_type, 6 as $rust_type],
                    ]);

                    assert_eq!(array.shape(), [2, 3]);
                    assert_eq!(array, expected);
                    Ok(())
                })
            }
        };
    }

    macro_rules! test_ndarray_to_numpy_dtype {
        ($test_name:ident, $struct_name:ty, $rust_type:ty, $np_dtype:expr) => {
            #[test]
            fn $test_name() -> PyResult<()> {
                pyo3::prepare_freethreaded_python();
                Python::with_gil(|py| {
                    let rust_array = arr2(&[
                        [1 as $rust_type, 2 as $rust_type, 3 as $rust_type],
                        [4 as $rust_type, 5 as $rust_type, 6 as $rust_type],
                    ]);
                    let rust_dlpack_obj = <$struct_name>::new(rust_array.into_dyn());

                    let locals = PyDict::new(py);
                    locals.set_item("np", py.import("numpy")?)?;
                    locals.set_item("rust_dlpack_obj", Py::new(py, rust_dlpack_obj)?)?;
                    locals.set_item("dtype", $np_dtype)?;

                    let code = c_str!(
                        "
arr = np.from_dlpack(rust_dlpack_obj)
expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
assert arr.shape == (2, 3)
assert np.allclose(arr, expected)
"
                    );
                    py.run(code, None, Some(&locals))?;
                    Ok(())
                })
            }
        };
    }

    test_numpy_to_ndarray_dtype!(test_numpy_to_ndarray_f32, f32, "float32");
    test_ndarray_to_numpy_dtype!(test_ndarray_to_numpy_f32, DLPackPyTensorF32, f32, "float32");

    test_numpy_to_ndarray_dtype!(test_numpy_to_ndarray_f64, f64, "float64");
    test_ndarray_to_numpy_dtype!(test_ndarray_to_numpy_f64, DLPackPyTensorF64, f64, "float64");

    test_numpy_to_ndarray_dtype!(test_numpy_to_ndarray_i32, i32, "int32");
    test_ndarray_to_numpy_dtype!(test_ndarray_to_numpy_i32, DLPackPyTensorI32, i32, "int32");

    test_numpy_to_ndarray_dtype!(test_numpy_to_ndarray_i64, i64, "int64");
    test_ndarray_to_numpy_dtype!(test_ndarray_to_numpy_i64, DLPackPyTensorI64, i64, "int64");
}
