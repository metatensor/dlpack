//! Convertions between DLPack tensors and Python objects using PyO3. This
//! module requires the `pyo3` feature to be enabled.
//!
//! This module provides the `PyDLPack` class, which implements the Python
//! DLPack protocol and can be used with any class offering a `from_dlpack`
//! function.
//!
//! The following conversions are supported:
//!
//! - `DLPackTensor` => `PyDLPack`: transfers ownership of the tensor from Rust
//!   to Python.
//! - `Py<PyCapsule>` and `Bound<'py, PyCapsule>` => `DLPackTensor`: transfers
//!   ownership of the tensor to Rust. The tensor is stored inside a PyCapsule,
//!   as returned by the `__dlpack__` method of a compatible Python object. See
//!   also
//!   <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html>.
//! - `Bound<'py, PyCapsule>` => `DLPackTensorRef`: get a read-only view of the
//!   tensor obtained from Python, without transferring ownership.
//!
//! # Examples
//!
//! ```
//! use pyo3::prelude::*;
//! use pyo3::types::IntoPyDict;
//! use pyo3::ffi::c_str;
//! use pyo3::types::PyCapsule;
//!
//! use dlpack::{DLPackTensor, DLPackTensorRef};
//!
//! Python::initialize();
//!
//! // pass data from rust to Python
//! let array = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let dlpack_tensor = DLPackTensor::try_from(array).unwrap();
//! let py_tensor = dlpack::pyo3::PyDLPack::try_from(dlpack_tensor).unwrap();
//!
//! Python::attach(|py| {
//!     let locals = [("np", py.import("numpy").unwrap())].into_py_dict(py).unwrap();
//!     locals.set_item("tensor", py_tensor).unwrap();
//!     py.run(c_str!("
//! array = np.from_dlpack(tensor)
//! expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
//! assert np.array_equal(array, expected)"), None, Some(&locals)).unwrap();
//! });
//!
//! // pass data from Python to Rust
//! Python::attach(|py| {
//!     let locals = [("np", py.import("numpy").unwrap())].into_py_dict(py).unwrap();
//!     py.run(c_str!("
//! array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
//! capsule = array.__dlpack__()"), None, Some(&locals)).unwrap();
//!     let capsule = locals.get_item("capsule").unwrap().unwrap().extract::<Bound<PyCapsule>>().unwrap();
//!     let dlpack_ref = DLPackTensorRef::try_from(capsule).unwrap();
//!     let array = ndarray::ArrayView2::<f64>::try_from(dlpack_ref).unwrap();
//!
//!     assert_eq!(array, ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
//! });
//! ```

use crate::sys::{self, DLManagedTensorVersioned};
use crate::{DLPackTensor, DLPackTensorRef};

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use std::ffi::CStr;
use std::ptr::NonNull;

/*****************************************************************************/
/*                      DLPackTensor => Python (via PyO3)                    */
/*****************************************************************************/

// The name for the PyCapsule, as per the DLPack standard.
const DLTENSOR_VERSIONED_NAME: &CStr = pyo3::ffi::c_str!("dltensor_versioned");
const USED_DLTENSOR_VERSIONED_NAME: &CStr = pyo3::ffi::c_str!("used_dltensor_versioned");
const DLTENSOR_NAME: &CStr = pyo3::ffi::c_str!("dltensor");

/// Python object implementing the dlpack protocol.
#[pyclass]
pub struct PyDLPack {
    capsule: Py<PyCapsule>,
    is_versioned: bool,
}

impl PyDLPack {
    fn as_dltensor<'py>(&self, py: Python<'py>) -> PyResult<&'py sys::DLTensor> {
        if self.is_versioned {
            let versioned_tensor = self.capsule.bind(py).pointer() as *const sys::DLManagedTensorVersioned;
            if versioned_tensor.is_null() {
                return Err(PyErr::new::<PyValueError, _>(
                    "PyCapsule pointer is null",
                ));
            }

            unsafe {
                return Ok(&(*versioned_tensor).dl_tensor);
            }
        } else {
            let tensor = self.capsule.bind(py).pointer() as *const sys::DLManagedTensor;
            if tensor.is_null() {
                return Err(PyErr::new::<PyValueError, _>(
                    "PyCapsule pointer is null",
                ));
            }

            unsafe {
                return Ok(&(*tensor).dl_tensor);
            }
        }
    }
}

#[allow(unused_variables)]
#[pymethods]
impl PyDLPack {
    #[new]
    fn new<'py>(py: Python<'py>, capsule: Py<PyCapsule>) -> PyResult<Self> {
        let name = capsule.bind(py).name()?;

        let is_versioned = if name == Some(DLTENSOR_NAME) {
            false
        } else if name == Some(DLTENSOR_VERSIONED_NAME) {
            true
        } else if name.is_none() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyCapsule name is not set",
            ));
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                format!("invalid capsule name: expected 'dltensor' or 'dltensor_versioned', got '{:?}'", name)
            ));
        };

        Ok(PyDLPack{ is_versioned, capsule })
    }

    /// Get the underlying PyCapsule containing the DLPack tensor.
    ///
    /// <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html>
    #[pyo3(signature=(*, stream=None, max_version=None, dl_device=None, copy=None))]
    pub fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
        max_version: Option<Bound<'py, PyAny>>,
        dl_device: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyCapsule>> {
        if stream.is_some() {
            return Err(PyErr::new::<PyValueError, _>("only `stream=None` is supported"));
        }

        // we can ignore `max_version`, the consumer is supposed to check it again
        // anyway

        if let Some(device) = dl_device {
            if device.ne(self.__dlpack_device__(py)?)? {
                return Err(PyErr::new::<PyBufferError, _>("unsupported `dl_device`"));
            }
        }

        if copy.is_some() {
            return Err(PyErr::new::<PyValueError, _>("only `copy=None` is supported"));
        }

        let capsule = self.capsule.clone_ref(py);
        let name = capsule.bind(py).name()?.expect("capsule name should be set").to_str().expect("name should be utf8");
        if name.starts_with("used_") {
            return Err(PyErr::new::<PyValueError, _>("this caspsule has already been used"));
        }

        return Ok(capsule);
    }

    /// Implementation of `__dlpack_device__`, returning a tuple with `(device_type, device_id)`.
    /// <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html>
    pub fn __dlpack_device__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyTuple>> {
        let tensor = self.as_dltensor(py)?;
        let device = tensor.device;

        let tuple = PyTuple::new(py, [device.device_type as i32, device.device_id])?;
        return Ok(tuple.unbind());
    }
}

impl<'py> TryFrom<&Bound<'py, PyCapsule>> for DLPackTensor {
    type Error = PyErr;

    fn try_from(capsule: &Bound<'py, PyCapsule>) -> Result<Self, Self::Error> {
        let name = capsule.name()?;

        let is_versioned = if name == Some(DLTENSOR_NAME) {
            false
        } else if name == Some(DLTENSOR_VERSIONED_NAME) {
            true
        } else if name.is_none() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyCapsule name is not set",
            ));
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                format!("invalid capsule name: expected 'dltensor' or 'dltensor_versioned', got '{:?}'", name)
            ));
        };

        if !is_versioned {
            return Err(PyErr::new::<PyValueError, _>(
                format!("invalid capsule, we only support 'dltensor_versioned' but got '{:?}'", name)
            ));
        }

        let pointer = capsule.pointer().cast::<DLManagedTensorVersioned>();
        if let Some(pointer) = NonNull::new(pointer) {
            unsafe {
                // set the name to "used_dltensor_versioned" so that
                // the capsule destructor does not free the tensor
                let status = pyo3::ffi::PyCapsule_SetName(
                    capsule.as_ptr(), USED_DLTENSOR_VERSIONED_NAME.as_ptr()
                );
                if status != 0 {
                    return Err(PyErr::fetch(capsule.py()));
                }

                return Ok(DLPackTensor::from_ptr(pointer));
            }
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                "invalid capsule, the pointer was null"
            ));
        }
    }
}

impl TryFrom<Py<PyCapsule>> for DLPackTensor {
    type Error = PyErr;

    fn try_from(value: Py<PyCapsule>) -> Result<Self, Self::Error> {
        Python::attach(|py| {
            let capsule = value.bind(py);
            DLPackTensor::try_from(capsule)
        })
    }
}

impl<'py> TryFrom<Bound<'py, PyCapsule>> for DLPackTensorRef<'py> {
    type Error = PyErr;

    fn try_from(value: Bound<'py, PyCapsule>) -> Result<Self, Self::Error> {
        Python::attach(|py| {
            let wrapper = PyDLPack::new(py, value.unbind())?;
            let dltensor = wrapper.as_dltensor(py)?;

            // SAFETY: The lifetime of the returned reference is tied to the
            // lifetime GIL lifetime.
            let tensor = unsafe {
                DLPackTensorRef::from_raw(dltensor.clone())
            };

            Ok(tensor)
        })
    }
}

unsafe extern "C" fn rust_capsule_deleter(object: *mut pyo3::ffi::PyObject) {
    if pyo3::ffi::PyCapsule_IsValid(object, USED_DLTENSOR_VERSIONED_NAME.as_ptr()) == 1 {
        // All good, the data was already transfered
        return;
    }

    if !pyo3::ffi::PyCapsule_IsValid(object, DLTENSOR_VERSIONED_NAME.as_ptr()) == 1 {
        // we got a bad capsule, send a warning
        pyo3::ffi::PyErr_WriteUnraisable(object);
        return;
    }

    let ptr = pyo3::ffi::PyCapsule_GetPointer(object, DLTENSOR_VERSIONED_NAME.as_ptr());

    // PyCapsule_IsValid checks the the pointer is not null
    let tensor = NonNull::new(ptr.cast::<DLManagedTensorVersioned>())
        .expect("the capsule should be non-null");
    std::mem::drop(DLPackTensor::from_ptr(tensor));
}

impl TryFrom<DLPackTensor> for PyDLPack {
    type Error = PyErr;

    fn try_from(value: DLPackTensor) -> Result<Self, Self::Error> {
        Python::attach(|py| {
            // SAFETY: we are holding the GIL here
            let capsule = unsafe {
                pyo3::ffi::PyCapsule_New(
                    value.raw.as_ptr().cast(),
                    DLTENSOR_VERSIONED_NAME.as_ptr(),
                    Some(rust_capsule_deleter),
                )
            };
            let capsule = unsafe {
                Bound::from_owned_ptr_or_err(py, capsule)?.cast_into_unchecked()
            };

            // do not run drop on the Rust side, the capsule now owns the tensor
            std::mem::forget(value);
            PyDLPack::new(py, capsule.unbind())
        })
    }
}


/*****************************************************************************/

#[cfg(test)]
mod tests {
    use crate::{DLPackTensor, DLPackTensorRef};

    use super::PyDLPack;

    use ndarray::{Array, ArrayView2};
    use pyo3::ffi::c_str;
    use pyo3::prelude::*;
    use pyo3::types::{PyCapsule, PyDict};

    macro_rules! test_numpy_to_ndarray_via_dlpack_dtype {
        ($test_name:ident, $rust_type:ty, $np_dtype:expr) => {
            #[test]
            fn $test_name() -> PyResult<()> {
                Python::initialize();
                Python::attach(|py| {
                    let locals = PyDict::new(py);
                    locals.set_item("np", py.import("numpy")?)?;
                    locals.set_item("dtype", $np_dtype)?;

                    let code = c_str!(
                        "
array = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
result_capsule = array.__dlpack__()
"
                    );
                    py.run(code, None, Some(&locals))?;

                    let result = locals.get_item("result_capsule")?.unwrap();
                    let capsule: Bound<PyCapsule> = result.extract()?;

                    let dlpack_ref = DLPackTensorRef::try_from(capsule)?;
                    let array = ArrayView2::<$rust_type>::try_from(dlpack_ref).unwrap();

                    let expected = ndarray::arr2(&[
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

    macro_rules! test_ndarray_to_numpy_via_dlpack_dtype {
        ($test_name:ident, $rust_type:ty, $np_dtype:expr) => {
            #[test]
            fn $test_name() -> PyResult<()> {
                Python::initialize();
                Python::attach(|py| {
                    let rust_array: Array<$rust_type, _> = ndarray::arr2(&[
                        [1 as $rust_type, 2 as $rust_type, 3 as $rust_type],
                        [4 as $rust_type, 5 as $rust_type, 6 as $rust_type],
                    ]);

                    let dl_tensor = DLPackTensor::try_from(rust_array).unwrap();
                    let tensor = PyDLPack::try_from(dl_tensor).unwrap();

                    let locals = PyDict::new(py);
                    locals.set_item("np", py.import("numpy")?)?;
                    locals.set_item("tensor", tensor)?;
                    locals.set_item("dtype", $np_dtype)?;

                    let code = c_str!(
                        "
array = np.from_dlpack(tensor)
expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
assert array.shape == (2, 3)
assert np.allclose(array, expected)
"
                    );
                    py.run(code, None, Some(&locals))?;
                    Ok(())
                })
            }
        };
    }

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_f32, f32, "float32");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_f32, f32, "float32");

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_f64, f64, "float64");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_f64, f64, "float64");

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_i32, i32, "int32");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_i32, i32, "int32");

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_i64, i64, "int64");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_i64, i64, "int64");
}
