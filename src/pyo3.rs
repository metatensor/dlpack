use crate::sys;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::CStr;

/*****************************************************************************/
/*                      ndarray => Python (via PyO3)                         */
/*****************************************************************************/

// The name for the PyCapsule, as per the DLPack standard.
const DLTENSOR_VERSIONED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"dltensor_versioned\0") };

// This deleter is called by the Python garbage collector when the PyCapsule's
// reference count goes to zero. It reclaims the Box'd DLManagedTensorVersioned
// and calls its *original* deleter, which is responsible for freeing the
// underlying array data.
unsafe extern "C" fn pycapsule_deleter(capsule: *mut ffi::PyObject) {
    if ffi::PyCapsule_IsValid(capsule, DLTENSOR_VERSIONED_NAME.as_ptr()) == 0 {
        return;
    }

    let managed_tensor_ptr = ffi::PyCapsule_GetPointer(capsule, DLTENSOR_VERSIONED_NAME.as_ptr())
        as *mut sys::DLManagedTensorVersioned;

    if managed_tensor_ptr.is_null() {
        return;
    }

    // Reclaim ownership of the Box<DLManagedTensorVersioned>.
    let mut boxed_managed_tensor = Box::from_raw(managed_tensor_ptr);

    // Call the original deleter function for the underlying data (e.g., the ndarray).
    if let Some(deleter) = boxed_managed_tensor.deleter {
        deleter(boxed_managed_tensor.as_mut());
    }
}

pub struct DLPackPyCapsule(pub Py<PyCapsule>);

pub trait IntoDLPackPyCapsule: Sized {
    fn into_dlpack_pycapsule(self) -> PyResult<DLPackPyCapsule>;
}

/// A generic implementation to convert any type that can be converted into a
/// `DLManagedTensorVersioned` into our local `DLPackPyCapsule`.
impl<T> IntoDLPackPyCapsule for T
where
    T: TryInto<sys::DLManagedTensorVersioned>,
    T::Error: std::fmt::Display,
{
    fn into_dlpack_pycapsule(self) -> PyResult<DLPackPyCapsule> {
        let managed_tensor = self
            .try_into()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let managed_tensor_ptr = Box::into_raw(Box::new(managed_tensor));

        Python::with_gil(|py| {
            let capsule = unsafe {
                Py::from_owned_ptr(
                    py,
                    ffi::PyCapsule_New(
                        managed_tensor_ptr as *mut _,
                        DLTENSOR_VERSIONED_NAME.as_ptr(),
                        Some(pycapsule_deleter),
                    ),
                )
            };
            Ok(DLPackPyCapsule(capsule))
        })
    }
}

/*****************************************************************************/

#[cfg(test)]
mod tests {
    use super::IntoDLPackPyCapsule;
    use crate::sys;
    use ndarray::{arr2, Array, ArrayView2};
    use pyo3::ffi::c_str;
    use pyo3::prelude::*;
    use pyo3::types::{PyCapsule, PyDict};

    macro_rules! test_numpy_to_ndarray_via_dlpack_dtype {
        ($test_name:ident, $rust_type:ty, $np_dtype:expr) => {
            #[test]
            fn $test_name() -> PyResult<()> {
                pyo3::prepare_freethreaded_python();
                Python::with_gil(|py| {
                    let locals = PyDict::new(py);
                    locals.set_item("np", py.import("numpy")?)?;
                    locals.set_item("dlpack", py.import("dlpack")?)?;
                    locals.set_item("dtype", $np_dtype)?;

                    let code = c_str!("
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
dl_obj = dlpack.asdlpack(arr)
result_capsule = dl_obj.__dlpack__()
"
                    );
                    py.run(code, None, Some(&locals))?;

                    let result = locals.get_item("result_capsule")?.unwrap();
                    let capsule: Bound<PyCapsule> = result.extract()?;

                    let dl_tensor_ptr = capsule.pointer() as *const sys::DLTensor;
                    let dlpack_ref =
                        unsafe { crate::DLPackTensorRef::from_raw((*dl_tensor_ptr).clone()) };
                    let array = ArrayView2::<$rust_type>::try_from(dlpack_ref).unwrap();

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

    macro_rules! test_ndarray_to_numpy_via_dlpack_dtype {
        ($test_name:ident, $rust_type:ty, $np_dtype:expr) => {
            #[test]
            fn $test_name() -> PyResult<()> {
                pyo3::prepare_freethreaded_python();
                Python::with_gil(|py| {
                    let rust_array: Array<$rust_type, _> = arr2(&[
                        [1 as $rust_type, 2 as $rust_type, 3 as $rust_type],
                        [4 as $rust_type, 5 as $rust_type, 6 as $rust_type],
                    ]);

                    let py_capsule_wrapper = rust_array.into_dlpack_pycapsule()?;
                    let py_capsule = py_capsule_wrapper.0;

                    let locals = PyDict::new(py);
                    locals.set_item("np", py.import("numpy")?)?;
                    locals.set_item("rust_capsule", py_capsule)?;
                    locals.set_item("dtype", $np_dtype)?;

                    let code = c_str!("
class DLPackWrapper:
    def __init__(self, capsule):
        self.capsule = capsule
    def __dlpack__(self, stream=None):
        return self.capsule

rust_dlpack_obj = DLPackWrapper(rust_capsule)
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

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_f32, f32, "float32");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_f32, f32, "float32");

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_f64, f64, "float64");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_f64, f64, "float64");

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_i32, i32, "int32");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_i32, i32, "int32");

    test_numpy_to_ndarray_via_dlpack_dtype!(test_from_numpy_i64, i64, "int64");
    test_ndarray_to_numpy_via_dlpack_dtype!(test_to_numpy_i64, i64, "int64");
}
