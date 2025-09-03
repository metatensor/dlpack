use crate::data_types::GetDLPackDataType;
use crate::sys;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::CStr;

/*****************************************************************************/
/*                      ndarray => Python (via PyO3)                         */
/*****************************************************************************/

// The name for the PyCapsule, as per the DLPack standard.
const DLTENSOR_VERSIONED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"dltensor_versioned\0") };

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

pub fn ndarray_to_py_capsule<'py, T, D>(
    py: Python<'py>,
    array: ndarray::Array<T, D>,
) -> PyResult<PyObject>
where
    D: ndarray::Dimension,
    // 'static bound is needed for the deleter XXX(rg): Not pretty..
    T: GetDLPackDataType + 'static,
{
    let shape = array.shape().iter().map(|&v| v as i64).collect();
    let strides = array.strides().iter().map(|v| *v as i64).collect();
    let mut data_ctx = Box::new(ShapeStrideI64 {
        array,
        shape,
        strides,
    });

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

// Array + other data needed alive inside the DLPackTensor
pub struct ShapeStrideI64<T> {
    // This field holds the actual array data.
    // When an instance of ShapeStrideI64 is dropped,
    // this array is dropped too.
    array: T,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

// This is the "deleter" for the DLManagedTensor.
pub unsafe extern "C" fn shape_strides_i64_box_deleter<T>(
    tensor: *mut sys::DLManagedTensorVersioned,
) {
    // The manager_ctx is a raw pointer to the Box<ShapeStrideI64<T>>.
    // The reconstructed box goes out of scope at the end of this function,
    // so ShapeStrideI64 is deallocated by the memory manager's drop, ergo removing
    // the ndarray::Array it contains.
    let data = (*tensor).manager_ctx.cast::<ShapeStrideI64<T>>();
    let boxed = Box::from_raw(data);
    std::mem::drop(boxed);
}

/*****************************************************************************/

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

            let code = c_str!(
                r"
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
dl_obj = dlpack.asdlpack(arr)
result_capsule = dl_obj.__dlpack__()
"
            );
            py.run(code, None, Some(&locals))?;

            let result = locals.get_item("result_capsule")?.unwrap();
            let capsule: Bound<PyCapsule> = result.extract()?;

            let dl_tensor_ptr = capsule.pointer() as *const sys::DLTensor;
            let dlpack_ref = unsafe { crate::DLPackTensorRef::from_raw((*dl_tensor_ptr).clone()) };
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
            let py_capsule = ndarray_to_py_capsule(py, rust_array).unwrap();
            let locals = PyDict::new(py);
            locals.set_item("np", py.import("numpy")?)?;
            locals.set_item("rust_capsule", py_capsule)?;

            let setup_code = c_str!(
                "
class DLPackWrapper:
    def __init__(self, capsule):
        self.capsule = capsule
    def __dlpack__(self, stream=None):
        return self.capsule

rust_dlpack_obj = DLPackWrapper(rust_capsule)
"
            );
            py.run(setup_code, None, Some(&locals))?;

            let test_code = c_str!(
                r"
arr = np.from_dlpack(rust_dlpack_obj)
expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert arr.shape == (2, 3)
assert np.allclose(arr, expected)
"
            );
            py.run(test_code, None, Some(&locals))?;
            Ok(())
        })
    }
}
