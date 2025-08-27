use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyCapsule};
use pyo3::ffi::c_str;

#[test]
fn run() -> PyResult<()> {
    Python::with_gil(|py| {
        let locals = [("np", py.import("numpy")?)].into_py_dict(py)?;
        let code = c_str!("np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).__dlpack__()");
        let value: Bound<PyCapsule> = py.eval(code, None, Some(&locals))?.extract()?;
        let dl_tensor = unsafe {
            &*value.pointer().cast::<dlpack::sys::DLTensor>()
        };

        let dlpack_ref = unsafe {
            dlpack::DLPackTensorRef::from_raw(dl_tensor.clone())
        };

        let array = ndarray::ArrayView2::<f32>::try_from(dlpack_ref).unwrap();
        assert_eq!(array.shape(), [2, 3]);
        assert_eq!(array, ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        Ok(())
    })
}
