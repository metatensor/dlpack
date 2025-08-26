use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyCapsule};

#[test]
fn run() {
    Python::with_gil(|py| {
        let numpy = py.import_bound("numpy").unwrap();
        let locals = [("np", numpy)].into_py_dict_bound(py);
        let code = "np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).__dlpack__()";
        let value = py.eval_bound(code, None, Some(&locals)).unwrap();
        let capsule: &Bound<PyCapsule> = value.downcast().unwrap();
        let dl_tensor = unsafe { &*capsule.pointer().cast::<dlpack::sys::DLTensor>() };

        let dlpack_ref = unsafe { dlpack::DLPackTensorRef::from_raw(dl_tensor.clone()) };

        let array = ndarray::ArrayView2::<f32>::try_from(dlpack_ref).unwrap();
        assert_eq!(array.shape(), [2, 3]);
        assert_eq!(array, ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    })
}
