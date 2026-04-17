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
//! # #[cfg(miri)] fn main() {}
//! # #[cfg(not(miri))]
//! # fn main() {
//! use pyo3::prelude::*;
//! use pyo3::types::IntoPyDict;
//! use pyo3::ffi::c_str;
//! use pyo3::types::PyCapsule;
//!
//! use dlpk::{DLPackTensor, DLPackTensorRef};
//!
//! Python::initialize();
//!
//! // pass data from rust to Python
//! let array = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let dlpack_tensor = DLPackTensor::try_from(array).unwrap();
//! let py_tensor = dlpk::pyo3::PyDLPack::try_from(dlpack_tensor).unwrap();
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
//! # }
//! ```

use crate::sys::{self, DLManagedTensorVersioned};
use crate::{DLPackTensor, DLPackTensorRef};

use pyo3::exceptions::{PyAttributeError, PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use std::ffi::{c_void, CStr};
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

    /// Producer-side DLPack fast exchange function table, exposed as a
    /// class attribute so consumers can look it up without instantiating
    /// the wrapper first.
    ///
    /// <https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h>
    #[classattr]
    #[allow(non_snake_case)]
    fn __dlpack_c_exchange_api__<'py>(py: Python<'py>) -> PyResult<Py<PyCapsule>> {
        // Leak a `Box<DLPackExchangeAPI>` once per process. `#[classattr]`
        // functions are invoked during class initialization, so this only
        // runs once; the pointer stays valid for the rest of the process,
        // which is what the DLPack spec requires.
        let api = Box::new(sys::DLPackExchangeAPI {
            header: sys::DLPackExchangeAPIHeader {
                version: sys::DLPackVersion::current(),
                prev_api: std::ptr::null_mut(),
            },
            managed_tensor_allocator: None,
            managed_tensor_from_py_object_no_sync:
                Some(pydlpack_managed_tensor_from_py_object),
            managed_tensor_to_py_object_no_sync:
                Some(pydlpack_managed_tensor_to_py_object),
            dltensor_from_py_object_no_sync:
                Some(pydlpack_dltensor_from_py_object),
            current_work_stream: Some(pydlpack_current_work_stream),
        });
        let ptr: *mut sys::DLPackExchangeAPI = Box::into_raw(api);

        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(
                ptr.cast(),
                sys::DLPACK_EXCHANGE_API_CAPSULE_NAME.as_ptr(),
                None, // no destructor: the struct lives for the process
            )
        };
        let capsule = unsafe {
            Bound::from_owned_ptr_or_err(py, capsule)?.cast_into_unchecked::<PyCapsule>()
        };
        Ok(capsule.unbind())
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
            let tensor_ref = unsafe { pointer.as_ref() };
            let version = tensor_ref.version;
            let is_v1_2_or_newer = version.major > 1 || (version.major == 1 && version.minor >= 2);
            let dltensor = &tensor_ref.dl_tensor;
            // Enforce v1.2+ stride requirement if ndim > 0
            if is_v1_2_or_newer && dltensor.ndim > 0 && dltensor.strides.is_null() {
                return Err(PyErr::new::<PyValueError, _>(
                    "DLPack v1.2+ requires non-NULL strides for non-scalar tensors"
                ));
            }
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

// ----- DLPackExchangeAPI producer callbacks (PyDLPack side) -----
//
// Python invokes these `extern "C"` callbacks through the function pointers
// stored in the static `DLPackExchangeAPI` capsule we hand out via the
// `__dlpack_c_exchange_api__` class attribute above. All callbacks run
// with the GIL held per the DLPack spec, catch any Rust panic (panicking
// across the C boundary would be UB) and map errors to the standard
// "return -1 with a Python exception set" contract.

fn into_ffi_rc(result: Result<(), PyErr>, py: Python<'_>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(err) => { err.restore(py); -1 }
    }
}

unsafe extern "C" fn pydlpack_managed_tensor_from_py_object(
    py_object: *mut std::os::raw::c_void,
    out: *mut *mut sys::DLManagedTensorVersioned,
) -> i32 {
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Python::attach(|py| {
            into_ffi_rc((|| -> PyResult<()> {
                // SAFETY: caller promises `py_object` is a valid borrowed PyObject*
                let obj = unsafe {
                    Bound::<PyAny>::from_borrowed_ptr(py, py_object.cast())
                };
                let bound: Bound<PyDLPack> = obj.downcast_into().map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!(
                        "exchange API got a non-PyDLPack object: {}", e,
                    ))
                })?;
                let self_ref = bound.borrow();
                let capsule = self_ref.capsule.bind(py);

                // The capsule must still be the unused, versioned tensor. If a
                // previous call transferred ownership, the name is already
                // `used_dltensor_versioned` and we refuse to re-hand-out.
                let name = capsule.name()?;
                if name != Some(DLTENSOR_VERSIONED_NAME) {
                    return Err(PyErr::new::<PyValueError, _>(
                        "PyDLPack has already been consumed or carries a legacy unversioned tensor",
                    ));
                }

                let ptr = capsule.pointer().cast::<sys::DLManagedTensorVersioned>();
                if ptr.is_null() {
                    return Err(PyErr::new::<PyValueError, _>(
                        "PyDLPack capsule pointer is null",
                    ));
                }

                // Transfer ownership out by renaming the capsule -- after this
                // the capsule's own destructor becomes a no-op, and the
                // consumer is responsible for calling the tensor's deleter.
                let status = unsafe {
                    pyo3::ffi::PyCapsule_SetName(
                        capsule.as_ptr(), USED_DLTENSOR_VERSIONED_NAME.as_ptr(),
                    )
                };
                if status != 0 { return Err(PyErr::fetch(py)); }

                unsafe { *out = ptr; }
                Ok(())
            })(), py)
        })
    }));
    match attempt {
        Ok(rc) => rc,
        Err(_) => {
            unsafe {
                pyo3::ffi::PyErr_SetString(
                    pyo3::ffi::PyExc_RuntimeError,
                    pyo3::ffi::c_str!(
                        "panic in dlpk managed_tensor_from_py_object_no_sync"
                    ).as_ptr(),
                );
            }
            -1
        }
    }
}

unsafe extern "C" fn pydlpack_dltensor_from_py_object(
    py_object: *mut std::os::raw::c_void,
    out: *mut sys::DLTensor,
) -> i32 {
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Python::attach(|py| {
            into_ffi_rc((|| -> PyResult<()> {
                let obj = unsafe {
                    Bound::<PyAny>::from_borrowed_ptr(py, py_object.cast())
                };
                let bound: Bound<PyDLPack> = obj.downcast_into().map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!(
                        "exchange API got a non-PyDLPack object: {}", e,
                    ))
                })?;
                let self_ref = bound.borrow();
                // `as_dltensor` reads through the capsule and borrows the
                // producer-owned shape/strides/data; that matches the
                // "valid until control returns" guarantee expected by this
                // callback.
                let dl = self_ref.as_dltensor(py)?.clone();
                unsafe { *out = dl; }
                Ok(())
            })(), py)
        })
    }));
    match attempt {
        Ok(rc) => rc,
        Err(_) => {
            unsafe {
                pyo3::ffi::PyErr_SetString(
                    pyo3::ffi::PyExc_RuntimeError,
                    pyo3::ffi::c_str!(
                        "panic in dlpk dltensor_from_py_object_no_sync"
                    ).as_ptr(),
                );
            }
            -1
        }
    }
}

unsafe extern "C" fn pydlpack_managed_tensor_to_py_object(
    tensor: *mut sys::DLManagedTensorVersioned,
    out_py_object: *mut *mut std::os::raw::c_void,
) -> i32 {
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Python::attach(|py| {
            into_ffi_rc((|| -> PyResult<()> {
                let ptr = NonNull::new(tensor).ok_or_else(|| {
                    PyErr::new::<PyValueError, _>(
                        "managed_tensor_to_py_object got a null tensor",
                    )
                })?;
                // SAFETY: the caller transferred ownership of the tensor to
                // us via the DLPack contract.
                let dlpack_tensor = unsafe { DLPackTensor::from_ptr(ptr) };
                let py_dlpack = PyDLPack::try_from(dlpack_tensor)?;
                let py_obj = Py::new(py, py_dlpack)?;
                unsafe { *out_py_object = py_obj.into_ptr().cast(); }
                Ok(())
            })(), py)
        })
    }));
    match attempt {
        Ok(rc) => rc,
        Err(_) => {
            unsafe {
                pyo3::ffi::PyErr_SetString(
                    pyo3::ffi::PyExc_RuntimeError,
                    pyo3::ffi::c_str!(
                        "panic in dlpk managed_tensor_to_py_object_no_sync"
                    ).as_ptr(),
                );
            }
            -1
        }
    }
}

unsafe extern "C" fn pydlpack_current_work_stream(
    _device_type: sys::DLDeviceType,
    _device_id: i32,
    out_current_stream: *mut *mut std::os::raw::c_void,
) -> i32 {
    // dlpk does not own an accelerator stream on either CPU or GPU, so the
    // spec-compliant answer is "no stream, no sync". Consumers that need
    // stream ordering must coordinate with whoever produced the underlying
    // DLPackTensor before it was wrapped in PyDLPack.
    unsafe { *out_current_stream = std::ptr::null_mut(); }
    0
}


/*****************************************************************************/
/*                 __dlpack_c_exchange_api__ consumer helpers                */
/*****************************************************************************/

/// Borrowed handle to a producer's [`DLPackExchangeAPI`][sys::DLPackExchangeAPI]
/// function table, as exposed via `type(obj).__dlpack_c_exchange_api__`.
///
/// The underlying pointer must remain valid for the lifetime of the Python
/// interpreter per the DLPack spec, so this wrapper is `Copy` and does not
/// track a Rust lifetime. All calls still require the GIL to be held (enforced
/// by taking `&Bound<PyAny>` or a `Python<'_>` token).
#[derive(Clone, Copy)]
pub struct ExchangeAPI {
    raw: NonNull<sys::DLPackExchangeAPI>,
}

// The function pointers stored in DLPackExchangeAPI are plain addresses
// managed by the producer; `prev_api` is a back-link into the same table.
// The producer keeps the whole structure alive for the process lifetime,
// so moving a borrowed handle across threads (with the GIL held when
// calling into it) is sound.
unsafe impl Send for ExchangeAPI {}
unsafe impl Sync for ExchangeAPI {}

impl ExchangeAPI {
    /// Look up the exchange API on the Python type of `obj`.
    ///
    /// Returns `Ok(None)` if the type does not expose
    /// `__dlpack_c_exchange_api__`, `Err(...)` if the attribute is present
    /// but malformed (wrong capsule name, null pointer, incompatible major
    /// version).
    pub fn for_pyobject<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Self>> {
        let ty = obj.get_type();
        let cap_any = match ty.getattr("__dlpack_c_exchange_api__") {
            Ok(cap) => cap,
            Err(err) if err.is_instance_of::<PyAttributeError>(obj.py()) => {
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        let capsule: Bound<'py, PyCapsule> = cap_any.downcast_into().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!(
                "__dlpack_c_exchange_api__ must be a PyCapsule: {}", e,
            ))
        })?;

        let name = capsule.name()?;
        if name != Some(sys::DLPACK_EXCHANGE_API_CAPSULE_NAME) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "invalid exchange API capsule name: expected {:?}, got {:?}",
                sys::DLPACK_EXCHANGE_API_CAPSULE_NAME, name,
            )));
        }

        let raw = NonNull::new(capsule.pointer().cast::<sys::DLPackExchangeAPI>())
            .ok_or_else(|| PyErr::new::<PyValueError, _>(
                "exchange API capsule pointer is null",
            ))?;

        // SAFETY: pointer is non-null and, per the DLPack spec, points at a
        // live DLPackExchangeAPI owned by the producer for the whole process.
        let version = unsafe { raw.as_ref().header.version };
        if version.major != sys::DLPACK_MAJOR_VERSION {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "incompatible exchange API major version: producer exposes {}, \
                 this library requires {}. Callers can walk `header.prev_api` \
                 to find an older compatible table.",
                version.major, sys::DLPACK_MAJOR_VERSION,
            )));
        }

        Ok(Some(ExchangeAPI { raw }))
    }

    /// DLPack version the producer implements.
    pub fn version(&self) -> sys::DLPackVersion {
        // SAFETY: pointer is valid for 'static per the spec
        unsafe { self.raw.as_ref().header.version }
    }

    /// Query the producer's current work stream for `device`. On CPU the
    /// producer returns a null stream (and the caller has no sync work to do).
    pub fn current_work_stream(
        &self, py: Python<'_>, device: sys::DLDevice,
    ) -> PyResult<*mut c_void> {
        let f = unsafe { self.raw.as_ref().current_work_stream }.ok_or_else(|| {
            PyErr::new::<PyValueError, _>(
                "exchange API does not expose `current_work_stream`",
            )
        })?;

        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: `f` is a producer-owned function pointer, args match the
        // DLPackCurrentWorkStream signature, and we hold the GIL.
        let rc = unsafe { f(device.device_type, device.device_id, &mut out) };
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(out)
    }

    /// Fast-path import: ask the producer for an owning
    /// `DLManagedTensorVersioned` without going through
    /// `obj.__dlpack__()`.
    ///
    /// The producer does not perform stream synchronization; if the tensor
    /// lives on an accelerator device, the caller must query
    /// [`current_work_stream`][Self::current_work_stream] and launch
    /// dependent kernels on the producer's stream.
    pub fn managed_tensor_from_pyobject(
        &self, obj: &Bound<'_, PyAny>,
    ) -> PyResult<DLPackTensor> {
        let f = unsafe { self.raw.as_ref().managed_tensor_from_py_object_no_sync }
            .ok_or_else(|| PyErr::new::<PyValueError, _>(
                "exchange API does not expose `managed_tensor_from_py_object_no_sync`",
            ))?;

        let mut out: *mut DLManagedTensorVersioned = std::ptr::null_mut();
        // SAFETY: `f` is a producer-owned function pointer, args match the
        // DLPackManagedTensorFromPyObjectNoSync signature, and we hold the GIL.
        let rc = unsafe { f(obj.as_ptr().cast(), &mut out) };
        if rc != 0 {
            return Err(PyErr::fetch(obj.py()));
        }
        let ptr = NonNull::new(out).ok_or_else(|| PyErr::new::<PyValueError, _>(
            "producer returned a null tensor without setting a Python exception",
        ))?;

        // SAFETY: the producer transferred ownership of the tensor to us per
        // the `DLPackManagedTensorFromPyObjectNoSync` contract.
        Ok(unsafe { DLPackTensor::from_ptr(ptr) })
    }

    /// Fast-path borrow: ask the producer to fill a stack `DLTensor` for
    /// `obj` and call `f` with a read-only `DLPackTensorRef`.
    ///
    /// The shape/strides/data pointers on the filled tensor alias
    /// producer-owned storage and are only guaranteed valid for the
    /// duration of the closure -- the tensor ref must not escape.
    pub fn with_dltensor_from<R>(
        &self, obj: &Bound<'_, PyAny>,
        f: impl FnOnce(DLPackTensorRef<'_>) -> PyResult<R>,
    ) -> PyResult<R> {
        let cb = unsafe { self.raw.as_ref().dltensor_from_py_object_no_sync }
            .ok_or_else(|| PyErr::new::<PyValueError, _>(
                "exchange API does not expose `dltensor_from_py_object_no_sync`",
            ))?;

        let mut dltensor = sys::DLTensor {
            data: std::ptr::null_mut(),
            device: sys::DLDevice::cpu(),
            ndim: 0,
            dtype: sys::DLDataType {
                code: sys::DLDataTypeCode::kDLInt, bits: 0, lanes: 0,
            },
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // SAFETY: `cb` is a producer-owned function pointer, args match
        // the DLPackDLTensorFromPyObjectNoSync signature, GIL is held.
        let rc = unsafe { cb(obj.as_ptr().cast(), &mut dltensor) };
        if rc != 0 {
            return Err(PyErr::fetch(obj.py()));
        }

        // SAFETY: the borrow is constrained to the closure body, matching
        // the producer's "valid until control returns" guarantee.
        let tensor_ref = unsafe { DLPackTensorRef::from_raw(dltensor) };
        f(tensor_ref)
    }

    /// Export a Rust-owning `DLPackTensor` into a Python object of the
    /// producer's type, using the producer's fast path.
    ///
    /// Ownership of `tensor` transfers to the producer, which is
    /// responsible for freeing it (including via its own deleter).
    pub fn managed_tensor_to_pyobject<'py>(
        &self, py: Python<'py>, tensor: DLPackTensor,
    ) -> PyResult<Bound<'py, PyAny>> {
        let f = unsafe { self.raw.as_ref().managed_tensor_to_py_object_no_sync }
            .ok_or_else(|| PyErr::new::<PyValueError, _>(
                "exchange API does not expose `managed_tensor_to_py_object_no_sync`",
            ))?;

        let raw = tensor.into_raw();
        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: `f` is a producer-owned function pointer; we hand off
        // ownership of `raw` per the contract, GIL is held.
        let rc = unsafe { f(raw.as_ptr(), &mut out) };
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }
        // SAFETY: on success the producer returns an owned PyObject* via
        // the `*mut c_void` slot.
        unsafe { Bound::from_owned_ptr_or_err(py, out.cast::<pyo3::ffi::PyObject>()) }
    }
}


/*****************************************************************************/

#[cfg(test)]
mod tests {
    use crate::{DLPackTensor, DLPackTensorRef, GetDLPackDataType};
    use crate::sys::{DLPackVersion, DLDevice, DLManagedTensorVersioned, DLTensor};

    use super::PyDLPack;
    use super::DLTENSOR_VERSIONED_NAME;

    use ndarray::{Array, ArrayView2};
    use pyo3::ffi::c_str;
    use pyo3::prelude::*;
    use pyo3::types::{PyCapsule, PyDict};

    macro_rules! test_numpy_to_ndarray_via_dlpack_dtype {
        ($test_name:ident, $rust_type:ty, $np_dtype:expr) => {
            #[test]
            #[cfg_attr(miri, ignore)]
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
            #[cfg_attr(miri, ignore)]
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

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_null_strides_fails_conversion() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let mut shape = vec![2i64];
            let mut data = vec![1.0f32, 2.0];

            let dl_tensor = DLTensor {
                data: data.as_mut_ptr().cast(),
                device: DLDevice::cpu(),
                ndim: 1,
                dtype: f32::get_dlpack_data_type(),
                shape: shape.as_mut_ptr(),
                strides: std::ptr::null_mut(), // Invalid for ndim > 0 in v1.2+
                byte_offset: 0,
            };

            let managed = Box::into_raw(Box::new(DLManagedTensorVersioned {
                version: DLPackVersion::current(),
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
                flags: 0,
                dl_tensor,
            }));

            let capsule = unsafe {
                Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyCapsule_New(
                    managed.cast(),
                    DLTENSOR_VERSIONED_NAME.as_ptr(),
                    None,
                ))?.cast_into_unchecked::<PyCapsule>()
            };

            let result = DLPackTensor::try_from(&capsule);
            assert!(result.is_err());

            // Cleanup the leaked memory since the capsule didn't take ownership
            unsafe { drop(Box::from_raw(managed)); }
            Ok(())
        })
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_v1_0_null_strides_allowed() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let mut shape = vec![2i64];
            let mut data = vec![1.0f32, 2.0];

            let dl_tensor = DLTensor {
                data: data.as_mut_ptr().cast(),
                device: DLDevice::cpu(),
                ndim: 1,
                dtype: f32::get_dlpack_data_type(),
                shape: shape.as_mut_ptr(),
                strides: std::ptr::null_mut(), // Legal in v1.0
                byte_offset: 0,
            };

            let managed = Box::into_raw(Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
                flags: 0,
                dl_tensor,
            }));

            let capsule = unsafe {
                Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyCapsule_New(
                    managed.cast(),
                    DLTENSOR_VERSIONED_NAME.as_ptr(),
                    None,
                ))?.cast_into_unchecked::<PyCapsule>()
            };

            let result = DLPackTensor::try_from(&capsule);
            assert!(result.is_ok(), "Legacy v1.0 tensors should permit NULL strides");

            Ok(())
        })
    }

    // ---- __dlpack_c_exchange_api__ ----

    use super::ExchangeAPI;
    use crate::sys::{self, DLPackExchangeAPI, DLPackExchangeAPIHeader};

    // The fake producer exposes a DLPackExchangeAPI whose
    // `managed_tensor_from_py_object_no_sync` ignores the incoming PyObject
    // and hands back a heap-allocated 1D float32 tensor with data [1, 2, 3].
    // We test the consumer wrapper against that.

    unsafe extern "C" fn mock_stream_cpu(
        _device_type: sys::DLDeviceType, _device_id: i32, out: *mut *mut std::ffi::c_void,
    ) -> i32 {
        *out = std::ptr::null_mut();
        0
    }

    struct MockTensorOwner {
        _data: Box<[f32; 3]>,
        _shape: Box<[i64; 1]>,
        _strides: Box<[i64; 1]>,
    }

    unsafe extern "C" fn mock_tensor_deleter(tensor: *mut DLManagedTensorVersioned) {
        let owner = (*tensor).manager_ctx.cast::<MockTensorOwner>();
        drop(Box::from_raw(owner));
        drop(Box::from_raw(tensor));
    }

    unsafe extern "C" fn mock_from_py_object(
        _py_object: *mut std::ffi::c_void,
        out: *mut *mut DLManagedTensorVersioned,
    ) -> i32 {
        let mut data = Box::new([1.0f32, 2.0, 3.0]);
        let mut shape = Box::new([3i64]);
        let mut strides = Box::new([1i64]);

        let dl_tensor = DLTensor {
            data: data.as_mut_ptr().cast(),
            device: DLDevice::cpu(),
            ndim: 1,
            dtype: f32::get_dlpack_data_type(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let owner = Box::into_raw(Box::new(MockTensorOwner {
            _data: data, _shape: shape, _strides: strides,
        }));

        let tensor = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion::current(),
            manager_ctx: owner.cast(),
            deleter: Some(mock_tensor_deleter),
            flags: sys::DLPACK_FLAG_BITMASK_IS_COPIED,
            dl_tensor,
        });

        *out = Box::into_raw(tensor);
        0
    }

    // Leak the exchange API struct into a PyCapsule and attach it as a
    // class attribute on a Python type. This mirrors what a real producer
    // library would do at startup.
    fn make_mock_producer_instance<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let api = Box::new(DLPackExchangeAPI {
            header: DLPackExchangeAPIHeader {
                version: DLPackVersion::current(),
                prev_api: std::ptr::null_mut(),
            },
            managed_tensor_allocator: None,
            managed_tensor_from_py_object_no_sync: Some(mock_from_py_object),
            managed_tensor_to_py_object_no_sync: None,
            dltensor_from_py_object_no_sync: None,
            current_work_stream: Some(mock_stream_cpu),
        });
        let api_ptr: *mut DLPackExchangeAPI = Box::into_raw(api);

        let capsule = unsafe {
            Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyCapsule_New(
                api_ptr.cast(),
                sys::DLPACK_EXCHANGE_API_CAPSULE_NAME.as_ptr(),
                None,  // the mock producer owns the struct for the process lifetime
            ))?.cast_into_unchecked::<PyCapsule>()
        };

        let locals = PyDict::new(py);
        locals.set_item("api", capsule)?;
        // Use the same dict as globals so the class-body lookup of `api`
        // resolves. With `exec(code, None, locals)` Python's class bodies
        // do not see names defined in the outer `locals`.
        py.run(c_str!("
class FakeProducer:
    __dlpack_c_exchange_api__ = api
obj = FakeProducer()
"), Some(&locals), Some(&locals))?;
        Ok(locals.get_item("obj")?.unwrap())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_exchange_api_missing_returns_none() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(c_str!("
class Plain: pass
obj = Plain()
"), Some(&locals), Some(&locals))?;
            let obj = locals.get_item("obj")?.unwrap();
            assert!(ExchangeAPI::for_pyobject(&obj)?.is_none());
            Ok(())
        })
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_exchange_api_fast_import() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let obj = make_mock_producer_instance(py)?;

            let api = ExchangeAPI::for_pyobject(&obj)?.expect("api present");
            assert_eq!(api.version().major, sys::DLPACK_MAJOR_VERSION);

            // CPU stream query: producer reports null.
            let stream = api.current_work_stream(py, DLDevice::cpu())?;
            assert!(stream.is_null());

            // Fast-path import: producer hands back the mock tensor [1, 2, 3].
            let tensor = api.managed_tensor_from_pyobject(&obj)?;
            let view: ndarray::ArrayView1<f32> = tensor.as_ref().try_into().unwrap();
            assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
            Ok(())
        })
    }

    // ----- PyDLPack as producer (the class attribute is wired up to a
    // static DLPackExchangeAPI with all four callbacks we implement). -----

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_pydlpack_exposes_exchange_api() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            // Build a Rust tensor, wrap it in PyDLPack, and look up the
            // exchange API through the class attribute -- the consumer
            // roundtrip.
            let array = ndarray::arr1(&[10.0f32, 20.0, 30.0, 40.0]);
            let dl_tensor = DLPackTensor::try_from(array).unwrap();
            let py_dlpack = PyDLPack::try_from(dl_tensor).unwrap();
            let obj = Py::new(py, py_dlpack)?.into_bound(py).into_any();

            let api = ExchangeAPI::for_pyobject(&obj)?
                .expect("PyDLPack should expose __dlpack_c_exchange_api__");
            assert_eq!(api.version().major, sys::DLPACK_MAJOR_VERSION);

            // Fast-path borrow through the producer's dltensor callback.
            let sum = api.with_dltensor_from(&obj, |t| {
                let view: ndarray::ArrayView1<f32> = t.try_into().unwrap();
                Ok(view.iter().sum::<f32>())
            })?;
            assert_eq!(sum, 100.0);

            // Fast-path consume: hand ownership off to the consumer side.
            let tensor: DLPackTensor = api.managed_tensor_from_pyobject(&obj)?;
            let view: ndarray::ArrayView1<f32> = tensor.as_ref().try_into().unwrap();
            assert_eq!(view.as_slice().unwrap(), &[10.0, 20.0, 30.0, 40.0]);

            // Second fast-path consume must fail -- the capsule is already
            // renamed to used_dltensor_versioned.
            let again = api.managed_tensor_from_pyobject(&obj);
            assert!(again.is_err(),
                "second fast-path consume should fail after ownership transfer");
            Ok(())
        })
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_pydlpack_managed_tensor_to_py_object_roundtrip() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            // Start with a Rust tensor, wrap as PyDLPack to get access to
            // the exchange API, then bounce a separate Rust tensor back
            // into a Python object via managed_tensor_to_pyobject.
            let pytensor = PyDLPack::try_from(
                DLPackTensor::try_from(ndarray::arr1(&[1.0f32])).unwrap(),
            ).unwrap();
            let handle = Py::new(py, pytensor)?.into_bound(py).into_any();
            let api = ExchangeAPI::for_pyobject(&handle)?.expect("api present");

            // Now ask the producer to re-wrap a different Rust-owned tensor
            // as one of its own Python objects (a PyDLPack).
            let outbound = DLPackTensor::try_from(
                ndarray::arr1(&[7.0f32, 8.0, 9.0]),
            ).unwrap();
            let wrapped = api.managed_tensor_to_pyobject(py, outbound)?;

            // Round-trip back through __dlpack__() and check the values.
            let capsule_obj = wrapped.call_method0("__dlpack__")?;
            let capsule: Bound<PyCapsule> = capsule_obj.extract()?;
            let imported = DLPackTensor::try_from(&capsule)?;
            let view: ndarray::ArrayView1<f32> = imported.as_ref().try_into().unwrap();
            assert_eq!(view.as_slice().unwrap(), &[7.0, 8.0, 9.0]);
            Ok(())
        })
    }
}
