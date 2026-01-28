//! DLPack integration for Rust.
//!
//! This crate provides wrappers around the DLPack C API, allowing to exchange
//! tensors with other libraries that support DLPack, such as PyTorch,
//! TensorFlow, MXNet, CuPy, JAX, etc.
//!
//! The C API is available in the `sys` module, and the following higher-level
//! wrappers are provided:
//!
//! - [`DLPackTensor`]: an owned DLPack tensor
//! - [`DLPackTensorRef`]: a borrowed DLPack tensor (immutable)
//! - [`DLPackTensorRefMut`]: a borrowed DLPack tensor (mutable)
//!
//! We also provide tools to convert from and to other rust types, using the
//! `TryFrom` trait.
//!
//! ## Features
//!
//! - [`ndarray`]: enable conversion from and to `ndarray::Array`
//! - [`pyo3`]: enable passing data from and to python, following the dlpack
//!   protocol (the data is passed through a `PyCapsule` object).


#![allow(clippy::needless_return, clippy::redundant_field_names)]

use std::{ffi::c_void, ptr::NonNull};

pub mod sys;

mod data_types;
pub use self::data_types::GetDLPackDataType;

pub use self::data_types::CastError;
use self::data_types::DLPackPointerCast;

impl sys::DLPackVersion {
    /// Returns the DLPack version supported by this library.
    pub const fn current() -> Self {
        Self {
            major: sys::DLPACK_MAJOR_VERSION,
            minor: sys::DLPACK_MINOR_VERSION,
        }
    }
}

impl sys::DLDevice {
    /// Get a CPU device (kDLCPU), the `device_id` is always 0.
    pub const fn cpu() -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLCPU,
            device_id: 0,
        }
    }

    /// Get a CUDA GPU device (kDLCUDA) with the given `device_id`.
    pub const fn cuda(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLCUDA,
            device_id,
        }
    }

    /// Get a pinned CUDA CPU memory device (kDLCUDAHost) allocated by `cudaMallocHost`.
    pub const fn cuda_host() -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLCUDAHost,
            device_id: 0,
        }
    }

    /// Get an OpenCL device (kDLOpenCL) with the given `device_id`.
    pub const fn opencl(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLOpenCL,
            device_id,
        }
    }

    /// Get a Vulkan device (kDLVulkan) with the given `device_id` for next generation graphics.
    pub const fn vulkan(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLVulkan,
            device_id,
        }
    }

    /// Get a Metal device (kDLMetal) with the given `device_id` for Apple GPUs.
    pub const fn metal(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLMetal,
            device_id,
        }
    }

    /// Get a VPI device (kDLVPI) with the given `device_id` for Verilog simulator buffers.
    pub const fn vpi(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLVPI,
            device_id,
        }
    }

    /// Get a ROCm device (kDLROCM) with the given `device_id` for AMD GPUs.
    pub const fn rocm(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLROCM,
            device_id,
        }
    }

    /// Get a pinned ROCm CPU memory device (kDLROCMHost) allocated by `hipMallocHost`.
    pub const fn rocm_host() -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLROCMHost,
            device_id: 0,
        }
    }

    /// Get a reserved extension device (kDLExtDev) with the given `device_id`.
    pub const fn ext_dev(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLExtDev,
            device_id,
        }
    }

    /// Get a CUDA managed/unified memory device (kDLCUDAManaged) allocated by
    /// `cudaMallocManaged`, the `device_id` is always 0.
    pub const fn cuda_managed() -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLCUDAManaged,
            device_id: 0,
        }
    }

    /// Get a oneAPI device (kDLOneAPI) for unified shared memory on a
    /// non-partitioned device, the `device_id` is always 0.
    pub const fn oneapi() -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLOneAPI,
            device_id: 0,
        }
    }

    /// Get a WebGPU device (kDLWebGPU) with the given `device_id` for the next
    /// generation standard.
    pub const fn webgpu(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLWebGPU,
            device_id,
        }
    }

    /// Get a Qualcomm Hexagon DSP device (kDLHexagon) with the given `device_id`.
    pub const fn hexagon(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLHexagon,
            device_id,
        }
    }

    /// Get a Microsoft MAIA device (kDLMAIA) with the given `device_id`.
    pub const fn maia(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLMAIA,
            device_id,
        }
    }

    /// Get an AWS Trainium device (kDLTrn) with the given `device_id`.
    pub const fn trn(device_id: i32) -> Self {
        Self {
            device_type: sys::DLDeviceType::kDLTrn,
            device_id,
        }
    }
}

/// A managed DLPack tensor, carrying ownership of the data.
///
/// Convertion from and to other array types is handled though the different
/// `TryFrom` implementations.
#[repr(transparent)]
pub struct DLPackTensor{
    raw: NonNull<sys::DLManagedTensorVersioned>,
}

impl Drop for DLPackTensor {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.raw.as_ref().deleter {
                deleter(self.raw.as_ptr());
            }
        }
    }
}

struct RustBoxedManager {
    // tensor: std::pin::Pin<sys::DLManagedTensorVersioned>,
    original_ctx: *mut c_void,
    original_deleter: Option<unsafe extern "C" fn(*mut sys::DLManagedTensorVersioned) -> ()>,
}

unsafe extern "C" fn rust_boxed_manager_deleter(tensor: *mut sys::DLManagedTensorVersioned) {
    if tensor.is_null() {
        return;
    }

    let manager = (*tensor).manager_ctx.cast::<RustBoxedManager>();
    assert!(!manager.is_null());

    (*tensor).manager_ctx = (*manager).original_ctx;
    (*tensor).deleter = (*manager).original_deleter;

    if let Some(deleter) = (*tensor).deleter {
        deleter(tensor);
    }

    std::mem::drop(Box::from_raw(manager));
    std::mem::drop(Box::from_raw(tensor));
}

impl DLPackTensor {
    /// Create a `DLPackTensor` from a raw `DLManagedTensorVersioned`.
    ///
    /// # Safety
    ///
    /// The `DLManagedTensorVersioned` should have a valid `deleter` that can
    /// be called from Rust, or have the deleter set to `None`.
    pub unsafe fn from_raw(mut tensor: sys::DLManagedTensorVersioned) -> DLPackTensor {
        // we need to move the tensor to the heap, so we need to wrap the
        // manager_ctx and deleter into another one that will also free the
        // tensor from the heap.
        let manager = Box::new(RustBoxedManager {
            original_ctx: tensor.manager_ctx,
            original_deleter: tensor.deleter,
        });
        tensor.manager_ctx = Box::into_raw(manager).cast();
        tensor.deleter = Some(rust_boxed_manager_deleter);

        let tensor = Box::new(tensor);

        return DLPackTensor{
            raw: NonNull::new_unchecked(Box::into_raw(tensor)),
        };
    }

    /// Create a `DLPackTensor` from a non-null pointer to `DLManagedTensorVersioned`.
    ///
    /// # Safety
    ///
    /// The `DLManagedTensorVersioned` should have a valid `deleter` that can
    /// be called from Rust, or have the deleter set to `None`.
    pub unsafe fn from_ptr(tensor: NonNull<sys::DLManagedTensorVersioned>) -> DLPackTensor {
        if tensor.as_ref().version.major != sys::DLPACK_MAJOR_VERSION {
            // from the spec, we need to call the deleter here (and it is the
            // only thing we can do)
            if let Some(deleter) = tensor.as_ref().deleter {
                deleter(tensor.as_ptr());
            }
            panic!(
                "Incompatible DLPack version, got {}, but this code only supports version {}",
                tensor.as_ref().version.major, sys::DLPACK_MAJOR_VERSION
            );
        }
        return DLPackTensor{
            raw: tensor
        };
    }

    /// Get a DLPack tensor reference from this owned tensor
    pub fn as_ref(&self) -> DLPackTensorRef<'_> {
        unsafe {
            // SAFETY: we are constaining the returned reference lifetime
            DLPackTensorRef::from_raw(self.raw.as_ref().dl_tensor.clone())
        }
    }

    /// Get a mutable DLPack tensor reference from this owned tensor
    pub fn as_mut(&mut self) -> DLPackTensorRefMut<'_> {
        unsafe {
            // only if NOT read only + unique
            assert!(self.raw.as_ref().flags & sys::DLPACK_FLAG_BITMASK_IS_COPIED == 0, "Can not create a mutable reference to a borrowed tensor");
            assert!(self.raw.as_ref().flags & sys::DLPACK_FLAG_BITMASK_READ_ONLY != 0, "Can not create a mutable reference to a read-only tensor");

            let version = self.raw.as_ref().version;
            // SAFETY: we are constraining the returned reference lifetime
            DLPackTensorRefMut::from_raw_with_version(self.raw.as_ref().dl_tensor.clone(), Some(version))
         }
     }

    /// Get a pointer to data in this tensor. This pointer can be a device
    /// pointer according to [`DLPackTensor::device`].
    pub fn data_ptr<T>(&self) -> Result<*const T, CastError>  where T: DLPackPointerCast {
        self.as_ref().data_ptr()
    }

    /// Get a mutable pointer to data in this tensor. This pointer can be a
    /// device pointer according to [`DLPackTensor::device`].
    pub fn data_ptr_mut<T>(&mut self) -> Result<*mut T, CastError>  where T: DLPackPointerCast {
        self.as_mut().data_ptr_mut()
    }

    /// Get the device where the data of this tensor lives.
    pub fn device(&self) -> sys::DLDevice {
        self.as_ref().device()
    }

    /// Get the data type of this tensor
    pub fn dtype(&self) -> sys::DLDataType {
        self.as_ref().dtype()
    }

    /// Get the number of dimensions of this tensor
    pub fn n_dims(&self) -> usize {
        self.as_ref().n_dims()
    }

    /// Get the shape of this tensor
    pub fn shape(&self) -> &[i64] {
        unsafe {
            assert!(!self.raw.as_ref().dl_tensor.shape.is_null());
            return std::slice::from_raw_parts(self.raw.as_ref().dl_tensor.shape, self.n_dims());
        }
    }

    /// Get the strides of this tensor, if any
    pub fn strides(&self) -> Option<&[i64]> {
        unsafe {
            if self.raw.as_ref().dl_tensor.strides.is_null() {
                return None;
            }
            return Some(std::slice::from_raw_parts(self.raw.as_ref().dl_tensor.strides, self.n_dims()));
        }
    }

    /// Get the byte offset of this tensor, i.e. how many bytes should be added
    /// to [`DLPackTensor::data_ptr`] and [`DLPackTensor::data_ptr_mut`] to get
    /// the first element of the tensor.
    pub fn byte_offset(&self) -> usize {
        self.as_ref().byte_offset()
    }

    /// Get a reference to the underlying DLTensor
    pub fn as_dltensor(&self) -> &sys::DLTensor {
        unsafe {
            &self.raw.as_ref().dl_tensor
        }
    }

    /// Consumes the `DLPackTensor`, returning the underlying raw pointer.
    ///
    /// # Safety
    /// 
    /// The caller is responsible for managing the memory and calling the deleter
    /// when the tensor is no longer needed.
    pub fn into_raw(self) -> NonNull<sys::DLManagedTensorVersioned>{
        let raw = self.raw;
        // forget so Drop and ergo the deleter are not called
        std::mem::forget(self);
        return raw;
    }
}

/// A reference to a DLPack tensor, with data borrowed from some owner,
/// potentially in another language.
pub struct DLPackTensorRef<'a> {
    pub raw: sys::DLTensor,
    version: Option<sys::DLPackVersion>,
    phantom: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> DLPackTensorRef<'a> {
    /// Create a `DLPackTensorRef` from a raw `DLTensor`
    ///
    /// # Safety
    ///
    /// The lifetime of the returned reference should be constrained to the
    /// actual lifetime of the `DLTensor`.
    pub unsafe fn from_raw(tensor: sys::DLTensor) -> DLPackTensorRef<'a> {
        DLPackTensorRef {
            raw: tensor,
            version: None,
            phantom: std::marker::PhantomData
        }
    }

    /// Create a `DLPackTensorRef` from a raw `DLTensor` with an explicit version
    pub unsafe fn from_raw_with_version(tensor: sys::DLTensor, version: Option<sys::DLPackVersion>) -> DLPackTensorRef<'a> {
        DLPackTensorRef {
            raw: tensor,
            version,
            phantom: std::marker::PhantomData
        }
    }

    /// Get a pointer to data in this tensor. This pointer can be a device
    /// pointer according to [`DLPackTensorRef::device`].
    pub fn data_ptr<T>(&self) -> Result<*const T, CastError>  where T: DLPackPointerCast {
        let ptr = self.raw.data.wrapping_add(self.byte_offset());
        return T::dlpack_ptr_cast(ptr, self.raw.dtype).map(|p| p.cast_const());
    }

    /// Get the device where the data of this tensor lives.
    pub fn device(&self) -> sys::DLDevice {
        return self.raw.device;
    }

    /// Get the data type of this tensor
    pub fn dtype(&self) -> sys::DLDataType {
        return self.raw.dtype;
    }

    /// Get the number of dimensions of this tensor
    pub fn n_dims(&self) -> usize {
        return self.raw.ndim as usize;
    }

    /// Get the shape of this tensor
    pub fn shape(&self) -> &[i64] {
        assert!(!self.raw.shape.is_null());
        unsafe {
            return std::slice::from_raw_parts(self.raw.shape, self.n_dims());
        }
    }

    /// Get the strides of this tensor, if any
    pub fn strides(&self) -> Option<&[i64]> {
        if self.raw.strides.is_null() {
            return None;
        }
        unsafe {
            return Some(std::slice::from_raw_parts(self.raw.strides, self.n_dims()));
        }
    }

    /// Get the byte offset of this tensor, i.e. how many bytes should be added
    /// to [`DLPackTensorRef::data_ptr`] to get the first element of the tensor.
    pub fn byte_offset(&self) -> usize {
        return self.raw.byte_offset as usize;
    }

    /// Get a reference to the underlying DLTensor
    pub fn as_dltensor(&self) -> &sys::DLTensor {
        &self.raw
    }

    /// Get the DLPack version of this tensor
    pub fn version(&self) -> Option<sys::DLPackVersion> {
        self.version
    }
}

/// A mutable reference to a DLPack tensor, with data borrowed from some owner,
/// potentially in another language.
pub struct DLPackTensorRefMut<'a> {
    raw: sys::DLTensor,
    pub version: Option<sys::DLPackVersion>,
    phantom: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> DLPackTensorRefMut<'a> {
    /// Create a `DLPackTensorRefMut` from a raw `DLTensor`
    ///
    /// # Safety
    ///
    /// The lifetime of the returned reference should be constrained to the
    /// actual lifetime of the `DLTensor`. The `DLTensor` should also be
    /// mutable, and there should not be any other references (mutable or not)
    /// to the same `DLTensor`.
    pub unsafe fn from_raw(tensor: sys::DLTensor) -> DLPackTensorRefMut<'a> {
        DLPackTensorRefMut {
            raw: tensor,
            version: None,
            phantom: std::marker::PhantomData
        }
    }

    /// Create a `DLPackTensorRefMut` from a raw `DLTensor` with an explicit version
    pub unsafe fn from_raw_with_version(tensor: sys::DLTensor, version: Option<sys::DLPackVersion>) -> DLPackTensorRefMut<'a> {
        DLPackTensorRefMut {
            raw: tensor,
            version,
            phantom: std::marker::PhantomData
        }
    }

    /// Convert this mutable reference to an immutable reference.
    pub fn as_ref(&self) -> DLPackTensorRef<'_> {
        unsafe {
            // SAFETY: we are constaining the returned reference lifetime
            DLPackTensorRef::from_raw_with_version(self.raw.clone(), self.version)
        }
    }

    /// Get a pointer to data in this tensor. This pointer can be a device
    /// pointer according to [`DLPackTensorRefMut::device`].
    pub fn data_ptr<T>(&self) -> Result<*const T, CastError>  where T: DLPackPointerCast {
        let ptr = self.raw.data.wrapping_add(self.byte_offset());
        return T::dlpack_ptr_cast(ptr, self.raw.dtype).map(|p| p.cast_const());
    }

    /// Get a mutable pointer to data in this tensor. This pointer can be a
    /// device pointer according to [`DLPackTensorRefMut::device`].
    pub fn data_ptr_mut<T>(&mut self) -> Result<*mut T, CastError>  where T: DLPackPointerCast {
        let ptr = self.raw.data.wrapping_add(self.byte_offset());
        return T::dlpack_ptr_cast(ptr, self.raw.dtype);
    }

    /// Get the device where the data of this tensor lives.
    pub fn device(&self) -> sys::DLDevice {
        return self.raw.device;
    }

    /// Get the data type of this tensor
    pub fn dtype(&self) -> sys::DLDataType {
        return self.raw.dtype;
    }

    /// Get the number of dimensions of this tensor
    pub fn n_dims(&self) -> usize {
        return self.raw.ndim as usize;
    }

    /// Get the shape of this tensor
    pub fn shape(&self) -> &[i64] {
        assert!(!self.raw.shape.is_null());
        unsafe {
            return std::slice::from_raw_parts(self.raw.shape, self.n_dims());
        }
    }

    /// Get the strides of this tensor, if any
    pub fn strides(&self) -> Option<&[i64]> {
        if self.raw.strides.is_null() {
            return None;
        }
        unsafe {
            return Some(std::slice::from_raw_parts(self.raw.strides, self.n_dims()));
        }
    }

    /// Get the byte offset of this tensor, i.e. how many bytes should be added
    /// to [`DLPackTensorRefMut::data_ptr`] and
    /// [`DLPackTensorRefMut::data_ptr_mut`] to get the first element of the
    /// tensor.
    pub fn byte_offset(&self) -> usize {
        return self.raw.byte_offset as usize;
    }

    /// Get a reference to the underlying DLTensor
    pub fn as_dltensor(&self) -> &sys::DLTensor {
        &self.raw
    }
    /// Get the version of the tensor if known
    pub fn version(&self) -> Option<sys::DLPackVersion> {
        self.version
    }
}

#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "pyo3")]
pub mod pyo3;
