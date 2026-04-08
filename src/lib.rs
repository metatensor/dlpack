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
//! `TryFrom` trait. Conversions to and from `std::vec::Vec` and slices are
//! always enabled, in the [`vec`] module.
//!
//! ## Features
//!
//! - [`sync`]: enable sharing data behind `Arc<Mutex<_>>` and `Arc<RwLock<_>>`
//!   with DLPack, enforcing Rust's borrowing rules at runtime through the
//!   locks.
//! - [`ndarray`]: enable conversion from and to `ndarray::Array`
//! - [`pyo3`]: enable passing data from and to python, following the dlpack
//!   protocol (the data is passed through a `PyCapsule` object).


#![allow(clippy::needless_return, clippy::redundant_field_names)]
#![forbid(clippy::as_ptr_cast_mut, clippy::ptr_cast_constness)]

use std::{ffi::c_void, ptr::NonNull};

pub mod sys;
pub use self::sys::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLPackVersion};

mod data_types;
pub use self::data_types::GetDLPackDataType;

pub use self::data_types::CastError;
pub use self::data_types::DLPackPointerCast;

impl DLPackVersion {
    /// Returns the DLPack version supported by this library.
    pub const fn current() -> Self {
        Self {
            major: sys::DLPACK_MAJOR_VERSION,
            minor: sys::DLPACK_MINOR_VERSION,
        }
    }
}

impl DLDevice {
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

struct DLTensorDebug<'a>(&'a sys::DLTensor);
impl std::fmt::Debug for DLTensorDebug<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tensor_ref = unsafe {
            DLPackTensorRef::from_raw(self.0.clone())
        };
        debug_tensor(&tensor_ref, f, "DLTensor")
    }
}

fn debug_tensor(tensor: &DLPackTensorRef<'_>, f: &mut std::fmt::Formatter<'_>, name: &str) -> std::fmt::Result {
    f.debug_struct(name)
            .field("data", &tensor.raw.data)
            .field("device", &tensor.device())
            .field("dtype", &tensor.dtype())
            .field("shape", &tensor.shape())
            .field("strides", &tensor.strides())
            .field("byte_offset", &tensor.byte_offset())
            .finish()
}

impl std::fmt::Debug for DLPackTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut flags_strings = Vec::new();
        let flags = self.flags();
        if flags & sys::DLPACK_FLAG_BITMASK_READ_ONLY != 0 {
            flags_strings.push("READ_ONLY");
        }
        if flags & sys::DLPACK_FLAG_BITMASK_IS_COPIED != 0 {
            flags_strings.push("IS_COPIED");
        }
        if flags & sys::DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED != 0 {
            flags_strings.push("IS_SUBBYTE_TYPE_PADDED");
        }

        let flags_string = if flags_strings.is_empty() {
            "0".into()
        } else {
            flags_strings.join(" | ")
        };

        f.debug_struct("DLPackTensor")
            .field("version", unsafe { &self.raw.as_ref().version })
            .field("manager_ctx", unsafe { &self.raw.as_ref().manager_ctx })
            .field("deleter", unsafe { &self.raw.as_ref().deleter })
            .field("flags", &flags_string)
            .field("dl_tensor", unsafe { &DLTensorDebug(&self.raw.as_ref().dl_tensor) })
            .finish()
    }
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
        if self.is_read_only() {
            panic!("Can not get a `DLPackTensorRefMut`: tensor is explicitly marked READ_ONLY");
        }

        // FIXME: ideally we should also check the IS_COPIED/IS_OWNED bit to
        // ensure that we are the unique owner of the tensor, but some libraries
        // (e.g. PyTorch) don't set this bit correctly, so for now we just
        // ignore it and let the user deal with potential issues if they mutate
        // a non-unique tensor.

        unsafe {
            // SAFETY: we are constraining the returned reference lifetime
            // the caller must ensure that the uniqueness check doesn't apply
            // i.e. they're fine mutating an ArcArray with refcount > 1
            DLPackTensorRefMut::from_raw(self.raw.as_ref().dl_tensor.clone())
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
        if self.n_dims() == 0 {
            // for 0-dim tensors, the shape pointer can be null, so we should
            // return an empty slice instead of panicking.
            return &[];
        }

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

    /// Get the raw flags bitfield.
    pub fn flags(&self) -> u64 {
        unsafe { self.raw.as_ref().flags }
    }

    /// Check if the tensor is explicitly marked as read-only.
    pub fn is_read_only(&self) -> bool {
        self.flags() & sys::DLPACK_FLAG_BITMASK_READ_ONLY != 0
    }

    /// Check if the tensor is unique/owned (IS_COPIED bit).
    pub fn is_copied(&self) -> bool {
        self.flags() & sys::DLPACK_FLAG_BITMASK_IS_COPIED != 0
    }

    /// Check if the sub-byte types (fp4, fp6) are padded to the next byte, or
    /// packed together.
    pub fn is_subbyte_type_padded(&self) -> bool {
        self.flags() & sys::DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED != 0
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
    phantom: std::marker::PhantomData<&'a [u8]>,
}

impl std::fmt::Debug for DLPackTensorRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        debug_tensor(self, f, "DLPackTensorRef")
    }
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
}

/// A mutable reference to a DLPack tensor, with data borrowed from some owner,
/// potentially in another language.
pub struct DLPackTensorRefMut<'a> {
    raw: sys::DLTensor,
    phantom: std::marker::PhantomData<&'a [u8]>,
}

impl std::fmt::Debug for DLPackTensorRefMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        debug_tensor(&self.as_ref(), f, "DLPackTensorRef")
    }
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
            phantom: std::marker::PhantomData
        }
    }

    /// Convert this mutable reference to an immutable reference.
    pub fn as_ref(&self) -> DLPackTensorRef<'_> {
        unsafe {
            // SAFETY: we are constaining the returned reference lifetime
            DLPackTensorRef::from_raw(self.raw.clone())
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
}

pub mod vec;

#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "pyo3")]
pub mod pyo3;
#[cfg(feature = "sync")]
pub mod sync;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug() {
        let data = vec![1, 2, 3, 4, 5];
        let tensor: DLPackTensor = data.try_into().unwrap();

        // get the pointer addresse that can change every time, so we can put it in the expected string
        let manager_ctx = format!("{:#?}", unsafe { tensor.raw.as_ref().manager_ctx });
        let deleter = format!("{:#?}", unsafe { tensor.raw.as_ref().deleter.unwrap() });
        let data = format!("{:#?}", unsafe { tensor.raw.as_ref().dl_tensor.data });

        let expected = format!(r#"DLPackTensor {{
    version: DLPackVersion {{
        major: 1,
        minor: 3,
    }},
    manager_ctx: {manager_ctx},
    deleter: Some(
        {deleter},
    ),
    flags: "IS_COPIED",
    dl_tensor: DLTensor {{
        data: {data},
        device: DLDevice {{
            device_type: kDLCPU,
            device_id: 0,
        }},
        dtype: DLDataType {{
            code: kDLInt,
            bits: 32,
            lanes: 1,
        }},
        shape: [
            5,
        ],
        strides: Some(
            [
                1,
            ],
        ),
        byte_offset: 0,
    }},
}}"#, );

        assert_eq!(format!("{:#?}", tensor), expected);
    }
}
