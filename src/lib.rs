#![allow(clippy::needless_return, clippy::redundant_field_names)]

pub mod sys;

mod data_types;
pub use self::data_types::CastError;
use self::data_types::DLPackPointerCast;

/// A managed DLPack tensor, carrying ownership of the data.
///
/// Convertion from and to other array types is handled though the different
/// `TryFrom` implementations.
pub struct DLPackTensor {
    raw: sys::DLManagedTensorVersioned,
}

impl Drop for DLPackTensor {
    fn drop(&mut self) {
        if let Some(deleter) = self.raw.deleter {
            unsafe {
                deleter(&mut self.raw);
            }
        }
    }
}

impl DLPackTensor {
    /// Create a `DLPackTensor` from a raw `DLManagedTensorVersioned`.
    ///
    /// # Safety
    ///
    /// The `DLManagedTensorVersioned` should have a valid `deleter` that can
    /// be called from Rust.
    pub unsafe fn from_raw(tensor: sys::DLManagedTensorVersioned) -> DLPackTensor {
        DLPackTensor {
            raw: tensor,
        }
    }

    /// Get a DLPack tensor reference from this owned tensor
    pub fn as_ref(&self) -> DLPackTensorRef<'_> {
        unsafe {
            // SAFETY: we are constaining the returned reference lifetime
            DLPackTensorRef::from_raw(self.raw.dl_tensor.clone())
        }
    }

    /// Get a mutable DLPack tensor reference from this owned tensor
    pub fn as_mut(&mut self) -> DLPackTensorRefMut<'_> {
        assert!(self.raw.flags & sys::DLPACK_FLAG_BITMASK_IS_COPIED == 0, "Can not create a mutable reference to a borrowed tensor");
        assert!(self.raw.flags & sys::DLPACK_FLAG_BITMASK_READ_ONLY != 0, "Can not create a mutable reference to a read-only tensor");
        // only if NOT read only + unique
        unsafe {
            // SAFETY: we are constaining the returned reference lifetime
            DLPackTensorRefMut::from_raw(self.raw.dl_tensor.clone())
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

    /// Get the number of dimensions of this tensor
    pub fn n_dims(&self) -> usize {
        self.as_ref().n_dims()
    }

    /// Get the shape of this tensor
    pub fn shape(&self) -> &[i64] {
        assert!(!self.raw.dl_tensor.shape.is_null());
        unsafe {
            return std::slice::from_raw_parts(self.raw.dl_tensor.shape, self.n_dims());
        }
    }

    /// Get the strides of this tensor, if any
    pub fn strides(&self) -> Option<&[i64]> {
        if self.raw.dl_tensor.strides.is_null() {
            return None;
        }
        unsafe {
            return Some(std::slice::from_raw_parts(self.raw.dl_tensor.strides, self.n_dims()));
        }
    }

    /// Get the byte offset of this tensor, i.e. how many bytes should be added
    /// to [`DLPackTensor::data_ptr`] and [`DLPackTensor::data_ptr_mut`] to get
    /// the first element of the tensor.
    pub fn byte_offset(&self) -> usize {
        self.as_ref().byte_offset()
    }
}

/// A reference to a DLPack tensor, with data borrowed from some owner,
/// potentially in another language.
pub struct DLPackTensorRef<'a> {
    pub raw: sys::DLTensor,
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
}

/// TODO
pub struct DLPackTensorRefMut<'a> {
    raw: sys::DLTensor,
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
}

#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "pyo3")]
pub mod pyo3;
