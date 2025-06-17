#![allow(clippy::needless_return, clippy::redundant_field_names)]

pub mod sys;

mod data_types;
pub use self::data_types::CastError;
use self::data_types::DLPackPointerCast;

/// TODO
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
    pub unsafe fn from_raw(tensor: sys::DLManagedTensorVersioned) -> DLPackTensor {
        DLPackTensor {
            raw: tensor,
        }
    }

    fn as_ref(&self) -> DLPackTensorRef {
        unsafe {
            // SAFETY: we are constaining the returned reference lifetime
            DLPackTensorRef::from_raw(self.raw.dl_tensor.clone())
        }
    }

    fn as_mut(&mut self) -> DLPackTensorRefMut {
        assert!(self.raw.flags & sys::DLPACK_FLAG_BITMASK_IS_COPIED == 0, "Can not create a mutable reference to a borrowed tensor");
        assert!(self.raw.flags & sys::DLPACK_FLAG_BITMASK_READ_ONLY != 0, "Can not create a mutable reference to a read-only tensor");
        // only if NOT read only + unique
        unsafe {
            // SAFETY: we are constaining the returned reference lifetime
            DLPackTensorRefMut::from_raw(self.raw.dl_tensor.clone())
        }
    }
}

/// TODO
pub struct DLPackTensorRef<'a> {
    pub raw: sys::DLTensor,
    phantom: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> DLPackTensorRef<'a> {
    pub unsafe fn from_raw(tensor: sys::DLTensor) -> DLPackTensorRef<'a> {
        DLPackTensorRef {
            raw: tensor,
            phantom: std::marker::PhantomData
        }
    }

    pub fn as_ptr<T>(&self) -> Result<*const T, CastError>  where T: DLPackPointerCast {
        let ptr = self.raw.data.wrapping_add(self.byte_offset());
        return T::dlpack_ptr_cast(ptr, self.raw.dtype).map(|p| p.cast_const());
    }

    pub fn device(&self) -> sys::DLDevice {
        return self.raw.device;
    }

    pub fn n_dims(&self) -> usize {
        return self.raw.ndim as usize;
    }

    pub fn shape(&self) -> &[i64] {
        assert!(!self.raw.shape.is_null());
        unsafe {
            return std::slice::from_raw_parts(self.raw.shape, self.n_dims());
        }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        if self.raw.strides.is_null() {
            return None;
        }
        unsafe {
            return Some(std::slice::from_raw_parts(self.raw.strides, self.n_dims()));
        }
    }

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
    pub unsafe fn from_raw(tensor: sys::DLTensor) -> DLPackTensorRefMut<'a> {
        DLPackTensorRefMut {
            raw: tensor,
            phantom: std::marker::PhantomData
        }
    }

    /// TODO
    pub fn as_ref(&self) -> DLPackTensorRef {
        todo!()
    }

    pub fn as_ptr<T>(&self) -> Result<*const T, CastError>  where T: DLPackPointerCast {
        let ptr = self.raw.data.wrapping_add(self.byte_offset());
        return T::dlpack_ptr_cast(ptr, self.raw.dtype).map(|p| p.cast_const());
    }

    pub fn as_ptr_mut<T>(&mut self) -> Result<*mut T, CastError>  where T: DLPackPointerCast {
        let ptr = self.raw.data.wrapping_add(self.byte_offset());
        return T::dlpack_ptr_cast(ptr, self.raw.dtype);
    }

    pub fn device(&self) -> sys::DLDevice {
        return self.raw.device;
    }

    pub fn n_dims(&self) -> usize {
        return self.raw.ndim as usize;
    }

    pub fn shape(&self) -> &[i64] {
        assert!(!self.raw.shape.is_null());
        unsafe {
            return std::slice::from_raw_parts(self.raw.shape, self.n_dims());
        }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        if self.raw.strides.is_null() {
            return None;
        }
        unsafe {
            return Some(std::slice::from_raw_parts(self.raw.strides, self.n_dims()));
        }
    }

    pub fn byte_offset(&self) -> usize {
        return self.raw.byte_offset as usize;
    }
}

#[cfg(feature = "ndarray")]
pub mod ndarray;
