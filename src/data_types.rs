use super::sys::{DLDataType, DLDataTypeCode};

#[derive(Debug)]
pub enum CastError {
    WrongType { dl_type: DLDataType, rust_type: &'static str },
    Lanes { expected: usize, given: usize },
    BadAlignment { ptr: usize, align: usize, rust_type: &'static str },
}

impl std::fmt::Display for CastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastError::WrongType { dl_type, rust_type } => {
                write!(f, "can not cast from {} to {}", dl_type, rust_type)
            }
            CastError::Lanes { expected, given } => {
                write!(f, "invalid number of lanes: expected {}, but got {}", expected, given)
            }
            CastError::BadAlignment { ptr, align, rust_type } => {
                write!(f, "invalid pointer alignment: pointer at {:x} should be aligned to {} for {}", ptr, align, rust_type)
            }
        }
    }
}

impl std::error::Error for CastError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// Trait to cast a DLPack pointer to a Rust pointer
pub trait DLPackPointerCast: Sized {
    /// Cast `ptr` (declared to have the `DLDataType` datatype) to a rust pointer.
    fn dlpack_ptr_cast(ptr: *mut std::os::raw::c_void, data_type: DLDataType) -> Result<*mut Self, CastError>;
}

macro_rules! impl_dlpack_pointer_cast {
    ($dlpack_code: expr, $($type: ty),+, ) => {
        $(impl DLPackPointerCast for $type {
            fn dlpack_ptr_cast(ptr: *mut std::os::raw::c_void, data_type: DLDataType) -> Result<*mut Self, CastError> {
                if (data_type.bits as usize) != 8 * ::std::mem::size_of::<$type>() || data_type.code != $dlpack_code {
                    return Err(CastError::WrongType { dl_type: data_type, rust_type: stringify!($type)});
                }

                if data_type.lanes != 1 {
                    return Err(CastError::Lanes { expected: 1, given: data_type.lanes as usize });
                }

                let ptr = ptr.cast::<Self>();
                if !ptr.is_aligned() {
                    return Err(CastError::BadAlignment {
                        ptr: ptr as usize,
                        align: ::std::mem::align_of::<$type>(),
                        rust_type: stringify!($type),
                    });
                }

                return Ok(ptr);
            }
        })+
    };
}

impl_dlpack_pointer_cast!(DLDataTypeCode::kDLUInt, u8, u16, u32, u64,);
impl_dlpack_pointer_cast!(DLDataTypeCode::kDLInt, i8, i16, i32, i64,);
impl_dlpack_pointer_cast!(DLDataTypeCode::kDLFloat, f32, f64,);
impl_dlpack_pointer_cast!(DLDataTypeCode::kDLBool, bool,);


/// Trait to get the DLPack datatype correspondong to a Rust datatype
#[allow(dead_code)]
pub trait GetDLPackDataType {
    fn get_dlpack_data_type() -> DLDataType;
}

macro_rules! impl_get_dlpack_data_type {
    ($dlpack_code: expr, $($type: ty),+, ) => {
        $(impl GetDLPackDataType for $type {
            fn get_dlpack_data_type() -> DLDataType {
                DLDataType {
                    code: $dlpack_code,
                    bits: (8 * std::mem::size_of::<$type>()).try_into().expect("failed to convert type size to u8"),
                    lanes: 1,
                }
            }
        })+
    };
}

impl_get_dlpack_data_type!(DLDataTypeCode::kDLUInt, u8, u16, u32, u64,);
impl_get_dlpack_data_type!(DLDataTypeCode::kDLInt, i8, i16, i32, i64,);
impl_get_dlpack_data_type!(DLDataTypeCode::kDLFloat, f32, f64,);
impl_get_dlpack_data_type!(DLDataTypeCode::kDLBool, bool,);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::{DLDataType, DLDataTypeCode};

    #[test]
    fn test_get_dlpack_data_type() {
        let dtype = u32::get_dlpack_data_type();
        assert_eq!(dtype.code, DLDataTypeCode::kDLUInt);
        assert_eq!(dtype.bits, 32);

        let dtype = f32::get_dlpack_data_type();
        assert_eq!(dtype.code, DLDataTypeCode::kDLFloat);
        assert_eq!(dtype.bits, 32);
    }

    #[test]
    fn test_dlpack_pointer_cast() {
        let value: u32 = 42;
        let mut mock_data = value;
        let ptr = &mut mock_data as *mut u32 as *mut std::os::raw::c_void;

        let dtype = DLDataType {
            code: DLDataTypeCode::kDLUInt,
            bits: 32,
            lanes: 1,
        };
        let result = u32::dlpack_ptr_cast(ptr, dtype);
        assert!(result.is_ok());
        let cast_ptr = result.unwrap();
        unsafe {
            assert_eq!(*cast_ptr, 42);
        }

        let wrong_type = DLDataType {
            code: DLDataTypeCode::kDLFloat,
            bits: 32,
            lanes: 1,
        };
        let result = u32::dlpack_ptr_cast(ptr, wrong_type);
        assert!(result.is_err());
        if let Err(CastError::WrongType { dl_type, .. }) = result {
            assert_eq!(dl_type.code, DLDataTypeCode::kDLFloat);
        } else {
            panic!("Expected a WrongType error");
        }
        
        let wrong_lanes = DLDataType {
            code: DLDataTypeCode::kDLUInt,
            bits: 32,
            lanes: 2,
        };
        let result = u32::dlpack_ptr_cast(ptr, wrong_lanes);
        assert!(result.is_err());
        if let Err(CastError::Lanes { given, .. }) = result {
            assert_eq!(given, 2);
        } else {
            panic!("Expected a Lanes error");
        }

        let mut correctly_aligned = [0_u8; 8];
        let base_ptr = correctly_aligned.as_mut_ptr().cast::<std::os::raw::c_void>();
        let unaligned_ptr = unsafe { base_ptr.add(1) };
        let result = u64::dlpack_ptr_cast(unaligned_ptr, DLDataType {
            code: DLDataTypeCode::kDLUInt,
            bits: 64,
            lanes: 1,
        });
        assert!(result.is_err());
        if let Err(CastError::BadAlignment { .. }) = result {
        } else {
            panic!("Expected a BadAlignment error");
        }
    }
}
