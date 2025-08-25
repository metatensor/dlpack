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
