#![allow(non_camel_case_types)]
#![allow(conflicting_repr_hints)]
//! This module contains the low-level API for dlpack. It was manually
//! translated from `dlpack.h` header at version 1.0; and contains types
//! suitable for use in C FFI.

/// The current major version of dlpack
pub const DLPACK_MAJOR_VERSION: u32 = 1;
/// The current minor version of dlpack
pub const DLPACK_MINOR_VERSION: u32 = 1;

/// bit mask to indicate that the tensor is read only.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;

/// bit mask to indicate that the tensor is a copy made by the producer.
///
/// If set, the tensor is considered solely owned throughout its lifetime by the
/// consumer, until the producer-provided deleter is invoked.
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;

/// bit mask to indicate that whether a sub-byte type is packed or padded.
///
/// The default for sub-byte types (ex: fp4/fp6) is assumed packed. This flag can
/// be set by the producer to signal that a tensor of sub-byte type is padded.
pub const DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED: u64 = 1 << 2;

/// The DLPack version.
///
/// A change in major version indicates that we have changed the data layout of
/// the ABI - DLManagedTensorVersioned.
///
/// A change in minor version indicates that we have added new code, such as a
/// new device type, but the ABI is kept the same.
///
/// If an obtained DLPack tensor has a major version that disagrees with the
/// version number specified in this header file (i.e. major !=
/// DLPACK_MAJOR_VERSION), the consumer must call the deleter (and it is safe to
/// do so). It is not safe to access any other fields as the memory layout will
/// have changed.
///
/// In the case of a minor version mismatch, the tensor can be safely used as
/// long as the consumer knows how to interpret all fields. Minor version
/// updates indicate the addition of enumeration values.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLPackVersion {
    /// DLPack major version.
    pub major: u32,
    /// DLPack minor version.
    pub minor: u32
}

/// The device type in DLDevice.
#[repr(C, u32)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDeviceType {
    /// CPU device
    kDLCPU = 1,
    /// CUDA GPU device
    kDLCUDA = 2,
    /// Pinned CUDA CPU memory by cudaMallocHost
    kDLCUDAHost = 3,
    /// OpenCL devices.
    kDLOpenCL = 4,
    /// Vulkan buffer for next generation graphics.
    kDLVulkan = 7,
    /// Metal for Apple GPU.
    kDLMetal = 8,
    /// Verilog simulator buffer
    kDLVPI = 9,
    /// ROCm GPUs for AMD GPUs
    kDLROCM = 10,
    /// Pinned ROCm CPU memory allocated by hipMallocHost
    kDLROCMHost = 11,
    /// Reserved extension device type, used for quickly test extension device.
    /// The semantics can differ depending on the implementation.
    kDLExtDev = 12,
    /// CUDA managed/unified memory allocated by cudaMallocManaged
    kDLCUDAManaged = 13,
    /// Unified shared memory allocated on a oneAPI non-partitioned
    /// device. Call to oneAPI runtime is required to determine the device
    /// type, the USM allocation type and the sycl context it is bound to.
    kDLOneAPI = 14,
    /// GPU support for next generation WebGPU standard.
    kDLWebGPU = 15,
    /// Qualcomm Hexagon DSP
    kDLHexagon = 16,
    /// Microsoft MAIA devices
    kDLMAIA = 17,
}

/// A Device for Tensor and operator.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    /// The device type used in the device.
    pub device_type: DLDeviceType,
    /// The device index. For vanilla CPU memory, pinned memory, or managed
    /// memory, this is set to 0.
    pub device_id: i32,
}

impl std::fmt::Display for DLDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DLDeviceType::kDLCPU => write!(f, "CPU"),
            DLDeviceType::kDLCUDA => write!(f, "CUDA"),
            DLDeviceType::kDLCUDAHost => write!(f, "CUDAHost"),
            DLDeviceType::kDLOpenCL => write!(f, "OpenCL"),
            DLDeviceType::kDLVulkan => write!(f, "Vulkan"),
            DLDeviceType::kDLMetal => write!(f, "Metal"),
            DLDeviceType::kDLVPI => write!(f, "VPI"),
            DLDeviceType::kDLROCM => write!(f, "ROCM"),
            DLDeviceType::kDLROCMHost => write!(f, "ROCMHost"),
            DLDeviceType::kDLExtDev => write!(f, "ExtDev"),
            DLDeviceType::kDLCUDAManaged => write!(f, "CUDAManaged"),
            DLDeviceType::kDLOneAPI => write!(f, "OneAPI"),
            DLDeviceType::kDLWebGPU => write!(f, "WebGPU"),
            DLDeviceType::kDLHexagon => write!(f, "Hexagon"),
            DLDeviceType::kDLMAIA => write!(f, "MAIA"),
        }
    }
}

impl std::fmt::Display for DLDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.device_type, self.device_id)
    }
}


/// The type code options DLDataType.
#[repr(C, u8)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DLDataTypeCode {
    /// signed integer
    kDLInt = 0,
    /// unsigned integer
    kDLUInt = 1,
    /// IEEE floating point
    kDLFloat = 2,
    /// Opaque handle type, reserved for testing purposes. Frameworks need to
    /// agree on the handle data type for the exchange to be well-defined.
    kDLOpaqueHandle = 3,
    /// bfloat16
    kDLBfloat = 4,
    /// complex number (C/C++/Python layout: compact struct per complex number)
    kDLComplex = 5,
    /// boolean
    kDLBool = 6,
    /// FP8 data types
    kDLFloat8_e3m4 = 7,
    kDLFloat8_e4m3 = 8,
    kDLFloat8_e4m3b11fnuz = 9,
    kDLFloat8_e4m3fn = 10,
    kDLFloat8_e4m3fnuz = 11,
    kDLFloat8_e5m2 = 12,
    kDLFloat8_e5m2fnuz = 13,
    kDLFloat8_e8m0fnu = 14,
    /// FP6 data types
    ///
    /// Setting bits != 6 is currently unspecified, and the producer must ensure it is set
    /// while the consumer must stop importing if the value is unexpected.
    ///
    kDLFloat6_e2m3fn = 15,
    kDLFloat6_e3m2fn = 16,
    /// FP4 data types
    ///
    /// Setting bits != 4 is currently unspecified, and the producer must ensure it is set
    /// while the consumer must stop importing if the value is unexpected.
    ///
    kDLFloat4_e2m1fn = 17,
}

/// The data type the tensor can hold. The data type is assumed to follow the
/// native endian-ness. An explicit error message should be raised when
/// attempting to export an array with non-native endianness
///
///  Examples
///   - `float`: type_code = 2, bits = 32, lanes = 1
///   - `float4(vectorized 4 float)`: type_code = 2, bits = 32, lanes = 4
///   - `int8`: type_code = 0, bits = 8, lanes = 1
///   - `std::complex<float>`: type_code = 5, bits = 64, lanes = 1
///   - `bool`: type_code = 6, bits = 8, lanes = 1 (as per common array library
///     convention, the underlying storage size of bool is 8 bits)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    /// Type code of base types.
    /// We keep it uint8_t instead of DLDataTypeCode for minimal memory
    /// footprint, but the value should be one of DLDataTypeCode enum values.
    pub code: DLDataTypeCode,
    /// Number of bits, common choices are 8, 16, 32.
    pub bits: u8,
    /// Number of lanes in the type, used for vector types.
    pub lanes: u16,
}

impl std::fmt::Display for DLDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.code == DLDataTypeCode::kDLBool && self.bits == 8 && self.lanes == 1 {
            return write!(f, "bool");
        }

        let type_ = match self.code {
            DLDataTypeCode::kDLInt => "i",
            DLDataTypeCode::kDLUInt => "u",
            DLDataTypeCode::kDLFloat => "f",
            DLDataTypeCode::kDLOpaqueHandle => "opaque",
            DLDataTypeCode::kDLBfloat => "bfloat",
            DLDataTypeCode::kDLComplex => "complex",
            DLDataTypeCode::kDLBool => "b",
            DLDataTypeCode::kDLFloat8_e3m4 => "f8_e3m4",
            DLDataTypeCode::kDLFloat8_e4m3 => "f8_e4m3",
            DLDataTypeCode::kDLFloat8_e4m3b11fnuz => "f8_e4m3b11fnuz",
            DLDataTypeCode::kDLFloat8_e4m3fn => "f8_e4m3fn",
            DLDataTypeCode::kDLFloat8_e4m3fnuz => "f8_e4m3fnuz",
            DLDataTypeCode::kDLFloat8_e5m2 => "f8_e5m2",
            DLDataTypeCode::kDLFloat8_e5m2fnuz => "f8_e5m2fnuz",
            DLDataTypeCode::kDLFloat8_e8m0fnu => "f8_e8m0fnu",
            DLDataTypeCode::kDLFloat6_e2m3fn => "f8_e2m3fn",
            DLDataTypeCode::kDLFloat6_e3m2fn => "f8_e3m2fn",
            DLDataTypeCode::kDLFloat4_e2m1fn => "f8_e2m1fn",
        };

        if self.lanes == 1 {
            write!(f, "{}{}", type_, self.bits)
        } else {
            write!(f, "{}{}x{}", type_, self.bits, self.lanes)
        }
    }
}

/// Plain C Tensor object, does not manage memory.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DLTensor {
    /// The data pointer points to the allocated data. This will be CUDA device
    /// pointer or cl_mem handle in OpenCL. It may be opaque on some device
    /// types. This pointer is always aligned to 256 bytes as in CUDA. The
    /// `byte_offset` field should be used to point to the beginning of the
    /// data.
    ///
    /// Note that as of Nov 2021, multiply libraries (CuPy, PyTorch, TensorFlow,
    /// TVM, perhaps others) do not adhere to this 256 byte alignment
    /// requirement on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must
    /// be fixed (after which this note will be updated); at the moment it is
    /// recommended to not rely on the data pointer being correctly aligned.
    ///
    /// For given DLTensor, the size of memory required to store the contents of
    /// data is calculated as follows:
    ///
    /// ```c
    /// static inline size_t GetDataSize(const DLTensor* t) {
    ///   size_t size = 1;
    ///   for (tvm_index_t i = 0; i < t->ndim; ++i) {
    ///     size *= t->shape[i];
    ///   }
    ///   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
    ///   return size;
    /// }
    /// ```
    ///
    /// Note that if the tensor is of size zero, then the data pointer should be
    /// set to `NULL`.
    pub data: *mut std::os::raw::c_void,
    /// The device of the tensor
    pub device: DLDevice,
    /// Number of dimensions
    pub ndim: i32,
    /// The data type of the pointer
    pub dtype: DLDataType,
    /// The shape of the tensor
    pub shape: *mut i64,
    /// strides of the tensor (in number of elements, not bytes)
    /// can be NULL, indicating tensor is compact and row-majored.
    pub strides: *mut i64,
    /// The offset in bytes to the beginning pointer to data
    pub byte_offset: u64,
}

/// C Tensor object, manage memory of DLTensor. This data structure is
/// intended to facilitate the borrowing of DLTensor by another framework. It is
/// not meant to transfer the tensor. When the borrowing framework doesn't need
/// the tensor, it should call the deleter to notify the host that the resource
/// is no longer needed.
///
/// NOTE: This data structure is used as Legacy DLManagedTensor
///      in DLPack exchange and is deprecated after DLPack v0.8
///      Use DLManagedTensorVersioned instead.
///      This data structure may get renamed or deleted in future versions.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensor {
    /// DLTensor which is being memory managed
    pub dl_tensor: DLTensor,
    /// the context of the original host framework of DLManagedTensor in which
    /// DLManagedTensor is used in the framework. It can also be NULL.
    pub manager_ctx: *mut std::os::raw::c_void,
    /// \brief Destructor - this should be called to destruct the manager_ctx
    /// which backs the DLManagedTensor. It can be NULL if there is no way for
    /// the caller to provide a reasonable destructor. The destructor deletes
    /// the argument self as well.
    pub deleter: Option<unsafe extern "C" fn(_self: *mut DLManagedTensor) -> ()>,
}


/// A versioned and managed C Tensor object, manage memory of DLTensor.
///
/// This data structure is intended to facilitate the borrowing of DLTensor by
/// another framework. It is not meant to transfer the tensor. When the borrowing
/// framework doesn't need the tensor, it should call the deleter to notify the
/// host that the resource is no longer needed.
///
/// NOTE:  This is the current standard DLPack exchange data structure.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensorVersioned {
    /// The API and ABI version of the current managed Tensor
    pub version: DLPackVersion,
    /// the context of the original host framework.
    ///
    /// Stores DLManagedTensorVersioned is used in the
    /// framework. It can also be NULL.
    pub manager_ctx: *mut std::os::raw::c_void,
    /// Destructor.
    ///
    /// This should be called to destruct manager_ctx which holds the
    /// DLManagedTensorVersioned. It can be NULL if there is no way for the
    /// caller to provide a reasonable destructor. The destructor deletes the
    /// argument self as well.
    pub deleter: Option<unsafe extern "C" fn(_self: *mut DLManagedTensorVersioned) -> ()>,

    /// Additional bitmask flags information about the tensor.
    ///
    /// By default the flags should be set to 0.
    ///
    /// NOTE: Future ABI changes should keep everything until this field
    ///       stable, to ensure that deleter can be correctly called.
    pub flags: u64,
    /// DLTensor which is being memory managed
    pub dl_tensor: DLTensor,
}
