#![allow(non_camel_case_types)]
#![allow(conflicting_repr_hints)]
//! This module contains the low-level API for dlpack. It was manually
//! translated from `dlpack.h` header at version 1.3; and contains types
//! suitable for use in C FFI.

/// The current major version of dlpack
pub const DLPACK_MAJOR_VERSION: u32 = 1;
/// The current minor version of dlpack
pub const DLPACK_MINOR_VERSION: u32 = 3;

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
#[repr(u32)]
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
    /// Reserved extension device type,
    /// used for quickly test extension device.
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
    /// AWS Trainium
    kDLTrn = 18,
}

/// A Device for Tensor and operator.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    /// The device type used in the device.
    pub device_type: DLDeviceType,
    /// The device index.
    /// For vanilla CPU memory, pinned memory, or managed memory, this is set to
    /// 0.
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
            DLDeviceType::kDLTrn => write!(f, "Trn"),
        }
    }
}

impl std::fmt::Display for DLDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.device_type, self.device_id)
    }
}


/// The type code options DLDataType.
#[repr(u8)]
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
///   - float8_e4m3: type_code = 8, bits = 8, lanes = 1 (packed in memory)
///   - float6_e3m2fn: type_code = 16, bits = 6, lanes = 1 (packed in memory)
///   - float4_e2m1fn: type_code = 17, bits = 4, lanes = 1 (packed in memory)
///
/// When a sub-byte type is packed, DLPack requires the data to be in little bit-endian, i.e.,
/// for a packed data set D ((D >> (i * bits)) && bit_mask) stores the i-th element.
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
            DLDataTypeCode::kDLFloat6_e2m3fn => "f6_e2m3fn",
            DLDataTypeCode::kDLFloat6_e3m2fn => "f6_e3m2fn",
            DLDataTypeCode::kDLFloat4_e2m1fn => "f4_e2m1fn",
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
    /// Strides of the tensor (in number of elements, not bytes).
    ///
    ///  can not be NULL if ndim != 0, must points to
    ///  an array of ndim elements that specifies the strides,
    ///  so consumer can always rely on strides[dim] being valid for 0 <= dim < ndim.
    ///
    ///  When ndim == 0, strides can be set to NULL.
    ///
    /// NOTE: Before DLPack v1.2, strides can be NULL to indicate contiguous data.
    ///       This is not allowed in DLPack v1.2 and later. The rationale
    ///       is to simplify the consumer handling.
    ///
    /// When ndim == 0, strides may represent NULL.
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
    /// Destructor - this should be called to destruct the manager_ctx
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
    ///
    /// See also:
    ///           DLPACK_FLAG_BITMASK_READ_ONLY
    ///           DLPACK_FLAG_BITMASK_IS_COPIED
    pub flags: u64,
    /// DLTensor which is being memory managed
    pub dl_tensor: DLTensor,
}

///----------------------------------------------------------------------
/// DLPack `__dlpack_c_exchange_api__` fast exchange protocol definitions
///----------------------------------------------------------------------

/// Request a producer library to create a new tensor.
///
/// Create a new `DLManagedTensorVersioned` within the context of the producer
/// library. The allocation is defined via the prototype DLTensor.
///
/// This function is exposed by the framework through the DLPackExchangeAPI.
///
/// # Arguments
///
/// * `prototype` - The prototype DLTensor. Only the dtype, ndim, shape,
///                 and device fields are used.
/// * `out`       - The output DLManagedTensorVersioned.
/// * `error_ctx` - Context for `SetError`.
/// * `SetError`  - The function to set the error.
///
/// # Returns
/// 
/// The owning DLManagedTensorVersioned* or NULL on failure.
/// SetError is called exactly when NULL is returned (the implementer
///         must ensure this).
///         
/// NOTE: - As a C function, must not thrown C++ exceptions.
///       - Error propagation via SetError to avoid any direct need
///         of Python API. Due to this `SetError` may have to ensure the GIL is
///         held since it will presumably set a Python error.
///
/// See also:
///          DLPackExchangeAPI
pub type DLPackManagedTensorAllocator = Option<unsafe extern "C" fn(
    prototype: *mut DLTensor,
    out: *mut *mut DLManagedTensorVersioned,
    error_ctx: *mut std::os::raw::c_void,
    set_error: Option<unsafe extern "C" fn(error_ctx: *mut std::os::raw::c_void,
                                           kind: *const std::os::raw::c_char,
                                           message: *const std::os::raw::c_char)>
) -> i32>;

/// Exports a PyObject* Tensor/NDArray to a DLManagedTensorVersioned.
///
/// This function does not perform any stream synchronization. The consumer should query
/// DLPackCurrentWorkStream to get the current work stream and launch kernels on it.
///
/// This function is exposed by the framework through the DLPackExchangeAPI.
///
/// # Arguments
/// 
/// * `py_object` - The Python object to convert. Must have the same type
///                 as the one the `DLPackExchangeAPI` was discovered from.
/// * `out` - The output DLManagedTensorVersioned.
/// 
/// # Returns
/// 
/// The owning DLManagedTensorVersioned* or NULL on failure with a
/// Python exception set. If the data cannot be described using DLPack
/// this should be a BufferError if possible.
/// 
/// NOTE: - As a C function, must not thrown C++ exceptions.
///
/// See also:
///          DLPackExchangeAPI, DLPackCurrentWorkStream
pub type DLPackManagedTensorFromPyObjectNoSync = Option<unsafe extern "C" fn(
    py_object: *mut std::os::raw::c_void,
    out: *mut *mut DLManagedTensorVersioned
) -> i32>;

/// Exports a PyObject* Tensor/NDArray to a provided DLTensor.
///
/// This function provides a faster interface for temporary, non-owning, exchange.
/// The producer (implementer) still owns the memory of data, strides, shape.
/// The liveness of the DLTensor and the data it views is only guaranteed until
/// control is returned.
///
/// This function currently assumes that the producer (implementer) can fill
/// in the DLTensor shape and strides without the need for temporary allocations.
///
/// This function does not perform any stream synchronization. The consumer should query
/// DLPackCurrentWorkStream to get the current work stream and launch kernels on it.
///
/// This function is exposed by the framework through the DLPackExchangeAPI.
///
/// # Arguments
/// 
///  * `py_object` - The Python object to convert. Must have the same type
///                  as the one the `DLPackExchangeAPI` was discovered from.
///  * `out` - The output DLTensor, whose space is pre-allocated on stack.
///
/// # Returns
/// 
/// 0 on success, -1 on failure with a Python exception set.
///
/// NOTE: - As a C function, must not thrown C++ exceptions.
///
/// See also:
///          DLPackExchangeAPI, DLPackCurrentWorkStream
pub type DLPackDLTensorFromPyObjectNoSync = Option<unsafe extern "C" fn(
    py_object: *mut std::os::raw::c_void,
    out: *mut DLTensor
) -> i32>;

/// \brief Obtain the current work stream of a device.
///
/// Obtain the current work stream of a device from the producer framework.
/// For example, it should map to torch.cuda.current_stream in PyTorch.
///
/// When device_type is kDLCPU, the consumer do not have to query the stream
/// and the producer can simply return NULL when queried.
/// The consumer do not have to do anything on stream sync or setting.
/// So CPU only framework can just provide a dummy implementation that
/// always set out_current_stream[0] to NULL.
///
/// # Arguments
/// 
/// * `device_type` - The device type.
/// * `device_id` - The device id.
/// * `out_current_stream` - The output current work stream.
///
/// # Returns
/// 
/// 0 on success, -1 on failure with a Python exception set.
/// 
/// NOTE: - As a C function, must not thrown C++ exceptions.
///
/// See also:
///          DLPackExchangeAPI
pub type DLPackCurrentWorkStream = Option<unsafe extern "C" fn(
    device_type: DLDeviceType,
    device_id: i32,
    out_current_stream: *mut *mut std::os::raw::c_void
) -> i32>;

/// Imports a DLManagedTensorVersioned to a PyObject* Tensor/NDArray.
///
/// Convert an owning DLManagedTensorVersioned* to the Python tensor of the
/// producer (implementer) library with the correct type.
///
/// This function does not perform any stream synchronization.
///
/// This function is exposed by the framework through the DLPackExchangeAPI.
///
/// # Arguments
/// 
/// * `tensor` - The DLManagedTensorVersioned to convert the ownership of the
///              tensor is stolen.
/// * `out_py_object` - The output Python object.
/// 
/// # Returns
/// 
/// 0 on success, -1 on failure with a Python exception set.
/// 
/// See also:
///          DLPackExchangeAPI
pub type DLPackManagedTensorToPyObjectNoSync = Option<unsafe extern "C" fn(
    tensor: *mut DLManagedTensorVersioned,
    out_py_object: *mut *mut std::os::raw::c_void
) -> i32>;

/// DLPackExchangeAPI stable header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLPackExchangeAPIHeader {
    /// The provided DLPack version the consumer must check major version
    /// compatibility before using this struct.
    pub version: DLPackVersion,
    /// Optional pointer to an older DLPackExchangeAPI in the chain.
    ///
    /// It must be NULL if the framework does not support older versions.
    /// If the current major version is larger than the one supported by the
    /// consumer, the consumer may walk this to find an earlier supported version.
    ///
    /// See also:
    ///          DLPackExchangeAPI
    pub prev_api: *mut DLPackExchangeAPIHeader,
}

/// Framework-specific function pointers table for DLPack exchange.
///
/// Additionally to `__dlpack__()` we define a C function table sharable by
///
/// Python implementations via `__dlpack_c_exchange_api__`.
/// This attribute must be set on the type as a Python PyCapsule
/// with name "dlpack_exchange_api".
///
/// A consumer library may use a pattern such as:
///
///
/// ```c
///  PyObject* api_capsule = PyObject_GetAttrString(
///    (PyObject*)Py_TYPE(tensor_obj), "__dlpack_c_exchange_api__")
///  );
///  if (api_capsule == NULL) { goto handle_error; }
///  MyDLPackExchangeAPI* api = (MyDLPackExchangeAPI*)PyCapsule_GetPointer(
///    api_capsule, "dlpack_exchange_api"
///  );
///  Py_DECREF(api_capsule);
///  if (api == NULL) { goto handle_error; }
/// ```
///
///
/// Note that this must be defined on the type. The consumer should look up the
/// attribute on the type and may cache the result for each unique type.
///
/// The precise API table is given by:
/// ```c
/// struct MyDLPackExchangeAPI : public DLPackExchangeAPI {
///   MyDLPackExchangeAPI() {
///     header.version.major = DLPACK_MAJOR_VERSION;
///     header.version.minor = DLPACK_MINOR_VERSION;
///     header.prev_version_api = nullptr;
///
///     managed_tensor_allocator = MyDLPackManagedTensorAllocator;
///     managed_tensor_from_py_object_no_sync = MyDLPackManagedTensorFromPyObjectNoSync;
///     managed_tensor_to_py_object_no_sync = MyDLPackManagedTensorToPyObjectNoSync;
///     dltensor_from_py_object_no_sync = MyDLPackDLTensorFromPyObjectNoSync;
///     current_work_stream = MyDLPackCurrentWorkStream;
///  }
///
///  static const DLPackExchangeAPI* Global() {
///     static MyDLPackExchangeAPI inst;
///     return &inst;
///  }
/// };
/// ```
///
/// Guidelines for leveraging DLPackExchangeAPI:
///
/// There are generally two kinds of consumer needs for DLPack exchange:
/// - N0: library support, where consumer.kernel(x, y, z) would like to run a kernel
///       with the data from x, y, z. The consumer is also expected to run the kernel with the same
///       stream context as the producer. For example, when x, y, z is torch.Tensor,
///       consumer should query exchange_api->current_work_stream to get the
///       current stream and launch the kernel with the same stream.
///       This setup is necessary for no synchronization in kernel launch and maximum compatibility
///       with CUDA graph capture in the producer.
///       This is the desirable behavior for library extension support for frameworks like PyTorch.
/// - N1: data ingestion and retention
///
/// Note that obj.__dlpack__() API should provide useful ways for N1.
/// The primary focus of the current DLPackExchangeAPI is to enable faster exchange N0
/// with the support of the function pointer current_work_stream.
///
/// Array/Tensor libraries should statically create and initialize this structure
/// then return a pointer to DLPackExchangeAPI as an int value in Tensor/Array.
/// The DLPackExchangeAPI* must stay alive throughout the lifetime of the process.
///
/// One simple way to do so is to create a static instance of DLPackExchangeAPI
/// within the framework and return a pointer to it. The following code
/// shows an example to do so in C++. It should also be reasonably easy
/// to do so in other languages.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLPackExchangeAPI {
    /// The header that remains stable across versions.
    pub header: DLPackExchangeAPIHeader,
    /// Producer function pointer for DLPackManagedTensorAllocator
    /// This function must not be NULL.
    ///
    /// See also:
    ///          DLPackManagedTensorAllocator
    pub managed_tensor_allocator: DLPackManagedTensorAllocator,
    /// Producer function pointer for DLPackManagedTensorFromPyObject
    /// This function must not be NULL.
    ///
    /// See also:
    ///          DLPackManagedTensorFromPyObject
    pub managed_tensor_from_py_object_no_sync: DLPackManagedTensorFromPyObjectNoSync,
    /// Producer function pointer for DLPackManagedTensorToPyObjectNoSync
    /// This function must not be NULL.
    ///
    /// See also:
    ///          DLPackManagedTensorToPyObjectNoSync
    pub managed_tensor_to_py_object_no_sync: DLPackManagedTensorToPyObjectNoSync,
    /// Producer function pointer for DLPackDLTensorFromPyObject
    /// This function must not be NULL.
    ///
    /// See also:
    ///          DLPackDLTensorFromPyObject
    pub dltensor_from_py_object_no_sync: DLPackDLTensorFromPyObjectNoSync,
    /// Producer function pointer for DLPackCurrentWorkStream
    /// This function must not be NULL.
    ///
    /// See also:
    ///          DLPackCurrentWorkStream
    pub current_work_stream: DLPackCurrentWorkStream,
}
