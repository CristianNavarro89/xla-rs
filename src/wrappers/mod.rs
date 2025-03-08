mod literal;
mod pjrt_buffer;
mod pjrt_client;
mod pjrt_device;
mod pjrt_loaded_executable;
mod shape;
mod xla_builder;
mod xla_op;

use crate::c_lib;
use crate::error::{Error, Result};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

pub use literal::Literal;
pub use pjrt_buffer::PjRtBuffer;
pub use pjrt_client::PjRtClient;
pub use pjrt_device::PjRtDevice;
pub use pjrt_loaded_executable::PjRtLoadedExecutable;
pub use shape::{ArrayShape, Shape};
pub use xla_builder::XlaBuilder;
pub use xla_op::XlaOp;

unsafe fn c_ptr_to_string(ptr: *const std::ffi::c_char) -> String {
    let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
    libc::free(ptr as *mut libc::c_void);
    str
}

/// The primitive types supported by XLA. `S8` is a signed 1 byte integer,
/// `U32` is an unsigned 4 bytes integer, etc.
#[derive(Clone, Copy, PartialEq, Eq, Debug, FromPrimitive)]
pub enum PrimitiveType {
    Invalid = 0,
    Pred = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    Bf16 = 16,
    F64 = 12,
    C64 = 15,
    C128 = 18,
    Tuple = 13,
    OpaqueType = 14,
    Token = 17,
}

impl PrimitiveType {
    fn element_type(self) -> Result<ElementType> {
        match self {
            Self::Pred => Ok(ElementType::Pred),
            Self::S8 => Ok(ElementType::S8),
            Self::S16 => Ok(ElementType::S16),
            Self::S32 => Ok(ElementType::S32),
            Self::S64 => Ok(ElementType::S64),
            Self::U8 => Ok(ElementType::U8),
            Self::U16 => Ok(ElementType::U16),
            Self::U32 => Ok(ElementType::U32),
            Self::U64 => Ok(ElementType::U64),
            Self::F16 => Ok(ElementType::F16),
            Self::F32 => Ok(ElementType::F32),
            Self::Bf16 => Ok(ElementType::Bf16),
            Self::F64 => Ok(ElementType::F64),
            Self::C64 => Ok(ElementType::C64),
            Self::C128 => Ok(ElementType::C128),
            Self::Invalid | Self::Tuple | Self::OpaqueType | Self::Token => {
                Err(Error::NotAnElementType { got: self })
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElementType {
    Pred,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    Bf16,
    F64,
    C64,
    C128,
}

impl ElementType {
    /// The size for this element type in bytes.
    pub fn element_size_in_bytes(&self) -> usize {
        match self {
            Self::Pred => 1,
            Self::S8 => 1,
            Self::S16 => 2,
            Self::S32 => 4,
            Self::S64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::F64 => 8,
            Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub fn primitive_type(&self) -> PrimitiveType {
        match self {
            Self::Pred => PrimitiveType::Pred,
            Self::S8 => PrimitiveType::S8,
            Self::S16 => PrimitiveType::S16,
            Self::S32 => PrimitiveType::S32,
            Self::S64 => PrimitiveType::S64,
            Self::U8 => PrimitiveType::U8,
            Self::U16 => PrimitiveType::U16,
            Self::U32 => PrimitiveType::U32,
            Self::U64 => PrimitiveType::U64,
            Self::F16 => PrimitiveType::F16,
            Self::F32 => PrimitiveType::F32,
            Self::Bf16 => PrimitiveType::Bf16,
            Self::F64 => PrimitiveType::F64,
            Self::C64 => PrimitiveType::C64,
            Self::C128 => PrimitiveType::C128,
        }
    }
}

pub trait ArrayElement: Copy {
    const TY: ElementType;
    const ELEMENT_SIZE_IN_BYTES: usize;
    const ZERO: Self;
}

#[allow(clippy::missing_safety_doc)]
/// A type implementing the `NativeType` trait can be directly converted to constant ops or
/// literals.
pub trait NativeType: Copy {
    unsafe fn constant_r0(b: c_lib::xla_builder, v: Self) -> c_lib::xla_op;
    unsafe fn constant_r1(b: c_lib::xla_builder, v: *const Self, l: usize) -> c_lib::xla_op;
    unsafe fn constant_r1c(b: c_lib::xla_builder, v: Self, l: usize) -> c_lib::xla_op;
    unsafe fn create_r0(v: Self) -> c_lib::literal;
    unsafe fn create_r1(v: *const Self, l: usize) -> c_lib::literal;
    unsafe fn literal_get_first_element(l: c_lib::literal) -> Self;
}

macro_rules! native_type {
    ($ty:ty, $cst0:ident, $cst1:ident, $cst1c:ident, $cre0:ident, $cre1:ident, $gf:ident) => {
        impl NativeType for $ty {
            unsafe fn constant_r0(b: c_lib::xla_builder, v: Self) -> c_lib::xla_op {
                c_lib::$cst0(b, v)
            }
            unsafe fn constant_r1(
                b: c_lib::xla_builder,
                v: *const Self,
                l: usize,
            ) -> c_lib::xla_op {
                c_lib::$cst1(b, v, l)
            }
            unsafe fn constant_r1c(b: c_lib::xla_builder, v: Self, l: usize) -> c_lib::xla_op {
                c_lib::$cst1c(b, v, l)
            }
            unsafe fn create_r0(v: Self) -> c_lib::literal {
                c_lib::$cre0(v)
            }
            unsafe fn create_r1(v: *const Self, l: usize) -> c_lib::literal {
                c_lib::$cre1(v, l)
            }
            unsafe fn literal_get_first_element(l: c_lib::literal) -> Self {
                c_lib::$gf(l)
            }
        }
    };
}