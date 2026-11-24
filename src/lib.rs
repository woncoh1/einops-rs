//! Readable tensor operations inspired by Alex Rogozhnikov's
//! [Einops](https://einops.rocks/).
//!
//! The default `burn` feature supports Burn 0.22 tensors. Enable an execution
//! backend in your application's Burn dependency. Patterns and dimension sizes
//! are checked at runtime; the output rank is also checked against its Rust type.
//!
//! ```
//! # #[cfg(feature = "burn")]
//! # fn main() -> einops_rs::Result<()> {
//! use burn_tensor::{Device, Tensor};
//! use einops_rs::{parse_shape, rearrange, reduce, repeat, Reduction};
//!
//! let device = Device::flex();
//! let x = Tensor::<3>::ones([2, 4, 12], &device);
//! let heads: Tensor<4> = rearrange(x, "b t (h d) -> b h t d", &[("h", 3)])?;
//! assert_eq!(heads.dims(), [2, 3, 4, 4]);
//!
//! let pooled: Tensor<2> = reduce(heads, "b h t d -> b d", Reduction::Mean, &[])?;
//! let copies: Tensor<3> = repeat(pooled, "b d -> b n d", &[("n", 2)])?;
//! let shape = parse_shape(&copies.dims(), "batch _ width")?;
//! assert_eq!(shape["batch"], 2);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "burn"))]
//! # fn main() {}
//! ```
//!
//! Transformations require positive dimension sizes and output rank at least one.
//! To reduce every axis, use an explicit singleton output such as `"b c -> 1"`.
//! Without default features, [`parse_shape`] is available without Burn.

#[cfg(feature = "burn")]
mod burn;
mod error;
mod pattern;
#[cfg(any(feature = "burn", test))]
mod plan;

#[cfg(feature = "burn")]
pub use burn::{Reduction, rearrange, reduce, repeat};
pub use error::{Error, ErrorKind, Result};
pub use pattern::parse_shape;
