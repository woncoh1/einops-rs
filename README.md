# einops-rs

Simple, readable tensor transformations for Rust and [Burn](https://burn.dev/).

Inspired by **Alex Rogozhnikov's [Einops](https://github.com/arogozhnikov/einops)**
and its [pattern notation](https://einops.rocks/). This is an independent Rust
implementation, not an official Einops project.

## Install

Version **0.1.0** requires **Rust 1.95+** and supports **Burn 0.22.1**.
This source release is ready for GitHub; it has not been published to crates.io.
Use a local checkout alongside your application:

```toml
[dependencies]
einops-rs = { path = "../einops-rs" }
burn = { version = "0.22.1", default-features = false, features = ["std", "flex"] }
```

`flex` selects Burn's CPU backend for this example. Choose the Burn backend
appropriate to your application. `einops-rs` does not select an execution backend.

## Quick start

```rust
use burn::prelude::*;
use einops_rs::{parse_shape, rearrange, reduce, repeat, Reduction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::flex();
    let x = Tensor::<3>::ones([2, 4, 12], &device);

    // Split the last dimension into three attention heads.
    let heads: Tensor<4> = rearrange(x, "b t (h d) -> b h t d", &[("h", 3)])?;
    assert_eq!(heads.dims(), [2, 3, 4, 4]);

    // Average across time.
    let pooled: Tensor<3> = reduce(heads, "b h t d -> b h d", Reduction::Mean, &[])?;
    assert_eq!(pooled.dims(), [2, 3, 4]);

    // Add a repeated axis.
    let copies: Tensor<4> = repeat(pooled, "b h d -> b h copy d", &[("copy", 2)])?;
    assert_eq!(copies.dims(), [2, 3, 2, 4]);

    let shape = parse_shape(&copies.dims(), "batch _ _ width")?;
    assert_eq!(shape["batch"], 2);
    assert_eq!(shape["width"], 4);
    Ok(())
}
```

The package name is `einops-rs`; the Rust import is `einops_rs`.
Annotate the output tensor's rank when Rust cannot infer it from surrounding code.
Pass `&[]` when no axis lengths are needed.

## Four operations

| Operation | Purpose |
| --- | --- |
| `rearrange(tensor, pattern, lengths)` | Reorder, split, merge, insert, or remove singleton axes. |
| `reduce(tensor, pattern, reduction, lengths)` | Reduce axes omitted from the output. |
| `repeat(tensor, pattern, lengths)` | Introduce repeated axes and reorder or merge them. |
| `parse_shape(shape, pattern)` | Return named dimensions as a `BTreeMap<String, usize>`. |

`rearrange` and `repeat` support Burn float, integer, and boolean tensors.
`reduce` supports float tensors with `Reduction::{Sum, Mean, Min, Max, Prod}`.
The tensor operations consume their input; clone a tensor first if you also need
to retain it.

## Patterns

| Pattern | Meaning |
| --- | --- |
| `b h w c -> b c h w` | Move channels before spatial dimensions. |
| `b c h w -> b (c h w)` | Flatten each image. |
| `b (h d) -> b h d`, with `h` supplied | Split a dimension and infer `d`. |
| `b 1 c -> b c` | Remove a singleton axis. |
| `... c -> c ...` | Move the last axis to the front. |
| `... c -> c (...)` | Merge the leading axes. |
| `b c h w -> b c` | Reduce spatial dimensions. |
| `b (h 2) (w 2) c -> b h w c` | Reduce each 2 × 2 patch. |
| `h w -> h w c`, with `c` supplied | Repeat a grayscale image across channels. |
| `h w -> (h 2) (w 2)` | Repeat each pixel into a 2 × 2 block. |

Parentheses group axes in row-major order: the last axis changes fastest.
An input group may have at most one unknown length. `1` and `()` represent
singleton dimensions. Numeric axes larger than one are supported on the reduced
side of `reduce` and the introduced side of `repeat`.
Axis names start with an ASCII letter and may contain letters, digits, and
internal underscores. Separate names with spaces.

`parse_shape` accepts a shape slice and a pattern without `->`. Use `_` to skip
one dimension, `...` to skip any number, or a number to check a dimension's size.
Groups are not supported by `parse_shape`.

```rust
use einops_rs::parse_shape;

let shape = parse_shape(&[2, 3, 16, 16], "batch 3 ...").unwrap();
assert_eq!(shape["batch"], 2);
```

## Scope of 0.1.0

- Patterns and dimension sizes are validated at runtime and return descriptive
  errors. The API uses ordinary functions, with output rank represented in the
  Rust type. It does not provide compile-time pattern macros.
- Tensor transformations require positive dimension sizes and at least one
  input and output axis. For a total reduction, use a pattern such as `b c -> 1`.
  Quantized tensors are not supported.
- Tensor work stays in Burn operations; this crate does not copy tensors to the
  host. Allocations, numerical behavior, and backend failures follow Burn.
  There is no guarantee that a transformation is a zero-copy view.
- Tensor-list stacking, `einsum`, `pack`, and `unpack` are outside this release.

The `burn` feature is enabled by default. For shape parsing without Burn:

```toml
einops-rs = { path = "../einops-rs", default-features = false }
```

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --no-default-features
cargo doc --all-features --no-deps
```

Tests include expected tensor values generated with Python Einops. The committed
fixtures allow normal Rust tests to run without Python.

See [RELEASING.md](RELEASING.md) to push the project to GitHub and tag `v0.1.0`.

## Acknowledgments and license

Alex Rogozhnikov created the original [Einops](https://einops.rocks/), which
introduced the notation and operations that inspired this library. See the
[upstream repository](https://github.com/arogozhnikov/einops) for the original
implementation, documentation, and contributors.

This Rust implementation is licensed under the [MIT License](LICENSE).
