use burn_tensor::{Device, Tensor};
use einops_rs::{Reduction, parse_shape, rearrange, reduce, repeat};

fn main() -> einops_rs::Result<()> {
    let device = Device::flex();
    let x = Tensor::<3>::ones([2, 4, 12], &device);

    // Split the feature dimension into attention heads.
    let heads: Tensor<4> = rearrange(x, "b t (h d) -> b h t d", &[("h", 3)])?;
    assert_eq!(heads.dims(), [2, 3, 4, 4]);
    println!("attention heads: {:?}", heads.dims());

    // Average over heads and time, then repeat along a new axis.
    let pooled: Tensor<2> = reduce(heads, "b h t d -> b d", Reduction::Mean, &[])?;
    let copies: Tensor<3> = repeat(pooled, "b d -> b n d", &[("n", 2)])?;
    assert_eq!(copies.dims(), [2, 2, 4]);
    println!("repeated features: {:?}", copies.dims());

    let sizes = parse_shape(&copies.dims(), "batch _ width")?;
    println!("named dimensions: {sizes:?}");
    Ok(())
}
