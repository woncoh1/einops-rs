//! Tensor operations for Burn 0.22.

use burn_tensor::{DType, Tensor, kind::Basic};

use crate::plan::{Operation, Plan, plan};
use crate::{Error, ErrorKind, Result};

/// The operation applied to axes omitted from a reduction's output pattern.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction {
    /// Add the elements along each reduced axis.
    Sum,
    /// Compute the arithmetic mean along each reduced axis.
    Mean,
    /// Keep the smallest element along each reduced axis.
    Min,
    /// Keep the largest element along each reduced axis.
    Max,
    /// Multiply the elements along each reduced axis.
    Prod,
}

/// Reorder, split, merge, insert, or remove singleton axes.
///
/// `axes` supplies lengths that cannot be inferred from the input shape. The
/// output rank is inferred from the surrounding Rust type, while the pattern
/// and dimension lengths are checked at runtime.
///
/// Float, integer, and boolean tensors are supported. Quantized tensors,
/// zero-length dimensions, and rank-zero inputs or outputs are not supported.
/// Device, element kind, and existing autodiff tracking are preserved.
pub fn rearrange<const IN: usize, const OUT: usize, K: Basic>(
    tensor: Tensor<IN, K>,
    pattern: &str,
    axes: &[(&str, usize)],
) -> Result<Tensor<OUT, K>> {
    let plan = prepare::<IN, OUT, K>(&tensor, pattern, axes, Operation::Rearrange)?;
    let flat = tensor.reshape([product(&plan.initial_shape)]);
    Ok(finish(flat, &plan))
}

/// Repeat a tensor along new axes, optionally reordering or grouping axes.
///
/// Every new named axis needs a length in `axes`. A new numeric axis, such as
/// `3` in `"h w -> h w 3"`, provides its own length.
/// Supports the same tensor kinds and shape restrictions as [`rearrange`].
pub fn repeat<const IN: usize, const OUT: usize, K: Basic>(
    tensor: Tensor<IN, K>,
    pattern: &str,
    axes: &[(&str, usize)],
) -> Result<Tensor<OUT, K>> {
    let plan = prepare::<IN, OUT, K>(&tensor, pattern, axes, Operation::Repeat)?;
    let flat = tensor.reshape([product(&plan.initial_shape)]);
    Ok(finish(flat, &plan))
}

/// Reduce omitted axes of a floating-point tensor, then arrange the result.
///
/// Use `"b c h w -> b c"` for global pooling, or
/// `"b c (h ph) (w pw) -> b c h w"` with patch sizes in `axes` for local pooling.
/// For a complete reduction, use a singleton output such as `"b c -> 1"`.
/// Existing Burn autodiff tracking is preserved.
pub fn reduce<const IN: usize, const OUT: usize>(
    tensor: Tensor<IN>,
    pattern: &str,
    reduction: Reduction,
    axes: &[(&str, usize)],
) -> Result<Tensor<OUT>> {
    let plan = prepare::<IN, OUT, _>(&tensor, pattern, axes, Operation::Reduce)?;
    let mut shape = plan.initial_shape.clone();
    let mut flat = tensor.reshape([product(&shape)]);

    // Removing axes from right to left keeps their original indexes valid.
    for &axis in plan.reduce_axes.iter().rev() {
        let before = product(&shape[..axis]);
        let after = product(&shape[axis + 1..]);
        if shape[axis] != 1 {
            let grouped = flat.reshape([before, shape[axis], after]);
            let reduced = match reduction {
                Reduction::Sum => grouped.sum_dim(1),
                Reduction::Mean => grouped.mean_dim(1),
                Reduction::Min => grouped.min_dim(1),
                Reduction::Max => grouped.max_dim(1),
                Reduction::Prod => grouped.prod_dim(1),
            };
            flat = reduced.reshape([before * after]);
        }
        shape.remove(axis);
    }
    Ok(finish(flat, &plan))
}

fn prepare<const IN: usize, const OUT: usize, K: Basic>(
    tensor: &Tensor<IN, K>,
    pattern: &str,
    axes: &[(&str, usize)],
    operation: Operation,
) -> Result<Plan> {
    if IN == 0 || OUT == 0 {
        return Err(Error::new(
            ErrorKind::Unsupported,
            pattern,
            "rank-zero tensors are not supported; use a singleton axis `1`",
        ));
    }
    if matches!(tensor.dtype(), DType::QFloat(_)) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            pattern,
            "quantized tensors are not supported; dequantize the tensor first",
        ));
    }
    let plan = plan(&tensor.dims(), pattern, axes, operation)?;
    if plan.output_shape.len() != OUT {
        return Err(Error::new(
            ErrorKind::RankMismatch,
            pattern,
            format!(
                "pattern produces rank {}, but the requested output tensor has rank {OUT}",
                plan.output_shape.len()
            ),
        ));
    }
    // Burn's public shape indexing uses signed sizes. The core planner already
    // checks usize overflow; this catches its stricter backend boundary.
    if product(&plan.initial_shape) > isize::MAX as usize
        || product(&plan.output_shape) > isize::MAX as usize
    {
        return Err(Error::new(
            ErrorKind::Overflow,
            pattern,
            "tensor size exceeds Burn's signed indexing range",
        ));
    }
    Ok(plan)
}

fn finish<const OUT: usize, K: Basic>(mut flat: Tensor<1, K>, plan: &Plan) -> Tensor<OUT, K> {
    let mut shape = plan.reduced_shape.clone();
    let mut order: Vec<_> = (0..shape.len()).collect();
    let total = product(&shape);

    // Move each desired axis into place. Grouping the untouched prefix/suffix
    // makes a public rank-4 swap sufficient even when a pattern has many axes.
    // No tensor values are read onto the host and no fixed rank cap is needed.
    for (target, &original_axis) in plan.permutation.iter().enumerate() {
        let source = order
            .iter()
            .position(|&axis| axis == original_axis)
            .expect("the planner creates a valid permutation");
        if source == target {
            continue;
        }
        let between = product(&shape[target..source]);
        if shape[source] != 1 && between != 1 {
            flat = flat
                .reshape([
                    product(&shape[..target]),
                    between,
                    shape[source],
                    product(&shape[source + 1..]),
                ])
                .swap_dims(1, 2)
                .reshape([total]);
        }
        let length = shape.remove(source);
        shape.insert(target, length);
        let axis = order.remove(source);
        order.insert(target, axis);
    }

    for &(position, length) in &plan.repeat_axes {
        if length != 1 {
            let before = product(&shape[..position]);
            let after = product(&shape[position..]);
            flat = flat
                .reshape([before, 1, after])
                .repeat_dim(1, length)
                .reshape([before * length * after]);
        }
        shape.insert(position, length);
    }

    let output: [usize; OUT] = std::array::from_fn(|axis| plan.output_shape[axis]);
    flat.reshape(output)
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use burn_tensor::{Device, TensorData};

    use super::*;

    #[test]
    fn reverses_ten_elementary_axes_without_a_rank_cap() {
        let tensor = Tensor::<1>::from_data(
            TensorData::new((0..1024).map(|n| n as f32).collect(), [1024]),
            &Device::flex(),
        );
        let output: Tensor<1> = rearrange(
            tensor,
            "(a b c d e f g h i j) -> (j i h g f e d c b a)",
            &[
                ("a", 2),
                ("b", 2),
                ("c", 2),
                ("d", 2),
                ("e", 2),
                ("f", 2),
                ("g", 2),
                ("h", 2),
                ("i", 2),
            ],
        )
        .unwrap();
        let expected: Vec<f32> = (0_u32..1024)
            .map(|index| (index.reverse_bits() >> 22) as f32)
            .collect();
        assert_eq!(output.into_data().try_to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn flattening_a_transposed_input_preserves_logical_order() {
        let tensor = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::flex())
            .transpose();
        let output: Tensor<1> = rearrange(tensor, "a b -> (a b)", &[]).unwrap();
        assert_eq!(
            output.into_data().try_to_vec::<f32>().unwrap(),
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
