//! End-to-end latency on Burn's eager Flex CPU backend, with f32 and no autodiff.
//!
//! Inputs are constructed and results compared before timing. Both paths include
//! an input-handle clone, the operation, device synchronization, and output drop.
//! Flex synchronization is a no-op; it is kept symmetric and explicit here.
//! No tensor-to-host conversion occurs inside a timed iteration.
//!
//! Direct Burn may retain a strided view for split_heads, while einops-rs may
//! materialize it. Logical values and shapes match; storage layouts need not.

use std::{hint::black_box, time::Duration};

use burn_tensor::{DType, Device, Tensor, TensorData};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use einops_rs::{Reduction, parse_shape, rearrange, reduce, repeat};

fn input<const D: usize>(shape: [usize; D], device: &Device) -> Tensor<D> {
    let values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|index| ((index * 37 % 257) as f32 - 128.0) / 32.0)
        .collect();
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn assert_equivalent<const D: usize>(actual: Tensor<D>, expected: Tensor<D>) {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.dtype(), DType::F32);
    assert_eq!(expected.dtype(), DType::F32);
    let actual = actual.into_data().try_to_vec::<f32>().unwrap();
    let expected = expected.into_data().try_to_vec::<f32>().unwrap();
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: einops-rs={actual}, direct Burn={expected}",
        );
    }
}

fn bench_pair<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    size: &str,
    input: Tensor<IN>,
    device: &Device,
    einops: impl Fn(Tensor<IN>) -> Tensor<OUT>,
    direct: impl Fn(Tensor<IN>) -> Tensor<OUT>,
) {
    assert!(!input.is_autodiff());
    assert_equivalent(einops(input.clone()), direct(input.clone()));
    device.sync().unwrap();

    group.bench_function(BenchmarkId::new("einops_rs", size), |b| {
        b.iter(|| {
            let output = einops(black_box(input.clone()));
            device.sync().unwrap();
            black_box(output)
        });
    });
    group.bench_function(BenchmarkId::new("direct_burn", size), |b| {
        b.iter(|| {
            let output = direct(black_box(input.clone()));
            device.sync().unwrap();
            black_box(output)
        });
    });
}

fn tensor_ops(c: &mut Criterion) {
    let device = Device::flex();

    let mut group = c.benchmark_group("rearrange_flatten");
    for (size, [batch, time, width]) in [("small", [2, 8, 32]), ("medium", [8, 128, 256])] {
        bench_pair(
            &mut group,
            size,
            input([batch, time, width], &device),
            &device,
            |x| rearrange(x, "b t d -> b (t d)", &[]).unwrap(),
            |x| x.reshape([batch, time * width]),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("rearrange_split_heads");
    for (size, [batch, time, width], heads) in
        [("small", [2, 8, 32], 4), ("medium", [8, 128, 256], 8)]
    {
        bench_pair(
            &mut group,
            size,
            input([batch, time, width], &device),
            &device,
            |x| rearrange(x, "b t (h d) -> b h t d", &[("h", heads)]).unwrap(),
            |x| {
                x.reshape([batch, time, heads, width / heads])
                    .swap_dims(1, 2)
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("rearrange_permute_merge");
    for (size, [batch, height, width, channels]) in
        [("small", [2, 4, 4, 8]), ("medium", [8, 32, 32, 32])]
    {
        bench_pair(
            &mut group,
            size,
            input([batch, height, width, channels], &device),
            &device,
            |x| rearrange(x, "b h w c -> b (c h w)", &[]).unwrap(),
            // Merging the permuted axes requires materialization in both paths.
            |x| {
                x.permute([0, 3, 1, 2])
                    .reshape([batch, channels * height * width])
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("repeat");
    let copies = 4;
    for (size, [batch, width]) in [("small", [4, 32]), ("medium", [128, 256])] {
        bench_pair(
            &mut group,
            size,
            input([batch, width], &device),
            &device,
            |x| repeat(x, "b d -> b r d", &[("r", copies)]).unwrap(),
            |x| x.reshape([batch, 1, width]).repeat_dim(1, copies),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("reduce_mean");
    for (size, [batch, time, width]) in [("small", [2, 8, 32]), ("medium", [8, 128, 256])] {
        bench_pair(
            &mut group,
            size,
            input([batch, time, width], &device),
            &device,
            |x| reduce(x, "b t d -> b d", Reduction::Mean, &[]).unwrap(),
            |x| x.mean_dim(1).reshape([batch, width]),
        );
    }
    group.finish();
}

fn shape_metadata(c: &mut Criterion) {
    c.bench_function("parse_shape/three_axes", |b| {
        b.iter(|| {
            black_box(
                parse_shape(black_box(&[8, 128, 256]), black_box("batch time width")).unwrap(),
            )
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = tensor_ops, shape_metadata
}
criterion_main!(benches);
