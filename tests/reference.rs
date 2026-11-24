//! Numerical compatibility with the documented Python Einops subset.
//!
//! Expected values are checked in; Python is only needed to regenerate them.

#![cfg(feature = "burn")]

use burn_tensor::{Device, Tensor, TensorData};
use einops_rs::{Reduction, rearrange, reduce, repeat};

struct Fixture {
    name: &'static str,
    operation: &'static str,
    input_shape: &'static [usize],
    input: &'static [f32],
    pattern: &'static str,
    axes: &'static [(&'static str, usize)],
    reduction: Option<&'static str>,
    output_shape: &'static [usize],
    output: &'static [f32],
}

include!("fixtures/einops.rs");

fn check<const IN: usize, const OUT: usize>(fixture: &Fixture) {
    let tensor = Tensor::<IN>::from_data(
        TensorData::new(fixture.input.to_vec(), fixture.input_shape.to_vec()),
        &Device::flex(),
    );
    let result: Tensor<OUT> = match fixture.operation {
        "rearrange" => rearrange(tensor, fixture.pattern, fixture.axes),
        "repeat" => repeat(tensor, fixture.pattern, fixture.axes),
        "reduce" => {
            let reduction = match fixture.reduction.expect("reduction fixture") {
                "sum" => Reduction::Sum,
                "mean" => Reduction::Mean,
                "min" => Reduction::Min,
                "max" => Reduction::Max,
                "prod" => Reduction::Prod,
                other => panic!("unknown reference reduction: {other}"),
            };
            reduce(tensor, fixture.pattern, reduction, fixture.axes)
        }
        other => panic!("unknown reference operation: {other}"),
    }
    .unwrap_or_else(|error| panic!("{}: {error}", fixture.name));
    assert_eq!(
        result.dims().as_slice(),
        fixture.output_shape,
        "{}",
        fixture.name
    );
    let actual = result.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(actual.len(), fixture.output.len(), "{}", fixture.name);
    for (index, (&actual, &expected)) in actual.iter().zip(fixture.output).enumerate() {
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{} at index {index}: actual {actual}, expected {expected}",
            fixture.name,
        );
    }
}

fn dispatch_output<const IN: usize>(fixture: &Fixture) {
    match fixture.output_shape.len() {
        1 => check::<IN, 1>(fixture),
        2 => check::<IN, 2>(fixture),
        3 => check::<IN, 3>(fixture),
        4 => check::<IN, 4>(fixture),
        5 => check::<IN, 5>(fixture),
        6 => check::<IN, 6>(fixture),
        rank => panic!("reference runner does not handle output rank {rank}"),
    }
}

#[test]
fn matches_python_einops_values_and_shapes() {
    for fixture in FIXTURES {
        match fixture.input_shape.len() {
            1 => dispatch_output::<1>(fixture),
            2 => dispatch_output::<2>(fixture),
            3 => dispatch_output::<3>(fixture),
            4 => dispatch_output::<4>(fixture),
            5 => dispatch_output::<5>(fixture),
            6 => dispatch_output::<6>(fixture),
            rank => panic!("reference runner does not handle input rank {rank}"),
        }
    }
}
