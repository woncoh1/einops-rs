use einops_rs::parse_shape;

#[test]
fn parses_names_and_skips_underscores() {
    let shape = parse_shape(&[2, 3, 5, 7], "batch _ h w").unwrap();
    assert_eq!(shape["batch"], 2);
    assert_eq!(shape["h"], 5);
    assert_eq!(shape["w"], 7);
    assert_eq!(shape.len(), 3);
    assert!(parse_shape(&[2, 3], "_ _").unwrap().is_empty());
}

#[test]
fn parses_ellipsis_at_any_position_including_no_dimensions() {
    let shape = parse_shape(&[2, 3, 4, 5], "batch ... channels").unwrap();
    assert_eq!(shape["batch"], 2);
    assert_eq!(shape["channels"], 5);
    assert_eq!(shape.len(), 2);
    assert_eq!(parse_shape(&[5], "... c").unwrap()["c"], 5);
    assert_eq!(parse_shape(&[2], "b ...").unwrap()["b"], 2);
    assert!(parse_shape(&[], "...").unwrap().is_empty());
}

#[test]
fn parses_zero_sizes_and_numeric_assertions() {
    let shape = parse_shape(&[2, 0, 1, 3], "batch empty 1 3").unwrap();
    assert_eq!(shape["batch"], 2);
    assert_eq!(shape["empty"], 0);
    assert_eq!(shape.len(), 2);
    assert!(parse_shape(&[2, 3], "2 width").is_ok());
    assert!(parse_shape(&[2, 3], "1 width").is_err());
}

#[test]
fn rejects_invalid_shape_patterns() {
    for (shape, pattern) in [
        (&[2, 3][..], "a a"),
        (&[2, 3][..], "a"),
        (&[2, 3][..], "a b c"),
        (&[2, 3][..], "... ..."),
        (&[6][..], "(a b)"),
        (&[1][..], "()"),
        (&[2][..], "a -> a"),
        (&[2][..], "a-b"),
    ] {
        assert!(
            parse_shape(shape, pattern).is_err(),
            "should reject {shape:?} / {pattern:?}",
        );
    }
}

#[cfg(feature = "burn")]
mod tensors {
    use burn_tensor::{Bool, Device, Int, Tensor};
    use einops_rs::{ErrorKind, Reduction, rearrange, reduce, repeat};

    fn matrix() -> Tensor<2> {
        Tensor::from_data([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::flex())
    }

    #[test]
    fn rejects_incompatible_requested_output_rank() {
        let error = rearrange::<2, 3, _>(matrix(), "h w -> w h", &[]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::RankMismatch);
    }

    #[test]
    fn rejects_scalar_output_with_an_error() {
        let error = reduce::<2, 0>(matrix(), "h w ->", Reduction::Sum, &[]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn rejects_ambiguous_or_inconsistent_axis_lengths() {
        let tensor = Tensor::<1>::from_data([0.0_f32; 6], &Device::flex());
        for lengths in [
            &[][..],
            &[("h", 4)][..],
            &[("h", 0)][..],
            &[("h", 2), ("h", 2)][..],
            &[("h", 2), ("typo", 3)][..],
        ] {
            assert!(rearrange::<1, 2, _>(tensor.clone(), "(h w) -> h w", lengths).is_err());
        }
        assert!(rearrange::<2, 2, _>(matrix(), "h w -> w h", &[("h", 3)]).is_err());
    }

    #[test]
    fn rejects_axis_removal_addition_and_duplication_in_rearrange() {
        for pattern in ["h w -> h", "h w -> h w c", "h w -> h w w", "h h -> h h"] {
            assert!(rearrange::<2, 2, _>(matrix(), pattern, &[]).is_err());
        }
        assert!(repeat::<2, 1, _>(matrix(), "h w -> h", &[]).is_err());
        assert!(reduce::<2, 3>(matrix(), "h w -> h w c", Reduction::Sum, &[("c", 2)]).is_err());
    }

    #[test]
    fn rejects_repeat_overflow_before_allocation() {
        let error = repeat::<2, 3, _>(matrix(), "h w -> h w r", &[("r", usize::MAX)]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Overflow);
    }

    #[test]
    fn preserves_integer_values_through_repeat_and_reorder() {
        let tensor = Tensor::<2, Int>::from_data([[1_i32, 2, 3], [4, 5, 6]], &Device::flex());
        let result: Tensor<2, Int> = repeat(tensor, "h w -> w (r h)", &[("r", 2)]).unwrap();
        assert_eq!(result.dims(), [3, 4]);
        assert_eq!(
            result.try_into_vec_as::<i32>().unwrap(),
            [1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6],
        );
    }

    #[test]
    fn preserves_boolean_values_through_rearrange_and_repeat() {
        let tensor = Tensor::<2, Bool>::from_data(
            [[true, false, true], [false, true, false]],
            &Device::flex(),
        );
        let transposed: Tensor<2, Bool> = rearrange(tensor, "h w -> w h", &[]).unwrap();
        let repeated: Tensor<2, Bool> = repeat(transposed, "w h -> w (h r)", &[("r", 2)]).unwrap();
        assert_eq!(repeated.dims(), [3, 4]);
        assert_eq!(
            repeated.into_data().try_to_vec::<bool>().unwrap(),
            [
                true, true, false, false, false, false, true, true, true, true, false, false
            ],
        );
    }

    #[test]
    fn preserves_gradients_through_rearrange_repeat_and_mean() {
        let device = Device::flex().autodiff();
        let tensor = Tensor::<2>::from_data([[1.0_f32, 2.0], [3.0, 4.0]], &device).require_grad();
        let transposed: Tensor<2> = rearrange(tensor.clone(), "h w -> w h", &[]).unwrap();
        let repeated: Tensor<3> = repeat(transposed, "w h -> r w h", &[("r", 3)]).unwrap();
        let reduced: Tensor<1> = reduce(repeated, "r w h -> h", Reduction::Mean, &[]).unwrap();
        let gradients = reduced.sum().backward();
        let gradient = tensor.grad(&gradients).expect("source gradient retained");
        for value in gradient.into_data().try_to_vec::<f32>().unwrap() {
            assert!((value - 0.5).abs() <= 1e-6, "expected 0.5, got {value}");
        }
    }
}
