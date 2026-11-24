//! Shape inference and operation planning, independent of the tensor engine.

use std::collections::{BTreeMap, BTreeSet};

use crate::pattern::{Axis, Expression, match_rank, parse_expression, valid_name};
use crate::{Error, ErrorKind, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operation {
    Rearrange,
    Reduce,
    Repeat,
}

#[derive(Debug)]
pub(crate) struct Plan {
    pub initial_shape: Vec<usize>,
    pub reduce_axes: Vec<usize>,
    pub reduced_shape: Vec<usize>,
    pub permutation: Vec<usize>,
    pub repeat_axes: Vec<(usize, usize)>,
    pub output_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Id {
    Named(String),
    Anonymous(usize),
    Ellipsis(usize),
}

impl Id {
    fn label(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Anonymous(index) => format!("anonymous axis {index}"),
            Self::Ellipsis(index) => format!("ellipsis axis {index}"),
        }
    }
}

pub(crate) fn plan(
    shape: &[usize],
    pattern: &str,
    axes: &[(&str, usize)],
    operation: Operation,
) -> Result<Plan> {
    let (left, right) = pattern.split_once("->").ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidPattern,
            pattern,
            "expected exactly one `->`",
        )
    })?;
    if right.contains("->") {
        return Err(Error::new(
            ErrorKind::InvalidPattern,
            pattern,
            "expected exactly one `->`",
        ));
    }
    let left = parse_expression(left, pattern, false, false)?;
    let right = parse_expression(right, pattern, false, true)?;
    check_ellipsis(&left, &right, pattern, operation)?;
    check_literals(&left, &right, pattern, operation)?;

    let explicit = left.groups.len() - usize::from(left.has_ellipsis);
    let ellipsis_len = match_rank(shape.len(), explicit, left.has_ellipsis, pattern)?;
    if shape.contains(&0) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            pattern,
            "zero-length dimensions are not supported by tensor operations in 0.1.0",
        ));
    }
    product(shape.iter().copied(), pattern)?;

    let mut lengths = argument_lengths(axes, &left, &right, pattern)?;
    let mut next_anonymous = 0;
    let left = expand(&left, ellipsis_len, &mut next_anonymous, &mut lengths);
    let right = expand(&right, ellipsis_len, &mut next_anonymous, &mut lengths);
    let input: Vec<_> = left.iter().flatten().cloned().collect();
    let output: Vec<_> = right.iter().flatten().cloned().collect();
    let input_set: BTreeSet<_> = input.iter().cloned().collect();
    let output_set: BTreeSet<_> = output.iter().cloned().collect();
    check_axis_sets(&input_set, &output_set, pattern, operation)?;

    for (index, (group, &actual)) in left.iter().zip(shape).enumerate() {
        let unknown: Vec<_> = group
            .iter()
            .filter(|id| !lengths.contains_key(*id))
            .collect();
        let known = product(
            group.iter().filter_map(|id| lengths.get(id).copied()),
            pattern,
        )?;
        match unknown.as_slice() {
            [] if known == actual => {}
            [] => {
                return Err(Error::new(
                    ErrorKind::InvalidShape,
                    pattern,
                    format!(
                        "input dimension {index} has length {actual}, but its axes require {known}"
                    ),
                ));
            }
            [id] => {
                if actual % known != 0 {
                    return Err(Error::new(
                        ErrorKind::InvalidShape,
                        pattern,
                        format!(
                            "input dimension {index} has length {actual}, which is not divisible by known factor {known} while inferring `{}`",
                            id.label()
                        ),
                    ));
                }
                lengths.insert((*id).clone(), actual / known);
            }
            _ => {
                let names = unknown
                    .iter()
                    .map(|id| id.label())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(Error::new(
                    ErrorKind::InvalidAxes,
                    pattern,
                    format!(
                        "cannot infer multiple axes ({names}) in input dimension {index}; supply all but one of their lengths"
                    ),
                ));
            }
        }
    }
    for id in &output {
        if !lengths.contains_key(id) {
            return Err(Error::new(
                ErrorKind::InvalidAxes,
                pattern,
                format!("supply a length for new repeat axis `{}`", id.label()),
            ));
        }
    }

    let output_shape = right
        .iter()
        .map(|group| product(group.iter().map(|id| lengths[id]), pattern))
        .collect::<Result<Vec<_>>>()?;
    product(output_shape.iter().copied(), pattern)?;
    let retained: Vec<_> = input.iter().filter(|id| output_set.contains(*id)).collect();
    let positions: BTreeMap<_, _> = retained
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    Ok(Plan {
        initial_shape: input.iter().map(|id| lengths[id]).collect(),
        reduce_axes: input
            .iter()
            .enumerate()
            .filter_map(|(i, id)| (!output_set.contains(id)).then_some(i))
            .collect(),
        reduced_shape: retained.iter().map(|id| lengths[*id]).collect(),
        permutation: output
            .iter()
            .filter_map(|id| positions.get(id).copied())
            .collect(),
        repeat_axes: output
            .iter()
            .enumerate()
            .filter_map(|(i, id)| (!input_set.contains(id)).then_some((i, lengths[id])))
            .collect(),
        output_shape,
    })
}

fn product(mut values: impl Iterator<Item = usize>, pattern: &str) -> Result<usize> {
    values.try_fold(1usize, |acc, value| {
        acc.checked_mul(value).ok_or_else(|| {
            Error::new(
                ErrorKind::Overflow,
                pattern,
                "dimension product exceeds usize",
            )
        })
    })
}

fn check_ellipsis(
    left: &Expression,
    right: &Expression,
    pattern: &str,
    operation: Operation,
) -> Result<()> {
    if right.has_ellipsis && !left.has_ellipsis {
        return Err(Error::new(
            ErrorKind::InvalidPattern,
            pattern,
            "an output ellipsis requires an input ellipsis",
        ));
    }
    if left.has_ellipsis && !right.has_ellipsis && operation != Operation::Reduce {
        return Err(Error::new(
            ErrorKind::InvalidPattern,
            pattern,
            "only reduce may remove an ellipsis",
        ));
    }
    Ok(())
}

fn check_literals(
    left: &Expression,
    right: &Expression,
    pattern: &str,
    operation: Operation,
) -> Result<()> {
    let has_nonunit_literal = |expr: &Expression| {
        expr.groups
            .iter()
            .flatten()
            .any(|axis| matches!(axis, Axis::Literal(n) if *n != 1))
    };
    if (has_nonunit_literal(left) && operation != Operation::Reduce)
        || (has_nonunit_literal(right) && operation != Operation::Repeat)
    {
        return Err(Error::new(
            ErrorKind::InvalidPattern,
            pattern,
            "literal lengths greater than 1 are allowed only on the input of reduce or the output of repeat; name other axes",
        ));
    }
    Ok(())
}

fn argument_lengths(
    axes: &[(&str, usize)],
    left: &Expression,
    right: &Expression,
    pattern: &str,
) -> Result<BTreeMap<Id, usize>> {
    let names: BTreeSet<_> = left
        .groups
        .iter()
        .chain(&right.groups)
        .flatten()
        .filter_map(|axis| match axis {
            Axis::Named(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    let mut lengths = BTreeMap::new();
    for &(name, length) in axes {
        if !valid_name(name) || !names.contains(name) {
            return Err(Error::new(
                ErrorKind::InvalidAxes,
                pattern,
                format!("axis-length argument `{name}` does not name an axis in the pattern"),
            ));
        }
        if length == 0 {
            return Err(Error::new(
                ErrorKind::InvalidAxes,
                pattern,
                format!("axis `{name}` must have a positive length"),
            ));
        }
        if lengths.insert(Id::Named(name.to_owned()), length).is_some() {
            return Err(Error::new(
                ErrorKind::InvalidAxes,
                pattern,
                format!("axis-length argument `{name}` was supplied more than once"),
            ));
        }
    }
    Ok(lengths)
}

fn expand(
    expression: &Expression,
    ellipsis_len: usize,
    next_anonymous: &mut usize,
    lengths: &mut BTreeMap<Id, usize>,
) -> Vec<Vec<Id>> {
    let mut groups = Vec::new();
    for (group, &grouped) in expression.groups.iter().zip(&expression.grouped) {
        if !grouped && group.as_slice() == [Axis::Ellipsis] {
            groups.extend((0..ellipsis_len).map(|i| vec![Id::Ellipsis(i)]));
            continue;
        }
        let mut expanded = Vec::new();
        for axis in group {
            match axis {
                Axis::Named(name) => expanded.push(Id::Named(name.clone())),
                Axis::Literal(1) => {}
                Axis::Literal(length) => {
                    let id = Id::Anonymous(*next_anonymous);
                    *next_anonymous += 1;
                    lengths.insert(id.clone(), *length);
                    expanded.push(id);
                }
                Axis::Ellipsis => expanded.extend((0..ellipsis_len).map(Id::Ellipsis)),
                Axis::Skip => unreachable!("underscores are rejected in transform patterns"),
            }
        }
        groups.push(expanded);
    }
    groups
}

fn check_axis_sets(
    input: &BTreeSet<Id>,
    output: &BTreeSet<Id>,
    pattern: &str,
    operation: Operation,
) -> Result<()> {
    let removed: Vec<_> = input.difference(output).map(Id::label).collect();
    let added: Vec<_> = output.difference(input).map(Id::label).collect();
    if !removed.is_empty() && operation != Operation::Reduce {
        return Err(Error::new(
            ErrorKind::InvalidAxes,
            pattern,
            format!(
                "only reduce may remove axes; missing from output: {}",
                removed.join(", ")
            ),
        ));
    }
    if !added.is_empty() && operation != Operation::Repeat {
        return Err(Error::new(
            ErrorKind::InvalidAxes,
            pattern,
            format!(
                "only repeat may introduce axes; new in output: {}",
                added.join(", ")
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_splits_then_permutes() {
        let p = plan(
            &[2, 4, 12],
            "b t (h d) -> b h t d",
            &[("h", 3)],
            Operation::Rearrange,
        )
        .unwrap();
        assert_eq!(p.initial_shape, [2, 4, 3, 4]);
        assert_eq!(p.output_shape, [2, 3, 4, 4]);
        assert_eq!(p.permutation, [0, 2, 1, 3]);
    }

    #[test]
    fn reductions_and_repetitions_have_explicit_axis_order() {
        let p = plan(&[2, 6, 8], "b (h 2) (w 2) -> w b h", &[], Operation::Reduce).unwrap();
        assert_eq!(p.reduce_axes, [2, 4]);
        assert_eq!(p.reduced_shape, [2, 3, 4]);
        assert_eq!(p.permutation, [2, 0, 1]);
        assert_eq!(p.output_shape, [4, 2, 3]);
        let p = plan(
            &[2, 3],
            "b c -> (r b) (c s)",
            &[("r", 2), ("s", 4)],
            Operation::Repeat,
        )
        .unwrap();
        assert_eq!(p.repeat_axes, [(0, 2), (3, 4)]);
        assert_eq!(p.output_shape, [4, 12]);
    }

    #[test]
    fn singletons_and_ellipsis_can_have_no_atomic_axes() {
        let p = plan(&[1], "1 ... -> (...) 1", &[], Operation::Rearrange).unwrap();
        assert!(p.initial_shape.is_empty());
        assert_eq!(p.output_shape, [1, 1]);
        let p = plan(&[2, 3, 4], "b ... -> (...) b", &[], Operation::Rearrange).unwrap();
        assert_eq!(p.output_shape, [12, 2]);
        assert_eq!(p.permutation, [1, 2, 0]);
    }

    #[test]
    fn shape_inference_does_not_guess_or_ignore_arguments() {
        let cases = [
            (vec![12], "(h w) -> h w", vec![], ErrorKind::InvalidAxes),
            (vec![1], "(h w) -> h w", vec![], ErrorKind::InvalidAxes),
            (
                vec![12],
                "(h w) -> h w",
                vec![("h", 5)],
                ErrorKind::InvalidShape,
            ),
            (
                vec![12],
                "(h w) -> h w",
                vec![("h", 3), ("w", 5)],
                ErrorKind::InvalidShape,
            ),
            (vec![2], "b -> b", vec![("x", 2)], ErrorKind::InvalidAxes),
            (
                vec![2],
                "b -> b",
                vec![("b", 2), ("b", 2)],
                ErrorKind::InvalidAxes,
            ),
            (vec![2], "b -> b", vec![("b", 0)], ErrorKind::InvalidAxes),
            (vec![2], "b -> b", vec![("b", 3)], ErrorKind::InvalidShape),
            (vec![0], "b -> b", vec![], ErrorKind::Unsupported),
        ];
        for (shape, pattern, axes, kind) in cases {
            assert_eq!(
                plan(&shape, pattern, &axes, Operation::Rearrange)
                    .unwrap_err()
                    .kind(),
                kind,
                "{pattern}"
            );
        }
    }

    #[test]
    fn malformed_patterns_and_axis_changes_fail_before_execution() {
        for pattern in [
            "a",
            "a -> a -> a",
            "a -> a a",
            "a -> (a",
            "((a)) -> a",
            "_ -> _",
            "(a ...) -> a",
            "a -> ... a",
            "a ... -> a",
            "0 -> 0",
            "2 -> 2",
            "a -> a-b",
        ] {
            assert!(
                plan(&[2], pattern, &[], Operation::Rearrange).is_err(),
                "{pattern}"
            );
        }
        assert_eq!(
            plan(&[2, 3], "a b -> a", &[], Operation::Repeat)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidAxes
        );
        assert_eq!(
            plan(&[2], "a -> a b", &[("b", 3)], Operation::Reduce)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidAxes
        );
        assert_eq!(
            plan(&[2], "a -> a b", &[], Operation::Repeat)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidAxes
        );
    }

    #[test]
    fn products_are_checked_without_allocating_tensors() {
        for (shape, pattern, axes, op) in [
            (
                vec![usize::MAX, 2],
                "a b -> (a b)",
                vec![],
                Operation::Rearrange,
            ),
            (
                vec![1],
                "a -> a b c",
                vec![("b", usize::MAX), ("c", 2)],
                Operation::Repeat,
            ),
            (
                vec![2],
                "(a b c) -> a b c",
                vec![("a", usize::MAX), ("b", 2)],
                Operation::Rearrange,
            ),
        ] {
            assert_eq!(
                plan(&shape, pattern, &axes, op).unwrap_err().kind(),
                ErrorKind::Overflow
            );
        }
    }
}
