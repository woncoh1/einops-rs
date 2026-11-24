use std::collections::{BTreeMap, BTreeSet};

use crate::{Error, ErrorKind, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Axis {
    Named(String),
    Literal(usize),
    Ellipsis,
    Skip,
}

#[derive(Debug, Clone)]
pub(crate) struct Expression {
    pub groups: Vec<Vec<Axis>>,
    pub grouped: Vec<bool>,
    pub has_ellipsis: bool,
}

pub(crate) fn valid_name(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        && !name.ends_with('_')
}

/// Read dimension lengths without evaluating a tensor operation.
///
/// Use `_` to skip a dimension, `...` to skip any number of dimensions, and
/// a positive integer to assert a fixed length. Named dimensions may be zero.
/// Parenthesized groups are not supported in shape queries.
///
/// ```
/// use einops_rs::parse_shape;
/// let sizes = parse_shape(&[2, 3, 5, 7], "batch _ height width")?;
/// assert_eq!(sizes["height"], 5);
/// assert_eq!(parse_shape(&[2, 3, 5, 7], "batch ... width")?["width"], 7);
/// # Ok::<(), einops_rs::Error>(())
/// ```
pub fn parse_shape(shape: &[usize], pattern: &str) -> Result<BTreeMap<String, usize>> {
    let expression = parse_expression(pattern, pattern, true, false)?;
    if expression.grouped.iter().any(|&grouped| grouped) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            pattern,
            "parse_shape does not support parenthesized groups; use one name per dimension",
        ));
    }
    let explicit = expression.groups.len() - usize::from(expression.has_ellipsis);
    let ellipsis_len = match_rank(shape.len(), explicit, expression.has_ellipsis, pattern)?;
    let mut sizes = BTreeMap::new();
    let mut index = 0;
    for group in expression.groups {
        match group.first() {
            Some(Axis::Ellipsis) => index += ellipsis_len,
            Some(Axis::Named(name)) => {
                sizes.insert(name.clone(), shape[index]);
                index += 1;
            }
            Some(Axis::Skip) => index += 1,
            Some(Axis::Literal(expected)) => {
                let actual = shape[index];
                if actual != *expected {
                    return Err(Error::new(
                        ErrorKind::InvalidShape,
                        pattern,
                        format!(
                            "dimension {index} is {actual}, but the pattern requires {expected}"
                        ),
                    ));
                }
                index += 1;
            }
            None => unreachable!("shape queries contain one token per group"),
        }
    }
    Ok(sizes)
}

pub(crate) fn match_rank(
    rank: usize,
    explicit: usize,
    has_ellipsis: bool,
    pattern: &str,
) -> Result<usize> {
    if (has_ellipsis && rank >= explicit) || (!has_ellipsis && rank == explicit) {
        Ok(rank - explicit)
    } else {
        let qualifier = if has_ellipsis { "at least " } else { "" };
        Err(Error::new(
            ErrorKind::RankMismatch,
            pattern,
            format!("input has {rank} dimensions; the pattern requires {qualifier}{explicit}"),
        ))
    }
}

pub(crate) fn parse_expression(
    expression: &str,
    pattern: &str,
    allow_skip: bool,
    allow_grouped_ellipsis: bool,
) -> Result<Expression> {
    let syntax = |message| Error::new(ErrorKind::InvalidPattern, pattern, message);
    let spaced = expression.replace('(', " ( ").replace(')', " ) ");
    let mut groups = Vec::new();
    let mut grouped = Vec::new();
    let mut current: Option<Vec<Axis>> = None;
    let mut names = BTreeSet::new();
    let mut has_ellipsis = false;

    for token in spaced.split_whitespace() {
        match token {
            "(" => {
                if current.is_some() {
                    return Err(syntax("nested parentheses are not supported"));
                }
                current = Some(Vec::new());
                continue;
            }
            ")" => {
                let group = current.take().ok_or_else(|| syntax("unmatched `)`"))?;
                groups.push(group);
                grouped.push(true);
                continue;
            }
            _ => {}
        }
        let axis = if token == "..." {
            if has_ellipsis {
                return Err(syntax("each side may contain only one ellipsis (`...`)"));
            }
            if current.is_some() && !allow_grouped_ellipsis {
                return Err(syntax("an input ellipsis cannot be inside parentheses"));
            }
            has_ellipsis = true;
            Axis::Ellipsis
        } else if token == "_" && allow_skip {
            Axis::Skip
        } else if token.bytes().all(|c| c.is_ascii_digit()) {
            let length = token.parse::<usize>().map_err(|_| {
                Error::new(
                    ErrorKind::Overflow,
                    pattern,
                    "a literal axis length exceeds usize",
                )
            })?;
            if length == 0 {
                return Err(syntax("literal axis lengths must be positive"));
            }
            Axis::Literal(length)
        } else if valid_name(token) {
            if !names.insert(token.to_owned()) {
                return Err(Error::new(
                    ErrorKind::InvalidPattern,
                    pattern,
                    format!("axis `{token}` appears more than once on the same side"),
                ));
            }
            Axis::Named(token.to_owned())
        } else {
            return Err(Error::new(
                ErrorKind::InvalidPattern,
                pattern,
                format!(
                    "invalid token `{token}`; use axis names, positive integers, `...`, and parentheses"
                ),
            ));
        };

        if let Some(group) = &mut current {
            group.push(axis);
        } else {
            groups.push(vec![axis]);
            grouped.push(false);
        }
    }
    if current.is_some() {
        return Err(syntax("unclosed `(`"));
    }
    Ok(Expression {
        groups,
        grouped,
        has_ellipsis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_queries_skip_assert_and_allow_empty_dimensions() {
        assert_eq!(parse_shape(&[2, 0, 7], "b _ c").unwrap()["c"], 7);
        assert_eq!(parse_shape(&[2, 3, 5, 7], "b ... c").unwrap()["c"], 7);
        assert_eq!(parse_shape(&[2, 7], "b ... c").unwrap().len(), 2);
        assert!(parse_shape(&[], "...").unwrap().is_empty());
        assert!(parse_shape(&[], "").unwrap().is_empty());
        assert!(parse_shape(&[1, 3], "1 3").unwrap().is_empty());
        assert!(parse_shape(&[2, 3], "_ _").unwrap().is_empty());
    }

    #[test]
    fn invalid_shape_queries_return_specific_errors() {
        for (shape, pattern, kind) in [
            (vec![2, 3], "b b", ErrorKind::InvalidPattern),
            (vec![2, 3], "b", ErrorKind::RankMismatch),
            (vec![2], "b ... c", ErrorKind::RankMismatch),
            (vec![2, 3], "1 c", ErrorKind::InvalidShape),
            (vec![2, 3], "... ...", ErrorKind::InvalidPattern),
            (vec![2, 3], "(b c)", ErrorKind::Unsupported),
            (vec![2], "0", ErrorKind::InvalidPattern),
            (vec![2], "b -> c", ErrorKind::InvalidPattern),
        ] {
            let error = parse_shape(&shape, pattern).unwrap_err();
            assert_eq!(error.kind(), kind, "{pattern}");
            assert_eq!(error.pattern(), pattern);
        }
    }
}
