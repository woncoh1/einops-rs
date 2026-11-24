use std::fmt;

/// The result of parsing or applying an Einops pattern.
pub type Result<T> = std::result::Result<T, Error>;

/// A broad error category. Inspect the message for axis and dimension details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Malformed notation, invalid names, or duplicate axes.
    InvalidPattern,
    /// A dimension cannot satisfy the pattern.
    InvalidShape,
    /// Missing, duplicate, unused, or incompatible axis lengths.
    InvalidAxes,
    /// The pattern rank does not match the input or output rank.
    RankMismatch,
    /// A dimension product does not fit in `usize`.
    Overflow,
    /// A feature is deliberately unsupported by this release.
    Unsupported,
}

/// A pattern error with deterministic, human-readable context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    pattern: String,
    message: String,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, pattern: &str, message: impl Into<String>) -> Self {
        Self {
            kind,
            pattern: pattern.to_owned(),
            message: message.into(),
        }
    }

    /// Return the error category.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Return the pattern that caused the error.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Return the explanation without the pattern prefix.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pattern `{}`: {}", self.pattern, self.message)
    }
}

impl std::error::Error for Error {}
