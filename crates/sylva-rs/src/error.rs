// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! What can go wrong, and the message it carries.
//!
//! Every fallible function in the core returns [`Result<T>`], which is "a `T`
//! or an [`Error`]". Callers either handle it or pass it on with `?`; nothing
//! is thrown, and a failure cannot be ignored by accident. The bindings turn
//! these into Python exceptions, so the text below is what the person running
//! the command finally reads — which is why it names the file and says what
//! would make it work, rather than describing the internals.

use std::path::PathBuf;

/// A failure, in one of the handful of kinds the core distinguishes.
///
/// The `#[error(...)]` lines are the messages; `#[from]` means an error of
/// that foreign type converts into this one on its own, which is what lets a
/// function working with files write `?` after a standard library call. A
/// variant added here is handled everywhere at once, because the compiler
/// refuses any `match` that forgets it.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("unsupported format: {0:?}")]
    UnsupportedFormat(String),
    #[error("{path}: {msg}")]
    File { path: PathBuf, msg: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("LAS: {0}")]
    Las(#[from] las::Error),
    #[error("RIEGL RiVLib: {0}")]
    Riegl(String),
}

/// The crate's own `Result`: `Result<T>` here means `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    pub fn file(path: impl Into<PathBuf>, msg: impl Into<String>) -> Self {
        Error::File { path: path.into(), msg: msg.into() }
    }
}
