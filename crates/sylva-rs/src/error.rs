// Sylva: LiDAR processing for forest ecology and remote sensing research.
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
    /// A bad argument or inconsistent input (lengths that disagree, an unknown mode,
    /// too few points); the message is the whole text.
    Invalid(String),
    #[error("unsupported format: {0:?}")]
    /// A file extension or format name no reader or writer handles, e.g. a point
    /// file that is not LAS/LAZ or a polygon file that is not .shp/.geojson.
    UnsupportedFormat(String),
    #[error("{path}: {msg}")]
    /// Something wrong with one file, named by `path`, with `msg` saying what
    /// (cannot be read, too few values, not a project directory).
    File {
        /// The file (or directory) the error is about.
        path: PathBuf,
        /// What was wrong with it, in plain words.
        msg: String,
    },
    #[error(transparent)]
    /// An operating-system I/O failure such as a missing file or denied permission,
    /// converted from `std::io::Error`.
    Io(#[from] std::io::Error),
    #[error("LAS: {0}")]
    /// A LAS/LAZ read or write failure reported by the `las` crate (bad header,
    /// unsupported point format, truncated file).
    Las(#[from] las::Error),
    #[error("RIEGL RiVLib: {0}")]
    /// A RIEGL RiVLib failure: the RXP library could not be loaded or reported an
    /// error while reading a scan, or an MTA model file is malformed.
    Riegl(String),
    #[error("RIEGL RDB: {0}")]
    /// A RIEGL RDB failure: librdb could not be loaded, or an .rdbx file could not
    /// be read or lacks a needed point attribute.
    Rdb(String),
}

/// The crate's own `Result`: `Result<T>` here means `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// An [`Error::Invalid`] carrying `msg`: the usual way to reject a bad argument.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    /// An [`Error::File`] naming `path` and saying what was wrong with it.
    pub fn file(path: impl Into<PathBuf>, msg: impl Into<String>) -> Self {
        Error::File { path: path.into(), msg: msg.into() }
    }
}
