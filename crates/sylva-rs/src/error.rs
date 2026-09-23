// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
use std::path::PathBuf;

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

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    pub fn file(path: impl Into<PathBuf>, msg: impl Into<String>) -> Self {
        Error::File { path: path.into(), msg: msg.into() }
    }
}
