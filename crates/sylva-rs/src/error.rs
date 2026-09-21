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
