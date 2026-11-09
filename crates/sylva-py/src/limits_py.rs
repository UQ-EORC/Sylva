// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::util::limits beyond the budget itself.
//!
//! Backs sylva.util.limits: a byte count comes in as a float and a
//! human-readable size string goes out. The memory budget itself is enforced
//! in Python; only the formatting is shared so both sides print the same.

use pyo3::prelude::*;
use sylva_rs::util::limits;

#[pyfunction]
fn memory_human(bytes: f64) -> String {
    limits::human_f64(bytes)
}

/// Adds this file's functions to the `_core` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(memory_human, m)?)?;
    Ok(())
}
