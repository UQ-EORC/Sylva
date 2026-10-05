// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::util::limits beyond the budget itself.

use pyo3::prelude::*;
use sylva_rs::util::limits;

#[pyfunction]
fn memory_human(bytes: f64) -> String {
    limits::human_f64(bytes)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(memory_human, m)?)?;
    Ok(())
}
