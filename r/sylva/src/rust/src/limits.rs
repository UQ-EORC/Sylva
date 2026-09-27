// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the memory budget (sylva_rs::limits). Sizes cross as doubles.

use extendr_api::prelude::*;
use sylva_rs::limits;

use crate::convert::{err, fail, Result};

fn opt_bytes(v: Option<u64>) -> Robj {
    v.map_or_else(|| ().into(), |b| (b as f64).into())
}

/// @noRd
#[extendr]
fn core_memory_available() -> Robj {
    opt_bytes(limits::available())
}

/// @noRd
#[extendr]
fn core_memory_budget() -> Robj {
    opt_bytes(limits::budget())
}

/// @noRd
#[extendr]
fn core_set_memory_budget(bytes: f64) -> Result<()> {
    if bytes.is_nan() || bytes < 0.0 || bytes.is_infinite() {
        return fail("the budget must be a non-negative number of bytes");
    }
    limits::set_budget(bytes as u64);
    Ok(())
}

/// @noRd
#[extendr]
fn core_memory_check(cells: f64, per_cell: f64, what: &str, hint: &str) -> Result<()> {
    if [cells, per_cell].iter().any(|v| v.is_nan() || *v < 0.0) {
        return fail("cells and per_cell must be non-negative");
    }
    limits::check_cells(cells as u128, per_cell as u64, what, hint).map_err(err)
}

/// @noRd
#[extendr]
fn core_memory_human(bytes: f64) -> String {
    limits::human_f64(bytes)
}

extendr_module! {
    mod limits;
    fn core_memory_available;
    fn core_memory_budget;
    fn core_set_memory_budget;
    fn core_memory_check;
    fn core_memory_human;
}
