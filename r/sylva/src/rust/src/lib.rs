// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! R bindings for sylva-rs.
//!
//! Point clouds cross the boundary as `list(xyz = <n x 3 double matrix>,
//! attrs = <named list of vectors>)`; see `R/` for the friendly wrappers.
//! Attribute types map to R as: floats and 64-bit or unsigned 32-bit
//! integers to double, smaller integers to integer, booleans to logical.

use extendr_api::prelude::*;

mod canopy;
mod quality;
mod coreg;
mod shots;
mod trees;
mod riscan;
mod convert;
mod filters;
mod ground;
mod io;
mod limits;
mod raster;
mod registration;

extendr_module! {
    mod sylva;
    use canopy;
    use quality;
    use coreg;
    use shots;
    use trees;
    use filters;
    use ground;
    use io;
    use limits;
    use raster;
    use registration;
    use riscan;
}
