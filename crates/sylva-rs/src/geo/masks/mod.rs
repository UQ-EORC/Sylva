// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Boolean masks over the points of a cloud.
//!
//! Every function here returns one `bool` per point, `true` where the point
//! is kept, so masks from different sources combine element-wise and index a
//! cloud directly. The sources are:
//!
//! * [`polygon`]: points whose x, y fall inside 2-D polygons with holes,
//!   read by [`vector_io`] from shapefiles and GeoJSON;
//! * [`raster`]: points whose raster cell satisfies a range or a value set;
//! * [`expr`]: a small expression language over the cloud's attributes;
//! * [`near`]: points within a distance of another cloud.
//!
//! Polygons and rasters are taken in the cloud's frame; nothing here
//! reprojects.

pub mod expr;
pub mod near;
pub mod polygon;
pub mod raster;
pub mod vector_io;

pub use expr::Expr;
pub use near::near_mask;
pub use polygon::{MultiPolygon, Polygon, PolygonIndex};
pub use raster::{raster_mask, RasterTest};
pub use vector_io::{read_polygons, Feature, Layer, Property};

/// Number of points handled by one parallel task. Masks are computed point by
/// point, so the result never depends on how the work is split.
pub(crate) const CHUNK: usize = 1 << 16;
