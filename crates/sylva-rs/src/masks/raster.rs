// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Masks from the raster cell under each point.
//!
//! A point belongs to the cell `col = floor((x - xmin) / resolution)`,
//! `row = floor((y - ymin) / resolution)`, so cells are closed on their
//! south and west edges and open on the others; a point on the grid's
//! northern or eastern edge is outside. Points outside the grid, and points
//! over a NaN cell, are never kept.

use rayon::prelude::*;

use super::CHUNK;
use crate::error::{Error, Result};
use crate::{Point, Raster};

/// What a cell value must satisfy.
#[derive(Debug, Clone, PartialEq)]
pub enum RasterTest {
    /// Any value that is not NaN.
    Valid,
    /// `min <= value <= max`, each bound optional.
    Range { min: Option<f64>, max: Option<f64> },
    /// The value equals one of these.
    Values(Vec<f64>),
}

/// Value of the cell under `(x, y)`, or `None` outside the grid.
#[inline]
fn cell_value(r: &Raster, x: f64, y: f64) -> Option<f64> {
    let c = ((x - r.xmin) / r.resolution).floor();
    let w = ((y - r.ymin) / r.resolution).floor();
    // NaN coordinates fail both comparisons.
    if c >= 0.0 && c < r.ncols as f64 && w >= 0.0 && w < r.nrows as f64 {
        Some(r.data[w as usize * r.ncols + c as usize])
    } else {
        None
    }
}

/// For each point, whether the cell under its x, y exists, is not NaN and
/// passes `test`.
///
/// # Errors
/// A resolution that is not finite and positive, a data length that does not
/// match the shape, a NaN bound, or `min > max`.
pub fn raster_mask(points: &[Point], raster: &Raster, test: &RasterTest) -> Result<Vec<bool>> {
    if !(raster.resolution.is_finite() && raster.resolution > 0.0) {
        return Err(Error::invalid(format!("raster resolution must be positive, got {}", raster.resolution)));
    }
    if raster.data.len() != raster.nrows * raster.ncols {
        return Err(Error::invalid(format!("raster data has {} values for a {} x {} grid", raster.data.len(), raster.nrows, raster.ncols)));
    }
    if !(raster.xmin.is_finite() && raster.ymin.is_finite()) {
        return Err(Error::invalid("raster origin must be finite"));
    }
    if let RasterTest::Range { min, max } = test {
        if min.is_some_and(f64::is_nan) || max.is_some_and(f64::is_nan) {
            return Err(Error::invalid("min and max must not be NaN"));
        }
        if let (Some(a), Some(b)) = (min, max) {
            if a > b {
                return Err(Error::invalid(format!("min ({a}) is greater than max ({b})")));
            }
        }
    }
    let keep = |v: f64| -> bool {
        !v.is_nan()
            && match test {
                RasterTest::Valid => true,
                RasterTest::Range { min, max } => min.is_none_or(|m| v >= m) && max.is_none_or(|m| v <= m),
                RasterTest::Values(vals) => vals.contains(&v),
            }
    };
    let mut out = vec![false; points.len()];
    out.par_chunks_mut(CHUNK).zip(points.par_chunks(CHUNK)).for_each(|(o, p)| {
        for (o, p) in o.iter_mut().zip(p) {
            *o = cell_value(raster, p[0], p[1]).is_some_and(keep);
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Raster {
        // 2 rows x 3 cols, 1 m cells from (10, 20); row 0 is the south.
        Raster { data: vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0], nrows: 2, ncols: 3, xmin: 10.0, ymin: 20.0, resolution: 1.0 }
    }

    #[test]
    fn cells_ranges_values_and_edges() {
        let r = grid();
        let pts = [[10.5, 20.5, 0.0], [12.5, 20.5, 0.0], [11.5, 21.5, 0.0], [9.9, 20.5, 0.0], [13.0, 20.5, 0.0], [10.0, 20.0, 0.0], [f64::NAN, 20.5, 0.0], [12.999, 21.999, 0.0]];
        assert_eq!(raster_mask(&pts, &r, &RasterTest::Valid).unwrap(), vec![true, false, true, false, false, true, false, true]);
        assert_eq!(raster_mask(&pts, &r, &RasterTest::Range { min: Some(2.0), max: Some(5.0) }).unwrap(), vec![false, false, true, false, false, false, false, false]);
        assert_eq!(raster_mask(&pts, &r, &RasterTest::Range { min: None, max: Some(1.0) }).unwrap(), vec![true, false, false, false, false, true, false, false]);
        assert_eq!(raster_mask(&pts, &r, &RasterTest::Values(vec![6.0, 1.0])).unwrap(), vec![true, false, false, false, false, true, false, true]);
    }

    #[test]
    fn bad_arguments() {
        let r = grid();
        assert!(raster_mask(&[], &r, &RasterTest::Range { min: Some(3.0), max: Some(1.0) }).is_err());
        assert!(raster_mask(&[], &r, &RasterTest::Range { min: Some(f64::NAN), max: None }).is_err());
        let mut bad = grid();
        bad.resolution = 0.0;
        assert!(raster_mask(&[], &bad, &RasterTest::Valid).is_err());
    }
}
