//! Georeferenced 2-D grid used for DTMs and CHMs.

use std::io::Write;
use std::path::Path;

use crate::error::{Error, Result};

/// A north-up grid; `data[row * ncols + col]`, row 0 is the *southern* edge.
#[derive(Debug, Clone, PartialEq)]
pub struct Raster {
    pub data: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
    pub xmin: f64,
    pub ymin: f64,
    pub resolution: f64,
}

/// How to reduce many point values into one cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reducer {
    Min,
    Max,
    Mean,
    Count,
}

impl Raster {
    pub fn filled(nrows: usize, ncols: usize, xmin: f64, ymin: f64, resolution: f64, value: f64) -> Self {
        Raster { data: vec![value; nrows * ncols], nrows, ncols, xmin, ymin, resolution }
    }

    pub fn xmax(&self) -> f64 {
        self.xmin + self.ncols as f64 * self.resolution
    }

    pub fn ymax(&self) -> f64 {
        self.ymin + self.nrows as f64 * self.resolution
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.ncols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, v: f64) {
        self.data[row * self.ncols + col] = v;
    }

    /// `(row, col)` of the cell containing `(x, y)`, possibly out of range.
    #[inline]
    pub fn cell_index(&self, x: f64, y: f64) -> (i64, i64) {
        (((y - self.ymin) / self.resolution).floor() as i64, ((x - self.xmin) / self.resolution).floor() as i64)
    }

    #[inline]
    pub fn in_bounds(&self, row: i64, col: i64) -> bool {
        row >= 0 && col >= 0 && (row as usize) < self.nrows && (col as usize) < self.ncols
    }

    /// Cell-centre coordinates.
    pub fn cell_center(&self, row: usize, col: usize) -> (f64, f64) {
        (self.xmin + (col as f64 + 0.5) * self.resolution, self.ymin + (row as f64 + 0.5) * self.resolution)
    }

    /// Bilinear sample with edge clamping. NaN cells propagate.
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        if self.nrows == 0 || self.ncols == 0 {
            return f64::NAN;
        }
        let fc = ((x - self.xmin) / self.resolution - 0.5).clamp(0.0, (self.ncols - 1) as f64);
        let fr = ((y - self.ymin) / self.resolution - 0.5).clamp(0.0, (self.nrows - 1) as f64);
        let c0 = fc.floor() as usize;
        let r0 = fr.floor() as usize;
        let c1 = (c0 + 1).min(self.ncols - 1);
        let r1 = (r0 + 1).min(self.nrows - 1);
        let tx = fc - c0 as f64;
        let ty = fr - r0 as f64;
        self.get(r0, c0) * (1.0 - tx) * (1.0 - ty)
            + self.get(r0, c1) * tx * (1.0 - ty)
            + self.get(r1, c0) * (1.0 - tx) * ty
            + self.get(r1, c1) * tx * ty
    }

    /// Sample many coordinates.
    pub fn sample_many(&self, xy: impl Iterator<Item = (f64, f64)>) -> Vec<f64> {
        xy.map(|(x, y)| self.sample(x, y)).collect()
    }

    /// Rasterise point values. `bounds` is `(xmin, ymin, xmax, ymax)`.
    pub fn from_points(
        xy: impl Iterator<Item = (f64, f64)> + Clone,
        values: impl Iterator<Item = f64>,
        resolution: f64,
        reducer: Reducer,
        bounds: Option<(f64, f64, f64, f64)>,
        fill: f64,
    ) -> Result<Raster> {
        let (xmin, ymin, xmax, ymax) = match bounds {
            Some(b) => b,
            None => {
                let mut lo = (f64::INFINITY, f64::INFINITY);
                let mut hi = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                for (x, y) in xy.clone() {
                    lo.0 = lo.0.min(x);
                    lo.1 = lo.1.min(y);
                    hi.0 = hi.0.max(x);
                    hi.1 = hi.1.max(y);
                }
                if !lo.0.is_finite() {
                    return Err(Error::invalid("cannot rasterise an empty point set"));
                }
                ((lo.0 / resolution).floor() * resolution, (lo.1 / resolution).floor() * resolution, hi.0, hi.1)
            }
        };
        let ncols = (((xmax - xmin) / resolution).floor() as usize + 1).max(1);
        let nrows = (((ymax - ymin) / resolution).floor() as usize + 1).max(1);
        let size = nrows * ncols;
        let init = match reducer {
            Reducer::Min => f64::INFINITY,
            Reducer::Max => f64::NEG_INFINITY,
            _ => 0.0,
        };
        let mut acc = vec![init; size];
        let mut counts = vec![0u32; size];
        let mut r = Raster::filled(nrows, ncols, xmin, ymin, resolution, fill);
        for ((x, y), v) in xy.zip(values) {
            let (row, col) = r.cell_index(x, y);
            if !r.in_bounds(row, col) {
                continue;
            }
            let i = row as usize * ncols + col as usize;
            counts[i] += 1;
            match reducer {
                Reducer::Min => acc[i] = acc[i].min(v),
                Reducer::Max => acc[i] = acc[i].max(v),
                Reducer::Mean => acc[i] += v,
                Reducer::Count => {}
            }
        }
        for i in 0..size {
            r.data[i] = match reducer {
                Reducer::Count => counts[i] as f64,
                _ if counts[i] == 0 => fill,
                Reducer::Mean => acc[i] / counts[i] as f64,
                _ => acc[i],
            };
        }
        Ok(r)
    }

    /// Fill NaN cells with the value of the nearest non-NaN cell (brute-force BFS by rings).
    pub fn fill_nearest(&mut self) {
        let nan: Vec<(usize, usize)> = (0..self.nrows)
            .flat_map(|r| (0..self.ncols).map(move |c| (r, c)))
            .filter(|&(r, c)| self.get(r, c).is_nan())
            .collect();
        if nan.is_empty() || nan.len() == self.data.len() {
            return;
        }
        // Multi-source BFS from valid cells (4-connectivity), a good enough
        // approximation of nearest-neighbour fill for terrain gaps.
        let mut out = self.data.clone();
        let mut queue: std::collections::VecDeque<(usize, usize)> = (0..self.nrows)
            .flat_map(|r| (0..self.ncols).map(move |c| (r, c)))
            .filter(|&(r, c)| !self.get(r, c).is_nan())
            .collect();
        while let Some((r, c)) = queue.pop_front() {
            let v = out[r * self.ncols + c];
            let nb = [
                (r.wrapping_sub(1), c),
                (r + 1, c),
                (r, c.wrapping_sub(1)),
                (r, c + 1),
            ];
            for (nr, nc) in nb {
                if nr < self.nrows && nc < self.ncols && out[nr * self.ncols + nc].is_nan() {
                    out[nr * self.ncols + nc] = v;
                    queue.push_back((nr, nc));
                }
            }
        }
        self.data = out;
    }

    /// Grey morphological opening (min then max) with a square window of `half` cells.
    pub fn grey_opening(&self, half: usize) -> Raster {
        let eroded = self.window_reduce(half, f64::min, f64::INFINITY);
        eroded.window_reduce(half, f64::max, f64::NEG_INFINITY)
    }

    /// Mean filter with a square window of `half` cells (NaN-aware).
    pub fn smooth(&self, half: usize) -> Raster {
        let mut out = self.clone();
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                let mut s = 0.0;
                let mut n = 0.0;
                for rr in r.saturating_sub(half)..=(r + half).min(self.nrows - 1) {
                    for cc in c.saturating_sub(half)..=(c + half).min(self.ncols - 1) {
                        let v = self.get(rr, cc);
                        if v.is_finite() {
                            s += v;
                            n += 1.0;
                        }
                    }
                }
                out.set(r, c, if n > 0.0 { s / n } else { f64::NAN });
            }
        }
        out
    }

    fn window_reduce(&self, half: usize, f: fn(f64, f64) -> f64, init: f64) -> Raster {
        let mut out = self.clone();
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                let mut v = init;
                for rr in r.saturating_sub(half)..=(r + half).min(self.nrows - 1) {
                    for cc in c.saturating_sub(half)..=(c + half).min(self.ncols - 1) {
                        let x = self.get(rr, cc);
                        if x.is_finite() {
                            v = f(v, x);
                        }
                    }
                }
                out.set(r, c, if v.is_finite() { v } else { f64::NAN });
            }
        }
        out
    }

    /// Write an ESRI ASCII grid.
    pub fn write_ascii_grid(&self, path: impl AsRef<Path>, nodata: f64) -> Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "ncols {}", self.ncols)?;
        writeln!(f, "nrows {}", self.nrows)?;
        writeln!(f, "xllcorner {}", self.xmin)?;
        writeln!(f, "yllcorner {}", self.ymin)?;
        writeln!(f, "cellsize {}", self.resolution)?;
        writeln!(f, "NODATA_value {nodata}")?;
        for r in (0..self.nrows).rev() {
            let row: Vec<String> = (0..self.ncols)
                .map(|c| {
                    let v = self.get(r, c);
                    if v.is_nan() {
                        format!("{nodata}")
                    } else {
                        format!("{v:.4}")
                    }
                })
                .collect();
            writeln!(f, "{}", row.join(" "))?;
        }
        Ok(())
    }

    /// Read an ESRI ASCII grid.
    pub fn read_ascii_grid(path: impl AsRef<Path>) -> Result<Raster> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let mut lines = text.lines();
        let mut ncols = 0usize;
        let mut nrows = 0usize;
        let mut xll = 0.0;
        let mut yll = 0.0;
        let mut cell = 1.0;
        let mut nodata = -9999.0;
        let mut centre = false;
        let mut data_lines = Vec::new();
        for line in lines.by_ref() {
            let mut it = line.split_whitespace();
            let Some(key) = it.next() else { continue };
            let val = it.next().unwrap_or("");
            match key.to_ascii_lowercase().as_str() {
                "ncols" => ncols = val.parse().map_err(|_| Error::file(path.as_ref(), "bad ncols"))?,
                "nrows" => nrows = val.parse().map_err(|_| Error::file(path.as_ref(), "bad nrows"))?,
                "xllcorner" => xll = val.parse().map_err(|_| Error::file(path.as_ref(), "bad xllcorner"))?,
                "yllcorner" => yll = val.parse().map_err(|_| Error::file(path.as_ref(), "bad yllcorner"))?,
                "xllcenter" => {
                    xll = val.parse().map_err(|_| Error::file(path.as_ref(), "bad xllcenter"))?;
                    centre = true;
                }
                "yllcenter" => {
                    yll = val.parse().map_err(|_| Error::file(path.as_ref(), "bad yllcenter"))?;
                    centre = true;
                }
                "cellsize" => cell = val.parse().map_err(|_| Error::file(path.as_ref(), "bad cellsize"))?,
                "nodata_value" => nodata = val.parse().unwrap_or(-9999.0),
                _ => {
                    data_lines.push(line);
                    break;
                }
            }
        }
        data_lines.extend(lines);
        if centre {
            xll -= cell / 2.0;
            yll -= cell / 2.0;
        }
        let mut r = Raster::filled(nrows, ncols, xll, yll, cell, f64::NAN);
        let mut vals = data_lines.iter().flat_map(|l| l.split_whitespace());
        for row in (0..nrows).rev() {
            for col in 0..ncols {
                let v: f64 = vals
                    .next()
                    .ok_or_else(|| Error::file(path.as_ref(), "truncated grid"))?
                    .parse()
                    .map_err(|_| Error::file(path.as_ref(), "bad grid value"))?;
                r.set(row, col, if v == nodata { f64::NAN } else { v });
            }
        }
        Ok(r)
    }
}
