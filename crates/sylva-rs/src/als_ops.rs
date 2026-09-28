// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Whole-area operations on an ALS catalogue, built on the chunk engine of
//! [`crate::als`]: ground classification, DTM, CHM, height normalisation,
//! noise filtering, retiling, decimation, and writing a cloud as tiles.
//!
//! Each runs the single-cloud function of [`crate::ground`],
//! [`crate::filters`] or [`crate::interpolate`] on a chunk with its buffer
//! and keeps what falls in the chunk's core, so that away from the outer
//! edge of the survey a result does not show where the tiles meet. Rasters
//! are computed on one grid for the whole catalogue ([`catalog_grid`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use las::Vlr;
use rayon::prelude::*;

use crate::als::{buffer_attr, catalog_grid, chunk_bounds, mosaic, output_path, plan, run, workers_for, write_like, Catalog, Chunk, ChunkData, Layout, BYTES_PER_POINT};
use crate::error::{Error, Result};
use crate::filters;
use crate::ground::{self, CsfParams, PmfParams, GROUND_CLASS};
use crate::interpolate::{self, GridParams};
use crate::io::las::{write_las_with_vlrs, LasWriteOptions};
use crate::pointcloud::Attr;
use crate::raster::Raster;
use crate::{Point, PointCloud};

/// ASPRS low noise class.
pub const NOISE_CLASS: u8 = 7;
/// ASPRS high noise class.
pub const HIGH_NOISE_CLASS: u8 = 18;

/// How to divide the work.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunOptions {
    pub layout: Layout,
    /// Buffer around each chunk (m).
    pub buffer: f64,
    /// Chunks processed at once; 0 for one per core (always capped by the
    /// memory budget).
    pub workers: usize,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions { layout: Layout::Tiles, buffer: 20.0, workers: 0 }
    }
}

fn chunks_and_workers(cat: &Catalog, opts: &RunOptions) -> Result<(Vec<Chunk>, usize)> {
    let chunks = plan(cat, opts.layout, opts.buffer)?;
    let w = workers_for(&chunks, opts.workers, BYTES_PER_POINT)?;
    Ok((chunks, w))
}

/// The extension of a chunk's output: `format` if given ("las" or "laz"),
/// else that of the chunk's own tile (or its first file).
fn out_ext(cat: &Catalog, chunk: &Chunk, format: Option<&str>) -> Result<String> {
    if let Some(f) = format {
        let f = f.trim_start_matches('.').to_ascii_lowercase();
        if f != "las" && f != "laz" {
            return Err(Error::invalid(format!("format must be 'las' or 'laz', got {f:?}")));
        }
        return Ok(f);
    }
    let t = &cat.tiles[like_tile(chunk)];
    Ok(t.path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).filter(|e| e == "las" || e == "laz").unwrap_or_else(|| "laz".to_string()))
}

/// The tile whose point format, scale and CRS a chunk's output copies.
fn like_tile(chunk: &Chunk) -> usize {
    chunk.own.unwrap_or(chunk.files[0])
}

/// Write the per-chunk outputs of `f` (which returns the cloud to write)
/// under `out_dir`, returning the paths written in chunk order.
fn write_chunks(cat: &Catalog, out_dir: &Path, format: Option<&str>, opts: &RunOptions, label: &str, f: impl Fn(&Chunk, ChunkData) -> Result<PointCloud> + Sync) -> Result<Vec<PathBuf>> {
    let (chunks, w) = chunks_and_workers(cat, opts)?;
    std::fs::create_dir_all(out_dir)?;
    // Check every output path before anything is written.
    for c in &chunks {
        output_path(cat, out_dir, &c.name, &out_ext(cat, c, format)?)?;
    }
    let written = run(cat, &chunks, w, label, |chunk, data| {
        let cloud = f(chunk, data)?;
        if cloud.is_empty() {
            return Ok(None);
        }
        let path = output_path(cat, out_dir, &chunk.name, &out_ext(cat, chunk, format)?)?;
        write_like(&cloud, &path, &cat.tiles[like_tile(chunk)])?;
        Ok(Some(path))
    })?;
    Ok(written.into_iter().flatten().flatten().collect())
}

const NO_GROUND: &str = "no chunk has 3 ground points (classification 2); classify ground first (als.classify_ground)";

fn classification(cloud: &PointCloud) -> Result<Vec<u8>> {
    let c = cloud.attr("classification").ok_or_else(|| Error::invalid("the tiles have no 'classification'; classify ground first (als.classify_ground)"))?;
    Ok((0..cloud.len()).map(|i| c.get_f64(i) as u8).collect())
}

fn ground_points(cloud: &PointCloud) -> Result<Vec<Point>> {
    let cls = classification(cloud)?;
    Ok((0..cloud.len()).filter(|&i| cls[i] == GROUND_CLASS).map(|i| cloud.xyz[i]).collect())
}

// ------------------------------------------------------------------ ground

/// Ground filter for [`classify_ground`].
#[derive(Debug, Clone)]
pub enum GroundMethod {
    Csf(CsfParams),
    Pmf(PmfParams),
}

/// Classify ground in every tile and write the classified tiles to `out_dir`.
///
/// The filter runs on each chunk with its buffer; the core points are
/// written with `classification` 2 (ground) or 1 (other), as
/// [`ground::classification_from_mask`] sets it. Points already classified
/// as noise (7 or 18) are left out of the filter and keep their class; with
/// `last_returns`, only last returns (`return_number == number_of_returns`)
/// can be ground. On a single tile with neither, the classes are those of
/// the single-cloud filter.
pub fn classify_ground(cat: &Catalog, out_dir: &Path, method: &GroundMethod, last_returns: bool, format: Option<&str>, opts: &RunOptions) -> Result<Vec<PathBuf>> {
    write_chunks(cat, out_dir, format, opts, "classifying ground", |_, data| {
        let cloud = &data.cloud;
        let old = cloud.attr("classification").map(|c| (0..cloud.len()).map(|i| c.get_f64(i) as u8).collect::<Vec<u8>>());
        let noise = |i: usize| old.as_ref().is_some_and(|c| c[i] == NOISE_CLASS || c[i] == HIGH_NOISE_CLASS);
        let last = |i: usize| -> bool {
            match (cloud.attr("return_number"), cloud.attr("number_of_returns")) {
                (Some(r), Some(n)) => r.get_f64(i) >= n.get_f64(i),
                _ => true,
            }
        };
        let used: Vec<usize> = (0..cloud.len()).filter(|&i| !noise(i) && (!last_returns || last(i))).collect();
        let xyz: Vec<Point> = if used.len() == cloud.len() { cloud.xyz.clone() } else { used.iter().map(|&i| cloud.xyz[i]).collect() };
        let mask = match method {
            GroundMethod::Csf(p) => ground::csf_ground_mask(&xyz, p),
            GroundMethod::Pmf(p) => ground::pmf_ground_mask(&xyz, p)?,
        };
        let mut is_ground = vec![false; cloud.len()];
        for (k, &i) in used.iter().enumerate() {
            is_ground[i] = mask[k];
        }
        let Attr::U8(mut cls) = ground::classification_from_mask(&is_ground) else { unreachable!() };
        for (i, c) in cls.iter_mut().enumerate() {
            if noise(i) {
                *c = old.as_ref().expect("noise implies a classification")[i];
            }
        }
        let idx = data.core_indices();
        let mut out = cloud.take(&idx);
        out.attrs.insert("classification".into(), Attr::U8(idx.iter().map(|&i| cls[i]).collect()));
        Ok(out)
    })
}

// ------------------------------------------------------------------ rasters

/// How [`dtm`] makes a surface from the ground points.
#[derive(Debug, Clone, Copy)]
pub enum DtmMethod {
    /// [`ground::make_dtm`]: lowest ground point per cell, gaps filled.
    Lowest,
    /// [`interpolate::grid`] at cell centres (TIN, natural neighbour or IDW),
    /// cells outside the triangulation filled from the nearest cell.
    Grid(GridParams),
}

/// A DTM of one chunk on the catalogue grid, None with fewer than 3 ground points.
fn chunk_dtm(grid: &Raster, chunk: &Chunk, ground: &[Point], method: &DtmMethod) -> Result<Option<Raster>> {
    if ground.len() < 3 {
        return Ok(None);
    }
    let b = chunk_bounds(grid, &chunk.outer);
    Ok(Some(match method {
        DtmMethod::Lowest => ground::make_dtm(ground, grid.resolution, Some(b))?,
        DtmMethod::Grid(p) => {
            let z: Vec<f64> = ground.iter().map(|q| q[2]).collect();
            let mut r = interpolate::grid(ground, &z, grid.resolution, Some(b), p)?;
            if r.data.iter().any(|v| v.is_nan()) {
                r.fill_nearest();
            }
            r
        }
    }))
}

/// Digital terrain model of the whole catalogue from its ground points
/// (`classification == 2`). Chunks with fewer than 3 ground points (with
/// their buffer) leave their cells NaN.
pub fn dtm(cat: &Catalog, resolution: f64, method: &DtmMethod, opts: &RunOptions) -> Result<Raster> {
    let grid = catalog_grid(cat, resolution)?;
    let (chunks, w) = chunks_and_workers(cat, opts)?;
    let parts = run(cat, &chunks, w, "DTM", |chunk, data| Ok(chunk_dtm(&grid, chunk, &ground_points(&data.cloud)?, method)?.map(|r| (r, chunk.core))))?;
    let parts: Vec<(Raster, [f64; 4])> = parts.into_iter().flatten().flatten().collect();
    if parts.is_empty() {
        return Err(Error::invalid(NO_GROUND));
    }
    mosaic(&grid, &parts)
}

/// Where heights above ground come from.
#[derive(Debug, Clone)]
pub enum Heights {
    /// z as it is: the tiles are already normalised (or a surface model is wanted).
    Z,
    /// z minus this DTM, sampled bilinearly (edge values beyond it).
    Dtm(Raster),
    /// z minus a DTM made per chunk from its ground points (with the
    /// buffer) at this resolution, as [`dtm`] with [`DtmMethod::Lowest`].
    Auto { resolution: f64 },
}

impl Heights {
    fn check(&self) -> Result<()> {
        match self {
            Heights::Auto { resolution } if !(resolution.is_finite() && *resolution > 0.0) => Err(Error::invalid(format!("dtm_resolution must be a positive number, got {resolution}"))),
            Heights::Dtm(r) if r.data.is_empty() => Err(Error::invalid("the DTM is empty")),
            _ => Ok(()),
        }
    }
}

/// Height above ground of every point of a chunk; None when it has too
/// little ground for [`Heights::Auto`].
pub fn chunk_heights(cat: &Catalog, chunk: &Chunk, cloud: &PointCloud, heights: &Heights) -> Result<Option<Vec<f64>>> {
pub(crate) fn chunk_heights(cat: &Catalog, chunk: &Chunk, cloud: &PointCloud, heights: &Heights) -> Result<Option<Vec<f64>>> {
    Ok(match heights {
        Heights::Z => Some(cloud.xyz.iter().map(|p| p[2]).collect()),
        Heights::Dtm(r) => Some(ground::heights_above(&cloud.xyz, r)),
        Heights::Auto { resolution } => {
            let grid = catalog_grid(cat, *resolution)?;
            chunk_dtm(&grid, chunk, &ground_points(cloud)?, &DtmMethod::Lowest)?.map(|d| ground::heights_above(&cloud.xyz, &d))
        }
    })
}

/// Canopy height model of the whole catalogue: the highest point above
/// ground per cell, as [`ground::make_chm`] makes it (0 where nothing is
/// above `min_height`). Chunks without enough ground for
/// [`Heights::Auto`] leave their cells NaN.
pub fn chm(cat: &Catalog, resolution: f64, heights: &Heights, min_height: f64, opts: &RunOptions) -> Result<Raster> {
    heights.check()?;
    let grid = catalog_grid(cat, resolution)?;
    let (chunks, w) = chunks_and_workers(cat, opts)?;
    let parts = run(cat, &chunks, w, "CHM", |chunk, data| {
        let Some(h) = chunk_heights(cat, chunk, &data.cloud, heights)? else { return Ok(None) };
        let b = chunk_bounds(&grid, &chunk.outer);
        Ok(Some((ground::make_chm(&data.cloud.xyz, &h, resolution, Some(b), min_height)?, chunk.core)))
    })?;
    let parts: Vec<(Raster, [f64; 4])> = parts.into_iter().flatten().flatten().collect();
    if parts.is_empty() && matches!(heights, Heights::Auto { .. }) {
        return Err(Error::invalid(NO_GROUND));
    }
    mosaic(&grid, &parts)
}

/// Write every tile with its height above ground: a `height` attribute, or
/// with `replace_z` z replaced by the height and the elevation kept in an
/// `elevation` attribute. Chunks without enough ground for
/// [`Heights::Auto`] are an error.
pub fn normalize(cat: &Catalog, out_dir: &Path, heights: &Heights, replace_z: bool, format: Option<&str>, opts: &RunOptions) -> Result<Vec<PathBuf>> {
    heights.check()?;
    if matches!(heights, Heights::Z) {
        return Err(Error::invalid("normalising needs a DTM or ground points to make one"));
    }
    write_chunks(cat, out_dir, format, opts, "normalising heights", |chunk, data| {
        let idx = data.core_indices();
        let core = data.cloud.take(&idx);
        let h = chunk_heights(cat, chunk, &core_with_buffer_ground(&data), heights)?
            .ok_or_else(|| Error::invalid("fewer than 3 ground points (classification 2) within the buffer; classify ground first (als.classify_ground) or give a DTM"))?;
        // chunk_heights saw the buffer's ground but returned heights for the
        // core points only (they come first, see core_with_buffer_ground).
        let h = &h[..core.len()];
        let mut out = core;
        if replace_z {
            out.attrs.insert("elevation".into(), Attr::F64(out.xyz.iter().map(|p| p[2]).collect()));
            for (p, &v) in out.xyz.iter_mut().zip(h) {
                p[2] = v;
            }
        } else {
            out.attrs.insert("height".into(), Attr::F64(h.to_vec()));
        }
        Ok(out)
    })
}

/// The chunk's core points followed by the buffer's ground points: enough
/// for a DTM over the whole buffered box, with heights needed for the core only.
fn core_with_buffer_ground(data: &ChunkData) -> PointCloud {
    let cloud = &data.cloud;
    let cls = cloud.attr("classification");
    let mut idx = data.core_indices();
    idx.extend((0..cloud.len()).filter(|&i| data.buffer[i] && cls.is_some_and(|c| c.get_f64(i) as u8 == GROUND_CLASS)));
    cloud.take(&idx)
}

// ------------------------------------------------------------------ noise

/// Noise test for [`filter_noise`].
#[derive(Debug, Clone, Copy)]
pub enum NoiseMethod {
    /// [`filters::statistical_outlier_mask`]; its threshold is the mean
    /// neighbour distance over the chunk and its buffer.
    Sor { k: usize, std_ratio: f64 },
    /// [`filters::radius_outlier_mask`]; purely local, so tile edges cannot show.
    Ror { radius: f64, min_neighbors: usize },
}

/// Remove noise from every tile (or with `classify`, keep every point and
/// set `classification` 7, ASPRS low noise, on the noise).
pub fn filter_noise(cat: &Catalog, out_dir: &Path, method: &NoiseMethod, classify: bool, format: Option<&str>, opts: &RunOptions) -> Result<Vec<PathBuf>> {
    match *method {
        NoiseMethod::Sor { k, std_ratio } if k == 0 || !(std_ratio.is_finite() && std_ratio > 0.0) => return Err(Error::invalid(format!("k must be positive and std_ratio a positive number, got k = {k}, std_ratio = {std_ratio}"))),
        NoiseMethod::Ror { radius, .. } if !(radius.is_finite() && radius > 0.0) => return Err(Error::invalid(format!("radius must be a positive number, got {radius}"))),
        _ => {}
    }
    write_chunks(cat, out_dir, format, opts, "filtering noise", |_, data| {
        let keep = match *method {
            NoiseMethod::Sor { k, std_ratio } => filters::statistical_outlier_mask(&data.cloud.xyz, k, std_ratio),
            NoiseMethod::Ror { radius, min_neighbors } => filters::radius_outlier_mask(&data.cloud.xyz, radius, min_neighbors),
        };
        let core = data.core_indices();
        if classify {
            let mut out = data.cloud.take(&core);
            let mut cls = match out.attr("classification") {
                Some(c) => (0..out.len()).map(|i| c.get_f64(i) as u8).collect(),
                None => vec![0u8; out.len()],
            };
            for (k, &i) in core.iter().enumerate() {
                if !keep[i] {
                    cls[k] = NOISE_CLASS;
                }
            }
            out.attrs.insert("classification".into(), Attr::U8(cls));
            Ok(out)
        } else {
            let idx: Vec<usize> = core.into_iter().filter(|&i| keep[i]).collect();
            Ok(data.cloud.take(&idx))
        }
    })
}

// ------------------------------------------------------------------ reorganising

/// Cut the catalogue into new square tiles of `size` m (on a grid anchored
/// at `origin`, by default the catalogue's minimum snapped down to a
/// multiple of `size`), named `<xmin>_<ymin>`. With a `buffer`, each tile
/// also holds the points within `buffer` m of it, flagged by a `buffer`
/// attribute (1 for buffer points).
pub fn retile(cat: &Catalog, out_dir: &Path, size: f64, buffer: f64, origin: Option<(f64, f64)>, format: Option<&str>, workers: usize) -> Result<Vec<PathBuf>> {
    let opts = RunOptions { layout: Layout::Grid { size, origin }, buffer, workers };
    write_chunks(cat, out_dir, format, &opts, "retiling", |_, data| {
        if buffer > 0.0 {
            let mut c = data.cloud;
            c.attrs.insert("buffer".into(), buffer_attr(&data.buffer));
            Ok(c)
        } else {
            Ok(data.cloud.take(&data.core_indices()))
        }
    })
}

/// How [`decimate`] thins the points.
#[derive(Debug, Clone, Copy)]
pub enum Decimation {
    /// Keep `round(fraction * n)` points of each chunk at random
    /// ([`filters::random_indices`], seeded with `seed` plus the tile's
    /// position in the catalogue, or the chunk's index on a grid).
    Random { fraction: f64, seed: u64 },
    /// Keep the first point (in file order) of each `size` m voxel, on a
    /// grid anchored at the origin of the coordinates.
    Voxel { size: f64 },
    /// Keep the highest point of each `size` m cell in x, y (first in file
    /// order on ties), on a grid anchored at the origin: a surface for CHMs.
    Highest { size: f64 },
}

/// Thin every tile. Needs no buffer: each point's fate depends only on the
/// points of its own chunk (so a voxel that straddles two tiles keeps a
/// point in each).
pub fn decimate(cat: &Catalog, out_dir: &Path, method: &Decimation, format: Option<&str>, opts: &RunOptions) -> Result<Vec<PathBuf>> {
    match *method {
        Decimation::Random { fraction, .. } if !(0.0..=1.0).contains(&fraction) => return Err(Error::invalid(format!("fraction must be between 0 and 1, got {fraction}"))),
        Decimation::Voxel { size } | Decimation::Highest { size } if !(size.is_finite() && size > 0.0) => return Err(Error::invalid(format!("size must be a positive number, got {size}"))),
        _ => {}
    }
    let opts = RunOptions { buffer: 0.0, ..*opts };
    write_chunks(cat, out_dir, format, &opts, "decimating", |chunk, data| {
        let core = data.cloud.take(&data.core_indices());
        let idx = decimate_indices(&core, method, chunk.own.unwrap_or(chunk.index) as u64);
        Ok(core.take(&idx))
    })
}

/// Indices [`decimate`] keeps of `cloud`, in order.
pub fn decimate_indices(cloud: &PointCloud, method: &Decimation, stream: u64) -> Vec<usize> {
    match *method {
        Decimation::Random { fraction, seed } => filters::random_indices(cloud.len(), (fraction * cloud.len() as f64).round() as usize, seed.wrapping_add(stream)),
        Decimation::Voxel { size } => {
            let mut seen = std::collections::HashSet::new();
            (0..cloud.len()).filter(|&i| seen.insert(cloud.xyz[i].map(|v| (v / size).floor() as i64))).collect()
        }
        Decimation::Highest { size } => {
            let mut best: BTreeMap<(i64, i64), usize> = BTreeMap::new();
            for (i, p) in cloud.xyz.iter().enumerate() {
                let key = ((p[0] / size).floor() as i64, (p[1] / size).floor() as i64);
                best.entry(key).and_modify(|j| if p[2] > cloud.xyz[*j][2] { *j = i }).or_insert(i);
            }
            let mut idx: Vec<usize> = best.into_values().collect();
            idx.sort_unstable();
            idx
        }
    }
}

// ------------------------------------------------------------------ writing a cloud as tiles

/// A GeoTIFF GeoKeyDirectory record declaring EPSG `code` as a projected
/// CRS (or geographic for the 4000-4999 range), which LAS readers take as
/// the file's CRS.
pub fn epsg_vlr(code: u16) -> Vlr {
    let geographic = (4000..5000).contains(&code);
    let keys: [u16; 12] = [1, 1, 0, 2, 1024, 0, 1, if geographic { 2 } else { 1 }, if geographic { 2048 } else { 3072 }, 0, 1, code];
    Vlr { user_id: "LASF_Projection".into(), record_id: 34735, description: "GeoTiff GeoKeyDirectoryTag".into(), data: keys.iter().flat_map(|k| k.to_le_bytes()).collect() }
}

/// Write `cloud` as square tiles of `size` m on a grid anchored at `origin`
/// (by default its minimum snapped down to a multiple of `size`), one file
/// `<out_dir>/<xmin>_<ymin>.<ext>` per non-empty tile. Returns the paths and
/// point counts, west to east within south-to-north rows.
pub fn write_tiles(cloud: &PointCloud, out_dir: &Path, size: f64, origin: Option<(f64, f64)>, ext: &str, opts: &LasWriteOptions, epsg: Option<u16>) -> Result<Vec<(PathBuf, usize)>> {
    if !(size.is_finite() && size > 0.0) {
        return Err(Error::invalid(format!("tile size must be a positive number of metres, got {size}")));
    }
    let ext = ext.trim_start_matches('.').to_ascii_lowercase();
    if ext != "las" && ext != "laz" {
        return Err(Error::invalid(format!("format must be 'las' or 'laz', got {ext:?}")));
    }
    let Some((lo, _)) = cloud.bounds() else { return Ok(Vec::new()) };
    let (ox, oy) = origin.unwrap_or(((lo[0] / size).floor() * size, (lo[1] / size).floor() * size));
    let mut groups: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    for (i, p) in cloud.xyz.iter().enumerate() {
        if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) {
            return Err(Error::invalid(format!("point {i} has a non-finite coordinate")));
        }
        groups.entry((((p[1] - oy) / size).floor() as i64, ((p[0] - ox) / size).floor() as i64)).or_default().push(i);
    }
    std::fs::create_dir_all(out_dir)?;
    let vlrs: Vec<Vlr> = epsg.map(epsg_vlr).into_iter().collect();
    let groups: Vec<((i64, i64), Vec<usize>)> = groups.into_iter().collect();
    groups
        .par_iter()
        .map(|((r, c), idx)| {
            let name = format!("{}_{}.{ext}", coord_name(ox + *c as f64 * size), coord_name(oy + *r as f64 * size));
            let path = out_dir.join(name);
            write_las_with_vlrs(&cloud.take(idx), &path, opts, &vlrs)?;
            Ok((path, idx.len()))
        })
        .collect()
}

fn coord_name(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.3}").trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::als::read_chunk;
    use crate::nprandom::Generator;

    /// A 200 x 200 m sloping plane with a few 10 m boxes of "canopy",
    /// written as 2 x 2 tiles of 100 m.
    fn scene(dir: &Path) -> (PointCloud, Catalog) {
        let mut rng = Generator::new(7);
        let n = 40_000;
        let xy = rng.uniform_n(0.0, 200.0, 2 * n);
        let mut xyz = Vec::with_capacity(n);
        let mut cls = Vec::with_capacity(n);
        for i in 0..n {
            let (x, y) = (xy[2 * i], xy[2 * i + 1]);
            let g = 0.05 * x + 0.02 * y;
            let canopy = ((x / 25.0).floor() as i64 + (y / 25.0).floor() as i64) % 3 == 0 && i % 2 == 0;
            xyz.push([x, y, if canopy { g + 10.0 } else { g }]);
            cls.push(if canopy { 1u8 } else { 2u8 });
        }
        let mut cloud = PointCloud::new(xyz);
        cloud.attrs.insert("classification".into(), Attr::U8(cls));
        #[allow(clippy::needless_update)]
        let opts = LasWriteOptions { point_format: 6, scale: 0.001, ..Default::default() };
        let tiles = write_tiles(&cloud, dir, 100.0, None, "laz", &opts, Some(28355)).unwrap();
        assert_eq!(tiles.len(), 4);
        let cat = Catalog::open(&tiles.iter().map(|t| t.0.clone()).collect::<Vec<_>>());
        // What was written, quantised, is the reference.
        let back = crate::io::read(&tiles[0].0).unwrap();
        assert_eq!(back.len(), tiles[0].1);
        (cloud, cat)
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva-als-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn tiles_carry_their_crs_and_chunks_their_buffer() {
        let d = tmp("chunks");
        let (cloud, cat) = scene(&d);
        assert_eq!(cat.crs().as_deref(), Some("EPSG:28355"));
        assert_eq!(cat.n_points() as usize, cloud.len());
        let chunks = plan(&cat, Layout::Tiles, 10.0).unwrap();
        let data = read_chunk(&cat, &chunks[0]).unwrap();
        assert_eq!(data.n_core() as u64, cat.tiles[0].n_points);
        // Buffer points are within 10 m of the tile and in other files.
        let b = chunks[0].outer;
        assert!(data.cloud.xyz.iter().all(|p| p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]));
        let expect = cloud.xyz.iter().filter(|p| p[0] <= b[2] && p[1] <= b[3]).count();
        assert!((data.cloud.len() as i64 - expect as i64).abs() <= 2, "{} vs {expect}", data.cloud.len());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_dtm_does_not_depend_on_the_tiling_or_the_workers() {
        let d = tmp("dtm");
        let (_, cat) = scene(&d);
        let one = dtm(&cat, 2.0, &DtmMethod::Lowest, &RunOptions { workers: 1, ..Default::default() }).unwrap();
        let four = dtm(&cat, 2.0, &DtmMethod::Lowest, &RunOptions { workers: 4, ..Default::default() }).unwrap();
        assert_eq!(one, four);
        let grid = dtm(&cat, 2.0, &DtmMethod::Lowest, &RunOptions { layout: Layout::Grid { size: 50.0, origin: None }, buffer: 10.0, workers: 3 }).unwrap();
        // The merged cloud, read back so that it is quantised as the tiles are.
        let merged = crate::als::read_region(&cat, [-1.0, -1.0, 201.0, 201.0]).unwrap();
        let g: Vec<Point> = merged.xyz.iter().zip(classification(&merged).unwrap()).filter(|(_, c)| *c == 2).map(|(p, _)| *p).collect();
        let whole = ground::make_dtm(&g, 2.0, Some((0.0, 0.0, one.xmax() - 1.0, one.ymax() - 1.0))).unwrap();
        assert_eq!((whole.nrows, whole.ncols), (one.nrows, one.ncols));
        for r in 5..one.nrows - 5 {
            for c in 5..one.ncols - 5 {
                assert_eq!(one.get(r, c), whole.get(r, c), "cell {r} {c}");
                assert_eq!(grid.get(r, c), whole.get(r, c), "cell {r} {c}");
            }
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn chm_normalise_and_ground_run_on_tiles() {
        let d = tmp("ops");
        let (_, cat) = scene(&d);
        let chm_r = chm(&cat, 5.0, &Heights::Auto { resolution: 2.0 }, 0.0, &RunOptions::default()).unwrap();
        let top = chm_r.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((top - 10.0).abs() < 0.4, "{top}");
        let out = normalize(&cat, &d.join("norm"), &Heights::Auto { resolution: 2.0 }, true, None, &RunOptions::default()).unwrap();
        assert_eq!(out.len(), 4);
        let n = crate::io::read(&out[0]).unwrap();
        assert!(n.attrs.contains_key("elevation"));
        let zmax = n.xyz.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
        assert!((zmax - 10.0).abs() < 0.4, "{zmax}");
        let g = classify_ground(&cat, &d.join("ground"), &GroundMethod::Pmf(PmfParams { cell_size: 2.0, max_window: 30.0, ..Default::default() }), false, Some("las"), &RunOptions::default()).unwrap();
        assert!(g[0].extension().unwrap() == "las");
        let c = crate::io::read(&g[1]).unwrap();
        let truth = crate::io::read(&cat.tiles[1].path).unwrap();
        let agree = classification(&c).unwrap().iter().zip(classification(&truth).unwrap()).filter(|(a, b)| **a == *b).count();
        assert!(agree as f64 > 0.98 * c.len() as f64, "{agree} of {}", c.len());
        // Outputs never overwrite inputs.
        assert!(classify_ground(&cat, &d, &GroundMethod::Csf(CsfParams::default()), false, None, &RunOptions::default()).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn retile_and_decimate_keep_every_point_once() {
        let d = tmp("retile");
        let (cloud, cat) = scene(&d);
        let out = retile(&cat, &d.join("re"), 60.0, 0.0, Some((0.0, 0.0)), None, 2).unwrap();
        assert_eq!(out.len(), 16);
        let total: usize = out.iter().map(|p| crate::io::read(p).unwrap().len()).sum();
        assert_eq!(total, cloud.len());
        let buffered = retile(&cat, &d.join("rb"), 100.0, 5.0, None, None, 2).unwrap();
        let mut core = 0;
        for p in &buffered {
            let t = crate::io::read(p).unwrap();
            let flags = t.attr("buffer").unwrap().to_f64();
            core += flags.iter().filter(|&&b| b == 0.0).count();
            assert!(flags.contains(&1.0));
        }
        assert_eq!(core, cloud.len());
        let dec = decimate(&cat, &d.join("dec"), &Decimation::Random { fraction: 0.25, seed: 1 }, None, &RunOptions::default()).unwrap();
        let kept: usize = dec.iter().map(|p| crate::io::read(p).unwrap().len()).sum();
        assert!((kept as f64 - 0.25 * cloud.len() as f64).abs() <= 4.0);
        let high = decimate_indices(&cloud, &Decimation::Highest { size: 1000.0 }, 0);
        assert_eq!(high.len(), 1);
        assert!(decimate(&cat, &d.join("x"), &Decimation::Voxel { size: 0.0 }, None, &RunOptions::default()).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
