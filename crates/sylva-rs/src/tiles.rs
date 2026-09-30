// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Point clouds of any size as tiles: building tiles from scans, and point
//! operations whose tiled result is that of the whole cloud.
//!
//! The tile engine is that of [`crate::als`] (a catalogue of LAS/LAZ files
//! known from their headers, chunks with buffers, results in chunk order);
//! nothing in it is specific to airborne data. This module adds what a
//! terrestrial plot needs:
//!
//! - [`ScanTiler`] writes square tiles from any number of scans, thinned on
//!   one global voxel grid, holding one scan and one tile at a time. The
//!   tiles hold exactly what [`filters::voxel_downsample_indices_at`] keeps of
//!   all the scans concatenated.
//! - [`thin`], [`sor`], [`ror`], [`features`] and [`detect_stems`] run one
//!   tile at a time with a buffer. Neighbourhood operations are exact for any
//!   buffer: a point whose `k` nearest neighbours (or search sphere) might
//!   reach beyond the buffered box is searched again in the tiles its reach
//!   meets, streamed one at a time ([`core_neighbourhoods`]). Statistical outlier
//!   removal takes two passes, since its threshold uses the mean and standard
//!   deviation over the whole cloud.
//!
//! Every operation uses one chunk per tile ([`Layout::Tiles`]), so that a
//! tile's own file holds its core points and the tiles, in catalogue order,
//! are the whole cloud in order.

#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;

use crate::als::{self, merge_clouds, output_path, plan, run, workers_for, workers_for_estimates, write_like, Catalog, Chunk, ChunkData, Layout, BYTES_PER_POINT};
use crate::als_ops::{epsg_vlr, NOISE_CLASS};
use crate::error::{Error, Result};
use crate::filters;
use crate::io::las::{write_las_with_vlrs, LasWriteOptions};
use crate::pointcloud::Attr;
use crate::spatial::{voxel_key, KdTree, VoxelKey};
use crate::stems::{self, StemFit, StemParams};
use crate::transform::Transform;
use crate::{progress, Point, PointCloud};

/// What a tiled run did, for checking that memory stayed bounded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunInfo {
    /// Tiles (chunks) processed.
    pub chunks: usize,
    /// Most points held at once by one chunk: a tile with its buffer, a
    /// widened read, or (building tiles) a scan or a tile being assembled.
    pub max_points: usize,
    /// Points read over the whole run, buffers and re-reads included.
    pub points_read: u64,
    /// Chunks read a second time, wider, for points whose neighbourhood
    /// could reach beyond the buffer.
    pub rereads: usize,
    /// Points evaluated from such a widened read.
    pub widened_points: u64,
}

/// Counters shared by the workers of a run.
#[derive(Debug, Default)]
pub struct Tracker {
    chunks: AtomicUsize,
    max_points: AtomicUsize,
    points_read: AtomicU64,
    rereads: AtomicUsize,
    widened: AtomicU64,
}

impl Tracker {
    fn read(&self, n: usize) {
        self.max_points.fetch_max(n, Ordering::Relaxed);
        self.points_read.fetch_add(n as u64, Ordering::Relaxed);
    }

    fn chunk(&self, n: usize) {
        self.chunks.fetch_add(1, Ordering::Relaxed);
        self.read(n);
    }

    /// The counts so far.
    pub fn info(&self) -> RunInfo {
        RunInfo {
            chunks: self.chunks.load(Ordering::Relaxed),
            max_points: self.max_points.load(Ordering::Relaxed),
            points_read: self.points_read.load(Ordering::Relaxed),
            rereads: self.rereads.load(Ordering::Relaxed),
            widened_points: self.widened.load(Ordering::Relaxed),
        }
    }
}

// ------------------------------------------------------------------ helpers

pub(crate) fn boxes_meet(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Tiles with points whose extent meets `outer`, in catalogue order.
pub(crate) fn files_in(cat: &Catalog, outer: &[f64; 4]) -> Vec<usize> {
    (0..cat.tiles.len()).filter(|&i| cat.tiles[i].n_points > 0 && boxes_meet(&cat.tiles[i].xy(), outer)).collect()
}

/// For each side of `outer` (west, south, east, north), whether the
/// catalogue has points beyond it.
pub(crate) fn open_sides(ext: &[f64; 4], outer: &[f64; 4]) -> [bool; 4] {
    [outer[0] > ext[0], outer[1] > ext[1], outer[2] < ext[2], outer[3] < ext[3]]
}

/// Distance from `p` to the nearest side of `outer` beyond which there may
/// be points; every point not read lies farther than this.
pub(crate) fn margin(outer: &[f64; 4], open: &[bool; 4], p: &Point) -> f64 {
    let d = [p[0] - outer[0], p[1] - outer[1], outer[2] - p[0], outer[3] - p[1]];
    (0..4).filter(|&s| open[s]).map(|s| d[s]).fold(f64::INFINITY, f64::min)
}

/// The extension of a tile's output: `format` if given, else that of the
/// tile itself.
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

fn like_tile(chunk: &Chunk) -> usize {
    chunk.own.unwrap_or(chunk.files[0])
}

/// One chunk per tile with `buffer`, the workers the memory budget allows,
/// and every output path checked before anything is written.
pub(crate) fn prepare(cat: &Catalog, out_dir: Option<&Path>, format: Option<&str>, buffer: f64, workers: usize) -> Result<(Vec<Chunk>, usize)> {
    let chunks = plan(cat, Layout::Tiles, buffer)?;
    let w = workers_for(&chunks, workers, BYTES_PER_POINT)?;
    if let Some(dir) = out_dir {
        fs::create_dir_all(dir)?;
        for c in &chunks {
            output_path(cat, dir, &c.name, &out_ext(cat, c, format)?)?;
        }
    }
    Ok((chunks, w))
}

pub(crate) fn write_tile(cat: &Catalog, out_dir: &Path, format: Option<&str>, chunk: &Chunk, cloud: &PointCloud) -> Result<Option<PathBuf>> {
    if cloud.is_empty() {
        return Ok(None);
    }
    let path = output_path(cat, out_dir, &chunk.name, &out_ext(cat, chunk, format)?)?;
    write_like(cloud, &path, &cat.tiles[like_tile(chunk)])?;
    Ok(Some(path))
}

/// Run `f` on every tile with its buffer and write what it returns, one
/// file per tile named as the tile.
fn write_each(cat: &Catalog, out_dir: &Path, format: Option<&str>, buffer: f64, workers: usize, label: &str, tracker: &Tracker, f: impl Fn(&Chunk, ChunkData) -> Result<PointCloud> + Sync) -> Result<Vec<PathBuf>> {
    let (chunks, w) = prepare(cat, Some(out_dir), format, buffer, workers)?;
    let written = run(cat, &chunks, w, label, |chunk, data| {
        tracker.chunk(data.cloud.len());
        let cloud = f(chunk, data)?;
        write_tile(cat, out_dir, format, chunk, &cloud)
    })?;
    Ok(written.into_iter().flatten().flatten().collect())
}

/// The core points with `keep` true, or with `classify` all of them with
/// `classification` 7 on the others.
fn keep_or_classify(core: PointCloud, keep: &[bool], classify: bool) -> PointCloud {
    if classify {
        let mut out = core;
        let mut cls: Vec<u8> = match out.attr("classification") {
            Some(c) => (0..out.len()).map(|i| c.get_f64(i) as u8).collect(),
            None => vec![0u8; out.len()],
        };
        for (c, &k) in cls.iter_mut().zip(keep) {
            if !k {
                *c = NOISE_CLASS;
            }
        }
        out.attrs.insert("classification".into(), Attr::U8(cls));
        out
    } else {
        core.filter(keep)
    }
}

pub(crate) fn check_buffer(buffer: f64) -> Result<()> {
    if !(buffer.is_finite() && buffer >= 0.0) {
        return Err(Error::invalid(format!("buffer must be a non-negative number of metres, got {buffer}")));
    }
    Ok(())
}

// ------------------------------------------------------------------ neighbourhoods

/// A neighbourhood around each point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Query {
    /// The `k` nearest points, the point itself included ([`KdTree::knn`]).
    Knn(usize),
    /// Every point within this distance, the point itself included ([`KdTree::within`]).
    Radius(f64),
}

impl Query {
    fn find(&self, tree: &KdTree, p: &Point) -> Vec<(usize, f64)> {
        match *self {
            Query::Knn(k) => tree.knn(p, k),
            Query::Radius(r) => tree.within(p, r),
        }
    }

    /// How far a neighbourhood found in a partial read may reach: the
    /// distance of the `k`-th neighbour (infinite with fewer than `k`), or
    /// the radius. The true neighbourhood lies within it.
    fn reach(&self, nb: &[(usize, f64)]) -> f64 {
        match *self {
            Query::Knn(k) if nb.len() < k => f64::INFINITY,
            Query::Knn(k) => nb[k - 1].1,
            Query::Radius(r) => r,
        }
    }
}

/// `f(points, neighbours)` for every core point of a chunk, in order, with
/// each point's neighbourhood exactly as it is in the whole catalogue.
///
/// The neighbourhood is first searched in the chunk as read. It is exact
/// when it reaches no farther than the nearest side of the buffered box
/// beyond which the catalogue has points (every point not read lies
/// farther). The other points are searched again within their reach (an
/// upper bound of the true neighbourhood): the tiles it meets are read one
/// at a time, each point keeping its nearest candidates over them, so the
/// result never depends on the buffer and memory stays at one tile however
/// far a point reaches. A buffer wider than the typical reach only saves
/// the second read.
pub fn core_neighbourhoods<T: Send>(cat: &Catalog, chunk: &Chunk, data: ChunkData, query: Query, tracker: &Tracker, f: impl Fn(&[Point], &[(usize, f64)]) -> T + Sync) -> Result<Vec<T>> {
    let ext = cat.xy_bounds().ok_or_else(|| Error::invalid("the catalogue has no tiles"))?;
    let core = data.core_indices();
    let pts = &data.cloud.xyz;
    let open = open_sides(&ext, &chunk.outer);
    let first: Vec<std::result::Result<T, f64>> = {
        let tree = KdTree::new(pts);
        core.par_iter()
            .map(|&i| {
                let nb = query.find(&tree, &pts[i]);
                let reach = query.reach(&nb);
                if reach <= margin(&chunk.outer, &open, &pts[i]) {
                    Ok(f(pts, &nb))
                } else {
                    Err(reach)
                }
            })
            .collect()
    };
    let redo: Vec<(usize, f64)> = first.iter().enumerate().filter_map(|(j, r)| r.as_ref().err().map(|&reach| (j, reach))).collect();
    if redo.is_empty() {
        return Ok(first.into_iter().map(|r| match r { Ok(v) => v, Err(_) => unreachable!("all exact") }).collect());
    }
    // The box each point left needs, within the catalogue; the tiles meeting
    // any of them are streamed one at a time, each point keeping its nearest
    // candidates, so memory stays at one tile however far a point reaches
    // (an isolated return far above the canopy reaches across the plot).
    let boxes: Vec<[f64; 4]> = redo
        .iter()
        .map(|&(j, reach)| {
            let p = pts[core[j]];
            let b = if reach.is_finite() { [p[0] - reach, p[1] - reach, p[0] + reach, p[1] + reach] } else { ext };
            [b[0].max(ext[0]), b[1].max(ext[1]), b[2].min(ext[2]), b[3].min(ext[3])]
        })
        .collect();
    let queries: Vec<Point> = redo.iter().map(|&(j, _)| pts[core[j]]).collect();
    drop(data);
    let mut outer = chunk.outer;
    for b in &boxes {
        outer = [outer[0].min(b[0]), outer[1].min(b[1]), outer[2].max(b[2]), outer[3].max(b[3])];
    }
    // Tiles nearest the points first, so that each point's k-th candidate
    // distance soon bounds its search and farther tiles are skipped; ties
    // are ordered by tile and position, so the order of reading is immaterial.
    let dist_to = |q: &Point, b: &[f64; 6]| -> f64 {
        let d = [0, 1, 2].map(|a| (b[a] - q[a]).max(0.0).max(q[a] - b[a + 3]));
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    let mut bound: Vec<f64> = redo.iter().map(|&(_, reach)| reach).collect();
    let mut order = files_in(cat, &outer);
    order.sort_by(|&a, &b| {
        let da = queries.iter().map(|q| dist_to(q, &cat.tiles[a].bounds)).fold(f64::INFINITY, f64::min);
        let db = queries.iter().map(|q| dist_to(q, &cat.tiles[b].bounds)).fold(f64::INFINITY, f64::min);
        da.total_cmp(&db).then(a.cmp(&b))
    });
    let mut found: Vec<Vec<(f64, usize, usize, Point)>> = vec![Vec::new(); redo.len()];
    for t in order {
        let tb = cat.tiles[t].xy();
        let want: Vec<usize> = (0..redo.len()).filter(|&q| boxes_meet(&tb, &boxes[q]) && dist_to(&queries[q], &cat.tiles[t].bounds) <= bound[q]).collect();
        if want.is_empty() {
            continue;
        }
        let o = outer;
        let c = crate::io::las::read_las_where(&cat.tiles[t].path, |p: &Point| p[0] >= o[0] && p[0] <= o[2] && p[1] >= o[1] && p[1] <= o[3]).map_err(|e| match e {
            Error::File { .. } => e,
            other => Error::file(&cat.tiles[t].path, other.to_string()),
        })?;
        tracker.read(c.len());
        if c.is_empty() {
            continue;
        }
        let tree = KdTree::new(&c.xyz);
        let near: Vec<Vec<(f64, usize, usize, Point)>> = want.par_iter().map(|&q| query.find(&tree, &queries[q]).into_iter().map(|(i, d)| (d, t, i, c.xyz[i])).collect()).collect();
        for (&q, cand) in want.iter().zip(near) {
            let all = &mut found[q];
            all.extend(cand);
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            if let Query::Knn(k) = query {
                all.truncate(k);
                if all.len() == k {
                    bound[q] = bound[q].min(all[k - 1].0);
                }
            }
        }
    }
    tracker.rereads.fetch_add(1, Ordering::Relaxed);
    tracker.widened.fetch_add(redo.len() as u64, Ordering::Relaxed);
    let again: Vec<T> = found
        .par_iter()
        .map(|cand| {
            let local: Vec<Point> = cand.iter().map(|c| c.3).collect();
            let nb: Vec<(usize, f64)> = cand.iter().enumerate().map(|(i, c)| (i, c.0)).collect();
            f(&local, &nb)
        })
        .collect();
    let mut out = first;
    for ((j, _), v) in redo.into_iter().zip(again) {
        out[j] = Ok(v);
    }
    Ok(out.into_iter().map(|r| match r { Ok(v) => v, Err(_) => unreachable!("all evaluated") }).collect())
}

// ------------------------------------------------------------------ thinning

/// Keep the first point (in catalogue order: tile by tile, each in file
/// order) of every `voxel` m voxel of a grid with a corner at `origin`, and
/// write the tiles. The result is [`filters::voxel_downsample_indices_at`]
/// of the whole catalogue: a voxel that straddles tiles keeps one point, in
/// the first tile that has one. The buffer is set to the voxel size, which
/// is all it needs.
pub fn thin(cat: &Catalog, out_dir: &Path, voxel: f64, origin: Point, format: Option<&str>, workers: usize) -> Result<(Vec<PathBuf>, RunInfo)> {
    if !(voxel.is_finite() && voxel > 0.0) {
        return Err(Error::invalid(format!("voxel size must be a positive number of metres, got {voxel}")));
    }
    if origin.iter().any(|v| !v.is_finite()) {
        return Err(Error::invalid("the grid origin must be finite"));
    }
    let tracker = Tracker::default();
    // Points of one voxel are less than a voxel apart; the margin covers rounding.
    let buffer = voxel * (1.0 + 1e-6) + 1e-9;
    let paths = write_each(cat, out_dir, format, buffer, workers, "thinning tiles", &tracker, |_, data| {
        let mut seen: HashSet<VoxelKey> = HashSet::with_capacity(data.cloud.len() / 4);
        let idx: Vec<usize> = (0..data.cloud.len()).filter(|&i| seen.insert(voxel_key(&data.cloud.xyz[i], &origin, voxel)) && !data.buffer[i]).collect();
        Ok(data.cloud.take(&idx))
    })?;
    Ok((paths, tracker.info()))
}

// ------------------------------------------------------------------ noise

/// Mean distance to the `k` nearest neighbours (the point itself left out),
/// as [`filters::statistical_outlier_mask`] computes it.
fn mean_distance(nb: &[(usize, f64)]) -> f64 {
    let s: f64 = nb.iter().skip(1).map(|(_, d)| d).sum();
    s / (nb.len().saturating_sub(1).max(1)) as f64
}

fn write_f64s(path: &Path, v: &[f64]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn read_f64s(path: &Path) -> Result<Vec<f64>> {
    let bytes = fs::read(path)?;
    Ok(bytes.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b)).collect())
}

/// A scratch directory under `out_dir`, removed when dropped.
pub(crate) struct Scratch(pub(crate) PathBuf);

impl Scratch {
    pub(crate) fn new(out_dir: &Path, name: &str) -> Result<Scratch> {
        let dir = out_dir.join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        Ok(Scratch(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The threshold of statistical outlier removal over values stored per
/// chunk, summed in chunk order as the whole-cloud filter sums them in point
/// order: `mean + std_ratio * std` (population standard deviation).
fn sor_threshold(files: &[PathBuf], n: usize, std_ratio: f64) -> Result<f64> {
    let mut sum = 0.0;
    for p in files {
        sum = read_f64s(p)?.iter().fold(sum, |a, b| a + b);
    }
    let mu = sum / n as f64;
    let mut ss = 0.0;
    for p in files {
        ss = read_f64s(p)?.iter().fold(ss, |a, d| a + (d - mu) * (d - mu));
    }
    Ok(mu + std_ratio * (ss / n as f64).sqrt())
}

/// Statistical outlier removal (CloudCompare's SOR, as
/// [`filters::statistical_outlier_mask`]) over a whole catalogue, in two
/// passes: each point's mean distance to its `k` nearest neighbours, tile by
/// tile with exact neighbourhoods ([`core_neighbourhoods`]), stored beside
/// the output; then the mean and standard deviation over all the points,
/// summed in catalogue order; then the tiles are written without the points
/// above `mean + std_ratio * std` (or, with `classify`, with those points
/// classified 7). The kept points are exactly those the single-cloud filter
/// keeps of the whole catalogue.
pub fn sor(cat: &Catalog, out_dir: &Path, k: usize, std_ratio: f64, classify: bool, format: Option<&str>, buffer: f64, workers: usize) -> Result<(Vec<PathBuf>, RunInfo)> {
    if k == 0 || !std_ratio.is_finite() {
        return Err(Error::invalid(format!("k must be positive and std_ratio a finite number, got k = {k}, std_ratio = {std_ratio}")));
    }
    check_buffer(buffer)?;
    let tracker = Tracker::default();
    let (chunks, w) = prepare(cat, Some(out_dir), format, buffer, workers)?;
    let scratch = Scratch::new(out_dir, ".sylva-sor")?;
    let store = |c: &Chunk| scratch.0.join(format!("{}.f64", c.index));
    let counts = run(cat, &chunks, w, "SOR: neighbour distances", |chunk, data| {
        tracker.chunk(data.cloud.len());
        let d = core_neighbourhoods(cat, chunk, data, Query::Knn(k + 1), &tracker, |_, nb| mean_distance(nb))?;
        write_f64s(&store(chunk), &d)?;
        Ok(d.len())
    })?;
    let done: Vec<usize> = (0..chunks.len()).filter(|&i| counts[i].is_some()).collect();
    let n: usize = counts.iter().flatten().sum();
    let thr = if n <= k { f64::INFINITY } else { sor_threshold(&done.iter().map(|&i| store(&chunks[i])).collect::<Vec<_>>(), n, std_ratio)? };
    let (chunks0, w0) = prepare(cat, None, format, 0.0, workers)?;
    let written = run(cat, &chunks0, w0, "SOR: writing tiles", |chunk, data| {
        tracker.read(data.cloud.len());
        let core = data.cloud.take(&data.core_indices());
        drop(data);
        let d = read_f64s(&store(chunk))?;
        if d.len() != core.len() {
            return Err(Error::invalid(format!("tile {} changed while it was being filtered", chunk.name)));
        }
        let keep: Vec<bool> = d.iter().map(|&v| v <= thr).collect();
        write_tile(cat, out_dir, format, chunk, &keep_or_classify(core, &keep, classify))
    })?;
    drop(scratch);
    Ok((written.into_iter().flatten().flatten().collect(), tracker.info()))
}

/// Radius outlier removal over a catalogue: points with fewer than
/// `min_neighbors` others within `radius` are removed (or classified 7 with
/// `classify`), exactly as [`filters::radius_outlier_mask`] does on the whole
/// catalogue. A buffer of at least `radius` avoids any second read.
pub fn ror(cat: &Catalog, out_dir: &Path, radius: f64, min_neighbors: usize, classify: bool, format: Option<&str>, buffer: f64, workers: usize) -> Result<(Vec<PathBuf>, RunInfo)> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(Error::invalid(format!("radius must be a positive number of metres, got {radius}")));
    }
    check_buffer(buffer)?;
    let tracker = Tracker::default();
    let paths = write_each(cat, out_dir, format, buffer, workers, "ROR", &tracker, |chunk, data| {
        let core = data.cloud.take(&data.core_indices());
        let keep = core_neighbourhoods(cat, chunk, data, Query::Radius(radius), &tracker, |_, nb| nb.len().saturating_sub(1) >= min_neighbors)?;
        Ok(keep_or_classify(core, &keep, classify))
    })?;
    Ok((paths, tracker.info()))
}

// ------------------------------------------------------------------ features

/// Local shape features from the covariance of each point's `k` nearest
/// neighbours ([`filters::local_pca`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// `normal_x`, `normal_y`, `normal_z`: the unoriented normal, as
    /// [`filters::estimate_normals`].
    Normals,
    /// `planarity` and `linearity`, as [`filters::planarity_linearity`].
    Shape,
}

/// Add local PCA features to every point of a catalogue and write the
/// tiles; each point's values are those of the single-cloud function on the
/// whole catalogue.
pub fn features(cat: &Catalog, out_dir: &Path, k: usize, feature: Feature, format: Option<&str>, buffer: f64, workers: usize) -> Result<(Vec<PathBuf>, RunInfo)> {
    if k == 0 {
        return Err(Error::invalid("k must be positive"));
    }
    check_buffer(buffer)?;
    let tracker = Tracker::default();
    let k = k.max(3);
    let paths = write_each(cat, out_dir, format, buffer, workers, "local features", &tracker, |chunk, data| {
        let mut core = data.cloud.take(&data.core_indices());
        let pca = core_neighbourhoods(cat, chunk, data, Query::Knn(k), &tracker, filters::pca_of_neighbours)?;
        match feature {
            Feature::Normals => {
                for (a, name) in ["normal_x", "normal_y", "normal_z"].iter().enumerate() {
                    core.attrs.insert(name.to_string(), Attr::F64(pca.iter().map(|(n, _)| n[a]).collect()));
                }
            }
            Feature::Shape => {
                let (pl, li): (Vec<f64>, Vec<f64>) = pca.iter().map(|(_, v)| filters::planarity_linearity_of(v)).unzip();
                core.attrs.insert("planarity".into(), Attr::F64(pl));
                core.attrs.insert("linearity".into(), Attr::F64(li));
            }
        }
        Ok(core)
    })?;
    Ok((paths, tracker.info()))
}

// ------------------------------------------------------------------ stems

/// Distance from `(x, y)` to a half-open box, 0 inside.
fn dist_to_core(b: &[f64; 4], x: f64, y: f64) -> f64 {
    if x >= b[0] && x < b[2] && y >= b[1] && y < b[3] {
        return 0.0;
    }
    let dx = (b[0] - x).max(0.0).max(x - b[2]);
    let dy = (b[1] - y).max(0.0).max(y - b[3]);
    (dx * dx + dy * dy).sqrt().max(f64::MIN_POSITIVE)
}

/// The chunk responsible for a position: the one whose core holds it, or
/// the nearest (the lowest index on a tie), so that every position has one.
pub fn owner(cores: &[[f64; 4]], x: f64, y: f64) -> usize {
    let mut best = (f64::INFINITY, 0);
    for (i, c) in cores.iter().enumerate() {
        let d = dist_to_core(c, x, y);
        if d < best.0 {
            best = (d, i);
        }
    }
    best.1
}

/// Detect stems over a catalogue of height-normalised tiles
/// ([`stems::detect_stems_full`], with heights from `height_attr`, or z
/// without it). Each tile is searched with its buffer and keeps the stems
/// whose position (at the reference height) it is responsible for
/// ([`owner`]). Clusters are given their own random streams
/// ([`StemParams::cluster_seeds`]), so a stem depends only on the points
/// around it; with a buffer wider than the clusters and circles of the stems
/// at a tile edge (about `max_cluster_extent`, 2 m), the stems are those
/// found in the whole cloud with the same parameters. Returned sorted by
/// quality with ids 1..n.
pub fn detect_stems(cat: &Catalog, height_attr: &str, params: &StemParams, buffer: f64, workers: usize) -> Result<(Vec<StemFit>, RunInfo)> {
    check_buffer(buffer)?;
    let mut p = params.clone();
    p.cluster_seeds = true;
    let tracker = Tracker::default();
    let (chunks, w) = prepare(cat, None, None, buffer, workers)?;
    let cores: Vec<[f64; 4]> = chunks.iter().map(|c| c.core).collect();
    let found = run(cat, &chunks, w, "detecting stems", |chunk, data| {
        tracker.chunk(data.cloud.len());
        let h = data.cloud.attr(height_attr).map(|a| a.to_f64()).unwrap_or_else(|| data.cloud.xyz.iter().map(|q| q[2]).collect());
        let stems = stems::detect_stems_full(&data.cloud.xyz, &h, &p);
        Ok(stems.into_iter().filter(|s| owner(&cores, s.tree.x, s.tree.y) == chunk.index).collect::<Vec<_>>())
    })?;
    let mut out: Vec<StemFit> = found.into_iter().flatten().flatten().collect();
    out.sort_by(|a, b| b.tree.quality.total_cmp(&a.tree.quality));
    for (i, s) in out.iter_mut().enumerate() {
        s.tree.tree_id = i as i64 + 1;
    }
    Ok((out, tracker.info()))
}

// ------------------------------------------------------------------ tiles from scans

const SPILL_MAGIC: [u8; 4] = *b"SYTB";

pub(crate) fn attr_code(a: &Attr) -> u8 {
    match a {
        Attr::F64(_) => 0,
        Attr::F32(_) => 1,
        Attr::I64(_) => 2,
        Attr::I32(_) => 3,
        Attr::U32(_) => 4,
        Attr::U16(_) => 5,
        Attr::U8(_) => 6,
        Attr::I8(_) => 7,
        Attr::Bool(_) => 8,
    }
}

pub(crate) fn put_attr(buf: &mut Vec<u8>, a: &Attr) {
    match a {
        Attr::F64(v) => v.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
        Attr::F32(v) => v.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
        Attr::I64(v) => v.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
        Attr::I32(v) => v.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
        Attr::U32(v) => v.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
        Attr::U16(v) => v.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
        Attr::U8(v) => buf.extend_from_slice(v),
        Attr::I8(v) => v.iter().for_each(|x| buf.push(*x as u8)),
        Attr::Bool(v) => v.iter().for_each(|x| buf.push(*x as u8)),
    }
}

pub(crate) fn get_attr(code: u8, n: usize, r: &mut impl Read) -> Result<Attr> {
    let size = match code {
        0 | 2 => 8,
        1 | 3 | 4 => 4,
        5 => 2,
        6..=8 => 1,
        _ => return Err(Error::invalid(format!("unknown attribute type {code} in a scratch file"))),
    };
    let mut b = vec![0u8; n * size];
    r.read_exact(&mut b)?;
    macro_rules! cols {
        ($t:ty, $s:expr) => {
            b.chunks_exact($s).map(|c| <$t>::from_le_bytes(c.try_into().expect("sized"))).collect()
        };
    }
    Ok(match code {
        0 => Attr::F64(cols!(f64, 8)),
        1 => Attr::F32(cols!(f32, 4)),
        2 => Attr::I64(cols!(i64, 8)),
        3 => Attr::I32(cols!(i32, 4)),
        4 => Attr::U32(cols!(u32, 4)),
        5 => Attr::U16(cols!(u16, 2)),
        6 => Attr::U8(b),
        7 => Attr::I8(b.into_iter().map(|x| x as i8).collect()),
        _ => Attr::Bool(b.into_iter().map(|x| x != 0).collect()),
    })
}

/// Append `cloud` as one block to a scratch file: its points at full
/// precision and every attribute with its type.
pub(crate) fn write_block(w: &mut impl Write, cloud: &PointCloud) -> Result<()> {
    let mut buf = Vec::with_capacity(16 + cloud.len() * 32);
    buf.extend_from_slice(&SPILL_MAGIC);
    buf.extend_from_slice(&(cloud.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(cloud.attrs.len() as u32).to_le_bytes());
    for p in &cloud.xyz {
        for v in p {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    for (name, a) in &cloud.attrs {
        buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(attr_code(a));
        put_attr(&mut buf, a);
    }
    w.write_all(&buf)?;
    Ok(())
}

/// The next block of a scratch file, None at its end.
pub(crate) fn read_block(r: &mut impl Read) -> Result<Option<PointCloud>> {
    let mut magic = [0u8; 4];
    let mut got = 0;
    while got < 4 {
        let k = r.read(&mut magic[got..])?;
        if k == 0 {
            break;
        }
        got += k;
    }
    if got == 0 {
        return Ok(None);
    }
    if got < 4 || magic != SPILL_MAGIC {
        return Err(Error::invalid("a scratch file of the tiles is damaged"));
    }
    let mut n8 = [0u8; 8];
    r.read_exact(&mut n8)?;
    let n = u64::from_le_bytes(n8) as usize;
    let mut a4 = [0u8; 4];
    r.read_exact(&mut a4)?;
    let n_attrs = u32::from_le_bytes(a4);
    let mut xb = vec![0u8; n * 24];
    r.read_exact(&mut xb)?;
    let xyz: Vec<Point> = xb.as_chunks::<24>().0.iter().map(|c| [0, 1, 2].map(|k| f64::from_le_bytes(c[8 * k..8 * k + 8].try_into().expect("8 bytes")))).collect();
    let mut cloud = PointCloud::new(xyz);
    for _ in 0..n_attrs {
        r.read_exact(&mut a4)?;
        let mut name = vec![0u8; u32::from_le_bytes(a4) as usize];
        r.read_exact(&mut name)?;
        let mut code = [0u8; 1];
        r.read_exact(&mut code)?;
        cloud.attrs.insert(String::from_utf8_lossy(&name).to_string(), get_attr(code[0], n, r)?);
    }
    Ok(Some(cloud))
}

/// How [`ScanTiler`] thins and cuts the scans.
#[derive(Debug, Clone)]
pub struct ScanTiling {
    /// Side of the square tiles (m); a whole number of voxels when thinning.
    pub tile_size: f64,
    /// Keep the first point of each voxel of this size (m), over all the
    /// scans in order; None keeps every point.
    pub voxel_size: Option<f64>,
    /// A corner of the voxel grid and of the tile grid.
    pub origin: Point,
    /// Keep only the points inside `[xmin, ymin, xmax, ymax]` (closed).
    pub bounds: Option<[f64; 4]>,
    /// `"las"` or `"laz"`.
    pub format: String,
    /// Point format and coordinate scale of the tiles.
    pub las: LasWriteOptions,
    /// EPSG code recorded in each tile.
    pub epsg: Option<u16>,
}

impl Default for ScanTiling {
    fn default() -> Self {
        ScanTiling { tile_size: 10.0, voxel_size: None, origin: [0.0; 3], bounds: None, format: "laz".into(), las: LasWriteOptions::default(), epsg: None }
    }
}

/// Builds square tiles from scans added one at a time.
///
/// Each scan (after its transform and the crop to `bounds`) keeps the first
/// point of each voxel it has, and those points are appended, tile by tile,
/// to scratch files at full precision with a `scan_id` attribute (the
/// scan's position in the order added). [`ScanTiler::finish`] then
/// assembles each tile from its scratch file, keeping the first point per
/// voxel in scan order, and writes it. Tiles are cut along voxel
/// boundaries, so no voxel is split between tiles, and the tiles together
/// hold exactly the points [`filters::voxel_downsample_indices_at`] keeps
/// of all the scans concatenated. Only one scan, and one tile per worker,
/// is held in memory; the scratch files take about 40 bytes per point per
/// scan on disk, plus the attributes.
#[derive(Debug)]
pub struct ScanTiler {
    out_dir: PathBuf,
    scratch: PathBuf,
    params: ScanTiling,
    /// Voxels per tile side, when thinning.
    per_tile: i64,
    n_scans: u32,
    spilled: BTreeMap<(i64, i64), u64>,
    max_points: usize,
    points_in: u64,
}

impl ScanTiler {
    /// Start a set of tiles in `out_dir` (created if needed).
    pub fn new(out_dir: &Path, params: ScanTiling) -> Result<ScanTiler> {
        let t = params.tile_size;
        if !(t.is_finite() && t > 0.0) {
            return Err(Error::invalid(format!("tile size must be a positive number of metres, got {t}")));
        }
        let mut per_tile = 0;
        if let Some(v) = params.voxel_size {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::invalid(format!("voxel size must be a positive number of metres, got {v}")));
            }
            let m = t / v;
            if (m - m.round()).abs() > 1e-6 * m.max(1.0) || m.round() < 1.0 {
                return Err(Error::invalid(format!("the tile size ({t} m) must be a whole number of voxels ({v} m), so that no voxel is split between tiles")));
            }
            per_tile = m.round() as i64;
        }
        if params.origin.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("the grid origin must be finite"));
        }
        if let Some(b) = params.bounds {
            if b.iter().any(|v| !v.is_finite()) || b[2] < b[0] || b[3] < b[1] {
                return Err(Error::invalid(format!("bounds must be (xmin, ymin, xmax, ymax), got {b:?}")));
            }
        }
        let f = params.format.trim_start_matches('.').to_ascii_lowercase();
        if f != "las" && f != "laz" {
            return Err(Error::invalid(format!("format must be 'las' or 'laz', got {f:?}")));
        }
        fs::create_dir_all(out_dir)?;
        let scratch = out_dir.join(".sylva-scans");
        let _ = fs::remove_dir_all(&scratch);
        fs::create_dir_all(&scratch)?;
        Ok(ScanTiler { out_dir: out_dir.to_path_buf(), scratch, params: ScanTiling { format: f, ..params }, per_tile, n_scans: 0, spilled: BTreeMap::new(), max_points: 0, points_in: 0 })
    }

    fn spill_path(&self, key: (i64, i64)) -> PathBuf {
        self.scratch.join(format!("{}_{}.bin", key.1, key.0))
    }

    /// Scans added so far.
    pub fn n_scans(&self) -> usize {
        self.n_scans as usize
    }

    /// Add the next scan, moved by `transform` if given. Returns the points
    /// it contributes (inside the bounds, first in their voxel within the
    /// scan).
    pub fn add(&mut self, mut cloud: PointCloud, transform: Option<&Transform>) -> Result<usize> {
        let scan_id = self.n_scans;
        self.n_scans += 1;
        self.max_points = self.max_points.max(cloud.len());
        self.points_in += cloud.len() as u64;
        if let Some(t) = transform {
            cloud.transform_in_place(t);
        }
        if let Some(i) = cloud.xyz.iter().position(|p| !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite())) {
            return Err(Error::invalid(format!("point {i} of scan {scan_id} has a non-finite coordinate")));
        }
        let p = &self.params;
        let (o, size) = (p.origin, p.tile_size);
        let mut seen: HashSet<VoxelKey> = HashSet::new();
        let mut groups: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
        for (i, q) in cloud.xyz.iter().enumerate() {
            if let Some(b) = p.bounds {
                if !(q[0] >= b[0] && q[0] <= b[2] && q[1] >= b[1] && q[1] <= b[3]) {
                    continue;
                }
            }
            // Row (y) first, so that tiles come south to north, west to east.
            let key = match p.voxel_size {
                Some(v) => {
                    let k = voxel_key(q, &o, v);
                    if !seen.insert(k) {
                        continue;
                    }
                    (k[1].div_euclid(self.per_tile), k[0].div_euclid(self.per_tile))
                }
                None => (((q[1] - o[1]) / size).floor() as i64, ((q[0] - o[0]) / size).floor() as i64),
            };
            groups.entry(key).or_default().push(i);
        }
        drop(seen);
        let mut kept = 0;
        for (key, idx) in groups {
            let mut block = cloud.take(&idx);
            block.attrs.insert("scan_id".into(), Attr::U32(vec![scan_id; idx.len()]));
            let path = self.spill_path(key);
            let mut w = BufWriter::new(fs::OpenOptions::new().create(true).append(true).open(&path)?);
            write_block(&mut w, &block)?;
            w.flush()?;
            *self.spilled.entry(key).or_default() += idx.len() as u64;
            kept += idx.len();
        }
        Ok(kept)
    }

    /// Assemble and write every tile (`<xmin>_<ymin>.<ext>`, on `workers`
    /// threads, fewer if memory is short), remove the scratch files and
    /// return the paths and point counts, south to north, west to east.
    pub fn finish(self, workers: usize) -> Result<(Vec<(PathBuf, usize)>, RunInfo)> {
        let keys: Vec<((i64, i64), u64)> = self.spilled.iter().map(|(k, n)| (*k, *n)).collect();
        let w = workers_for_estimates(&keys.iter().map(|k| k.1).collect::<Vec<_>>(), workers, BYTES_PER_POINT)?;
        let vlrs: Vec<las::Vlr> = self.params.epsg.map(epsg_vlr).into_iter().collect();
        let task = progress::start("writing tiles", keys.len() as u64);
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let held = AtomicUsize::new(0);
        let results: Mutex<Vec<Option<(PathBuf, usize)>>> = Mutex::new(vec![None; keys.len()]);
        let errors: Mutex<Vec<(usize, Error)>> = Mutex::new(Vec::new());
        let one = |i: usize| -> Result<Option<(PathBuf, usize)>> {
            let (key, _) = keys[i];
            let mut r = BufReader::new(File::open(self.spill_path(key))?);
            let mut seen: HashSet<VoxelKey> = HashSet::new();
            let mut parts = Vec::new();
            let mut n = 0usize;
            while let Some(block) = read_block(&mut r)? {
                held.fetch_max(n + block.len(), Ordering::Relaxed);
                let part = match self.params.voxel_size {
                    Some(v) => {
                        let idx: Vec<usize> = (0..block.len()).filter(|&j| seen.insert(voxel_key(&block.xyz[j], &self.params.origin, v))).collect();
                        if idx.len() == block.len() { block } else { block.take(&idx) }
                    }
                    None => block,
                };
                n += part.len();
                parts.push(part);
            }
            drop(seen);
            let cloud = merge_clouds(parts);
            if cloud.is_empty() {
                return Ok(None);
            }
            let (x0, y0) = (self.params.origin[0] + key.1 as f64 * self.params.tile_size, self.params.origin[1] + key.0 as f64 * self.params.tile_size);
            let path = self.out_dir.join(format!("{}_{}.{}", coord_name(x0), coord_name(y0), self.params.format));
            write_las_with_vlrs(&cloud, &path, &self.params.las, &vlrs)?;
            Ok(Some((path, cloud.len())))
        };
        let work = || loop {
            if failed.load(Ordering::Relaxed) {
                break;
            }
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= keys.len() {
                break;
            }
            match one(i) {
                Ok(v) => results.lock().expect("results lock")[i] = v,
                Err(e) => {
                    failed.store(true, Ordering::Relaxed);
                    errors.lock().expect("errors lock").push((i, e));
                }
            }
            task.inc(1);
        };
        std::thread::scope(|s| {
            for _ in 0..w.max(1).min(keys.len().max(1)) {
                s.spawn(work);
            }
        });
        let _ = fs::remove_dir_all(&self.scratch);
        let mut errors = errors.into_inner().expect("errors lock");
        errors.sort_by_key(|e| e.0);
        if let Some((_, e)) = errors.into_iter().next() {
            return Err(e);
        }
        let written: Vec<(PathBuf, usize)> = results.into_inner().expect("results lock").into_iter().flatten().collect();
        let info = RunInfo { chunks: written.len(), max_points: self.max_points.max(held.load(Ordering::Relaxed)), points_read: self.points_in, rereads: 0, widened_points: 0 };
        Ok((written, info))
    }
}

impl Drop for ScanTiler {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.scratch);
    }
}

/// A coordinate as a file name part: integral values without decimals.
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
    use crate::nprandom::Generator;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva-tiles-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    /// Clustered points over 30 x 30 m (blobs of a few thousand points and
    /// sparse noise between them), with a `pid` attribute.
    fn plot(n: usize, seed: u64) -> PointCloud {
        let mut rng = Generator::new(seed);
        let centres = rng.uniform_n(0.0, 30.0, 40);
        let u = rng.uniform_n(0.0, 1.0, 4 * n);
        let mut xyz = Vec::with_capacity(n);
        for i in 0..n {
            let c = (i % 20) * 2;
            if i % 50 == 0 {
                xyz.push([30.0 * u[4 * i], 30.0 * u[4 * i + 1], 10.0 * u[4 * i + 2]]);
            } else {
                let r = 1.5 * u[4 * i + 3];
                let a = std::f64::consts::TAU * u[4 * i];
                xyz.push([centres[c] + r * a.cos(), centres[c + 1] + r * a.sin(), 8.0 * u[4 * i + 1]]);
            }
        }
        let mut cloud = PointCloud::new(xyz);
        cloud.attrs.insert("pid".into(), Attr::I64((0..n as i64).collect()));
        cloud
    }

    /// `cloud` written as `size` m tiles, the catalogue, and all its points
    /// read back (quantised as the tiles are).
    fn tiled(cloud: &PointCloud, dir: &Path, size: f64) -> (Catalog, PointCloud) {
        let opts = LasWriteOptions { point_format: 6, scale: 0.0001, ..Default::default() };
        let written = crate::als_ops::write_tiles(cloud, dir, size, Some((0.0, 0.0)), "laz", &opts, None).unwrap();
        let cat = Catalog::open(&written.iter().map(|w| w.0.clone()).collect::<Vec<_>>());
        let whole = als::read_region(&cat, [-1e9, -1e9, 1e9, 1e9]).unwrap();
        (cat, whole)
    }

    fn read_all(paths: &[PathBuf]) -> PointCloud {
        merge_clouds(paths.iter().map(|p| crate::io::read(p).unwrap()).collect())
    }

    fn pids(c: &PointCloud) -> Vec<i64> {
        let mut v: Vec<i64> = c.attr("pid").unwrap().to_f64().iter().map(|&x| x as i64).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn scratch_blocks_round_trip() {
        let mut c = plot(100, 1);
        c.attrs.insert("f".into(), Attr::F32((0..100).map(|i| i as f32 * 0.5).collect()));
        c.attrs.insert("b".into(), Attr::Bool((0..100).map(|i| i % 3 == 0).collect()));
        c.attrs.insert("s".into(), Attr::I8((0..100).map(|i| -(i as i8)).collect()));
        let mut buf = Vec::new();
        write_block(&mut buf, &c).unwrap();
        write_block(&mut buf, &c.take(&[3, 4])).unwrap();
        let mut r = &buf[..];
        let a = read_block(&mut r).unwrap().unwrap();
        let b = read_block(&mut r).unwrap().unwrap();
        assert!(read_block(&mut r).unwrap().is_none());
        assert_eq!(a.xyz, c.xyz);
        assert_eq!(a.attrs, c.attrs);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn thinning_and_noise_filters_match_the_whole_cloud() {
        let d = tmp("ops");
        let (cat, whole) = tiled(&plot(30_000, 2), &d.join("in"), 10.0);
        assert!(cat.len() >= 9);
        // Voxel thinning on a global grid.
        let (paths, info) = thin(&cat, &d.join("thin"), 0.25, [0.0; 3], None, 3).unwrap();
        let expect = whole.take(&filters::voxel_downsample_indices_at(&whole.xyz, &[0.0; 3], 0.25));
        assert_eq!(pids(&read_all(&paths)), pids(&expect));
        assert!(info.max_points < whole.len() / 2, "{info:?}");
        // SOR, with a buffer too narrow for some neighbourhoods.
        let (paths, info) = sor(&cat, &d.join("sor"), 6, 1.0, false, None, 0.05, 2).unwrap();
        let keep = filters::statistical_outlier_mask(&whole.xyz, 6, 1.0);
        assert!(keep.iter().any(|k| !k));
        assert_eq!(pids(&read_all(&paths)), pids(&whole.filter(&keep)));
        assert!(info.rereads > 0 && info.widened_points > 0, "{info:?}");
        // ROR with and without a buffer.
        let keep = filters::radius_outlier_mask(&whole.xyz, 0.3, 4);
        for buffer in [0.0, 0.3] {
            let (paths, info) = ror(&cat, &d.join(format!("ror{buffer}")), 0.3, 4, false, None, buffer, 4).unwrap();
            assert_eq!(pids(&read_all(&paths)), pids(&whole.filter(&keep)));
            assert_eq!(info.rereads > 0, buffer == 0.0, "{info:?}");
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn features_match_the_whole_cloud() {
        let d = tmp("features");
        let (cat, whole) = tiled(&plot(12_000, 3), &d.join("in"), 15.0);
        let (paths, _) = features(&cat, &d.join("n"), 10, Feature::Shape, None, 0.5, 2).unwrap();
        let (pl, li) = filters::planarity_linearity(&whole.xyz, 10);
        let got = read_all(&paths);
        let gid = got.attr("pid").unwrap().to_f64();
        let wid = whole.attr("pid").unwrap().to_f64();
        let pos: BTreeMap<i64, usize> = wid.iter().enumerate().map(|(i, &p)| (p as i64, i)).collect();
        let (gpl, gli) = (got.attr("planarity").unwrap().to_f64(), got.attr("linearity").unwrap().to_f64());
        for (j, &p) in gid.iter().enumerate() {
            let i = pos[&(p as i64)];
            assert_eq!((gpl[j], gli[j]), (pl[i], li[i]), "point {p}");
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn tiles_from_scans_hold_the_thinned_concatenation() {
        let d = tmp("scans");
        let scans: Vec<PointCloud> = (0..4).map(|s| plot(5_000, 10 + s)).collect();
        let shift = Transform::from_row_major(&[1.0, 0.0, 0.0, 0.3, 0.0, 1.0, 0.0, -0.2, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let params = ScanTiling { tile_size: 7.5, voxel_size: Some(0.25), bounds: Some([1.0, 1.0, 28.0, 29.0]), las: LasWriteOptions { scale: 0.0001, ..Default::default() }, ..Default::default() };
        let mut tiler = ScanTiler::new(&d, params).unwrap();
        let mut all = Vec::new();
        for (s, c) in scans.iter().enumerate() {
            let c = if s == 2 { c.transformed(&shift) } else { c.clone() };
            let mut tagged = c.clone();
            tagged.attrs.insert("scan_id".into(), Attr::U32(vec![s as u32; c.len()]));
            all.push(tagged);
            tiler.add(scans[s].clone(), if s == 2 { Some(&shift) } else { None }).unwrap();
        }
        let (written, info) = tiler.finish(3).unwrap();
        assert!(!d.join(".sylva-scans").exists());
        let concat = merge_clouds(all);
        let crop: Vec<usize> = (0..concat.len()).filter(|&i| { let q = concat.xyz[i]; q[0] >= 1.0 && q[0] <= 28.0 && q[1] >= 1.0 && q[1] <= 29.0 }).collect();
        let concat = concat.take(&crop);
        let expect = concat.take(&filters::voxel_downsample_indices_at(&concat.xyz, &[0.0; 3], 0.25));
        let got = read_all(&written.iter().map(|w| w.0.clone()).collect::<Vec<_>>());
        let key = |c: &PointCloud| {
            let (s, p) = (c.attr("scan_id").unwrap().to_f64(), c.attr("pid").unwrap().to_f64());
            let mut v: Vec<(i64, i64)> = s.iter().zip(&p).map(|(a, b)| (*a as i64, *b as i64)).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(key(&got), key(&expect));
        assert_eq!(written.iter().map(|w| w.1).sum::<usize>(), expect.len());
        assert!(written[0].0.file_name().unwrap().to_string_lossy().starts_with("0_0."));
        assert!(info.max_points <= 5_000 + expect.len() / 2, "{info:?}");
        assert!(ScanTiler::new(&d.join("x"), ScanTiling { tile_size: 1.0, voxel_size: Some(0.3), ..Default::default() }).is_err());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn owners_partition_the_plane() {
        let cores = [[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]];
        assert_eq!(owner(&cores, 10.0, 5.0), 1);
        assert_eq!(owner(&cores, 9.99, 5.0), 0);
        assert_eq!(owner(&cores, 25.0, 5.0), 1);
        assert_eq!(owner(&cores, -3.0, 50.0), 0);
    }
}
