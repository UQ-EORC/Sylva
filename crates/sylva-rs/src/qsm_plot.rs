// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A QSM for every tree of a segmented plot, and the plot's tables and files.
//!
//! [`build_plot`] is the loop around [`crate::qsm::build_qsm`]: each tree's
//! points are taken from the labels, thinned, put through the wood filter
//! and fitted, and a buttress is looked for and meshed. A tree that cannot
//! be fitted is recorded rather than failing the plot. The trees are done
//! one after another, as before; each fit is parallel inside.
//!
//! The plot's table ([`table`], [`table_csv`]), meshes ([`write_meshes`])
//! and cylinder files ([`write_cylinders`]) are read from [`PlotEntry`]
//! views of the models, as the Python package's `PlotQSMs` holds them.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::buttress_detect as bd;
use crate::filters::voxel_downsample_indices;
use crate::json::py_float_repr;
use crate::leaf_model::qsm_from_rows;
use crate::qsm::buttress::{self as bm, Buttress};
use crate::qsm::metrics::tree_metrics;
use crate::qsm::wood::{wood_mask, WoodParams};
use crate::qsm::{build_qsm, QsmParams};
use crate::qsm_ops::{self as ops, Row};
use crate::{Error, Point, Result};

/// `np.median`, NaN if any value is.
fn median(v: &[f64]) -> f64 {
    if v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    crate::numeric::median(v)
}

/// `max` of Python floats: the first unless the second is larger.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Terrain elevation under a stem, from the points within 1 m of `(cx, cy)`
/// (all points if none are): the median of `z - height`.
pub fn buttress_ground_z(points: &[Point], heights: &[f64], cx: f64, cy: f64) -> f64 {
    let base: Vec<f64> = points.iter().zip(heights).map(|(p, h)| p[2] - h).collect();
    let near: Vec<f64> = points.iter().zip(&base).filter(|(p, _)| (p[0] - cx).hypot(p[1] - cy) <= 1.0).map(|(_, &b)| b).collect();
    median(if near.is_empty() { &base } else { &near })
}

/// Refuse a buttress raster that would not fit in memory. The raster is
/// fixed by the settings, not by the cloud, so a fine resolution over a
/// wide reach is a large grid whatever was scanned.
pub fn buttress_raster_check(p: &bm::ButtressParams) -> Result<()> {
    let nx = (2.0 * p.max_radius / py_max(p.resolution, 1e-6)).trunc() as i128 + 1;
    let nz = (p.max_height / py_max(p.slice, 1e-6)).trunc() as i128 + 1;
    let cells = (nx * nx * nz).max(0) as u128;
    crate::limits::check_cells(
        cells,
        2,
        &format!("a {nx} x {nx} x {nz} buttress raster at {} m", py_float_repr(p.resolution)),
        "a coarser resolution, a smaller max_radius, or a lower max_height",
    )
}

/// [`bm::buttress_mesh`] with the ground found from the points when
/// `ground_z` is None ([`buttress_ground_z`]) and the raster size checked.
pub fn buttress_mesh(points: &[Point], heights: &[f64], cx: f64, cy: f64, ground_z: Option<f64>, p: &bm::ButtressParams) -> Result<Buttress> {
    if heights.len() != points.len() {
        return Err(Error::invalid("heights must have one value per point"));
    }
    let ground_z = ground_z.unwrap_or_else(|| buttress_ground_z(points, heights, cx, cy));
    buttress_raster_check(p)?;
    if p.resolution <= 0.0 || p.slice <= 0.0 || p.max_height <= 0.0 {
        return Err(Error::invalid("resolution, slice and max_height must be positive"));
    }
    Ok(bm::buttress_mesh(points, heights, cx, cy, ground_z, p))
}

/// The wood of one tree's points, thinned to 2 cm first, with the filter's
/// default settings (`sylva.qsm.wood_points`).
pub fn wood_points(points: &[Point]) -> Vec<Point> {
    let thin: Vec<Point> = voxel_downsample_indices(points, 0.02).into_iter().map(|i| points[i]).collect();
    let mask = wood_mask(&thin, &WoodParams::default());
    thin.into_iter().zip(mask).filter(|(_, m)| *m).map(|(p, _)| p).collect()
}

/// Settings of [`build_plot`].
#[derive(Debug, Clone)]
pub struct PlotParams {
    /// Thin each tree to this spacing first (m); 0 keeps every point.
    pub voxel_size: f64,
    /// Run the wood filter on each tree first.
    pub wood: bool,
    /// Look for a buttress on each tree and mesh it (needs heights).
    pub buttress: bool,
    /// Trees with fewer points are skipped.
    pub min_points: f64,
    /// Passed to [`build_qsm`].
    pub qsm: QsmParams,
}

impl Default for PlotParams {
    fn default() -> Self {
        PlotParams { voxel_size: 0.01, wood: true, buttress: false, min_points: 2000.0, qsm: QsmParams::default() }
    }
}

/// The models of a plot, each list in tree order.
#[derive(Debug, Clone, Default)]
pub struct PlotModels {
    /// Cylinder rows of the trees that were fitted.
    pub models: Vec<(i64, Vec<Row>)>,
    /// Buttresses found and meshed.
    pub buttresses: Vec<(i64, Buttress)>,
    /// Why a tree was not modelled: too few points, or the fit's error.
    pub skipped: Vec<(i64, String)>,
    /// Points per tree that had enough.
    pub points: Vec<(i64, usize)>,
    /// Highest point of those trees above ground (only with heights).
    pub heights: Vec<(i64, f64)>,
}

/// A QSM for every tree of a segmented plot.
///
/// `labels` holds the tree id per point (below 0 is not a tree); `heights`,
/// if given, the height above ground per point; `stems` the stem centre and
/// DBH `(tree_id, [x, y], dbh)` each model is built around (a later entry
/// for the same tree wins). Without a centre it is the median xy of the
/// thinned tree's points between 0.5 and 1.5 m above ground (all its points
/// if 20 or fewer are there). A finite, positive stem DBH anchors the model's
/// base radius (`base_radius = dbh / 2`) unless `p.qsm.base_radius` is set:
/// without an anchor, a tree with too few good stem circles takes its taper
/// prior from its widest fitted circle, which on a small tree with a leafy
/// crown is often foliage, and the thin stem was then replaced by a trunk of
/// up to a metre.
pub fn build_plot(points: &[Point], labels: &[i64], heights: Option<&[f64]>, stems: &[(i64, [f64; 2], f64)], p: &PlotParams) -> Result<PlotModels> {
    if labels.len() != points.len() {
        return Err(Error::invalid("labels must have one value per point"));
    }
    if heights.is_some_and(|h| h.len() != points.len()) {
        return Err(Error::invalid("heights must have one value per point"));
    }
    let mut members: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &t) in labels.iter().enumerate() {
        if t >= 0 {
            members.entry(t).or_default().push(i);
        }
    }
    let centres: HashMap<i64, [f64; 2]> = stems.iter().map(|&(t, c, _)| (t, c)).collect();
    let dbh: HashMap<i64, f64> = stems.iter().map(|&(t, _, d)| (t, d)).collect();
    let mut out = PlotModels::default();
    let task = crate::progress::start("fitting QSMs", members.len() as u64);
    for (&tid, idx) in &members {
        task.inc(1);
        let n = idx.len();
        if (n as f64) < p.min_points {
            out.skipped.push((tid, format!("{n} points")));
            continue;
        }
        let tree: Vec<Point> = idx.iter().map(|&i| points[i]).collect();
        let tree_h: Option<Vec<f64>> = heights.map(|h| idx.iter().map(|&i| h[i]).collect());
        out.points.push((tid, n));
        if let Some(h) = &tree_h {
            let top = if h.iter().any(|x| x.is_nan()) { f64::NAN } else { h.iter().copied().fold(f64::NEG_INFINITY, f64::max) };
            out.heights.push((tid, top));
        }
        let keep: Vec<usize> = if p.voxel_size > 0.0 { voxel_downsample_indices(&tree, p.voxel_size) } else { (0..n).collect() };
        let thin: Vec<Point> = keep.iter().map(|&i| tree[i]).collect();
        let base = match centres.get(&tid) {
            Some(&c) => c,
            None => {
                let low: Vec<Point> = match &tree_h {
                    Some(h) => keep.iter().filter(|&&i| h[i] > 0.5 && h[i] < 1.5).map(|&i| tree[i]).collect(),
                    None => thin.clone(),
                };
                let from = if low.len() > 20 { &low } else { &thin };
                let xs: Vec<f64> = from.iter().map(|q| q[0]).collect();
                let ys: Vec<f64> = from.iter().map(|q| q[1]).collect();
                [median(&xs), median(&ys)]
            }
        };
        let input = if p.wood { wood_points(&thin) } else { thin };
        let mut qp = p.qsm.clone();
        if qp.base_radius <= 0.0 {
            if let Some(&d) = dbh.get(&tid) {
                if d.is_finite() && d > 0.0 {
                    qp.base_radius = d / 2.0;
                }
            }
        }
        match build_qsm(&input, Some(base), &qp) {
            Ok(q) => out.models.push((tid, q.to_rows())),
            Err(e) => {
                out.skipped.push((tid, e.to_string()));
                continue;
            }
        }
        if let (true, Some(h)) = (p.buttress, &tree_h) {
            let found = bd::detect_buttress(&tree, h, Some(base), &bd::ButtressParams::default())?;
            if found.buttressed {
                let bp = bm::ButtressParams { top: Some(found.top), ..Default::default() };
                let b = buttress_mesh(&tree, h, found.centre[0], found.centre[1], None, &bp)?;
                if !b.faces.is_empty() {
                    out.buttresses.push((tid, b));
                }
            }
        }
    }
    Ok(out)
}

/// Median share of the models' length that was fitted to points rather than
/// taken from the priors; None without models. A low share usually means a
/// cloud too sparse for the shell width.
pub fn median_measured_length<'a>(models: impl IntoIterator<Item = &'a [Row]>) -> Option<f64> {
    let shares: Vec<f64> = models.into_iter().map(|rows| tree_metrics(&qsm_from_rows(rows), 1.0, 0.5).measured_length_fraction).collect();
    if shares.is_empty() {
        None
    } else {
        Some(median(&shares))
    }
}

/// A tree's buttress as a plot holds it.
#[derive(Debug, Clone, Copy)]
pub struct ButtressView<'a> {
    pub vertices: &'a [Point],
    pub faces: &'a [[u32; 3]],
    pub volume: f64,
    pub top: f64,
    pub top_z: f64,
}

/// One tree of a plot: its model, and what `build_plot` recorded about it.
#[derive(Debug, Clone, Copy)]
pub struct PlotEntry<'a> {
    pub tree_id: i64,
    pub rows: &'a [Row],
    pub points: Option<i64>,
    pub height: Option<f64>,
    pub buttress: Option<ButtressView<'a>>,
}

/// Wood volume of one tree (m³): the buttress below its top plus the
/// cylinders above it where there is one, the cylinders otherwise.
pub fn tree_volume(e: &PlotEntry) -> f64 {
    match &e.buttress {
        Some(b) => b.volume + ops::volume_above(e.rows, b.top_z),
        None => ops::totals(e.rows).total_volume,
    }
}

/// Wood volume of the whole plot (m³), summed in the order given.
pub fn total_volume(entries: &[PlotEntry]) -> f64 {
    entries.iter().fold(0.0, |s, e| s + tree_volume(e))
}

/// Python's `round(x, digits)`: to the nearest multiple of `10^-digits`,
/// ties to even on the exact binary value.
pub fn py_round(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// One row of the plot table.
#[derive(Debug, Clone, PartialEq)]
pub struct TableRow {
    pub tree_id: i64,
    /// Points of the tree (None for a model not from `build_plot`).
    pub points: Option<i64>,
    pub volume_m3: f64,
    pub dbh_m: f64,
    pub height_m: f64,
    pub n_cylinders: usize,
    /// Share of the volume and length fitted to points rather than priors.
    pub measured_volume: f64,
    pub measured_length: f64,
    pub buttress_m3: Option<f64>,
    pub buttress_top_m: Option<f64>,
}

/// Column names of the plot table.
pub const TABLE_COLUMNS: [&str; 10] = ["tree_id", "points", "volume_m3", "dbh_m", "height_m", "n_cylinders", "measured_volume", "measured_length", "buttress_m3", "buttress_top_m"];

/// One row per tree in tree order, rounded as the CSV carries them: volumes
/// to 5 decimals, DBH to 4, heights to 2 and shares to 3.
pub fn table(entries: &[PlotEntry]) -> Vec<TableRow> {
    let mut sorted: Vec<&PlotEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| e.tree_id);
    sorted
        .into_iter()
        .map(|e| {
            let q = qsm_from_rows(e.rows);
            let fit = tree_metrics(&q, 1.0, 0.5);
            TableRow {
                tree_id: e.tree_id,
                points: e.points,
                volume_m3: py_round(tree_volume(e), 5),
                dbh_m: py_round(q.dbh(), 4),
                height_m: py_round(e.height.unwrap_or(f64::NAN), 2),
                n_cylinders: e.rows.len(),
                measured_volume: py_round(fit.measured_volume_fraction, 3),
                measured_length: py_round(fit.measured_length_fraction, 3),
                buttress_m3: e.buttress.map(|b| py_round(b.volume, 5)),
                buttress_top_m: e.buttress.map(|b| py_round(b.top, 2)),
            }
        })
        .collect()
}

/// The table as CSV text, as Python's `csv` module writes it (`\r\n` line
/// ends, floats as `repr`, a missing value empty); just the `tree_id`
/// header for no rows.
pub fn table_csv(rows: &[TableRow]) -> String {
    if rows.is_empty() {
        return "tree_id\r\n".into();
    }
    let mut s = TABLE_COLUMNS.join(",") + "\r\n";
    let opt = |v: Option<f64>| v.map(py_float_repr).unwrap_or_default();
    for r in rows {
        let cells = [
            r.tree_id.to_string(),
            r.points.map(|p| p.to_string()).unwrap_or_default(),
            py_float_repr(r.volume_m3),
            py_float_repr(r.dbh_m),
            py_float_repr(r.height_m),
            r.n_cylinders.to_string(),
            py_float_repr(r.measured_volume),
            py_float_repr(r.measured_length),
            opt(r.buttress_m3),
            opt(r.buttress_top_m),
        ];
        s += &cells.join(",");
        s += "\r\n";
    }
    s
}

/// Write the plot table as CSV ([`table_csv`]).
pub fn write_table_csv(path: impl AsRef<Path>, entries: &[PlotEntry]) -> Result<()> {
    std::fs::write(path, table_csv(&table(entries)))?;
    Ok(())
}

/// Write a surface mesh per tree into `directory` (created if need be), as
/// `<prefix><tree_id>.<fmt>`, in tree order. A tree with a buttress is
/// written joined to it ([`ops::fuse`], wood cut 0.1 m into the base);
/// every other tree is its cylinder mesh. `fmt` is `"ply"` (binary, face
/// colours) or `"obj"` (text, named objects).
pub fn write_meshes(directory: impl AsRef<Path>, entries: &[PlotEntry], fmt: &str, sides: usize, contiguous: bool, prefix: &str) -> Result<Vec<PathBuf>> {
    if fmt != "ply" && fmt != "obj" {
        return Err(Error::invalid("fmt must be 'ply' or 'obj'"));
    }
    let d = directory.as_ref();
    std::fs::create_dir_all(d)?;
    let mut sorted: Vec<&PlotEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| e.tree_id);
    let task = crate::progress::start("writing meshes", sorted.len() as u64);
    let mut out = Vec::with_capacity(sorted.len());
    for e in sorted {
        let path = d.join(format!("{prefix}{}.{fmt}", e.tree_id));
        match &e.buttress {
            Some(b) => {
                let t = ops::fuse(b.vertices, b.faces, b.volume, b.top_z, e.rows, sides, contiguous, 0.1)?;
                if fmt == "ply" {
                    ops::write_tree_mesh_ply(&path, &t.vertices, &t.faces, &t.part, None)?;
                } else {
                    ops::write_tree_mesh_obj(&path, &t.vertices, &t.faces, &t.part)?;
                }
            }
            None if fmt == "ply" => ops::write_model_ply(&path, e.rows, sides, None, contiguous)?,
            None => ops::write_model_obj(&path, e.rows, sides, contiguous)?,
        }
        out.push(path);
        task.inc(1);
    }
    Ok(out)
}

/// Write one cylinder CSV per tree into `directory` (created if need be),
/// as `<prefix><tree_id>.csv`.
pub fn write_cylinders(directory: impl AsRef<Path>, entries: &[PlotEntry], prefix: &str) -> Result<()> {
    let d = directory.as_ref();
    std::fs::create_dir_all(d)?;
    for e in entries {
        qsm_from_rows(e.rows).write_csv(d.join(format!("{prefix}{}.csv", e.tree_id)))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nprandom::Generator;

    fn stem(rng: &mut Generator, cx: f64, r: f64, h: f64, n: usize) -> Vec<Point> {
        (0..n)
            .map(|_| {
                let t = rng.random() * std::f64::consts::TAU;
                let z = rng.random() * h;
                [cx + r * t.cos(), r * t.sin(), z]
            })
            .collect()
    }

    #[test]
    fn rounding_is_pythons() {
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round(0.125, 2), 0.12);
        assert!(py_round(-1e-7, 5).is_sign_negative());
        assert!(py_round(f64::NAN, 2).is_nan());
        assert_eq!(py_round(3e-5, 5), 3e-5);
    }

    #[test]
    fn a_plot_models_its_trees_and_skips_the_small() {
        let mut rng = Generator::new(1);
        let mut pts = stem(&mut rng, 0.0, 0.15, 4.0, 8000);
        let mut labels = vec![1i64; pts.len()];
        pts.extend(stem(&mut rng, 5.0, 0.1, 3.0, 6000));
        labels.resize(pts.len(), 3);
        pts.extend(stem(&mut rng, 9.0, 0.05, 0.4, 100));
        labels.resize(pts.len(), 2);
        pts.push([0.0, 9.0, 0.0]);
        labels.push(-1);
        let h: Vec<f64> = pts.iter().map(|p| p[2]).collect();
        let p = PlotParams { wood: false, min_points: 1000.0, ..Default::default() };
        let plot = build_plot(&pts, &labels, Some(&h), &[(3, [5.0, 0.0], f64::NAN)], &p).unwrap();
        assert_eq!(plot.models.iter().map(|m| m.0).collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(plot.skipped, vec![(2, "100 points".to_string())]);
        assert_eq!(plot.points, vec![(1, 8000), (3, 6000)]);
        assert!(plot.heights[0].1 <= 4.0 && plot.heights[0].1 > 3.9);
        let entries: Vec<PlotEntry> = plot.models.iter().map(|(t, r)| PlotEntry { tree_id: *t, rows: r, points: None, height: None, buttress: None }).collect();
        let rows = table(&entries);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].volume_m3 > rows[1].volume_m3);
        let csv = table_csv(&rows);
        assert!(csv.starts_with("tree_id,points,volume_m3") && csv.contains("\r\n1,,"));
        assert_eq!(table_csv(&[]), "tree_id\r\n");
        assert!((total_volume(&entries) - entries.iter().map(tree_volume).sum::<f64>()).abs() < 1e-12);
        assert!(median_measured_length(plot.models.iter().map(|m| m.1.as_slice())).unwrap() > 0.5);
        assert!(build_plot(&pts, &labels[1..], None, &[], &p).is_err());
        let dir = std::env::temp_dir().join(format!("sylva-plot-{}", std::process::id()));
        assert!(write_meshes(&dir, &entries, "stl", 8, true, "t").is_err());
        let files = write_meshes(&dir, &entries, "obj", 8, true, "t").unwrap();
        assert_eq!(files.iter().map(|f| f.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>(), vec!["t1.obj", "t3.obj"]);
        write_cylinders(&dir, &entries, "c").unwrap();
        assert!(dir.join("c3.csv").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
