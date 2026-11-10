// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The coregistration pipeline: scans and pairs.
//!
//! **Per scan** ([`prepare_scan`]): a terrain model (refitted without the
//! directions a tilted scanner could not see, [`refit_visible_ground`]), the
//! stem map and a planarity-filtered subsample for ICP. **Per pair**
//! ([`register_pair`]): reflective targets where both scans saw them,
//! otherwise a global stem-map match with its height taken from the two
//! terrain models ([`on_ground`]), refined by ICP; the pair is accepted only
//! if it fits over all points and above the ground and its terrain agrees,
//! and a strong target match can be trusted where ICP fails. The whole
//! survey is in [`crate::coreg::survey`].
//!
//! Every decision, message and number follows the NumPy implementation this
//! replaced (`sylva.coreg.pipeline`), including NumPy's arithmetic where it
//! matters: 4x4 products are sequential fused multiply-adds (NumPy hands
//! them to BLAS), heights are compared in float32 as NumPy compares a
//! float32 array with a Python float, and the ICP points are float32.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use nalgebra::Matrix4;

use crate::coreg::matching::{match_stem_maps, MatchParams, StemMap, StemMatch};
use crate::coreg::geometry::{planar_filter, voxel_centroids};
use crate::coreg::ground::{fit_ground, GroundModel, GroundParams};
use crate::coreg::icp::{evaluate_registration, icp_prepared, plane_information, IcpConfig, IcpResult, IcpTarget};
use crate::coreg::reflectors::{match_reflectors, Reflector, ReflectorMatch};
use crate::coreg::stemmap::StemRecord;
use crate::coreg::transforms::{invert, transform_difference, transform_points, transform_vectors, Mat4};
use crate::error::Error;
use crate::util::numeric::median;
use crate::util::pyformat::{fixed, percent, signed, thousands};
use crate::trees::stems::{detect_stems_full, StemParams};
use crate::Point;

/// Result of a pipeline function, failing with the crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

// ----------------------------------------------------------------- arithmetic

/// `a @ b` as NumPy computes a 4x4 product (BLAS: one fused multiply-add
/// after another along the inner dimension).
pub fn matmul(a: &Mat4, b: &Mat4) -> Mat4 {
    Matrix4::from_fn(|i, j| {
        let mut acc = a[(i, 0)] * b[(0, j)];
        for k in 1..4 {
            acc = a[(i, k)].mul_add(b[(k, j)], acc);
        }
        acc
    })
}

/// `np.linalg.norm` of a 3-vector.
fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// `np.degrees`.
fn degrees(x: f64) -> f64 {
    x * (180.0 / std::f64::consts::PI)
}

/// Python's `max(a, b)`.
fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// `values[:end]` with Python's slice bounds.
pub(crate) fn py_head<T: Clone>(values: &[T], end: i64) -> Vec<T> {
    let n = values.len() as i64;
    let stop = if end < 0 { (n + end).max(0) } else { end.min(n) };
    values[..stop as usize].to_vec()
}

/// `str(x)` of a Python float.
pub(crate) fn py_float(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        crate::util::json::py_float_repr(x)
    }
}

/// Python's `repr()` of a string.
pub(crate) fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// `f32` rounding of a point, as NumPy's `astype(np.float32)`.
fn f32_point(p: &Point) -> Point {
    [p[0] as f32 as f64, p[1] as f32 as f64, p[2] as f32 as f64]
}

// --------------------------------------------------------------------- config

/// Reading bounds and RiVLib options (`CoregConfig.riegl_options`).
#[derive(Debug, Clone)]
pub struct ReadingOptions {
    /// RiVLib directory or library file to load for `.rxp`; `None` finds it
    /// the usual way.
    pub library: Option<PathBuf>,
    /// Drop RiVLib's pseudo echoes (flag bit 4) when reading `.rxp`.
    pub drop_pseudo_echoes: bool,
    /// Echo selection for `.rxp`: `"all"`, `"first"`, `"last"` or `"single"`.
    pub echoes: String,
    /// Keep every `stride`-th point (1 = all); splits multi-echo pulses.
    pub stride: usize,
    /// Keep every `shot_stride`-th laser pulse with all its echoes (1 = all).
    pub shot_stride: usize,
    /// `(min, max)` of range (m, from the scanner), deviation, reflectance
    /// and amplitude, in that order; `None` leaves a side open.
    pub bounds: [(Option<f64>, Option<f64>); 4],
    /// Whether any option was given (a reason then says "after filtering").
    pub any: bool,
}

impl Default for ReadingOptions {
    fn default() -> Self {
        ReadingOptions { library: None, drop_pseudo_echoes: true, echoes: "all".into(), stride: 1, shot_stride: 1, bounds: [(None, None); 4], any: false }
    }
}

/// The attributes the reading bounds apply to, in [`ReadingOptions::bounds`] order.
pub const GATES: [&str; 4] = ["range", "deviation", "reflectance", "amplitude"];

/// Settings of the whole pipeline (`sylva.coreg.CoregConfig`).
#[derive(Debug, Clone)]
pub struct CoregConfig {
    /// Cell size (m) of each scan's raster terrain model.
    pub ground_cell_size: f64,
    /// Least fraction (0-1) of the elevations from -30 to +5 degrees an azimuth
    /// of the scan must sample for the terrain in that direction to be fitted
    /// from it ([`refit_visible_ground`]); `None` fits the terrain from every
    /// return. Tilted scans sample almost none along their tilt axis.
    pub ground_min_coverage: Option<f64>,
    /// Stem detection, in the detector's coregistration mode.
    pub stems: StemParams,
    /// Settings of the global stem-map match that gives a pair its coarse transform.
    pub matching: MatchParams,
    /// Settings of the ICP that refines each pair.
    pub icp: IcpConfig,
    /// Voxel size (m) the ICP subsample of each scan is thinned to.
    pub icp_voxel: f64,
    /// Planarity gate (0-1) on the ICP points of every scan; 0 keeps foliage,
    /// which costs accuracy and can trap ICP in a local minimum.
    pub icp_min_planarity: f64,
    /// Returns higher above ground than this (m) are left out of ICP: the upper
    /// canopy is the least repeatable part of the scene and the most moved by wind.
    pub icp_max_height: f64,
    /// Try reflective targets before stems where both scans saw at least
    /// `min_reflector_matches`; far more precise, but they usually cover only
    /// part of a survey.
    pub use_reflectors: bool,
    /// Targets both scans must hold, and a reflector match must bring into
    /// agreement, for it to be attempted and to succeed; never fewer than 3.
    pub min_reflector_matches: i64,
    /// Distance (m) within which a transformed target counts as matched; also
    /// how far ICP may move from a trusted reflector pose before the targets'
    /// own pose is kept instead.
    pub reflector_tolerance: f64,
    /// A reflector match with at least this many targets, agreeing within
    /// `trusted_reflector_rmse`, is accepted even when ICP then fails its
    /// tests; 0 always requires ICP to agree.
    pub trusted_reflector_matches: i64,
    /// RMS target disagreement (m) a reflector match must stay under to be trusted.
    pub trusted_reflector_rmse: f64,
    /// Stem correspondences a coarse match needs to reach ICP; permissive,
    /// since ICP fitness decides.
    pub min_match_inliers: i64,
    /// A coarse match whose best distinctly different rival has more than this
    /// share (0-1) of its inliers is ambiguous: both are refined by ICP and the
    /// pair is accepted only if one fits the above-ground points clearly better.
    pub max_match_ambiguity: f64,
    /// Factor by which the winner's above-ground fitness must exceed its
    /// rival's for an ambiguous match to be accepted.
    pub ambiguity_margin: f64,
    /// Largest RMS scatter (m) of a coarse match's own stem correspondences.
    pub max_match_rmse: f64,
    /// Cap (m) on the median horizontal disagreement of the matched stems under
    /// the coarse transform; infinite (the default) disables it.
    pub max_coarse_stem_rmse: f64,
    /// Take a stem match's vertical offset from the two terrain models where
    /// both saw ground rather than from the matched stems, whose heights rest
    /// on the terrain exactly where it is least certain.
    pub height_from_ground: bool,
    /// Terrain cells further than this (m) from a scanner are left out of
    /// height comparisons: there ground returns are sparse and grazing and the
    /// model is mostly understorey.
    pub ground_radius: f64,
    /// Shared observed terrain cells a height comparison needs.
    pub min_ground_cells: i64,
    /// A pair whose terrain models still differ by more than this (m, median
    /// over the shared ground) after ICP is refused; `None` disables the test.
    pub max_ground_disagreement: Option<f64>,
    /// Pairs whose approximate scanner positions are further apart (m) are not tried.
    pub max_pair_distance: f64,
    /// Match every pair's stems first and run ICP only on the survivors.
    pub screen_pairs: bool,
    /// 0 keeps every pair (Python's None and 0 alike).
    pub max_pairs_per_scan: i64,
    /// Least ICP fitness (0-1, over all source points) of an accepted pair; it
    /// falls with the distance between scans as less of each is shared.
    pub min_icp_fitness: f64,
    /// Largest ICP inlier RMSE (m) of an accepted pair.
    pub max_icp_rmse: f64,
    /// Least fitness (0-1) over source points higher than `fitness_min_height`:
    /// ground is planar and fits under any horizontal shift, so stems, branches
    /// and logs must agree too.
    pub min_icp_fitness_above_ground: f64,
    /// Height above ground (m) from which points count in the above-ground
    /// fitness; <= 0 skips that test.
    pub fitness_min_height: f64,
    /// If ICP leaves the matched stems this much (m) further apart than the
    /// coarse match did, the coarse transform is kept: ICP slid somewhere unrelated.
    pub stem_agreement_tolerance: f64,
    /// ICP moving further than this (m) from the coarse solution has diverged,
    /// and the pair is refused.
    pub max_coarse_to_fine_shift: f64,
    /// Retry each scan left unregistered against the combined registered
    /// survey: it can overlap several registered scans partly without
    /// overlapping any one of them enough.
    pub recover_unregistered: bool,
    /// Passes of that recovery, each able to tie in scans the previous one
    /// registered; treated as at least 1.
    pub recovery_rounds: i64,
    /// Nearest registered scans whose points are merged into the target a
    /// recovery attempt is registered against.
    pub recovery_neighbours: i64,
    /// Refine every registered pose in one joint Gauss-Newton over the
    /// point-to-plane correspondences of every accepted pair plus the matched
    /// stems ([`crate::coreg::refine::refine_joint`]).
    pub refine_multiview: bool,
    /// Associations per level of that refinement; treated as at least 1.
    pub refinement_rounds: i64,
    /// Voxel sizes (m) of its levels, coarse to fine.
    pub refinement_voxel_sizes: Vec<f64>,
    /// Correspondence cut-off (m) at each of those levels.
    pub refinement_max_distances: Vec<f64>,
    /// Cap on the points of each scan at each level, a seeded random subset beyond it.
    pub refinement_points_per_scan: usize,
    /// Share (0-1) of each pair's weight carried by its matched stems.
    pub refinement_stem_weight: f64,
    /// Horizontal radius (m) within which mutually nearest stems of a pair are
    /// tied together.
    pub refinement_stem_radius: f64,
    /// A voxel needs at least this many points for its centroid to take part.
    pub refinement_min_voxel_points: usize,
    /// A refinement moving any scan further than this (m) is discarded.
    pub refinement_max_shift: f64,
    /// Solve all poses together over the pose graph of accepted pairs; off, the
    /// poses stay as the spanning tree over the edges placed them.
    pub optimise_globally: bool,
    /// Index of the scan whose levelled frame is the world frame, held fixed;
    /// clamped into range, and replaced if that scan was set aside or no
    /// accepted pair touches it.
    pub reference_scan: i64,
    /// Drop accepted pairs whose measured transform the rest of the graph
    /// contradicts (solve-and-reject passes).
    pub reject_outlier_edges: bool,
    /// ICP correspondences counted as one independent observation when
    /// weighting pose-graph edges: neighbouring residuals share the error of
    /// the surface they sample.
    pub information_patch_points: f64,
    /// Floor (m) on the point-to-plane residual that weights an edge.
    pub information_min_sigma: f64,
    /// With priors: results putting a scanner further than this (m) from its
    /// prior position are refused.
    pub max_prior_shift: f64,
    /// With priors: if set, results rotated more than this (degrees) from the
    /// prior are refused too; `None` by default, since recorded headings can
    /// be wrong by any amount.
    pub max_prior_rotation: Option<f64>,
    /// Scans prepared and pairs registered at once; 0 picks from the cores
    /// and free memory.
    pub workers: i64,
    /// 0 leaves memory out of the choice of workers.
    pub memory_per_worker_gb: f64,
    /// RiSCAN PRO's RXP import filter applied when reading
    /// ([`crate::riscan::riscan_like_mask`]): `"none"`, `"current"` (drops
    /// echoes within 0.5 m of the scanner) or `"legacy"` (also the weak,
    /// isolated echoes the older conversion discarded).
    pub riscan_filter: String,
    /// Bounds and RiVLib options applied while reading each scan file.
    pub reading: ReadingOptions,
    /// Scans with fewer points (after the reading filters) are set aside
    /// rather than processed.
    pub min_points_per_scan: i64,
    /// Most points a scan is prepared from, after the reading filters; a
    /// larger scan is thinned by an even stride through its recording order
    /// ([`stride_subsample`]), so the whole rotation stays covered. 0 reads
    /// every point (Python's None and 0 alike). Only preparation is capped:
    /// [`crate::coreg::survey::merge_clouds`] always merges every point.
    pub max_points_per_scan: usize,
}

impl Default for CoregConfig {
    /// The settings a survey runs with unless the caller changes them.
    ///
    /// These are the values the published interface documents, chosen on TLS
    /// plots of the kind the package was written for; a caller overrides the
    /// few it cares about and leaves the rest - `..Default::default()`. They
    /// live here rather than in the Python wrapper so the R bindings and the
    /// command line get the same run from the same request.
    fn default() -> Self {
        CoregConfig {
            ground_cell_size: 0.5,
            ground_min_coverage: Some(0.8),
            stems: StemParams::coreg(),
            matching: MatchParams::default(),
            icp: IcpConfig::default(),
            icp_voxel: 0.05,
            icp_min_planarity: 0.35,
            icp_max_height: 12.0,
            use_reflectors: true,
            min_reflector_matches: 3,
            reflector_tolerance: 0.05,
            trusted_reflector_matches: 5,
            trusted_reflector_rmse: 0.03,
            min_match_inliers: 5,
            max_match_ambiguity: 0.8,
            ambiguity_margin: 1.25,
            max_match_rmse: 0.30,
            max_coarse_stem_rmse: f64::INFINITY,
            height_from_ground: true,
            ground_radius: 30.0,
            min_ground_cells: 50,
            max_ground_disagreement: Some(0.25),
            max_pair_distance: 40.0,
            screen_pairs: true,
            max_pairs_per_scan: 0,
            min_icp_fitness: 0.04,
            max_icp_rmse: 0.15,
            min_icp_fitness_above_ground: 0.03,
            fitness_min_height: 1.0,
            stem_agreement_tolerance: 0.25,
            max_coarse_to_fine_shift: 2.0,
            recover_unregistered: true,
            recovery_rounds: 2,
            recovery_neighbours: 6,
            refine_multiview: false,
            refinement_rounds: 3,
            refinement_voxel_sizes: vec![0.10, 0.05, 0.03],
            refinement_max_distances: vec![0.30, 0.15, 0.08],
            refinement_points_per_scan: 400_000,
            refinement_stem_weight: 0.05,
            refinement_stem_radius: 0.15,
            refinement_min_voxel_points: 1,
            refinement_max_shift: 0.30,
            optimise_globally: true,
            reference_scan: 0,
            reject_outlier_edges: true,
            information_patch_points: 100.0,
            information_min_sigma: 0.005,
            max_prior_shift: 5.0,
            max_prior_rotation: None,
            workers: 0,
            memory_per_worker_gb: 4.0,
            riscan_filter: "none".into(),
            reading: ReadingOptions::default(),
            min_points_per_scan: 1000,
            max_points_per_scan: 0,
        }
    }
}

// ---------------------------------------------------------------------- scans

/// Everything later stages need from one scan (`ScanFeatures`).
#[derive(Debug, Clone)]
pub struct ScanFeatures {
    /// Scan position name (the file's stem when none was given for a path).
    pub name: String,
    /// Points the scan was prepared from, after the reading filters.
    pub n_points: usize,
    /// Raster terrain model of the scan in its levelled frame; `None` for an
    /// unusable scan.
    pub ground: Option<GroundModel>,
    /// Stems, best first.
    pub stems: Vec<StemRecord>,
    /// Name the stem map is saved under (the scan's name).
    pub stem_map_name: String,
    /// Planar subsample for ICP; float32 values held as f64.
    pub icp_points: Vec<Point>,
    /// Retro-reflective targets seen from this scan, in its levelled frame.
    pub reflectors: Vec<Reflector>,
    /// Height above ground of each ICP point.
    pub icp_heights: Vec<f32>,
    /// `level_from_scan` applied to the raw points first.
    pub levelling: Mat4,
    /// Scanner position in the levelled frame.
    pub origin: Point,
    /// Path of the file the scan was read from; `None` for points given directly.
    pub source: Option<String>,
    /// Wall-clock time (s) spent preparing the scan.
    pub seconds: f64,
    /// Why the scan is unusable; empty if it is fine.
    pub error: String,
}

impl ScanFeatures {
    /// Can the scan take part in registration at all?
    pub fn usable(&self) -> bool {
        self.error.is_empty() && !self.icp_points.is_empty()
    }

    /// Where the scanner stood, given this scan's `world_from_scan`.
    pub fn location(&self, pose: &Mat4) -> Point {
        transform_points(pose, &[self.origin])[0]
    }

    /// Stem positions, `(x, y, z)` at the reference height.
    pub fn stem_positions(&self) -> Vec<Point> {
        self.stems.iter().map(|s| [s.x, s.y, s.z]).collect()
    }

    /// The stem map as the matcher takes it.
    pub fn stem_map(&self) -> StemMap {
        stem_map(&self.stems)
    }
}

/// Positions, diameters and qualities of stems, for [`match_stem_maps`].
pub fn stem_map(stems: &[StemRecord]) -> StemMap {
    StemMap { positions: stems.iter().map(|s| [s.x, s.y, s.z]).collect(), diameters: stems.iter().map(|s| s.dbh).collect(), qualities: stems.iter().map(|s| s.quality()).collect() }
}

/// Stems moved by a rigid transform (`StemMap.transformed`).
pub fn transform_stems(stems: &[StemRecord], t: &Mat4) -> Vec<StemRecord> {
    let pos = transform_points(t, &stems.iter().map(|s| [s.x, s.y, s.z]).collect::<Vec<_>>());
    let ax = transform_vectors(t, &stems.iter().map(|s| s.axis).collect::<Vec<_>>());
    stems.iter().zip(pos).zip(ax).map(|((s, p), a)| StemRecord { x: p[0], y: p[1], z: p[2], axis: a, ..s.clone() }).collect()
}

/// What a scan is prepared from.
#[derive(Debug, Clone)]
pub enum ScanInput {
    /// A file: `.rxp` through RiVLib, anything else through [`crate::io::read`].
    Path(PathBuf),
    /// Points in the scan's own frame.
    Points(Vec<Point>),
    /// An input that could not even be converted: set aside with this reason.
    Failed {
        /// Why the input could not be converted.
        reason: String,
        /// Path of the file it came from, if any.
        source: Option<PathBuf>,
    },
}

/// Why a scan could not be read or prepared.
#[derive(Debug)]
pub enum PrepareError {
    /// A reading bound or filter needs an attribute the file does not carry
    /// (the Python package's `KeyError`).
    MissingAttribute(String),
    /// Any other error of the core library (reading the file, fitting the
    /// ground, ...).
    Core(Error),
}

impl From<Error> for PrepareError {
    fn from(e: Error) -> Self {
        PrepareError::Core(e)
    }
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrepareError::MissingAttribute(m) => f.write_str(m),
            PrepareError::Core(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PrepareError {}

impl PrepareError {
    /// `f"{type(exc).__name__}: {exc}"` of the exception the Python package raises.
    pub fn describe(&self) -> String {
        match self {
            PrepareError::MissingAttribute(m) => format!("KeyError: {}", py_repr(m)),
            PrepareError::Core(e) => format!("{}: {e}", python_exception(e)),
        }
    }
}

/// The Python exception a core error is raised as (the bindings' mapping).
pub fn python_exception(e: &Error) -> &'static str {
    match e {
        Error::Io(_) | Error::File { .. } | Error::Las(_) => "OSError",
        _ => "ValueError",
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Python's `Path.stem`.
pub fn path_stem(path: &Path) -> String {
    let name = file_name(path);
    match name.rfind('.') {
        Some(k) if k > 0 && k + 1 < name.len() => name[..k].to_string(),
        _ => name,
    }
}

/// Points of a scan file in its own frame, filtered for coregistration
/// (`_read_scan`): RiSCAN's import filter on the whole stream first, then
/// the closed intervals on range (from the scanner), deviation, reflectance
/// and amplitude, then the cap on the number of points
/// (`max_points_per_scan`, an even stride through the file: [`stride_subsample`]).
pub fn read_scan(path: &Path, cfg: &CoregConfig) -> std::result::Result<Vec<Point>, PrepareError> {
    let is_rxp = path.extension().map(|e| e.to_string_lossy().to_lowercase() == "rxp").unwrap_or(false);
    let cloud = if is_rxp {
        let o = &cfg.reading;
        let opts = crate::io::riegl::RxpOptions { library: o.library.clone(), drop_pseudo_echoes: o.drop_pseudo_echoes, min_range: 0.0, max_range: f64::INFINITY, stride: o.stride.max(1), shot_stride: o.shot_stride.max(1), max_points: None, echoes: o.echoes.clone(),
            // coregistration reads many scans thinned by stride; plain zone-1 reading keeps it fast
            mta: crate::io::rxp_mta::MtaOptions { mode: crate::io::rxp_mta::MtaMode::Off, ..Default::default() }, ..Default::default() };
        crate::io::riegl::read_rxp(path, &opts)?
    } else {
        crate::io::read(path)?
    };
    let xyz = cloud.xyz;
    let mut keep = vec![true; xyz.len()];
    if cfg.riscan_filter != "none" {
        let Some(amplitude) = cloud.attrs.get("amplitude") else {
            return Err(PrepareError::MissingAttribute(format!("the RiSCAN filter needs amplitude, which {} does not carry", file_name(path))));
        };
        let p: Vec<[f32; 3]> = xyz.iter().map(|q| [q[0] as f32, q[1] as f32, q[2] as f32]).collect();
        let a: Vec<f32> = amplitude.to_f64().into_iter().map(|v| v as f32).collect();
        let mask = crate::riscan::riscan_like_mask(&p, &a, &cfg.riscan_filter, 0.5, &crate::riscan::LegacyFilter::default())?;
        for (k, m) in keep.iter_mut().zip(mask) {
            *k &= m;
        }
    }
    for (g, name) in GATES.iter().enumerate() {
        let (lo, hi) = cfg.reading.bounds[g];
        if lo.is_none() && hi.is_none() {
            continue;
        }
        let (values, lo, hi) = if *name == "range" {
            (xyz.iter().map(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).collect::<Vec<f64>>(), lo.map(|v| v * v), hi.map(|v| v * v))
        } else if let Some(a) = cloud.attrs.get(*name) {
            let mut v = a.to_f64();
            if *name == "deviation" {
                for x in v.iter_mut() {
                    if *x == 65535.0 {
                        *x = -1.0; // RIEGL's "not measured"
                    }
                }
            }
            (v, lo, hi)
        } else {
            return Err(PrepareError::MissingAttribute(format!("reading bounds {name}, which {} does not carry", file_name(path))));
        };
        for (k, v) in keep.iter_mut().zip(values) {
            if let Some(lo) = lo {
                *k &= v >= lo;
            }
            if let Some(hi) = hi {
                *k &= v <= hi;
            }
        }
    }
    let mut out: Vec<Point> = xyz.into_iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| p).collect();
    if cfg.max_points_per_scan > 0 && out.len() > cfg.max_points_per_scan {
        out = stride_subsample(&out, cfg.max_points_per_scan);
    }
    Ok(out)
}

/// `count` of `points` spread evenly through their order (point
/// `floor(i * n / count)` for `i` in `0..count`), or all of them if there are
/// no more than `count`.
///
/// A scan file is in recording order, so its first `count` points are one
/// wedge of the rotation; an even stride keeps the whole field of view at a
/// lower density, and is the same every run.
pub fn stride_subsample(points: &[Point], count: usize) -> Vec<Point> {
    let n = points.len();
    if n <= count {
        return points.to_vec();
    }
    (0..count).map(|i| points[(i as u128 * n as u128 / count as u128) as usize]).collect()
}

fn ground_params(cfg: &CoregConfig) -> GroundParams {
    GroundParams { cell_size: cfg.ground_cell_size, ..GroundParams::default() }
}

/// Heights above ground of points, as float32 (`GroundModel.normalise(..., dtype=float32)`).
pub fn normalise_f32(ground: &GroundModel, points: &[Point]) -> Vec<f32> {
    let xy: Vec<[f64; 2]> = points.iter().map(|p| [p[0], p[1]]).collect();
    let h = ground.height_at(&xy);
    points.iter().zip(h).map(|(p, g)| (p[2] - g) as f32).collect()
}

/// A placeholder for a scan that cannot be registered (`_unusable_scan`).
pub fn unusable_scan(name: &str, n_points: usize, source: Option<String>, start: Instant, reason: &str) -> ScanFeatures {
    ScanFeatures {
        name: name.into(),
        n_points,
        ground: None,
        stems: Vec::new(),
        stem_map_name: name.into(),
        icp_points: Vec::new(),
        reflectors: Vec::new(),
        icp_heights: Vec::new(),
        levelling: Matrix4::identity(),
        origin: [0.0; 3],
        source,
        seconds: start.elapsed().as_secs_f64(),
        error: reason.into(),
    }
}

/// Stems of one scan, best first, with `z` on the scan's own terrain
/// (`sylva.coreg.detect_stems` with heights given).
pub fn detect_scan_stems(points: &[Point], ground: &GroundModel, heights: &[f64], p: &StemParams) -> Vec<StemRecord> {
    if points.len() < 100 {
        return Vec::new();
    }
    let found = detect_stems_full(points, heights, p);
    let xy: Vec<[f64; 2]> = found.iter().map(|s| [s.tree.x, s.tree.y]).collect();
    let terrain = ground.height_at(&xy);
    let mut stems: Vec<StemRecord> = found
        .iter()
        .zip(terrain)
        .map(|(s, z0)| StemRecord { x: s.tree.x, y: s.tree.y, z: z0 + p.reference_height, dbh: s.tree.dbh, axis: s.axis, reference_height: p.reference_height, n_slices: s.tree.n_slices as i64, n_points: s.tree.n_points as i64, rmse: s.tree.rmse, coverage: s.coverage, lean_deg: s.tree.lean_deg })
        .collect();
    let q: Vec<f64> = stems.iter().map(|s| -s.quality() + 0.0).collect();
    let mut order: Vec<usize> = (0..stems.len()).collect();
    order.sort_by(|&a, &b| q[a].total_cmp(&q[b]));
    let sorted = order.iter().map(|&k| stems[k].clone()).collect();
    stems = sorted;
    stems
}

/// Fit the ground, detect stems and build the ICP subsample of one scan.
///
/// `name` empty takes the file's stem for a path. `levelling` rotates the
/// points and targets first; `origin` is the scanner's position in the
/// levelled cloud (the origin if `None`), and with it, or for a scan read
/// from file, the terrain is refitted without the directions the scanner
/// could not see.
pub fn prepare_scan(input: &ScanInput, cfg: &CoregConfig, name: &str, reflectors: &[Reflector], levelling: Option<&Mat4>, origin: Option<Point>) -> std::result::Result<ScanFeatures, PrepareError> {
    let start = Instant::now();
    let level = levelling.copied().unwrap_or_else(Matrix4::identity);
    let mut name = name.to_string();
    let (mut points, source) = match input {
        ScanInput::Path(p) => {
            let pts = read_scan(p, cfg)?;
            if name.is_empty() {
                name = path_stem(p);
            }
            (pts, Some(p.to_string_lossy().into_owned()))
        }
        ScanInput::Points(p) => (p.clone(), None),
        ScanInput::Failed { reason, source } => return Ok(unusable_scan(&name, 0, source.as_ref().map(|s| s.to_string_lossy().into_owned()), start, reason)),
    };
    let mut reflectors = reflectors.to_vec();
    if levelling.is_some() {
        points = transform_points(&level, &points);
        for r in reflectors.iter_mut() {
            let p = transform_points(&level, &[r.position()])[0];
            (r.x, r.y, r.z) = (p[0], p[1], p[2]);
        }
    }
    let scanner = origin.unwrap_or([0.0; 3]);
    if (points.len() as i64) < cfg.min_points_per_scan {
        let reason = format!("only {} points{}", thousands(points.len() as i64), if cfg.reading.any { " after filtering" } else { "" });
        return Ok(unusable_scan(&name, points.len(), source, start, &reason));
    }
    let mut ground = fit_ground(&points, &ground_params(cfg))?;
    if cfg.ground_min_coverage.is_some() && (source.is_some() || origin.is_some()) {
        if let Some(g) = refit_visible_ground(&points, scanner, cfg)? {
            ground = g;
        }
    }
    let heights = normalise_f32(&ground, &points);
    let heights64: Vec<f64> = heights.iter().map(|&h| h as f64).collect();
    let stems = detect_scan_stems(&points, &ground, &heights64, &cfg.stems);
    let top = cfg.icp_max_height as f32;
    let below: Vec<Point> = points.iter().zip(&heights).filter(|(_, &h)| h <= top).map(|(p, _)| *p).collect();
    let icp_points: Vec<Point> = if cfg.icp_min_planarity > 0.0 {
        if cfg.icp_voxel < 0.0 {
            return Err(Error::invalid("voxel size must be positive").into());
        }
        planar_filter(&below, cfg.icp_min_planarity, Some(cfg.icp_voxel), 20, Some(0.15))
    } else {
        if cfg.icp_voxel.is_nan() || cfg.icp_voxel <= 0.0 {
            return Err(Error::invalid("voxel size must be positive").into());
        }
        voxel_centroids(&below, cfg.icp_voxel, true).0
    };
    let icp_points: Vec<Point> = icp_points.iter().map(f32_point).collect();
    let icp_heights = if icp_points.is_empty() { Vec::new() } else { normalise_f32(&ground, &icp_points) };
    Ok(ScanFeatures { name: name.clone(), n_points: points.len(), ground: Some(ground), stems, stem_map_name: name, icp_points, reflectors, icp_heights, levelling: level, origin: scanner, source, seconds: start.elapsed().as_secs_f64(), error: String::new() })
}

/// Fewest returns the terrain refit of [`refit_visible_ground`] is made from;
/// with fewer left (the returns steeper than 45 degrees below the scanner and
/// those at sighted azimuths), the first fit stands.
const REFIT_MIN_RETURNS: usize = 100;

/// The terrain refitted without the directions in which the scanner could
/// not see it (`_refit_visible_ground`), or `None` where every azimuth, or
/// none, is blind, or fewer than 100 returns would be left
/// to fit, and the fit stands.
///
/// An azimuth (1 degree) is blind when the scan sampled less than
/// `ground_min_coverage` of the elevations from -30 to +5 degrees in it;
/// blind azimuths, widened by 3 degrees either side, keep only their returns
/// more than 45 degrees below the horizon.
pub fn refit_visible_ground(points: &[Point], scanner: Point, cfg: &CoregConfig) -> Result<Option<GroundModel>> {
    let Some(min_coverage) = cfg.ground_min_coverage else { return Ok(None) };
    let n = points.len();
    let mut azimuth = vec![0usize; n];
    let mut elevation = vec![0.0f64; n];
    let mut sampled = vec![[false; 35]; 360];
    for (k, p) in points.iter().enumerate() {
        let rel = [p[0] - scanner[0], p[1] - scanner[1], p[2] - scanner[2]];
        let horizontal = rel[0].hypot(rel[1]);
        let a = (degrees(rel[1].atan2(rel[0])) + 180.0).floor() as i64;
        azimuth[k] = a.rem_euclid(360) as usize;
        elevation[k] = degrees(rel[2].atan2(horizontal));
        if elevation[k] >= -30.0 && elevation[k] < 5.0 {
            sampled[azimuth[k]][(elevation[k] + 30.0).floor() as usize] = true;
        }
    }
    let blind: Vec<bool> = sampled.iter().map(|row| (row.iter().filter(|&&s| s).count() as f64 / 35.0) < min_coverage).collect();
    if !blind.iter().any(|&b| b) || blind.iter().all(|&b| b) {
        return Ok(None);
    }
    let widen = 3i64;
    let widened: Vec<bool> = (0..360i64).map(|a| (-widen..=widen).any(|d| blind[(a + d).rem_euclid(360) as usize])).collect();
    let kept: Vec<Point> = (0..n).filter(|&k| !widened[azimuth[k]] || elevation[k] < -45.0).map(|k| points[k]).collect();
    // Scattered blind azimuths, once widened, can cover the whole circle and
    // leave nothing (an upright scanner sees nothing 45 degrees down): a
    // terrain cannot be fitted to that, so the fit from every return stands.
    if kept.len() < REFIT_MIN_RETURNS {
        return Ok(None);
    }
    Ok(Some(fit_ground(&kept, &ground_params(cfg))?))
}

// ------------------------------------------------------------------- terrain

/// Observed terrain cells within `radius` of the scanner, in the scan's
/// frame, as `(x, y, height)` (`_terrain_samples`).
pub fn terrain_samples(scan: &ScanFeatures, radius: f64) -> Vec<Point> {
    let Some(g) = &scan.ground else { return Vec::new() };
    let mut xy = Vec::new();
    for iy in 0..g.ny {
        for ix in 0..g.nx {
            if g.observed[iy * g.nx + ix] {
                let p = [g.origin[0] + ix as f64 * g.cell_size, g.origin[1] + iy as f64 * g.cell_size];
                if (p[0] - scan.origin[0]).hypot(p[1] - scan.origin[1]) <= radius {
                    xy.push(p);
                }
            }
        }
    }
    let h = g.height_at(&xy);
    xy.iter().zip(h).map(|(p, z)| [p[0], p[1], z]).collect()
}

/// Median height of the targets' terrain over the source's where both saw
/// ground (`_height_offset`): metres to add to the source's height, NaN
/// with fewer than `min_ground_cells` shared cells.
///
/// `targets` are `(scan, world_from_scan)` of the scans compared against,
/// `world_from_source` the source's pose (or the pair transform, the
/// target's frame being the world).
pub fn height_offset(source: &ScanFeatures, world_from_source: &Mat4, targets: &[(&ScanFeatures, Mat4)], cfg: &CoregConfig) -> f64 {
    let samples = terrain_samples(source, cfg.ground_radius);
    if samples.is_empty() {
        return f64::NAN;
    }
    let world = transform_points(world_from_source, &samples);
    let mut offsets = Vec::new();
    for (scan, pose) in targets {
        let Some(g) = &scan.ground else { continue };
        let local = transform_points(&invert(pose), &world);
        let (rows, cols) = (g.ny as f64, g.nx as f64);
        let inside: Vec<&Point> = local
            .iter()
            .filter(|p| {
                let (cx, cy) = ((p[0] - g.origin[0]) / g.cell_size, (p[1] - g.origin[1]) / g.cell_size);
                cx >= -0.5 && cx <= cols - 0.5 && cy >= -0.5 && cy <= rows - 0.5 && (p[0] - scan.origin[0]).hypot(p[1] - scan.origin[1]) <= cfg.ground_radius
            })
            .collect();
        let support = g.support(&inside.iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>());
        let shared: Vec<&Point> = inside.into_iter().zip(support).filter(|(_, s)| *s).map(|(p, _)| p).collect();
        let z: Vec<f64> = shared.iter().map(|p| p[2]).collect();
        let shared: Vec<[f64; 2]> = shared.iter().map(|p| [p[0], p[1]]).collect();
        let h = g.height_at(&shared);
        offsets.extend(h.iter().zip(&z).map(|(h, z)| h - z));
    }
    if (offsets.len() as i64) < cfg.min_ground_cells.max(1) {
        return f64::NAN;
    }
    median(&offsets)
}

/// `transform` with its height set from the terrain (`_on_ground`), when
/// `height_from_ground` and the scans share enough ground.
pub fn on_ground(transform: &Mat4, source: &ScanFeatures, targets: &[(&ScanFeatures, Mat4)], cfg: &CoregConfig) -> Mat4 {
    if !cfg.height_from_ground {
        return *transform;
    }
    let dz = height_offset(source, transform, targets, cfg);
    if !dz.is_finite() {
        return *transform;
    }
    let mut out = *transform;
    out[(2, 3)] += dz;
    out
}

/// Does the terrain disagree beyond `max_ground_disagreement`?
pub fn ground_disagrees(offset: f64, cfg: &CoregConfig) -> bool {
    cfg.max_ground_disagreement.is_some_and(|m| offset.is_finite() && offset.abs() > m)
}

/// Median horizontal distance between matched stems under `transform`; NaN
/// without matches (`_stem_median_residual`).
pub fn stem_median_residual(transform: &Mat4, matched_source: &[Point], matched_target: &[Point]) -> f64 {
    if matched_source.is_empty() {
        return f64::NAN;
    }
    let moved = transform_points(transform, matched_source);
    let d: Vec<f64> = moved
        .iter()
        .zip(matched_target)
        .map(|(p, q)| {
            let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    median(&d)
}

/// ICP fitness over source points more than `fitness_min_height` up; NaN
/// ("no evidence") if heights are missing or fewer than 100 points qualify
/// (`_above_ground_fitness`).
pub fn above_ground_fitness(source: &[Point], heights: &[f32], target: &[Point], transform: &Mat4, cfg: &CoregConfig) -> f64 {
    if cfg.fitness_min_height <= 0.0 || heights.len() != source.len() {
        return f64::NAN;
    }
    let bar = cfg.fitness_min_height as f32;
    let above: Vec<Point> = source.iter().zip(heights).filter(|(_, &h)| h > bar).map(|(p, _)| *p).collect();
    if above.len() < 100 {
        return f64::NAN;
    }
    evaluate_registration(&above, target, transform, cfg.icp.fitness_threshold, 200_000, Some(0.05), 0).0
}

/// Does `pose` put the scanner where its prior says it stood (`_prior_ok`)?
/// `(false, why)` if it is more than `max_prior_shift` away, or rotated more
/// than `max_prior_rotation` degrees from it.
pub fn prior_ok(pose: &Mat4, prior: &Mat4, origin: Point, cfg: &CoregConfig) -> (bool, String) {
    let (a, b) = (transform_points(pose, &[origin])[0], transform_points(prior, &[origin])[0]);
    let shift = norm3([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
    if shift > cfg.max_prior_shift {
        return (false, format!("scanner {} m from its prior position", fixed(shift, 1)));
    }
    if let Some(limit) = cfg.max_prior_rotation {
        let xi = crate::coreg::transforms::se3_log(&matmul(&invert(prior), pose));
        let rot = degrees(norm3([xi[0], xi[1], xi[2]]));
        if rot > limit {
            return (false, format!("{} deg from the prior orientation", fixed(rot, 1)));
        }
    }
    (true, String::new())
}

// ---------------------------------------------------------------------- pairs

/// Registration of one pair (`PairResult`); `transform` maps scan `i` into
/// scan `j`.
#[derive(Debug, Clone)]
pub struct PairResult {
    /// Index of the source scan, whose points `transform` moves.
    pub i: i64,
    /// Index of the target scan.
    pub j: i64,
    /// Name of scan `i`.
    pub name_i: String,
    /// Name of scan `j`.
    pub name_j: String,
    /// `j_from_i`, the transform the pair reports: ICP's, or the coarse one
    /// when `used_icp` is false; identity until something was tried.
    pub transform: Mat4,
    /// The transform ICP started from: the reflector match's, or the stem
    /// match's with its height set from the two terrain models.
    pub coarse_transform: Mat4,
    /// Global stem-map match behind the coarse transform; `None` when targets
    /// were used or stems were never matched.
    pub stem_match: Option<StemMatch>,
    /// Alignment of the shared retro-reflective targets, where both scans had
    /// enough and it succeeded; else `None`.
    pub reflector_match: Option<ReflectorMatch>,
    /// ICP outcome; when `used_icp` is false it scores the coarse transform,
    /// not the refinement that was discarded.
    pub icp: Option<IcpResult>,
    /// Whether the pair passed every test and becomes a pose-graph edge.
    pub success: bool,
    /// Why the pair failed, or how it was accepted; empty while undecided.
    pub reason: String,
    /// Wall-clock time (s) spent registering the pair.
    pub seconds: f64,
    /// Positions of the matched stems in each scan's own frame.
    pub matched_source: Vec<Point>,
    /// The same stems in the target scan's frame, row for row with `matched_source`.
    pub matched_target: Vec<Point>,
    /// Median horizontal distance (m) between matched stems under the coarse
    /// transform; NaN until known.
    pub coarse_stem_rmse: f64,
    /// The same under the ICP transform; NaN if ICP did not run or fewer than
    /// 3 stems matched.
    pub fine_stem_rmse: f64,
    /// ICP fitness over source points above ground; NaN if unknown.
    pub fitness_above: f64,
    /// A distinctly different coarse hypothesis nearly as well supported
    /// (`ambiguity` above `max_match_ambiguity`), resolved by ICP; else `None`.
    pub rival: Option<StemMatch>,
    /// False when the coarse transform was kept over ICP's.
    pub used_icp: bool,
    /// Median height (m) of the target's terrain over the source's where both
    /// saw ground, under `transform`; NaN if they share too little.
    pub ground_offset: f64,
    /// Accepted on its reflector match although ICP failed.
    pub trusted: bool,
}

impl PairResult {
    /// An undecided pair of scans `i` and `j`: identity transforms, no match,
    /// NaN statistics.
    pub fn new(i: i64, j: i64, name_i: &str, name_j: &str) -> Self {
        PairResult {
            i,
            j,
            name_i: name_i.into(),
            name_j: name_j.into(),
            transform: Matrix4::identity(),
            coarse_transform: Matrix4::identity(),
            stem_match: None,
            reflector_match: None,
            icp: None,
            success: false,
            reason: String::new(),
            seconds: 0.0,
            matched_source: Vec::new(),
            matched_target: Vec::new(),
            coarse_stem_rmse: f64::NAN,
            fine_stem_rmse: f64::NAN,
            fitness_above: f64::NAN,
            rival: None,
            used_icp: true,
            ground_offset: f64::NAN,
            trusted: false,
        }
    }

    /// ICP fitness (0-1) of the reported transform; 0 when ICP did not run.
    pub fn fitness(&self) -> f64 {
        self.icp.as_ref().map_or(0.0, |r| r.fitness)
    }

    /// ICP inlier RMSE (m); infinite when ICP did not run.
    pub fn rmse(&self) -> f64 {
        self.icp.as_ref().map_or(f64::INFINITY, |r| r.inlier_rmse)
    }

    /// Stems the coarse stem match paired; 0 without one.
    pub fn n_stem_matches(&self) -> usize {
        self.stem_match.as_ref().map_or(0, |m| m.n_inliers)
    }

    /// Stem agreement of the transform this pair reports.
    pub fn stem_rmse(&self) -> f64 {
        if self.used_icp {
            self.fine_stem_rmse
        } else {
            self.coarse_stem_rmse
        }
    }

    /// One line for the report (`PairResult.summary`).
    pub fn summary(&self) -> String {
        let status = if self.success { "ok  " } else { "FAIL" };
        let stem = match &self.reflector_match {
            Some(r) => format!(" targets={}", r.n_inliers),
            None if self.stem_rmse().is_finite() => format!(" stem={}cm", fixed(self.stem_rmse() * 100.0, 1)),
            None => String::new(),
        };
        let mut s = format!("[{status}] {} -> {}: stems={:2} ", self.name_i, self.name_j, self.n_stem_matches());
        if let Some(m) = self.stem_match.as_ref().filter(|m| m.ambiguity > 0.0) {
            s += &format!("amb={} ", fixed(m.ambiguity, 2));
        }
        s += &format!("fitness={} ", fixed(self.fitness(), 3));
        if self.fitness_above.is_finite() {
            s += &format!("above={} ", fixed(self.fitness_above, 3));
        }
        if self.ground_offset.is_finite() {
            s += &format!("dz={}cm ", signed(self.ground_offset * 100.0, 1));
        }
        s + &format!("rmse={} mm{stem} ({})", crate::util::pyformat::fixed_width(self.rmse() * 1000.0, 6, 1), self.reason)
    }
}

/// Store the positions of the matched stems (`_attach_matched_stems`).
fn attach_matched_stems(result: &mut PairResult, source: &ScanFeatures, target: &ScanFeatures, m: &StemMatch) {
    if !m.success || m.correspondences.is_empty() {
        return;
    }
    result.matched_source = m.correspondences.iter().map(|&(a, _)| { let s = &source.stems[a]; [s.x, s.y, s.z] }).collect();
    result.matched_target = m.correspondences.iter().map(|&(_, b)| { let s = &target.stems[b]; [s.x, s.y, s.z] }).collect();
}

/// Screen a coarse match before paying for ICP (`_coarse_is_acceptable`);
/// sets `reason` on failure.
fn coarse_is_acceptable(result: &mut PairResult, m: &StemMatch, cfg: &CoregConfig) -> bool {
    if !m.success || (m.n_inliers as i64) < cfg.min_match_inliers {
        result.reason = format!("stem matching failed ({} inliers, need {})", m.n_inliers, cfg.min_match_inliers);
        return false;
    }
    if m.inlier_rmse > cfg.max_match_rmse {
        result.reason = format!("coarse match too loose ({} cm scatter, limit {} cm)", fixed(m.inlier_rmse * 100.0, 1), fixed(cfg.max_match_rmse * 100.0, 0));
        return false;
    }
    if m.ambiguity > cfg.max_match_ambiguity {
        match &m.rival {
            Some(r) if r.success => result.rival = Some((**r).clone()),
            _ => {
                result.reason = format!("ambiguous stem pattern (a rival alignment has {} of the inliers, limit {})", percent(m.ambiguity, 0), percent(cfg.max_match_ambiguity, 0));
                return false;
            }
        }
    }
    result.coarse_stem_rmse = stem_median_residual(&m.transform.0, &result.matched_source, &result.matched_target);
    if result.coarse_stem_rmse > cfg.max_coarse_stem_rmse {
        result.reason = format!("coarse stems disagree by {} m (limit {} m)", fixed(result.coarse_stem_rmse, 2), fixed(cfg.max_coarse_stem_rmse, 2));
        return false;
    }
    true
}

/// A screened pair's probe (`_screen`): the matched stems attached and the
/// coarse match judged; whether it is worth ICP on its stems.
pub fn screen_probe(probe: &mut PairResult, source: &ScanFeatures, target: &ScanFeatures, m: &StemMatch, cfg: &CoregConfig) -> bool {
    attach_matched_stems(probe, source, target, m);
    coarse_is_acceptable(probe, m, cfg)
}

/// ICP from `initial` against `target`'s points or its prepared pyramid.
fn run_icp(source: &ScanFeatures, target: &ScanFeatures, pyramid: Option<&IcpTarget>, initial: &Mat4, cfg: &IcpConfig) -> Result<IcpResult> {
    match pyramid {
        Some(t) => icp_prepared(&source.icp_points, t, Some(*initial), cfg),
        None => crate::coreg::icp::icp(&source.icp_points, &target.icp_points, Some(*initial), cfg),
    }
}

/// The settings `plane_information` is computed with: the ICP's, but always
/// point-to-plane, as the Python binding builds them.
fn plane_config(cfg: &IcpConfig) -> IcpConfig {
    IcpConfig { voxel_sizes: cfg.voxel_sizes.clone(), max_distances: cfg.max_distances.clone(), robust: cfg.robust.clone(), robust_scale: cfg.robust_scale, trim_fraction: cfg.trim_fraction, trim_ramp: cfg.trim_ramp, min_planarity: cfg.min_planarity, normal_neighbours: cfg.normal_neighbours, max_points: cfg.max_points, seed: cfg.seed, ..IcpConfig::default() }
}

/// ICP-style quality of a transform ICP did not produce (`_score`).
fn score(source: &ScanFeatures, target: &ScanFeatures, pyramid: Option<&IcpTarget>, transform: &Mat4, cfg: &CoregConfig, refined: &IcpResult) -> Result<IcpResult> {
    let (fitness, rmse, n) = evaluate_registration(&source.icp_points, &target.icp_points, transform, cfg.icp.fitness_threshold, 200_000, Some(0.05), 0);
    let pc = plane_config(&cfg.icp);
    let information = match pyramid {
        // A pyramid prepared for point-to-point ICP has no normals: the
        // information is always point-to-plane, so that pyramid is built here.
        Some(t) if t.matches(&pc) => plane_information(&source.icp_points, t, transform, &pc),
        _ => plane_information(&source.icp_points, &IcpTarget::new(&target.icp_points, &pc), transform, &pc),
    };
    Ok(IcpResult { transform: *transform, fitness, inlier_rmse: rmse, n_correspondences: n, iterations: refined.iterations, converged: false, history: refined.history.clone(), information })
}

/// Accept a pair ICP refused if its reflector match is strong enough alone
/// (`_trust_reflectors`).
pub fn trust_reflectors(result: &mut PairResult, coarse: &Mat4, shift: f64, cfg: &CoregConfig) {
    let Some(found) = &result.reflector_match else { return };
    if cfg.trusted_reflector_matches <= 0 || (found.n_inliers as i64) < cfg.trusted_reflector_matches || found.rmse > cfg.trusted_reflector_rmse || result.reason.starts_with("ambiguous") {
        return;
    }
    let (n, rmse) = (found.n_inliers, found.rmse);
    let why = result.reason.clone();
    if shift > cfg.reflector_tolerance {
        result.transform = *coarse;
        result.used_icp = false;
    }
    result.success = true;
    result.trusted = true;
    result.reason = format!("{n} reflectors, {} mm, trusted ({} pose; ICP alone: {why})", fixed(rmse * 1000.0, 1), if result.used_icp { "ICP" } else { "targets" });
}

fn yaw(t: &Mat4) -> f64 {
    t[(1, 0)].atan2(t[(0, 0)])
}

/// Refine a coarse transform with ICP and decide whether to accept it
/// (`_refine_and_judge`).
///
/// Every test, and later the pose-graph edge, judges the transform the pair
/// reports: when the stem cross-check keeps the coarse transform, it is
/// scored afresh rather than by the ICP it replaced.
pub fn refine_and_judge(result: &mut PairResult, source: &ScanFeatures, target: &ScanFeatures, cfg: &CoregConfig, coarse: &Mat4, prepared: Option<&IcpTarget>) -> Result<()> {
    let built;
    let pyramid = match prepared {
        Some(t) => Some(t),
        None if result.rival.is_some() => {
            built = IcpTarget::new(&target.icp_points, &cfg.icp); // two ICPs against it
            Some(&built)
        }
        None => None,
    };
    let mut coarse = *coarse;
    result.coarse_transform = coarse;
    let mut refined = run_icp(source, target, pyramid, &coarse, &cfg.icp)?;
    result.icp = Some(refined.clone());
    result.transform = refined.transform;
    // Cross-check ICP against the stems it was meant to refine.
    if result.matched_source.len() >= 3 {
        result.coarse_stem_rmse = stem_median_residual(&coarse, &result.matched_source, &result.matched_target);
        result.fine_stem_rmse = stem_median_residual(&refined.transform, &result.matched_source, &result.matched_target);
        if result.fine_stem_rmse > result.coarse_stem_rmse + cfg.stem_agreement_tolerance {
            result.transform = coarse;
            result.used_icp = false;
        }
    }
    result.fitness_above = above_ground_fitness(&source.icp_points, &source.icp_heights, &target.icp_points, &result.transform, cfg);
    if let Some(rival) = result.rival.clone() {
        // An ambiguous stem pattern: refine the rival too and keep whichever
        // fits the above-ground points better, if the margin is clear.
        let rival_coarse = on_ground(&rival.transform.0, source, &[(target, Matrix4::identity())], cfg);
        let other = run_icp(source, target, pyramid, &rival_coarse, &cfg.icp)?;
        let other_above = above_ground_fitness(&source.icp_points, &source.icp_heights, &target.icp_points, &other.transform, cfg);
        let mut mine = if result.fitness_above.is_finite() { result.fitness_above } else { 0.0 };
        let mut theirs = if other_above.is_finite() { other_above } else { 0.0 };
        // Two hypotheses ICP pulls to the same pose were never rivals.
        let (_, apart) = transform_difference(&result.transform, &other.transform);
        let mut yaw_apart = degrees((yaw(&result.transform) - yaw(&other.transform)).abs());
        yaw_apart = yaw_apart.min(360.0 - yaw_apart);
        let converged = apart < cfg.matching.distinct_translation && yaw_apart < cfg.matching.distinct_yaw_deg;
        if converged {
            result.reason = "rivals converged in ICP; ".into();
        } else if theirs > mine {
            result.stem_match = Some(rival.clone());
            attach_matched_stems(result, source, target, &rival);
            coarse = rival_coarse;
            result.coarse_transform = coarse;
            refined = other.clone();
            result.icp = Some(other.clone());
            result.transform = other.transform;
            result.fitness_above = other_above;
            std::mem::swap(&mut mine, &mut theirs);
            result.used_icp = true;
            if result.matched_source.len() >= 3 {
                result.coarse_stem_rmse = stem_median_residual(&coarse, &result.matched_source, &result.matched_target);
                result.fine_stem_rmse = stem_median_residual(&refined.transform, &result.matched_source, &result.matched_target);
            }
        }
        if !converged {
            if mine < cfg.ambiguity_margin * theirs {
                result.reason = format!("ambiguous stem pattern; ICP cannot separate the rivals (above-ground fitness {} vs {})", fixed(mine, 3), fixed(theirs, 3));
                return Ok(());
            }
            result.reason = format!("rival resolved by ICP ({} vs {} above ground); ", fixed(mine, 3), fixed(theirs, 3));
        }
    }
    if !result.used_icp {
        result.icp = Some(score(source, target, pyramid, &result.transform, cfg, &refined)?);
    }
    let scored = result.icp.clone().expect("scored");
    let (_, shift) = transform_difference(&coarse, &refined.transform);
    result.ground_offset = height_offset(source, &result.transform, &[(target, Matrix4::identity())], cfg);
    if result.used_icp && shift > cfg.max_coarse_to_fine_shift {
        result.reason = format!("ICP diverged from the coarse solution by {} m", fixed(shift, 2));
    } else if scored.fitness < cfg.min_icp_fitness {
        result.reason = format!("low ICP fitness ({} < {})", fixed(scored.fitness, 3), py_float(cfg.min_icp_fitness));
    } else if result.fitness_above < cfg.min_icp_fitness_above_ground {
        result.reason = format!("low above-ground fitness ({} < {}); ground alone matched", fixed(result.fitness_above, 3), py_float(cfg.min_icp_fitness_above_ground));
    } else if scored.inlier_rmse > cfg.max_icp_rmse {
        result.reason = format!("high ICP rmse ({} m > {})", fixed(scored.inlier_rmse, 3), py_float(cfg.max_icp_rmse));
    } else if ground_disagrees(result.ground_offset, cfg) {
        result.reason = format!("terrain heights disagree by {} m", signed(result.ground_offset, 2));
    } else {
        result.success = true;
        if result.reflector_match.is_none() {
            let kept = if result.reason.starts_with("rival resolved") || result.reason.starts_with("rivals converged") { result.reason.clone() } else { String::new() };
            result.reason = kept
                + &if result.used_icp {
                    format!("shift from coarse {} cm", fixed(shift * 100.0, 1))
                } else {
                    format!("kept coarse (ICP moved the stems {} cm)", fixed(shift * 100.0, 1))
                };
        } else if !result.used_icp {
            result.reason += " (kept coarse)";
        }
    }
    if !result.success {
        trust_reflectors(result, &coarse, shift, cfg);
    }
    Ok(())
}

/// A coarse transform by one method, or `None` (`_coarse_transform`).
fn coarse_transform(result: &mut PairResult, source: &ScanFeatures, target: &ScanFeatures, cfg: &CoregConfig, given: Option<&StemMatch>, reflectors: bool) -> Result<Option<Mat4>> {
    if reflectors {
        let src: Vec<Point> = source.reflectors.iter().map(|r| r.position()).collect();
        let dst: Vec<Point> = target.reflectors.iter().map(|r| r.position()).collect();
        let found = match_reflectors(&src, &dst, cfg.reflector_tolerance, cfg.min_reflector_matches, 0.03)?;
        if !found.success {
            return Ok(None);
        }
        result.reason = format!("{} reflectors, {} mm", found.n_inliers, fixed(found.rmse * 1000.0, 1));
        let t = found.transform;
        result.reflector_match = Some(found);
        return Ok(Some(t));
    }
    let m = match given {
        Some(m) => m.clone(),
        None => match_stem_maps(&source.stem_map(), &target.stem_map(), &cfg.matching),
    };
    result.stem_match = Some(m.clone());
    attach_matched_stems(result, source, target, &m);
    if !coarse_is_acceptable(result, &m, cfg) {
        return Ok(None);
    }
    Ok(Some(on_ground(&m.transform.0, source, &[(target, Matrix4::identity())], cfg)))
}

/// Coarse-match then ICP-refine one pair of prepared scans (`register_pair`).
///
/// With `initial`, ICP starts there. Otherwise reflective targets are tried
/// first where both scans have them, then stems (with `stem_match` if the
/// pair was screened already); the first that ICP accepts wins, and a
/// target match ICP rejects does not cost the pair its stem match.
/// `prepared` is `target`'s ICP pyramid, if built already.
#[allow(clippy::too_many_arguments)]
pub fn register_pair(source: &ScanFeatures, target: &ScanFeatures, cfg: &CoregConfig, initial: Option<&Mat4>, stem_match: Option<&StemMatch>, i: i64, j: i64, prepared: Option<&IcpTarget>) -> Result<PairResult> {
    let start = Instant::now();
    let mut result = PairResult::new(i, j, &source.name, &target.name);
    if let Some(initial) = initial {
        refine_and_judge(&mut result, source, target, cfg, initial, prepared)?;
        result.seconds = start.elapsed().as_secs_f64();
        return Ok(result);
    }
    // Targets first: a target is located to millimetres and three fix all
    // six degrees of freedom. Stems stay as a fallback, because three
    // targets can form a congruent triangle by coincidence.
    let mut attempts = Vec::new();
    if cfg.use_reflectors && !source.reflectors.is_empty() && !target.reflectors.is_empty() {
        attempts.push(true);
    }
    attempts.push(false);
    let mut last = result;
    for reflectors in attempts {
        let mut attempt = PairResult::new(i, j, &source.name, &target.name);
        let Some(coarse) = coarse_transform(&mut attempt, source, target, cfg, stem_match, reflectors)? else {
            if !attempt.reason.is_empty() {
                last = attempt;
            }
            continue;
        };
        refine_and_judge(&mut attempt, source, target, cfg, &coarse, prepared)?;
        attempt.seconds = start.elapsed().as_secs_f64();
        if attempt.success {
            return Ok(attempt);
        }
        last = attempt;
    }
    last.seconds = start.elapsed().as_secs_f64();
    Ok(last)
}

/// `(fitness, rmse, correspondences)` weighting a pair's pose-graph edge
/// (`_edge_quality`): a trusted reflector pair that kept the targets' pose
/// is weighted by the targets' own residual and count.
pub fn edge_quality(pair: &PairResult) -> (f64, f64, i64) {
    if pair.trusted && !pair.used_icp {
        if let Some(found) = &pair.reflector_match {
            return (1.0, pymax(found.rmse, 1e-3), found.n_inliers as i64);
        }
    }
    (pair.fitness(), pair.rmse(), pair.icp.as_ref().map_or(0, |r| r.n_correspondences as i64))
}

/// Information of a pair's edge from its point-to-plane correspondences, or
/// `None` (the default information) without them (`_edge_information`).
pub fn edge_information(pair: &PairResult, cfg: &CoregConfig) -> Option<nalgebra::Matrix6<f64>> {
    let info = pair.icp.as_ref()?.information.as_ref()?;
    if pair.trusted && !pair.used_icp {
        return None;
    }
    Some(crate::coreg::posegraph::plane_edge_information(&info.hessian, info.sigma, info.n as i64, &pair.transform, cfg.information_patch_points, cfg.information_min_sigma))
}

// ------------------------------------------------------------ shared targets

/// The ICP pyramids of the most recently used targets (`_TargetCache`).
///
/// Each pyramid is built once, outside the lock, by whichever thread asks
/// first; the others wait for it. The pyramid depends only on the target
/// and the settings, so the cache changes no result.
pub struct TargetCache<'a> {
    scans: &'a [ScanFeatures],
    config: &'a IcpConfig,
    capacity: usize,
    built: Mutex<Vec<(usize, Arc<OnceLock<IcpTarget>>)>>,
}

impl<'a> TargetCache<'a> {
    /// A cache over `scans` for pyramids built with `config`, keeping the
    /// `capacity` (at least 1) most recently used.
    pub fn new(scans: &'a [ScanFeatures], config: &'a IcpConfig, capacity: usize) -> Self {
        TargetCache { scans, config, capacity: capacity.max(1), built: Mutex::new(Vec::new()) }
    }

    /// The pyramid of scan `k`, built on first use and marked most recently
    /// used; `get()` on the returned cell is always `Some`.
    pub fn get(&self, k: usize) -> Arc<OnceLock<IcpTarget>> {
        let cell = {
            let mut built = self.built.lock().expect("target cache");
            let cell = match built.iter().position(|(m, _)| *m == k) {
                Some(pos) => {
                    let entry = built.remove(pos);
                    let cell = Arc::clone(&entry.1);
                    built.push(entry);
                    cell
                }
                None => {
                    let cell = Arc::new(OnceLock::new());
                    built.push((k, Arc::clone(&cell)));
                    while built.len() > self.capacity {
                        built.remove(0);
                    }
                    cell
                }
            };
            cell
        };
        cell.get_or_init(|| IcpTarget::new(&self.scans[k].icp_points, self.config));
        cell
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stand(seed: u64) -> Vec<Point> {
        let mut rng = crate::util::nprandom::Generator::new(seed);
        let mut pts = Vec::new();
        for _ in 0..30000 {
            let (x, y) = (rng.random() * 30.0 - 15.0, rng.random() * 30.0 - 15.0);
            pts.push([x, y, 0.02 * x + 0.01 * rng.random()]);
        }
        for (cx, cy, r) in [(3.0, 2.0, 0.2), (-4.0, 5.0, 0.3), (6.0, -5.0, 0.25), (-6.0, -3.0, 0.15), (0.5, 8.0, 0.22), (9.0, 4.0, 0.35), (-9.0, 7.0, 0.18), (1.0, -8.0, 0.28)] {
            for _ in 0..3000 {
                let a = rng.random() * std::f64::consts::TAU;
                let h = rng.random() * 6.0;
                pts.push([cx + r * a.cos(), cy + r * a.sin(), 0.02 * cx + h]);
            }
        }
        pts
    }

    #[test]
    fn products_are_numpys() {
        let a = crate::coreg::transforms::se3_exp(&nalgebra::Vector6::new(0.1, -0.2, 0.3, 1.0, 2.0, -3.0));
        let b = crate::coreg::transforms::se3_exp(&nalgebra::Vector6::new(-0.3, 0.1, 0.2, -1.0, 0.5, 4.0));
        let c = matmul(&a, &b);
        assert!((c - a * b).abs().max() < 1e-12);
        assert_eq!(matmul(&Matrix4::identity(), &b), b);
    }

    #[test]
    fn python_helpers() {
        assert_eq!(py_head(&[1, 2, 3], 2), vec![1, 2]);
        assert_eq!(py_head(&[1, 2, 3], -1), vec![1, 2]);
        assert_eq!(py_head(&[1, 2, 3], -5), Vec::<i32>::new());
        assert_eq!(py_head(&[1, 2, 3], 9), vec![1, 2, 3]);
        assert_eq!(py_float(0.04), "0.04");
        assert_eq!(py_float(1.0), "1.0");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a"), "'a'");
        assert_eq!(path_stem(Path::new("/x/a.b.laz")), "a.b");
        assert_eq!(path_stem(Path::new("/x/.hidden")), ".hidden");
        assert_eq!(PrepareError::MissingAttribute("x".into()).describe(), "KeyError: 'x'");
    }

    #[test]
    fn a_scan_is_prepared_and_registers_to_itself() {
        let cfg = CoregConfig::default();
        let pts = stand(3);
        let scan = prepare_scan(&ScanInput::Points(pts.clone()), &cfg, "a", &[], None, None).unwrap();
        assert!(scan.usable());
        assert_eq!(scan.stems.len(), 8);
        assert!(scan.stems.windows(2).all(|w| w[0].quality() >= w[1].quality()));
        assert!(scan.icp_points.iter().all(|p| p.iter().all(|&v| v as f32 as f64 == v)));
        let moved = crate::coreg::transforms::yaw_transform(0.4, 2.0, -1.0, 0.3);
        let other = prepare_scan(&ScanInput::Points(transform_points(&invert(&moved), &pts)), &cfg, "b", &[], None, None).unwrap();
        let pair = register_pair(&scan, &other, &cfg, None, None, 0, 1, None).unwrap();
        assert!(pair.success, "{}", pair.reason);
        let truth = invert(&moved);
        let (_, t) = transform_difference(&pair.transform, &truth);
        assert!(t < 0.02, "{t}");
        assert!(pair.summary().starts_with("[ok  ] a -> b: stems= 8"));
        let few = prepare_scan(&ScanInput::Points(pts[..500].to_vec()), &cfg, "few", &[], None, None).unwrap();
        assert_eq!(few.error, "only 500 points");
        assert!(!few.usable());
        let offset = height_offset(&other, &moved, &[(&scan, Matrix4::identity())], &cfg);
        assert!(offset.abs() < 0.02, "{offset}");
    }

    #[test]
    fn a_kept_coarse_transform_is_scored_against_a_point_to_point_pyramid() {
        // A negative tolerance makes the stem cross-check keep the coarse
        // transform, which is then scored: with a pyramid prepared for
        // point-to-point ICP that used to fail the pair, and the survey with it.
        let mut cfg = CoregConfig { stem_agreement_tolerance: -1.0, ..CoregConfig::default() };
        cfg.icp.method = "point_to_point".into();
        let pts = stand(3);
        let moved = crate::coreg::transforms::yaw_transform(0.4, 2.0, -1.0, 0.3);
        let a = prepare_scan(&ScanInput::Points(pts.clone()), &cfg, "a", &[], None, None).unwrap();
        let b = prepare_scan(&ScanInput::Points(transform_points(&invert(&moved), &pts)), &cfg, "b", &[], None, None).unwrap();
        let prepared = IcpTarget::new(&b.icp_points, &cfg.icp);
        let pair = register_pair(&a, &b, &cfg, None, None, 0, 1, Some(&prepared)).unwrap();
        assert!(pair.success && !pair.used_icp, "{}", pair.reason);
        assert!(pair.icp.unwrap().information.is_some());
    }

    #[test]
    fn the_point_cap_strides_through_the_whole_scan() {
        let pts: Vec<Point> = (0..10).map(|k| [k as f64, 0.0, 0.0]).collect();
        let x = |v: Vec<Point>| v.iter().map(|p| p[0] as usize).collect::<Vec<_>>();
        assert_eq!(x(stride_subsample(&pts, 4)), vec![0, 2, 5, 7]);
        assert_eq!(x(stride_subsample(&pts, 10)), (0..10).collect::<Vec<_>>());
        assert_eq!(stride_subsample(&pts, 99).len(), 10);
        assert!(stride_subsample(&pts, 0).is_empty());
        // Through a file: the cap reaches the far end of the recording.
        let dir = std::env::temp_dir().join(format!("sylva-cap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scan.xyz");
        let text: String = (0..1000).map(|k| format!("{k} 0 0\n")).collect();
        std::fs::write(&path, text).unwrap();
        let cfg = CoregConfig { max_points_per_scan: 10, ..CoregConfig::default() };
        let read = read_scan(&path, &cfg);
        std::fs::remove_dir_all(&dir).unwrap();
        let read = read.unwrap();
        assert_eq!(read.len(), 10);
        assert_eq!(read[9][0], 900.0);
    }

    #[test]
    fn a_refit_that_would_leave_no_ground_keeps_the_first_fit() {
        // Flat ground seen from 1.6 m at 3-30 m (elevations -28 to -3 degrees,
        // under 80 % of the band) and a thin pole every 6 degrees filling the
        // band there: the blind azimuths, widened, cover the circle, and an
        // upright scanner has nothing 45 degrees down to keep.
        let mut rng = crate::util::nprandom::Generator::new(5);
        let mut pts: Vec<Point> = Vec::new();
        for _ in 0..60000 {
            let (a, r) = (rng.random() * std::f64::consts::TAU, 3.0 + 27.0 * rng.random());
            pts.push([r * a.cos(), r * a.sin(), 0.0]);
        }
        for k in 0..60 {
            let a = (k as f64 * 6.0 + 0.5).to_radians();
            pts.extend((0..500).map(|i| [8.0 * a.cos(), 8.0 * a.sin(), 6.0 * i as f64 / 500.0]));
        }
        let cfg = CoregConfig::default();
        assert!(refit_visible_ground(&pts, [0.0, 0.0, 1.6], &cfg).unwrap().is_none());
        let scan = prepare_scan(&ScanInput::Points(pts), &cfg, "open", &[], None, Some([0.0, 0.0, 1.6])).unwrap();
        assert!(scan.usable(), "{}", scan.error);
    }
}
