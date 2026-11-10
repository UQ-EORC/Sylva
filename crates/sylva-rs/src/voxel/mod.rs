// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Adapted from rayvoxel (Josh Rivory, unpublished), a port of AMAPVox (UMR AMAP);
// see THIRD_PARTY_NOTICES.md.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Ray-traced voxel grid with Beer–Lambert statistics (Vincent et al. 2017).
//!
//! A port of `rayvoxel` (J. Rivory, unpublished), a reimplementation of the
//! method of Vincent et al. (2017) on raycloudtools, working on [`Shots`] instead of ray-cloud files. Every pulse is
//! traced through the grid twice: once unweighted (beam counts, potential
//! path length, sub-voxel exploration) and once per echo segment with the
//! fraction of the beam still travelling (free path length, beam sections,
//! mean angles). Attenuation is then estimated per voxel by free path length
//! (FPL, with Pimont et al. 2018 bias correction), exact potential path
//! length (PPL), transmittance or Bailey & Mahaffee (2017) eq. 10, and
//! turned into plant / leaf / wood area density with a projection function
//! `G` from an analytic leaf angle distribution or from per-tree inclination
//! angle distributions estimated on the echoes (Vicari et al. 2019).
//!
//! What is not ported: the out-of-core shard path (grids too large for memory
//! are traced block by block instead, see [`voxelize_blocks`]), the NetCDF
//! writer and the per-voxel LAS class histogram (only plant / leaf / wood hit
//! counts are kept).

mod blocked;
mod blocks;
pub mod grid;
mod iad;
pub(crate) mod metrics;
pub mod quality;
mod refine;
mod traverse;
mod wood;
mod write;

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::qsm::Cylinder;
use crate::transform::{add, scale};
use crate::{Point, Raster, Shots};

pub use blocked::BlockedGrid;
pub use blocks::{voxelize_blocks, BlockOptions, BlockStats, Pulses};
pub use iad::TreeIad;
pub use metrics::{classify_de_wit, compute_g, compute_g_from_histogram, laser_spec, solve_bailey_pad, Lad};
pub use write::WriteOptions;

/// How the energy of a pulse is shared between its echoes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMethod {
    /// `1 / n` per echo (equal echo weighting).
    Equal,
    /// The last echo carries the whole pulse.
    Full,
    /// The first echo carries the whole pulse.
    First,
    /// Proportional to echo intensity (falls back to `Equal` when all zero).
    Relative,
    /// The most intense echo carries the whole pulse.
    Strongest,
}

impl WeightMethod {
    /// A method from its rayvoxel name: `equal`, `full`, `first`, `relative` or `strongest`.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "equal" => WeightMethod::Equal,
            "full" => WeightMethod::Full,
            "first" => WeightMethod::First,
            "relative" => WeightMethod::Relative,
            "strongest" => WeightMethod::Strongest,
            other => return Err(Error::invalid(format!("unknown weighting {other:?} (equal|full|first|relative|strongest)"))),
        })
    }
}

/// Attenuation coefficient estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attenuation {
    /// Free path length: the maximum-likelihood estimate of Pimont et al. (2018) over the
    /// path actually travelled, with their bias correction and the beam footprint of Pimont et al. (2019).
    Fpl,
    /// Potential path length: the exact maximum-likelihood solve over the full voxel chords
    /// (mean-chord form of Pimont et al. 2018 where the exact solve is unavailable).
    Ppl,
    /// `−ln(1 − intercepted / entering) / mean chord` (m⁻¹): the Beer–Lambert attenuation that
    /// gives the voxel's beam-section transmittance over the mean length of the pulses' paths
    /// across it (`path_length / num_beams`, m).
    Transmittance,
    /// Bailey & Mahaffee (2017) eq. 10 per class, from the mean gap probability over the mean
    /// path; needs leaf and wood echoes and the inclination distributions.
    Bailey,
}

impl Attenuation {
    /// A method from its name (`fpl`, `ppl`, `transmittance`, `bailey`; case-insensitive).
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "fpl" => Attenuation::Fpl,
            "ppl" => Attenuation::Ppl,
            "transmittance" => Attenuation::Transmittance,
            "bailey" => Attenuation::Bailey,
            other => return Err(Error::invalid(format!("unknown attenuation method {other:?} (fpl|ppl|transmittance|bailey)"))),
        })
    }

    /// The lower-case name used in column names (`pad_fpl`, `attenuation_ppl`, ...).
    pub fn name(&self) -> &'static str {
        match self {
            Attenuation::Fpl => "fpl",
            Attenuation::Ppl => "ppl",
            Attenuation::Transmittance => "transmittance",
            Attenuation::Bailey => "bailey",
        }
    }
}

/// Laser beam geometry for beam-section weighted metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamSpec {
    /// Beam diameter at the exit aperture (m).
    pub diameter: f64,
    /// Full beam divergence (rad).
    pub divergence: f64,
}

#[derive(Debug, Clone, PartialEq)]
/// Options of a ray-traced voxelisation (the defaults are rayvoxel's).
pub struct VoxelParams {
    /// Voxel edge (m).
    pub voxel_size: f64,
    /// Grid corners; `None` fits the grid to the (non-ground and ground) echoes.
    /// The max corner is snapped up to a whole number of voxels.
    pub bounds: Option<(Point, Point)>,
    /// How a pulse's energy is shared between its echoes.
    pub weighting: WeightMethod,
    /// Also trace each pulse beyond its last echo to map occluded space.
    pub occlusion: bool,
    /// Shorten the path in a column's top voxel to start at the highest echo
    /// (for flat-topped canopies such as crops).
    pub flat_top: bool,
    /// Borrow statistics from the 26 neighbours for voxels crossed by fewer
    /// weighted beams than this (0 = off). Border voxels are left alone.
    pub neighbour_prior_min_rays: u32,
    /// Enables beam-section metrics (`bs_*`, transmittance).
    pub beam: Option<BeamSpec>,
    /// `N` for an `N³` sub-voxel exploration grid (0 = off, at most 4).
    pub subvoxel_split: usize,
    /// Beams through a sub-voxel cell for it to count as explored.
    pub subvoxel_min_beams: u8,
    /// Mean single-leaf area (m²) for the effective free path; 0 disables it.
    pub average_leaf_area: f64,
    /// Analytic leaf angle distribution for `pad_g_corrected` and as the
    /// fallback `G` when no inclination distribution is available.
    pub lad: Lad,
    /// Attenuation estimators to compute (at least one); the first is the one behind `pad_g_corrected`.
    pub attenuation: Vec<Attenuation>,
    /// Estimate inclination angle distributions from echo normals.
    pub inclination: bool,
    /// Bins of the inclination histograms over 0–90 degrees.
    pub n_iad_bins: usize,
    /// Neighbours in the PCA that gives each echo its normal.
    pub knn_normal: usize,
    /// Longest edge of a Bailey triangle facet (m).
    pub triangle_lmax: f64,
    /// Length traced for pulses without an echo; infinite = to the grid edge.
    pub unbounded_range: f64,
}

impl Default for VoxelParams {
    fn default() -> Self {
        VoxelParams {
            voxel_size: 0.1,
            bounds: None,
            weighting: WeightMethod::Equal,
            occlusion: false,
            flat_top: false,
            neighbour_prior_min_rays: 0,
            beam: None,
            subvoxel_split: 0,
            subvoxel_min_beams: 10,
            average_leaf_area: 0.005,
            lad: Lad::Spherical,
            attenuation: vec![Attenuation::Fpl],
            inclination: false,
            n_iad_bins: 18,
            knn_normal: 10,
            triangle_lmax: 0.05,
            unbounded_range: f64::INFINITY,
        }
    }
}

/// Echo foliage classes (values of [`VoxelInputs::foliage`]).
pub mod foliage {
    /// Not vegetation: a hit, but in no plant / leaf / wood count.
    pub const EXCLUDED: u8 = 0;
    /// Vegetation of no particular class: counts as plant only.
    pub const PLANT: u8 = 1;
    /// Leaf: counts as plant and leaf.
    pub const LEAF: u8 = 2;
    /// Wood: counts as plant and wood.
    pub const WOOD: u8 = 3;
}

/// Pulses plus optional per-echo annotations (all of length `shots.n_echoes()`).
#[derive(Debug, Clone, Copy)]
pub struct VoxelInputs<'a> {
    /// The pulses to trace.
    pub shots: &'a Shots,
    /// Ground echoes: the pulse is traced up to them but they are never hits.
    pub ground: Option<&'a [bool]>,
    /// See [`foliage`]; `None` treats every non-ground echo as plant.
    pub foliage: Option<&'a [u8]>,
    /// For `Relative` / `Strongest` weighting.
    pub intensity: Option<&'a [f64]>,
    /// Groups echoes for the inclination distributions (negative = none);
    /// `None` pools all echoes as tree 0.
    pub tree_id: Option<&'a [i32]>,
    /// Terrain: stops occlusion rays below ground and gives `distance_from_ground`.
    pub dtm: Option<&'a Raster>,
}

impl<'a> VoxelInputs<'a> {
    /// Pulses with no annotations: every echo is a non-ground plant echo.
    pub fn new(shots: &'a Shots) -> Self {
        VoxelInputs { shots, ground: None, foliage: None, intensity: None, tree_id: None, dtm: None }
    }
}

/// Per-voxel `f32` accumulators. Names follow the rayvoxel output columns.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F {
    /// Σ beam fraction entering the voxel (observed traversals, count). A pulse enters a voxel
    /// once, with the fraction it still carries on the way in: one that leaves an echo in the
    /// voxel and travels on is not counted again for the part that goes on.
    NumBeamsWeighted,
    /// Potential path length: Σ full voxel chord.
    PathLength,
    /// Σ squared voxel chord (m²), for the spread of the chord lengths.
    PathLengthSq,
    /// Σ fraction × free path (entry → echo, or the full chord on a miss).
    FreePathLength,
    /// The part of `FreePathLength` from segments ending in a plant-class echo.
    FreePathLengthPlant,
    /// The part of `FreePathLength` from segments ending in a leaf echo.
    FreePathLengthLeaf,
    /// The part of `FreePathLength` from segments ending in a wood echo.
    FreePathLengthWood,
    /// Σ fraction × effective free path `−ln(1 − λ₁ z) / λ₁` (m), the finite-leaf
    /// correction of Pimont et al. (2018) with `λ₁ = average_leaf_area / (4 voxel volume)`.
    EffectiveFreePathLength,
    /// Σ path (m) of the rays traced beyond a pulse's last echo (occlusion rays, unit weight).
    PathLengthOccluded,
    /// Σ fraction × angle of the beam from the vertical (rad, 0 to π/2: 0 for a beam travelling
    /// straight up or straight down, π/2 for a horizontal one); over `NumBeamsWeighted` the
    /// mean zenith at which `G` is evaluated.
    SumOfAngles,
    /// Σ fraction × sin of the beam azimuth (clockwise from +y), for the circular mean.
    SumSinAzimuth,
    /// Σ fraction × cos of the beam azimuth, for the circular mean.
    SumCosAzimuth,
    /// Σ fraction × distance from the scanner to the voxel centre (m).
    SumOfLaserDistances,
    /// Σ full chord (m) of the voxel for every echo in it: the hit term of the mean-chord PPL estimator.
    SumHitDelta,
    /// Σ full chord (m) of every pulse that passed through the voxel without ending in it,
    /// whatever it ends in further on (an echo, the ground, or nothing): the miss term of
    /// the mean-chord PPL estimator. This is a larger set of pulses than `NumMissRays`
    /// counts, which leaves out those that end in a ground echo or in no echo at all, so
    /// `SumMissDelta / NumMissRays` is not a mean chord.
    SumMissDelta,
    /// Σ fraction × free path (m) of pulses with no echo, or whose segment ends in a ground echo.
    PathLengthUnbound,
    /// Σ fraction × beam cross-section (m²) at the voxel centre, for beams entering it
    /// (each pulse once, as for `NumBeamsWeighted`).
    BsEntering,
    /// Σ echo share × beam cross-section (m²) of the echoes in the voxel.
    BsIntercepted,
    /// Σ fraction × beam cross-section (m²) of the beams that passed right through the voxel.
    BsPotential,
    /// Σ fraction × beam cross-section × free path (m³).
    BsFreePath,
    /// Σ fraction × beam cross-section × effective free path (m³): the denominator of the FPL estimate.
    BsEffectiveFreePath,
    /// Σ echo share × beam cross-section × effective free path from the voxel entry to the echo (m³),
    /// over the echoes in the voxel: the Pimont et al. (2018) bias term.
    BsEffFreePathHits,
    /// Σ fraction of the pulse leaving the voxel × beam cross-section × chord (m³, or m without
    /// beam metrics): the miss side of the exact PPL solve. Only filled when `Ppl` is requested.
    PplMissWl,
    /// Σ echo share of the pulse over the echoes in the voxel: hits on the
    /// same footing as the share-weighted path lengths. `NumHits` counts
    /// every echo of a multi-echo pulse as one whole hit.
    HitsWeighted,
}

impl F {
    /// Number of `f32` accumulators.
    pub const COUNT: usize = 24;
    /// Every accumulator in slot order (`F::X as usize` indexes it).
    pub const ALL: [F; F::COUNT] = [
        F::NumBeamsWeighted, F::PathLength, F::PathLengthSq, F::FreePathLength, F::FreePathLengthPlant,
        F::FreePathLengthLeaf, F::FreePathLengthWood, F::EffectiveFreePathLength, F::PathLengthOccluded,
        F::SumOfAngles, F::SumSinAzimuth, F::SumCosAzimuth, F::SumOfLaserDistances, F::SumHitDelta,
        F::SumMissDelta, F::PathLengthUnbound, F::BsEntering, F::BsIntercepted, F::BsPotential,
        F::BsFreePath, F::BsEffectiveFreePath, F::BsEffFreePathHits, F::PplMissWl, F::HitsWeighted,
    ];

    /// The rayvoxel output column name.
    pub fn name(&self) -> &'static str {
        match self {
            F::NumBeamsWeighted => "num_beams_weighted",
            F::PathLength => "path_length",
            F::PathLengthSq => "path_length_sq",
            F::FreePathLength => "free_path_length",
            F::FreePathLengthPlant => "free_path_length_plant",
            F::FreePathLengthLeaf => "free_path_length_leaf",
            F::FreePathLengthWood => "free_path_length_wood",
            F::EffectiveFreePathLength => "effective_free_path_length",
            F::PathLengthOccluded => "path_length_occluded",
            F::SumOfAngles => "sum_of_angles",
            F::SumSinAzimuth => "sum_sin_azimuth",
            F::SumCosAzimuth => "sum_cos_azimuth",
            F::SumOfLaserDistances => "sum_of_laser_distances",
            F::SumHitDelta => "sum_hit_delta",
            F::SumMissDelta => "sum_miss_delta",
            F::PathLengthUnbound => "path_length_unbound",
            F::BsEntering => "bs_entering",
            F::BsIntercepted => "bs_intercepted",
            F::BsPotential => "bs_potential",
            F::BsFreePath => "bs_free_path",
            F::BsEffectiveFreePath => "bs_effective_free_path",
            F::BsEffFreePathHits => "bs_eff_free_path_hits",
            F::PplMissWl => "ppl_miss_wl",
            F::HitsWeighted => "num_hits_weighted",
        }
    }

    fn is_beam(&self) -> bool {
        matches!(self, F::BsEntering | F::BsIntercepted | F::BsPotential | F::BsFreePath | F::BsEffectiveFreePath | F::BsEffFreePathHits)
    }
}

/// Per-voxel `i32` counters.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I {
    /// Echoes in the voxel, one per echo (ground echoes and echoes carrying no share of the pulse excluded).
    NumHits,
    /// Pulses entering the voxel.
    NumBeams,
    /// Occlusion rays (traced beyond a pulse's last echo) through the voxel; column `num_beams_occluded`.
    NumRaysOccluded,
    /// Pulse segments crossing the voxel that end in a ground echo or in no echo at all (rayvoxel's unbounded rays).
    NumUnboundRays,
    /// Pulses (count) that crossed the voxel without ending in it, on their way to a non-ground
    /// echo beyond. Pulses that cross it on their way to a ground echo or to no echo at all are
    /// not counted here (they are rayvoxel's unbounded rays, see `NumUnboundRays`), although
    /// their chords are in `SumMissDelta`.
    NumMissRays,
    /// Echoes in the voxel with the leaf class.
    NumHitLeaf,
    /// Echoes in the voxel with the wood class.
    NumHitWood,
    /// Echoes in the voxel of any vegetation class (plant, leaf or wood).
    NumHitPlant,
}

impl I {
    /// Number of `i32` counters.
    pub const COUNT: usize = 8;
    /// Every counter in slot order.
    pub const ALL: [I; I::COUNT] = [
        I::NumHits, I::NumBeams, I::NumRaysOccluded, I::NumUnboundRays, I::NumMissRays, I::NumHitLeaf,
        I::NumHitWood, I::NumHitPlant,
    ];

    /// The rayvoxel output column name.
    pub fn name(&self) -> &'static str {
        match self {
            I::NumHits => "num_hits",
            I::NumBeams => "num_beams",
            I::NumRaysOccluded => "num_beams_occluded",
            I::NumUnboundRays => "num_unbound_rays",
            I::NumMissRays => "num_miss_rays",
            I::NumHitLeaf => "num_hit_leaf",
            I::NumHitWood => "num_hit_wood",
            I::NumHitPlant => "num_hit_plant",
        }
    }
}

/// What the rays saw of a voxel.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelState {
    /// Never traversed.
    Unobserved = 0,
    /// Only traversed beyond a pulse's last echo.
    Occluded = 1,
    /// Traversed, no echo.
    Empty = 2,
    /// Holds at least one echo.
    Filled = 3,
}

/// Dense voxel statistics; flat index is `i + nx (j + ny k)`.
#[derive(Debug, Clone)]
pub struct RayVoxels {
    /// Minimum corner of the grid (m).
    pub origin: Point,
    /// Voxel edge (m).
    pub voxel_size: f64,
    /// `[nx, ny, nz]`.
    pub shape: [usize; 3],
    /// The options the grid was traced with.
    pub params: VoxelParams,
    /// Whether any echo carried the leaf class (enables the leaf columns).
    pub has_leaf: bool,
    /// Whether any echo carried the wood class (enables the wood columns).
    pub has_wood: bool,
    /// One vector per [`F`]; empty when that group is switched off (reads as 0).
    pub f: Vec<Vec<f32>>,
    /// One vector per [`I`].
    pub i: Vec<Vec<i32>>,
    /// Exact PPL attenuation (−1 where unsolved); `None` unless `Ppl` was requested.
    pub ppl_lambda: Option<Vec<f32>>,
    /// `split³` saturating beam counts per voxel.
    pub subvoxel_counts: Option<Vec<u8>>,
    /// Terrain height under each column (`i + nx j`), NaN where unknown.
    pub ground_height: Option<Vec<f64>>,
    /// Woody volume per voxel (m³) from [`RayVoxels::add_wood_volume`].
    pub wood_volume: Option<Vec<f32>>,
    /// Tree with the most echoes in each voxel (−1 = none).
    pub predominant_tree: Option<Vec<i32>>,
    /// Inclination distributions by tree id; empty unless `inclination` or `Bailey` was requested.
    pub tree_iad: BTreeMap<i32, TreeIad>,
}

impl RayVoxels {
    /// Voxels in the grid.
    pub fn n_voxels(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    #[inline]
    /// An `f32` accumulator of the voxel at flat index `idx` (0 when that group was not traced).
    pub fn get_f(&self, field: F, idx: usize) -> f32 {
        self.f[field as usize].get(idx).copied().unwrap_or(0.0)
    }

    #[inline]
    /// An `i32` counter of the voxel at flat index `idx`.
    pub fn get_i(&self, field: I, idx: usize) -> i32 {
        self.i[field as usize][idx]
    }

    /// `(i, j, k)` of a flat index.
    #[inline]
    pub fn unravel(&self, idx: usize) -> [usize; 3] {
        let (nx, ny) = (self.shape[0], self.shape[1]);
        [idx % nx, (idx / nx) % ny, idx / (nx * ny)]
    }

    /// Centre of the voxel at flat index `idx` (m).
    pub fn center(&self, idx: usize) -> Point {
        let ijk = self.unravel(idx);
        [
            self.origin[0] + (ijk[0] as f64 + 0.5) * self.voxel_size,
            self.origin[1] + (ijk[1] as f64 + 0.5) * self.voxel_size,
            self.origin[2] + (ijk[2] as f64 + 0.5) * self.voxel_size,
        ]
    }

    /// What the rays saw of the voxel: filled if it holds an echo, else empty if a beam
    /// entered it, else occluded if only occlusion rays did, else unobserved.
    pub fn state(&self, idx: usize) -> VoxelState {
        if self.get_i(I::NumHits, idx) > 0 {
            VoxelState::Filled
        } else if self.get_f(F::NumBeamsWeighted, idx) > 0.0 {
            VoxelState::Empty
        } else if self.get_i(I::NumRaysOccluded, idx) > 0 {
            VoxelState::Occluded
        } else {
            VoxelState::Unobserved
        }
    }

    /// Rasterise QSM cylinders into the grid (adds to any earlier call).
    pub fn add_wood_volume(&mut self, cylinders: &[Cylinder]) {
        let n = self.n_voxels();
        let out = self.wood_volume.get_or_insert_with(|| vec![0.0; n]);
        wood::rasterise(cylinders, &self.origin, self.voxel_size, &self.shape, out);
    }
}

/// Ground echoes by height: at most `max_above` over the terrain, as
/// rayvoxel's `--dtm_filter_distance`, but echoes under the DTM surface count
/// as ground too rather than as vegetation.
pub fn ground_mask_from_dtm(shots: &Shots, dtm: &Raster, max_above: f64) -> Vec<bool> {
    shots.echo_xyz().iter().map(|p| p[2] - dtm.sample(p[0], p[1]) <= max_above).collect()
}

fn check_len<T>(name: &str, v: Option<&[T]>, n: usize) -> Result<()> {
    match v {
        Some(a) if a.len() != n => Err(Error::invalid(format!("{name} has {} values for {n} echoes", a.len()))),
        _ => Ok(()),
    }
}

/// Refuse pulses with an echo whose range is NaN or infinite. Such an echo
/// has no position: it cannot be placed in a voxel, and the pulse cannot be
/// traced to it.
pub(crate) fn check_ranges(shots: &Shots) -> Result<()> {
    match shots.echo_range.iter().position(|r| !r.is_finite()) {
        Some(e) => Err(Error::invalid(format!(
            "echo {e} has a range of {} m; echo ranges must be finite (drop these echoes, or their pulses, before voxelising)",
            shots.echo_range[e]
        ))),
        None => Ok(()),
    }
}

/// Incremental voxelisation: pulses can be added in any number of batches
/// (e.g. the row groups of a [`crate::io::shots::ShotsFile`]), so only one
/// batch has to be in memory next to the grid.
pub struct Voxelizer {
    engine: traverse::Engine,
    peaks: Option<Vec<f64>>,
    ground_height: Option<Vec<f64>>,
    echoes: Option<iad::EchoPoints>,
    has_leaf: bool,
    has_wood: bool,
}

impl Voxelizer {
    /// `bounds` are the grid corners (the max corner is snapped up to a whole
    /// number of voxels); `params.bounds` is ignored.
    pub fn new(params: &VoxelParams, bounds: (Point, Point), dtm: Option<&Raster>) -> Result<Self> {
        if !(params.voxel_size > 0.0) {
            return Err(Error::invalid("voxel_size must be positive"));
        }
        if params.subvoxel_split > 4 {
            return Err(Error::invalid("subvoxel_split must be at most 4"));
        }
        if params.attenuation.is_empty() {
            return Err(Error::invalid("at least one attenuation method is needed"));
        }
        let lo = bounds.0;
        let shape = grid_shape(params, bounds)?;
        // A grid is nx*ny*nz cells whatever the cloud holds: a plot asked for
        // at a centimetre is tens of billions of them.
        let cells = shape.iter().map(|&n| n as u128).product::<u128>();
        let per_cell = std::mem::size_of::<traverse::Cell>() as u64
            + (params.subvoxel_split.max(1).pow(3) as u64);
        crate::util::limits::check_cells(
            cells,
            per_cell,
            &format!("a {} x {} x {} voxel grid at {} m", shape[0], shape[1], shape[2], params.voxel_size),
            "a larger voxel, a smaller area, or splitting the plot into tiles",
        )?;
        let ground_height = dtm.map(|dtm| column_ground(dtm, lo, params.voxel_size, shape));
        let wants_iad = params.inclination || params.attenuation.contains(&Attenuation::Bailey);
        Ok(Voxelizer {
            engine: traverse::Engine::new(params, traverse::Geom::new(lo, params.voxel_size, shape)),
            peaks: None,
            ground_height,
            echoes: wants_iad.then(iad::EchoPoints::default),
            has_leaf: false,
            has_wood: false,
        })
    }

    /// Flat-top compensation needs the highest echo of every column before
    /// any pulse is traced: with `params.flat_top`, call this for every batch
    /// first, then [`Voxelizer::add`] them all.
    pub fn add_peaks(&mut self, shots: &Shots) {
        let n = self.engine.geom.shape[0] * self.engine.geom.shape[1];
        let peaks = self.peaks.get_or_insert_with(|| vec![f64::MIN; n]);
        refine::update_peaks(peaks, &shots.echo_xyz(), &self.engine.geom);
    }

    /// Trace one batch of pulses into the grid. The annotations must have one value per
    /// echo, every echo range (m) must be finite, and `Relative` / `Strongest` weighting
    /// needs intensities.
    pub fn add(&mut self, inputs: &VoxelInputs) -> Result<()> {
        let ne = inputs.shots.n_echoes();
        check_len("ground", inputs.ground, ne)?;
        check_len("foliage", inputs.foliage, ne)?;
        check_len("intensity", inputs.intensity, ne)?;
        check_len("tree_id", inputs.tree_id, ne)?;
        check_ranges(inputs.shots)?;
        if matches!(self.engine.params.weighting, WeightMethod::Relative | WeightMethod::Strongest) && inputs.intensity.is_none() {
            return Err(Error::invalid("relative / strongest weighting needs echo intensities"));
        }
        self.has_leaf |= inputs.foliage.is_some_and(|f| f.contains(&foliage::LEAF));
        self.has_wood |= inputs.foliage.is_some_and(|f| f.contains(&foliage::WOOD));
        let peaks = if self.engine.params.flat_top { self.peaks.as_deref() } else { None };
        self.engine.add(inputs, peaks, self.ground_height.as_deref());
        if let Some(e) = &mut self.echoes {
            e.collect(inputs, &self.engine.geom);
        }
        Ok(())
    }

    /// The finished grid: neighbour priors applied, inclination distributions built. The
    /// Bailey method needs both leaf and wood echoes among the batches added.
    pub fn finish(self) -> Result<RayVoxels> {
        let bailey = self.engine.params.attenuation.contains(&Attenuation::Bailey);
        if bailey && !(self.has_leaf && self.has_wood) {
            return Err(Error::invalid("the bailey method needs both leaf and wood echoes"));
        }
        let prior = self.engine.params.neighbour_prior_min_rays;
        let mut vox = self.engine.finish();
        vox.has_leaf = self.has_leaf;
        vox.has_wood = self.has_wood;
        vox.ground_height = self.ground_height;
        if prior > 0 {
            refine::apply_neighbour_priors(&mut vox, prior as f32);
        }
        if let Some(e) = self.echoes {
            iad::build(&mut vox, e);
        }
        Ok(vox)
    }
}

/// Voxels per axis of a grid over `bounds` (the max corner snapped up).
pub(crate) fn grid_shape(params: &VoxelParams, bounds: (Point, Point)) -> Result<[usize; 3]> {
    let (lo, hi) = bounds;
    let mut shape = [0usize; 3];
    for k in 0..3 {
        if !(hi[k] > lo[k]) {
            return Err(Error::invalid("grid bounds are empty"));
        }
        shape[k] = ((hi[k] - lo[k]) / params.voxel_size).ceil().max(1.0) as usize;
    }
    Ok(shape)
}

/// Terrain height under the centre of every column, `i + nx j`.
pub(crate) fn column_ground(dtm: &Raster, lo: Point, size: f64, shape: [usize; 3]) -> Vec<f64> {
    let mut g = Vec::with_capacity(shape[0] * shape[1]);
    for j in 0..shape[1] {
        for i in 0..shape[0] {
            g.push(dtm.sample(lo[0] + (i as f64 + 0.5) * size, lo[1] + (j as f64 + 0.5) * size));
        }
    }
    g
}

/// Bounds fitted to the echoes, grown by 0.1 mm: an echo exactly on the max
/// face would otherwise fall in the cell beyond the last one and be dropped,
/// and echoes decoded from a file's f32 angles and ranges can sit a hair
/// outside the box recorded from the f64 originals. The same margin in memory
/// and from a file keeps the two grids identical.
pub(crate) fn pad_bounds(b: (Point, Point)) -> (Point, Point) {
    let (mut lo, mut hi) = b;
    for k in 0..3 {
        let eps = 1e-4f64.max(hi[k].abs() * 1e-9);
        lo[k] -= eps;
        hi[k] += eps;
    }
    (lo, hi)
}

/// Trace `inputs.shots` through a voxel grid and accumulate the statistics.
pub fn voxelize(inputs: &VoxelInputs, params: &VoxelParams) -> Result<RayVoxels> {
    let bounds = match params.bounds {
        Some(b) => b,
        None => {
            let xyz = inputs.shots.echo_xyz();
            if xyz.is_empty() {
                return Err(Error::invalid("no echoes to fit the grid to; pass bounds"));
            }
            pad_bounds((crate::util::spatial::min_corner(&xyz), crate::util::spatial::max_corner(&xyz)))
        }
    };
    let mut v = Voxelizer::new(params, bounds, inputs.dtm)?;
    if params.flat_top {
        v.add_peaks(inputs.shots);
    }
    v.add(inputs)?;
    v.finish()
}

/// How to derive the per-echo annotations of [`VoxelInputs`] from echo
/// attributes, for pulses that are streamed from a file.
#[derive(Debug, Clone, PartialEq)]
pub struct EchoLabels {
    /// Attribute holding the class codes below.
    pub class_attr: String,
    /// Class code meaning ground; `None` tells ground by height over the DTM instead.
    pub ground_class: Option<i64>,
    /// Without a `ground_class`: echoes at most this far above the DTM are ground.
    pub ground_distance: f64,
    /// Codes that mean leaf / wood. With either given, other codes below 3
    /// are excluded and the rest are plant (as rayvoxel).
    pub leaf_classes: Vec<i64>,
    /// Codes that mean wood (see `leaf_classes`).
    pub wood_classes: Vec<i64>,
    /// Attribute holding each echo's tree id, for the inclination distributions.
    pub tree_attr: String,
    /// Attribute holding each echo's intensity, for `Relative` / `Strongest` weighting.
    pub intensity_attr: String,
}

impl Default for EchoLabels {
    fn default() -> Self {
        EchoLabels { class_attr: "classification".into(), ground_class: None, ground_distance: 0.2, leaf_classes: vec![], wood_classes: vec![], tree_attr: "tree_id".into(), intensity_attr: "intensity".into() }
    }
}

/// Owned per-echo annotations.
#[derive(Debug, Clone, Default)]
pub struct EchoAnnotations {
    /// Ground flag per echo; `None` when no ground rule applied.
    pub ground: Option<Vec<bool>>,
    /// Class code per echo (see [`foliage`]); `None` when no leaf or wood classes were given.
    pub foliage: Option<Vec<u8>>,
    /// Intensity per echo, when the attribute exists.
    pub intensity: Option<Vec<f64>>,
    /// Tree id per echo, when the attribute exists.
    pub tree_id: Option<Vec<i32>>,
}

impl EchoLabels {
    /// The per-echo annotations of `shots` from its echo attributes (and the DTM for
    /// ground by height); an absent attribute leaves its annotation `None`, except
    /// the class attribute, which is required when a rule needs it.
    pub fn annotate(&self, shots: &Shots, dtm: Option<&Raster>) -> Result<EchoAnnotations> {
        let class = shots.echo_attrs.get(&self.class_attr).map(|a| a.to_f64());
        let need_class = || class.as_ref().ok_or_else(|| Error::invalid(format!("the shots have no {:?} echo attribute", self.class_attr)));
        let ground = match (self.ground_class, dtm) {
            (Some(c), _) => Some(need_class()?.iter().map(|&v| v == c as f64).collect()),
            (None, Some(d)) if self.ground_distance > 0.0 => Some(ground_mask_from_dtm(shots, d, self.ground_distance)),
            _ => None,
        };
        let foliage = if self.leaf_classes.is_empty() && self.wood_classes.is_empty() {
            None
        } else {
            let code = |v: f64| {
                if self.leaf_classes.iter().any(|&c| c as f64 == v) {
                    foliage::LEAF
                } else if self.wood_classes.iter().any(|&c| c as f64 == v) {
                    foliage::WOOD
                } else if v < 3.0 {
                    foliage::EXCLUDED
                } else {
                    foliage::PLANT
                }
            };
            Some(need_class()?.iter().map(|&v| code(v)).collect())
        };
        Ok(EchoAnnotations {
            ground,
            foliage,
            intensity: shots.echo_attrs.get(&self.intensity_attr).map(|a| a.to_f64()),
            tree_id: shots.echo_attrs.get(&self.tree_attr).map(|a| a.to_f64().iter().map(|&v| v as i32).collect()),
        })
    }
}

impl EchoAnnotations {
    /// Borrow these annotations, `shots` and the DTM as the inputs of a trace.
    pub fn inputs<'a>(&'a self, shots: &'a Shots, dtm: Option<&'a Raster>) -> VoxelInputs<'a> {
        VoxelInputs { shots, ground: self.ground.as_deref(), foliage: self.foliage.as_deref(), intensity: self.intensity.as_deref(), tree_id: self.tree_id.as_deref(), dtm }
    }
}

/// Voxelise a shots file one row group at a time, decoding ahead while the
/// current group is traced; the grid defaults to the echo bounding
/// box recorded in the file.
pub fn voxelize_file(file: &crate::io::shots::ShotsFile, params: &VoxelParams, labels: &EchoLabels, dtm: Option<&Raster>) -> Result<RayVoxels> {
    // The recorded box comes from f64 echoes; the stored angles and ranges
    // may be f32, so decoded echoes can sit a hair outside it.
    let mut v = Voxelizer::new(params, params.bounds.unwrap_or_else(|| pad_bounds(file.bounds)), dtm)?;
    if params.flat_top {
        for g in 0..file.n_groups() {
            v.add_peaks(&file.read_group(g)?);
        }
    }
    for_each_file_batch(file, |shots| {
        let notes = labels.annotate(&shots, dtm)?;
        v.add(&notes.inputs(&shots, dtm))
    })?;
    v.finish()
}

/// Hand the pulses of a shots file to `f` in batches of a few row groups,
/// decoding ahead while a batch is traced. The order is fixed: the same
/// batches in the same order on every call.
pub(crate) fn for_each_file_batch(file: &crate::io::shots::ShotsFile, mut f: impl FnMut(Shots) -> Result<()>) -> Result<()> {
    // Pulses fired together cross the same voxels, and threads adding to the
    // same voxels stall each other. So row groups are visited in strides that
    // put distant parts of the file next to each other, and traced a few at a time.
    const BATCH: usize = 4;
    let n = file.n_groups();
    let stride = (1..=n).rev().find(|s| *s <= n.div_ceil(BATCH).max(1) && gcd(*s, n) == 1).unwrap_or(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::sync_channel(BATCH);
        for _ in 0..n.clamp(1, BATCH) {
            let (tx, next, handle) = (tx.clone(), &next, file.reopen()?);
            scope.spawn(move || loop {
                let g = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if g >= n || tx.send((g, handle.read_group(g * stride % n))).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        // Decoders finish out of turn; tracing in a fixed order keeps results reproducible.
        let mut early = BTreeMap::new();
        let mut batch = Shots::default();
        let mut due = 0;
        for (g, part) in rx {
            early.insert(g, part);
            while let Some(part) = early.remove(&due) {
                crate::io::shots::append(&mut batch, part?)?;
                due += 1;
                if due % BATCH == 0 || due == n {
                    f(std::mem::take(&mut batch))?;
                }
            }
        }
        Ok::<(), Error>(())
    })
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A pulse without echoes is traced this far.
pub(crate) fn unbounded_end(origin: &Point, dir: &Point, geom: &traverse::Geom, range: f64) -> Point {
    let r = if range.is_finite() {
        range
    } else {
        let mut to_centre = 0.0;
        let mut diag = 0.0;
        for k in 0..3 {
            let ext = geom.shape[k] as f64 * geom.size;
            to_centre += (geom.origin[k] + 0.5 * ext - origin[k]).powi(2);
            diag += ext * ext;
        }
        to_centre.sqrt() + diag.sqrt()
    };
    add(origin, &scale(dir, r))
}
