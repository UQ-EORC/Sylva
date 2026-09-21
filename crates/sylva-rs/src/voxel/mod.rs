//! Ray-traced voxel grid with AMAPVox-style Beer–Lambert statistics.
//!
//! A port of the `rayvoxel` tool from the raycloudtools fork (G. Eaton,
//! CSIRO), working on [`Shots`] instead of ray-cloud files. Every pulse is
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
//! What is not ported: the out-of-core shard path, the NetCDF writer and the
//! per-voxel LAS class histogram (only plant / leaf / wood hit counts are kept).

mod iad;
mod metrics;
mod refine;
mod traverse;
mod wood;
mod write;

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::qsm::Cylinder;
use crate::transform::{add, scale};
use crate::{Point, Raster, Shots};

pub use iad::TreeIad;
pub use metrics::{classify_de_wit, compute_g, compute_g_from_histogram, laser_spec, solve_bailey_pad, Lad};
pub use write::WriteOptions;

/// How the energy of a pulse is shared between its echoes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightMethod {
    /// `1 / n` per echo (AMAPVox `EqualEchoWeight`).
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
    Fpl,
    Ppl,
    Transmittance,
    Bailey,
}

impl Attenuation {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "fpl" => Attenuation::Fpl,
            "ppl" => Attenuation::Ppl,
            "transmittance" => Attenuation::Transmittance,
            "bailey" => Attenuation::Bailey,
            other => return Err(Error::invalid(format!("unknown attenuation method {other:?} (fpl|ppl|transmittance|bailey)"))),
        })
    }

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
pub struct VoxelParams {
    pub voxel_size: f64,
    /// Grid corners; `None` fits the grid to the (non-ground and ground) echoes.
    /// The max corner is snapped up to a whole number of voxels.
    pub bounds: Option<(Point, Point)>,
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
    pub attenuation: Vec<Attenuation>,
    /// Estimate inclination angle distributions from echo normals.
    pub inclination: bool,
    pub n_iad_bins: usize,
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
    pub const PLANT: u8 = 1;
    pub const LEAF: u8 = 2;
    pub const WOOD: u8 = 3;
}

/// Pulses plus optional per-echo annotations (all of length `shots.n_echoes()`).
#[derive(Debug, Clone, Copy)]
pub struct VoxelInputs<'a> {
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
    pub fn new(shots: &'a Shots) -> Self {
        VoxelInputs { shots, ground: None, foliage: None, intensity: None, tree_id: None, dtm: None }
    }
}

/// Per-voxel `f32` accumulators. Names follow the rayvoxel output columns.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F {
    /// Σ beam fraction entering the voxel (observed traversals).
    NumBeamsWeighted,
    /// Potential path length: Σ full voxel chord (AMAPVox `lgTotal`).
    PathLength,
    PathLengthSq,
    /// Σ fraction × free path (entry → echo, or the full chord on a miss).
    FreePathLength,
    FreePathLengthPlant,
    FreePathLengthLeaf,
    FreePathLengthWood,
    EffectiveFreePathLength,
    PathLengthOccluded,
    SumOfAngles,
    SumSinAzimuth,
    SumCosAzimuth,
    SumOfLaserDistances,
    SumHitDelta,
    SumMissDelta,
    PathLengthUnbound,
    BsEntering,
    BsIntercepted,
    BsPotential,
    BsFreePath,
    BsEffectiveFreePath,
    BsEffFreePathHits,
    PplMissWl,
}

impl F {
    pub const COUNT: usize = 23;
    pub const ALL: [F; F::COUNT] = [
        F::NumBeamsWeighted, F::PathLength, F::PathLengthSq, F::FreePathLength, F::FreePathLengthPlant,
        F::FreePathLengthLeaf, F::FreePathLengthWood, F::EffectiveFreePathLength, F::PathLengthOccluded,
        F::SumOfAngles, F::SumSinAzimuth, F::SumCosAzimuth, F::SumOfLaserDistances, F::SumHitDelta,
        F::SumMissDelta, F::PathLengthUnbound, F::BsEntering, F::BsIntercepted, F::BsPotential,
        F::BsFreePath, F::BsEffectiveFreePath, F::BsEffFreePathHits, F::PplMissWl,
    ];

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
    NumHits,
    NumBeams,
    NumRaysOccluded,
    NumUnboundRays,
    NumMissRays,
    NumHitLeaf,
    NumHitWood,
    NumHitPlant,
}

impl I {
    pub const COUNT: usize = 8;
    pub const ALL: [I; I::COUNT] = [
        I::NumHits, I::NumBeams, I::NumRaysOccluded, I::NumUnboundRays, I::NumMissRays, I::NumHitLeaf,
        I::NumHitWood, I::NumHitPlant,
    ];

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
    pub origin: Point,
    pub voxel_size: f64,
    /// `[nx, ny, nz]`.
    pub shape: [usize; 3],
    pub params: VoxelParams,
    pub has_leaf: bool,
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
    pub tree_iad: BTreeMap<i32, TreeIad>,
}

impl RayVoxels {
    pub fn n_voxels(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    #[inline]
    pub fn get_f(&self, field: F, idx: usize) -> f32 {
        self.f[field as usize].get(idx).copied().unwrap_or(0.0)
    }

    #[inline]
    pub fn get_i(&self, field: I, idx: usize) -> i32 {
        self.i[field as usize][idx]
    }

    /// `(i, j, k)` of a flat index.
    #[inline]
    pub fn unravel(&self, idx: usize) -> [usize; 3] {
        let (nx, ny) = (self.shape[0], self.shape[1]);
        [idx % nx, (idx / nx) % ny, idx / (nx * ny)]
    }

    pub fn center(&self, idx: usize) -> Point {
        let ijk = self.unravel(idx);
        [
            self.origin[0] + (ijk[0] as f64 + 0.5) * self.voxel_size,
            self.origin[1] + (ijk[1] as f64 + 0.5) * self.voxel_size,
            self.origin[2] + (ijk[2] as f64 + 0.5) * self.voxel_size,
        ]
    }

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
        let (lo, hi) = bounds;
        let mut shape = [0usize; 3];
        for k in 0..3 {
            if !(hi[k] > lo[k]) {
                return Err(Error::invalid("grid bounds are empty"));
            }
            shape[k] = ((hi[k] - lo[k]) / params.voxel_size).ceil().max(1.0) as usize;
        }
        let ground_height = dtm.map(|dtm| {
            let mut g = Vec::with_capacity(shape[0] * shape[1]);
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    g.push(dtm.sample(lo[0] + (i as f64 + 0.5) * params.voxel_size, lo[1] + (j as f64 + 0.5) * params.voxel_size));
                }
            }
            g
        });
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

    pub fn add(&mut self, inputs: &VoxelInputs) -> Result<()> {
        let ne = inputs.shots.n_echoes();
        check_len("ground", inputs.ground, ne)?;
        check_len("foliage", inputs.foliage, ne)?;
        check_len("intensity", inputs.intensity, ne)?;
        check_len("tree_id", inputs.tree_id, ne)?;
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

/// Bounds fitted to the echoes, grown by 0.1 mm: an echo exactly on the max
/// face would otherwise fall in the cell beyond the last one and be dropped,
/// and echoes decoded from a file's f32 angles and ranges can sit a hair
/// outside the box recorded from the f64 originals. The same margin in memory
/// and from a file keeps the two grids identical.
fn pad_bounds(b: (Point, Point)) -> (Point, Point) {
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
            pad_bounds((crate::spatial::min_corner(&xyz), crate::spatial::max_corner(&xyz)))
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
    pub ground_class: Option<i64>,
    /// Without a `ground_class`: echoes at most this far above the DTM are ground.
    pub ground_distance: f64,
    /// Codes that mean leaf / wood. With either given, other codes below 3
    /// are excluded and the rest are plant (as rayvoxel).
    pub leaf_classes: Vec<i64>,
    pub wood_classes: Vec<i64>,
    pub tree_attr: String,
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
    pub ground: Option<Vec<bool>>,
    pub foliage: Option<Vec<u8>>,
    pub intensity: Option<Vec<f64>>,
    pub tree_id: Option<Vec<i32>>,
}

impl EchoLabels {
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
                    let shots = std::mem::take(&mut batch);
                    let notes = labels.annotate(&shots, dtm)?;
                    v.add(&notes.inputs(&shots, dtm))?;
                }
            }
        }
        Ok::<(), Error>(())
    })?;
    v.finish()
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
