// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! The foliage model the language packages present: leaf area grids, leaf
//! shapes, textbook angle distributions, leaf insertion from a grid or a
//! total, and the tree OBJ file.
//!
//! [`crate::leaves`] holds the algorithms (normals, area from points, the
//! leaf/wood filters, placing blades); this module holds the containers and
//! the steps that turn a user's request into calls to them, as the Python
//! package did in NumPy. Arrays follow its layout: a grid's `density` is
//! row-major `(nz, ny, nx)`.

use std::f64::consts::PI;
use std::path::Path;

use crate::leaves::{self, LeafAngles, LeafBlade, LeafMesh, LeafParams};
use crate::io::mesh::{self, ObjMesh};
use crate::util::numeric::pairwise_sum;
use crate::qsm::model::{Cylinder, Qsm};
use crate::qsm::wood::GbsParams;
use crate::voxel::RayVoxels;
use crate::{Error, Point, Result};

// ------------------------------------------------------------ leaf / wood

/// The graph-based separation's shell sizes and direction limit for a tree:
/// the authors' two settings, 0.5-3 m shells and 27 degrees for trees 15 m
/// or taller, 0.1-1 m and 45 degrees below that. Given values are kept; an
/// empty cloud keeps the defaults of [`GbsParams`].
pub fn gbs_params_for(points: &[Point], intervals: Option<Vec<f64>>, max_angle: Option<f64>, base: GbsParams) -> GbsParams {
    let mut p = base;
    match intervals {
        Some(iv) => p.intervals = iv,
        None if !points.is_empty() => {
            let (lo, hi) = points.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| (lo.min(q[2]), hi.max(q[2])));
            let tall = hi - lo >= 15.0;
            p.intervals = if tall { vec![0.5, 1.0, 1.5, 2.0, 3.0] } else { vec![0.1, 0.2, 0.3, 0.5, 1.0] };
            p.max_angle = max_angle.unwrap_or(if tall { 0.15 * PI } else { 0.25 * PI });
            return p;
        }
        None => {}
    }
    if let Some(a) = max_angle {
        p.max_angle = a;
    }
    p
}

// ------------------------------------------------------------ angle types

/// Names of the de Wit (1965) distributions, as [`de_wit`] takes them.
pub const DE_WIT_TYPES: [&str; 6] = ["spherical", "uniform", "planophile", "erectophile", "plagiophile", "extremophile"];

/// A textbook de Wit (1965) distribution over `n_bins` bins of 0-90
/// degrees; `None` for an unknown name.
pub fn de_wit(name: &str, n_bins: usize) -> Option<LeafAngles> {
    let t: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * (PI / 2.0) / n_bins as f64).collect();
    let f: Vec<f64> = match name {
        "spherical" => t.iter().map(|x| x.sin()).collect(),
        "uniform" => vec![1.0; n_bins],
        "planophile" => t.iter().map(|x| 1.0 + (2.0 * x).cos()).collect(),
        "erectophile" => t.iter().map(|x| 1.0 - (2.0 * x).cos()).collect(),
        "plagiophile" => t.iter().map(|x| 1.0 - (4.0 * x).cos()).collect(),
        "extremophile" => t.iter().map(|x| 1.0 + (4.0 * x).cos()).collect(),
        _ => return None,
    };
    let s = pairwise_sum(&f);
    let f: Vec<f64> = f.iter().map(|v| v / s).collect();
    Some(leaves::angle_distribution(&t, Some(&f), n_bins))
}

// ------------------------------------------------------------ leaf area grid

/// One-sided leaf area density (m² m⁻³) on a regular grid.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafAreaGrid {
    /// Minimum corner.
    pub origin: Point,
    pub voxel_size: f64,
    /// `[nz, ny, nx]`.
    pub shape: [usize; 3],
    /// Row-major `(nz, ny, nx)`.
    pub density: Vec<f64>,
}

/// NumPy's `nan_to_num`: NaN to 0, infinities to the largest finite values.
fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        f64::MIN
    } else {
        v
    }
}

impl LeafAreaGrid {
    pub fn new(origin: Point, voxel_size: f64, shape: [usize; 3], density: Vec<f64>) -> Result<Self> {
        if density.len() != shape[0] * shape[1] * shape[2] {
            return Err(Error::invalid(format!("density has {} values for a {}x{}x{} grid", density.len(), shape[0], shape[1], shape[2])));
        }
        Ok(LeafAreaGrid { origin, voxel_size, shape, density })
    }

    /// Leaf area per voxel (m²).
    pub fn area(&self) -> Vec<f64> {
        let v = self.voxel_size.powf(3.0);
        self.density.iter().map(|d| d * v).collect()
    }

    /// Total one-sided leaf area (m²), NaN voxels left out.
    pub fn total_area(&self) -> f64 {
        let a: Vec<f64> = self.area().into_iter().map(|x| if x.is_nan() { 0.0 } else { x }).collect();
        pairwise_sum(&a)
    }

    /// The same pattern rescaled to a total area (all zero if the grid holds none).
    pub fn scaled_to(&self, total_area: f64) -> LeafAreaGrid {
        let t = self.total_area();
        let k = if t > 0.0 { total_area / t } else { 0.0 };
        LeafAreaGrid { density: self.density.iter().map(|d| d * k).collect(), ..self.clone() }
    }

    /// Layer centres (grid z) and the leaf area of each layer.
    pub fn profile(&self) -> (Vec<f64>, Vec<f64>) {
        let [nz, ny, nx] = self.shape;
        let z = (0..nz).map(|k| self.origin[2] + (k as f64 + 0.5) * self.voxel_size).collect();
        let a: Vec<f64> = self.area().into_iter().map(|x| if x.is_nan() { 0.0 } else { x }).collect();
        let per = ny * nx;
        let area = (0..nz).map(|k| pairwise_sum(&a[k * per..(k + 1) * per])).collect();
        (z, area)
    }

    /// Centres and leaf area of the voxels holding any, in row-major order.
    pub fn cells(&self) -> (Vec<Point>, Vec<f64>) {
        let [_, ny, nx] = self.shape;
        let mut centres = Vec::new();
        let mut area = Vec::new();
        for (idx, a) in self.area().into_iter().enumerate() {
            let a = nan_to_num(a);
            if a > 0.0 {
                let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
                centres.push([
                    self.origin[0] + (i as f64 + 0.5) * self.voxel_size,
                    self.origin[1] + (j as f64 + 0.5) * self.voxel_size,
                    self.origin[2] + (k as f64 + 0.5) * self.voxel_size,
                ]);
                area.push(a);
            }
        }
        (centres, area)
    }

    /// Leaf (or plant) area density of a ray-traced grid by field or metric
    /// name (`"lad_fpl"`, `"pad_fpl"`, ...); unobserved (NaN) voxels become 0.
    pub fn from_voxels(grid: &RayVoxels, field: &str) -> Result<LeafAreaGrid> {
        let [nx, ny, nz] = grid.shape;
        let d = grid.values(field)?.into_iter().map(nan_to_num).collect();
        LeafAreaGrid::new(grid.origin, grid.voxel_size, [nz, ny, nx], d)
    }
}

/// Leaf area density from leaf points alone (see
/// [`leaves::point_leaf_area`]), on a grid aligned to multiples of
/// `voxel_size`; `res <= 0` picks the thinning from the point spacing.
pub fn leaf_area_density(points: &[Point], voxel_size: f64, res: f64, k: usize) -> Result<LeafAreaGrid> {
    let (pts, area, _) = leaves::point_leaf_area(points, res, k);
    if pts.is_empty() {
        return Ok(LeafAreaGrid { origin: [0.0; 3], voxel_size, shape: [1, 1, 1], density: vec![0.0] });
    }
    if area.len() != pts.len() {
        return Err(Error::invalid("too few leaf points to estimate their area"));
    }
    let mut origin = [f64::INFINITY; 3];
    for p in &pts {
        for a in 0..3 {
            origin[a] = origin[a].min(p[a]);
        }
    }
    for o in &mut origin {
        *o = (*o / voxel_size).floor() * voxel_size;
    }
    let idx: Vec<[usize; 3]> = pts.iter().map(|p| std::array::from_fn(|a| ((p[a] - origin[a]) / voxel_size).floor() as usize)).collect();
    let mut n = [0usize; 3];
    for i in &idx {
        for a in 0..3 {
            n[a] = n[a].max(i[a] + 1);
        }
    }
    let mut dens = vec![0.0; n[0] * n[1] * n[2]];
    for (i, a) in idx.iter().zip(&area) {
        dens[i[0] + n[0] * (i[1] + n[1] * i[2])] += a;
    }
    let v = voxel_size.powf(3.0);
    dens.iter_mut().for_each(|d| *d /= v);
    Ok(LeafAreaGrid { origin, voxel_size, shape: [n[2], n[1], n[0]], density: dens })
}

// ------------------------------------------------------------ leaf shapes

/// The blade one leaf is cut from, in unit leaf space (see [`LeafBlade`]),
/// and its size.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafShape {
    pub vertices: Vec<Point>,
    pub faces: Vec<[u32; 3]>,
    /// Blade length and greatest width (m).
    pub length: f64,
    pub width: f64,
}

impl Default for LeafShape {
    /// The built-in six-sided blade at 8 x 4 cm.
    fn default() -> Self {
        let b = LeafBlade::default();
        LeafShape { vertices: b.vertices, faces: b.faces, length: 0.08, width: 0.04 }
    }
}

impl LeafShape {
    /// A shape, checked: faces must be non-empty and index the vertices, and
    /// the size positive. Missing vertices or faces are the built-in blade's.
    pub fn new(vertices: Option<Vec<Point>>, faces: Option<Vec<[u32; 3]>>, length: f64, width: f64) -> Result<Self> {
        let b = LeafBlade::default();
        let vertices = vertices.unwrap_or(b.vertices);
        let faces = faces.unwrap_or(b.faces);
        if faces.is_empty() || faces.iter().flatten().any(|&i| i as usize >= vertices.len()) {
            return Err(Error::invalid("faces must be non-empty and index vertices"));
        }
        if !(length > 0.0 && width > 0.0) {
            return Err(Error::invalid("length and width must be positive"));
        }
        Ok(LeafShape { vertices, faces, length, width })
    }

    fn blade(&self) -> LeafBlade {
        LeafBlade { vertices: self.vertices.clone(), faces: self.faces.clone() }
    }

    /// One-sided area of one leaf (m²): the sum of its triangles.
    pub fn area(&self) -> f64 {
        self.blade().area(self.length, self.width)
    }

    /// The same blade at a new size; `None` keeps the current one.
    pub fn resized(&self, length: Option<f64>, width: Option<f64>) -> Result<Self> {
        LeafShape::new(Some(self.vertices.clone()), Some(self.faces.clone()), length.unwrap_or(self.length), width.unwrap_or(self.width))
    }

    /// The same blade and aspect ratio at a one-sided area (m²).
    pub fn scaled_to(&self, area: f64) -> Result<Self> {
        if area.is_nan() || area <= 0.0 {
            return Err(Error::invalid("area must be positive"));
        }
        let k = (area / self.area()).sqrt();
        self.resized(Some(self.length * k), Some(self.width * k))
    }

    /// A custom blade from a mesh of one leaf: base at the smallest x, tip
    /// along +x, blade across y, curl in z. With `normalise` the mesh is
    /// mapped into unit leaf space and its own extent is the default size;
    /// without, the vertices are taken as unit leaf space and the size
    /// defaults to 8 x 4 cm (a zero size counts as none).
    pub fn from_mesh(vertices: Vec<Point>, faces: Vec<[u32; 3]>, length: Option<f64>, width: Option<f64>, normalise: bool) -> Result<Self> {
        if vertices.is_empty() {
            return Err(Error::invalid("vertices must be (V, 3) or (V, 2)"));
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in &vertices {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        if !normalise {
            let or = |v: Option<f64>, d: f64| v.filter(|&x| x != 0.0).unwrap_or(d);
            return LeafShape::new(Some(vertices), Some(faces), or(length, 0.08), or(width, 0.04));
        }
        if !(size[0] > 0.0 && size[1] > 0.0) {
            return Err(Error::invalid("the mesh has no extent along or across the blade"));
        }
        let shift = [lo[0], 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
        let div = [size[0], size[1], size[1]];
        let v = vertices.iter().map(|p| std::array::from_fn(|a| (p[a] - shift[a]) / div[a])).collect();
        LeafShape::new(Some(v), Some(faces), length.unwrap_or(size[0]), width.unwrap_or(size[1]))
    }

    /// A custom blade from an OBJ file holding one leaf (see
    /// [`mesh::parse_obj`] for what is read), oriented as for [`LeafShape::from_mesh`].
    pub fn from_obj(path: impl AsRef<Path>, length: Option<f64>, width: Option<f64>, normalise: bool) -> Result<Self> {
        let (v, f) = mesh::read_obj(path)?;
        LeafShape::from_mesh(v, f, length, width, normalise)
    }
}

/// One-sided area of one leaf of `shape` (the built-in blade if `None`) at
/// the given length and width.
pub fn single_leaf_area(length: f64, width: f64, shape: Option<&LeafShape>) -> Result<f64> {
    let base = shape.cloned().unwrap_or_default();
    Ok(base.resized(Some(length), Some(width))?.area())
}

// ------------------------------------------------------------ leaf insertion

/// Where leaf area goes: a grid, or a total to spread over the leaf points.
#[derive(Debug, Clone, Copy)]
pub enum LeafArea<'a> {
    Grid(&'a LeafAreaGrid),
    Total(f64),
}

/// Options of [`add_leaves`] other than the inputs.
#[derive(Debug, Clone)]
pub struct AddLeaves {
    /// Blade and size of every leaf.
    pub shape: LeafShape,
    pub max_branch_distance: f64,
    pub jitter: f64,
    pub seed: u64,
}

/// Leaf polygons for a QSM (`cylinders` may be empty): each voxel of the
/// grid receives leaves until its area is met, centred on `seeds` in it (see
/// [`leaves::insert_leaves`]). A total area is first spread over `seeds` by
/// [`leaf_area_density`] at 0.25 m. Placement runs in the grid's frame, so
/// the result does not depend on how far the plot is from the coordinate
/// origin.
pub fn add_leaves(area: LeafArea, seeds: &[Point], angles: &LeafAngles, cylinders: &[Cylinder], opts: &AddLeaves) -> Result<LeafMesh> {
    if angles.bin_centres.len() != angles.density.len() || angles.density.is_empty() {
        return Err(Error::invalid("bin_centres and density must be non-empty and equal in length"));
    }
    let scaled;
    let grid = match area {
        LeafArea::Grid(g) => g,
        LeafArea::Total(t) => {
            if seeds.is_empty() {
                return Err(Error::invalid("a total leaf area needs leaf_points to distribute it over"));
            }
            scaled = leaf_area_density(seeds, 0.25, 0.0, 12)?.scaled_to(t);
            &scaled
        }
    };
    let (centres, cell_area) = grid.cells();
    let o = grid.origin;
    let shift = |p: &Point| [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    let cells: Vec<(Point, f64)> = centres.iter().map(shift).zip(cell_area).collect();
    let seeds: Vec<Point> = seeds.iter().map(shift).collect();
    let cyl: Vec<Cylinder> = cylinders.iter().map(|c| Cylinder { start: shift(&c.start), ..c.clone() }).collect();
    let params = LeafParams { length: opts.shape.length, width: opts.shape.width, blade: opts.shape.blade(), max_branch_distance: opts.max_branch_distance, jitter: opts.jitter, seed: opts.seed };
    let mut mesh = leaves::insert_leaves(&cells, grid.voxel_size, &seeds, angles, &cyl, &params);
    let back = |p: &mut Point| {
        for a in 0..3 {
            p[a] += o[a];
        }
    };
    mesh.vertices.iter_mut().for_each(back);
    mesh.centres.iter_mut().for_each(back);
    Ok(mesh)
}

// ------------------------------------------------------------ OBJ files

/// A QSM from `(n, 12)` cylinder rows (see [`Qsm::to_rows`]).
pub fn qsm_from_rows(rows: &[[f64; 12]]) -> Qsm {
    Qsm {
        cylinders: rows
            .iter()
            .map(|r| Cylinder {
                start: [r[0], r[1], r[2]],
                axis: [r[3], r[4], r[5]],
                length: r[6],
                radius: r[7],
                parent: r[8] as i64,
                branch_order: r[9] as u32,
                branch_id: r[10] as u32,
                n_points: r[11] as usize,
            })
            .collect(),
    }
}

/// Write leaves as one OBJ object named `leaves`.
pub fn write_leaf_obj(path: impl AsRef<Path>, vertices: &[Point], faces: &[[u32; 3]]) -> Result<()> {
    mesh::write_obj(path, &[ObjMesh { name: "leaves", vertices, faces }])
}

/// Write the wood cylinders (`sides` facets round, one tube per branch if
/// `contiguous`) and the leaves to one OBJ, as objects `wood` and `leaves`.
pub fn write_tree_obj(path: impl AsRef<Path>, model: &Qsm, sides: usize, contiguous: bool, vertices: &[Point], faces: &[[u32; 3]]) -> Result<()> {
    let (wv, wf, _) = if contiguous { model.mesh_contiguous(sides) } else { model.mesh(sides) };
    mesh::write_obj(path, &[ObjMesh { name: "wood", vertices: &wv, faces: &wf }, ObjMesh { name: "leaves", vertices, faces }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn de_wit_types_order_by_mean_inclination() {
        let m = |n: &str| de_wit(n, 18).unwrap().mean;
        assert!(m("planophile") < m("plagiophile") && m("plagiophile") < m("erectophile"));
        assert!((de_wit("spherical", 18).unwrap().density.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(de_wit("round", 18).is_none());
    }

    #[test]
    fn grid_area_profile_cells_and_scaling() {
        let g = LeafAreaGrid::new([1.0, 2.0, 3.0], 0.5, [2, 1, 2], vec![1.0, f64::NAN, 0.0, 2.0]).unwrap();
        assert!((g.total_area() - 3.0 * 0.125).abs() < 1e-15);
        let (z, a) = g.profile();
        assert_eq!(z, vec![3.25, 3.75]);
        assert_eq!(a, vec![0.125, 0.25]);
        let (c, ca) = g.cells();
        assert_eq!(c, vec![[1.25, 2.25, 3.25], [1.75, 2.25, 3.75]]);
        assert_eq!(ca, vec![0.125, 0.25]);
        assert!((g.scaled_to(6.0).total_area() - 6.0).abs() < 1e-12);
        let empty = LeafAreaGrid::new([0.0; 3], 1.0, [1, 1, 1], vec![0.0]).unwrap();
        assert_eq!(empty.scaled_to(5.0).total_area(), 0.0);
    }

    #[test]
    fn density_from_points_sits_on_multiples_of_the_voxel() {
        let mut pts = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                pts.push([1.03 + i as f64 * 0.005, 2.01 + j as f64 * 0.005, 0.77]);
            }
        }
        let g = leaf_area_density(&pts, 0.25, 0.0, 12).unwrap();
        assert_eq!(g.origin, [1.0, 2.0, 0.75]);
        assert!((g.total_area() - 0.04).abs() < 0.01, "{}", g.total_area());
        assert_eq!(leaf_area_density(&[], 0.25, 0.0, 12).unwrap().shape, [1, 1, 1]);
    }

    #[test]
    fn shapes_check_scale_and_normalise() {
        let s = LeafShape::default();
        assert!((s.area() - single_leaf_area(0.08, 0.04, None).unwrap()).abs() < 1e-15);
        assert!((s.scaled_to(0.01).unwrap().area() - 0.01).abs() < 1e-12);
        assert!(LeafShape::new(None, Some(vec![[0, 1, 9]]), 0.1, 0.1).is_err());
        assert!(LeafShape::new(None, None, 0.0, 0.1).is_err());
        let v = vec![[0.3, -0.2, 1.0], [0.4, -0.17, 1.01], [0.5, -0.2, 1.0], [0.4, -0.23, 1.01]];
        let m = LeafShape::from_mesh(v.clone(), vec![[0, 1, 2], [0, 2, 3]], None, None, true).unwrap();
        assert!((m.length - 0.2).abs() < 1e-12 && (m.width - 0.06).abs() < 1e-12);
        assert!((m.vertices[2][0] - 1.0).abs() < 1e-12 && m.vertices[0][1].abs() < 1e-12);
        let raw = LeafShape::from_mesh(v, vec![[0, 1, 2]], Some(0.0), None, false).unwrap();
        assert_eq!((raw.length, raw.width), (0.08, 0.04));
    }

    #[test]
    fn a_total_needs_points_and_is_met() {
        let a = de_wit("spherical", 18).unwrap();
        let opts = AddLeaves { shape: LeafShape::default(), max_branch_distance: 0.5, jitter: 0.01, seed: 1 };
        assert!(add_leaves(LeafArea::Total(1.0), &[], &a, &[], &opts).is_err());
        let pts: Vec<Point> = (0..3000).map(|i| [100.0 + (i % 50) as f64 * 0.01, 200.0 + (i / 50) as f64 * 0.01, 5.0 + 0.001 * (i % 7) as f64]).collect();
        let m = add_leaves(LeafArea::Total(1.0), &pts, &a, &[], &opts).unwrap();
        assert!((m.total_area() - 1.0).abs() < m.leaf_area);
        assert!(m.centres.iter().all(|c| c[0] > 99.9 && c[1] > 199.9));
    }
}
