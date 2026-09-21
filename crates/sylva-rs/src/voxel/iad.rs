//! Inclination angle distributions from echo normals, per tree.
//!
//! Normals come from a PCA of each echo's nearest neighbours; their
//! inclination `acos |n_z|` is binned over `[0, π/2]` into plant / leaf / wood
//! histograms per tree, and `G` is the projection kernel integrated over both
//! the histogram and the tree's own distribution of beam zeniths (Vicari,
//! Pisek & Disney 2019, Agric. For. Meteorol. 264). For the Bailey method the
//! histograms are built from area-weighted triangle facets between
//! neighbouring same-class echoes (Bailey & Mahaffee 2017).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::FRAC_PI_2;

use rayon::prelude::*;

use super::metrics::{classify_de_wit, compute_g_from_histogram};
use super::traverse::Geom;
use super::{foliage, Attenuation, RayVoxels, VoxelInputs};
use crate::spatial::KdTree;
use crate::transform::{add, cross, norm, scale, sub};
use crate::Point;

/// Inclination distributions and projection coefficients of one tree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TreeIad {
    /// Bin centres (rad).
    pub bin_centres: Vec<f64>,
    /// Normalised leaf / wood / plant inclination histograms.
    pub liad: Vec<f64>,
    pub wiad: Vec<f64>,
    pub piad: Vec<f64>,
    /// Triangle-facet histograms (Bailey only, otherwise empty).
    pub liad_bailey: Vec<f64>,
    pub wiad_bailey: Vec<f64>,
    pub piad_bailey: Vec<f64>,
    pub g_leaf: f64,
    pub g_wood: f64,
    pub g_plant: f64,
    pub bailey_g_leaf: f64,
    pub bailey_g_wood: f64,
    pub leaf_hits: u32,
    pub wood_hits: u32,
    /// Closest de Wit type of each histogram.
    pub liad_de_wit: Option<&'static str>,
    pub wiad_de_wit: Option<&'static str>,
    pub piad_de_wit: Option<&'static str>,
}

fn normalize(h: &mut [f64]) {
    let s: f64 = h.iter().sum();
    if s > 0.0 {
        h.iter_mut().for_each(|v| *v /= s);
    }
}

fn bin_of(theta: f64, n: usize) -> usize {
    ((theta / FRAC_PI_2 * n as f64) as usize).min(n - 1)
}

#[derive(Default, Clone)]
struct Facets {
    leaf: Vec<f64>,
    wood: Vec<f64>,
    g_num: [f64; 2],
    g_den: [f64; 2],
}

/// Non-ground echoes inside the grid, gathered while pulses are added.
#[derive(Default)]
pub(crate) struct EchoPoints {
    pts: Vec<Point>,
    /// Voxel, foliage code, tree and beam zenith of each point.
    meta: Vec<(usize, u8, i32, f64)>,
}

impl EchoPoints {
    pub fn collect(&mut self, inputs: &VoxelInputs, geom: &Geom) {
        let shots = inputs.shots;
        for s in 0..shots.n_shots() {
            let (o, d) = (shots.origin[s], shots.direction[s]);
            let beam = d[2].abs().min(1.0).acos();
            let first = shots.echo_start[s];
            for e in first..first + shots.echo_count[s] as usize {
                if inputs.ground.is_some_and(|g| g[e]) {
                    continue;
                }
                let p = add(&o, &scale(&d, shots.echo_range[e]));
                let Some(idx) = geom.flat(geom.cell_of(&p)) else { continue };
                self.pts.push(p);
                self.meta.push((idx, inputs.foliage.map_or(foliage::PLANT, |f| f[e]), inputs.tree_id.map_or(0, |t| t[e]), beam));
            }
        }
    }
}

pub(crate) fn build(vox: &mut RayVoxels, echoes: EchoPoints) {
    let nb = vox.params.n_iad_bins.max(1);
    let bailey = vox.params.attenuation.contains(&Attenuation::Bailey);
    let EchoPoints { pts, meta } = echoes;
    let mut predominant = vec![-1i32; vox.n_voxels()];
    if pts.len() < 3 {
        vox.predominant_tree = Some(predominant);
        return;
    }

    let k = vox.params.knn_normal.clamp(3, pts.len());
    let tree = KdTree::new(&pts);
    let neighbours: Vec<Vec<usize>> = pts.par_iter().map(|p| tree.knn(p, k).into_iter().map(|(i, _)| i).collect()).collect();
    let inclination: Vec<Option<f64>> = neighbours
        .par_iter()
        .map(|nbrs| {
            if nbrs.len() < 3 {
                return None;
            }
            let n = nbrs.len() as f64;
            let mut c = [0.0; 3];
            for &i in nbrs {
                for a in 0..3 {
                    c[a] += pts[i][a] / n;
                }
            }
            let mut cov = nalgebra::Matrix3::<f64>::zeros();
            for &i in nbrs {
                let d = sub(&pts[i], &c);
                cov += nalgebra::Vector3::new(d[0], d[1], d[2]) * nalgebra::RowVector3::new(d[0], d[1], d[2]);
            }
            let eig = cov.symmetric_eigen();
            let imin = (0..3).min_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap()).unwrap();
            Some(eig.eigenvectors.column(imin)[2].abs().min(1.0).acos())
        })
        .collect();

    #[derive(Default)]
    struct Hist {
        all: Vec<f64>,
        leaf: Vec<f64>,
        wood: Vec<f64>,
        beam: Vec<f64>,
        leaf_hits: u32,
        wood_hits: u32,
    }
    let mut hists: BTreeMap<i32, Hist> = BTreeMap::new();
    let mut tree_hits: HashMap<usize, BTreeMap<i32, u32>> = HashMap::new();
    for (i, theta) in inclination.iter().enumerate() {
        let Some(theta) = *theta else { continue };
        let (idx, fol, tid, beam) = meta[i];
        if tid < 0 {
            continue;
        }
        *tree_hits.entry(idx).or_default().entry(tid).or_default() += 1;
        let h = hists.entry(tid).or_insert_with(|| Hist { all: vec![0.0; nb], leaf: vec![0.0; nb], wood: vec![0.0; nb], beam: vec![0.0; nb], ..Default::default() });
        let b = bin_of(theta, nb);
        h.all[b] += 1.0;
        h.beam[bin_of(beam, nb)] += 1.0;
        match fol {
            foliage::LEAF => {
                h.leaf[b] += 1.0;
                h.leaf_hits += 1;
            }
            foliage::WOOD => {
                h.wood[b] += 1.0;
                h.wood_hits += 1;
            }
            _ => {}
        }
    }

    // Most frequent tree per voxel; ties go to the lowest id.
    for (idx, counts) in &tree_hits {
        let mut best = (0u32, -1i32);
        for (&tid, &c) in counts {
            if c > best.0 {
                best = (c, tid);
            }
        }
        predominant[*idx] = best.1;
    }

    let bin_centres: Vec<f64> = (0..nb).map(|b| (b as f64 + 0.5) * FRAC_PI_2 / nb as f64).collect();
    let mut out: BTreeMap<i32, TreeIad> = BTreeMap::new();
    for (tid, mut h) in hists {
        normalize(&mut h.all);
        normalize(&mut h.leaf);
        normalize(&mut h.wood);
        normalize(&mut h.beam);
        let g = |hist: &[f64]| h.beam.iter().zip(&bin_centres).filter(|(w, _)| **w > 0.0).map(|(w, &c)| w * compute_g_from_histogram(c, &bin_centres, hist)).sum::<f64>();
        out.insert(
            tid,
            TreeIad {
                g_plant: g(&h.all),
                g_leaf: g(&h.leaf),
                g_wood: g(&h.wood),
                liad_de_wit: classify_de_wit(&bin_centres, &h.leaf),
                wiad_de_wit: classify_de_wit(&bin_centres, &h.wood),
                piad_de_wit: classify_de_wit(&bin_centres, &h.all),
                bin_centres: bin_centres.clone(),
                liad: h.leaf,
                wiad: h.wood,
                piad: h.all,
                leaf_hits: h.leaf_hits,
                wood_hits: h.wood_hits,
                ..Default::default()
            },
        );
    }

    if bailey {
        let facets = triangle_facets(&pts, &neighbours, &meta, nb, vox.params.triangle_lmax);
        // Fold each voxel's facets into its predominant tree, area weighted.
        let mut per_tree: BTreeMap<i32, Facets> = BTreeMap::new();
        for (idx, f) in facets {
            let tid = predominant[idx];
            if tid < 0 {
                continue;
            }
            let t = per_tree.entry(tid).or_insert_with(|| Facets { leaf: vec![0.0; nb], wood: vec![0.0; nb], ..Default::default() });
            for b in 0..nb {
                t.leaf[b] += f.leaf[b];
                t.wood[b] += f.wood[b];
            }
            for c in 0..2 {
                t.g_num[c] += f.g_num[c];
                t.g_den[c] += f.g_den[c];
            }
        }
        for (tid, mut f) in per_tree {
            let Some(iad) = out.get_mut(&tid) else { continue };
            let mut plant: Vec<f64> = f.leaf.iter().zip(&f.wood).map(|(a, b)| a + b).collect();
            normalize(&mut plant);
            normalize(&mut f.leaf);
            normalize(&mut f.wood);
            iad.piad_bailey = plant;
            iad.liad_bailey = f.leaf;
            iad.wiad_bailey = f.wood;
            if f.g_den[0] > 0.0 {
                iad.bailey_g_leaf = f.g_num[0] / f.g_den[0];
            }
            if f.g_den[1] > 0.0 {
                iad.bailey_g_wood = f.g_num[1] / f.g_den[1];
            }
        }
    }

    vox.predominant_tree = Some(predominant);
    vox.tree_iad = out;
}

/// Triangles between each echo and pairs of its same-class neighbours with
/// all edges under `l_max`, binned by facet inclination with weight
/// `area · sin θ`; `G` per facet is `|n_z|` (vertical mean beam direction,
/// as in rayvoxel). Facets belong to the voxel of their first vertex.
fn triangle_facets(pts: &[Point], neighbours: &[Vec<usize>], meta: &[(usize, u8, i32, f64)], nb: usize, l_max: f64) -> HashMap<usize, Facets> {
    let l2 = l_max * l_max;
    let d2 = |a: usize, b: usize| {
        let d = sub(&pts[a], &pts[b]);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    let class = |i: usize| match meta[i].1 {
        foliage::LEAF => Some(0),
        foliage::WOOD => Some(1),
        _ => None,
    };
    let mut seen: HashSet<[usize; 3]> = HashSet::new();
    let mut out: HashMap<usize, Facets> = HashMap::new();
    for i in 0..pts.len() {
        let Some(ci) = class(i) else { continue };
        let nbrs = &neighbours[i];
        for (ja, &a) in nbrs.iter().enumerate() {
            if a == i || class(a) != Some(ci) || d2(a, i) > l2 {
                continue;
            }
            for &b in &nbrs[ja + 1..] {
                if b == i || b == a || class(b) != Some(ci) || d2(b, i) > l2 || d2(b, a) > l2 {
                    continue;
                }
                let mut tri = [i, a, b];
                tri.sort_unstable();
                if !seen.insert(tri) {
                    continue;
                }
                let n = cross(&sub(&pts[a], &pts[i]), &sub(&pts[b], &pts[i]));
                let len = norm(&n);
                if len < 1e-12 {
                    continue;
                }
                let cos_t = (n[2].abs() / len).clamp(0.0, 1.0);
                let theta = cos_t.acos();
                let w = 0.5 * len * theta.sin();
                let f = out.entry(meta[i].0).or_insert_with(|| Facets { leaf: vec![0.0; nb], wood: vec![0.0; nb], ..Default::default() });
                if ci == 0 { f.leaf[bin_of(theta, nb)] += w } else { f.wood[bin_of(theta, nb)] += w }
                f.g_num[ci] += cos_t * w;
                f.g_den[ci] += w;
            }
        }
    }
    out
}
