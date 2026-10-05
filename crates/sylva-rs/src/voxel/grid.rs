// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A ray-traced voxel grid as the language packages present it: arrays by
//! name, layer summaries of what the scan saw, voxel coordinates, the
//! per-tree sampling table, and the echo labels of in-memory pulses.
//!
//! Arrays are row-major `(nz, ny, nx)` (flat index `i + nx (j + ny k)`, as
//! [`RayVoxels`] stores them); maps are `(ny, nx)`. These are the NumPy
//! computations of the Python package's `RayVoxelGrid`, and follow them to
//! the bit, down to single-precision sums over single-precision fields.

use crate::util::numeric::{arange, pairwise_sum, pairwise_sum_f32, searchsorted_right};
use crate::raster::Raster;
use crate::shots::Shots;
use crate::voxel::quality::{tree_sampling, TreeSampling};
use crate::voxel::{self, EchoAnnotations, EchoLabels, RayVoxels, VoxelParams, WeightMethod, F, I};
use crate::{Error, Point, Result};

/// Values of the `state` metric.
pub const UNOBSERVED: u8 = 0;
pub const OCCLUDED: u8 = 1;
pub const EMPTY: u8 = 2;
pub const FILLED: u8 = 3;

/// A raw accumulator by name, in its stored precision.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldData {
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
}

impl FieldData {
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            FieldData::I32(v) => v.iter().map(|&x| x as f64).collect(),
            FieldData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            FieldData::F64(v) => v.clone(),
            FieldData::U8(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }
}

/// What the scan saw of the canopy space, layer by layer (see
/// [`RayVoxels::occlusion_profile`]).
#[derive(Debug, Clone, PartialEq)]
pub struct OcclusionProfile {
    /// Layer centres (m above the ground or the grid floor); one fewer than
    /// the other columns when the canopy space is thinner than a voxel.
    pub height: Vec<f64>,
    pub n_voxels: Vec<i64>,
    pub observed: Vec<f64>,
    pub occluded: Vec<f64>,
    pub unobserved: Vec<f64>,
    pub mean_beams: Vec<f64>,
    /// Plot totals: shares observed, occluded, unobserved, and the top used.
    pub total_observed: f64,
    pub total_occluded: f64,
    pub total_unobserved: f64,
    pub top: f64,
}

/// Heights, states and membership of every voxel, and the top used.
type CanopySpace = (Vec<f64>, Vec<u8>, Vec<bool>, f64);

impl RayVoxels {
    /// Names of the raw accumulators this grid holds (depends on the options).
    pub fn field_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = I::ALL.iter().map(|f| f.name()).collect();
        names.extend(F::ALL.iter().filter(|f| !self.f[**f as usize].is_empty()).map(|f| f.name()));
        for (name, on) in [("ppl_lambda", self.ppl_lambda.is_some()), ("wood_volume", self.wood_volume.is_some()), ("predominant_tree", self.predominant_tree.is_some()), ("subvoxel_counts", self.subvoxel_counts.is_some()), ("ground_height", self.ground_height.is_some())] {
            if on {
                names.push(name);
            }
        }
        names
    }

    /// A raw accumulator by name. `ground_height` is `(ny, nx)` and
    /// `subvoxel_counts` `(nz, ny, nx, split³)`.
    pub fn field(&self, name: &str) -> Result<FieldData> {
        let missing = || Error::invalid(format!("no voxel field {name:?} (see field_names())"));
        if let Some(f) = I::ALL.iter().find(|f| f.name() == name) {
            return Ok(FieldData::I32(self.i[*f as usize].clone()));
        }
        if let Some(f) = F::ALL.iter().find(|f| f.name() == name) {
            let data = &self.f[*f as usize];
            return if data.is_empty() { Err(missing()) } else { Ok(FieldData::F32(data.clone())) };
        }
        match name {
            "ppl_lambda" => Ok(FieldData::F32(self.ppl_lambda.clone().ok_or_else(missing)?)),
            "wood_volume" => Ok(FieldData::F32(self.wood_volume.clone().ok_or_else(missing)?)),
            "predominant_tree" => Ok(FieldData::I32(self.predominant_tree.clone().ok_or_else(missing)?)),
            "subvoxel_counts" => Ok(FieldData::U8(self.subvoxel_counts.clone().ok_or_else(missing)?)),
            "ground_height" => Ok(FieldData::F64(self.ground_height.clone().ok_or_else(missing)?)),
            _ => Err(missing()),
        }
    }

    /// A field (see [`RayVoxels::field_names`]) or a metric by name, widened
    /// to f64.
    pub fn values(&self, name: &str) -> Result<Vec<f64>> {
        if self.field_names().contains(&name) {
            Ok(self.field(name)?.to_f64())
        } else {
            self.metric(name)
        }
    }

    fn state_values(&self) -> Vec<u8> {
        (0..self.n_voxels()).map(|i| self.state(i) as u8).collect()
    }

    /// Height of every voxel: above the ground where the grid has a DTM with
    /// any finite height, else of its centre above the grid floor.
    fn heights(&self) -> Result<Vec<f64>> {
        let h = self.metric("distance_from_ground")?;
        if h.iter().any(|v| v.is_finite()) {
            return Ok(h);
        }
        let per = self.shape[0] * self.shape[1];
        Ok((0..self.n_voxels()).map(|i| ((i / per) as f64 + 0.5) * self.voxel_size).collect())
    }

    /// Canopy space: every voxel from `min_height` up to `max_height`, by
    /// default the highest filled voxel (0 if none). Returns the heights,
    /// the states, the membership and the top.
    fn canopy_space(&self, min_height: f64, max_height: Option<f64>) -> Result<CanopySpace> {
        let state = self.state_values();
        let h = self.heights()?;
        let top = match max_height {
            Some(t) => t,
            None => {
                let filled: Vec<f64> = h.iter().zip(&state).filter(|(_, &s)| s == FILLED).map(|(&v, _)| v).collect();
                if filled.is_empty() {
                    0.0
                } else {
                    let m = filled.iter().copied().filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, f64::max);
                    if m == f64::NEG_INFINITY {
                        f64::NAN
                    } else {
                        m
                    }
                }
            }
        };
        let space = h.iter().map(|&v| v.is_finite() && v >= min_height && v <= top).collect();
        Ok((h, state, space, top))
    }

    /// What the scan saw of the canopy space (see `canopy_space`), in layers
    /// of one voxel from `min_height`: per layer the voxel count, the shares
    /// observed (a pulse went through or ended in it), occluded (only pulses
    /// already stopped reached it) and unobserved, and the mean pulses
    /// entering a voxel; NaN for layers without voxels.
    pub fn occlusion_profile(&self, min_height: f64, max_height: Option<f64>) -> Result<OcclusionProfile> {
        let (h, state, space, top) = self.canopy_space(min_height, max_height)?;
        let vs = self.voxel_size;
        if !(top + vs - min_height).is_finite() {
            return Err(Error::invalid("arange: cannot compute length"));
        }
        let edges = arange(min_height, top + vs, vs);
        let n_layers = edges.len().saturating_sub(1).max(1);
        let kmax = edges.len().saturating_sub(2);
        let beams = self.values("num_beams")?;
        let mut n = vec![0i64; n_layers];
        let mut obs = vec![0i64; n_layers];
        let mut occ = vec![0i64; n_layers];
        let mut bm = vec![0.0f64; n_layers];
        for idx in 0..h.len() {
            if !space[idx] {
                continue;
            }
            let k = (searchsorted_right(&edges, h[idx]) as i64 - 1).clamp(0, kmax as i64) as usize;
            if k >= n_layers {
                continue;
            }
            n[k] += 1;
            if state[idx] >= EMPTY {
                obs[k] += 1;
            }
            if state[idx] == OCCLUDED {
                occ[k] += 1;
            }
            bm[k] += beams[idx];
        }
        let height: Vec<f64> = edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).take(n_layers).collect();
        let div = |a: f64, b: i64| a / b as f64;
        let tot = n.iter().sum::<i64>().max(1) as f64;
        let (sn, so, sc) = (n.iter().sum::<i64>(), obs.iter().sum::<i64>(), occ.iter().sum::<i64>());
        Ok(OcclusionProfile {
            height,
            observed: obs.iter().zip(&n).map(|(&o, &c)| div(o as f64, c)).collect(),
            occluded: occ.iter().zip(&n).map(|(&o, &c)| div(o as f64, c)).collect(),
            unobserved: (0..n_layers).map(|k| div((n[k] - obs[k] - occ[k]) as f64, n[k])).collect(),
            mean_beams: bm.iter().zip(&n).map(|(&b, &c)| div(b, c)).collect(),
            n_voxels: n,
            total_observed: so as f64 / tot,
            total_occluded: sc as f64 / tot,
            total_unobserved: (sn - so - sc) as f64 / tot,
            top,
        })
    }

    /// Share of each column's canopy space (as for
    /// [`RayVoxels::occlusion_profile`]) that was observed, `(ny, nx)`; NaN
    /// for columns without canopy space.
    pub fn observed_map(&self, min_height: f64, max_height: Option<f64>) -> Result<Vec<f64>> {
        let (_, state, space, _) = self.canopy_space(min_height, max_height)?;
        let per = self.shape[0] * self.shape[1];
        let mut seen = vec![0i64; per];
        let mut all = vec![0i64; per];
        for idx in 0..state.len() {
            if space[idx] {
                all[idx % per] += 1;
                if state[idx] >= EMPTY {
                    seen[idx % per] += 1;
                }
            }
        }
        Ok(seen.iter().zip(&all).map(|(&s, &a)| s as f64 / a as f64).collect())
    }

    /// Mean of a field or metric per layer over the voxels entered by at
    /// least `min_beams` pulses; NaN for layers where none is.
    pub fn profile(&self, name: &str, min_beams: f64) -> Result<Vec<f64>> {
        let beams = &self.i[I::NumBeams as usize];
        let ok: Vec<bool> = beams.iter().map(|&b| b as f64 >= min_beams).collect();
        let per = self.shape[0] * self.shape[1];
        let nz = self.shape[2];
        let count: Vec<i64> = (0..nz).map(|k| ok[k * per..(k + 1) * per].iter().filter(|&&b| b).count() as i64).collect();
        // NumPy keeps single-precision fields single while summing.
        let single = self.field_names().contains(&name) && matches!(self.field(name)?, FieldData::F32(_));
        let values: Vec<f64> = if single {
            let FieldData::F32(v) = self.field(name)? else { unreachable!() };
            check_len(v.len(), per * nz, name)?;
            let w: Vec<f32> = v.iter().zip(&ok).map(|(&x, &b)| if b { x } else { 0.0 }).collect();
            (0..nz).map(|k| pairwise_sum_f32(&w[k * per..(k + 1) * per]) as f64).collect()
        } else {
            let v = self.values(name)?;
            check_len(v.len(), per * nz, name)?;
            let w: Vec<f64> = v.iter().zip(&ok).map(|(&x, &b)| if b { x } else { 0.0 }).collect();
            (0..nz).map(|k| pairwise_sum(&w[k * per..(k + 1) * per])).collect()
        };
        Ok(values.iter().zip(&count).map(|(&v, &c)| if c > 0 { v / c as f64 } else { f64::NAN }).collect())
    }

    /// Bottom z of each voxel layer.
    pub fn z_levels(&self) -> Vec<f64> {
        (0..self.shape[2]).map(|k| self.origin[2] + k as f64 * self.voxel_size).collect()
    }

    /// Voxel-centre x, y and z, each row-major `(nz, ny, nx)`.
    pub fn centers(&self) -> [Vec<f64>; 3] {
        let n = self.n_voxels();
        let mut out = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
        for idx in 0..n {
            let ijk = self.unravel(idx);
            for a in 0..3 {
                out[a].push(self.origin[a] + (ijk[a] as f64 + 0.5) * self.voxel_size);
            }
        }
        out
    }

    /// How well each tree was seen (see [`tree_sampling`]), from this grid's
    /// states and pulse counts.
    pub fn tree_sampling(&self, points: &[Point], labels: &[i64], min_beams: f64, above: f64) -> Result<Vec<TreeSampling>> {
        if labels.len() != points.len() {
            return Err(Error::invalid("labels must match the points, state and beams the grid"));
        }
        let beams: Vec<f64> = self.i[I::NumBeams as usize].iter().map(|&b| b as f64).collect();
        Ok(tree_sampling(points, labels, self.origin, self.voxel_size, self.shape, &self.state_values(), &beams, min_beams, above))
    }
}

fn check_len(n: usize, want: usize, name: &str) -> Result<()> {
    if n != want {
        return Err(Error::invalid(format!("{name} is not a per-voxel array")));
    }
    Ok(())
}

/// Echo labels of in-memory pulses: `ground` and `foliage` arrays when
/// given, else derived from echo attributes by `labels` as for a shots file
/// ([`EchoLabels::annotate`]), with the intensities only for the weightings
/// that use them.
pub fn annotate(shots: &Shots, labels: &EchoLabels, ground: Option<Vec<bool>>, foliage: Option<Vec<u8>>, dtm: Option<&Raster>, weighting: WeightMethod) -> Result<EchoAnnotations> {
    let mut l = labels.clone();
    if ground.is_some() {
        l.ground_class = None;
        l.ground_distance = 0.0;
    }
    if foliage.is_some() {
        l.leaf_classes.clear();
        l.wood_classes.clear();
    }
    let mut a = l.annotate(shots, dtm)?;
    if ground.is_some() {
        a.ground = ground;
    }
    if foliage.is_some() {
        a.foliage = foliage;
    }
    if !matches!(weighting, WeightMethod::Relative | WeightMethod::Strongest) {
        a.intensity = None;
    }
    Ok(a)
}

/// Voxelise in-memory pulses with echo labels from [`annotate`].
pub fn voxelize_labelled(shots: &Shots, params: &VoxelParams, labels: &EchoLabels, ground: Option<Vec<bool>>, foliage: Option<Vec<u8>>, dtm: Option<&Raster>) -> Result<RayVoxels> {
    let a = annotate(shots, labels, ground, foliage, dtm, params.weighting)?;
    voxel::voxelize(&a.inputs(shots, dtm), params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column() -> RayVoxels {
        // Pulses up a 1 x 1 x 4 column; half end in the second voxel.
        let n = 40;
        let origin: Vec<Point> = (0..n).map(|i| [0.3 + 0.01 * (i % 10) as f64, 0.5, 0.0]).collect();
        let count: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let start: Vec<usize> = count.iter().scan(0usize, |s, &c| { let v = *s; *s += c as usize; Some(v) }).collect();
        let shots = Shots { origin, direction: vec![[0.0, 0.0, 1.0]; n], echo_start: start, echo_count: count, echo_range: vec![1.5; n / 2], echo_attrs: Default::default() };
        let params = VoxelParams { voxel_size: 1.0, bounds: Some(([0.0; 3], [1.0, 1.0, 4.0])), occlusion: true, attenuation: vec![voxel::Attenuation::Transmittance], unbounded_range: 4.0, ..Default::default() };
        voxelize_labelled(&shots, &params, &EchoLabels::default(), None, None, None).unwrap()
    }

    #[test]
    fn layers_count_what_was_seen() {
        let g = column();
        assert_eq!(g.z_levels(), vec![0.0, 1.0, 2.0, 3.0]);
        let p = g.occlusion_profile(0.0, Some(4.0)).unwrap();
        assert_eq!(p.n_voxels, vec![1, 1, 1, 1]);
        assert_eq!(p.observed, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(p.mean_beams[0], 40.0);
        assert!(g.occlusion_profile(0.0, Some(6.0)).unwrap().observed[5].is_nan());
        assert_eq!(p.top, 4.0);
        let top = g.occlusion_profile(0.0, None).unwrap();
        assert_eq!(top.top, 1.5);
        assert_eq!(g.observed_map(0.0, None).unwrap(), vec![1.0]);
        let prof = g.profile("num_hits", 1.0).unwrap();
        assert_eq!(prof, vec![0.0, 20.0, 0.0, 0.0]);
        assert!(g.profile("num_hits", 1000.0).unwrap()[0].is_nan());
        assert_eq!(g.centers()[2], vec![0.5, 1.5, 2.5, 3.5]);
    }

    #[test]
    fn arrays_override_attributes() {
        let g = column();
        assert!(g.field_names().contains(&"path_length"));
        assert!(g.values("pad_transmittance").is_ok());
        assert!(g.values("nope").is_err());
    }
}
