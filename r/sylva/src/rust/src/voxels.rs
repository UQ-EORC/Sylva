// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for ray-traced voxel grids. The grid stays on the Rust side as
//! a `RayVoxelGrid` external pointer whose methods copy arrays out as flat
//! row-major vectors (`(nz, ny, nx)`); `R/voxels.R` gives them dimensions.

use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::leaf_model::qsm_from_rows;
use sylva_rs::qsm::Qsm;
use sylva_rs::voxel;
use sylva_rs::voxel_grid::{self, FieldData};

use crate::convert::{doubles, err, fail, optional_f64, raster_from_r, shots_from_r, xyz_from_r, Result};

/// QSM cylinders from an `n x 12` matrix (or `NULL` for none).
pub fn qsm_from_r(cylinders: &Robj) -> Result<Qsm> {
    if cylinders.is_null() {
        return Ok(Qsm::default());
    }
    let m: RMatrix<f64> = cylinders.try_into().map_err(|_| Error::Other("cylinders must be a double matrix with 12 columns".into()))?;
    if m.ncols() != 12 {
        return fail("cylinder array must have 12 columns");
    }
    let n = m.nrows();
    let d = m.data();
    let rows: Vec<[f64; 12]> = (0..n).map(|i| std::array::from_fn(|c| d[c * n + i])).collect();
    Ok(qsm_from_rows(&rows))
}

/// Ray-traced voxel statistics, held on the Rust side.
#[extendr]
pub struct RayVoxelGrid {
    pub inner: voxel::RayVoxels,
}

fn field_to_r(f: FieldData) -> Robj {
    match f {
        FieldData::I32(v) => v.into(),
        FieldData::U8(v) => v.iter().map(|&x| x as i32).collect::<Vec<i32>>().into(),
        FieldData::F32(v) => v.iter().map(|&x| x as f64).collect::<Vec<f64>>().into(),
        FieldData::F64(v) => v.into(),
    }
}

#[extendr]
impl RayVoxelGrid {
    /// Minimum corner.
    fn origin(&self) -> Vec<f64> {
        self.inner.origin.to_vec()
    }

    fn voxel_size(&self) -> f64 {
        self.inner.voxel_size
    }

    /// `c(nx, ny, nz)`.
    fn shape(&self) -> Vec<i32> {
        self.inner.shape.iter().map(|&v| v as i32).collect()
    }

    fn has_leaf(&self) -> bool {
        self.inner.has_leaf
    }

    fn has_wood(&self) -> bool {
        self.inner.has_wood
    }

    fn field_names(&self) -> Vec<String> {
        self.inner.field_names().into_iter().map(String::from).collect()
    }

    fn metric_names(&self) -> Vec<String> {
        voxel::RayVoxels::METRICS.iter().map(|s| s.to_string()).collect()
    }

    /// A raw accumulator, row-major; integers for counts and ids.
    fn field(&self, name: &str) -> Result<Robj> {
        Ok(field_to_r(self.inner.field(name).map_err(err)?))
    }

    /// A derived quantity, row-major; `state` as integers.
    fn metric(&self, name: &str) -> Result<Robj> {
        let v = self.inner.metric(name).map_err(err)?;
        if name == "state" {
            return Ok(v.iter().map(|&s| s as i32).collect::<Vec<i32>>().into());
        }
        Ok(v.into())
    }

    /// Per-tree inclination distributions, named by tree id.
    fn tree_iad(&self) -> List {
        let mut names = Vec::new();
        let mut values = Vec::new();
        let s = |o: Option<&'static str>| o.map_or_else(|| Robj::from(()), Robj::from);
        for (tid, t) in &self.inner.tree_iad {
            names.push(tid.to_string());
            values.push(Robj::from(list!(
                bin_centres = t.bin_centres.clone(),
                liad = t.liad.clone(),
                wiad = t.wiad.clone(),
                piad = t.piad.clone(),
                liad_bailey = t.liad_bailey.clone(),
                wiad_bailey = t.wiad_bailey.clone(),
                piad_bailey = t.piad_bailey.clone(),
                g_leaf = t.g_leaf,
                g_wood = t.g_wood,
                g_plant = t.g_plant,
                bailey_g_leaf = t.bailey_g_leaf,
                bailey_g_wood = t.bailey_g_wood,
                leaf_hits = t.leaf_hits as f64,
                wood_hits = t.wood_hits as f64,
                liad_de_wit = s(t.liad_de_wit),
                wiad_de_wit = s(t.wiad_de_wit),
                piad_de_wit = s(t.piad_de_wit)
            )));
        }
        List::from_names_and_values(names, values).expect("names match values")
    }

    /// Rasterise QSM cylinders (an `n x 12` matrix) into woody volume.
    fn add_wood_volume(&mut self, cylinders: Robj) -> Result<()> {
        let q = qsm_from_r(&cylinders)?;
        self.inner.add_wood_volume(&q.cylinders);
        Ok(())
    }

    /// Write `.vox` (AMAPVox) or text; the number of voxels written.
    fn write(&self, path: &str, format: &str, include_unobserved: bool, filled_only: bool) -> Result<f64> {
        let opts = voxel::WriteOptions { include_unobserved, filled_only };
        let n = match format {
            "vox" => self.inner.write_vox(path, opts),
            "text" => self.inner.write_text(path, opts),
            other => return fail(format!("unknown voxel format {other:?} (vox|text)")),
        };
        Ok(n.map_err(err)? as f64)
    }

    fn write_iad_csv(&self, path: &str) -> Result<()> {
        self.inner.write_iad_csv(path).map_err(err)
    }

    fn occlusion_profile(&self, min_height: f64, max_height: Robj) -> Result<List> {
        let p = self.inner.occlusion_profile(min_height, optional_f64(&max_height, "max_height")?).map_err(err)?;
        Ok(list!(
            height = p.height,
            n_voxels = p.n_voxels.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            observed = p.observed,
            occluded = p.occluded,
            unobserved = p.unobserved,
            mean_beams = p.mean_beams,
            total = list!(observed = p.total_observed, occluded = p.total_occluded, unobserved = p.total_unobserved, top = p.top)
        ))
    }

    /// Row-major `(ny, nx)`.
    fn observed_map(&self, min_height: f64, max_height: Robj) -> Result<Vec<f64>> {
        self.inner.observed_map(min_height, optional_f64(&max_height, "max_height")?).map_err(err)
    }

    fn profile(&self, name: &str, min_beams: f64) -> Result<Vec<f64>> {
        self.inner.profile(name, min_beams).map_err(err)
    }

    fn z_levels(&self) -> Vec<f64> {
        self.inner.z_levels()
    }

    fn centers(&self) -> List {
        let [x, y, z] = self.inner.centers();
        list!(x = x, y = y, z = z)
    }

    fn tree_sampling(&self, xyz: Robj, labels: &[f64], min_beams: f64, above: f64) -> Result<List> {
        let p = xyz_from_r(&xyz)?;
        let l: Vec<i64> = labels.iter().map(|&v| v as i64).collect();
        let r = self.inner.tree_sampling(&p, &l, min_beams, above).map_err(err)?;
        let col = |f: &dyn Fn(&sylva_rs::voxel::quality::TreeSampling) -> f64| r.iter().map(f).collect::<Vec<f64>>();
        let q: Vec<f64> = (0..4).flat_map(|c| r.iter().map(move |x| x.beams_by_quarter[c])).collect();
        Ok(list!(
            tree_id = col(&|x| x.tree_id as f64),
            n_voxels = col(&|x| x.n_voxels as f64),
            volume = col(&|x| x.volume),
            observed_fraction = col(&|x| x.observed_fraction),
            occluded_fraction = col(&|x| x.occluded_fraction),
            unobserved_fraction = col(&|x| x.unobserved_fraction),
            median_beams = col(&|x| x.median_beams),
            p10_beams = col(&|x| x.p10_beams),
            well_sampled_fraction = col(&|x| x.well_sampled_fraction),
            above_observed_fraction = col(&|x| x.above_observed_fraction),
            beams_by_quarter = RMatrix::new_matrix(r.len(), 4, |i, c| q[c * r.len() + i])
        ))
    }
}

fn opt<'a>(m: &'a HashMap<&str, Robj>, k: &str) -> Result<&'a Robj> {
    m.get(k).ok_or_else(|| Error::Other(format!("missing option `{k}`")))
}

fn f(m: &HashMap<&str, Robj>, k: &str) -> Result<f64> {
    optional_f64(opt(m, k)?, k)?.ok_or_else(|| Error::Other(format!("option `{k}` is NULL")))
}

fn s(m: &HashMap<&str, Robj>, k: &str) -> Result<String> {
    opt(m, k)?.as_str().map(String::from).ok_or_else(|| Error::Other(format!("option `{k}` must be a string")))
}

fn b(m: &HashMap<&str, Robj>, k: &str) -> Result<bool> {
    opt(m, k)?.as_bool().ok_or_else(|| Error::Other(format!("option `{k}` must be TRUE or FALSE")))
}

fn ints(m: &HashMap<&str, Robj>, k: &str) -> Result<Vec<i64>> {
    let v = opt(m, k)?;
    if v.is_null() {
        return Ok(Vec::new());
    }
    Ok(doubles(v, k)?.iter().map(|&x| x as i64).collect())
}

/// Voxel parameters and echo labels from the options list of `R/voxels.R`.
fn params_from_r(opts: &List) -> Result<(voxel::VoxelParams, voxel::EchoLabels)> {
    let m: HashMap<&str, Robj> = opts.clone().try_into()?;
    let bounds = {
        let v = opt(&m, "bounds")?;
        if v.is_null() {
            None
        } else {
            let d = doubles(v, "bounds")?;
            if d.len() != 6 {
                return fail("bounds must hold six values: the min and max corners");
            }
            Some(([d[0], d[1], d[2]], [d[3], d[4], d[5]]))
        }
    };
    let beam = {
        let v = opt(&m, "beam")?;
        if v.is_null() {
            None
        } else {
            let d = doubles(v, "beam")?;
            if d.len() != 2 {
                return fail("beam must be c(diameter, divergence)");
            }
            Some(voxel::BeamSpec { diameter: d[0], divergence: d[1] })
        }
    };
    let lad_params = doubles(opt(&m, "lad_params")?, "lad_params").unwrap_or_default();
    let attenuation: Vec<String> = opt(&m, "attenuation")?.as_str_vector().ok_or_else(|| Error::Other("attenuation must be character".into()))?.iter().map(|s| s.to_string()).collect();
    let params = voxel::VoxelParams {
        voxel_size: f(&m, "voxel_size")?,
        bounds,
        weighting: voxel::WeightMethod::parse(&s(&m, "weighting")?).map_err(err)?,
        occlusion: b(&m, "occlusion")?,
        flat_top: b(&m, "flat_top")?,
        neighbour_prior_min_rays: f(&m, "neighbour_prior_min_rays")?.max(0.0) as u32,
        beam,
        subvoxel_split: f(&m, "subvoxel_split")?.max(0.0) as usize,
        subvoxel_min_beams: f(&m, "subvoxel_min_beams")?.clamp(0.0, 255.0) as u8,
        average_leaf_area: f(&m, "average_leaf_area")?,
        lad: voxel::Lad::parse(&s(&m, "lad")?, &lad_params).map_err(err)?,
        attenuation: attenuation.iter().map(|a| voxel::Attenuation::parse(a)).collect::<std::result::Result<_, _>>().map_err(err)?,
        inclination: b(&m, "inclination")?,
        n_iad_bins: f(&m, "n_iad_bins")?.max(0.0) as usize,
        knn_normal: f(&m, "knn_normal")?.max(0.0) as usize,
        triangle_lmax: f(&m, "triangle_lmax")?,
        unbounded_range: f(&m, "unbounded_range")?,
    };
    let ground_class = optional_f64(opt(&m, "ground_class")?, "ground_class")?.map(|v| v as i64);
    let labels = voxel::EchoLabels {
        class_attr: s(&m, "class_attr")?,
        ground_class,
        ground_distance: f(&m, "ground_distance")?,
        leaf_classes: ints(&m, "leaf_classes")?,
        wood_classes: ints(&m, "wood_classes")?,
        tree_attr: s(&m, "tree_attr")?,
        intensity_attr: s(&m, "intensity_attr")?,
    };
    Ok((params, labels))
}

fn dtm_from_r(dtm: &Robj) -> Result<Option<sylva_rs::Raster>> {
    if dtm.is_null() {
        return Ok(None);
    }
    let l: HashMap<&str, Robj> = List::try_from(dtm)?.try_into()?;
    let g = |k: &str| l.get(k).ok_or_else(|| Error::Other(format!("dtm has no `{k}`")));
    Ok(Some(raster_from_r(g("data")?, doubles(g("xmin")?, "xmin")?[0], doubles(g("ymin")?, "ymin")?[0], doubles(g("resolution")?, "resolution")?[0])?))
}

/// @noRd
#[extendr]
fn core_ray_voxelize(shots: List, ground: Robj, foliage: Robj, dtm: Robj, opts: List) -> Result<RayVoxelGrid> {
    let s = shots_from_r(&shots)?;
    let (params, labels) = params_from_r(&opts)?;
    let n = s.echo_range.len();
    let ground = if ground.is_null() {
        None
    } else {
        let g = ground.as_logical_slice().ok_or_else(|| Error::Other("ground must be logical".into()))?;
        Some(g.iter().map(|v| v.is_true()).collect::<Vec<bool>>())
    };
    let foliage = if foliage.is_null() { None } else { Some(doubles(&foliage, "foliage")?.iter().map(|&v| v as u8).collect::<Vec<u8>>()) };
    for (name, len) in [("ground", ground.as_ref().map(|g| g.len())), ("foliage", foliage.as_ref().map(|f| f.len()))] {
        if let Some(l) = len {
            if l != n {
                return fail(format!("{name} has {l} values for {n} echoes"));
            }
        }
    }
    let dtm = dtm_from_r(&dtm)?;
    let inner = voxel_grid::voxelize_labelled(&s, &params, &labels, ground, foliage, dtm.as_ref()).map_err(err)?;
    Ok(RayVoxelGrid { inner })
}

/// @noRd
#[extendr]
fn core_ray_voxelize_file(path: &str, dtm: Robj, opts: List) -> Result<RayVoxelGrid> {
    let (params, labels) = params_from_r(&opts)?;
    let dtm = dtm_from_r(&dtm)?;
    let file = sylva_rs::io::shots::ShotsFile::open(path).map_err(err)?;
    let inner = voxel::voxelize_file(&file, &params, &labels, dtm.as_ref()).map_err(err)?;
    Ok(RayVoxelGrid { inner })
}

/// @noRd
#[extendr]
fn core_laser_spec(name: &str) -> Robj {
    voxel::laser_spec(name).map_or_else(|| Robj::from(()), |(d, v)| Robj::from(vec![d, v]))
}

/// @noRd
#[extendr]
fn core_leaf_projection(theta: &[f64], lad: &str, lad_params: &[f64]) -> Result<Vec<f64>> {
    let lad = voxel::Lad::parse(lad, lad_params).map_err(err)?;
    Ok(theta.iter().map(|&t| voxel::compute_g(t, &lad)).collect())
}

extendr_module! {
    mod voxels;
    impl RayVoxelGrid;
    fn core_ray_voxelize;
    fn core_ray_voxelize_file;
    fn core_laser_spec;
    fn core_leaf_projection;
}
