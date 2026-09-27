// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva.voxels: the ray-traced grid (held on the Rust side),
//! its layer summaries and files, scanner beams and leaf projection.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::voxel;
use sylva_rs::voxel_grid::{self, FieldData};

use crate::{err, qsm_from_rows, raster_from_py, shots_from_py, xyz_from_py};

/// Ray-traced voxel statistics. The grid stays on the Rust side; arrays are
/// copied out on request, shaped `(nz, ny, nx)`.
#[pyclass(name = "RayVoxels")]
pub(crate) struct PyRayVoxels {
    pub(crate) inner: voxel::RayVoxels,
}

impl PyRayVoxels {
    fn grid_shape(&self) -> [usize; 3] {
        let s = self.inner.shape;
        [s[2], s[1], s[0]]
    }
}

#[pymethods]
impl PyRayVoxels {
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.origin.to_vec().into_pyarray(py)
    }

    #[getter]
    fn voxel_size(&self) -> f64 {
        self.inner.voxel_size
    }

    /// `(nx, ny, nz)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.inner.shape;
        (s[0], s[1], s[2])
    }

    #[getter]
    fn has_leaf(&self) -> bool {
        self.inner.has_leaf
    }

    #[getter]
    fn has_wood(&self) -> bool {
        self.inner.has_wood
    }

    /// Names accepted by `field`.
    fn field_names(&self) -> Vec<&'static str> {
        self.inner.field_names()
    }

    /// A raw accumulator as an array, in its stored type.
    fn field<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let shp = self.grid_shape();
        let f = self.inner.field(name).map_err(err)?;
        let dims: Vec<usize> = match name {
            "ground_height" => vec![shp[1], shp[2]],
            "subvoxel_counts" => vec![shp[0], shp[1], shp[2], self.inner.params.subvoxel_split.pow(3)],
            _ => shp.to_vec(),
        };
        Ok(match f {
            FieldData::I32(v) => PyArray1::from_vec(py, v).reshape(dims)?.into_any(),
            FieldData::F32(v) => PyArray1::from_vec(py, v).reshape(dims)?.into_any(),
            FieldData::F64(v) => PyArray1::from_vec(py, v).reshape(dims)?.into_any(),
            FieldData::U8(v) => PyArray1::from_vec(py, v).reshape(dims)?.into_any(),
        })
    }

    /// A derived per-voxel quantity (`pad_fpl`, `attenuation_ppl`, `transmittance`, ...).
    fn metric<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let v = &self.inner;
        let data = py.detach(|| v.metric(name)).map_err(err)?;
        let shp = self.grid_shape();
        if name == "state" {
            return Ok(PyArray1::from_vec(py, data.iter().map(|&s| s as u8).collect()).reshape(shp)?.into_any());
        }
        Ok(PyArray1::from_vec(py, data).reshape(shp)?.into_any())
    }

    fn metric_names(&self) -> Vec<&'static str> {
        voxel::RayVoxels::METRICS.to_vec()
    }

    /// Per-tree inclination distributions: `{tree_id: {...}}`.
    fn tree_iad<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (tid, t) in &self.inner.tree_iad {
            let d = PyDict::new(py);
            d.set_item("bin_centres", t.bin_centres.clone().into_pyarray(py))?;
            for (k, h) in [("liad", &t.liad), ("wiad", &t.wiad), ("piad", &t.piad), ("liad_bailey", &t.liad_bailey), ("wiad_bailey", &t.wiad_bailey), ("piad_bailey", &t.piad_bailey)] {
                d.set_item(k, h.clone().into_pyarray(py))?;
            }
            for (k, g) in [("g_leaf", t.g_leaf), ("g_wood", t.g_wood), ("g_plant", t.g_plant), ("bailey_g_leaf", t.bailey_g_leaf), ("bailey_g_wood", t.bailey_g_wood)] {
                d.set_item(k, g)?;
            }
            d.set_item("leaf_hits", t.leaf_hits)?;
            d.set_item("wood_hits", t.wood_hits)?;
            d.set_item("liad_de_wit", t.liad_de_wit)?;
            d.set_item("wiad_de_wit", t.wiad_de_wit)?;
            d.set_item("piad_de_wit", t.piad_de_wit)?;
            out.set_item(*tid, d)?;
        }
        Ok(out)
    }

    /// Rasterise QSM cylinders (12-column rows) into per-voxel woody volume.
    fn add_wood_volume(&mut self, py: Python<'_>, cylinders: PyReadonlyArray2<f64>) -> PyResult<()> {
        let q = qsm_from_rows(cylinders)?;
        let v = &mut self.inner;
        py.detach(|| v.add_wood_volume(&q.cylinders));
        Ok(())
    }

    /// Write `.vox` (AMAPVox) or `.txt`; returns the number of voxels written.
    #[pyo3(signature = (path, format="vox", include_unobserved=false, filled_only=false))]
    fn write(&self, py: Python<'_>, path: PathBuf, format: &str, include_unobserved: bool, filled_only: bool) -> PyResult<usize> {
        let opts = voxel::WriteOptions { include_unobserved, filled_only };
        let v = &self.inner;
        match format {
            "vox" => py.detach(|| v.write_vox(&path, opts)).map_err(err),
            "text" => py.detach(|| v.write_text(&path, opts)).map_err(err),
            other => Err(PyValueError::new_err(format!("unknown voxel format {other:?} (vox|text)"))),
        }
    }

    fn write_iad_csv(&self, path: PathBuf) -> PyResult<()> {
        self.inner.write_iad_csv(path).map_err(err)
    }

    /// Layer shares of the canopy space observed, occluded and unobserved.
    #[pyo3(signature = (min_height=0.0, max_height=None))]
    fn occlusion_profile<'py>(&self, py: Python<'py>, min_height: f64, max_height: Option<f64>) -> PyResult<Bound<'py, PyDict>> {
        let p = self.inner.occlusion_profile(min_height, max_height).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("height", p.height.into_pyarray(py))?;
        d.set_item("n_voxels", p.n_voxels.into_pyarray(py))?;
        d.set_item("observed", p.observed.into_pyarray(py))?;
        d.set_item("occluded", p.occluded.into_pyarray(py))?;
        d.set_item("unobserved", p.unobserved.into_pyarray(py))?;
        d.set_item("mean_beams", p.mean_beams.into_pyarray(py))?;
        let t = PyDict::new(py);
        t.set_item("observed", p.total_observed)?;
        t.set_item("occluded", p.total_occluded)?;
        t.set_item("unobserved", p.total_unobserved)?;
        t.set_item("top", p.top)?;
        d.set_item("total", t)?;
        Ok(d)
    }

    /// `(ny, nx)` share of each column's canopy space observed.
    #[pyo3(signature = (min_height=0.0, max_height=None))]
    fn observed_map<'py>(&self, py: Python<'py>, min_height: f64, max_height: Option<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.inner.observed_map(min_height, max_height).map_err(err)?;
        let s = self.inner.shape;
        PyArray1::from_vec(py, m).reshape([s[1], s[0]])
    }

    /// Mean of a field or metric per layer over voxels with `min_beams` pulses.
    #[pyo3(signature = (name="pad_fpl", min_beams=1.0))]
    fn profile<'py>(&self, py: Python<'py>, name: &str, min_beams: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.inner.profile(name, min_beams).map_err(err)?.into_pyarray(py))
    }

    /// Bottom z of each layer.
    fn z_levels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.z_levels().into_pyarray(py)
    }

    /// Voxel-centre `(X, Y, Z)`, each `(nz, ny, nx)`.
    #[allow(clippy::type_complexity)]
    fn centers<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let shp = self.grid_shape();
        let [x, y, z] = self.inner.centers();
        let arr = |v: Vec<f64>| -> PyResult<Bound<'py, PyAny>> { Ok(PyArray1::from_vec(py, v).reshape(shp)?.into_any()) };
        Ok((arr(x)?, arr(y)?, arr(z)?))
    }

    /// How well each labelled tree was seen; columns as arrays.
    #[pyo3(signature = (xyz, labels, min_beams=10.0, above=2.0))]
    fn tree_sampling<'py>(&self, py: Python<'py>, xyz: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<i64>, min_beams: f64, above: f64) -> PyResult<Bound<'py, PyDict>> {
        let p = xyz_from_py(xyz)?;
        let l = labels.as_array().to_vec();
        let v = &self.inner;
        let r = py.detach(|| v.tree_sampling(&p, &l, min_beams, above)).map_err(err)?;
        let d = PyDict::new(py);
        macro_rules! col {
            ($name:literal, $f:expr) => {
                d.set_item($name, r.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
            };
        }
        col!("tree_id", |x| x.tree_id);
        col!("n_voxels", |x| x.n_voxels as i64);
        col!("volume", |x| x.volume);
        col!("observed_fraction", |x| x.observed_fraction);
        col!("occluded_fraction", |x| x.occluded_fraction);
        col!("unobserved_fraction", |x| x.unobserved_fraction);
        col!("median_beams", |x| x.median_beams);
        col!("p10_beams", |x| x.p10_beams);
        col!("well_sampled_fraction", |x| x.well_sampled_fraction);
        col!("above_observed_fraction", |x| x.above_observed_fraction);
        let q: Vec<f64> = r.iter().flat_map(|x| x.beams_by_quarter).collect();
        d.set_item("beams_by_quarter", PyArray1::from_vec(py, q).reshape([r.len(), 4])?)?;
        Ok(d)
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn params(voxel_size: f64, bounds: Option<((f64, f64, f64), (f64, f64, f64))>, weighting: &str, occlusion: bool, flat_top: bool, neighbour_prior_min_rays: u32, beam: Option<(f64, f64)>, subvoxel_split: usize, subvoxel_min_beams: u8, average_leaf_area: f64, lad: &str, lad_params: Vec<f64>, attenuation: Vec<String>, inclination: bool, n_iad_bins: usize, knn_normal: usize, triangle_lmax: f64, unbounded_range: f64) -> PyResult<voxel::VoxelParams> {
    Ok(voxel::VoxelParams {
        voxel_size,
        bounds: bounds.map(|(lo, hi)| ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2])),
        weighting: voxel::WeightMethod::parse(weighting).map_err(err)?,
        occlusion,
        flat_top,
        neighbour_prior_min_rays,
        beam: beam.map(|(diameter, divergence)| voxel::BeamSpec { diameter, divergence }),
        subvoxel_split,
        subvoxel_min_beams,
        average_leaf_area,
        lad: voxel::Lad::parse(lad, &lad_params).map_err(err)?,
        attenuation: attenuation.iter().map(|m| voxel::Attenuation::parse(m)).collect::<Result<_, _>>().map_err(err)?,
        inclination,
        n_iad_bins,
        knn_normal,
        triangle_lmax,
        unbounded_range,
    })
}

/// Voxelise a shots file one batch of row groups at a time.
#[pyfunction]
#[pyo3(signature = (path, voxel_size=0.1, bounds=None, dtm=None, class_attr="classification", ground_class=None, ground_distance=0.2, leaf_classes=vec![], wood_classes=vec![], tree_attr="tree_id", intensity_attr="intensity", weighting="equal", occlusion=false, flat_top=false, neighbour_prior_min_rays=0, beam=None, subvoxel_split=0, subvoxel_min_beams=10, average_leaf_area=0.005, lad="spherical", lad_params=vec![], attenuation=vec!["fpl".to_string()], inclination=false, n_iad_bins=18, knn_normal=10, triangle_lmax=0.05, unbounded_range=f64::INFINITY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ray_voxelize_file(py: Python<'_>, path: PathBuf, voxel_size: f64, bounds: Option<((f64, f64, f64), (f64, f64, f64))>, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, class_attr: &str, ground_class: Option<i64>, ground_distance: f64, leaf_classes: Vec<i64>, wood_classes: Vec<i64>, tree_attr: &str, intensity_attr: &str, weighting: &str, occlusion: bool, flat_top: bool, neighbour_prior_min_rays: u32, beam: Option<(f64, f64)>, subvoxel_split: usize, subvoxel_min_beams: u8, average_leaf_area: f64, lad: &str, lad_params: Vec<f64>, attenuation: Vec<String>, inclination: bool, n_iad_bins: usize, knn_normal: usize, triangle_lmax: f64, unbounded_range: f64) -> PyResult<PyRayVoxels> {
    let params = params(voxel_size, bounds, weighting, occlusion, flat_top, neighbour_prior_min_rays, beam, subvoxel_split, subvoxel_min_beams, average_leaf_area, lad, lad_params, attenuation, inclination, n_iad_bins, knn_normal, triangle_lmax, unbounded_range)?;
    let labels = voxel::EchoLabels { class_attr: class_attr.into(), ground_class, ground_distance, leaf_classes, wood_classes, tree_attr: tree_attr.into(), intensity_attr: intensity_attr.into() };
    let dtm = dtm.map(|(data, xmin, ymin, res)| raster_from_py(data, xmin, ymin, res));
    let inner = py.detach(|| voxel::voxelize_file(&sylva_rs::io::shots::ShotsFile::open(&path)?, &params, &labels, dtm.as_ref())).map_err(err)?;
    Ok(PyRayVoxels { inner })
}

/// Voxelise in-memory pulses; echo labels come from the `ground` and
/// `foliage` arrays or, without them, from echo attributes as for a file.
#[pyfunction]
#[pyo3(signature = (shots, voxel_size=0.1, bounds=None, ground=None, foliage=None, dtm=None, class_attr="classification", ground_class=None, ground_distance=0.2, leaf_classes=vec![], wood_classes=vec![], tree_attr="tree_id", intensity_attr="intensity", weighting="equal", occlusion=false, flat_top=false, neighbour_prior_min_rays=0, beam=None, subvoxel_split=0, subvoxel_min_beams=10, average_leaf_area=0.005, lad="spherical", lad_params=vec![], attenuation=vec!["fpl".to_string()], inclination=false, n_iad_bins=18, knn_normal=10, triangle_lmax=0.05, unbounded_range=f64::INFINITY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ray_voxelize(py: Python<'_>, shots: &Bound<'_, PyDict>, voxel_size: f64, bounds: Option<((f64, f64, f64), (f64, f64, f64))>, ground: Option<PyReadonlyArray1<bool>>, foliage: Option<PyReadonlyArray1<u8>>, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, class_attr: &str, ground_class: Option<i64>, ground_distance: f64, leaf_classes: Vec<i64>, wood_classes: Vec<i64>, tree_attr: &str, intensity_attr: &str, weighting: &str, occlusion: bool, flat_top: bool, neighbour_prior_min_rays: u32, beam: Option<(f64, f64)>, subvoxel_split: usize, subvoxel_min_beams: u8, average_leaf_area: f64, lad: &str, lad_params: Vec<f64>, attenuation: Vec<String>, inclination: bool, n_iad_bins: usize, knn_normal: usize, triangle_lmax: f64, unbounded_range: f64) -> PyResult<PyRayVoxels> {
    let s = shots_from_py(shots)?;
    let params = params(voxel_size, bounds, weighting, occlusion, flat_top, neighbour_prior_min_rays, beam, subvoxel_split, subvoxel_min_beams, average_leaf_area, lad, lad_params, attenuation, inclination, n_iad_bins, knn_normal, triangle_lmax, unbounded_range)?;
    let labels = voxel::EchoLabels { class_attr: class_attr.into(), ground_class, ground_distance, leaf_classes, wood_classes, tree_attr: tree_attr.into(), intensity_attr: intensity_attr.into() };
    let ground = ground.map(|a| a.as_array().to_vec());
    let foliage = foliage.map(|a| a.as_array().to_vec());
    let dtm = dtm.map(|(data, xmin, ymin, res)| raster_from_py(data, xmin, ymin, res));
    let inner = py.detach(|| voxel_grid::voxelize_labelled(&s, &params, &labels, ground, foliage, dtm.as_ref())).map_err(err)?;
    Ok(PyRayVoxels { inner })
}

/// `(beam diameter at exit [m], divergence [rad])` of a scanner known to AMAPVox.
#[pyfunction]
fn laser_spec(name: &str) -> Option<(f64, f64)> {
    voxel::laser_spec(name)
}

/// Projection function G of an analytic leaf angle distribution at beam zenith `theta` (rad).
#[pyfunction]
#[pyo3(signature = (theta, lad="spherical", lad_params=vec![]))]
fn leaf_projection<'py>(py: Python<'py>, theta: PyReadonlyArray1<f64>, lad: &str, lad_params: Vec<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lad = voxel::Lad::parse(lad, &lad_params).map_err(err)?;
    Ok(theta.as_array().iter().map(|&t| voxel::compute_g(t, &lad)).collect::<Vec<_>>().into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRayVoxels>()?;
    for f in [wrap_pyfunction!(ray_voxelize, m)?, wrap_pyfunction!(ray_voxelize_file, m)?, wrap_pyfunction!(laser_spec, m)?, wrap_pyfunction!(leaf_projection, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}
