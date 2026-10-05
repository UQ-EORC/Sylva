// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for block tracing (sylva.voxels.blocks): the blocked trace and
//! the grid it leaves on disk.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::voxel::{self, BlockOptions, BlockStats, BlockedGrid, Pulses};

use crate::voxels_py::{field_array, occlusion_dict, params, sampling_dict, PyRayVoxels};
use crate::{err, raster_from_py, shots_from_py, xyz_from_py};

/// A ray-traced voxel grid stored block by block in a directory.
#[pyclass(name = "BlockedVoxels")]
pub(crate) struct PyBlockedVoxels {
    inner: BlockedGrid,
}

#[pymethods]
impl PyBlockedVoxels {
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.dir.clone()
    }

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

    /// Voxels per block `(bx, by, bz)`.
    #[getter]
    fn block_size(&self) -> (usize, usize, usize) {
        let b = self.inner.block;
        (b[0], b[1], b[2])
    }

    /// Blocks along x, y and z.
    #[getter]
    fn n_blocks(&self) -> (usize, usize, usize) {
        let b = self.inner.n_blocks();
        (b[0], b[1], b[2])
    }

    /// Blocks on disk, `(bx, by, bz)`.
    fn blocks_present(&self) -> Vec<(usize, usize, usize)> {
        self.inner.blocks_present().into_iter().map(|b| (b[0], b[1], b[2])).collect()
    }

    fn field_names(&self) -> Vec<&'static str> {
        self.inner.field_names()
    }

    fn metric_names(&self) -> Vec<&'static str> {
        voxel::RayVoxels::METRICS.to_vec()
    }

    fn field<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let g = &self.inner;
        let f = py.detach(|| g.field(name)).map_err(err)?;
        field_array(py, f, name, g.shape, g.params.subvoxel_split)
    }

    fn metric<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let g = &self.inner;
        let data = py.detach(|| g.metric(name)).map_err(err)?;
        let s = g.shape;
        let shp = [s[2], s[1], s[0]];
        if name == "state" {
            return Ok(PyArray1::from_vec(py, data.iter().map(|&v| v as u8).collect()).reshape(shp)?.into_any());
        }
        Ok(PyArray1::from_vec(py, data).reshape(shp)?.into_any())
    }

    /// Voxels `lo..hi` (`(i, j, k)` indices, `hi` exclusive) as a grid.
    fn read_box(&self, py: Python<'_>, lo: (usize, usize, usize), hi: (usize, usize, usize)) -> PyResult<PyRayVoxels> {
        let g = &self.inner;
        let inner = py.detach(|| g.read_box([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2])).map_err(err)?;
        Ok(PyRayVoxels { inner })
    }

    fn block(&self, py: Python<'_>, b: (usize, usize, usize)) -> PyResult<PyRayVoxels> {
        let g = &self.inner;
        let inner = py.detach(|| g.block([b.0, b.1, b.2])).map_err(err)?;
        Ok(PyRayVoxels { inner })
    }

    fn to_grid(&self, py: Python<'_>) -> PyResult<PyRayVoxels> {
        let g = &self.inner;
        let inner = py.detach(|| g.to_grid()).map_err(err)?;
        Ok(PyRayVoxels { inner })
    }

    #[pyo3(signature = (name="pad_fpl", min_beams=1.0))]
    fn profile<'py>(&self, py: Python<'py>, name: &str, min_beams: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let g = &self.inner;
        Ok(py.detach(|| g.profile(name, min_beams)).map_err(err)?.into_pyarray(py))
    }

    #[pyo3(signature = (min_height=0.0, max_height=None))]
    fn occlusion_profile<'py>(&self, py: Python<'py>, min_height: f64, max_height: Option<f64>) -> PyResult<Bound<'py, PyDict>> {
        let g = &self.inner;
        occlusion_dict(py, py.detach(|| g.occlusion_profile(min_height, max_height)).map_err(err)?)
    }

    #[pyo3(signature = (min_height=0.0, max_height=None))]
    fn observed_map<'py>(&self, py: Python<'py>, min_height: f64, max_height: Option<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let g = &self.inner;
        let m = py.detach(|| g.observed_map(min_height, max_height)).map_err(err)?;
        PyArray1::from_vec(py, m).reshape([g.shape[1], g.shape[0]])
    }

    #[pyo3(signature = (xyz, labels, min_beams=10.0, above=2.0))]
    fn tree_sampling<'py>(&self, py: Python<'py>, xyz: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<i64>, min_beams: f64, above: f64) -> PyResult<Bound<'py, PyDict>> {
        let p = xyz_from_py(xyz)?;
        let l = labels.as_array().to_vec();
        let g = &self.inner;
        let r = py.detach(|| g.tree_sampling(&p, &l, min_beams, above)).map_err(err)?;
        sampling_dict(py, r)
    }

    #[pyo3(signature = (path, format="vox", include_unobserved=false, filled_only=false))]
    fn write(&self, py: Python<'_>, path: PathBuf, format: &str, include_unobserved: bool, filled_only: bool) -> PyResult<usize> {
        let opts = voxel::WriteOptions { include_unobserved, filled_only };
        let text = match format {
            "vox" => false,
            "text" => true,
            other => return Err(PyValueError::new_err(format!("unknown voxel format {other:?} (vox|text)"))),
        };
        let g = &self.inner;
        py.detach(|| g.write(&path, text, opts)).map_err(err)
    }
}

fn stats_dict(py: Python<'_>, s: BlockStats) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_blocks", (s.n_blocks[0], s.n_blocks[1], s.n_blocks[2]))?;
    d.set_item("n_passes", s.n_passes)?;
    d.set_item("peak_voxels", s.peak_voxels)?;
    d.set_item("bytes_per_voxel", s.bytes_per_voxel)?;
    d.set_item("block_pulses", s.block_pulses)?;
    d.set_item("blocks_written", s.blocks_written)?;
    Ok(d)
}

/// Trace pulses block by block: in-memory `shots` (a dict, as for
/// `ray_voxelize`) or the shots file `path`. Returns `(grid, stats)`, the
/// grid assembled in memory, or on disk when `out` is given.
#[pyfunction]
#[pyo3(signature = (shots=None, path=None, block=(64, 64, 64), max_memory=None, workers=0, out=None, voxel_size=0.1, bounds=None, ground=None, foliage=None, dtm=None, class_attr="classification", ground_class=None, ground_distance=0.2, leaf_classes=vec![], wood_classes=vec![], tree_attr="tree_id", intensity_attr="intensity", weighting="equal", occlusion=false, flat_top=false, neighbour_prior_min_rays=0, beam=None, subvoxel_split=0, subvoxel_min_beams=10, average_leaf_area=0.005, lad="spherical", lad_params=vec![], attenuation=vec!["fpl".to_string()], inclination=false, n_iad_bins=18, knn_normal=10, triangle_lmax=0.05, unbounded_range=f64::INFINITY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ray_voxelize_blocks<'py>(py: Python<'py>, shots: Option<&Bound<'py, PyDict>>, path: Option<PathBuf>, block: (usize, usize, usize), max_memory: Option<f64>, workers: usize, out: Option<PathBuf>, voxel_size: f64, bounds: Option<((f64, f64, f64), (f64, f64, f64))>, ground: Option<PyReadonlyArray1<bool>>, foliage: Option<PyReadonlyArray1<u8>>, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, class_attr: &str, ground_class: Option<i64>, ground_distance: f64, leaf_classes: Vec<i64>, wood_classes: Vec<i64>, tree_attr: &str, intensity_attr: &str, weighting: &str, occlusion: bool, flat_top: bool, neighbour_prior_min_rays: u32, beam: Option<(f64, f64)>, subvoxel_split: usize, subvoxel_min_beams: u8, average_leaf_area: f64, lad: &str, lad_params: Vec<f64>, attenuation: Vec<String>, inclination: bool, n_iad_bins: usize, knn_normal: usize, triangle_lmax: f64, unbounded_range: f64) -> PyResult<(Py<PyAny>, Bound<'py, PyDict>)> {
    let params = params(voxel_size, bounds, weighting, occlusion, flat_top, neighbour_prior_min_rays, beam, subvoxel_split, subvoxel_min_beams, average_leaf_area, lad, lad_params, attenuation, inclination, n_iad_bins, knn_normal, triangle_lmax, unbounded_range)?;
    let labels = voxel::EchoLabels { class_attr: class_attr.into(), ground_class, ground_distance, leaf_classes, wood_classes, tree_attr: tree_attr.into(), intensity_attr: intensity_attr.into() };
    let dtm = dtm.map(|(data, xmin, ymin, res)| raster_from_py(data, xmin, ymin, res));
    if let Some(m) = max_memory {
        if m.is_nan() || m <= 0.0 {
            return Err(PyValueError::new_err("max_memory must be a positive number of bytes"));
        }
    }
    let opts = BlockOptions { block: [block.0, block.1, block.2], max_memory: max_memory.map(|m| m as u64), workers };
    let (grid, stats) = match (shots, path) {
        (Some(d), None) => {
            let s = shots_from_py(d)?;
            let ground = ground.map(|a| a.as_array().to_vec());
            let foliage = foliage.map(|a| a.as_array().to_vec());
            py.detach(|| {
                let notes = sylva_rs::voxel::grid::annotate(&s, &labels, ground, foliage, dtm.as_ref(), params.weighting)?;
                voxel::voxelize_blocks(&Pulses::Memory(notes.inputs(&s, dtm.as_ref())), &params, dtm.as_ref(), &opts, out.as_deref())
            })
        }
        (None, Some(p)) => py.detach(|| {
            let file = sylva_rs::io::shots::ShotsFile::open(&p)?;
            voxel::voxelize_blocks(&Pulses::File { file: &file, labels: &labels }, &params, dtm.as_ref(), &opts, out.as_deref())
        }),
        _ => return Err(PyValueError::new_err("give shots or a path, not both")),
    }
    .map_err(err)?;
    let stats = stats_dict(py, stats)?;
    let grid: Py<PyAny> = match (grid, out) {
        (Some(inner), _) => Py::new(py, PyRayVoxels { inner })?.into_any(),
        (None, Some(dir)) => Py::new(py, PyBlockedVoxels { inner: BlockedGrid::open(&dir).map_err(err)? })?.into_any(),
        (None, None) => return Err(PyValueError::new_err("the blocked trace returned no grid")),
    };
    Ok((grid, stats))
}

/// Open a grid written block by block.
#[pyfunction]
fn open_blocked_voxels(path: PathBuf) -> PyResult<PyBlockedVoxels> {
    Ok(PyBlockedVoxels { inner: BlockedGrid::open(&path).map_err(err)? })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlockedVoxels>()?;
    m.add_function(wrap_pyfunction!(ray_voxelize_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(open_blocked_voxels, m)?)?;
    Ok(())
}
