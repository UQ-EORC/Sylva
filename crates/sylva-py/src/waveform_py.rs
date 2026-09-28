// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::waveform: waveform files, decomposition and the
//! synthetic waveform generator. Waveforms and echoes cross as dicts of
//! arrays.
#![allow(clippy::type_complexity, clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]

use std::collections::BTreeMap;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use sylva_rs::waveform::decompose::{self, DecomposeOptions, PeakMethod};
use sylva_rs::waveform::las::{self as wlas, LasReadOptions, LasWriteWaveOptions, PacketStore};
use sylva_rs::waveform::pulsewaves::{self as pls, Kinds, PlsReadOptions, PlsWriteOptions};
use sylva_rs::waveform::simulate::{self, SimulateOptions};
use sylva_rs::waveform::{self, Echoes, FileFormat, Waveforms};

use crate::{attrs_from_py, attrs_to_py, err, shots_from_py, shots_to_py, xyz_from_py, xyz_to_py};

fn get<'py>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("waveforms dict missing {k:?}")))
}

fn f64s(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<f64>> {
    Ok(get(d, k)?.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec())
}

fn i64s(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<i64>> {
    Ok(get(d, k)?.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec())
}

pub(crate) fn waveforms_from_py(d: &Bound<'_, PyDict>) -> PyResult<Waveforms> {
    let start = i64s(d, "sample_start")?;
    let count = i64s(d, "sample_count")?;
    let samples = get(d, "samples")?.extract::<PyReadonlyArray1<f32>>()?.as_array().to_vec();
    if start.iter().chain(&count).any(|&v| v < 0) || count.iter().any(|&c| c > u32::MAX as i64) {
        return Err(PyValueError::new_err("waveforms: sample_start and sample_count must be non-negative"));
    }
    let attrs = match d.get_item("attrs")? {
        Some(a) => attrs_from_py(Some(a.cast::<PyDict>()?))?,
        None => BTreeMap::new(),
    };
    let w = Waveforms {
        pulse: i64s(d, "pulse")?,
        gps_time: f64s(d, "gps_time")?,
        origin: xyz_from_py(get(d, "origin")?.extract()?)?,
        anchor: xyz_from_py(get(d, "anchor")?.extract()?)?,
        direction: xyz_from_py(get(d, "direction")?.extract()?)?,
        offset: f64s(d, "offset")?,
        interval: f64s(d, "interval")?,
        metres_per_ns: f64s(d, "metres_per_ns")?,
        sample_start: start.iter().map(|&v| v as usize).collect(),
        sample_count: count.iter().map(|&v| v as u32).collect(),
        samples,
        attrs,
    };
    w.validate().map_err(err)?;
    Ok(w)
}

pub(crate) fn waveforms_to_py<'py>(py: Python<'py>, w: &Waveforms) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("pulse", w.pulse.clone().into_pyarray(py))?;
    d.set_item("gps_time", w.gps_time.clone().into_pyarray(py))?;
    d.set_item("origin", xyz_to_py(py, &w.origin))?;
    d.set_item("anchor", xyz_to_py(py, &w.anchor))?;
    d.set_item("direction", xyz_to_py(py, &w.direction))?;
    d.set_item("offset", w.offset.clone().into_pyarray(py))?;
    d.set_item("interval", w.interval.clone().into_pyarray(py))?;
    d.set_item("metres_per_ns", w.metres_per_ns.clone().into_pyarray(py))?;
    d.set_item("sample_start", w.sample_start.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("sample_count", w.sample_count.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("samples", w.samples.clone().into_pyarray(py))?;
    d.set_item("attrs", attrs_to_py(py, &w.attrs)?)?;
    Ok(d)
}

fn echoes_from_py(d: &Bound<'_, PyDict>) -> PyResult<Echoes> {
    let wfrow = i64s(d, "waveform")?;
    if wfrow.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("echoes: waveform indices must be non-negative"));
    }
    let e = Echoes {
        waveform: wfrow.iter().map(|&v| v as usize).collect(),
        time: f64s(d, "time")?,
        amplitude: f64s(d, "amplitude")?,
        width: f64s(d, "width")?,
        xyz: xyz_from_py(get(d, "xyz")?.extract()?)?,
        range: f64s(d, "range")?,
    };
    let n = e.waveform.len();
    if [e.time.len(), e.amplitude.len(), e.width.len(), e.xyz.len(), e.range.len()].iter().any(|&l| l != n) {
        return Err(PyValueError::new_err("echoes: every array must have one value per echo"));
    }
    if e.waveform.windows(2).any(|w| w[1] < w[0]) {
        return Err(PyValueError::new_err("echoes must be grouped by waveform in ascending order"));
    }
    Ok(e)
}

fn echoes_to_py<'py>(py: Python<'py>, e: &Echoes) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("waveform", e.waveform.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("time", e.time.clone().into_pyarray(py))?;
    d.set_item("amplitude", e.amplitude.clone().into_pyarray(py))?;
    d.set_item("width", e.width.clone().into_pyarray(py))?;
    d.set_item("xyz", xyz_to_py(py, &e.xyz))?;
    d.set_item("range", e.range.clone().into_pyarray(py))?;
    Ok(d)
}

fn kinds_from(s: &str) -> PyResult<Kinds> {
    match s {
        "returning" => Ok(Kinds::Returning),
        "outgoing" => Ok(Kinds::Outgoing),
        "all" => Ok(Kinds::All),
        _ => Err(PyValueError::new_err(format!("kind must be 'returning', 'outgoing' or 'all', got {s:?}"))),
    }
}

/// Describe a waveform file without reading its waveforms.
#[pyfunction]
fn waveform_info<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match waveform::detect_format(path).map_err(err)? {
        FileFormat::Las => {
            let i = wlas::read_info(path).map_err(err)?;
            d.set_item("format", "las")?;
            d.set_item("version", format!("{}.{}", i.version.0, i.version.1))?;
            d.set_item("point_format", i.point_format)?;
            d.set_item("n_records", i.n_points)?;
            d.set_item("compressed", i.compressed)?;
            d.set_item("bounds", i.bounds.to_vec())?;
            let (store, wdp) = match &i.store {
                PacketStore::Internal(_) => ("internal", None),
                PacketStore::External(p) => ("external", Some(p.to_string_lossy().to_string())),
                PacketStore::Missing => ("missing", None),
            };
            d.set_item("packets", store)?;
            d.set_item("wdp", wdp)?;
            let desc = PyDict::new(py);
            for (k, v) in &i.descriptors {
                let e = PyDict::new(py);
                e.set_item("bits_per_sample", v.bits_per_sample)?;
                e.set_item("n_samples", v.n_samples)?;
                e.set_item("interval", v.spacing_ps as f64 / 1000.0)?;
                e.set_item("gain", v.gain)?;
                e.set_item("offset", v.offset)?;
                e.set_item("compression", v.compression)?;
                desc.set_item(*k, e)?;
            }
            d.set_item("descriptors", desc)?;
        }
        FileFormat::PulseWaves => {
            let i = pls::read_info(path).map_err(err)?;
            d.set_item("format", "pulsewaves")?;
            d.set_item("version", format!("{}.{}", i.version.0, i.version.1))?;
            d.set_item("n_pulses", i.n_pulses)?;
            d.set_item("compressed", i.pulse_compression != 0)?;
            d.set_item("bounds", i.bounds.to_vec())?;
            d.set_item("system", i.system.clone())?;
            d.set_item("software", i.software.clone())?;
            d.set_item("waves", i.waves.as_ref().map(|p| p.to_string_lossy().to_string()))?;
            let desc = PyDict::new(py);
            for (k, v) in &i.descriptors {
                let e = PyDict::new(py);
                e.set_item("units", v.units)?;
                e.set_item("optical_center_to_anchor", v.optical_center_to_anchor)?;
                let s: Vec<(u8, u8, u16, f32, u16)> = v.samplings.iter().map(|s| (s.kind, s.channel, s.bits_per_sample, s.units, s.lookup_table)).collect();
                e.set_item("samplings", s)?;
                desc.set_item(*k, e)?;
            }
            d.set_item("descriptors", desc)?;
            d.set_item("lookup_tables", i.tables.keys().cloned().collect::<Vec<_>>())?;
        }
    }
    Ok(d)
}

/// Read waveforms `start .. start + count` (records for LAS, pulses for
/// PulseWaves); returns the waveforms and where the next chunk starts.
#[pyfunction]
#[pyo3(signature = (path, start=0, count=None, dedupe=true, kind="returning", lookup=true))]
fn waveform_read<'py>(py: Python<'py>, path: &str, start: u64, count: Option<u64>, dedupe: bool, kind: &str, lookup: bool) -> PyResult<(Bound<'py, PyDict>, u64)> {
    let kinds = kinds_from(kind)?;
    let (w, next) = py
        .detach(|| match waveform::detect_format(path)? {
            FileFormat::Las => wlas::read_waveforms(path, &LasReadOptions { start, count, dedupe }),
            FileFormat::PulseWaves => pls::read_waveforms(path, &PlsReadOptions { start, count, kinds, lookup }),
        })
        .map_err(err)?;
    Ok((waveforms_to_py(py, &w)?, next))
}

#[pyfunction]
#[pyo3(signature = (wf, path, scale=0.001, bits=16, external=false, crs_wkt=None))]
fn waveform_write_las(py: Python<'_>, wf: &Bound<'_, PyDict>, path: &str, scale: f64, bits: u8, external: bool, crs_wkt: Option<String>) -> PyResult<()> {
    let w = waveforms_from_py(wf)?;
    py.detach(|| wlas::write_waveforms(&w, path, &LasWriteWaveOptions { scale, bits, external, crs_wkt })).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (wf, path, scale=1e-4))]
fn waveform_write_pulsewaves(py: Python<'_>, wf: &Bound<'_, PyDict>, path: &str, scale: f64) -> PyResult<()> {
    let w = waveforms_from_py(wf)?;
    py.detach(|| pls::write_waveforms(&w, path, &PlsWriteOptions { scale })).map_err(err)
}

/// Position of every sample.
#[pyfunction]
fn waveform_sample_positions<'py>(py: Python<'py>, wf: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w = waveforms_from_py(wf)?;
    Ok(xyz_to_py(py, &py.detach(|| w.sample_positions())))
}

/// Background and noise standard deviation of each waveform.
#[pyfunction]
fn waveform_noise<'py>(py: Python<'py>, wf: &Bound<'_, PyDict>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let w = waveforms_from_py(wf)?;
    let r: Vec<(f64, f64)> = py.detach(|| (0..w.len()).into_par_iter().with_min_len(256).map(|i| decompose::estimate_noise(w.samples_of(i))).collect());
    Ok((r.iter().map(|x| x.0).collect::<Vec<_>>().into_pyarray(py), r.iter().map(|x| x.1).collect::<Vec<_>>().into_pyarray(py)))
}

/// Samples smoothed with a Gaussian of `sigma` ns.
#[pyfunction]
fn waveform_smooth<'py>(py: Python<'py>, wf: &Bound<'_, PyDict>, sigma: f64) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if !(sigma >= 0.0) || !sigma.is_finite() {
        return Err(PyValueError::new_err(format!("sigma must be zero or positive, got {sigma}")));
    }
    let w = waveforms_from_py(wf)?;
    let parts: Vec<Vec<f32>> = py.detach(|| {
        (0..w.len())
            .into_par_iter()
            .with_min_len(256)
            .map(|i| {
                let y: Vec<f64> = w.samples_of(i).iter().map(|&v| v as f64).collect();
                decompose::smooth(&y, sigma / w.interval[i]).into_iter().map(|v| v as f32).collect()
            })
            .collect()
    });
    Ok(parts.concat().into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (wf, smooth=1.0, threshold=4.0, min_amplitude=0.0, peaks="inflection", max_echoes=10, min_width=0.3, max_width=20.0, noise=None, background=None, max_iter=100, refine=true))]
fn waveform_decompose<'py>(py: Python<'py>, wf: &Bound<'_, PyDict>, smooth: f64, threshold: f64, min_amplitude: f64, peaks: &str, max_echoes: usize, min_width: f64, max_width: f64, noise: Option<f64>, background: Option<f64>, max_iter: usize, refine: bool) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let peaks = match peaks {
        "inflection" => PeakMethod::Inflection,
        "derivative" => PeakMethod::Derivative,
        _ => return Err(PyValueError::new_err(format!("peaks must be 'inflection' or 'derivative', got {peaks:?}"))),
    };
    let w = waveforms_from_py(wf)?;
    let opts = DecomposeOptions { smooth, threshold, min_amplitude, peaks, max_echoes, min_width, max_width, noise, background, max_iter, refine };
    let (e, st) = py.detach(|| decompose::decompose(&w, &opts)).map_err(err)?;
    let s = PyDict::new(py);
    s.set_item("background", st.background.into_pyarray(py))?;
    s.set_item("noise", st.noise.into_pyarray(py))?;
    s.set_item("rmse", st.rmse.into_pyarray(py))?;
    s.set_item("n_echoes", st.n_echoes.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    s.set_item("iterations", st.iterations.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    Ok((echoes_to_py(py, &e)?, s))
}

#[pyfunction]
#[pyo3(signature = (wf, echoes, origin=None))]
fn waveform_to_shots<'py>(py: Python<'py>, wf: &Bound<'_, PyDict>, echoes: &Bound<'_, PyDict>, origin: Option<PyReadonlyArray2<f64>>) -> PyResult<Bound<'py, PyDict>> {
    let w = waveforms_from_py(wf)?;
    let e = echoes_from_py(echoes)?;
    let o = origin.map(xyz_from_py).transpose()?;
    let s = waveform::to_shots(&w, &e, o.as_deref()).map_err(err)?;
    shots_to_py(py, &s)
}

#[pyfunction]
fn waveform_cross_section<'py>(py: Python<'py>, range: PyReadonlyArray1<f64>, amplitude: PyReadonlyArray1<f64>, width: PyReadonlyArray1<f64>, calibration: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (r, a, s) = (range.as_array().to_vec(), amplitude.as_array().to_vec(), width.as_array().to_vec());
    if a.len() != r.len() || s.len() != r.len() {
        return Err(PyValueError::new_err("range, amplitude and width must have the same length"));
    }
    Ok(waveform::backscatter_cross_section(&r, &a, &s, calibration).into_pyarray(py))
}

#[pyfunction]
fn waveform_calibration(range: PyReadonlyArray1<f64>, amplitude: PyReadonlyArray1<f64>, width: PyReadonlyArray1<f64>, reflectance: PyReadonlyArray1<f64>, beam_divergence: f64, incidence: PyReadonlyArray1<f64>) -> PyResult<f64> {
    waveform::calibration_constant(&range.as_array().to_vec(), &amplitude.as_array().to_vec(), &width.as_array().to_vec(), &reflectance.as_array().to_vec(), beam_divergence, &incidence.as_array().to_vec()).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (shots, gps_time=None, pulse_width=1.5, interval=1.0, n_samples=120, margin=3.0, start_range=None, background=10.0, noise=1.0, digitise=true, bits=16, amplitude=100.0, metres_per_ns=waveform::C_HALF, seed=0))]
fn synthetic_waveforms<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, gps_time: Option<PyReadonlyArray1<f64>>, pulse_width: f64, interval: f64, n_samples: usize, margin: f64, start_range: Option<f64>, background: f64, noise: f64, digitise: bool, bits: u8, amplitude: f64, metres_per_ns: f64, seed: u64) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let s = shots_from_py(shots)?;
    let t = gps_time.map(|t| t.as_array().to_vec());
    let opts = SimulateOptions { pulse_width, interval, n_samples, margin, start_range, background, noise, digitise, bits, amplitude, metres_per_ns, seed };
    let (w, e) = py.detach(|| simulate::waveforms_from_shots(&s, t.as_deref(), &opts)).map_err(err)?;
    Ok((waveforms_to_py(py, &w)?, echoes_to_py(py, &e)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(waveform_info, m)?,
        wrap_pyfunction!(waveform_read, m)?,
        wrap_pyfunction!(waveform_write_las, m)?,
        wrap_pyfunction!(waveform_write_pulsewaves, m)?,
        wrap_pyfunction!(waveform_sample_positions, m)?,
        wrap_pyfunction!(waveform_noise, m)?,
        wrap_pyfunction!(waveform_smooth, m)?,
        wrap_pyfunction!(waveform_decompose, m)?,
        wrap_pyfunction!(waveform_to_shots, m)?,
        wrap_pyfunction!(waveform_cross_section, m)?,
        wrap_pyfunction!(waveform_calibration, m)?,
        wrap_pyfunction!(synthetic_waveforms, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
