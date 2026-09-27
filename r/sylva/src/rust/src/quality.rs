// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for point-cloud quality on stems: stem noise, its summary and
//! scan ids from sensor origins. Tables cross as lists of columns.
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::quality::{self as ql, NoiseParams};
use sylva_rs::quality_summary as qs;

use crate::convert::{doubles, err, fail, xyz_from_r, Result};
use crate::trees::xy_from_r;

/// @noRd
#[extendr]
fn core_stem_noise(xyz: Robj, heights: &[f64], stems: Robj, scan_ids: Robj, height_min: f64, height_max: f64, step: f64, thickness: f64, min_radius: f64, max_radius: f64, min_arc: f64, min_inlier_fraction: f64, cut_min: f64, cut_fraction: f64, min_scan_points: f64, iterations: f64) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    let ids: Option<Vec<i64>> = if scan_ids.is_null() { None } else { Some(doubles(&scan_ids, "scan_id")?.iter().map(|&v| v as i64).collect()) };
    if heights.len() != p.len() || ids.as_ref().is_some_and(|s| s.len() != p.len()) {
        return fail("heights and scan_id need one value per point");
    }
    let st = xy_from_r(&stems)?;
    let params = NoiseParams { height_min, height_max, step, thickness, min_radius, max_radius, min_arc, min_inlier_fraction, cut_min, cut_fraction, min_scan_points: min_scan_points.max(0.0) as usize };
    let r = ql::stem_noise(&p, heights, ids.as_deref(), &st, &params, iterations.max(0.0) as usize);
    macro_rules! col {
        ($src:expr, $f:expr) => {
            $src.iter().map($f).collect::<Vec<f64>>()
        };
    }
    let slices = list!(
        stem = col!(r.slices, |x| x.stem as f64),
        height = col!(r.slices, |x| x.height),
        cx = col!(r.slices, |x| x.cx),
        cy = col!(r.slices, |x| x.cy),
        radius = col!(r.slices, |x| x.radius),
        n_points = col!(r.slices, |x| x.n_points as f64),
        sigma = col!(r.slices, |x| x.sigma),
        sigma_first = col!(r.slices, |x| x.sigma_first),
        arc = col!(r.slices, |x| x.arc),
        tail_fraction = col!(r.slices, |x| x.tail_fraction)
    );
    let scan_slices = list!(
        scan = col!(r.scan_slices, |x| x.scan as f64),
        slice = col!(r.scan_slices, |x| x.slice as f64),
        n_points = col!(r.scan_slices, |x| x.n_points as f64),
        median_residual = col!(r.scan_slices, |x| x.median_residual),
        sigma_within = col!(r.scan_slices, |x| x.sigma_within),
        sigma_local = col!(r.scan_slices, |x| x.sigma_local)
    );
    let scans = list!(
        scan = col!(r.scans, |x| x.scan as f64),
        n_points = col!(r.scans, |x| x.n_points as f64),
        n_slices = col!(r.scans, |x| x.n_slices as f64),
        tx = col!(r.scans, |x| x.tx),
        ty = col!(r.scans, |x| x.ty),
        sigma_within = col!(r.scans, |x| x.sigma_within),
        sigma_local = col!(r.scans, |x| x.sigma_local)
    );
    Ok(list!(slices = slices, scan_slices = scan_slices, scans = scans, residual = r.residual))
}

/// @noRd
#[extendr]
fn core_stem_noise_summary(slices: List, scan_slices: List, scans: List, min_scan_slices: f64) -> Result<List> {
    let get = |l: &List, k: &str| -> Result<Vec<f64>> {
        let m: HashMap<&str, Robj> = l.clone().try_into()?;
        doubles(m.get(k).ok_or_else(|| Error::Other(format!("table has no `{k}`")))?, k)
    };
    let ids = |v: Vec<f64>| v.iter().map(|&x| x as i64).collect::<Vec<i64>>();
    let height = get(&slices, "height")?;
    let n = height.len();
    let empty = |l: &List, k: &str| -> Result<Vec<f64>> { if n > 0 { get(l, k) } else { Ok(get(l, k).unwrap_or_default()) } };
    let stem = ids(empty(&slices, "stem")?);
    let scan = ids(get(&scans, "scan")?);
    let (sl_n, sl_sigma, sl_first, sl_tail) = (empty(&slices, "n_points")?, empty(&slices, "sigma")?, empty(&slices, "sigma_first")?, empty(&slices, "tail_fraction")?);
    let (ss_n, ss_within, ss_local) = (empty(&scan_slices, "n_points")?, empty(&scan_slices, "sigma_within")?, empty(&scan_slices, "sigma_local")?);
    let (sc_n, sc_slices, tx, ty) = (empty(&scans, "n_points")?, empty(&scans, "n_slices")?, empty(&scans, "tx")?, empty(&scans, "ty")?);
    let t = qs::NoiseTables {
        slice_count: n,
        slice_stem: &stem,
        slice_n_points: &sl_n,
        slice_sigma: &sl_sigma,
        slice_sigma_first: &sl_first,
        slice_tail_fraction: &sl_tail,
        scan_slice_n_points: &ss_n,
        scan_slice_sigma_within: &ss_within,
        scan_slice_sigma_local: &ss_local,
        scan: &scan,
        scan_n_points: &sc_n,
        scan_n_slices: &sc_slices,
        scan_tx: &tx,
        scan_ty: &ty,
    };
    let s = qs::noise_summary(&t, min_scan_slices).map_err(err)?;
    let mut names = vec!["n_stems", "n_slices", "n_scans"];
    let mut values: Vec<Robj> = vec![(s.n_stems as f64).into(), (s.n_slices as f64).into(), (s.n_scans as f64).into()];
    if let Some(m) = s.measured {
        names.extend(["sigma_total", "sigma_corrected", "sigma_within", "sigma_local", "tail_fraction", "n_scans_registered"]);
        values.extend([m.sigma_total.into(), m.sigma_corrected.into(), m.sigma_within.into(), m.sigma_local.into(), m.tail_fraction.into(), (m.n_scans_registered as f64).into()]);
        if let Some((rms, max, worst)) = m.registration {
            names.extend(["registration_rms", "registration_max", "worst_scan"]);
            values.extend([rms.into(), max.into(), (worst as f64).into()]);
        }
    }
    Ok(List::from_names_and_values(names, values).expect("names match values"))
}

/// @noRd
#[extendr]
fn core_scan_ids_from_origins(origins: Robj, tolerance: f64) -> Result<Vec<f64>> {
    let m: RMatrix<f64> = origins.try_into().map_err(|_| Error::Other("origins must be a numeric matrix".into()))?;
    let (nr, nc) = (m.nrows(), m.ncols());
    let d = m.data();
    let flat: Vec<f64> = (0..nr).flat_map(|r| (0..nc).map(move |c| d[c * nr + r])).collect();
    Ok(qs::scan_ids_from_origins(&flat, nc, tolerance).map_err(err)?.iter().map(|&v| v as f64).collect())
}

extendr_module! {
    mod quality;
    fn core_stem_noise;
    fn core_stem_noise_summary;
    fn core_scan_ids_from_origins;
}
