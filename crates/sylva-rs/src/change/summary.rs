// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Plot-level growth, mortality and recruitment between two epochs.
//!
//! Basal area is `pi / 4 * dbh²` per tree and stem volume a form factor
//! times basal area times height; both are summed per hectare and divided
//! by the years between the epochs. Every quantity is also computed for
//! `n_draws` Monte Carlo draws in which each tree's DBH, height and
//! increments are perturbed by their standard errors (the registration
//! uncertainty is already part of the increments' errors), and survivors
//! whose increment could not be measured take the increment of a randomly
//! drawn measured survivor. The interval is the central `confidence` range
//! of the draws. It covers measurement noise only: the counts of deaths and
//! recruits are taken as exact, and the sampling error of a plot as a
//! sample of a stand is not included.

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::util::nprandom::Generator;
use super::positive;

/// A survivor: its first-epoch size and its increments, with standard
/// errors. A NaN increment (not measured) is imputed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Survivor {
    pub dbh: f64,
    pub dbh_se: f64,
    pub height: f64,
    pub height_se: f64,
    pub d_dbh: f64,
    pub d_dbh_se: f64,
    pub d_height: f64,
    pub d_height_se: f64,
}

/// A tree seen in one epoch only (dead or recruit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Single {
    pub dbh: f64,
    pub dbh_se: f64,
    pub height: f64,
    pub height_se: f64,
}

/// Settings of [`plot_summary`].
#[derive(Debug, Clone)]
pub struct SummaryParams {
    /// Plot area (m²).
    pub area: f64,
    /// Years between the epochs.
    pub years: f64,
    pub n_draws: usize,
    pub confidence: f64,
    /// Stem volume = `form_factor * basal area * height`.
    pub form_factor: f64,
    /// Oven-dry wood density (t/m³) for biomass; NaN leaves biomass out.
    pub wood_density: f64,
    pub seed: u64,
}

impl Default for SummaryParams {
    fn default() -> Self {
        SummaryParams { area: 1.0, years: 1.0, n_draws: 2000, confidence: 0.95, form_factor: 0.5, wood_density: f64::NAN, seed: 0 }
    }
}

/// One plot-level quantity: its estimate and interval.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub name: &'static str,
    pub unit: &'static str,
    pub estimate: f64,
    pub low: f64,
    pub high: f64,
}

fn ba(d: f64) -> f64 {
    PI / 4.0 * d * d
}

struct Draw {
    growth_d: f64,
    ba: [f64; 3],
    vol: [f64; 3],
}

/// Sums of one realisation; `noise` draws a perturbation for a value and
/// its standard error, `pick` a measured survivor for imputation.
fn realise(s: &[Survivor], dead: &[Single], rec: &[Single], ff: f64, noise: &mut dyn FnMut(f64, f64) -> f64, pick: &mut dyn FnMut() -> Option<(f64, f64)>) -> Draw {
    let (mut gd, mut n) = (0.0, 0usize);
    let (mut ba_g, mut v_g) = (0.0, 0.0);
    for t in s {
        let d1 = noise(t.dbh, t.dbh_se);
        let h1 = noise(t.height, t.height_se);
        let (dd, dh) = if t.d_dbh.is_finite() {
            (noise(t.d_dbh, t.d_dbh_se), if t.d_height.is_finite() { noise(t.d_height, t.d_height_se) } else { pick().map(|p| p.1).unwrap_or(0.0) })
        } else {
            pick().unwrap_or((0.0, 0.0))
        };
        gd += dd;
        n += 1;
        let (d2, h2) = (d1 + dd, h1 + dh);
        ba_g += ba(d2) - ba(d1);
        v_g += ff * (ba(d2) * h2 - ba(d1) * h1);
    }
    let sum = |set: &[Single], noise: &mut dyn FnMut(f64, f64) -> f64| -> (f64, f64) {
        set.iter().fold((0.0, 0.0), |acc, t| {
            let (d, h) = (noise(t.dbh, t.dbh_se), noise(t.height, t.height_se));
            (acc.0 + ba(d), acc.1 + ff * ba(d) * h)
        })
    };
    let (ba_m, v_m) = sum(dead, noise);
    let (ba_r, v_r) = sum(rec, noise);
    Draw { growth_d: if n > 0 { gd / n as f64 } else { f64::NAN }, ba: [ba_g, ba_m, ba_r], vol: [v_g, v_m, v_r] }
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f64::NAN;
    }
    f.sort_by(f64::total_cmp);
    let pos = q * (f.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    f[lo] + (f[hi] - f[lo]) * (pos - lo as f64)
}

/// Growth, mortality and recruitment of a plot; see the module
/// documentation. `n_ambiguous` trees (merged or split stems) are left out
/// of every sum and only reported.
pub fn plot_summary(survivors: &[Survivor], dead: &[Single], recruits: &[Single], n_ambiguous: usize, p: &SummaryParams) -> Result<Vec<Quantity>> {
    if !positive(p.area) || !positive(p.years) || !positive(p.form_factor) {
        return Err(Error::invalid("area, years and form_factor must be positive"));
    }
    if !(p.confidence > 0.0 && p.confidence < 1.0) {
        return Err(Error::invalid(format!("confidence must be in (0, 1), got {}", p.confidence)));
    }
    if survivors.iter().any(|t| !t.dbh.is_finite()) || dead.iter().chain(recruits).any(|t| !t.dbh.is_finite()) {
        return Err(Error::invalid("every tree needs a finite DBH"));
    }
    let per_ha = 1e4 / p.area / p.years;
    let measured: Vec<(f64, f64)> = survivors.iter().filter(|t| t.d_dbh.is_finite()).map(|t| (t.d_dbh, if t.d_height.is_finite() { t.d_height } else { 0.0 })).collect();
    let mean_measured = if measured.is_empty() { (0.0, 0.0) } else { (measured.iter().map(|m| m.0).sum::<f64>() / measured.len() as f64, measured.iter().map(|m| m.1).sum::<f64>() / measured.len() as f64) };
    // Point estimate: no noise, unmeasured survivors at the measured mean.
    let point = realise(survivors, dead, recruits, p.form_factor, &mut |v, _| v, &mut || Some(mean_measured));
    let mut noise_rng = Generator::new(p.seed);
    let mut pick_rng = Generator::new(p.seed.wrapping_add(1));
    let mut draws: Vec<Draw> = Vec::with_capacity(p.n_draws);
    for _ in 0..p.n_draws {
        draws.push(realise(
            survivors,
            dead,
            recruits,
            p.form_factor,
            &mut |v, se| if se.is_finite() && se > 0.0 { v + se * noise_rng.standard_normal() } else { v },
            &mut || if measured.is_empty() { None } else { Some(measured[pick_rng.bounded(measured.len() as u64 - 1) as usize]) },
        ));
    }
    let (qlo, qhi) = ((1.0 - p.confidence) / 2.0, (1.0 + p.confidence) / 2.0);
    let mut out = Vec::new();
    let mut push = |name: &'static str, unit: &'static str, est: f64, f: &dyn Fn(&Draw) -> f64| {
        let mut v: Vec<f64> = draws.iter().map(f).collect();
        let (low, high) = if draws.is_empty() { (f64::NAN, f64::NAN) } else { (percentile(&mut v, qlo), percentile(&mut v, qhi)) };
        out.push(Quantity { name, unit, estimate: est, low, high });
    };
    let (ns, nd, nr) = (survivors.len() as f64, dead.len() as f64, recruits.len() as f64);
    let exact = |v: f64| move |_: &Draw| v;
    push("n_survivors", "trees", ns, &exact(ns));
    push("n_deaths", "trees", nd, &exact(nd));
    push("n_recruits", "trees", nr, &exact(nr));
    push("n_ambiguous", "trees", n_ambiguous as f64, &exact(n_ambiguous as f64));
    let n0 = ns + nd;
    let n1 = ns + nr;
    let mort = if n0 > 0.0 { 1.0 - (ns / n0).powf(1.0 / p.years) } else { f64::NAN };
    let recr = if n1 > 0.0 { 1.0 - (ns / n1).powf(1.0 / p.years) } else { f64::NAN };
    push("mortality_rate", "1/yr", mort, &exact(mort));
    push("recruitment_rate", "1/yr", recr, &exact(recr));
    let py = p.years;
    push("dbh_increment", "m/yr", point.growth_d / py, &|d| d.growth_d / py);
    let names_ba = ["basal_area_growth", "basal_area_mortality", "basal_area_recruitment"];
    let names_v = ["volume_growth", "volume_mortality", "volume_recruitment"];
    for (k, name) in names_ba.into_iter().enumerate() {
        push(name, "m²/ha/yr", point.ba[k] * per_ha, &|d| d.ba[k] * per_ha);
    }
    push("basal_area_net", "m²/ha/yr", (point.ba[0] - point.ba[1] + point.ba[2]) * per_ha, &|d| (d.ba[0] - d.ba[1] + d.ba[2]) * per_ha);
    for (k, name) in names_v.into_iter().enumerate() {
        push(name, "m³/ha/yr", point.vol[k] * per_ha, &|d| d.vol[k] * per_ha);
    }
    push("volume_net", "m³/ha/yr", (point.vol[0] - point.vol[1] + point.vol[2]) * per_ha, &|d| (d.vol[0] - d.vol[1] + d.vol[2]) * per_ha);
    if p.wood_density.is_finite() {
        let w = p.wood_density;
        let names_b = ["biomass_growth", "biomass_mortality", "biomass_recruitment"];
        for (k, name) in names_b.into_iter().enumerate() {
            push(name, "Mg/ha/yr", point.vol[k] * per_ha * w, &|d| d.vol[k] * per_ha * w);
        }
        push("biomass_net", "Mg/ha/yr", (point.vol[0] - point.vol[1] + point.vol[2]) * per_ha * w, &|d| (d.vol[0] - d.vol[1] + d.vol[2]) * per_ha * w);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surv(dbh: f64, d: f64) -> Survivor {
        Survivor { dbh, dbh_se: 0.002, height: 15.0, height_se: 0.1, d_dbh: d, d_dbh_se: 0.001, d_height: 0.5, d_height_se: 0.1 }
    }

    #[test]
    fn sums_and_intervals() {
        let s = [surv(0.3, 0.01), surv(0.2, 0.02)];
        let dead = [Single { dbh: 0.4, dbh_se: 0.002, height: 20.0, height_se: 0.1 }];
        let p = SummaryParams { area: 1000.0, years: 5.0, ..Default::default() };
        let q = plot_summary(&s, &dead, &[], 0, &p).unwrap();
        let get = |n: &str| q.iter().find(|x| x.name == n).unwrap().clone();
        let g = get("basal_area_growth");
        let expect = ((ba(0.31) - ba(0.3)) + (ba(0.22) - ba(0.2))) * 10.0 / 5.0;
        assert!((g.estimate - expect).abs() < 1e-12);
        assert!(g.low < g.estimate && g.estimate < g.high);
        assert!((get("basal_area_mortality").estimate - ba(0.4) * 2.0).abs() < 1e-12);
        assert!((get("mortality_rate").estimate - (1.0 - (2.0f64 / 3.0).powf(0.2))).abs() < 1e-12);
        assert!((get("dbh_increment").estimate - 0.003).abs() < 1e-12);
        assert!(q.iter().all(|x| !x.name.starts_with("biomass")));
    }

    #[test]
    fn unmeasured_survivors_take_the_measured_increments() {
        let s = [surv(0.3, 0.01), surv(0.3, f64::NAN)];
        let q = plot_summary(&s, &[], &[], 0, &SummaryParams { area: 1e4, ..Default::default() }).unwrap();
        let g = q.iter().find(|x| x.name == "dbh_increment").unwrap();
        assert!((g.estimate - 0.01).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_settings() {
        assert!(plot_summary(&[], &[], &[], 0, &SummaryParams { area: 0.0, ..Default::default() }).is_err());
        assert!(plot_summary(&[surv(f64::NAN, 0.0)], &[], &[], 0, &SummaryParams::default()).is_err());
    }
}
