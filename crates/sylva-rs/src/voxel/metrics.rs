//! Per-voxel estimators: attenuation, projection functions, area densities.

use std::f64::consts::{FRAC_PI_2, PI};

use super::{Attenuation, RayVoxels, F, I};
use crate::error::{Error, Result};

const EPS: f64 = 1e-10;

/// Analytic leaf angle distribution (de Wit types and AMAPVox's parametric ones).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lad {
    Spherical,
    Uniform,
    Planophile,
    Erectophile,
    Plagiophile,
    Extremophile,
    /// Campbell's ellipsoidal distribution with axis ratio `chi`.
    Ellipsoidal(f64),
    /// Two-parameter beta distribution (`mu`, `nu`).
    TwoParamBeta(f64, f64),
}

impl Lad {
    /// From a name and up to two parameters, as rayvoxel's `--lad` / `--lad_params`.
    pub fn parse(name: &str, p: &[f64]) -> Result<Self> {
        let need = |n: usize| if p.len() < n { Err(Error::invalid(format!("LAD {name:?} needs {n} parameter(s)"))) } else { Ok(()) };
        Ok(match name {
            "spherical" => Lad::Spherical,
            "uniform" => Lad::Uniform,
            "planophile" => Lad::Planophile,
            "erectophile" => Lad::Erectophile,
            "plagiophile" => Lad::Plagiophile,
            "extremophile" => Lad::Extremophile,
            "ellipsoidal" => {
                need(1)?;
                Lad::Ellipsoidal(p[0])
            }
            "twoParamBeta" => {
                need(2)?;
                if p[0] <= 0.0 || p[1] <= 0.0 {
                    return Err(Error::invalid("beta LAD parameters must be positive"));
                }
                Lad::TwoParamBeta(p[0], p[1])
            }
            other => return Err(Error::invalid(format!("unknown leaf angle distribution {other:?}"))),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Lad::Spherical => "spherical",
            Lad::Uniform => "uniform",
            Lad::Planophile => "planophile",
            Lad::Erectophile => "erectophile",
            Lad::Plagiophile => "plagiophile",
            Lad::Extremophile => "extremophile",
            Lad::Ellipsoidal(_) => "ellipsoidal",
            Lad::TwoParamBeta(..) => "twoParamBeta",
        }
    }

    /// Probability density of the leaf inclination `t` on `[0, π/2]` (the
    /// zenith angle of the leaf normal, 0 for a horizontal leaf).
    pub fn pdf(&self, t: f64) -> f64 {
        match *self {
            Lad::Spherical => t.sin(),
            Lad::Uniform => 2.0 / PI,
            Lad::Planophile => 2.0 / PI * (1.0 + (2.0 * t).cos()),
            Lad::Erectophile => 2.0 / PI * (1.0 - (2.0 * t).cos()),
            Lad::Plagiophile => 2.0 / PI * (1.0 - (4.0 * t).cos()),
            Lad::Extremophile => 2.0 / PI * (1.0 + (4.0 * t).cos()),
            Lad::Ellipsoidal(chi) => {
                if chi == 1.0 {
                    return t.sin();
                }
                let lambda = if chi < 1.0 {
                    let e = (1.0 - chi * chi).sqrt();
                    chi + e.asin() / e
                } else {
                    let e = (1.0 - 1.0 / (chi * chi)).sqrt();
                    chi + ((1.0 + e) / (1.0 - e)).ln() / (2.0 * e * chi)
                };
                2.0 * chi.powi(3) * t.sin() / (lambda * (t.cos().powi(2) + chi * chi * t.sin().powi(2)).powi(2))
            }
            Lad::TwoParamBeta(mu, nu) => {
                let x = 2.0 * t / PI;
                if !(0.0..=1.0).contains(&x) {
                    return 0.0;
                }
                let ln_b = ln_gamma(mu) + ln_gamma(nu) - ln_gamma(mu + nu);
                x.powf(mu - 1.0) * (1.0 - x).powf(nu - 1.0) / ln_b.exp() * (2.0 / PI)
            }
        }
    }
}

/// Lanczos approximation of `ln Γ(x)` for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let s = G.iter().enumerate().skip(1).fold(G[0], |s, (i, g)| s + g / (x + i as f64));
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

/// Mean projection of a unit leaf at inclination `leaf` onto the plane normal
/// to a beam at zenith `beam`, for uniform leaf azimuth.
fn projection_kernel(beam: f64, leaf: f64) -> f64 {
    let cotcot = 1.0 / (beam.tan() * leaf.tan());
    if cotcot.abs() > 1.0 || cotcot.is_infinite() {
        return beam.cos() * leaf.cos();
    }
    let a = cotcot.acos();
    beam.cos() * leaf.cos() * (1.0 + 2.0 / PI * (a.tan() - a))
}

fn fold_beam_angle(theta: f64) -> f64 {
    let mut t = theta % PI;
    if t > FRAC_PI_2 {
        t = PI - t;
    }
    t.min(FRAC_PI_2 - 1e-9)
}

/// Projection function `G(θ)` of a leaf angle distribution for a beam at
/// zenith `theta` (rad). 180-step trapezoid, as AMAPVox.
pub fn compute_g(theta: f64, lad: &Lad) -> f64 {
    if *lad == Lad::Spherical {
        return 0.5;
    }
    let theta = fold_beam_angle(theta);
    let n = 180;
    let h = FRAC_PI_2 / n as f64;
    let f = |t: f64| projection_kernel(theta, t) * lad.pdf(t);
    let mut sum = 0.5 * (f(0.0) + f(FRAC_PI_2));
    for i in 1..n {
        sum += f(i as f64 * h);
    }
    h * sum
}

/// `G(θ)` against an empirical (normalised) inclination histogram.
pub fn compute_g_from_histogram(theta: f64, bin_centres: &[f64], hist: &[f64]) -> f64 {
    if bin_centres.is_empty() || bin_centres.len() != hist.len() {
        return 0.5;
    }
    let theta = fold_beam_angle(theta);
    bin_centres.iter().zip(hist).map(|(&c, &h)| projection_kernel(theta, c) * h).sum()
}

/// Closest de Wit distribution to an inclination histogram (L2 on the
/// normalised bins); `None` for an empty histogram.
pub fn classify_de_wit(bin_centres: &[f64], hist: &[f64]) -> Option<&'static str> {
    let total: f64 = hist.iter().sum();
    if bin_centres.is_empty() || hist.len() != bin_centres.len() || total <= 0.0 {
        return None;
    }
    let mut best = (f64::INFINITY, None);
    for lad in [Lad::Planophile, Lad::Erectophile, Lad::Plagiophile, Lad::Extremophile, Lad::Spherical, Lad::Uniform] {
        let reference: Vec<f64> = bin_centres.iter().map(|&c| lad.pdf(c)).collect();
        let ref_sum: f64 = reference.iter().sum();
        if ref_sum <= 0.0 {
            continue;
        }
        let d2: f64 = hist.iter().zip(&reference).map(|(h, r)| (h / total - r / ref_sum).powi(2)).sum();
        if d2 < best.0 {
            best = (d2, Some(lad.name()));
        }
    }
    best.1
}

/// `(beam diameter at exit [m], divergence [rad])` of a scanner known to AMAPVox.
pub fn laser_spec(name: &str) -> Option<(f64, f64)> {
    Some(match name.to_ascii_uppercase().replace('_', "-").as_str() {
        "LMS-Q560" => (0.0003, 0.0005),
        "LMS-Q780" => (0.005, 0.00025),
        "VZ-400" | "VZ-400I" => (0.007, 0.00035),
        "LEICA-SCANSTATION-P30-40" => (0.0035, 0.00023),
        "LEICA-SCANSTATION-C10" => (0.004, 0.0001),
        "FARO-FOCUS-X330" => (0.0025, 0.00019),
        "MINIVUX-1UAV" => (0.0145, 0.00105),
        "TRIMBLE-X7" => (0.0026, 0.0008),
        "UNITARY-BEAM-SECTION" => (0.0, 0.0),
        _ => return None,
    })
}

/// Bailey & Mahaffee (2017) eq. 10: area density from the mean gap
/// probability `1 − hits / beams` over the mean path, by the secant method.
pub fn solve_bailey_pad(path_length: f64, beams_weighted: f64, hits: f64, g: f64) -> f64 {
    if g <= 0.0 || beams_weighted < 1.0 {
        return 0.0;
    }
    let r_bar = path_length / beams_weighted;
    if r_bar <= 0.0 {
        return 0.0;
    }
    let p_bar = (1.0 - hits / beams_weighted).clamp(1e-12, 1.0 - 1e-12);
    let residual = |a: f64| p_bar - (-a * g * r_bar).exp();
    let mut a0 = -p_bar.ln() / (g * r_bar);
    let mut a1 = a0 * 1.0001 + 1e-6;
    let (mut f0, mut f1) = (residual(a0), residual(a1));
    for _ in 0..50 {
        if f1.abs() < 1e-9 || (f1 - f0).abs() < 1e-18 {
            break;
        }
        let a2 = a1 - f1 * (a1 - a0) / (f1 - f0);
        (a0, f0) = (a1, f1);
        a1 = a2;
        f1 = residual(a1);
    }
    a1.max(0.0)
}

/// Area densities of one voxel for one attenuation method.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AreaDensity {
    pub pad: f64,
    pub lad: f64,
    pub wad: f64,
}

impl RayVoxels {
    /// Attenuation coefficient λ (m⁻¹). `Bailey` has no λ of its own and
    /// falls back to `hits / path_length`.
    ///
    /// * FPL: intercepted beam section over the effective free path
    ///   (hits over free path without beam metrics) — the biased MLE.
    /// * Transmittance: `−ln(1 − intercepted / entering)`.
    /// * PPL: the exact solve where available, otherwise the mean-chord form
    ///   (Pimont et al. 2018).
    pub fn attenuation(&self, idx: usize, method: Attenuation) -> f64 {
        let f = |fld: F| self.get_f(fld, idx) as f64;
        // Share-weighted: the free path and beam counts below carry each
        // segment's share of the pulse, so the hits must too.
        let hits_w = self.get_f(F::HitsWeighted, idx) as f64;
        let transmittance = || {
            if f(F::BsEntering) > EPS {
                return Some(-((f(F::BsEntering) - f(F::BsIntercepted)) / f(F::BsEntering)).max(EPS).ln());
            }
            None
        };
        match method {
            Attenuation::Fpl => {
                if f(F::BsEffectiveFreePath) > EPS {
                    f(F::BsIntercepted) / f(F::BsEffectiveFreePath)
                } else if f(F::FreePathLength) > EPS {
                    hits_w / f(F::FreePathLength)
                } else {
                    0.0
                }
            }
            Attenuation::Transmittance => transmittance().unwrap_or_else(|| {
                if f(F::NumBeamsWeighted) > EPS {
                    -(1.0 - (hits_w / f(F::NumBeamsWeighted)).min(1.0 - EPS)).ln()
                } else {
                    0.0
                }
            }),
            Attenuation::Ppl => {
                let hits = self.get_i(I::NumHits, idx) as f64;
                if let Some(k) = self.ppl_lambda.as_ref().map(|v| v[idx]).filter(|&k| k >= 0.0) {
                    return k as f64;
                }
                let n = hits;
                let m = (self.get_i(I::NumBeams, idx) as f64 - hits).max(0.0);
                let d_n = if n > EPS { f(F::SumHitDelta) / n } else { 0.0 };
                let d_m = if m > EPS { f(F::SumMissDelta) / m } else { 0.0 };
                if n < EPS || d_n < EPS {
                    return transmittance().unwrap_or(0.0);
                }
                if m < EPS || d_m < EPS {
                    return 50.0 / d_n;
                }
                // m δ_m = n δ_n e^{−λ δ_n} / (1 − e^{−λ δ_n})
                let (mut lo, mut hi) = (EPS, 50.0 / d_n.min(d_m));
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    let e = (-mid * d_n).exp();
                    if 1.0 - e < EPS {
                        hi = mid;
                    } else if m * d_m < n * d_n * e / (1.0 - e) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            }
            Attenuation::Bailey => {
                if f(F::PathLength) > EPS {
                    hits_w / f(F::PathLength)
                } else {
                    0.0
                }
            }
        }
    }

    /// Pimont et al. (2018) bias correction of the FPL estimate.
    pub fn fpl_bias(&self, idx: usize) -> f64 {
        let w_eff = self.get_f(F::BsEffectiveFreePath, idx) as f64;
        let beams = self.get_i(I::NumBeams, idx);
        if w_eff > EPS && beams > 0 {
            self.get_f(F::BsEffFreePathHits, idx) as f64 * self.get_f(F::BsEntering, idx) as f64 / (w_eff * w_eff * beams as f64)
        } else {
            0.0
        }
    }

    /// Bias-corrected contact-frequency density for a spherical leaf angle
    /// distribution: `2 (N − 1) / N · hits / path_length`, with each echo
    /// counted as its share of the pulse (a pulse is intercepted at most once).
    pub fn pad_g0_5(&self, idx: usize) -> f64 {
        let n = self.get_f(F::NumBeamsWeighted, idx) as f64;
        if n < 2.0 {
            return 0.0;
        }
        2.0 * (n - 1.0) * self.get_f(F::HitsWeighted, idx) as f64 / (EPS + n * self.get_f(F::PathLength, idx) as f64)
    }

    pub fn transmittance(&self, idx: usize) -> f64 {
        let f = |fld: F| self.get_f(fld, idx) as f64;
        if f(F::BsFreePath) > EPS {
            (-f(F::BsIntercepted) / f(F::BsFreePath)).exp()
        } else if f(F::BsEntering) > EPS {
            ((f(F::BsEntering) - f(F::BsIntercepted)) / f(F::BsEntering)).max(0.0)
        } else {
            1.0
        }
    }

    /// Weighted mean beam zenith (rad).
    pub fn mean_zenith(&self, idx: usize) -> f64 {
        let n = self.get_f(F::NumBeamsWeighted, idx);
        if n > 0.0 { (self.get_f(F::SumOfAngles, idx) / n) as f64 } else { 0.0 }
    }

    /// Circular mean beam azimuth (rad, `atan2(x, y)` in `[0, 2π)`) and its
    /// concentration (mean resultant length).
    pub fn mean_azimuth(&self, idx: usize) -> (f64, f64) {
        let n = self.get_f(F::NumBeamsWeighted, idx) as f64;
        if n <= 0.0 {
            return (0.0, 0.0);
        }
        let (s, c) = (self.get_f(F::SumSinAzimuth, idx) as f64 / n, self.get_f(F::SumCosAzimuth, idx) as f64 / n);
        (s.atan2(c).rem_euclid(2.0 * PI), s.hypot(c))
    }

    pub fn mean_laser_distance(&self, idx: usize) -> f64 {
        let n = self.get_f(F::NumBeamsWeighted, idx);
        if n > 0.0 { (self.get_f(F::SumOfLaserDistances, idx) / n) as f64 } else { 0.0 }
    }

    /// Standard deviation of the chords of the beams crossing the voxel.
    pub fn sd_path_length(&self, idx: usize) -> f64 {
        let n = self.get_i(I::NumBeams, idx) as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let mean = self.get_f(F::PathLength, idx) as f64 / n;
        (self.get_f(F::PathLengthSq, idx) as f64 / n - mean * mean).max(0.0).sqrt()
    }

    /// Centre height above the terrain (NaN without a DTM).
    pub fn distance_from_ground(&self, idx: usize) -> f64 {
        match &self.ground_height {
            Some(g) => self.center(idx)[2] - g[idx % (self.shape[0] * self.shape[1])],
            None => f64::NAN,
        }
    }

    /// Fraction of sub-voxel cells crossed by enough beams, and their bitmap.
    pub fn exploration(&self, idx: usize) -> (f64, u64) {
        let n_sub = self.params.subvoxel_split.pow(3);
        let Some(counts) = self.subvoxel_counts.as_ref().filter(|_| n_sub > 0) else { return (0.0, 0) };
        let mut bits = 0u64;
        for (b, &c) in counts[idx * n_sub..(idx + 1) * n_sub].iter().enumerate() {
            if c >= self.params.subvoxel_min_beams {
                bits |= 1 << b;
            }
        }
        (bits.count_ones() as f64 / n_sub as f64, bits)
    }

    /// `G` at the voxel's mean beam zenith for the analytic [`Lad`].
    pub fn g_analytic(&self, idx: usize) -> f64 {
        compute_g(self.mean_zenith(idx), &self.params.lad)
    }

    /// Plant area density `λ / G(mean zenith)` with the first attenuation
    /// method and the analytic leaf angle distribution.
    pub fn pad_g_corrected(&self, idx: usize) -> f64 {
        if self.get_f(F::PathLength, idx) <= 0.0 {
            return 0.0;
        }
        let g = self.g_analytic(idx);
        if g > 0.0 { self.attenuation(idx, self.params.attenuation[0]) / g } else { 0.0 }
    }

    /// The inclination distribution governing a voxel (its predominant tree's).
    pub fn voxel_iad(&self, idx: usize) -> Option<&super::TreeIad> {
        let tid = *self.predominant_tree.as_ref()?.get(idx)?;
        self.tree_iad.get(&tid)
    }

    /// Plant / leaf / wood area density. With an inclination distribution the
    /// `G` values are the tree's angle-integrated ones (Vicari et al. 2019),
    /// FPL is taken per class from the class free paths, and Bailey solves
    /// eq. 10 per class; otherwise `G` comes from the analytic [`Lad`] and
    /// leaf / wood split λ by their share of the hits.
    pub fn area_density(&self, idx: usize, method: Attenuation) -> AreaDensity {
        let mut out = AreaDensity::default();
        let path = self.get_f(F::PathLength, idx) as f64;
        if path <= 0.0 {
            return out;
        }
        let hits = self.get_i(I::NumHits, idx) as f64;
        let hit_total = hits.max(EPS);
        let leaf = self.get_i(I::NumHitLeaf, idx) as f64;
        let wood = self.get_i(I::NumHitWood, idx) as f64;
        let Some(iad) = self.voxel_iad(idx) else {
            let g = self.g_analytic(idx);
            if g > 0.0 {
                let lambda = self.attenuation(idx, method);
                out.pad = lambda / g;
                if self.has_leaf {
                    out.lad = lambda * leaf / hit_total / g;
                }
                if self.has_wood {
                    out.wad = lambda * wood / hit_total / g;
                }
            }
            return out;
        };
        match method {
            Attenuation::Bailey => {
                let beams = self.get_f(F::NumBeamsWeighted, idx) as f64;
                if iad.bailey_g_leaf > 0.0 && leaf > 0.0 {
                    out.lad = solve_bailey_pad(path, beams, leaf, iad.bailey_g_leaf);
                }
                if iad.bailey_g_wood > 0.0 && wood > 0.0 {
                    out.wad = solve_bailey_pad(path, beams, wood, iad.bailey_g_wood);
                }
                out.pad = match (leaf > 0.0, wood > 0.0) {
                    (false, true) => out.wad,
                    (true, false) => out.lad,
                    (true, true) if iad.g_plant > 0.0 => solve_bailey_pad(path, beams, hits, iad.g_plant),
                    _ => 0.0,
                };
            }
            Attenuation::Fpl => {
                let class = |n: i32, fp: f32, g: f64| if g > 0.0 && fp as f64 > EPS { n as f64 / (fp as f64 * g) } else { 0.0 };
                // Plant hits count every vegetation echo, so pair them with the free
                // path of every vegetation segment (rayvoxel uses the unclassified
                // ones only, which gives 0 for a fully leaf / wood labelled cloud).
                let plant_path = self.get_f(F::FreePathLengthPlant, idx) + self.get_f(F::FreePathLengthLeaf, idx) + self.get_f(F::FreePathLengthWood, idx);
                out.pad = class(self.get_i(I::NumHitPlant, idx), plant_path, iad.g_plant);
                out.lad = class(self.get_i(I::NumHitLeaf, idx), self.get_f(F::FreePathLengthLeaf, idx), iad.g_leaf);
                out.wad = class(self.get_i(I::NumHitWood, idx), self.get_f(F::FreePathLengthWood, idx), iad.g_wood);
            }
            _ => {
                let lambda = self.attenuation(idx, method);
                if iad.g_plant > 0.0 {
                    out.pad = lambda / iad.g_plant;
                }
                if iad.g_leaf > 0.0 {
                    out.lad = lambda * leaf / hit_total / iad.g_leaf;
                }
                if iad.g_wood > 0.0 {
                    out.wad = lambda * wood / hit_total / iad.g_wood;
                }
            }
        }
        out
    }

    /// A derived quantity for every voxel, by name. See [`RayVoxels::METRICS`];
    /// `pad_<method>`, `lad_<method>`, `wad_<method>` and
    /// `attenuation_<method>` are accepted for any attenuation method.
    pub fn metric(&self, name: &str) -> Result<Vec<f64>> {
        use rayon::prelude::*;
        let n = self.n_voxels();
        let map = |f: &(dyn Fn(usize) -> f64 + Sync)| Ok((0..n).into_par_iter().map(f).collect());
        let volume = self.voxel_size.powi(3);
        match name {
            "state" => map(&|i| self.state(i) as u8 as f64),
            "pad_g0_5" => map(&|i| self.pad_g0_5(i)),
            "surface_area" => map(&|i| self.pad_g0_5(i) * volume),
            "pad_g_corrected" => map(&|i| self.pad_g_corrected(i)),
            "transmittance" => map(&|i| self.transmittance(i)),
            "mean_zenith_angle" => map(&|i| self.mean_zenith(i).to_degrees()),
            "mean_azimuth_angle" => map(&|i| self.mean_azimuth(i).0.to_degrees()),
            "azimuth_concentration" => map(&|i| self.mean_azimuth(i).1),
            "mean_laser_dist" => map(&|i| self.mean_laser_distance(i)),
            "sd_path_length" => map(&|i| self.sd_path_length(i)),
            "distance_from_ground" => map(&|i| self.distance_from_ground(i)),
            "exploration_rate" => map(&|i| self.exploration(i).0),
            "attenuation_fpl_biased" => map(&|i| self.attenuation(i, Attenuation::Fpl)),
            "attenuation_fpl_correction" => map(&|i| self.fpl_bias(i)),
            "attenuation_fpl" => map(&|i| self.attenuation(i, Attenuation::Fpl) - self.fpl_bias(i)),
            "g_plant" => map(&|i| self.voxel_iad(i).map_or_else(|| self.g_analytic(i), |t| t.g_plant)),
            "g_leaf" => map(&|i| self.voxel_iad(i).map_or(0.0, |t| t.g_leaf)),
            "g_wood" => map(&|i| self.voxel_iad(i).map_or(0.0, |t| t.g_wood)),
            "wood_volume_density" => match &self.wood_volume {
                Some(w) => map(&|i| w[i] as f64 / volume),
                None => Err(Error::invalid("no wood volume has been added")),
            },
            other => {
                let (kind, method) = other.split_once('_').ok_or_else(|| Error::invalid(format!("unknown voxel metric {other:?}")))?;
                let m = Attenuation::parse(method).map_err(|_| Error::invalid(format!("unknown voxel metric {other:?}")))?;
                match kind {
                    "attenuation" => map(&|i| self.attenuation(i, m)),
                    "pad" => map(&|i| self.area_density(i, m).pad),
                    "lad" => map(&|i| self.area_density(i, m).lad),
                    "wad" => map(&|i| self.area_density(i, m).wad),
                    _ => Err(Error::invalid(format!("unknown voxel metric {other:?}"))),
                }
            }
        }
    }

    pub const METRICS: [&'static str; 19] = [
        "state", "pad_g0_5", "surface_area", "pad_g_corrected", "transmittance", "mean_zenith_angle",
        "mean_azimuth_angle", "azimuth_concentration", "mean_laser_dist", "sd_path_length", "distance_from_ground",
        "exploration_rate", "attenuation_fpl_biased", "attenuation_fpl_correction", "attenuation_fpl", "g_plant",
        "g_leaf", "g_wood", "wood_volume_density",
    ];
}
