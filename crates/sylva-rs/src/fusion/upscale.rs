// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Plot values from TLS carried over an ALS survey with a regression.
//!
//! The area-based approach (Næsset 2002): a plot value `y` measured on the
//! ground (here from TLS: above-ground biomass from QSM wood volumes and a
//! wood density, basal area, stem density) is regressed on ALS metrics `x`
//! of the same plots, and the model is applied to every cell of a grid of the
//! same metrics. Two ordinary least-squares forms are offered:
//!
//! * `linear`: `y = b0 + Σ bk xk`;
//! * `loglog`: `ln y = b0 + Σ bk ln xk`, the usual form for biomass. Its
//!   predictions are taken back with Baskerville's (1972) correction,
//!   `exp(ŷ + σ²/2)`, which estimates the mean rather than the median.
//!
//! Leave-one-out predictions come in closed form from the hat matrix (the
//! residual `e_i / (1 - h_ii)`, and for `loglog` the residual variance
//! without plot `i`, `((n - k) σ² - e_i² / (1 - h_ii)) / (n - k - 1)`), and
//! equal refitting the model without each plot. Each prediction carries the
//! standard error of a new observation, `σ √(1 + x₀ᵀ (XᵀX)⁻¹ x₀)`, and the
//! Student t interval at `level`; for `loglog` the interval is that of
//! `ln y` taken back, and the standard error that of the log-normal
//! distribution it implies. Cells whose metrics lie outside the range of the
//! plots are flagged as extrapolated.

use nalgebra::{DMatrix, DVector};

use crate::error::{Error, Result};

/// Model form of [`fit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Linear,
    LogLog,
}

impl Form {
    pub fn parse(s: &str) -> Result<Form> {
        match s {
            "linear" => Ok(Form::Linear),
            "loglog" => Ok(Form::LogLog),
            _ => Err(Error::invalid(format!("unknown model {s:?}; expected 'linear' or 'loglog'"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Form::Linear => "linear",
            Form::LogLog => "loglog",
        }
    }
}

/// A fitted model.
#[derive(Debug, Clone)]
pub struct Fit {
    pub form: Form,
    /// Intercept first, then one per predictor (on the model's scale).
    pub coef: Vec<f64>,
    pub se: Vec<f64>,
    /// Residual standard error and its degrees of freedom `n - k`.
    pub sigma: f64,
    pub df: usize,
    pub n: usize,
    /// On the model's scale (of `ln y` for `loglog`).
    pub r2: f64,
    pub adj_r2: f64,
    /// `(XᵀX)⁻¹`, `k x k` row-major.
    pub xtx_inv: Vec<f64>,
    /// Back-transformation factor: `exp(σ²/2)` for `loglog`, 1 otherwise.
    pub correction: f64,
    /// Fitted values and leave-one-out predictions, both on the scale of `y`.
    pub fitted: Vec<f64>,
    pub loo: Vec<f64>,
    /// Leave-one-out root mean square error, mean error (prediction minus
    /// observation), `1 - PRESS / SS`, and RMSE relative to the mean of `y` (%).
    pub loo_rmse: f64,
    pub loo_bias: f64,
    pub loo_r2: f64,
    pub loo_rrmse: f64,
    /// Range of each predictor over the plots.
    pub x_min: Vec<f64>,
    pub x_max: Vec<f64>,
}

fn design(x: &[Vec<f64>], form: Form, rows: usize) -> Option<DMatrix<f64>> {
    let k = x.len() + 1;
    let mut m = DMatrix::zeros(rows, k);
    for i in 0..rows {
        m[(i, 0)] = 1.0;
        for (j, col) in x.iter().enumerate() {
            let v = col[i];
            let v = if form == Form::LogLog { if v > 0.0 { v.ln() } else { return None } } else { v };
            if !v.is_finite() {
                return None;
            }
            m[(i, j + 1)] = v;
        }
    }
    Some(m)
}

/// Fit `y` on the columns of `x` (one per predictor, one value per plot).
///
/// # Errors
/// With fewer than `k + 2` plots (`k` coefficients), non-finite values,
/// values that are not positive for `loglog`, or collinear predictors.
pub fn fit(y: &[f64], x: &[Vec<f64>], form: Form) -> Result<Fit> {
    let n = y.len();
    let k = x.len() + 1;
    if x.iter().any(|c| c.len() != n) {
        return Err(Error::invalid("every predictor needs one value per plot"));
    }
    if n < k + 2 {
        return Err(Error::invalid(format!("{n} plots are too few for {k} coefficients and a leave-one-out check; need at least {}", k + 2)));
    }
    if y.iter().chain(x.iter().flatten()).any(|v| !v.is_finite()) {
        return Err(Error::invalid("plot values and predictors must be finite"));
    }
    if form == Form::LogLog && (y.iter().chain(x.iter().flatten()).any(|&v| v <= 0.0)) {
        return Err(Error::invalid("a loglog model needs positive plot values and predictors; use 'linear' or drop the zeros"));
    }
    let xm = design(x, form, n).ok_or_else(|| Error::invalid("predictors must be finite (and positive for loglog)"))?;
    let ym = DVector::from_iterator(n, y.iter().map(|&v| if form == Form::LogLog { v.ln() } else { v }));
    // Collinear (or constant) predictors: a column of X that the others
    // reproduce to within 1e-9 of its size, relative to the largest
    // singular value of X with its columns scaled to unit length.
    let mut scaled = xm.clone();
    for mut c in scaled.column_iter_mut() {
        let n = c.norm();
        if n > 0.0 {
            c /= n;
        }
    }
    let sv = scaled.singular_values();
    let (smax, smin) = (sv.max(), sv.min());
    if !(smin > 1e-9 * smax) {
        return Err(Error::invalid("the predictors are collinear (or constant); drop one"));
    }
    let xtx = xm.transpose() * &xm;
    let inv = xtx.clone().cholesky().map(|c| c.inverse()).ok_or_else(|| Error::invalid("the predictors are collinear (or constant); drop one"))?;
    let b = &inv * xm.transpose() * &ym;
    let fitted_m = &xm * &b;
    let e = &ym - &fitted_m;
    let df = n - k;
    let rss: f64 = e.iter().map(|v| v * v).sum();
    let sigma2 = rss / df as f64;
    let mean = ym.mean();
    let ss: f64 = ym.iter().map(|v| (v - mean) * (v - mean)).sum();
    let r2 = if ss > 0.0 { 1.0 - rss / ss } else { f64::NAN };
    let adj_r2 = if ss > 0.0 { 1.0 - (rss / df as f64) / (ss / (n - 1) as f64) } else { f64::NAN };
    let back = |m: f64, s2: f64| if form == Form::LogLog { (m + 0.5 * s2).exp() } else { m };
    let fitted: Vec<f64> = fitted_m.iter().map(|&m| back(m, sigma2)).collect();
    let mut loo = Vec::with_capacity(n);
    for i in 0..n {
        let xi = xm.row(i).transpose();
        let h = (xi.transpose() * &inv * &xi)[(0, 0)];
        if h >= 1.0 - 1e-12 {
            return Err(Error::invalid(format!("plot {i} alone determines the fit (leverage 1); leave-one-out is undefined")));
        }
        let el = e[i] / (1.0 - h);
        let s2 = (((df as f64) * sigma2 - e[i] * e[i] / (1.0 - h)) / (df as f64 - 1.0)).max(0.0);
        loo.push(back(ym[i] - el, s2));
    }
    let ymean = y.iter().sum::<f64>() / n as f64;
    let press: f64 = loo.iter().zip(y).map(|(p, o)| (p - o) * (p - o)).sum();
    let ss_y: f64 = y.iter().map(|v| (v - ymean) * (v - ymean)).sum();
    let loo_rmse = (press / n as f64).sqrt();
    Ok(Fit {
        form,
        coef: b.iter().copied().collect(),
        se: (0..k).map(|j| (sigma2 * inv[(j, j)]).sqrt()).collect(),
        sigma: sigma2.sqrt(),
        df,
        n,
        r2,
        adj_r2,
        xtx_inv: inv.transpose().iter().copied().collect(),
        correction: if form == Form::LogLog { (0.5 * sigma2).exp() } else { 1.0 },
        fitted,
        loo_bias: loo.iter().zip(y).map(|(p, o)| p - o).sum::<f64>() / n as f64,
        loo,
        loo_rmse,
        loo_r2: if ss_y > 0.0 { 1.0 - press / ss_y } else { f64::NAN },
        loo_rrmse: 100.0 * loo_rmse / ymean,
        x_min: x.iter().map(|c| c.iter().copied().fold(f64::INFINITY, f64::min)).collect(),
        x_max: x.iter().map(|c| c.iter().copied().fold(f64::NEG_INFINITY, f64::max)).collect(),
    })
}

/// Predictions of [`predict`], one per row.
#[derive(Debug, Clone, Default)]
pub struct Prediction {
    pub mean: Vec<f64>,
    pub se: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub extrapolated: Vec<bool>,
}

/// Apply a model to new predictor values (columns as in [`fit`]); rows with
/// a missing or (for `loglog`) non-positive value are NaN.
pub fn predict(f: &Fit, x: &[Vec<f64>], level: f64) -> Result<Prediction> {
    let k = f.coef.len();
    if x.len() + 1 != k {
        return Err(Error::invalid(format!("the model has {} predictors, {} were given", k - 1, x.len())));
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(Error::invalid("level must be between 0 and 1"));
    }
    let rows = x.first().map(|c| c.len()).unwrap_or(0);
    if x.iter().any(|c| c.len() != rows) {
        return Err(Error::invalid("every predictor needs the same number of values"));
    }
    // How many standard errors wide the interval is, from Student's t with
    // the model's degrees of freedom: narrower for a model fitted on many
    // plots, wider for one fitted on a handful.
    let t = t_quantile(0.5 * (1.0 + level), f.df as f64);
    let s2 = f.sigma * f.sigma;
    let mut out = Prediction::default();
    // One row's predictors, with a leading 1 for the intercept. The vector is
    // reused across rows rather than allocated per row.
    let mut x0 = vec![0.0; k];
    for i in 0..rows {
        x0[0] = 1.0;
        let mut ok = true;
        let mut extra = false;
        for (j, c) in x.iter().enumerate() {
            let v = c[i];
            // A predictor outside the range the model was fitted on: the
            // prediction is still made, but flagged, because a regression
            // says nothing dependable beyond the data behind it.
            extra |= v < f.x_min[j] || v > f.x_max[j];
            let v = if f.form == Form::LogLog { if v > 0.0 { v.ln() } else { f64::NAN } } else { v };
            ok &= v.is_finite();
            x0[j + 1] = v;
        }
        if !ok {
            out.mean.push(f64::NAN);
            out.se.push(f64::NAN);
            out.lower.push(f64::NAN);
            out.upper.push(f64::NAN);
            out.extrapolated.push(false);
            continue;
        }
        // The fitted value, then its uncertainty. `h` is this row's leverage,
        // which grows the further its predictors sit from the middle of the
        // fitting data, so a prediction out at the edge of the plots comes
        // with a wider interval. The `1.0 +` is the scatter of a single new
        // observation about the line, on top of the uncertainty in the line.
        let m: f64 = x0.iter().zip(&f.coef).map(|(a, b)| a * b).sum();
        let mut h = 0.0;
        for a in 0..k {
            for b in 0..k {
                h += x0[a] * f.xtx_inv[a * k + b] * x0[b];
            }
        }
        let s = (s2 * (1.0 + h)).sqrt();
        match f.form {
            Form::Linear => {
                out.mean.push(m);
                out.se.push(s);
                out.lower.push(m - t * s);
                out.upper.push(m + t * s);
            }
            // A log-log model was fitted on logarithms, so exponentiating the
            // fitted value gives a median, not a mean: the `0.5 * s2` term is
            // the usual correction back to the mean. The interval needs no
            // correction, since exponentiating is monotonic, which is why it
            // is not symmetric about the mean.
            Form::LogLog => {
                out.mean.push((m + 0.5 * s2).exp());
                out.se.push((m + 0.5 * s * s).exp() * ((s * s).exp() - 1.0).sqrt());
                out.lower.push((m - t * s).exp());
                out.upper.push((m + t * s).exp());
            }
        }
        out.extrapolated.push(extra);
    }
    Ok(out)
}

/// Plot totals per hectare from a plot's trees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlotSummary {
    /// Above-ground biomass (Mg/ha), wood volume (m³/ha), basal area
    /// (m²/ha) and stems per hectare, of the trees with `dbh >= min_dbh`.
    pub agb: f64,
    pub volume: f64,
    pub basal_area: f64,
    pub stems: f64,
    pub n_trees: usize,
    /// Trees counted without a volume (they add nothing to `agb`, `volume`).
    pub missing_volume: usize,
}

/// Sum a plot's trees: `dbh` (m), wood `volume` (m³), over `area` (m²),
/// with `wood_density` (kg/m³).
pub fn plot_summary(dbh: &[f64], volume: &[f64], area: f64, wood_density: f64, min_dbh: f64) -> Result<PlotSummary> {
    if dbh.len() != volume.len() {
        return Err(Error::invalid("one volume per tree is needed"));
    }
    if !(area.is_finite() && area > 0.0) || !(wood_density.is_finite() && wood_density > 0.0) || !min_dbh.is_finite() {
        return Err(Error::invalid("area and wood_density must be positive and min_dbh finite"));
    }
    let ha = area / 10_000.0;
    let (mut v, mut ba, mut n, mut miss) = (0.0, 0.0, 0usize, 0usize);
    for (&d, &vol) in dbh.iter().zip(volume) {
        if min_dbh > 0.0 && !(d >= min_dbh) {
            continue;
        }
        n += 1;
        if d.is_finite() {
            ba += std::f64::consts::PI * 0.25 * d * d;
        }
        if vol.is_finite() {
            v += vol;
        } else {
            miss += 1;
        }
    }
    Ok(PlotSummary { agb: v * wood_density / 1000.0 / ha, volume: v / ha, basal_area: ba / ha, stems: n as f64 / ha, n_trees: n, missing_volume: miss })
}

// ------------------------------------------------------------ distributions

/// `ln Γ(x)` for `x > 0` (Lanczos, g = 7, 9 terms).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8, 771.323_428_777_653_1, -176.615_029_162_140_6, 12.507_343_278_686_905, -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6, 1.505_632_735_149_311_6e-7];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued fraction of the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..400 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Regularised incomplete beta function `I_x(a, b)`.
pub(crate) fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Student's t distribution function with `df` degrees of freedom.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    let tail = 0.5 * inc_beta(0.5 * df, 0.5, df / (df + t * t));
    if t >= 0.0 { 1.0 - tail } else { tail }
}

/// Quantile of Student's t distribution (bisection on [`t_cdf`]).
pub fn t_quantile(p: f64, df: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) || !(df > 0.0) {
        return f64::NAN;
    }
    if p < 0.5 {
        return -t_quantile(1.0 - p, df);
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    while t_cdf(hi, df) < p {
        hi *= 2.0;
        if hi > 1e12 {
            return f64::INFINITY;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-13 * hi.max(1.0) {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_quantiles_match_tables() {
        // scipy.stats.t.ppf(0.975, df)
        for (df, q) in [(1.0, 12.706204736174698), (2.0, 4.302652729749464), (5.0, 2.5705818356363146), (10.0, 2.2281388519862744), (30.0, 2.0422724563012373), (1000.0, 1.9623390808264078)] {
            assert!((t_quantile(0.975, df) - q).abs() < 1e-9 * q, "{df}: {}", t_quantile(0.975, df));
        }
        assert!((t_quantile(0.95, 3.0) - 2.3533634348018264).abs() < 1e-9);
        assert!((t_quantile(0.025, 5.0) + 2.5705818356363146).abs() < 1e-9);
        assert!((t_cdf(0.0, 4.0) - 0.5).abs() < 1e-15);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
    }

    fn brute_loo(y: &[f64], x: &[Vec<f64>], form: Form) -> Vec<f64> {
        (0..y.len())
            .map(|i| {
                let yy: Vec<f64> = y.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, v)| *v).collect();
                let xx: Vec<Vec<f64>> = x.iter().map(|c| c.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, v)| *v).collect()).collect();
                let f = fit(&yy, &xx, form).unwrap();
                let xi: Vec<Vec<f64>> = x.iter().map(|c| vec![c[i]]).collect();
                predict(&f, &xi, 0.95).unwrap().mean[0]
            })
            .collect()
    }

    #[test]
    fn exact_relations_and_leave_one_out() {
        let x1: Vec<f64> = vec![5.0, 8.0, 12.0, 15.0, 18.0, 22.0, 25.0, 30.0];
        let x2: Vec<f64> = vec![40.0, 55.0, 60.0, 70.0, 72.0, 80.0, 85.0, 90.0];
        // An exact power law is recovered exactly, with no residual.
        let y: Vec<f64> = x1.iter().zip(&x2).map(|(a, b)| 3.0 * a.powf(1.5) * b.powf(0.4)).collect();
        let f = fit(&y, &[x1.clone(), x2.clone()], Form::LogLog).unwrap();
        assert!((f.coef[0] - 3f64.ln()).abs() < 1e-9 && (f.coef[1] - 1.5).abs() < 1e-10 && (f.coef[2] - 0.4).abs() < 1e-10, "{:?}", f.coef);
        assert!(f.sigma < 1e-9 && (f.r2 - 1.0).abs() < 1e-12);
        // With noise: closed-form leave-one-out equals refitting.
        let noise = [0.1, -0.05, 0.08, -0.12, 0.02, 0.06, -0.09, 0.03];
        let yn: Vec<f64> = y.iter().zip(noise).map(|(v, e)| v * f64::exp(e)).collect();
        for form in [Form::LogLog, Form::Linear] {
            let f = fit(&yn, &[x1.clone(), x2.clone()], form).unwrap();
            let b = brute_loo(&yn, &[x1.clone(), x2.clone()], form);
            for (a, c) in f.loo.iter().zip(&b) {
                assert!((a - c).abs() < 1e-9 * c.abs(), "{form:?}: {a} vs {c}");
            }
            assert_eq!(f.df, 5);
        }
        // The prediction interval of a linear model: t * sigma * sqrt(1 + h).
        let f = fit(&yn, std::slice::from_ref(&x1), Form::Linear).unwrap();
        let p = predict(&f, &[vec![10.0, f64::NAN, 100.0]], 0.9).unwrap();
        let h = f.xtx_inv[0] + 2.0 * 10.0 * f.xtx_inv[1] + 100.0 * f.xtx_inv[3];
        let s = f.sigma * (1.0 + h).sqrt();
        assert!((p.se[0] - s).abs() < 1e-12 && (p.upper[0] - p.mean[0] - t_quantile(0.95, 6.0) * s).abs() < 1e-9);
        assert!(p.mean[1].is_nan() && !p.extrapolated[0] && p.extrapolated[2]);
    }

    #[test]
    fn bad_fits_are_refused() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!(fit(&[1.0, 2.0, 3.0], &[vec![1.0, 2.0, 3.0]], Form::Linear).is_err());
        assert!(fit(&[1.0, 2.0, 3.0, 0.0], std::slice::from_ref(&x), Form::LogLog).is_err());
        assert!(fit(&[1.0, 2.0, 3.0, 4.0], &[x.clone(), x.iter().map(|v| 2.0 * v).collect()], Form::Linear).is_err());
        assert!(fit(&[1.0, 2.0, f64::NAN, 4.0], std::slice::from_ref(&x), Form::Linear).is_err());
        assert!(fit(&[1.0, 2.0, 3.0, 4.0], &[vec![1.0; 4]], Form::Linear).is_err());
        let f = fit(&[1.0, 2.1, 2.9, 4.2], &[x], Form::Linear).unwrap();
        assert!(predict(&f, &[], 0.95).is_err() && predict(&f, &[vec![1.0]], 1.0).is_err());
        assert!(Form::parse("log").is_err());
    }

    #[test]
    fn plot_totals() {
        let s = plot_summary(&[0.2, 0.4, 0.05, f64::NAN], &[0.5, 2.0, 0.01, 1.0], 1000.0, 500.0, 0.1).unwrap();
        assert_eq!(s.n_trees, 2);
        assert!((s.agb - 2.5 * 0.5 / 0.1).abs() < 1e-12);
        assert!((s.basal_area - std::f64::consts::PI * 0.25 * (0.04 + 0.16) / 0.1).abs() < 1e-12);
        assert!((s.stems - 20.0).abs() < 1e-12);
        let all = plot_summary(&[0.2, f64::NAN], &[0.5, f64::NAN], 10_000.0, 500.0, 0.0).unwrap();
        assert_eq!((all.n_trees, all.missing_volume), (2, 1));
        assert!(plot_summary(&[0.2], &[], 1.0, 1.0, 0.0).is_err());
        assert!(plot_summary(&[0.2], &[1.0], 0.0, 1.0, 0.0).is_err());
    }
}
