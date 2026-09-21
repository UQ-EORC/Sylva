//! Small dense least-squares solver (Levenberg–Marquardt with numeric Jacobian).

use nalgebra::{DMatrix, DVector};

/// Result of a least-squares fit.
#[derive(Debug, Clone)]
pub struct LmResult {
    pub x: Vec<f64>,
    pub residuals: Vec<f64>,
    pub rmse: f64,
    pub iterations: usize,
}

/// Minimise `sum r_i(x)^2` starting from `x0`.
///
/// `f(x, r)` must fill `r` with the residual vector; its length is fixed on
/// the first call. Intended for tiny parameter vectors (circles, cylinders).
pub fn levenberg_marquardt<F>(mut f: F, x0: &[f64], max_iter: usize, tol: f64) -> LmResult
where
    F: FnMut(&[f64], &mut Vec<f64>),
{
    let n = x0.len();
    let mut x = DVector::from_column_slice(x0);
    let mut r = Vec::new();
    f(x.as_slice(), &mut r);
    let m = r.len();
    let mut cost = r.iter().map(|v| v * v).sum::<f64>();
    let mut lambda = 1e-3;
    let mut jac = DMatrix::<f64>::zeros(m, n);
    let mut r_pert = Vec::with_capacity(m);
    let mut iterations = 0;
    for it in 0..max_iter {
        iterations = it + 1;
        // Numeric Jacobian (forward differences).
        for j in 0..n {
            let h = 1e-6 * x[j].abs().max(1.0);
            let mut xp = x.clone();
            xp[j] += h;
            f(xp.as_slice(), &mut r_pert);
            for i in 0..m {
                jac[(i, j)] = (r_pert[i] - r[i]) / h;
            }
        }
        let rv = DVector::from_column_slice(&r);
        let jt = jac.transpose();
        let jtj = &jt * &jac;
        let g = &jt * &rv;
        if g.amax() < tol {
            break;
        }
        let mut improved = false;
        for _ in 0..10 {
            let mut a = jtj.clone();
            for j in 0..n {
                a[(j, j)] += lambda * (jtj[(j, j)].abs().max(1e-12));
            }
            let Some(chol) = a.cholesky() else {
                lambda *= 10.0;
                continue;
            };
            let step = chol.solve(&g);
            let xn = &x - &step;
            f(xn.as_slice(), &mut r_pert);
            let cn = r_pert.iter().map(|v| v * v).sum::<f64>();
            if cn < cost {
                let rel = (cost - cn) / cost.max(1e-300);
                x = xn;
                std::mem::swap(&mut r, &mut r_pert);
                cost = cn;
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                if rel < tol || step.amax() < tol {
                    return finish(x, r, iterations);
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    finish(x, r, iterations)
}

fn finish(x: DVector<f64>, r: Vec<f64>, iterations: usize) -> LmResult {
    let m = r.len().max(1) as f64;
    let rmse = (r.iter().map(|v| v * v).sum::<f64>() / m).sqrt();
    LmResult { x: x.as_slice().to_vec(), residuals: r, rmse, iterations }
}
