// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Array helpers with NumPy's semantics.
//!
//! Code moved here from the Python package must give what it gave there, so
//! these follow NumPy exactly where it matters: histogram bins are
//! half-open except the last, quantiles interpolate linearly, `arange`
//! takes its length from `ceil((stop - start) / step)`, and `gradient` uses
//! one-sided differences at the ends.

/// `np.arange(start, stop, step)`.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil();
    let n = if n.is_finite() && n > 0.0 { n as usize } else { 0 };
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// `np.histogram(values, bins=edges)[0]`: bin `i` holds `edges[i] <= v <
/// edges[i + 1]`, the last bin also its right edge; NaN and values outside
/// are dropped. `edges` must be increasing.
pub fn histogram(values: impl IntoIterator<Item = f64>, edges: &[f64]) -> Vec<u64> {
    let nb = edges.len().saturating_sub(1);
    let mut counts = vec![0u64; nb];
    if nb == 0 {
        return counts;
    }
    let (lo, hi) = (edges[0], edges[nb]);
    for v in values {
        if !(v >= lo && v <= hi) {
            continue;
        }
        let i = if v == hi { nb - 1 } else { edges.partition_point(|&e| e <= v) - 1 };
        counts[i] += 1;
    }
    counts
}

/// `np.histogram(values, bins=edges, weights=w)[0]`.
pub fn histogram_weighted(values: &[f64], weights: &[f64], edges: &[f64]) -> Vec<f64> {
    let nb = edges.len().saturating_sub(1);
    let mut sums = vec![0.0; nb];
    if nb == 0 {
        return sums;
    }
    let (lo, hi) = (edges[0], edges[nb]);
    for (&v, &w) in values.iter().zip(weights) {
        if !(v >= lo && v <= hi) {
            continue;
        }
        let i = if v == hi { nb - 1 } else { edges.partition_point(|&e| e <= v) - 1 };
        sums[i] += w;
    }
    sums
}

/// `np.searchsorted(sorted, v, side="right")`.
pub fn searchsorted_right(sorted: &[f64], v: f64) -> usize {
    sorted.partition_point(|&e| e <= v)
}

/// `np.searchsorted(sorted, v, side="left")`.
pub fn searchsorted_left(sorted: &[f64], v: f64) -> usize {
    sorted.partition_point(|&e| e < v)
}

/// `np.quantile(values, q)` (linear interpolation). NaN for an empty input.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&v, q)
}

/// [`quantile`] of an already sorted slice.
pub fn quantile_sorted(v: &[f64], q: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    // NumPy's lerp: a + (b - a) * t, taken from the far end when t >= 0.5.
    let (a, b) = (v[lo], v[hi]);
    if frac >= 0.5 {
        b - (b - a) * (1.0 - frac)
    } else {
        a + (b - a) * frac
    }
}

/// `np.median(values)`: the mean of the two middle values for an even count.
pub fn median(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `np.gradient(f, h)` (second-order interior, first-order ends).
pub fn gradient(f: &[f64], h: f64) -> Vec<f64> {
    let n = f.len();
    match n {
        0 => vec![],
        1 => vec![0.0],
        _ => (0..n)
            .map(|i| {
                if i == 0 {
                    (f[1] - f[0]) / h
                } else if i == n - 1 {
                    (f[n - 1] - f[n - 2]) / h
                } else {
                    (f[i + 1] - f[i - 1]) / (2.0 * h)
                }
            })
            .collect(),
    }
}

/// `np.nanmean`, NaN when no value is finite.
pub fn nanmean(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for v in values {
        if !v.is_nan() {
            s += v;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

/// `np.sum` of a contiguous float array: NumPy's pairwise summation (eight
/// accumulators over blocks of up to 128 values), bit for bit. 0 when empty.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        return values.iter().fold(-0.0, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += values[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[i..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&values[..n2]) + pairwise_sum(&values[n2..])
}

/// [`pairwise_sum`] of a float32 array, which NumPy sums in float32.
pub fn pairwise_sum_f32(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        return values.iter().fold(-0.0, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += values[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[i..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f32(&values[..n2]) + pairwise_sum_f32(&values[n2..])
}

/// `np.nanmedian`: the median of the values that are not NaN, NaN if none are.
pub fn nanmedian(values: impl IntoIterator<Item = f64>) -> f64 {
    let v: Vec<f64> = values.into_iter().filter(|x| !x.is_nan()).collect();
    median(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_follows_numpy_blocks() {
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
        let v: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37).sin() * 1e3).collect();
        let naive: f64 = v.iter().sum();
        assert!((pairwise_sum(&v) - naive).abs() < 1e-9);
        let (a, b) = (0.1 + 0.1, 0.1 + 0.1);
        assert_eq!(pairwise_sum(&[0.1; 16]), ((a + b) + (a + b)) + ((a + b) + (a + b)));
    }

    #[test]
    fn nanmedian_skips_nan() {
        assert_eq!(nanmedian([3.0, f64::NAN, 1.0, 2.0]), 2.0);
        assert!(nanmedian([f64::NAN]).is_nan());
    }

    #[test]
    fn arange_matches_numpy_lengths() {
        assert_eq!(arange(0.0, 95.0, 5.0).len(), 19);
        assert_eq!(arange(0.0, 1.0 + 0.1, 0.1).len(), 11);
        assert!(arange(5.0, 1.0, 1.0).is_empty());
    }

    #[test]
    fn histogram_closes_the_last_bin() {
        let c = histogram([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, f64::NAN], &[0.0, 1.0, 2.0]);
        assert_eq!(c, vec![2, 3]);
    }

    #[test]
    fn quantile_and_median_follow_numpy() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&v), 2.5);
        assert!((quantile(&v, 0.98) - 3.94).abs() < 1e-12);
        assert_eq!(quantile(&v, 0.0), 1.0);
        assert_eq!(quantile(&v, 1.0), 4.0);
    }

    #[test]
    fn gradient_uses_one_sided_ends() {
        assert_eq!(gradient(&[0.0, 1.0, 4.0, 9.0], 1.0), vec![1.0, 2.0, 4.0, 5.0]);
    }
}
