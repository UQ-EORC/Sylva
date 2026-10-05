// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Numbers laid out as Python's format specifications lay them out.
//!
//! Reports and log messages written by the core must read as the Python
//! package wrote them, so these follow Python's `format()`: fixed-point
//! digits are correctly rounded (Rust's formatting agrees), `nan` and `inf`
//! are spelled in lower case, `g` switches to scientific notation below
//! 1e-4 and from `10**precision` and drops trailing zeros, and `,` groups
//! thousands.

/// `format(x, f".{prec}f")`.
pub fn fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        format!("{x:.prec$}")
    }
}

/// `format(x, f"+.{prec}f")`.
pub fn signed(x: f64, prec: usize) -> String {
    let s = fixed(x, prec);
    if s.starts_with('-') { s } else { format!("+{s}") }
}

/// `format(x, f"{width}.{prec}f")`: right-aligned in `width` characters.
pub fn fixed_width(x: f64, width: usize, prec: usize) -> String {
    format!("{:>width$}", fixed(x, prec))
}

/// `format(x, f".{prec}%")`: the value times 100, fixed, and a percent sign.
pub fn percent(x: f64, prec: usize) -> String {
    format!("{}%", fixed(x * 100.0, prec))
}

/// `format(x, f".{prec}g")`.
pub fn general(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        return fixed(x, 0);
    }
    let p = prec.max(1);
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    // The exponent after rounding to p significant digits decides the notation.
    let sci = format!("{:.*e}", p - 1, x);
    let (mantissa, exp) = sci.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("exponent");
    if exp < -4 || exp >= p as i32 {
        let mantissa = strip_zeros(mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        strip_zeros(&format!("{:.*}", (p as i32 - 1 - exp) as usize, x)).to_string()
    }
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// `format(n, ",d")`.
pub fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    if n < 0 {
        out.push('-');
    }
    for (k, c) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `format(s, f">{width}")`: right-aligned, never truncated.
pub fn right(s: &str, width: usize) -> String {
    let pad = width.saturating_sub(s.chars().count());
    format!("{}{s}", " ".repeat(pad))
}

/// `format(s, f"<{width}")`: left-aligned, never truncated.
pub fn left(s: &str, width: usize) -> String {
    let pad = width.saturating_sub(s.chars().count());
    format!("{s}{}", " ".repeat(pad))
}

/// `str(b)` of a Python bool.
pub fn boolean(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_as_python() {
        assert_eq!(fixed(0.125, 2), "0.12");
        assert_eq!(fixed(2.675, 2), "2.67");
        assert_eq!(fixed(-0.0, 1), "-0.0");
        assert_eq!(fixed(f64::NAN, 3), "nan");
        assert_eq!(fixed(f64::NEG_INFINITY, 3), "-inf");
        assert_eq!(signed(0.004, 2), "+0.00");
        assert_eq!(signed(-0.004, 2), "-0.00");
        assert_eq!(signed(f64::INFINITY, 1), "+inf");
        assert_eq!(signed(f64::NAN, 1), "+nan");
        assert_eq!(fixed_width(41.13, 6, 1), "  41.1");
        assert_eq!(fixed_width(f64::INFINITY, 6, 1), "   inf");
        assert_eq!(percent(0.8, 0), "80%");
        assert_eq!(percent(-1.0, 0), "-100%");
        assert_eq!(percent(0.165, 0), "16%");
    }

    #[test]
    fn general_as_python() {
        for (x, s) in [(2537.0, "2537"), (4.613e-12, "4.613e-12"), (6.877, "6.877"), (1.759, "1.759"), (400.1, "400.1"), (123456.789, "1.235e+05"), (0.000123456, "0.0001235"), (1e-20, "1e-20"), (12345678.0, "1.235e+07"), (0.0, "0"), (-0.0, "-0"), (9999.5, "1e+04"), (0.00001, "1e-05"), (1.0, "1"), (0.5, "0.5"), (f64::NAN, "nan"), (f64::INFINITY, "inf"), (-3.25, "-3.25"), (1e100, "1e+100")] {
            assert_eq!(general(x, 4), s, "{x}");
        }
    }

    #[test]
    fn grouping_and_alignment() {
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(-1234), "-1,234");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(0), "0");
        assert_eq!(right("12", 4), "  12");
        assert_eq!(left("\u{e9}t\u{e9}", 5), "\u{e9}t\u{e9}  ");
        assert_eq!(left("longer", 2), "longer");
    }
}
