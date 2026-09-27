// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! What will not fit in memory, refused before it is asked for.
//!
//! A voxel grid is `nx * ny * nz` cells whatever the cloud holds, so a plot
//! asked for at 1 cm is tens of billions of cells and the process dies with
//! no message worth reading - or, worse, takes the machine down with it. The
//! same goes for a neighbour graph over tens of millions of points. Every
//! such allocation is sized first and checked here, and an honest error says
//! what it would have needed and what would make it fit.
//!
//! The budget is the memory the system says is available, less a margin, or
//! whatever `SYLVA_MEM_BUDGET` asks for (in GB). It is a guard against the
//! obvious mistake, not a guarantee: nothing here tracks what is already
//! held, and a machine can still be pushed over by many smaller pieces.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, Result};

/// Set by [`set_budget`]; 0 means "work it out from the system".
static BUDGET: AtomicU64 = AtomicU64::new(0);

/// Memory the system reports as available (bytes), if it will say.
///
/// Linux only: `MemAvailable` from `/proc/meminfo`, which already accounts
/// for reclaimable cache. Elsewhere this is None and nothing is refused.
pub fn available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The most one allocation may ask for, in bytes.
///
/// `SYLVA_MEM_BUDGET` (GB) wins, then anything given to [`set_budget`], then
/// 80 % of what the system says is available. None when nothing is known.
pub fn budget() -> Option<u64> {
    if let Ok(v) = std::env::var("SYLVA_MEM_BUDGET") {
        if let Ok(gb) = v.trim().parse::<f64>() {
            if gb > 0.0 {
                return Some((gb * 1e9) as u64);
            }
        }
    }
    match BUDGET.load(Ordering::Relaxed) {
        0 => available().map(|a| a * 4 / 5),
        b => Some(b),
    }
}

/// Set the budget in bytes; 0 goes back to reading the system.
pub fn set_budget(bytes: u64) {
    BUDGET.store(bytes, Ordering::Relaxed);
}

/// Bytes as a human reads them.
pub fn human(bytes: u64) -> String {
    human_f64(bytes as f64)
}

/// Any size in bytes as a human reads it ("3.2 TB"): whole bytes below
/// 1000, then kB, MB, GB and TB to one decimal, never beyond TB.
pub fn human_f64(bytes: f64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut v = bytes;
    for (u, unit) in UNITS.iter().enumerate() {
        if v < 1000.0 || u == UNITS.len() - 1 {
            let n = if v.is_nan() { "nan".to_string() } else if u == 0 { format!("{v:.0}") } else { format!("{v:.1}") };
            return format!("{n} {unit}");
        }
        v /= 1000.0;
    }
    unreachable!()
}

/// Refuse an allocation of `bytes` that `what` is about to make.
///
/// `hint` says what would make it fit ("a larger voxel, or a smaller area").
/// Passes when the size is within budget, or when the system will not say
/// what it has.
pub fn check(bytes: u64, what: &str, hint: &str) -> Result<()> {
    match budget() {
        Some(b) if bytes > b => Err(Error::Invalid(format!(
            "{what} needs {}, and only {} is available: try {hint}. \
             Set SYLVA_MEM_BUDGET (GB) to raise the limit.",
            human(bytes),
            human(b),
        ))),
        _ => Ok(()),
    }
}

/// `cells * per_cell`, refusing sizes that cannot be counted at all.
pub fn check_cells(cells: u128, per_cell: u64, what: &str, hint: &str) -> Result<()> {
    let bytes = cells.saturating_mul(per_cell as u128);
    if bytes > u64::MAX as u128 {
        return Err(Error::Invalid(format!("{what} needs more memory than can be counted: try {hint}")));
    }
    check(bytes as u64, what, hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_as_python_writes_them() {
        let got: Vec<String> = [0.0, 999.6, 1050.0, 12345678.0, 3.2e15, -5.0, f64::INFINITY].iter().map(|&v| human_f64(v)).collect();
        assert_eq!(got, ["0 B", "1000 B", "1.1 kB", "12.3 MB", "3200.0 TB", "-5 B", "inf TB"]);
        assert_eq!(human(2_500_000_000), "2.5 GB");
    }

    #[test]
    fn a_budget_can_be_set_and_put_back() {
        let was = budget();
        set_budget(1_000_000);
        assert_eq!(budget(), Some(1_000_000));
        assert!(check(999_999, "a thing", "less of it").is_ok());
        let e = check(2_000_000, "a grid of 10 cells", "a larger voxel").unwrap_err().to_string();
        assert!(e.contains("2.0 MB") && e.contains("1.0 MB") && e.contains("a larger voxel"), "{e}");
        set_budget(0);
        assert_eq!(budget().is_some(), was.is_some());
    }

    #[test]
    fn a_count_that_cannot_fit_in_bytes_is_refused() {
        set_budget(1_000_000);
        let e = check_cells(u128::MAX / 2, 64, "a grid", "a larger voxel").unwrap_err().to_string();
        set_budget(0);
        assert!(e.contains("than can be counted"), "{e}");
    }

    #[test]
    fn sizes_read_the_way_people_write_them() {
        assert_eq!(human(512), "512 B");
        assert_eq!(human(1_500), "1.5 kB");
        assert_eq!(human(2_400_000_000), "2.4 GB");
        assert_eq!(human(3_200_000_000_000), "3.2 TB");
    }
}
