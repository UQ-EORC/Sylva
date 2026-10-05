// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Delimited text point clouds (XYZ, TXT, CSV, PTS).

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::{Point, PointCloud};

fn split<'a>(line: &'a str, delim: Option<char>) -> impl Iterator<Item = &'a str> {
    let it: Box<dyn Iterator<Item = &'a str>> = match delim {
        Some(d) => Box::new(line.split(d).map(str::trim).filter(|t| !t.is_empty())),
        None => Box::new(line.split_whitespace()),
    };
    it
}

fn detect_delim(line: &str) -> Option<char> {
    [',', ';', '\t'].into_iter().find(|&d| line.contains(d))
}

/// Read a delimited text file. The first three numeric columns are x, y, z;
/// remaining columns become attributes named from `columns`, from a header
/// line if present, else `col3`, `col4`, ...
///
/// A leading line holding a single integer (PTS point count) is skipped.
pub fn read_ascii(path: impl AsRef<Path>, columns: Option<&[String]>) -> Result<PointCloud> {
    let path = path.as_ref();
    let mut r = BufReader::new(std::fs::File::open(path)?);
    let mut first = String::new();
    r.read_line(&mut first)?;
    let mut second = String::new();
    r.read_line(&mut second)?;
    let probe = if second.trim().is_empty() { first.as_str() } else { second.as_str() };
    let delim = detect_delim(probe);

    let first_tokens: Vec<&str> = split(first.trim(), delim).collect();
    let numeric = |t: &str| t.parse::<f64>().is_ok();
    let mut header_names: Option<Vec<String>> = None;
    let mut pending: Vec<&str> = Vec::new();
    if first_tokens.len() == 1 && first_tokens[0].parse::<u64>().is_ok() && !second.trim().is_empty() {
        // PTS count line
    } else if !first_tokens.iter().all(|t| numeric(t)) {
        header_names = Some(first_tokens.iter().map(|t| t.trim_start_matches(['#', '/']).trim().to_string()).collect());
    } else {
        pending.push(first.as_str());
    }
    pending.push(second.as_str());

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut ncols = 0usize;
    let mut push_line = |line: &str| -> Result<()> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(());
        }
        let vals: Vec<f64> = split(line, delim)
            .map(|t| t.parse::<f64>().map_err(|_| Error::file(path, format!("bad number {t:?}"))))
            .collect::<Result<_>>()?;
        if rows.is_empty() {
            ncols = vals.len();
        } else if vals.len() != ncols {
            return Err(Error::file(path, format!("inconsistent column count ({} vs {ncols})", vals.len())));
        }
        rows.push(vals);
        Ok(())
    };
    for l in pending {
        push_line(l)?;
    }
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            break;
        }
        push_line(&line)?;
    }
    if !rows.is_empty() && ncols < 3 {
        return Err(Error::file(path, format!("expected at least 3 columns, found {ncols}")));
    }
    let xyz: Vec<Point> = rows.iter().map(|v| [v[0], v[1], v[2]]).collect();
    let mut cloud = PointCloud::new(xyz);
    let names: Vec<String> = match columns.map(|c| c.to_vec()).or(header_names) {
        Some(n) if n.len() == ncols => n[3..].to_vec(),
        Some(n) if n.len() + 3 == ncols => n,
        _ => (3..ncols).map(|i| format!("col{i}")).collect(),
    };
    for (k, name) in names.into_iter().enumerate() {
        let col: Vec<f64> = rows.iter().map(|v| v[3 + k]).collect();
        cloud.attrs.insert(name, Attr::F64(col));
    }
    Ok(cloud)
}

/// Write x, y, z and all attributes as delimited text.
pub fn write_ascii(cloud: &PointCloud, path: impl AsRef<Path>, delim: &str, header: bool, precision: usize) -> Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    if header {
        let mut names = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        names.extend(cloud.attrs.keys().cloned());
        writeln!(w, "{}", names.join(delim))?;
    }
    for i in 0..cloud.len() {
        let p = cloud.xyz[i];
        write!(w, "{:.prec$}{delim}{:.prec$}{delim}{:.prec$}", p[0], p[1], p[2], prec = precision)?;
        for a in cloud.attrs.values() {
            match a {
                Attr::F64(_) | Attr::F32(_) => write!(w, "{delim}{}", a.get_f64(i))?,
                _ => write!(w, "{delim}{}", a.get_f64(i) as i64)?,
            }
        }
        writeln!(w)?;
    }
    w.flush()?;
    Ok(())
}
