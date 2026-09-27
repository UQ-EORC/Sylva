// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Large-area airborne lidar: a catalogue of LAS/LAZ tiles and processing in
//! buffered chunks.
//!
//! An airborne survey arrives as hundreds of tiles, each too small to be
//! processed on its own (a DTM cell or a ground filter at a tile edge needs the
//! points across it) and all of them together too large for memory. The
//! approach is that of lidR's `LAScatalog` (Roussel et al. 2020):
//!
//! 1. [`Catalog::open`] reads only the headers: extent, point count, point
//!    format and CRS of every file, from which overlaps, gaps and mixed
//!    formats are found without touching a point.
//! 2. [`plan`] divides the area into chunks, one per tile or a regular grid,
//!    each with a *core* (the area it is responsible for) and a *buffer*
//!    around it.
//! 3. [`read_chunk`] streams only the files that overlap a chunk's buffered
//!    box and keeps only the points inside it, flagging the buffer points.
//! 4. [`run`] processes chunks on a bounded number of threads (sized against
//!    the memory budget of [`crate::limits`]) and returns the results in chunk
//!    order, so that nothing depends on how many threads ran or which
//!    finished first. Point outputs keep only core points; raster outputs are
//!    joined by [`mosaic`], each cell taken from the chunk whose core holds it.
//!
//! The built-in operations (ground classification, DTM, CHM, normalisation,
//! noise filtering, retiling, decimation) are in [`crate::als_ops`].

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use las::{Header, Vlr};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::io::las::{read_las, read_las_where, write_las_with_vlrs, LasWriteOptions};
use crate::pointcloud::Attr;
use crate::raster::Raster;
use crate::{limits, progress, Point, PointCloud};

/// One LAS/LAZ file of a catalogue, as its header describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    pub path: PathBuf,
    /// `[xmin, ymin, zmin, xmax, ymax, zmax]` from the header.
    pub bounds: [f64; 6],
    pub n_points: u64,
    /// LAS point data record format (0-10).
    pub point_format: u8,
    /// LAS version as `(major, minor)`.
    pub version: (u8, u8),
    /// `EPSG:<code>` when the CRS records name one, else the WKT, else None.
    pub crs: Option<String>,
    /// Coordinate quantisation per axis.
    pub scale: [f64; 3],
    pub offset: [f64; 3],
    /// A LAStools `.lax` index beside the file, or a COPC file (which is
    /// organised as an octree).
    pub spatial_index: bool,
    pub file_size: u64,
}

impl Tile {
    /// `[xmin, ymin, xmax, ymax]`.
    pub fn xy(&self) -> [f64; 4] {
        [self.bounds[0], self.bounds[1], self.bounds[3], self.bounds[4]]
    }
}

/// EPSG code of the outermost CRS of an OGC WKT string (WKT1 `AUTHORITY` or
/// WKT2 `ID`): the last one in the text, since a CRS names its authority
/// after its components.
pub fn epsg_from_wkt(wkt: &str) -> Option<u32> {
    let upper = wkt.to_ascii_uppercase();
    let mut best: Option<(usize, u32)> = None;
    for key in ["AUTHORITY[\"EPSG\",", "ID[\"EPSG\","] {
        if let Some(pos) = upper.rfind(key) {
            let rest = &upper[pos + key.len()..];
            let digits: String = rest.trim_start_matches(|c: char| c == '"' || c.is_whitespace()).chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(code) = digits.parse::<u32>() {
                if best.is_none_or(|(p, _)| pos > p) {
                    best = Some((pos, code));
                }
            }
        }
    }
    best.map(|(_, c)| c)
}

/// The CRS a LAS header declares: `EPSG:<code>` when its WKT or GeoTIFF keys
/// name an EPSG code, the WKT text when it does not, None without CRS records.
pub fn header_crs(header: &Header) -> Option<String> {
    if let Some(bytes) = header.get_wkt_crs_bytes() {
        let wkt = String::from_utf8_lossy(bytes).trim_end_matches('\0').trim().to_string();
        if !wkt.is_empty() {
            return Some(match epsg_from_wkt(&wkt) {
                Some(code) => format!("EPSG:{code}"),
                None => wkt,
            });
        }
    }
    if let Ok(Some(g)) = header.get_geotiff_crs() {
        let epsg = |k: Option<u16>| k.filter(|&k| (1024..=32766).contains(&k));
        if let Some(k) = epsg(g.get_projected_crs_geo_key_value()).or_else(|| epsg(g.get_geodetic_crs_geo_key_value())) {
            return Some(format!("EPSG:{k}"));
        }
        return Some("user-defined (GeoTIFF keys)".to_string());
    }
    None
}

fn read_header(path: &Path) -> Result<Header> {
    let f = File::open(path).map_err(|e| Error::file(path, e.to_string()))?;
    Header::new(BufReader::new(f)).map_err(|e| Error::file(path, e.to_string()))
}

/// The coordinate-system records of a file, to copy into files derived from it.
pub fn crs_vlrs(path: &Path) -> Result<Vec<Vlr>> {
    Ok(read_header(path)?.all_vlrs().filter(|v| v.is_crs()).cloned().collect())
}

/// Read one file's header into a [`Tile`].
pub fn read_tile(path: &Path) -> Result<Tile> {
    let header = read_header(path)?;
    let b = header.bounds();
    let t = header.transforms();
    let v = header.version();
    let spatial_index = path.with_extension("lax").exists()
        || path.with_extension("LAX").exists()
        || header.all_vlrs().any(|v| v.user_id.eq_ignore_ascii_case("copc"));
    Ok(Tile {
        path: path.to_path_buf(),
        bounds: [b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z],
        n_points: header.number_of_points(),
        point_format: header.point_format().to_u8().unwrap_or(255),
        version: (v.major, v.minor),
        crs: header_crs(&header),
        scale: [t.x.scale, t.y.scale, t.z.scale],
        offset: [t.x.offset, t.y.offset, t.z.offset],
        spatial_index,
        file_size: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
    })
}

/// A problem found by [`Catalog::issues`].
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    /// `missing`, `unreadable`, `empty`, `mixed_crs`, `no_crs`,
    /// `mixed_point_format`, `mixed_scale`, `overlap` or `gap`.
    pub kind: String,
    pub message: String,
}

/// Tiles of an airborne survey, known from their headers alone.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Catalog {
    pub tiles: Vec<Tile>,
    /// Paths that were asked for but do not exist.
    pub missing: Vec<PathBuf>,
    /// Paths whose header could not be read, with the reason.
    pub unreadable: Vec<(PathBuf, String)>,
}

/// Area of the intersection of two `[xmin, ymin, xmax, ymax]` boxes, with
/// its width and height (zero or negative when they do not meet).
fn intersection(a: &[f64; 4], b: &[f64; 4]) -> (f64, f64) {
    (a[2].min(b[2]) - a[0].max(b[0]), a[3].min(b[3]) - a[1].max(b[1]))
}

fn boxes_meet(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Is `(x, y)` in the half-open box `[xmin, xmax) x [ymin, ymax)`?
#[inline]
pub fn in_core(b: &[f64; 4], x: f64, y: f64) -> bool {
    x >= b[0] && x < b[2] && y >= b[1] && y < b[3]
}

/// Is `(x, y)` in the closed box?
#[inline]
fn in_box(b: &[f64; 4], x: f64, y: f64) -> bool {
    x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("1 {word}")
    } else {
        format!("{n} {word}s")
    }
}

impl Catalog {
    /// Read the headers of `paths` (in parallel; the tiles keep the order
    /// given). Files that do not exist or cannot be read are recorded, not
    /// raised, so that [`Catalog::issues`] can report them all at once.
    pub fn open(paths: &[PathBuf]) -> Catalog {
        let read: Vec<std::result::Result<Tile, (PathBuf, Option<String>)>> = paths
            .par_iter()
            .map(|p| {
                if !p.exists() {
                    return Err((p.clone(), None));
                }
                read_tile(p).map_err(|e| (p.clone(), Some(e.to_string())))
            })
            .collect();
        let mut cat = Catalog::default();
        for r in read {
            match r {
                Ok(t) => cat.tiles.push(t),
                Err((p, None)) => cat.missing.push(p),
                Err((p, Some(msg))) => cat.unreadable.push((p, msg)),
            }
        }
        cat
    }

    /// Number of readable tiles.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Points in all tiles, from the headers.
    pub fn n_points(&self) -> u64 {
        self.tiles.iter().map(|t| t.n_points).sum()
    }

    /// `[xmin, ymin, zmin, xmax, ymax, zmax]` over all tiles; None if there are none.
    pub fn bounds(&self) -> Option<[f64; 6]> {
        let first = self.tiles.first()?;
        let mut b = first.bounds;
        for t in &self.tiles[1..] {
            for k in 0..3 {
                b[k] = b[k].min(t.bounds[k]);
                b[k + 3] = b[k + 3].max(t.bounds[k + 3]);
            }
        }
        Some(b)
    }

    /// `[xmin, ymin, xmax, ymax]` over all tiles.
    pub fn xy_bounds(&self) -> Option<[f64; 4]> {
        self.bounds().map(|b| [b[0], b[1], b[3], b[4]])
    }

    /// The one CRS every tile declares, if they agree (None if they differ
    /// or declare none).
    pub fn crs(&self) -> Option<String> {
        let first = self.tiles.first()?.crs.clone()?;
        self.tiles.iter().all(|t| t.crs.as_deref() == Some(first.as_str())).then_some(first)
    }

    /// Refuse to process a catalogue with files missing or unreadable, or
    /// with no tiles at all.
    pub fn check_usable(&self) -> Result<()> {
        if let Some(p) = self.missing.first() {
            let n = self.missing.len();
            return Err(Error::invalid(format!("{} of the catalogue {} missing (first: {}); rebuild it from the files that exist", plural(n, "file"), if n == 1 { "is" } else { "are" }, p.display())));
        }
        if let Some((p, msg)) = self.unreadable.first() {
            return Err(Error::invalid(format!("{} could not be read (first: {}: {msg})", plural(self.unreadable.len(), "file"), p.display())));
        }
        if self.tiles.is_empty() {
            return Err(Error::invalid("the catalogue has no tiles"));
        }
        Ok(())
    }

    /// Pairs of tiles whose extents overlap by more than `tolerance` (m) in
    /// both x and y, as `(i, j, area)` with `i < j`.
    pub fn overlaps(&self, tolerance: f64) -> Vec<(usize, usize, f64)> {
        let mut order: Vec<usize> = (0..self.tiles.len()).collect();
        order.sort_by(|&a, &b| self.tiles[a].bounds[0].total_cmp(&self.tiles[b].bounds[0]).then(a.cmp(&b)));
        let mut out = Vec::new();
        for (k, &i) in order.iter().enumerate() {
            let a = self.tiles[i].xy();
            for &j in &order[k + 1..] {
                let b = self.tiles[j].xy();
                if b[0] > a[2] - tolerance {
                    break;
                }
                let (w, h) = intersection(&a, &b);
                if w > tolerance && h > tolerance {
                    out.push((i.min(j), i.max(j), w * h));
                }
            }
        }
        out.sort_by_key(|o| (o.0, o.1));
        out
    }

    /// Holes in the coverage: areas enclosed by tiles but inside none of them
    /// (tile extents grown by `tolerance`, so that the slivers between tiles
    /// whose points stop just short of their edges do not count). The area is
    /// sampled on a grid of at most 1000 x 1000 cells; each hole is returned
    /// as its bounding box `[xmin, ymin, xmax, ymax]` with its area. Concave
    /// outlines are not holes.
    pub fn gaps(&self, tolerance: f64) -> Vec<([f64; 4], f64)> {
        let Some(b) = self.xy_bounds() else { return Vec::new() };
        let (w, h) = (b[2] - b[0], b[3] - b[1]);
        let cell = (w.max(h) / 1000.0).max(tolerance.max(1e-9));
        let nx = ((w / cell).ceil() as usize).max(1);
        let ny = ((h / cell).ceil() as usize).max(1);
        let mut covered = vec![false; nx * ny];
        for t in &self.tiles {
            let tb = t.xy();
            let c0 = (((tb[0] - tolerance - b[0]) / cell - 0.5).ceil().max(0.0)) as usize;
            let c1 = (((tb[2] + tolerance - b[0]) / cell - 0.5).floor()).min(nx as f64 - 1.0);
            let r0 = (((tb[1] - tolerance - b[1]) / cell - 0.5).ceil().max(0.0)) as usize;
            let r1 = (((tb[3] + tolerance - b[1]) / cell - 0.5).floor()).min(ny as f64 - 1.0);
            if c1 < 0.0 || r1 < 0.0 {
                continue;
            }
            for r in r0..=r1 as usize {
                for c in c0..=c1 as usize {
                    covered[r * nx + c] = true;
                }
            }
        }
        // Uncovered cells reachable from the border are outside the survey.
        let mut outside = vec![false; nx * ny];
        let mut stack: Vec<usize> = (0..nx * ny).filter(|&i| !covered[i] && (i % nx == 0 || i % nx == nx - 1 || i / nx == 0 || i / nx == ny - 1)).collect();
        for &i in &stack {
            outside[i] = true;
        }
        let neighbours = |i: usize| {
            let (r, c) = (i / nx, i % nx);
            let mut v = Vec::with_capacity(4);
            if r > 0 {
                v.push(i - nx);
            }
            if r + 1 < ny {
                v.push(i + nx);
            }
            if c > 0 {
                v.push(i - 1);
            }
            if c + 1 < nx {
                v.push(i + 1);
            }
            v
        };
        while let Some(i) = stack.pop() {
            for j in neighbours(i) {
                if !covered[j] && !outside[j] {
                    outside[j] = true;
                    stack.push(j);
                }
            }
        }
        // The rest of the uncovered cells are holes; group them.
        let mut seen = vec![false; nx * ny];
        let mut holes = Vec::new();
        for start in 0..nx * ny {
            if covered[start] || outside[start] || seen[start] {
                continue;
            }
            seen[start] = true;
            let mut stack = vec![start];
            let (mut lo, mut hi, mut n) = ([usize::MAX; 2], [0usize; 2], 0usize);
            while let Some(i) = stack.pop() {
                n += 1;
                let (r, c) = (i / nx, i % nx);
                lo = [lo[0].min(c), lo[1].min(r)];
                hi = [hi[0].max(c), hi[1].max(r)];
                for j in neighbours(i) {
                    if !covered[j] && !outside[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            let bbox = [b[0] + lo[0] as f64 * cell, b[1] + lo[1] as f64 * cell, b[0] + (hi[0] + 1) as f64 * cell, b[1] + (hi[1] + 1) as f64 * cell];
            holes.push((bbox, n as f64 * cell * cell));
        }
        holes
    }

    /// Everything that could make processing go wrong or give a wrong
    /// answer: missing and unreadable files, empty tiles, tiles in
    /// different (or no) CRS, mixed point formats or scales, overlapping
    /// tiles and holes in the coverage. `tolerance` (m) is how far tile
    /// extents may overlap, or fall short of each other, before it counts.
    pub fn issues(&self, tolerance: f64) -> Vec<Issue> {
        let mut out = Vec::new();
        let mut push = |kind: &str, message: String| out.push(Issue { kind: kind.to_string(), message });
        for p in &self.missing {
            push("missing", format!("{} does not exist", p.display()));
        }
        for (p, msg) in &self.unreadable {
            push("unreadable", format!("{}: {msg}", p.display()));
        }
        let empty: Vec<&Tile> = self.tiles.iter().filter(|t| t.n_points == 0).collect();
        if !empty.is_empty() {
            push("empty", format!("{} with no points (first: {})", plural(empty.len(), "tile"), empty[0].path.display()));
        }
        let mut crs: BTreeMap<String, usize> = BTreeMap::new();
        for t in &self.tiles {
            *crs.entry(t.crs.clone().unwrap_or_else(|| "none".to_string())).or_default() += 1;
        }
        if crs.len() > 1 {
            let list: Vec<String> = crs.iter().map(|(k, n)| format!("{} ({n})", short_crs(k))).collect();
            push("mixed_crs", format!("tiles declare different coordinate systems: {}", list.join(", ")));
        } else if crs.contains_key("none") {
            push("no_crs", "no tile declares a coordinate system".to_string());
        }
        let mut formats: BTreeMap<u8, usize> = BTreeMap::new();
        for t in &self.tiles {
            *formats.entry(t.point_format).or_default() += 1;
        }
        if formats.len() > 1 {
            let list: Vec<String> = formats.iter().map(|(k, n)| format!("{k} ({n})")).collect();
            push("mixed_point_format", format!("tiles use different point formats: {}; attributes not in every format (gps_time, colour) are dropped where tiles meet", list.join(", ")));
        }
        let mut scales: Vec<[u64; 3]> = self.tiles.iter().map(|t| t.scale.map(f64::to_bits)).collect();
        scales.sort_unstable();
        scales.dedup();
        if scales.len() > 1 {
            push("mixed_scale", format!("tiles use {} different coordinate scales", scales.len()));
        }
        let overlaps = self.overlaps(tolerance);
        if !overlaps.is_empty() {
            let (i, j, _) = overlaps[0];
            let area: f64 = overlaps.iter().map(|o| o.2).sum();
            push("overlap", format!("{} of tiles overlap by more than {tolerance} m ({area:.1} m² in all; first: {} and {}); points in the overlaps are counted twice", plural(overlaps.len(), "pair"), self.tiles[i].path.display(), self.tiles[j].path.display()));
        }
        let gaps = self.gaps(tolerance);
        if !gaps.is_empty() {
            let area: f64 = gaps.iter().map(|g| g.1).sum();
            let b = gaps[0].0;
            push("gap", format!("{} in the coverage ({area:.1} m² in all; first near x {:.1}..{:.1}, y {:.1}..{:.1})", plural(gaps.len(), "hole"), b[0], b[2], b[1], b[3]));
        }
        out
    }

    /// A plain-text report: extent, counts, density, formats, CRS and issues.
    pub fn report(&self, tolerance: f64) -> String {
        let mut s = String::new();
        let n = self.n_points();
        s.push_str(&format!("ALS catalogue: {}, {} points\n", plural(self.tiles.len(), "tile"), group_thousands(n)));
        if let Some(b) = self.bounds() {
            let area: f64 = self.tiles.iter().map(|t| (t.bounds[3] - t.bounds[0]).max(0.0) * (t.bounds[4] - t.bounds[1]).max(0.0)).sum();
            s.push_str(&format!("  extent   x {:.2} .. {:.2}, y {:.2} .. {:.2}, z {:.2} .. {:.2}\n", b[0], b[3], b[1], b[4], b[2], b[5]));
            s.push_str(&format!("  area     {:.4} km² covered by tile extents ({:.1} x {:.1} m bounding box)\n", area / 1e6, b[3] - b[0], b[4] - b[1]));
            if area > 0.0 {
                s.push_str(&format!("  density  {:.2} points/m²\n", n as f64 / area));
            }
            let mut sizes: Vec<f64> = self.tiles.iter().map(|t| (t.bounds[3] - t.bounds[0]).max(t.bounds[4] - t.bounds[1])).collect();
            sizes.sort_by(f64::total_cmp);
            s.push_str(&format!("  tiles    {:.1} m across (median), {} to {} points\n", sizes[sizes.len() / 2], group_thousands(self.tiles.iter().map(|t| t.n_points).min().unwrap_or(0)), group_thousands(self.tiles.iter().map(|t| t.n_points).max().unwrap_or(0))));
        }
        let mut formats: BTreeMap<String, usize> = BTreeMap::new();
        for t in &self.tiles {
            *formats.entry(format!("LAS {}.{} format {}", t.version.0, t.version.1, t.point_format)).or_default() += 1;
        }
        let list: Vec<String> = formats.iter().map(|(k, v)| format!("{k} ({v})")).collect();
        if !list.is_empty() {
            s.push_str(&format!("  format   {}\n", list.join(", ")));
        }
        let indexed = self.tiles.iter().filter(|t| t.spatial_index).count();
        s.push_str(&format!("  index    {} of {} with a spatial index\n", indexed, self.tiles.len()));
        s.push_str(&format!("  crs      {}\n", match (self.crs(), self.tiles.iter().any(|t| t.crs.is_some())) {
            (Some(c), _) => short_crs(&c),
            (None, true) => "mixed".to_string(),
            (None, false) => "none declared".to_string(),
        }));
        let issues = self.issues(tolerance);
        if issues.is_empty() {
            s.push_str("  checks   no problems found\n");
        } else {
            s.push_str(&format!("  checks   {}:\n", plural(issues.len(), "problem")));
            for i in &issues {
                s.push_str(&format!("    [{}] {}\n", i.kind, i.message));
            }
        }
        s
    }
}

fn short_crs(c: &str) -> String {
    if c.len() > 60 {
        format!("{}...", &c[..c.char_indices().nth(57).map(|(i, _)| i).unwrap_or(c.len())])
    } else {
        c.to_string()
    }
}

fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

// ------------------------------------------------------------------ chunks

/// How the area is divided into chunks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Layout {
    /// One chunk per tile. The core is the tile itself: its own points are
    /// core points and every other file's are buffer points.
    Tiles,
    /// A regular grid of `size` m squares anchored at `origin` (by default
    /// the catalogue's minimum snapped down to a multiple of `size`). A
    /// point is core in the chunk whose half-open square holds it.
    Grid { size: f64, origin: Option<(f64, f64)> },
}

/// A piece of the catalogue to process.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub index: usize,
    /// `[xmin, ymin, xmax, ymax]` of the area the chunk is responsible for.
    pub core: [f64; 4],
    /// The core grown by the buffer: every point in it is read.
    pub outer: [f64; 4],
    /// For [`Layout::Tiles`], the tile whose points are the core.
    pub own: Option<usize>,
    /// Tiles overlapping `outer`, in catalogue order.
    pub files: Vec<usize>,
    /// Points expected in `outer`, from the header counts and the share of
    /// each tile's extent that falls in it.
    pub est_points: u64,
    /// A file name for the chunk's outputs (no extension): the tile's own
    /// stem, or `<xmin>_<ymin>` of a grid chunk.
    pub name: String,
}

/// A coordinate as a file name part: integral values without decimals.
fn coord_name(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.3}").trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn est_points(cat: &Catalog, outer: &[f64; 4], files: &[usize]) -> u64 {
    files
        .iter()
        .map(|&i| {
            let t = &cat.tiles[i];
            let tb = t.xy();
            let area = (tb[2] - tb[0]) * (tb[3] - tb[1]);
            let (w, h) = intersection(&tb, outer);
            if area <= 0.0 {
                t.n_points
            } else {
                ((w.max(0.0) * h.max(0.0) / area).min(1.0) * t.n_points as f64).ceil() as u64
            }
        })
        .sum()
}

/// Divide the catalogue into chunks with a `buffer` (m) around each.
/// Grid chunks that no tile reaches are left out; chunk indices count the
/// chunks kept, west to east within south-to-north rows.
pub fn plan(cat: &Catalog, layout: Layout, buffer: f64) -> Result<Vec<Chunk>> {
    cat.check_usable()?;
    if !(buffer.is_finite() && buffer >= 0.0) {
        return Err(Error::invalid(format!("buffer must be a non-negative number of metres, got {buffer}")));
    }
    let grow = |c: &[f64; 4]| [c[0] - buffer, c[1] - buffer, c[2] + buffer, c[3] + buffer];
    let files_in = |outer: &[f64; 4]| -> Vec<usize> { (0..cat.tiles.len()).filter(|&i| cat.tiles[i].n_points > 0 && boxes_meet(&cat.tiles[i].xy(), outer)).collect() };
    let mut chunks = Vec::new();
    match layout {
        Layout::Tiles => {
            for (i, t) in cat.tiles.iter().enumerate() {
                if t.n_points == 0 {
                    continue;
                }
                let core = t.xy();
                let outer = grow(&core);
                let files = files_in(&outer);
                let name = t.path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| format!("tile_{i}"));
                chunks.push(Chunk { index: chunks.len(), core, outer, own: Some(i), est_points: est_points(cat, &outer, &files), files, name });
            }
        }
        Layout::Grid { size, origin } => {
            if !(size.is_finite() && size > 0.0) {
                return Err(Error::invalid(format!("chunk size must be a positive number of metres, got {size}")));
            }
            let b = cat.xy_bounds().expect("checked non-empty");
            let (ox, oy) = origin.unwrap_or(((b[0] / size).floor() * size, (b[1] / size).floor() * size));
            if ox > b[0] || oy > b[1] {
                return Err(Error::invalid(format!("the chunk origin ({ox}, {oy}) is east or north of the catalogue's south-west corner ({}, {})", b[0], b[1])));
            }
            let nx = ((b[2] - ox) / size).floor() as usize + 1;
            let ny = ((b[3] - oy) / size).floor() as usize + 1;
            limits::check_cells(nx as u128 * ny as u128, 200, &format!("a grid of {nx} x {ny} chunks"), "a larger chunk_size")?;
            for r in 0..ny {
                for c in 0..nx {
                    let core = [ox + c as f64 * size, oy + r as f64 * size, ox + (c + 1) as f64 * size, oy + (r + 1) as f64 * size];
                    // Only chunks whose core some tile reaches have work to do.
                    if !cat.tiles.iter().any(|t| t.n_points > 0 && { let tb = t.xy(); tb[0] < core[2] && core[0] <= tb[2] && tb[1] < core[3] && core[1] <= tb[3] }) {
                        continue;
                    }
                    let outer = grow(&core);
                    let files = files_in(&outer);
                    let name = format!("{}_{}", coord_name(core[0]), coord_name(core[1]));
                    chunks.push(Chunk { index: chunks.len(), core, outer, own: None, est_points: est_points(cat, &outer, &files), files, name });
                }
            }
        }
    }
    Ok(chunks)
}

/// The points of a chunk: `cloud` holds everything in the buffered box and
/// `buffer[i]` is true for the points outside the core.
#[derive(Debug, Clone, Default)]
pub struct ChunkData {
    pub cloud: PointCloud,
    pub buffer: Vec<bool>,
}

impl ChunkData {
    /// Number of core points.
    pub fn n_core(&self) -> usize {
        self.buffer.iter().filter(|&&b| !b).count()
    }

    /// Indices of the core points.
    pub fn core_indices(&self) -> Vec<usize> {
        (0..self.buffer.len()).filter(|&i| !self.buffer[i]).collect()
    }
}

/// Stack clouds, keeping the attributes that every part has with the same
/// type (a column that differs in type between files is dropped).
pub fn merge_clouds(parts: Vec<PointCloud>) -> PointCloud {
    let parts: Vec<PointCloud> = parts.into_iter().collect();
    let Some(first) = parts.first() else { return PointCloud::default() };
    let names: Vec<String> = first.attrs.iter().filter(|(k, a)| parts.iter().all(|p| p.attrs.get(*k).is_some_and(|b| b.dtype() == a.dtype()))).map(|(k, _)| k.clone()).collect();
    let mut out = PointCloud::new(Vec::with_capacity(parts.iter().map(|p| p.len()).sum()));
    for p in &parts {
        out.xyz.extend_from_slice(&p.xyz);
    }
    for name in names {
        let mut col = first.attrs[&name].clone();
        for p in &parts[1..] {
            col.extend(&p.attrs[&name]).expect("types checked");
        }
        out.attrs.insert(name, col);
    }
    out
}

/// Read a chunk: every file overlapping its buffered box, streamed, keeping
/// the points inside the box (a file wholly inside is read without testing).
pub fn read_chunk(cat: &Catalog, chunk: &Chunk) -> Result<ChunkData> {
    let mut parts = Vec::with_capacity(chunk.files.len());
    let mut buffer = Vec::new();
    for &i in &chunk.files {
        let t = cat.tiles.get(i).ok_or_else(|| Error::invalid(format!("chunk {} names tile {i}, which the catalogue does not have", chunk.index)))?;
        let tb = t.xy();
        let whole = chunk.own == Some(i) || (in_box(&chunk.outer, tb[0], tb[1]) && in_box(&chunk.outer, tb[2], tb[3]));
        let o = chunk.outer;
        let c = if whole { read_las(&t.path) } else { read_las_where(&t.path, |p: &Point| in_box(&o, p[0], p[1])) }.map_err(|e| match e {
            Error::File { .. } => e,
            other => Error::file(&t.path, other.to_string()),
        })?;
        match chunk.own {
            Some(own) => buffer.extend(std::iter::repeat_n(own != i, c.len())),
            None => buffer.extend(c.xyz.iter().map(|p| !in_core(&chunk.core, p[0], p[1]))),
        }
        parts.push(c);
    }
    Ok(ChunkData { cloud: merge_clouds(parts), buffer })
}

/// Read every point of the catalogue inside `bounds` (`[xmin, ymin, xmax, ymax]`, closed).
pub fn read_region(cat: &Catalog, bounds: [f64; 4]) -> Result<PointCloud> {
    cat.check_usable()?;
    let files: Vec<usize> = (0..cat.tiles.len()).filter(|&i| boxes_meet(&cat.tiles[i].xy(), &bounds)).collect();
    let chunk = Chunk { index: 0, core: bounds, outer: bounds, own: None, est_points: est_points(cat, &bounds, &files), files, name: String::new() };
    Ok(read_chunk(cat, &chunk)?.cloud)
}

// ------------------------------------------------------------------ running

/// Default bytes held per point while a chunk is processed: the cloud
/// (coordinates and the LAS attributes, about 60 bytes) plus room for the
/// working copies, neighbour searches and grids of a typical step.
pub const BYTES_PER_POINT: u64 = 256;

/// How many chunks may be in memory at once: `workers` (all cores when 0),
/// fewer if the largest chunk times that many would exceed the memory
/// budget. Refuses outright when a single chunk will not fit.
pub fn workers_for(chunks: &[Chunk], workers: usize, bytes_per_point: u64) -> Result<usize> {
    workers_for_estimates(&chunks.iter().map(|c| c.est_points).collect::<Vec<_>>(), workers, bytes_per_point)
}

/// [`workers_for`] from the chunks' estimated point counts alone.
pub fn workers_for_estimates(est_points: &[u64], workers: usize, bytes_per_point: u64) -> Result<usize> {
    let largest = est_points.iter().copied().max().unwrap_or(0);
    let need = largest.saturating_mul(bytes_per_point);
    limits::check(need, &format!("a chunk of about {} points", group_thousands(largest)), "a smaller chunk_size or buffer")?;
    let w = if workers == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { workers };
    let fit = match limits::budget() {
        Some(b) if need > 0 => (b / need).max(1) as usize,
        _ => usize::MAX,
    };
    Ok(w.min(fit).min(est_points.len().max(1)).max(1))
}

/// Run `f` on every chunk that has core points, on `workers` threads (see
/// [`workers_for`]), and return what it gave in chunk order (None for
/// chunks with no core points, which `f` never sees). Each worker takes the
/// next chunk, reads it and processes it before taking another, so at most
/// `workers` chunks are in memory at a time. The first error stops the
/// workers taking new chunks and is returned.
pub fn run<T: Send>(cat: &Catalog, chunks: &[Chunk], workers: usize, label: &str, f: impl Fn(&Chunk, ChunkData) -> Result<T> + Sync) -> Result<Vec<Option<T>>> {
    let task = progress::start(label, chunks.len() as u64);
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..chunks.len()).map(|_| None).collect());
    let errors: Mutex<Vec<(usize, Error)>> = Mutex::new(Vec::new());
    let work = || loop {
        if failed.load(Ordering::Relaxed) {
            break;
        }
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= chunks.len() {
            break;
        }
        let out = read_chunk(cat, &chunks[i]).and_then(|data| if data.n_core() == 0 { Ok(None) } else { f(&chunks[i], data).map(Some) });
        match out {
            Ok(v) => results.lock().expect("results lock")[i] = v,
            Err(e) => {
                failed.store(true, Ordering::Relaxed);
                errors.lock().expect("errors lock").push((i, e));
            }
        }
        task.inc(1);
    };
    let n = workers.max(1).min(chunks.len().max(1));
    if n == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..n {
                s.spawn(work);
            }
        });
    }
    let mut errors = errors.into_inner().expect("errors lock");
    errors.sort_by_key(|e| e.0);
    if let Some((i, e)) = errors.into_iter().next() {
        return Err(match e {
            Error::Invalid(m) => Error::Invalid(format!("chunk {} ({}): {m}", i, chunks[i].name)),
            other => other,
        });
    }
    Ok(results.into_inner().expect("results lock"))
}

// ------------------------------------------------------------------ rasters

/// The grid every raster output of a catalogue shares: south-west corner at
/// the catalogue's minimum snapped down to a multiple of `resolution`
/// (as [`Raster::from_points`] snaps it), covering its maximum.
pub fn catalog_grid(cat: &Catalog, resolution: f64) -> Result<Raster> {
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(Error::invalid(format!("resolution must be a positive number, got {resolution}")));
    }
    let b = cat.xy_bounds().ok_or_else(|| Error::invalid("the catalogue has no tiles"))?;
    let xmin = (b[0] / resolution).floor() * resolution;
    let ymin = (b[1] / resolution).floor() * resolution;
    let ncols = ((b[2] - xmin) / resolution).floor() as usize + 1;
    let nrows = ((b[3] - ymin) / resolution).floor() as usize + 1;
    limits::check_cells(nrows as u128 * ncols as u128, 17, &format!("a {nrows} x {ncols} raster at {resolution} m"), "a coarser resolution")?;
    Ok(Raster::filled(nrows, ncols, xmin, ymin, resolution, f64::NAN))
}

/// Bounds `(xmin, ymin, xmax, ymax)` to pass to [`Raster::from_points`] for
/// the cells of `grid` that cover `outer`, so that a chunk's raster lies on
/// the catalogue grid.
pub fn chunk_bounds(grid: &Raster, outer: &[f64; 4]) -> (f64, f64, f64, f64) {
    let res = grid.resolution;
    let c0 = ((outer[0] - grid.xmin) / res).floor();
    let r0 = ((outer[1] - grid.ymin) / res).floor();
    let c1 = ((outer[2] - grid.xmin) / res).floor() + 1.0;
    let r1 = ((outer[3] - grid.ymin) / res).floor() + 1.0;
    let xmin = grid.xmin + c0 * res;
    let ymin = grid.ymin + r0 * res;
    // from_points makes floor((xmax - xmin) / res) + 1 columns.
    (xmin, ymin, xmin + (c1 - c0 - 0.5) * res, ymin + (r1 - r0 - 0.5) * res)
}

fn dist_to_box(b: &[f64; 4], x: f64, y: f64) -> f64 {
    let dx = (b[0] - x).max(0.0).max(x - b[2]);
    let dy = (b[1] - y).max(0.0).max(y - b[3]);
    if in_core(b, x, y) {
        0.0
    } else {
        (dx * dx + dy * dy).sqrt().max(f64::MIN_POSITIVE)
    }
}

/// Join per-chunk rasters onto `grid` (from [`catalog_grid`]). Each cell of
/// the result comes from the chunk whose core is nearest its centre
/// (zero inside a core, which is half-open, so a cell on a shared edge
/// belongs to exactly one chunk); ties go to the lower chunk index. Cells
/// in no chunk's raster stay NaN. The rasters must share the grid's
/// resolution and alignment.
pub fn mosaic(grid: &Raster, parts: &[(Raster, [f64; 4])]) -> Result<Raster> {
    let res = grid.resolution;
    let mut out = grid.clone();
    out.data.iter_mut().for_each(|v| *v = f64::NAN);
    let mut best = vec![f64::INFINITY; out.data.len()];
    for (k, (r, core)) in parts.iter().enumerate() {
        if (r.resolution - res).abs() > 1e-9 * res {
            return Err(Error::invalid(format!("raster {k} has resolution {}, expected {res} like the others", r.resolution)));
        }
        let fc = (r.xmin - grid.xmin) / res;
        let fr = (r.ymin - grid.ymin) / res;
        if (fc - fc.round()).abs() > 1e-6 || (fr - fr.round()).abs() > 1e-6 {
            return Err(Error::invalid(format!("raster {k} is not aligned with the catalogue grid: its corner ({}, {}) is not a whole number of {res} m cells from ({}, {})", r.xmin, r.ymin, grid.xmin, grid.ymin)));
        }
        let (dc, dr) = (fc.round() as i64, fr.round() as i64);
        for row in 0..r.nrows {
            let gr = dr + row as i64;
            if gr < 0 || gr >= out.nrows as i64 {
                continue;
            }
            for col in 0..r.ncols {
                let gc = dc + col as i64;
                if gc < 0 || gc >= out.ncols as i64 {
                    continue;
                }
                let (x, y) = out.cell_center(gr as usize, gc as usize);
                let d = dist_to_box(core, x, y);
                let i = gr as usize * out.ncols + gc as usize;
                if d < best[i] {
                    best[i] = d;
                    out.data[i] = r.get(row, col);
                }
            }
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------ writing

/// A point format the writer can fill: waveform and NIR formats become the
/// nearest format without them (4 -> 1, 5 -> 3, 8 -> 7, 9 -> 6, 10 -> 7).
pub fn writable_format(format: u8) -> u8 {
    match format {
        4 => 1,
        5 => 3,
        8 | 10 => 7,
        9 => 6,
        f if f <= 10 => f,
        _ => 6,
    }
}

/// Write `cloud` as LAS/LAZ (by extension) in the point format and scale of
/// `like`, copying its coordinate-system records.
pub fn write_like(cloud: &PointCloud, path: &Path, like: &Tile) -> Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    // The struct update keeps this building as the options gain fields.
    #[allow(clippy::needless_update)]
    let opts = LasWriteOptions { point_format: writable_format(like.point_format), scale: like.scale[0], ..Default::default() };
    write_las_with_vlrs(cloud, path, &opts, &crs_vlrs(&like.path)?)
}

/// Where a per-chunk output goes: `<out_dir>/<name>.<ext>`, refusing to
/// overwrite any file of the catalogue.
pub fn output_path(cat: &Catalog, out_dir: &Path, name: &str, ext: &str) -> Result<PathBuf> {
    let path = out_dir.join(format!("{name}.{ext}"));
    let canon = |p: &Path| std::fs::canonicalize(p).ok();
    if let Some(c) = canon(&path) {
        if cat.tiles.iter().any(|t| canon(&t.path).as_ref() == Some(&c)) {
            return Err(Error::invalid(format!("{} is a file of the catalogue; write the output somewhere else", path.display())));
        }
    }
    Ok(path)
}

/// The cloud without the named attribute.
pub fn without(mut cloud: PointCloud, name: &str) -> PointCloud {
    cloud.attrs.remove(name);
    cloud
}

/// A `u8` column flagging buffer points, as written by retiling.
pub fn buffer_attr(buffer: &[bool]) -> Attr {
    Attr::U8(buffer.iter().map(|&b| b as u8).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(path: &str, b: [f64; 4], n: u64, crs: Option<&str>, format: u8) -> Tile {
        Tile { path: PathBuf::from(path), bounds: [b[0], b[1], 0.0, b[2], b[3], 10.0], n_points: n, point_format: format, version: (1, 4), crs: crs.map(String::from), scale: [0.001; 3], offset: [0.0; 3], spatial_index: false, file_size: 0 }
    }

    fn grid2x2() -> Catalog {
        let mut cat = Catalog::default();
        for (i, (x, y)) in [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)].iter().enumerate() {
            cat.tiles.push(tile(&format!("t{i}.laz"), [*x, *y, x + 99.99, y + 99.99], 1000, Some("EPSG:28355"), 6));
        }
        cat
    }

    #[test]
    fn epsg_is_read_from_the_outermost_authority() {
        let wkt = r#"PROJCS["GDA94 / MGA zone 55",GEOGCS["GDA94",AUTHORITY["EPSG","4283"]],UNIT["metre",1],AUTHORITY["EPSG","28355"]]"#;
        assert_eq!(epsg_from_wkt(wkt), Some(28355));
        assert_eq!(epsg_from_wkt(r#"PROJCRS["x",BASEGEOGCRS["y",ID["EPSG",4326]],ID["EPSG",32617]]"#), Some(32617));
        assert_eq!(epsg_from_wkt("LOCAL_CS[\"x\"]"), None);
    }

    #[test]
    fn a_regular_grid_has_no_issues() {
        let cat = grid2x2();
        assert!(cat.issues(1.0).is_empty(), "{:?}", cat.issues(1.0));
        assert_eq!(cat.crs().as_deref(), Some("EPSG:28355"));
        assert!(cat.report(1.0).contains("no problems found"));
    }

    #[test]
    fn overlaps_gaps_and_mixtures_are_found() {
        let mut cat = grid2x2();
        cat.tiles[3].crs = Some("EPSG:28356".into());
        cat.tiles[2].point_format = 1;
        cat.tiles.push(tile("over.laz", [50.0, 50.0, 150.0, 60.0], 10, Some("EPSG:28355"), 6));
        let kinds: Vec<String> = cat.issues(1.0).into_iter().map(|i| i.kind).collect();
        assert_eq!(kinds, ["mixed_crs", "mixed_point_format", "overlap"]);
        assert_eq!(cat.crs(), None);

        // A 3 x 3 layout with the middle tile missing is a hole; a missing
        // corner is only a concave outline.
        let mut ring = Catalog::default();
        for r in 0..3 {
            for c in 0..3 {
                if (r, c) != (1, 1) && (r, c) != (2, 2) {
                    let (x, y) = (c as f64 * 100.0, r as f64 * 100.0);
                    ring.tiles.push(tile("t.laz", [x, y, x + 99.99, y + 99.99], 5, None, 6));
                }
            }
        }
        let gaps = ring.gaps(1.0);
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert!((gaps[0].1 - 100.0 * 100.0).abs() < 0.05 * 1e4, "{}", gaps[0].1);
        let kinds: Vec<String> = ring.issues(1.0).into_iter().map(|i| i.kind).collect();
        assert_eq!(kinds, ["no_crs", "gap"]);
    }

    #[test]
    fn missing_files_stop_processing() {
        let cat = Catalog::open(&[PathBuf::from("/nonexistent/a.laz")]);
        assert_eq!(cat.missing.len(), 1);
        assert_eq!(cat.issues(1.0)[0].kind, "missing");
        assert!(plan(&cat, Layout::Tiles, 10.0).unwrap_err().to_string().contains("missing"));
    }

    #[test]
    fn chunks_cover_the_area_once() {
        let cat = grid2x2();
        let tiles = plan(&cat, Layout::Tiles, 20.0).unwrap();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].files, vec![0, 1, 2, 3]);
        assert_eq!(tiles[0].own, Some(0));
        // 1000 own points plus a 20 m strip of each neighbour (a 20 x 20 corner of the diagonal one).
        assert!((tiles[0].est_points as f64 - (1000.0 + 2.0 * 200.0 + 40.0)).abs() < 5.0, "{}", tiles[0].est_points);
        let grid = plan(&cat, Layout::Grid { size: 50.0, origin: None }, 5.0).unwrap();
        assert_eq!(grid.len(), 16);
        assert_eq!(grid[5].core, [50.0, 50.0, 100.0, 100.0]);
        assert_eq!(grid[5].name, "50_50");
        assert_eq!(grid[0].files, vec![0]);
        assert_eq!(grid[5].files, vec![0, 1, 2, 3]);
        assert!(plan(&cat, Layout::Grid { size: 0.0, origin: None }, 5.0).is_err());
        assert!(plan(&cat, Layout::Tiles, -1.0).is_err());
    }

    #[test]
    fn mosaic_takes_each_cell_from_its_core() {
        let cat = grid2x2();
        let grid = catalog_grid(&cat, 10.0).unwrap();
        assert_eq!((grid.nrows, grid.ncols), (20, 20));
        let chunks = plan(&cat, Layout::Tiles, 20.0).unwrap();
        let parts: Vec<(Raster, [f64; 4])> = chunks
            .iter()
            .map(|c| {
                let b = chunk_bounds(&grid, &c.outer);
                let r = Raster::from_points(std::iter::empty(), std::iter::empty(), 10.0, crate::raster::Reducer::Min, Some(b), c.index as f64).unwrap();
                assert!((r.xmin - grid.xmin) % 10.0 == 0.0);
                (r, c.core)
            })
            .collect();
        let m = mosaic(&grid, &parts).unwrap();
        assert_eq!(m.get(0, 0), 0.0);
        assert_eq!(m.get(0, 19), 1.0);
        assert_eq!(m.get(19, 0), 2.0);
        assert_eq!(m.get(19, 19), 3.0);
        assert_eq!(m.get(9, 9), 0.0);
        assert_eq!(m.get(10, 10), 3.0);
        assert!(m.data.iter().all(|v| v.is_finite()));
        let mut bad = parts[0].0.clone();
        bad.xmin += 3.0;
        assert!(mosaic(&grid, &[(bad, parts[0].1)]).is_err());
    }

    #[test]
    fn workers_shrink_to_the_budget() {
        let cat = grid2x2();
        let chunks = plan(&cat, Layout::Tiles, 20.0).unwrap();
        let need = chunks.iter().map(|c| c.est_points).max().unwrap() * 100;
        limits::set_budget(3 * need + 1);
        let w = workers_for(&chunks, 8, 100);
        limits::set_budget(1000);
        let refused = workers_for(&chunks, 8, 100);
        limits::set_budget(0);
        assert_eq!(w.unwrap(), 3);
        assert!(refused.unwrap_err().to_string().contains("smaller chunk_size"));
    }
}
