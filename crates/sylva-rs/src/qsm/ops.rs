// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Operations on a QSM's cylinder table, and a buttress joined to it.
//!
//! A model is its `(n, 12)` table of rows (start, axis, length, radius,
//! parent, branch order, branch id, points), as the Python package's `QSM`
//! holds it; everything here reads and writes those rows as they are, so a
//! value that is not a whole number survives a cut unchanged. Sums are
//! NumPy's pairwise sums, and the cuts, cross-sections and the join fit
//! follow the NumPy code they replace operation for operation.
//!
//! [`fuse`] joins a buttress mesh to the cylinders above it
//! ([`TreeMesh`]); the model and joined meshes are written as OBJ
//! ([`crate::io::mesh`]) or binary PLY, coloured by branch order.

use std::collections::BTreeSet;
use std::path::Path;

use crate::leaves::model::qsm_from_rows;
use crate::io::mesh::{self, ObjMesh};
use crate::util::numeric::{arange, pairwise_sum};
use crate::{Error, Point, Result};

/// One cylinder: `sx, sy, sz, ax, ay, az, length, radius, parent,
/// branch_order, branch_id, n_points`.
pub type Row = [f64; 12];

/// A segment of a cross-section, two `(x, y)` ends.
pub type Segment = [[f64; 2]; 2];

/// Face colours by branch order, brown stem to green twigs; orders past the
/// last take the last.
pub const ORDER_COLORS: [[u8; 3]; 6] = [[139, 90, 43], [205, 133, 63], [222, 184, 135], [60, 179, 113], [46, 139, 87], [34, 139, 34]];

/// Face colour of a buttress, darker than the stem.
pub const BUTTRESS_COLOR: [u8; 3] = [101, 67, 33];

/// `x.astype(int)` in NumPy on x86-64: truncated toward zero, NaN and values
/// out of range the smallest integer.
fn np_int(x: f64) -> i64 {
    if x.is_nan() || x >= 9.223_372_036_854_776e18 || x < -9.223_372_036_854_776e18 {
        i64::MIN
    } else {
        x as i64
    }
}

/// Cylinder end points, `start + axis * length`.
pub fn ends(rows: &[Row]) -> Vec<Point> {
    rows.iter().map(|r| [r[0] + r[3] * r[6], r[1] + r[4] * r[6], r[2] + r[5] * r[6]]).collect()
}

/// Volume of each cylinder, `pi r^2 length` (m³).
pub fn volumes(rows: &[Row]) -> Vec<f64> {
    rows.iter().map(|r| std::f64::consts::PI * (r[7] * r[7]) * r[6]).collect()
}

/// Totals of a model, as the `QSM` properties give them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Totals {
    pub total_volume: f64,
    pub stem_volume: f64,
    pub branch_volume: f64,
    pub total_length: f64,
    /// Highest branch order (NaN if any order is NaN; 0 for no rows).
    pub max_branch_order: f64,
}

/// Volumes, lengths and the highest branch order of a model.
pub fn totals(rows: &[Row]) -> Totals {
    let v = volumes(rows);
    let total_volume = pairwise_sum(&v);
    let stem: Vec<f64> = rows.iter().zip(&v).filter(|(r, _)| r[9] == 0.0).map(|(_, &x)| x).collect();
    let stem_volume = pairwise_sum(&stem);
    let lengths: Vec<f64> = rows.iter().map(|r| r[6]).collect();
    let max_branch_order = if rows.is_empty() {
        0.0
    } else if rows.iter().any(|r| r[9].is_nan()) {
        f64::NAN
    } else {
        rows.iter().map(|r| r[9]).fold(f64::NEG_INFINITY, f64::max)
    };
    Totals { total_volume, stem_volume, branch_volume: total_volume - stem_volume, total_length: pairwise_sum(&lengths), max_branch_order }
}

/// Cylinder volume above the plane `z`, a cylinder crossing it counted by
/// the share of its axis above it (m³).
pub fn volume_above(rows: &[Row], z: f64) -> f64 {
    let e = ends(rows);
    let v = volumes(rows);
    let parts: Vec<f64> = rows
        .iter()
        .zip(&e)
        .zip(&v)
        .map(|((r, e), &vol)| {
            let (lo, hi) = (r[2].min(e[2]), r[2].max(e[2]));
            let lo = if r[2].is_nan() || e[2].is_nan() { f64::NAN } else { lo };
            let hi = if r[2].is_nan() || e[2].is_nan() { f64::NAN } else { hi };
            let span = (hi - lo).max(1e-12);
            let share = if hi <= z {
                0.0
            } else if lo >= z {
                1.0
            } else {
                (hi - z) / span
            };
            vol * share
        })
        .collect();
    pairwise_sum(&parts)
}

/// The model above the plane `z`: a cylinder crossing it keeps the part
/// above (cut where its axis crosses the plane), one wholly below is
/// dropped, and parents are renumbered, a child whose parent went becoming
/// a branch base (-1). A parent past the last row counts as the last row.
pub fn above(rows: &[Row], z: f64) -> Vec<Row> {
    let e = ends(rows);
    let n = rows.len();
    let mut idx = vec![-1i64; n];
    let mut out: Vec<Row> = Vec::new();
    for (i, (r, e)) in rows.iter().zip(&e).enumerate() {
        if r[2].is_nan() || e[2].is_nan() || r[2].max(e[2]).partial_cmp(&z) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        let mut row = *r;
        let len = r[6];
        let dz = r[5] * len;
        let steep = dz.abs() > 1e-12;
        if r[2] < z && steep {
            // Starts below, so trim the base.
            let t = ((z - r[2]) / dz).clamp(0.0, 1.0);
            let tl = t * len;
            row[0] = r[0] + r[3] * tl;
            row[1] = r[1] + r[4] * tl;
            row[2] = r[2] + r[5] * tl;
            row[6] = len * (1.0 - t);
        }
        idx[i] = out.len() as i64;
        out.push(row);
    }
    for row in &mut out {
        let par = np_int(row[8]);
        row[8] = if par >= 0 && n > 0 { idx[(par as usize).min(n - 1)] as f64 } else { -1.0 };
    }
    out
}

/// Cylinder table from CSV text: the first line is a header, blank lines
/// and `#` comments are skipped, and the values (all rows as long as the
/// first) must make whole rows of twelve.
pub fn parse_csv(text: &str) -> Result<Vec<Row>> {
    let mut vals: Vec<f64> = Vec::new();
    let mut width: Option<usize> = None;
    for (no, line) in text.lines().enumerate().skip(1) {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(|s| {
                let s = s.trim();
                s.parse::<f64>().map_err(|_| Error::invalid(format!("could not convert string to float: '{s}'")))
            })
            .collect::<Result<Vec<f64>>>()?;
        if width.is_some_and(|w| w != row.len()) {
            return Err(Error::invalid(format!("the number of columns changed from {} to {} at row {}", width.unwrap_or(0), row.len(), no + 1)));
        }
        width = Some(row.len());
        vals.extend(row);
    }
    if !vals.len().is_multiple_of(12) {
        return Err(Error::invalid(format!("cannot reshape array of size {} into shape (12,)", vals.len())));
    }
    Ok(vals.as_chunks::<12>().0.to_vec())
}

/// Read cylinders written by `Qsm::write_csv` ([`parse_csv`]).
pub fn read_csv(path: impl AsRef<Path>) -> Result<Vec<Row>> {
    parse_csv(&std::fs::read_to_string(path)?)
}

/// Triangle mesh of the cylinders: `(vertices, faces, cylinder of each
/// face)`, one closed tube per cylinder or, `contiguous`, per branch.
pub fn mesh(rows: &[Row], sides: usize, contiguous: bool) -> (Vec<Point>, Vec<[u32; 3]>, Vec<u32>) {
    let q = qsm_from_rows(rows);
    if contiguous {
        q.mesh_contiguous(sides)
    } else {
        q.mesh(sides)
    }
}

fn index_color(order: i64) -> Result<[u8; 3]> {
    let n = ORDER_COLORS.len() as i64;
    let k = order.min(n - 1);
    if k < -n {
        return Err(Error::invalid(format!("index {k} is out of bounds for axis 0 with size {n}")));
    }
    let k = if k < 0 { k + n } else { k };
    Ok(ORDER_COLORS[k as usize])
}

/// Colour of each face by the branch order of the cylinder it belongs to.
pub fn order_colors(rows: &[Row], owner: &[u32]) -> Result<Vec<[u8; 3]>> {
    owner.iter().map(|&o| index_color(np_int(rows[o as usize][9]))).collect()
}

fn signed_faces(faces: &[[u32; 3]]) -> Vec<[i32; 3]> {
    faces.iter().map(|f| [f[0] as i32, f[1] as i32, f[2] as i32]).collect()
}

/// Write the cylinder mesh as OBJ, one object `tree_1`.
pub fn write_model_obj(path: impl AsRef<Path>, rows: &[Row], sides: usize, contiguous: bool) -> Result<()> {
    let (v, f, _) = mesh(rows, sides, contiguous);
    mesh::write_obj(path, &[ObjMesh { name: "tree_1", vertices: &v, faces: &f }])
}

/// Write the cylinder mesh as binary PLY, faces in `color` or by branch order.
pub fn write_model_ply(path: impl AsRef<Path>, rows: &[Row], sides: usize, color: Option<[u8; 3]>, contiguous: bool) -> Result<()> {
    let (v, f, owner) = mesh(rows, sides, contiguous);
    let rgb = match color {
        Some(c) => vec![c; f.len()],
        None => order_colors(rows, &owner)?,
    };
    mesh::write_ply(path, &v, &signed_faces(&f), Some(&rgb))
}

fn check_faces(vertices: &[Point], faces: &[[u32; 3]]) -> Result<()> {
    if let Some(&k) = faces.iter().flatten().find(|&&k| k as usize >= vertices.len()) {
        return Err(Error::invalid(format!("index {k} is out of bounds for axis 0 with size {}", vertices.len())));
    }
    Ok(())
}

/// Segments where a mesh crosses the plane `z`, in xy, face by face.
///
/// A face with every vertex on one side is skipped; a vertex exactly on the
/// plane counts as below it, so a face touching the plane at a vertex gives
/// nothing. Each crossing face gives the segment between its first and last
/// crossed edges.
pub fn section(vertices: &[Point], faces: &[[u32; 3]], z: f64) -> Result<Vec<Segment>> {
    check_faces(vertices, faces)?;
    let mut out = Vec::new();
    for f in faces {
        let t = [vertices[f[0] as usize], vertices[f[1] as usize], vertices[f[2] as usize]];
        let d = [t[0][2] - z, t[1][2] - z, t[2][2] - z];
        if d.iter().all(|&x| x > 0.0) || d.iter().all(|&x| x < 0.0) {
            continue;
        }
        let mut ends: Vec<[f64; 2]> = Vec::with_capacity(3);
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (da, db) = (d[a], d[b]);
            let w = if (da > 0.0) != (db > 0.0) { da / if da == db { 1e-12 } else { da - db } } else { f64::NAN };
            let e = [t[a][0] + w * (t[b][0] - t[a][0]), t[a][1] + w * (t[b][1] - t[a][1])];
            if !(e[0].is_nan() || e[1].is_nan()) {
                ends.push(e);
            }
        }
        if ends.len() >= 2 {
            out.push([ends[0], ends[ends.len() - 1]]);
        }
    }
    Ok(out)
}

/// Even-odd test of each `(x, y)` point against a soup of segments
/// ([`section`]): a point is inside if a ray towards +x crosses an odd
/// number of them.
pub fn inside(section: &[Segment], points: &[[f64; 2]]) -> Vec<bool> {
    let mut hits = vec![0u32; points.len()];
    for &[[x0, y0], [x1, y1]] in section {
        if y0 == y1 {
            continue;
        }
        let (lo, hi) = (y0.min(y1), y0.max(y1));
        for (h, p) in hits.iter_mut().zip(points) {
            if p[1] >= lo && p[1] < hi {
                let f = (p[1] - y0) / (y1 - y0);
                if x0 + f * (x1 - x0) > p[0] {
                    *h += 1;
                }
            }
        }
    }
    hits.iter().map(|h| h % 2 == 1).collect()
}

/// How well the wood sits inside the base at the join: the distance (m)
/// between the means of their section ends, and the share of the wood's
/// cross-section (on a grid of `cell`) outside the base. NaN for an empty
/// section, and the share NaN where the wood's section encloses no cell.
pub fn join_fit(base: &[Segment], wood: &[Segment], cell: f64) -> (f64, f64) {
    if base.is_empty() || wood.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = |s: &[Segment]| -> [f64; 2] {
        let pts: Vec<[f64; 2]> = s.iter().flatten().copied().collect();
        let n = pts.len() as f64;
        let mut acc = pts[0];
        for p in &pts[1..] {
            acc[0] += p[0];
            acc[1] += p[1];
        }
        [acc[0] / n, acc[1] / n]
    };
    let (mw, mb) = (mean(wood), mean(base));
    let d = [mw[0] - mb[0], mw[1] - mb[1]];
    let offset = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in wood.iter().flatten() {
        for k in 0..2 {
            lo[k] = if p[k].is_nan() || lo[k].is_nan() { f64::NAN } else { lo[k].min(p[k]) };
            hi[k] = if p[k].is_nan() || hi[k].is_nan() { f64::NAN } else { hi[k].max(p[k]) };
        }
    }
    let xs = arange(lo[0] - cell, hi[0] + cell, cell);
    let ys = arange(lo[1] - cell, hi[1] + cell, cell);
    let grid: Vec<[f64; 2]> = ys.iter().flat_map(|&y| xs.iter().map(move |&x| [x, y])).collect();
    let inw = inside(wood, &grid);
    let n_in = inw.iter().filter(|&&b| b).count();
    if n_in == 0 {
        return (offset, f64::NAN);
    }
    let inb = inside(base, &grid);
    let out = inw.iter().zip(&inb).filter(|(&w, &b)| w && !b).count();
    (offset, out as f64 / n_in as f64)
}

/// A buttress and a QSM as one mesh; see [`fuse`].
#[derive(Debug, Clone, Default)]
pub struct TreeMesh {
    pub vertices: Vec<Point>,
    pub faces: Vec<[u32; 3]>,
    /// Per face: 0 for the buttress, 1 for the wood above it.
    pub part: Vec<u8>,
    pub buttress_volume: f64,
    pub wood_volume: f64,
    pub top_z: f64,
    /// Distance (m) between the middles of base and wood at the join.
    pub offset: f64,
    /// Share of the wood's cross-section at the join outside the base.
    pub overhang: f64,
}

impl TreeMesh {
    /// Whole-stem volume: buttress plus the wood above it (m³).
    pub fn volume(&self) -> f64 {
        self.buttress_volume + self.wood_volume
    }
}

/// Join a buttress mesh (`vertices`, `faces`, its `volume` below `top_z`)
/// to the model `rows`: the model is cut `overlap` below `top_z` so the
/// tubes reach down inside the base, both surfaces go into one mesh with a
/// part label per face, and the volumes are read at `top_z`. The join is
/// judged on sections 5 cm below (base) and above (wood) the top.
#[allow(clippy::too_many_arguments)]
pub fn fuse(vertices: &[Point], faces: &[[u32; 3]], volume: f64, top_z: f64, rows: &[Row], sides: usize, contiguous: bool, overlap: f64) -> Result<TreeMesh> {
    check_faces(vertices, faces)?;
    let (wv, wf, _) = mesh(&above(rows, top_z - overlap), sides, contiguous);
    let nb = vertices.len() as u32;
    let mut v = vertices.to_vec();
    v.extend_from_slice(&wv);
    let mut f = faces.to_vec();
    f.extend(wf.iter().map(|t| [t[0].wrapping_add(nb), t[1].wrapping_add(nb), t[2].wrapping_add(nb)]));
    let mut part = vec![0u8; faces.len()];
    part.extend(std::iter::repeat_n(1u8, wf.len()));
    let base = section(vertices, faces, top_z - 0.05)?;
    let stem = section(&wv, &wf, top_z + 0.05)?;
    let (offset, overhang) = join_fit(&base, &stem, 0.02);
    Ok(TreeMesh { vertices: v, faces: f, part, buttress_volume: volume, wood_volume: volume_above(rows, top_z), top_z, offset, overhang })
}

/// A named part of a joined mesh: its vertices and faces.
pub type MeshPart = (&'static str, Vec<Point>, Vec<[u32; 3]>);

/// The parts of a joined mesh as separate meshes, `buttress` then `wood`
/// (a part without faces is left out), each with only the vertices its
/// faces use, in index order.
pub fn tree_mesh_parts(vertices: &[Point], faces: &[[u32; 3]], part: &[u8]) -> Result<Vec<MeshPart>> {
    check_faces(vertices, faces)?;
    if part.len() != faces.len() {
        return Err(Error::invalid("part must have one label per face"));
    }
    let mut out = Vec::new();
    for (k, name) in [(0u8, "buttress"), (1u8, "wood")] {
        let f: Vec<[u32; 3]> = faces.iter().zip(part).filter(|(_, &p)| p == k).map(|(f, _)| *f).collect();
        if f.is_empty() {
            continue;
        }
        let used: Vec<u32> = f.iter().flatten().copied().collect::<BTreeSet<u32>>().into_iter().collect();
        let mut new = vec![0u32; used.last().map_or(0, |&m| m as usize + 1)];
        for (i, &u) in used.iter().enumerate() {
            new[u as usize] = i as u32;
        }
        let v = used.iter().map(|&u| vertices[u as usize]).collect();
        let f = f.iter().map(|t| [new[t[0] as usize], new[t[1] as usize], new[t[2] as usize]]).collect();
        out.push((name, v, f));
    }
    Ok(out)
}

/// Write a joined mesh as OBJ objects `buttress` and `wood`.
pub fn write_tree_mesh_obj(path: impl AsRef<Path>, vertices: &[Point], faces: &[[u32; 3]], part: &[u8]) -> Result<()> {
    let parts = tree_mesh_parts(vertices, faces, part)?;
    let objs: Vec<ObjMesh> = parts.iter().map(|(n, v, f)| ObjMesh { name: n, vertices: v, faces: f }).collect();
    mesh::write_obj(path, &objs)
}

/// Write a joined mesh as binary PLY, faces in `color` or the buttress in
/// [`BUTTRESS_COLOR`] and the wood in the stem colour.
pub fn write_tree_mesh_ply(path: impl AsRef<Path>, vertices: &[Point], faces: &[[u32; 3]], part: &[u8], color: Option<[u8; 3]>) -> Result<()> {
    if part.len() != faces.len() {
        return Err(Error::invalid("part must have one label per face"));
    }
    let rgb: Vec<[u8; 3]> = match color {
        Some(c) => vec![c; faces.len()],
        None => part.iter().map(|&p| if p == 0 { BUTTRESS_COLOR } else { ORDER_COLORS[0] }).collect(),
    };
    mesh::write_ply(path, vertices, &signed_faces(faces), Some(&rgb))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stem() -> Vec<Row> {
        let mut rows: Vec<Row> = (0..4).map(|k| [0.0, 0.0, k as f64, 0.0, 0.0, 1.0, 1.0, 0.2, k as f64 - 1.0, 0.0, 0.0, 10.0]).collect();
        rows.push([0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 1.0, 0.1, 1.0, 1.0, 1.0, 10.0]);
        rows
    }

    #[test]
    fn a_cut_drops_what_is_below_and_renumbers() {
        let rows = stem();
        let cut = above(&rows, 2.0);
        assert_eq!(cut.len(), 2);
        assert_eq!(cut[0][2], 2.0);
        assert_eq!([cut[0][8], cut[1][8]], [-1.0, 0.0]);
        let v = volume_above(&rows, 2.0);
        assert!((v - totals(&cut).total_volume).abs() < 1e-15);
        let lean = [[0.0, 0.0, 0.0, 0.0, 0.6, 0.8, 5.0, 0.2, -1.0, 0.0, 0.0, 10.0]];
        assert!((above(&lean, 2.0)[0][6] - (5.0 - 2.0 / 0.8)).abs() < 1e-12);
        assert!(above(&lean, 100.0).is_empty());
        assert_eq!(totals(&[]).max_branch_order, 0.0);
    }

    #[test]
    fn csv_skips_blanks_and_comments() {
        let r = parse_csv("h\n1,2,3,4,5,6,7,8,9,10,11,12\n\n# c\n 1, 2,3,4,5,6,7,8,9,10,11,nan # x\n").unwrap();
        assert_eq!(r.len(), 2);
        assert!(r[1][11].is_nan());
        assert!(parse_csv("h\n1,2\n").is_err());
        assert!(parse_csv("h\n").unwrap().is_empty());
    }

    #[test]
    fn a_square_section_holds_its_middle() {
        // A unit cube from 0 to 1, cut at 0.5.
        let v: Vec<Point> = (0..8).map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64]).collect();
        let f = [[0, 1, 5], [0, 5, 4], [1, 3, 7], [1, 7, 5], [3, 2, 6], [3, 6, 7], [2, 0, 4], [2, 4, 6], [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6]];
        let s = section(&v, &f, 0.5).unwrap();
        assert_eq!(s.len(), 8);
        assert_eq!(inside(&s, &[[0.5, 0.5], [1.5, 0.5], [0.5, -0.1]]), vec![true, false, false]);
        let (off, over) = join_fit(&s, &s, 0.1);
        assert_eq!((off, over), (0.0, 0.0));
        assert!(join_fit(&s, &[], 0.1).0.is_nan());
        let parts = tree_mesh_parts(&v, &f, &[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].1.len(), 8);
        assert!(section(&v, &[[0, 1, 9]], 0.5).is_err());
    }

    #[test]
    fn fused_volumes_are_read_at_the_top() {
        let v: Vec<Point> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let f = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
        let rows = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.2, -1.0, 0.0, 0.0, 10.0]];
        let t = fuse(&v, &f, 0.16, 1.0, &rows, 8, true, 0.1).unwrap();
        assert_eq!(t.part.iter().filter(|&&p| p == 0).count(), 4);
        assert!((t.wood_volume - volume_above(&rows, 1.0)).abs() < 1e-15);
        assert!((t.volume() - 0.16 - std::f64::consts::PI * 0.04 * 2.0).abs() < 1e-12);
        let low = t.faces[4..].iter().flatten().map(|&k| t.vertices[k as usize][2]).fold(f64::INFINITY, f64::min);
        assert!((low - 0.9).abs() < 1e-12);
        assert_eq!(index_color(-1).unwrap(), ORDER_COLORS[5]);
        assert!(index_color(-7).is_err());
    }
}
