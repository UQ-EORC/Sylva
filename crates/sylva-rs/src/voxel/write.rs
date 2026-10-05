// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Adapted from rayvoxel (Josh Rivory, unpublished), a port of AMAPVox (UMR AMAP);
// see THIRD_PARTY_NOTICES.md.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! AMAPVox `.vox`, plain text and per-tree inclination CSV writers.

use std::io::{BufWriter, Write};
use std::path::Path;

use super::{Attenuation, RayVoxels, VoxelState, F, I};
use crate::error::{Error, Result};

/// Which voxels a file holds. By default every observed or occluded voxel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WriteOptions {
    /// Also write voxels no ray reached.
    pub include_unobserved: bool,
    /// Only voxels holding an echo (overrides `include_unobserved`).
    pub filled_only: bool,
}

pub(crate) type Column<'a> = (String, Box<dyn Fn(usize) -> String + Sync + 'a>);

fn real(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    // Values that round to zero print without a sign.
    let v = if v.abs() < 5e-7 { 0.0 } else { v };
    format!("{v:.6}")
}

impl RayVoxels {
    /// Output columns after the voxel indices, in rayvoxel's order.
    ///
    /// A column is a name and a function from a voxel's index to its text, so
    /// the table is described here once and the writers simply walk it. The
    /// functions are boxed because each is a different closure; `Box<dyn Fn>`
    /// is "some function, decided at run time", which is what lets them sit in
    /// one list. Nothing is computed until a writer asks for a voxel, so the
    /// derived quantities cost nothing for a column that is not written.
    ///
    /// The order is rayvoxel's, and some columns appear only when the trace
    /// recorded them — leaf and wood counts, sub-voxel splits, the beam
    /// fields — so the header matches what is actually in the grid.
    pub(crate) fn columns(&self) -> Vec<Column<'_>> {
        let mut cols: Vec<Column> = Vec::new();
        macro_rules! col {
            ($name:expr, $f:expr) => {
                cols.push(($name.to_string(), Box::new($f)))
            };
        }
        // Shorthands: a raw counter, and a raw sum formatted as rayvoxel does.
        let int = |fld: I| move |i: usize| self.get_i(fld, i).to_string();
        let flt = |fld: F| move |i: usize| real(self.get_f(fld, i) as f64);
        let incl = self.predominant_tree.is_some();

        col!("num_hits", int(I::NumHits));
        col!("num_hit_plant", int(I::NumHitPlant));
        col!("free_path_length_plant", flt(F::FreePathLengthPlant));
        col!("free_path_length_leaf", flt(F::FreePathLengthLeaf));
        col!("free_path_length_wood", flt(F::FreePathLengthWood));
        if self.has_leaf {
            col!("num_hit_leaf", int(I::NumHitLeaf));
        }
        if self.has_wood {
            col!("num_hit_wood", int(I::NumHitWood));
        }
        for fld in [I::NumBeams, I::NumMissRays, I::NumUnboundRays, I::NumRaysOccluded] {
            col!(fld.name(), int(fld));
        }
        for fld in [F::PathLength, F::FreePathLength, F::EffectiveFreePathLength, F::PathLengthOccluded, F::PathLengthUnbound] {
            col!(fld.name(), flt(fld));
        }
        col!("voxel_size", |_| real(self.voxel_size));
        col!("surface_area", |i| real(self.pad_g0_5(i) * self.voxel_size.powi(3)));
        let split = self.params.subvoxel_split;
        if split > 0 {
            col!("subvoxel_split", move |_| split.to_string());
        }
        col!("mean_zenith_angle", |i| real(self.mean_zenith(i).to_degrees()));
        col!("mean_azimuth_angle_deg", |i| real(self.mean_azimuth(i).0.to_degrees()));
        col!("azimuth_concentration", |i| real(self.mean_azimuth(i).1));
        col!("mean_laser_dist", |i| real(self.mean_laser_distance(i)));
        col!("attenuation_fpl", |i| real(self.attenuation(i, Attenuation::Fpl) - self.fpl_bias(i)));
        col!("attenuation_ppl", |i| real(self.attenuation(i, Attenuation::Ppl)));
        if self.ground_height.is_some() {
            col!("distance_from_ground", |i| real(self.distance_from_ground(i)));
        }
        if self.params.beam.is_some() {
            col!("transmittance", |i| real(self.transmittance(i)));
            col!("bs_entering", flt(F::BsEntering));
            col!("bs_intercepted", flt(F::BsIntercepted));
        }
        if split > 0 {
            col!("exploration_rate", |i| real(self.exploration(i).0));
            col!("subvoxel_bitmap", |i| self.exploration(i).1.to_string());
        }
        if !incl {
            col!("pad_g_corrected", |i| real(self.pad_g_corrected(i)));
        } else {
            let na = |s: Option<&'static str>| s.unwrap_or("NA").to_string();
            col!("predominant_tree", |i| self.predominant_tree.as_ref().map_or(-1, |p| p[i]).to_string());
            col!("piad_dewit", move |i| na(self.voxel_iad(i).and_then(|t| t.piad_de_wit)));
            if self.has_leaf {
                col!("liad_dewit", move |i| na(self.voxel_iad(i).filter(|_| self.get_i(I::NumHitLeaf, i) > 0).and_then(|t| t.liad_de_wit)));
            }
            if self.has_wood {
                col!("wiad_dewit", move |i| na(self.voxel_iad(i).filter(|_| self.get_i(I::NumHitWood, i) > 0).and_then(|t| t.wiad_de_wit)));
            }
            col!("g_plant", |i| real(self.voxel_iad(i).map_or_else(|| self.g_analytic(i), |t| t.g_plant)));
            if self.has_leaf {
                col!("g_leaf", |i| real(self.voxel_iad(i).filter(|_| self.get_i(I::NumHitLeaf, i) > 0).map_or(0.0, |t| t.g_leaf)));
            }
            if self.has_wood {
                col!("g_wood", |i| real(self.voxel_iad(i).filter(|_| self.get_i(I::NumHitWood, i) > 0).map_or(0.0, |t| t.g_wood)));
            }
            for &m in &self.params.attenuation {
                col!(format!("pad_{}", m.name()), move |i| real(self.area_density(i, m).pad));
                if self.has_leaf {
                    col!(format!("lad_{}", m.name()), move |i| real(self.area_density(i, m).lad));
                }
                if self.has_wood {
                    col!(format!("wad_{}", m.name()), move |i| real(self.area_density(i, m).wad));
                }
            }
        }
        if let Some(w) = &self.wood_volume {
            col!("wood_volume", move |i| real(w[i] as f64));
            col!("wood_volume_density", move |i| real(w[i] as f64 / self.voxel_size.powi(3)));
        }
        cols
    }

    /// Rows of the voxels `opts` keeps; `offset` is added to the printed
    /// indices (for a part of a larger grid).
    pub(crate) fn write_rows(&self, out: &mut impl Write, cols: &[Column], opts: WriteOptions, with_xyz: bool, offset: [usize; 3]) -> Result<usize> {
        use rayon::prelude::*;
        let keep = |i: usize| match self.state(i) {
            VoxelState::Filled => true,
            VoxelState::Unobserved => opts.include_unobserved && !opts.filled_only,
            _ => !opts.filled_only,
        };
        let mut written = 0;
        let n = self.n_voxels();
        const CHUNK: usize = 1 << 16;
        for start in (0..n).step_by(CHUNK) {
            let lines: Vec<String> = (start..(start + CHUNK).min(n))
                .into_par_iter()
                .filter(|&i| keep(i))
                .map(|i| {
                    let [a, b, c] = self.unravel(i);
                    let [a, b, c] = [a + offset[0], b + offset[1], c + offset[2]];
                    let mut line = format!("{a} {b} {c}");
                    if with_xyz {
                        let p = self.center(i);
                        line += &format!(" {:.6} {:.6} {:.6}", p[0], p[1], p[2]);
                    }
                    for (_, f) in cols {
                        line.push(' ');
                        line += &f(i);
                    }
                    line.push('\n');
                    line
                })
                .collect();
            written += lines.len();
            for l in lines {
                out.write_all(l.as_bytes())?;
            }
        }
        Ok(written)
    }

    /// Write an AMAPVox voxel-space file. Returns the number of voxels written.
    pub fn write_vox(&self, path: impl AsRef<Path>, opts: WriteOptions) -> Result<usize> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::file(path, e.to_string()))?;
        let mut out = BufWriter::new(file);
        let cols = self.columns();
        self.vox_header(&mut out, &cols, self.origin, self.shape)?;
        let n = self.write_rows(&mut out, &cols, opts, false, [0; 3])?;
        out.flush()?;
        Ok(n)
    }

    /// The `.vox` header of a grid at `origin` of `shape` voxels with this
    /// grid's options and columns.
    pub(crate) fn vox_header(&self, out: &mut impl Write, cols: &[Column], origin: crate::Point, shape: [usize; 3]) -> Result<()> {
        let g_desc = if self.predominant_tree.is_none() {
            format!("analytic LAD ({})", self.params.lad.name())
        } else {
            let vicari: Vec<&str> = self.params.attenuation.iter().filter(|m| **m != Attenuation::Bailey).map(|m| m.name()).collect();
            let mut parts = Vec::new();
            if !vicari.is_empty() {
                parts.push(format!("Vicari: {}", vicari.join(",")));
            }
            if self.params.attenuation.contains(&Attenuation::Bailey) {
                parts.push("Bailey".into());
            }
            format!("estimated IAD ({})", parts.join("; "))
        };
        let s = self.voxel_size;
        writeln!(out, "VOXEL SPACE")?;
        writeln!(out, "#g_correction:{g_desc}")?;
        writeln!(out, "#max_corner:{:.6} {:.6} {:.6}", origin[0] + shape[0] as f64 * s, origin[1] + shape[1] as f64 * s, origin[2] + shape[2] as f64 * s)?;
        writeln!(out, "#min_corner:{} {} {}", real(origin[0]), real(origin[1]), real(origin[2]))?;
        writeln!(out, "#res:{s:.6} {s:.6} {s:.6}")?;
        writeln!(out, "#split:{} {} {}", shape[0], shape[1], shape[2])?;
        if self.params.subvoxel_split > 0 {
            writeln!(out, "#subvoxel_min_beams:{}", self.params.subvoxel_min_beams)?;
            writeln!(out, "#subvoxel_split:{}", self.params.subvoxel_split)?;
        }
        let names: Vec<&str> = cols.iter().map(|c| c.0.as_str()).collect();
        writeln!(out, "i j k {}", names.join(" "))?;
        Ok(())
    }

    /// The header line of the text table.
    pub(crate) fn text_header(&self, out: &mut impl Write, cols: &[Column]) -> Result<()> {
        let names: Vec<&str> = cols.iter().map(|c| c.0.as_str()).collect();
        writeln!(out, "i j k x y z {}", names.join(" "))?;
        Ok(())
    }

    /// Write a space-delimited table with voxel centres. Returns the number of voxels written.
    pub fn write_text(&self, path: impl AsRef<Path>, opts: WriteOptions) -> Result<usize> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::file(path, e.to_string()))?;
        let mut out = BufWriter::new(file);
        let cols = self.columns();
        self.text_header(&mut out, &cols)?;
        let n = self.write_rows(&mut out, &cols, opts, true, [0; 3])?;
        out.flush()?;
        Ok(n)
    }

    /// Write the per-tree inclination angle distributions, one row per tree.
    pub fn write_iad_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::file(path, e.to_string()))?;
        let mut out = BufWriter::new(file);
        let nb = self.params.n_iad_bins;
        let bailey = self.params.attenuation.contains(&Attenuation::Bailey);
        let vicari = self.params.attenuation.iter().any(|m| *m != Attenuation::Bailey);
        let mut sets: Vec<&str> = Vec::new();
        if self.has_leaf {
            sets.push("liad");
        }
        if self.has_wood {
            sets.push("wiad");
        }
        sets.push("piad");

        let mut header = vec!["tree_id".to_string()];
        if vicari {
            header.extend(sets.iter().map(|s| format!("{s}_dewit")));
            header.extend(sets.iter().flat_map(|s| (0..nb).map(move |b| format!("{s}_{b}"))));
        }
        if bailey {
            header.extend(sets.iter().flat_map(|s| (0..nb).map(move |b| format!("{s}_bailey_{b}"))));
        }
        writeln!(out, "{}", header.join(","))?;
        for (tid, t) in &self.tree_iad {
            let mut row = vec![tid.to_string()];
            let bins = |h: &[f64]| (0..nb).map(|b| format!("{:.6}", h.get(b).copied().unwrap_or(0.0))).collect::<Vec<_>>();
            if vicari {
                for s in &sets {
                    row.push(match *s { "liad" => t.liad_de_wit, "wiad" => t.wiad_de_wit, _ => t.piad_de_wit }.unwrap_or("").to_string());
                }
                for s in &sets {
                    row.extend(bins(match *s { "liad" => &t.liad, "wiad" => &t.wiad, _ => &t.piad }));
                }
            }
            if bailey {
                for s in &sets {
                    row.extend(bins(match *s { "liad" => &t.liad_bailey, "wiad" => &t.wiad_bailey, _ => &t.piad_bailey }));
                }
            }
            writeln!(out, "{}", row.join(","))?;
        }
        out.flush()?;
        Ok(())
    }
}
